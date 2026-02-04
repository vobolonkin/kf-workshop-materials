# 🎮 Workshop: My First ML Model
**Goal:** Predict if a player is a VIP (High Engagement) or likely to Churn (Low Engagement) using AI.

---

## **Step 0: Getting Started**

### **1. Enter the Environment**
1.  **Click this link:** [https://kubeflow.89.169.115.198.sslip.io/](https://kubeflow.89.169.115.198.sslip.io/)
2.  If asked to log in, use the credentials provided by the instructor (Default: `user@example.com` / `12341234`).
3.  Look for the notebook named **`workshop`** in the list.
4.  Click the blue **CONNECT** button on the right side.
5.  This will open **JupyterLab** in a new tab.

### **2. Navigate to the Workshop Folder**
*Look at the file browser on the **left** side of the screen.*
1.  Double-click the folder named **`workshop`**.
2.  Inside that, double-click the folder named **`materials`**.
    *   *Note: You must be inside this folder so the code can find the dataset.*

### **3. Create your Notebook**
*Look at the main "Launcher" area on the right.*
1.  Under the **Notebook** header, click the big square button with the Python logo labeled **Python 3**.
2.  A new file named `Untitled.ipynb` will open.
3.  (Optional) Right-click the file name `Untitled.ipynb` in the left sidebar, select **Rename**, and name it `MyModel.ipynb`.

### **4. How to Run Code**
*   **To create a NEW empty cell:** Click the **`+`** (Plus) icon in the top toolbar.
*   **To RUN the code:** Click inside the cell to select it, then press **`Shift + Enter`** on your keyboard (or click the **▶** Play button).

---

## **Step 1: The Setup**
*What are we doing?*
We are importing the "tools" we need. `pandas` is for reading data (like Excel), and `sklearn` is the brain that contains the Artificial Intelligence algorithms. We also set some technical settings to make sure the code runs smoothly on the cloud.

**Copy and Paste this into the first Cell:**

```python
import os
# These settings help the code run stable on Cloud environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

print("✅ Setup complete! Libraries loaded.")
```
*(Press **Shift + Enter** to run. Wait for the ✅ message.)*

---

## **Step 2: Load & Visualize**
*What are we doing?*
We load the dataset. Then, we ask a simple question: **"Do players who play more frequently actually have higher engagement scores?"** We generate a Boxplot to check this hypothesis visually.

**1. Click the `+` button to create a new empty cell.**
**2. Copy and Paste this code:**

```python
print("\n--- PART 1: Investigation ---")

# 1. Load Data
try:
    df = pd.read_csv('online_gaming_behavior_dataset.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'online_gaming_behavior_dataset.csv' not found.") 
    print("👉 Check that you are inside the 'workshop/materials' folder in the sidebar!")

# 2. Visualize: Do frequent sessions actually mean higher engagement?
custom_order = ['Low', 'Medium', 'High'] 

plt.figure(figsize=(8, 5))

# We plot the data to see patterns before we do any math
sns.boxplot(
    data=df, 
    x='EngagementLevel', 
    y='SessionsPerWeek',
    hue='EngagementLevel', # Fixes the warning
    legend=False,          # Fixes the warning
    order=custom_order, 
    palette='coolwarm'
)

plt.title("Hypothesis Check: Do Frequent Sessions = Higher Engagement?")
plt.grid(True, alpha=0.3)
plt.show()
```

---

## **Step 3: Prepare Data & The "Baseline" Model**
*What are we doing?*
1.  **Translation:** Computers don't understand words like "Male" or "High". We convert them into numbers (0, 1, 2).
2.  **Splitting:** We hide 20% of the data to use as a "Final Exam" for the AI later.
3.  **Baseline:** We run a simple **Logistic Regression**. This is a basic statistical model. If our fancy AI can't beat this score, it's not worth using.

**1. Click the `+` button to create a new empty cell.**
**2. Copy and Paste this code:**

```python
print("\n--- PART 2: The Baseline (Linear Model) ---")

# 1. Clean Data (Remove ID as it predicts nothing)
df_clean = df.drop(columns=['PlayerID'])

# 2. Manual Mapping (Crucial Step)
# We map 'Low' to 0, 'Medium' to 1, 'High' to 2 so the charts makes sense
target_map = {'Low': 0, 'Medium': 1, 'High': 2}
df_clean['EngagementLevel'] = df_clean['EngagementLevel'].map(target_map)
target_names = ['Low', 'Medium', 'High'] 

# 3. Encode other text columns automatically (Gender, Location, etc)
label_encoders = {}
for col in ['Gender', 'Location', 'GameGenre', 'GameDifficulty']:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# 4. Split Features (X) and Target (y)
X = df_clean.drop(columns=['EngagementLevel'])
y = df_clean['EngagementLevel']

# 80% Training (Study), 20% Testing (Exam)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale Data (Normalize) - Helps the AI learn faster
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Linear Model (Logistic Regression)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)

print(f"Linear Model Accuracy: {accuracy_score(y_test, log_pred)*100:.2f}%")

# Plot Confusion Matrix (Blue squares show correct guesses)
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, log_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix: Linear Baseline")
plt.show()
```

---

## **Step 4: The Random Forest**
*What are we doing?*
We train a **Random Forest**. Imagine 100 decision trees (flowcharts) voting on the answer. This is usually much more accurate than a linear model for gaming data.

**1. Click the `+` button to create a new empty cell.**
**2. Copy and Paste this code:**

```python
print("\n--- PART 3: Random Forest (Decision Trees) ---")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) 
rf_pred = rf_model.predict(X_test)

print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred)*100:.2f}%")

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
cm_rf = confusion_matrix(y_test, rf_pred)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=target_names)
disp_rf.plot(cmap='Blues', values_format='d') 
plt.title("Confusion Matrix: Random Forest")
plt.show()
```

---

## **Step 5: The Neural Network**
*What are we doing?*
We build a small **Neural Network** (simulating a brain) to see if it can beat the Forest. We also measure how long it takes to train compared to other models.

**1. Click the `+` button to create a new empty cell.**
**2. Copy and Paste this code:**

```python
print("\n--- PART 4: Neural Network ---")

t0 = time.perf_counter()
# 1. Train Neural Network (Deep Learning)
nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
nn_model.fit(X_train_scaled, y_train) 
nn_pred = nn_model.predict(X_test_scaled)

print(f"Neural Network Accuracy: {accuracy_score(y_test, nn_pred)*100:.2f}%")

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
cm_nn = confusion_matrix(y_test, nn_pred)
disp_nn = ConfusionMatrixDisplay(confusion_matrix=cm_nn, display_labels=target_names)
disp_nn.plot(cmap='Blues', values_format='d') 
plt.title("Confusion Matrix: Neural Network")
plt.show()

t1 = time.perf_counter()
print(f"NN Training Time: {t1 - t0:.2f} seconds")
```

---

## **Step 6: The "Stress Test" (Cross Validation)**
*What are we doing?*
We perform **Cross-Validation** on our Random Forest. We divide the data into 5 chunks and run the test 5 separate times. This proves that our high accuracy wasn't just luck (or a "fluke") based on the specific 20% we selected earlier.

**1. Click the `+` button to create a new empty cell.**
**2. Copy and Paste this code:**

```python
print("\n--- PART 5: Cross Validation (Stress Test) ---")
print("Running Cross-Validation on Random Forest...")

# We use the full dataset (X, y) and let the computer split it 5 different ways
cv_scores = cross_val_score(rf_model, X, y, cv=5)

print(f"Test Run 1: {cv_scores[0]*100:.2f}%")
print(f"Test Run 2: {cv_scores[1]*100:.2f}%")
print(f"Test Run 3: {cv_scores[2]*100:.2f}%")
print(f"Test Run 4: {cv_scores[3]*100:.2f}%")
print(f"Test Run 5: {cv_scores[4]*100:.2f}%")
print("-" * 30)
print(f"AVERAGE ACCURACY: {cv_scores.mean() * 100:.2f}%")
```

---
**🎉 Congratulations! You just built and evaluated 3 different AI models.**
