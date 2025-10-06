import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load model and training features
# -------------------------------
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Telecom Churn Predictor", layout="centered")

st.title("📞 Telecom Customer Churn Prediction")
st.markdown("Predict whether a telecom customer will churn based on their details.")

# -------------------------------
# User Input Section
# -------------------------------
st.subheader("Enter Customer Details:")

# Only show the features used during training (excluding dropped ones)
# Replace the following feature names with YOUR actual model feature names
input_data = {}

# Example numeric features
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

# Example categorical features
categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

# Collect numeric inputs
for col in numeric_features:
    input_data[col] = st.number_input(f"{col}", min_value=0.0, step=1.0)

# Collect categorical inputs
for col in categorical_features:
    unique_vals = ["Yes", "No"]
    if col == "gender":
        unique_vals = ["Male", "Female"]
    elif col == "InternetService":
        unique_vals = ["DSL", "Fiber optic", "No"]
    elif col == "Contract":
        unique_vals = ["Month-to-month", "One year", "Two year"]
    elif col == "PaymentMethod":
        unique_vals = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]

    input_data[col] = st.selectbox(f"{col}", unique_vals)

# -------------------------------
# Data Preprocessing
# -------------------------------
df = pd.DataFrame([input_data])

# Convert categorical columns to dummies (same as training)
df = pd.get_dummies(df)

# Align columns with training features
df = df.reindex(columns=features, fill_value=0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Churn"):
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer is **likely to churn** (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer is **not likely to churn** (Probability: {prob:.2f})")
