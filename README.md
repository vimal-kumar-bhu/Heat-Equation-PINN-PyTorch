# 1D Heat Equation using Physics-Informed Neural Networks (PINNs) in PyTorch

This repository contains a complete, self-contained PyTorch implementation for solving the **1D Heat Equation (Diffusion Equation)** using **Physics-Informed Neural Networks (PINNs)**. The implementation leverages both **Adam** and **L-BFGS** optimizers to achieve high-precision convergence, computes relative $L_2$ errors, compares exact vs predicted numerical values, and generates multi-time snapshots and spatio-temporal heatmaps.

---

## 📌 Mathematical Problem

We consider the 1D linear heat equation (diffusion equation):
$$\frac{\partial u}{\partial t} - \nu \frac{\partial^2 u}{\partial x^2} = 0, \quad x \in [-1, 1], \quad t \in [0, 1]$$

with diffusion coefficient $\nu = \frac{0.01}{\pi}$ and initial condition:
$$u(0, x) = -\sin(\pi x)$$

The analytical (exact) solution for comparison is derived via **Separation of Variables**:
$$u(t, x) = -\sin(\pi x) \exp\left(-\nu \pi^2 t\right)$$

---

## 🚀 Key Features

* **Deep Neural Network (DNN) Architecture:** Multi-layer feedforward neural network utilizing $\tanh$ activation functions.
* **Two-Stage Optimization Strategy:** Initial fast convergence via **Adam**, followed by high-precision refinement using **L-BFGS**.
* **Automatic Differentiation:** PyTorch `autograd` is used to compute exact spatio-temporal derivatives for the PDE residual loss ($\mathcal{L}_f$).
* **Comprehensive Evaluation:** 
  * Computes overall and snapshot-specific **Relative $L_2$ Errors** at multiple time instances ($t = 0.25, 0.50, 0.75$).
  * Prints precise numerical comparisons between analytical exact solutions and PINN predictions.

---

## 🛠️ Requirements

Make sure you have the following libraries installed in your Python environment:
* Python 3.8+
* PyTorch
* NumPy
* Matplotlib

---

## 💻 Usage

Clone the repository and run the main python script to train the PINN model, compute errors, and generate comparison plots:

```bash
python main.py
