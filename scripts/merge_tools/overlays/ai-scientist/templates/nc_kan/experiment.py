"""Synthetic neural-collapse benchmark (Hermes fork template for AI-Scientist)."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser(description="Run nc_kan experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
args = parser.parse_args()


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _kan_layer(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Minimal KAN-inspired univariate transform: sum_j phi_j(x_j)."""
    hidden = _relu(x @ w1)
    return hidden @ w2


def _train_probe(x: np.ndarray, y: np.ndarray, lr: float = 0.05, steps: int = 120) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(42)
    w1 = rng.normal(scale=0.2, size=(x.shape[1], 16))
    w2 = rng.normal(scale=0.2, size=(16, len(np.unique(y))))
    for _ in range(steps):
        logits = _kan_layer(x, w1, w2)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y)), y] = 1.0
        grad_logits = (probs - one_hot) / len(y)
        grad_w2 = _relu(x @ w1).T @ grad_logits
        grad_hidden = grad_logits @ w2.T
        grad_w1 = x.T @ (grad_hidden * (x @ w1 > 0))
        w1 -= lr * grad_w1
        w2 -= lr * grad_w2
    preds = np.argmax(_kan_layer(x, w1, w2), axis=1)
    acc = float((preds == y).mean())
    return _kan_layer(x, w1, w2), acc


def _nc_ratio(features: np.ndarray, labels: np.ndarray) -> float:
    classes = np.unique(labels)
    global_mean = features.mean(axis=0)
    within = 0.0
    between = 0.0
    for cls in classes:
        mask = labels == cls
        cluster = features[mask]
        center = cluster.mean(axis=0)
        within += float(np.mean(np.sum((cluster - center) ** 2, axis=1)))
        between += float(np.sum((center - global_mean) ** 2))
    return within / max(between, 1e-8)


if __name__ == "__main__":
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(7)
    n_per_class = 64
    centers = rng.normal(size=(4, 8))
    xs, ys = [], []
    for label, center in enumerate(centers):
        block = center + rng.normal(scale=0.35, size=(n_per_class, 8))
        xs.append(block)
        ys.append(np.full(n_per_class, label, dtype=int))
    x = np.vstack(xs)
    y = np.concatenate(ys)

    features, accuracy = _train_probe(x, y)
    nc_ratio = _nc_ratio(features, y)
    class_sep = float(np.mean([np.linalg.norm(centers[i] - centers[j]) for i in range(4) for j in range(i + 1, 4)]))

    means = {
        "accuracy": accuracy,
        "nc_ratio": nc_ratio,
        "class_separation": class_sep,
    }
    payload = {
        "nc_kan": {
            "means": means,
            "features": features[:16].tolist(),
        }
    }
    with open(os.path.join(out_dir, "final_info.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
