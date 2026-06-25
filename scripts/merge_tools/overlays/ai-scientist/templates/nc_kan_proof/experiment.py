"""Proof-oriented NC bound benchmark (Hermes / ShinkaEvolve alignment)."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser(description="Run nc_kan_proof experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
args = parser.parse_args()


def _features(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.tanh(x @ w)


def _bound_gap(features: np.ndarray, labels: np.ndarray) -> float:
    """Larger is better: between-class margin minus within-class spread."""
    classes = np.unique(labels)
    centers = np.array([features[labels == c].mean(axis=0) for c in classes])
    within = float(np.mean([np.std(features[labels == c], axis=0).mean() for c in classes]))
    pairs = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            pairs.append(np.linalg.norm(centers[i] - centers[j]))
    between = float(np.mean(pairs)) if pairs else 0.0
    return between - within


if __name__ == "__main__":
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(11)
    x = rng.normal(size=(200, 6))
    y = (x[:, 0] + 0.4 * x[:, 1] > 0).astype(int)
    w = rng.normal(scale=0.3, size=(6, 4))
    feats = _features(x, w)
    gap = _bound_gap(feats, y)
    margin = float(np.min(np.abs(feats.mean(axis=0))))
    proof_score = gap * margin

    means = {
        "bound_gap": gap,
        "margin": margin,
        "proof_score": proof_score,
    }
    payload = {"nc_kan_proof": {"means": means}}
    with open(os.path.join(out_dir, "final_info.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
