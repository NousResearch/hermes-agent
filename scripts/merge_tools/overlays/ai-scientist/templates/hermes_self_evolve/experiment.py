"""Simulated Hermes agent loop benchmark for self-evolution research."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser(description="Run hermes_self_evolve experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
args = parser.parse_args()


def _simulate_agent_loop(
    tool_precision: float,
    retry_budget: int,
    context_window: int,
    n_tasks: int = 200,
) -> dict[str, float]:
    rng = np.random.default_rng(99)
    success = 0
    latencies = []
    for _ in range(n_tasks):
        difficulty = rng.uniform(0.2, 1.0)
        attempts = 0
        solved = False
        while attempts <= retry_budget and not solved:
            attempts += 1
            noise = rng.normal(scale=0.08)
            prob = min(0.98, max(0.02, tool_precision - 0.25 * difficulty + noise))
            solved = rng.random() < prob
        if solved:
            success += 1
        latencies.append(attempts * (1.1 - min(context_window, 8192) / 10000.0))
    return {
        "success_rate": success / n_tasks,
        "mean_latency": float(np.mean(latencies)),
        "retry_pressure": float(retry_budget / max(n_tasks, 1)),
    }


if __name__ == "__main__":
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    means = _simulate_agent_loop(tool_precision=0.62, retry_budget=2, context_window=4096)
    payload = {"hermes_self_evolve": {"means": means}}
    with open(os.path.join(out_dir, "final_info.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
