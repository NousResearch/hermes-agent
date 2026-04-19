"""Multi-objective fitness template (NSGA-II mode).

Returning a ``dict[str, float]`` tells the runner to select with NSGA-II
non-dominated sort + crowding distance. Each key is one objective; the
order is preserved and the first objective is the primary display key.

The example balances an accuracy proxy against prompt length (treated
here as a cost signal — shorter is cheaper to serve). Replace with your
real measurements.
"""

from __future__ import annotations

from evolver_sdk import fitness_spec


@fitness_spec(
    held_out_frac=0.2,
    timeout_s=30,
    objectives=["accuracy", "cost"],   # NSGA-II uses this list
)
def fitness(candidate: str, context: dict) -> dict[str, float]:
    # Accuracy stub: +0.5 base, +0.3 if the prompt is imperative.
    acc = 0.5 + (0.3 if candidate.strip().endswith(".") else 0.0)

    # Cost stub: negate length so "higher is better" semantics hold.
    cost = -float(len(candidate)) / 100.0

    return {"accuracy": acc, "cost": cost}
