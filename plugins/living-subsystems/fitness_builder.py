"""FitnessBuilder: weighted multi-dimension fitness functions and their evaluation history."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from .base import Subsystem, new_id, now, result

FITNESS_FILE = "fitness_functions.json"


class FitnessBuilder(Subsystem):
    def create_function(self, name: str, target: str, dimensions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not dimensions:
            return result(False, "At least one dimension is required")
        try:
            weights = [float(d["weight"]) for d in dimensions]
            names = [d["name"] for d in dimensions]
        except (KeyError, TypeError, ValueError):
            return result(False, "Each dimension needs a name and numeric weight")
        if any(w <= 0 or not math.isfinite(w) for w in weights) or len(set(names)) != len(names):
            return result(False, "Weights must be positive and dimension names unique")
        total = sum(weights)
        fn = {"id": new_id("fit"), "name": name, "target": target, "created": now(), "history": [],
              "dimensions": [{"name": n, "weight": w / total} for n, w in zip(names, weights)]}
        with self.lock(FITNESS_FILE):
            data = self.load(FITNESS_FILE)
            data.setdefault("functions", {})[fn["id"]] = fn
            self.save(FITNESS_FILE, data)
        return result(True, "Fitness function created", function=fn)

    def evaluate(self, fitness_id: str, scores: Dict[str, float]) -> Dict[str, Any]:
        """Weighted score in [0, 1]; ``scores`` must cover every dimension."""
        with self.lock(FITNESS_FILE):
            data = self.load(FITNESS_FILE)
            fn = data.get("functions", {}).get(fitness_id)
            if fn is None:
                return result(False, f"Unknown fitness function: {fitness_id}")
            missing = [d["name"] for d in fn["dimensions"] if d["name"] not in scores]
            if missing:
                return result(False, f"Missing scores for: {', '.join(missing)}")
            score = sum(d["weight"] * max(0.0, min(1.0, float(scores[d["name"]]))) for d in fn["dimensions"])
            fn["history"].append({"ts": now(), "scores": scores, "score": score})
            self.save(FITNESS_FILE, data)
        return result(True, f"Score {score:.3f}", score=score)

    def _functions(self) -> Dict[str, Dict[str, Any]]:
        return self.load(FITNESS_FILE).get("functions", {})

    def status(self) -> Dict[str, Any]:
        return result(True, f"{len(self._functions())} fitness functions", total=len(self._functions()))

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Latest score and trend (last minus previous) per function."""
        latest = {}
        for fid, fn in self._functions().items():
            hist = fn["history"]
            if hist:
                latest[fid] = {"name": fn["name"], "score": hist[-1]["score"],
                               "trend": hist[-1]["score"] - hist[-2]["score"] if len(hist) > 1 else None}
        return result(True, f"{len(latest)} evaluated functions", latest=latest)
