"""dream_rsi.py - Dream-RSI inspired meta-policy over training recipes.

Logs training attempt outcomes as DiscoveryTraces and uses ReplaySimulator to estimate
optimal recipe choices (epochs, augmentation, batch order) before running real fine-tuning compute.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryTrace:
    trace_id: str
    recipe: Dict[str, Any]  # epochs, augment, count, order
    eval_score: float
    confidence_avg: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DiscoveryTrace:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ReplaySimulator:
    """Estimates recipe performance from past discovery trace history."""

    def __init__(self, traces: List[DiscoveryTrace]):
        self.traces = traces

    def estimate_recipe_score(self, recipe: Dict[str, Any]) -> float:
        if not self.traces:
            return 0.5

        # Weighted nearest neighbor search in recipe space
        weights = []
        for tr in self.traces:
            epoch_diff = abs(recipe.get("epochs", 3) - tr.recipe.get("epochs", 3))
            aug_diff = 0.0 if recipe.get("augment") == tr.recipe.get("augment") else 1.0
            dist = epoch_diff * 0.2 + aug_diff * 0.5
            w = 1.0 / (1.0 + dist)
            weights.append((tr.eval_score or tr.confidence_avg, w))

        total_w = sum(w for _, w in weights)
        if total_w == 0:
            return 0.5
        return sum(s * w for s, w in weights) / total_w


class MetaExplorationPolicy:
    """Selects best fine-tuning recipe guided by ReplaySimulator."""

    def __init__(self, trace_store_path: str | Path):
        self.trace_store_path = Path(trace_store_path)
        self.trace_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.traces: List[DiscoveryTrace] = []
        self.load()

    def load(self) -> None:
        if self.trace_store_path.exists():
            try:
                data = json.loads(self.trace_store_path.read_text(encoding="utf-8"))
                self.traces = [DiscoveryTrace.from_dict(d) for d in data]
            except Exception as e:
                logger.warning("Failed to load discovery traces: %s", e)
                self.traces = []

    def save(self) -> None:
        self.trace_store_path.write_text(json.dumps([t.to_dict() for t in self.traces], indent=2), encoding="utf-8")

    def record_attempt(self, recipe: Dict[str, Any], eval_score: float, confidence_avg: float) -> DiscoveryTrace:
        trace_id = f"trace_{int(time.time()*1000)}_{len(self.traces)+1}"
        trace = DiscoveryTrace(
            trace_id=trace_id,
            recipe=recipe,
            eval_score=eval_score,
            confidence_avg=confidence_avg,
            timestamp=time.time(),
        )
        self.traces.append(trace)
        self.save()
        return trace

    def select_best_recipe(
        self,
        candidate_recipes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        candidates = candidate_recipes or [
            {"epochs": 3, "augment": True, "count": 20},
            {"epochs": 5, "augment": True, "count": 30},
            {"epochs": 2, "augment": False, "count": 15},
        ]
        sim = ReplaySimulator(self.traces)
        scored = [(r, sim.estimate_recipe_score(r)) for r in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
