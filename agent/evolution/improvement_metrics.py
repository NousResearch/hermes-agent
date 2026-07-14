"""Self-Improvement Metrics — statistical proof that HAEE makes agents better.

Two SOTA components that close the research gap:

1. ImprovementTracker: Measures agent performance over time with statistical
   significance. Uses paired testing (Wilcoxon signed-rank) to determine whether
   observed improvements are real or noise. Produces publishable metrics.

2. LearnedFailurePatterns: Trains a lightweight logistic regression model on
   historical failure data to predict which trajectory patterns lead to failure.
   Goes beyond deterministic rules — learns from actual outcomes.

Research basis:
  - EvoPolicyGym (2026): budget-constrained improvement measurement
  - GAIA critic model (2026): learned evaluation from trajectory data
  - UI-Genie (NeurIPS 2025): self-improving reward models
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction for learned models
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryFeatures:
    """Feature vector extracted from a trajectory for ML models."""
    total_turns: int = 0
    total_tool_calls: int = 0
    unique_tools: int = 0
    error_count: int = 0
    tool_error_rate: float = 0.0
    verification_attempted: bool = False
    premature_completion_score: float = 0.0  # 0-1: how likely premature
    loop_detected: bool = False
    tool_diversity: float = 0.0  # unique_tools / total_tool_calls
    turns_since_last_tool: int = 0  # turns since last action

    def to_vector(self) -> List[float]:
        return [
            float(self.total_turns),
            float(self.total_tool_calls),
            float(self.unique_tools),
            float(self.error_count),
            self.tool_error_rate,
            1.0 if self.verification_attempted else 0.0,
            self.premature_completion_score,
            1.0 if self.loop_detected else 0.0,
            self.tool_diversity,
            float(self.turns_since_last_tool),
        ]

    @classmethod
    def from_trajectory(cls, trace_json: str) -> "TrajectoryFeatures":
        """Extract features from a trajectory trace."""
        try:
            trace = json.loads(trace_json)
        except (json.JSONDecodeError, TypeError):
            return cls()

        steps = trace.get("steps", [])
        errors = trace.get("errors", [])

        # Count tool calls
        tool_calls = [s for s in steps if s.get("type") == "tool_execution"]
        tool_names = [s.get("extra", {}).get("tool", "") for s in tool_calls]
        unique_tools = len(set(t for t in tool_names if t))

        # Error analysis
        error_count = len(errors)
        tool_error_count = sum(1 for s in tool_calls if s.get("status") != "success")
        tool_error_rate = tool_error_count / len(tool_calls) if tool_calls else 0.0

        # Verification detection
        model_calls = [s for s in steps if s.get("type") == "model_call"]
        last_model = model_calls[-1] if model_calls else {}
        premature = 0.0
        if last_model and not last_model.get("extra", {}).get("tool_calls"):
            # Last model call had no tool calls — might be premature
            if error_count > 0 or tool_error_rate > 0:
                premature = 0.8
            else:
                premature = 0.3

        # Loop detection
        tool_seq = tool_names[-9:]  # Last 9 tool calls
        loop_detected = False
        if len(tool_seq) >= 6:
            for i in range(len(tool_seq) - 3):
                if tool_seq[i:i+3] == tool_seq[i+3:i+6]:
                    loop_detected = True
                    break

        # Turns since last tool
        turns_since_last = 0
        for s in reversed(steps):
            if s.get("type") == "tool_execution":
                break
            turns_since_last += 1

        return cls(
            total_turns=trace.get("total_turns", len(steps)),
            total_tool_calls=len(tool_calls),
            unique_tools=unique_tools,
            error_count=error_count,
            tool_error_rate=tool_error_rate,
            verification_attempted=any(
                t in ("terminal", "read_file", "search_files")
                for t in tool_names
            ),
            premature_completion_score=premature,
            loop_detected=loop_detected,
            tool_diversity=unique_tools / len(tool_calls) if tool_calls else 0.0,
            turns_since_last_tool=turns_since_last,
        )


# ---------------------------------------------------------------------------
# Lightweight logistic regression for failure prediction
# ---------------------------------------------------------------------------


class LearnedFailurePredictor:
    """Logistic regression trained on historical trajectories.

    Learns which feature patterns predict failure. More accurate than
    deterministic rules because it adapts to the specific agent + task domain.

    Uses online gradient descent — updates with each new trajectory.
    No external ML libraries needed. ~50 lines of math.
    """

    def __init__(self, n_features: int = 10, learning_rate: float = 0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.n_updates = 0
        self.accuracy_window: List[bool] = []  # Last 20 predictions

    def predict(self, features: TrajectoryFeatures) -> float:
        """Predict failure probability (0.0-1.0)."""
        x = features.to_vector()
        z = self.bias + sum(w * v for w, v in zip(self.weights, x))
        return 1.0 / (1.0 + math.exp(-z))  # Sigmoid

    def update(self, features: TrajectoryFeatures, actual_failure: bool) -> None:
        """Online gradient descent update."""
        predicted = self.predict(features)
        error = (1.0 if actual_failure else 0.0) - predicted
        x = features.to_vector()

        # Gradient step
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * x[i]
        self.bias += self.learning_rate * error

        self.n_updates += 1

        # Track accuracy
        correct = (predicted > 0.5) == actual_failure
        self.accuracy_window.append(correct)
        if len(self.accuracy_window) > 20:
            self.accuracy_window.pop(0)

    @property
    def accuracy(self) -> float:
        """Rolling accuracy over last 20 predictions."""
        if not self.accuracy_window:
            return 0.5
        return sum(self.accuracy_window) / len(self.accuracy_window)

    @property
    def is_trained(self) -> bool:
        return self.n_updates >= 10

    def feature_importance(self) -> List[Tuple[str, float]]:
        """Return features ranked by absolute weight (importance)."""
        names = [
            "total_turns", "total_tool_calls", "unique_tools", "error_count",
            "tool_error_rate", "verification_attempted", "premature_completion",
            "loop_detected", "tool_diversity", "turns_since_last_tool",
        ]
        ranked = sorted(
            zip(names, [abs(w) for w in self.weights]),
            key=lambda x: x[1], reverse=True,
        )
        return ranked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "n_updates": self.n_updates,
            "accuracy": self.accuracy,
            "feature_importance": [
                {"feature": name, "importance": imp}
                for name, imp in self.feature_importance()
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearnedFailurePredictor":
        model = cls(n_features=len(d["weights"]))
        model.weights = d["weights"]
        model.bias = d.get("bias", 0.0)
        model.n_updates = d.get("n_updates", 0)
        return model


# ---------------------------------------------------------------------------
# Improvement Tracker — statistical proof of improvement
# ---------------------------------------------------------------------------


@dataclass
class ImprovementRecord:
    """A single before/after measurement."""
    task_name: str
    score_before: float
    score_after: float
    delta: float
    timestamp: str
    run_id: str


class ImprovementTracker:
    """Tracks agent performance over time with statistical significance.

    Uses Wilcoxon signed-rank test (non-parametric, paired) to determine
    whether observed improvements are statistically significant or noise.

    This is the PROOF that HAEE makes agents better.
    """

    def __init__(self):
        self._records: List[ImprovementRecord] = []
        self._task_history: Dict[str, List[float]] = defaultdict(list)
        self._predictor = LearnedFailurePredictor()
        self._load()

    def record_improvement(
        self,
        task_name: str,
        score_before: float,
        score_after: float,
        run_id: str = "",
        trace_json: Optional[str] = None,
        actual_failure: Optional[bool] = None,
    ) -> None:
        """Record a single improvement event."""
        record = ImprovementRecord(
            task_name=task_name,
            score_before=score_before,
            score_after=score_after,
            delta=score_after - score_before,
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
        )
        self._records.append(record)
        self._task_history[task_name].append(score_after)

        # Update learned predictor
        if trace_json and actual_failure is not None:
            features = TrajectoryFeatures.from_trajectory(trace_json)
            self._predictor.update(features, actual_failure)

        self._save()

    # ── Statistical tests ──────────────────────────────────────────────

    def wilcoxon_p_value(self) -> float:
        """Wilcoxon signed-rank test p-value for improvement.

        H0: median delta = 0 (no improvement)
        H1: median delta > 0 (improvement)

        p < 0.05: statistically significant improvement
        p < 0.01: highly significant improvement
        """
        deltas = [r.delta for r in self._records if r.delta != 0]
        if len(deltas) < 5:
            return 1.0  # Not enough data

        # Compute signed ranks
        abs_deltas = [(abs(d), i) for i, d in enumerate(deltas)]
        abs_deltas.sort()

        ranks = [0] * len(deltas)
        i = 0
        while i < len(abs_deltas):
            j = i
            while j < len(abs_deltas) and abs_deltas[j][0] == abs_deltas[i][0]:
                j += 1
            avg_rank = (i + j + 1) / 2.0  # 1-indexed average
            for k in range(i, j):
                ranks[abs_deltas[k][1]] = avg_rank
            i = j

        # W = sum of ranks for positive deltas
        W = sum(rank for rank, d in zip(ranks, deltas) if d > 0)
        n = len(deltas)

        # Normal approximation to Wilcoxon distribution
        mean_W = n * (n + 1) / 4.0
        std_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)

        if std_W == 0:
            return 1.0

        z = (W - mean_W) / std_W

        # One-tailed p-value (improvement only)
        # Normal CDF approximation
        p = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        return max(0.0, min(1.0, p))

    @property
    def is_significant(self) -> bool:
        """Is the improvement statistically significant? (p < 0.05)"""
        return self.wilcoxon_p_value() < 0.05

    @property
    def is_highly_significant(self) -> bool:
        """Is the improvement highly significant? (p < 0.01)"""
        return self.wilcoxon_p_value() < 0.01

    # ── Summary statistics ─────────────────────────────────────────────

    @property
    def mean_improvement(self) -> float:
        if not self._records:
            return 0.0
        return sum(r.delta for r in self._records) / len(self._records)

    @property
    def median_improvement(self) -> float:
        if not self._records:
            return 0.0
        sorted_deltas = sorted(r.delta for r in self._records)
        n = len(sorted_deltas)
        if n % 2 == 0:
            return (sorted_deltas[n//2 - 1] + sorted_deltas[n//2]) / 2.0
        return sorted_deltas[n//2]

    @property
    def improvement_rate(self) -> float:
        """Fraction of tasks that improved."""
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r.delta > 0) / len(self._records)

    @property
    def effect_size(self) -> float:
        """Cohen's d effect size. >0.2 small, >0.5 medium, >0.8 large."""
        deltas = [r.delta for r in self._records]
        if len(deltas) < 2:
            return 0.0
        mean = sum(deltas) / len(deltas)
        variance = sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        return mean / std if std > 0 else 0.0

    # ── Task-specific metrics ──────────────────────────────────────────

    def task_improvement(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get improvement metrics for a specific task."""
        scores = self._task_history.get(task_name, [])
        if len(scores) < 2:
            return None

        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]

        return {
            "task_name": task_name,
            "total_attempts": len(scores),
            "first_half_mean": sum(first_half) / len(first_half),
            "second_half_mean": sum(second_half) / len(second_half),
            "improvement": (
                (sum(second_half) / len(second_half)) -
                (sum(first_half) / len(first_half))
            ),
            "trend": "improving" if second_half and first_half and
                     (sum(second_half)/len(second_half)) > (sum(first_half)/len(first_half))
                     else "stable",
        }

    def tasks_improving(self) -> List[str]:
        """List tasks showing improvement trend."""
        improving = []
        for task_name in self._task_history:
            info = self.task_improvement(task_name)
            if info and info["trend"] == "improving" and info["improvement"] > 0.05:
                improving.append(task_name)
        return sorted(improving)

    # ── Full report ────────────────────────────────────────────────────

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive improvement report."""
        return {
            "summary": {
                "total_records": len(self._records),
                "unique_tasks": len(self._task_history),
                "mean_improvement": self.mean_improvement,
                "median_improvement": self.median_improvement,
                "improvement_rate": self.improvement_rate,
                "effect_size": self.effect_size,
                "effect_size_label": (
                    "large" if self.effect_size > 0.8
                    else "medium" if self.effect_size > 0.5
                    else "small" if self.effect_size > 0.2
                    else "negligible"
                ),
                "wilcoxon_p_value": self.wilcoxon_p_value(),
                "statistically_significant": self.is_significant,
                "highly_significant": self.is_highly_significant,
            },
            "tasks_improving": self.tasks_improving(),
            "task_details": {
                name: self.task_improvement(name)
                for name in self._task_history
                if self.task_improvement(name)
            },
            "learned_predictor": {
                "accuracy": self._predictor.accuracy,
                "n_updates": self._predictor.n_updates,
                "is_trained": self._predictor.is_trained,
                "top_features": [
                    {"feature": f, "importance": round(i, 4)}
                    for f, i in self._predictor.feature_importance()[:5]
                ],
            },
        }

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        path = self._store_path()
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f)
            self._records = [
                ImprovementRecord(**r) for r in data.get("records", [])
            ]
            for r in self._records:
                self._task_history[r.task_name].append(r.score_after)
            if "predictor" in data:
                self._predictor = LearnedFailurePredictor.from_dict(data["predictor"])
        except Exception as e:
            logger.debug("Failed to load improvement tracker: %s", e)

    def _save(self) -> None:
        path = self._store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "records": [r.__dict__ for r in self._records[-500:]],
                    "predictor": self._predictor.to_dict(),
                }, f, indent=2, default=str)
        except Exception as e:
            logger.debug("Failed to save improvement tracker: %s", e)

    def _store_path(self) -> Path:
        return get_hermes_home() / "evolution" / "improvement_metrics.json"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_tracker: Optional[ImprovementTracker] = None


def get_tracker() -> ImprovementTracker:
    global _tracker
    if _tracker is None:
        _tracker = ImprovementTracker()
    return _tracker
