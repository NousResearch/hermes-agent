"""deepeval_adapter.py — Thin, dependency-free adapter for deepeval-style agentic metrics.

Implements the core production-relevant metrics from deepeval without requiring the
external package (falls back to pure Python heuristics + optional G-Eval hook).

Metrics implemented:
- TaskCompletion
- ToolCorrectness
- StepEfficiency
- PlanAdherence
- GoalAccuracy
- ArgumentCorrectness
- G-Eval (custom LLM-as-a-judge criteria via callback)

All metrics return a float in [0.0, 1.0] (or None if inapplicable).
Registered as a provider for the future FitnessEngine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol


class MetricProtocol(Protocol):
    name: str

    def compute(self, trajectory: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[float]:
        ...


@dataclass
class DeepevalMetricResult:
    name: str
    score: float
    explanation: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


class DeepevalAdapter:
    """Registry + executor for deepeval-style metrics."""

    def __init__(self, geval_callback: Optional[Callable[[str, Dict], float]] = None):
        self._metrics: Dict[str, MetricProtocol] = {}
        self.geval_callback = geval_callback  # Optional external LLM judge
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(TaskCompletionMetric())
        self.register(ToolCorrectnessMetric())
        self.register(StepEfficiencyMetric())
        self.register(PlanAdherenceMetric())
        self.register(GoalAccuracyMetric())
        self.register(ArgumentCorrectnessMetric())

    def register(self, metric: MetricProtocol) -> None:
        self._metrics[metric.name] = metric

    def compute_all(
        self, trajectory: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DeepevalMetricResult]:
        results: Dict[str, DeepevalMetricResult] = {}
        for name, metric in self._metrics.items():
            try:
                score = metric.compute(trajectory, context)
                if score is not None:
                    results[name] = DeepevalMetricResult(
                        name=name,
                        score=round(float(score), 4),
                        explanation=f"{name} computed via deepeval_adapter",
                    )
            except Exception as e:
                results[name] = DeepevalMetricResult(
                    name=name,
                    score=0.0,
                    explanation=f"Error computing {name}: {e}",
                )
        return results

    def geval(self, criteria: str, data: Dict[str, Any]) -> Optional[float]:
        """Run custom G-Eval style criteria if a callback is provided."""
        if self.geval_callback:
            try:
                return self.geval_callback(criteria, data)
            except Exception:
                return None
        return None


# ------------------------------------------------------------------
# Concrete metric implementations (heuristic + production-friendly)
# ------------------------------------------------------------------

class TaskCompletionMetric:
    name = "TaskCompletion"

    def compute(self, trajectory: Dict[str, Any], context: Optional[Dict] = None) -> Optional[float]:
        # Heuristic: ratio of successful tool calls or final response containing "success"/"completed"
        tools = trajectory.get("tool_results", [])
        if not tools:
            return None
        successes = sum(1 for t in tools if "error" not in str(t).lower() and "fail" not in str(t).lower())
        return successes / max(len(tools), 1)


class ToolCorrectnessMetric:
    name = "ToolCorrectness"

    def compute(self, trajectory: Dict[str, Any], context: Optional[Dict] = None) -> Optional[float]:
        tools = trajectory.get("tool_calls", [])
        if not tools:
            return None
        # Simple heuristic: argument schema presence + no obvious type errors
        correct = 0
        for call in tools:
            args = call.get("arguments", {})
            if isinstance(args, dict) and args:
                correct += 1
        return correct / max(len(tools), 1)


class StepEfficiencyMetric:
    name = "StepEfficiency"

    def compute(self, trajectory: Dict[str, Any], context: Optional[Dict] = None) -> Optional[float]:
        steps = trajectory.get("steps", len(trajectory.get("messages", [])))
        target_steps = context.get("target_steps", 8) if context else 8
        if steps <= 0:
            return None
        efficiency = max(0.0, min(1.0, target_steps / steps))
        return efficiency


class PlanAdherenceMetric:
    name = "PlanAdherence"

    def compute(self, trajectory: Dict[str, Any], context: Optional[Dict] = None) -> Optional[float]:
        plan = trajectory.get("plan", [])
        actions = trajectory.get("actions_taken", [])
        if not plan:
            return None
        adherence = sum(1 for a in actions if any(p.lower() in str(a).lower() for p in plan)) / max(len(plan), 1)
        return min(1.0, adherence)


class GoalAccuracyMetric:
    name = "GoalAccuracy"

    def compute(self, trajectory: Dict[str, Any], context: Optional[Dict] = None) -> Optional[float]:
        goal = (context or {}).get("goal", "")
        final_response = str(trajectory.get("final_response", ""))
        if not goal:
            return None
        # Simple keyword overlap (production would use embedding similarity)
        goal_words = set(re.findall(r"\w+", goal.lower()))
        resp_words = set(re.findall(r"\w+", final_response.lower()))
        overlap = len(goal_words & resp_words) / max(len(goal_words), 1)
        return min(1.0, overlap)


class ArgumentCorrectnessMetric:
    name = "ArgumentCorrectness"

    def compute(self, trajectory: Dict[str, Any], context: Optional[Dict] = None) -> Optional[float]:
        calls = trajectory.get("tool_calls", [])
        if not calls:
            return None
        correct = 0
        for call in calls:
            args = call.get("arguments", {})
            if isinstance(args, dict) and all(isinstance(v, (str, int, float, bool, list, dict)) for v in args.values()):
                correct += 1
        return correct / max(len(calls), 1)


# Convenience singleton
default_adapter = DeepevalAdapter()