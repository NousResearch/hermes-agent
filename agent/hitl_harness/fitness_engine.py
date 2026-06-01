"""fitness_engine.py — Executable Fitness Function Runner (Phase 2).

Central component that turns a GOAL.md (or in-memory FitnessGoal) into
a runnable, dual-scored evaluation.

Key features from the research:
- Parses GOAL.md YAML frontmatter + action catalog
- Runs deterministic + deepeval-style agentic metrics
- Always returns dual scores (primary + instrument_trust)
- Exposes clean `evaluate(trajectory, context) -> FitnessResult`
- Designed to be called from HITLHarness and future GOAL.md skill
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from agent.hitl_harness.deepeval_adapter import DeepevalAdapter, default_adapter
from agent.hitl_harness.fitness_wizard import FitnessFunctionWizard, FitnessGoal


@dataclass
class FitnessResult:
    """Result of a single fitness evaluation."""

    primary_score: float
    instrument_trust_score: float
    combined_score: float
    metric_breakdown: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    action_catalog: List[str] = field(default_factory=list)
    goal_id: Optional[str] = None


class MetricProvider(Protocol):
    def compute(self, trajectory: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[float]:
        ...


class FitnessEngine:
    """Executable fitness function runner.

    Usage:
        engine = FitnessEngine(goal=goal, deepeval_adapter=adapter)
        result = engine.evaluate(trajectory, context={"session_id": "..."})
    """

    def __init__(
        self,
        goal: Optional[FitnessGoal] = None,
        deepeval_adapter: Optional[DeepevalAdapter] = None,
        wizard: Optional[FitnessFunctionWizard] = None,
        swe_bench_adapter: Optional[Any] = None,
    ):
        self.goal = goal
        self.deepeval = deepeval_adapter or default_adapter
        self.wizard = wizard
        self.swe_bench = swe_bench_adapter
        self._metric_providers: Dict[str, MetricProvider] = {}

        # Register deepeval metrics by default
        if self.deepeval:
            for name, metric in self.deepeval._metrics.items():
                self.register_metric_provider(name, metric)

    def register_metric_provider(self, name: str, provider: MetricProvider) -> None:
        self._metric_providers[name] = provider

    def load_goal(self, goal: FitnessGoal) -> None:
        """Load or replace the active fitness goal."""
        self.goal = goal

    def evaluate(
        self,
        trajectory: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> FitnessResult:
        """Run the full fitness evaluation and return dual-scored result."""
        context = context or {}
        metric_scores: Dict[str, float] = {}

        # 1. Run all registered metric providers (deepeval + any custom)
        for name, provider in self._metric_providers.items():
            try:
                score = provider.compute(trajectory, context)
                if score is not None:
                    metric_scores[name] = round(float(score), 4)
            except Exception:
                metric_scores[name] = 0.0

        # 2. Compute primary score from goal weights (if goal exists)
        primary_score = 0.5
        if self.goal and self.goal.composite_weights:
            total_weight = 0.0
            weighted_sum = 0.0
            for prop in self.goal.properties:
                name = prop.get("name")
                weight = self.goal.composite_weights.get(name, 0.1)
                if name in metric_scores:
                    weighted_sum += metric_scores[name] * weight
                    total_weight += weight
            if total_weight > 0:
                primary_score = weighted_sum / total_weight

        # 3. Instrument trust = average of deepeval/agentic metrics
        agentic_names = {"TaskCompletion", "ToolCorrectness", "StepEfficiency", "PlanAdherence", "GoalAccuracy"}
        agentic_scores = [v for k, v in metric_scores.items() if k in agentic_names]
        instrument_trust = sum(agentic_scores) / len(agentic_scores) if agentic_scores else 0.75

        # 4. Combined dual score
        combined = round(primary_score * 0.7 + instrument_trust * 0.3, 4)

        # 5. SWE-bench resolution rate (if adapter available)
        if self.swe_bench and context and context.get("swe_bench_issues"):
            try:
                corpus_result = self.swe_bench.evaluate_corpus(
                    context["swe_bench_issues"],
                    context.get("harness"),
                )
                metric_scores["swe_bench_resolution_rate"] = corpus_result["resolution_rate"]
            except Exception as e:
                logger.debug("SWE-bench evaluation failed: %s", e)

        # 6. Action catalog from goal
        action_catalog = []
        if self.goal:
            action_catalog = [p.get("name", "") for p in self.goal.properties]

        return FitnessResult(
            primary_score=round(primary_score, 4),
            instrument_trust_score=round(instrument_trust, 4),
            combined_score=combined,
            metric_breakdown=metric_scores,
            explanation=f"Fitness evaluated with {len(metric_scores)} metrics",
            action_catalog=action_catalog,
            goal_id=self.goal.id if self.goal else None,
        )

    def get_action_suggestions(self, result: FitnessResult) -> List[str]:
        """Return ranked suggestions from the action catalog based on weak metrics."""
        if not self.goal:
            return []
        weak = sorted(
            result.metric_breakdown.items(),
            key=lambda x: x[1]
        )[:3]
        return [name for name, _ in weak]