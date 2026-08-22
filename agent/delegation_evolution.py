"""Delegation Evolution — AIDE²-inspired multi-strategy agent dispatch (skeleton).

Inspired by AIDE²'s bandit + greedy + fork search strategy: when
``delegate_task`` is called with ``evolution=True``, it dispatches multiple
subagents with different strategies, scores their results, and forks to new
strategies when stagnation is detected.

⚠️  STUB IMPLEMENTATION WARNING ⚠️
The execution paths that *dispatch* a strategy to a subagent
(``_dispatch_strategy``, ``_fork_strategy``) are intentional stubs that
raise ``NotImplementedError`` until Phase 3 (after the real EvalHarness
execution path lands). They previously returned fake scores via
``random.uniform``; that has been removed.

Working parts (fully functional):
- Bandit strategy selection from historical scores
- Stagnation detection (consecutive non-improvements)
- Strategy fork logic (pick untried strategy on stagnation)
- Lineage tracking (parent → child relationships)
- Persistent state for both strategy scores and lineage history

Key principles from AIDE² (for the full Phase 3 implementation):
- Bandit dispatch: weight strategies by historical success
- Stagnation detection: if N consecutive runs show no improvement, fork
- Strategy fork: take the best result and try a completely new approach
- Lineage tracking: parent-child relationships between attempts

Usage (Phase 3):
    from agent.delegation_evolution import DelegationEvolution
    de = DelegationEvolution(hermes_home=Path.home() / ".hermes")

    # Evolve a task — strategy selection works; dispatch raises until Phase 3
    results = await de.evolve_task(
        goal="Optimize this script",
        max_agents=3,
        stagnation_threshold=3,
    )
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Result from one strategy's execution."""

    strategy: str
    score: float = 0.0
    cost_usd: float = 0.0
    duration_sec: float = 0.0
    output: str = ""
    error: str = ""
    lineage_id: str = ""
    parent_lineage: str = ""
    improved: bool = False


@dataclass
class EvolutionResult:
    """Result of the full evolution cycle."""

    task_id: str
    best_strategy: str = ""
    best_score: float = 0.0
    total_attempts: int = 0
    total_cost_usd: float = 0.0
    stagnation_detected: bool = False
    fork_performed: bool = False
    results: List[StrategyResult] = field(default_factory=list)
    duration_sec: float = 0.0


# Strategy templates for different approaches
STRATEGY_TEMPLATES = {
    "aggressive": {
        "role": "You are an aggressive optimizer. Make bold changes, refactor aggressively, and prioritize performance over safety.",
        "name": "aggressive",
    },
    "conservative": {
        "role": "You are a conservative optimizer. Make minimal changes, preserve existing behavior, and prioritize correctness over novelty.",
        "name": "conservative",
    },
    "creative": {
        "role": "You are a creative optimizer. Think outside the box, try unconventional approaches, and explore novel solutions.",
        "name": "creative",
    },
    "analytical": {
        "role": "You are an analytical optimizer. Break down the problem systematically, analyze each component, and build an optimal solution from first principles.",
        "name": "analytical",
    },
    "minimal": {
        "role": "You are a minimal optimizer. Find the smallest possible change that produces the biggest improvement.",
        "name": "minimal",
    },
}


class DelegationEvolution:
    """Multi-strategy delegation with stagnation detection and strategy fork.

    Implements AIDE's bandit + greedy + fork search for agent dispatch:
    1. Dispatch multiple subagents with different strategies
    2. Score results using objective criteria
    3. Track lineage (parent-child relationships)
    4. Detect stagnation (N consecutive runs without improvement)
    5. Fork: when stagnated, take best result and try new strategy
    """

    def __init__(
        self,
        hermes_home: Optional[Path] = None,
        default_strategies: Optional[List[str]] = None,
        max_history_per_strategy: int = 100,
        max_lineage_history: int = 500,
    ):
        self.hermes_home = hermes_home or Path.home() / ".hermes"
        self.state_dir = self.hermes_home / "state" / "delegation_evolution"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.default_strategies = default_strategies or [
            "aggressive",
            "conservative",
            "creative",
        ]
        self._strategy_scores: Dict[str, List[float]] = {}
        self._lineage_history: Dict[str, List[StrategyResult]] = {}
        self._stagnation_counter: int = 0
        self._last_best_score: float = 0.0
        self.max_history_per_strategy = max_history_per_strategy
        self.max_lineage_history = max_lineage_history

        self._load_state()

    def _load_state(self) -> None:
        """Load historical strategy performance data."""
        scores_file = self.state_dir / "strategy_scores.json"
        if scores_file.exists():
            try:
                self._strategy_scores = json.loads(
                    scores_file.read_text(encoding="utf-8"),
                )
            except (OSError, ValueError) as e:
                logger.warning(
                    "DelegationEvolution: failed to load strategy_scores.json: %s",
                    e,
                )
                self._strategy_scores = {}

        lineage_file = self.state_dir / "lineage_history.json"
        if lineage_file.exists():
            try:
                data = json.loads(lineage_file.read_text(encoding="utf-8"))
                for task_id, results in data.items():
                    self._lineage_history[task_id] = [
                        StrategyResult(**r) for r in results
                    ]
            except (OSError, ValueError, TypeError) as e:
                logger.warning(
                    "DelegationEvolution: failed to load lineage_history.json: %s",
                    e,
                )

    def _save_state(self) -> None:
        """Persist state for future runs.

        Disk errors are logged but never raised: a transient write failure
        should not lose an in-progress evolution cycle.
        """
        scores_file = self.state_dir / "strategy_scores.json"
        try:
            scores_file.write_text(
                json.dumps(self._strategy_scores, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(
                "DelegationEvolution: failed to write strategy_scores.json: %s",
                e,
            )

        lineage_data = {
            task_id: [r.__dict__ for r in results]
            for task_id, results in self._lineage_history.items()
        }
        lineage_file = self.state_dir / "lineage_history.json"
        try:
            lineage_file.write_text(
                json.dumps(lineage_data, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(
                "DelegationEvolution: failed to write lineage_history.json: %s",
                e,
            )

    async def evolve_task(
        self,
        goal: str,
        max_agents: int = 3,
        stagnation_threshold: int = 3,
        context: Optional[str] = None,
    ) -> EvolutionResult:
        """Evolve a task through multi-strategy dispatch.

        Args:
            goal: What the subagents should accomplish
            max_agents: Max concurrent subagents (default 3)
            stagnation_threshold: Consecutive non-improvements before fork
            context: Background information for subagents

        Returns:
            EvolutionResult with best strategy and all results
        """
        task_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.info(
            "Delegation evolution: starting task %s (goal='%s')",
            task_id,
            goal[:50],
        )

        result = EvolutionResult(task_id=task_id)

        # Select strategies based on bandit weights
        strategies = self._select_strategies(max_agents)
        logger.info("Delegation evolution: selected strategies: %s", strategies)

        # Dispatch subagents (in production, this calls delegate_task)
        for strategy in strategies:
            strat_result = await self._dispatch_strategy(
                strategy,
                goal,
                context,
                task_id,
            )
            result.results.append(strat_result)
            result.total_attempts += 1
            result.total_cost_usd += strat_result.cost_usd

        # Find best result
        if result.results:
            best = max(result.results, key=lambda r: r.score)
            result.best_strategy = best.strategy
            result.best_score = best.score

            # Check for improvement
            improved = best.score > self._last_best_score
            for r in result.results:
                r.improved = improved

            if improved:
                self._last_best_score = best.score
                self._stagnation_counter = 0
            else:
                self._stagnation_counter += 1

            # Check for stagnation
            if self._stagnation_counter >= stagnation_threshold:
                result.stagnation_detected = True
                logger.warning(
                    "Delegation evolution: stagnation detected (%d consecutive non-improvements)",
                    self._stagnation_counter,
                )

                # Fork: try a new strategy
                fork_result = await self._fork_strategy(
                    best,
                    goal,
                    context,
                    task_id,
                )
                result.results.append(fork_result)
                result.fork_performed = True
                result.total_attempts += 1

                if fork_result.score > best.score:
                    result.best_strategy = fork_result.strategy
                    result.best_score = fork_result.score
                    self._last_best_score = fork_result.score
                    self._stagnation_counter = 0

            # Update strategy scores
            for r in result.results:
                if r.strategy not in self._strategy_scores:
                    self._strategy_scores[r.strategy] = []
                self._strategy_scores[r.strategy].append(r.score)
                # Trim per-strategy history to bound growth on long-running
                # deployments where the same strategy may be retried many
                # times.
                if (
                    len(self._strategy_scores[r.strategy])
                    > self.max_history_per_strategy
                ):
                    self._strategy_scores[r.strategy] = self._strategy_scores[
                        r.strategy
                    ][-self.max_history_per_strategy :]

            # Store lineage, then evict oldest entries once we exceed the
            # lineage history cap. We evict in insertion order which roughly
            # matches task completion order.
            self._lineage_history[task_id] = result.results
            if len(self._lineage_history) > self.max_lineage_history:
                excess = len(self._lineage_history) - self.max_lineage_history
                for old_task_id in list(self._lineage_history)[:excess]:
                    del self._lineage_history[old_task_id]

        result.duration_sec = time.time() - start_time
        self._save_state()

        logger.info(
            "Delegation evolution: task %s done — best=%s (score=%.2f), attempts=%d, fork=%s",
            task_id,
            result.best_strategy,
            result.best_score,
            result.total_attempts,
            result.fork_performed,
        )

        return result

    def _select_strategies(self, max_agents: int) -> List[str]:
        """Select strategies using bandit-weighted selection.

        Strategies with higher historical scores get higher probability.
        New strategies get exploration bonus.
        """
        available = list(STRATEGY_TEMPLATES.keys())

        if not self._strategy_scores:
            # No history: use default strategies
            return self.default_strategies[:max_agents]

        # Calculate average scores
        avg_scores = {
            s: sum(scores) / len(scores) if scores else 0.5
            for s, scores in self._strategy_scores.items()
        }

        # Sort by score descending
        sorted_strategies = sorted(
            available,
            key=lambda s: avg_scores.get(s, 0.5),
            reverse=True,
        )

        # Add exploration bonus for untried strategies
        for s in available:
            if s not in self._strategy_scores:
                avg_scores[s] = 0.6  # Exploration bonus

        # Select top N
        selected = []
        for s in sorted_strategies:
            if len(selected) >= max_agents:
                break
            selected.append(s)

        return selected[:max_agents]

    async def _dispatch_strategy(
        self,
        strategy: str,
        goal: str,
        context: Optional[str],
        task_id: str,
    ) -> StrategyResult:
        """Dispatch a single strategy to a subagent and return the result.

        ⚠️  STUB: raises ``NotImplementedError`` until Phase 3. The real
        implementation will call ``delegate_task`` (or equivalent
        subagent dispatch) with ``strategy``'s role prompt + ``goal``,
        then return the subagent's measured score and cost.

        Tests and callers that need a non-stub result should monkeypatch
        this method.
        """
        raise NotImplementedError(
            f"DelegationEvolution._dispatch_strategy is a stub until Phase 3. "
            f"Cannot dispatch strategy {strategy!r} for task {task_id!r} "
            f"without a real subagent runtime. "
            f"See docs/aide-squared-roadmap.md."
        )

    async def _fork_strategy(
        self,
        best_result: StrategyResult,
        goal: str,
        context: Optional[str],
        task_id: str,
    ) -> StrategyResult:
        """Fork to a new strategy when stagnation is detected.

        ⚠️  STUB: raises ``NotImplementedError`` until Phase 3. The real
        implementation will pick an untried strategy, dispatch it via
        ``_dispatch_strategy``, and return the measured result. The
        previous random.uniform stub has been removed.
        """
        raise NotImplementedError(
            f"DelegationEvolution._fork_strategy is a stub until Phase 3. "
            f"Cannot fork from {best_result.strategy!r} for task "
            f"{task_id!r} without a real subagent runtime. "
            f"See docs/aide-squared-roadmap.md."
        )

    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance stats for all strategies."""
        stats = {}
        for strategy, scores in self._strategy_scores.items():
            if not scores:
                continue
            stats[strategy] = {
                "avg_score": round(sum(scores) / len(scores), 3),
                "max_score": round(max(scores), 3),
                "min_score": round(min(scores), 3),
                "attempts": len(scores),
            }
        return stats

    def get_lineage(self, task_id: str) -> List[StrategyResult]:
        """Get the lineage of attempts for a task."""
        return list(self._lineage_history.get(task_id, []))
