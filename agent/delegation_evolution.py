"""Delegation Evolution — AIDE²-inspired multi-strategy agent dispatch.

Inspired by AIDE's bandit + greedy + fork search strategy: when delegate_task
is called with evolution=True, it dispatches multiple subagents with different
strategies, scores their results, and forks to new strategies when stagnation
is detected.

Key principles from AIDE²:
- Bandit dispatch: weight strategies by historical success
- Stagnation detection: if N consecutive runs show no improvement, fork
- Strategy fork: take the best result and try a completely new approach
- Lineage tracking: parent-child relationships between attempts

Usage:
    from agent.delegation_evolution import DelegationEvolution
    de = DelegationEvolution(hermes_home=Path.home() / ".hermes")

    # Evolve a task
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
    ):
        self.hermes_home = hermes_home or Path.home() / ".hermes"
        self.state_dir = self.hermes_home / "state" / "delegation_evolution"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.default_strategies = default_strategies or ["aggressive", "conservative", "creative"]
        self._strategy_scores: Dict[str, List[float]] = {}
        self._lineage_history: Dict[str, List[StrategyResult]] = {}
        self._stagnation_counter: int = 0
        self._last_best_score: float = 0.0

        self._load_state()

    def _load_state(self) -> None:
        """Load historical strategy performance data."""
        scores_file = self.state_dir / "strategy_scores.json"
        if scores_file.exists():
            self._strategy_scores = json.loads(scores_file.read_text())

        lineage_file = self.state_dir / "lineage_history.json"
        if lineage_file.exists():
            data = json.loads(lineage_file.read_text())
            for task_id, results in data.items():
                self._lineage_history[task_id] = [
                    StrategyResult(**r) for r in results
                ]

    def _save_state(self) -> None:
        """Persist state for future runs."""
        scores_file = self.state_dir / "strategy_scores.json"
        scores_file.write_text(json.dumps(self._strategy_scores, indent=2))

        lineage_data = {
            task_id: [r.__dict__ for r in results]
            for task_id, results in self._lineage_history.items()
        }
        lineage_file = self.state_dir / "lineage_history.json"
        lineage_file.write_text(json.dumps(lineage_data, indent=2))

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
            task_id, goal[:50],
        )

        result = EvolutionResult(task_id=task_id)

        # Select strategies based on bandit weights
        strategies = self._select_strategies(max_agents)
        logger.info("Delegation evolution: selected strategies: %s", strategies)

        # Dispatch subagents (in production, this calls delegate_task)
        for strategy in strategies:
            strat_result = await self._dispatch_strategy(
                strategy, goal, context, task_id,
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
                    best, goal, context, task_id,
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

            # Store lineage
            self._lineage_history[task_id] = result.results

        result.duration_sec = time.time() - start_time
        self._save_state()

        logger.info(
            "Delegation evolution: task %s done — best=%s (score=%.2f), attempts=%d, fork=%s",
            task_id, result.best_strategy, result.best_score,
            result.total_attempts, result.fork_performed,
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
        """Dispatch a single strategy (simulated).

        In production, this would call delegate_task with the strategy's
        role and goal.
        """
        start = time.time()
        template = STRATEGY_TEMPLATES.get(strategy, {})
        role = template.get("role", "")

        lineage_id = f"{task_id}-{strategy}"

        # Simulate execution with strategy-specific characteristics
        import random

        # Different strategies have different success profiles
        if strategy == "aggressive":
            score = random.uniform(0.4, 0.95)  # High variance
            cost = random.uniform(0.1, 0.5)
        elif strategy == "conservative":
            score = random.uniform(0.6, 0.85)  # Low variance, moderate
            cost = random.uniform(0.05, 0.2)
        elif strategy == "creative":
            score = random.uniform(0.3, 0.9)  # High variance, high ceiling
            cost = random.uniform(0.2, 0.6)
        elif strategy == "analytical":
            score = random.uniform(0.5, 0.8)  # Moderate variance
            cost = random.uniform(0.15, 0.4)
        else:  # minimal
            score = random.uniform(0.5, 0.75)
            cost = random.uniform(0.05, 0.15)

        return StrategyResult(
            strategy=strategy,
            score=round(score, 3),
            cost_usd=round(cost, 4),
            duration_sec=round(time.time() - start, 3),
            output=f"Strategy '{strategy}' completed the task.",
            lineage_id=lineage_id,
        )

    async def _fork_strategy(
        self,
        best_result: StrategyResult,
        goal: str,
        context: Optional[str],
        task_id: str,
    ) -> StrategyResult:
        """Fork to a new strategy when stagnation is detected.

        Takes the best result so far and tries a completely different approach.
        """
        # Pick a strategy that hasn't been tried yet
        tried = {r.strategy for r in [best_result]}
        available = [s for s in STRATEGY_TEMPLATES if s not in tried]

        if not available:
            # All strategies tried — retry best with variation
            fork_strategy = f"{best_result.strategy}-variant"
        else:
            import random
            fork_strategy = random.choice(available)

        lineage_id = f"{task_id}-fork-{fork_strategy}"

        logger.info(
            "Delegation evolution: forking to '%s' (from '%s')",
            fork_strategy, best_result.strategy,
        )

        # Simulate forked execution
        import random
        # Fork has 50% chance of breaking stagnation
        score = random.uniform(0.6, 0.95) if random.random() < 0.5 else random.uniform(0.3, 0.6)
        cost = random.uniform(0.1, 0.4)

        return StrategyResult(
            strategy=fork_strategy,
            score=round(score, 3),
            cost_usd=round(cost, 4),
            output=f"Fork strategy '{fork_strategy}' completed.",
            lineage_id=lineage_id,
            parent_lineage=best_result.lineage_id,
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
