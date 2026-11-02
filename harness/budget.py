"""Multi-dimensional execution budgets with hard stops.

Hermes already caps agent iterations (:class:`IterationBudget`); the harness
additionally governs tool calls, retries, replans, tokens, and elapsed time.
A hard limit ends the task — it is never silently exceeded.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List

from .state import ExecutionBudget


@dataclass
class BudgetUsage:
    tool_calls: int = 0
    retries: int = 0
    replans: int = 0
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "tool_calls": self.tool_calls,
            "retries": self.retries,
            "replans": self.replans,
            "iterations": self.iterations,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


_LIMIT_CHECKS = (
    ("max_tool_calls", "tool_calls", "tool calls"),
    ("max_retries", "retries", "retries"),
    ("max_replans", "replans", "replans"),
    ("max_iterations", "iterations", "iterations"),
)


class BudgetGovernor:
    """Thread-safe usage counters against an :class:`ExecutionBudget`."""

    def __init__(self, budget: ExecutionBudget) -> None:
        self.budget = budget
        self.usage = BudgetUsage()
        self._lock = threading.Lock()

    def consume_tool_call(self) -> bool:
        return self._consume("tool_calls", self.budget.max_tool_calls)

    def consume_retry(self) -> bool:
        return self._consume("retries", self.budget.max_retries)

    def consume_replan(self) -> bool:
        return self._consume("replans", self.budget.max_replans)

    def consume_iteration(self) -> bool:
        return self._consume("iterations", self.budget.max_iterations)

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.usage.input_tokens += max(0, input_tokens)
            self.usage.output_tokens += max(0, output_tokens)

    def _consume(self, attr: str, limit: int) -> bool:
        with self._lock:
            if limit > 0 and getattr(self.usage, attr) >= limit:
                return False
            setattr(self.usage, attr, getattr(self.usage, attr) + 1)
            return True

    def exhausted(self) -> List[str]:
        """Names of exhausted dimensions; empty means budget remains."""
        used = self.usage
        hit = [
            label
            for limit_attr, use_attr, label in _LIMIT_CHECKS
            if getattr(self.budget, limit_attr) > 0
            and getattr(used, use_attr) >= getattr(self.budget, limit_attr)
        ]
        max_elapsed = self.budget.max_elapsed_seconds
        if max_elapsed is not None and used.elapsed_seconds >= max_elapsed:
            hit.append("elapsed time")
        return hit
