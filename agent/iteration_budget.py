"""Per-agent iteration budget — thread-safe consume/refund counter.

Extracted from ``run_agent.py``.  Each ``AIAgent`` instance (parent or
subagent) holds an :class:`IterationBudget`; the parent's cap comes from
``max_iterations`` (default 90), each subagent's cap comes from
``delegation.max_iterations`` (default 50).

``run_agent`` re-exports ``IterationBudget`` so existing
``from run_agent import IterationBudget`` imports keep working unchanged.
"""

from __future__ import annotations

import threading


class IterationBudget:
    """Thread-safe iteration counter for an agent.

    Each agent (parent or subagent) gets its own ``IterationBudget``.
    The parent's budget is capped at ``max_iterations`` (default 90).
    Each subagent gets an independent budget capped at
    ``delegation.max_iterations`` (default 50) — this means total
    iterations across parent + subagents can exceed the parent's cap.
    Users control the per-subagent limit via ``delegation.max_iterations``
    in config.yaml.

    A non-positive max_total means "unbounded".  This is an explicit operator
    escape hatch for long-running/durable sessions: usage is still counted for
    observability, but ``consume()`` never denies an iteration.

    ``execute_code`` (programmatic tool calling) iterations are refunded via
    :meth:`refund` so they don't eat into the budget.
    """

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    @property
    def is_unbounded(self) -> bool:
        """True when this budget has no hard iteration ceiling."""
        if isinstance(self.max_total, bool):
            return False
        try:
            return int(self.max_total) <= 0
        except (TypeError, ValueError):
            return False

    def consume(self) -> bool:
        """Try to consume one iteration.  Returns True if allowed."""
        with self._lock:
            if not self.is_unbounded and self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """Give back one iteration (e.g. for execute_code turns)."""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self):
        with self._lock:
            if self.is_unbounded:
                return float("inf")
            return max(0, self.max_total - self._used)


def is_unbounded_iteration_limit(max_iterations: int) -> bool:
    """Return True when a configured max-iteration value disables the cap."""
    if isinstance(max_iterations, bool):
        return False
    try:
        return int(max_iterations) <= 0
    except (TypeError, ValueError):
        return False


__all__ = ["IterationBudget", "is_unbounded_iteration_limit"]
