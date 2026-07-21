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

    ``execute_code`` (programmatic tool calling) iterations are refunded via
    :meth:`refund` so they don't eat into the budget.  Refunds are capped
    at ``max_refunds`` (default 15) per turn to prevent infinite loops when
    the model repeatedly calls execute_code without making progress.
    """

    def __init__(self, max_total: int, *, max_refunds: int = 15):
        self.max_total = max_total
        self.max_refunds = max_refunds
        self._used = 0
        self._refunds_given = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Try to consume one iteration.  Returns True if allowed."""
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> bool:
        """Give back one iteration (e.g. for execute_code turns).

        Returns True if the refund was granted, False if the refund cap
        has been reached.  When the cap is hit, the budget erodes normally
        which ensures the loop eventually terminates even when the model
        keeps calling execute_code without progress.
        """
        with self._lock:
            if self._used > 0 and self._refunds_given < self.max_refunds:
                self._used -= 1
                self._refunds_given += 1
                return True
            return False

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)

    @property
    def refunds_remaining(self) -> int:
        """How many refunds are still available this turn."""
        with self._lock:
            return max(0, self.max_refunds - self._refunds_given)


__all__ = ["IterationBudget"]
