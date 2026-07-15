"""Per-agent slices and shared execution leases.

Extracted from ``run_agent.py``.  Each ``AIAgent`` instance (parent or
subagent) holds an :class:`IterationBudget`; the parent's cap comes from
``max_iterations`` (default 90), each subagent's cap comes from
``delegation.max_iterations`` (default 50).

``run_agent`` re-exports ``IterationBudget`` so existing
``from run_agent import IterationBudget`` imports keep working unchanged.

An :class:`ExecutionLease` is deliberately separate from the renewable
per-agent slice.  It is a monotonic provider-call ceiling shared by a parent
and every descendant.  Renewing a slice or spawning a child must never replace
or refund that lease.  An explicitly injected lease also survives subsequent
turn calls unchanged; only a cached root's owned default is refreshed for a
new, unrelated ``run_conversation`` invocation.  It is the deterministic
safety boundary underneath model-authored plan-to-completion behavior.
"""

from __future__ import annotations

import threading


# Ten full slices is intentionally generous: with Hermes' production default
# of 90 iterations this permits 900 aggregate provider calls across the parent
# and all descendants before a resumable safety boundary.  That is large enough
# for genuinely complex multi-slice work without turning every slice into an
# approval prompt, while remaining finite if a stale Todo or provider starts a
# valid-but-nonterminating tool loop.  Advanced embedders can inject an explicit
# ExecutionLease; this is not a user-facing environment setting.
DEFAULT_EXECUTION_LEASE_SLICES = 10


def default_execution_lease_calls(max_iterations: int) -> int:
    """Return the deterministic aggregate provider-call ceiling for an agent."""

    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    return max_iterations * DEFAULT_EXECUTION_LEASE_SLICES


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
    :meth:`refund` so they don't eat into the budget.
    """

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Try to consume one iteration.  Returns True if allowed."""
        with self._lock:
            if self._used >= self.max_total:
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
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)


class ExecutionLease:
    """Thread-safe, monotonic provider-call authority shared by an agent tree.

    Unlike :class:`IterationBudget`, a lease has no refund operation.  Every
    provider request consumes one unit, including grace/closing requests and
    execute-code turns.  Sharing the same object gives concurrent descendants
    one atomic aggregate ceiling rather than independent resettable limits.
    """

    def __init__(self, max_total: int):
        if isinstance(max_total, bool) or not isinstance(max_total, int):
            raise TypeError("execution lease max_total must be an integer")
        if max_total <= 0:
            raise ValueError("execution lease max_total must be positive")
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Atomically reserve one provider call, returning whether it is allowed."""

        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)


__all__ = [
    "DEFAULT_EXECUTION_LEASE_SLICES",
    "ExecutionLease",
    "IterationBudget",
    "default_execution_lease_calls",
]
