"""Per-agent iteration budget — thread-safe consume/refund counter.

Each ``AIAgent`` (parent or subagent) holds its own :class:`IterationBudget`: the parent's
cap is ``max_iterations`` (default 500), each subagent's ``delegation.max_iterations``
(default 50), so total iterations across parent + subagents can exceed the parent's cap.
"""

from __future__ import annotations

import math
import threading


def normalize_budget_warning_ratio(value) -> float | None:
    """A finite ratio strictly between zero and one, or None (feature off)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    return ratio if math.isfinite(ratio) and 0 < ratio < 1 else None


class IterationBudget:
    """Thread-safe iteration counter; ``execute_code`` (programmatic tool calling)
    iterations are refunded via :meth:`refund` so they don't eat into the budget."""

    _extensions_granted: int = 0

    def __init__(self, max_total: int):
        self.max_total = max_total
        self.initial_max_total = max_total
        self._initial_max_total = max_total
        self._used = 0
        self._extensions_granted = 0
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

    def extend_on_progress(
        self,
        amount: int = 1,
        *,
        min_initial_max: int = 20,
        low_watermark: int = 5,
        max_multiplier: int = 4,
    ) -> bool:
        """Extend ``max_total`` by ``amount`` when a real session (``initial_max_total >= min_initial_max``)
        is making verified progress and within ``low_watermark`` iterations of exhaustion."""
        initial_max = getattr(self, "initial_max_total", None)
        if initial_max is None:
            initial_max = getattr(self, "_initial_max_total", self.max_total)
            self.initial_max_total = initial_max
        if amount <= 0 or initial_max < min_initial_max:
            return False
        with self._lock:
            if self.max_total - self._used > low_watermark:
                return False
            ceiling = initial_max * max_multiplier
            if self.max_total + amount > ceiling:
                return False
            self.max_total += amount
            self._extensions_granted = getattr(self, "_extensions_granted", 0) + amount
            return True

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)


__all__ = ["IterationBudget"]
