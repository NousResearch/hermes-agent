"""Monotonic helpers for the optional per-conversation run budget."""

from __future__ import annotations

import math
import time
from typing import Any, Optional


def _finite_seconds(value: Any) -> Optional[float]:
    """Return a finite numeric duration, rejecting bools and duck types."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds = float(value)
    return seconds if math.isfinite(seconds) else None


def elapsed_run_budget_seconds(
    agent: Any,
    *,
    now: Optional[float] = None,
) -> Optional[float]:
    """Return monotonic elapsed time for an active, well-typed run budget."""
    budget = _finite_seconds(getattr(agent, "run_budget_seconds", None))
    started_at = _finite_seconds(getattr(agent, "_run_budget_started_at", None))
    if budget is None or budget <= 0 or started_at is None:
        return None
    current = time.monotonic() if now is None else _finite_seconds(now)
    if current is None:
        return None
    return max(0.0, current - started_at)


def remaining_run_budget_seconds(
    agent: Any,
    *,
    now: Optional[float] = None,
) -> Optional[float]:
    """Return the bounded remaining budget, or ``None`` when inactive."""
    budget = _finite_seconds(getattr(agent, "run_budget_seconds", None))
    if budget is None or budget <= 0:
        return None
    elapsed = elapsed_run_budget_seconds(agent, now=now)
    if elapsed is None:
        return None
    return max(0.0, budget - elapsed)
