"""Shared deadline arithmetic for the optional per-turn wall-clock run budget."""

from __future__ import annotations

import time
from typing import Any


def remaining_run_budget_seconds(agent: Any) -> float | None:
    """Return the active turn's remaining wall-clock budget, or ``None`` when disabled."""
    budget = getattr(agent, "run_budget_seconds", None)
    started = getattr(agent, "_run_budget_started_at", None)
    if (
        not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0
        or not isinstance(started, (int, float)) or isinstance(started, bool)
    ):
        return None
    return max(0.0, float(budget) - (time.time() - float(started)))


def run_budget_expired(agent: Any) -> bool:
    """Whether the configured turn deadline has elapsed."""
    remaining = remaining_run_budget_seconds(agent)
    return remaining is not None and remaining <= 0.0
