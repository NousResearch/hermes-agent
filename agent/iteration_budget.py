"""Per-agent iteration budget — thread-safe consume/refund counter.

Each ``AIAgent`` (parent or subagent) holds its own :class:`IterationBudget`: the parent's
 cap is ``max_iterations`` (default 500), each subagent's ``delegation.max_iterations``
 (default 50), so total iterations across parent + subagents can exceed the parent's cap.
 A turn uses an explicit invocation limit before a skill override before its configured baseline.
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


def normalize_skill_max_turns(value) -> dict[str, int]:
    """Keep valid skill names and positive integer turn limits."""
    if not isinstance(value, dict):
        return {}
    return {
        name: limit for name, limit in value.items()
        if isinstance(name, str) and name.strip()
        and isinstance(limit, int) and not isinstance(limit, bool) and limit > 0
    }


def skill_max_turns_from_config(cfg) -> dict[str, int]:
    """Read the shared skill limit setting from a surface's config."""
    if not isinstance(cfg, dict):
        return {}
    agent_cfg = cfg.get("agent")
    return normalize_skill_max_turns(agent_cfg.get("skill_max_turns")) if isinstance(agent_cfg, dict) else {}


def arm_turn_iteration_limit(agent, user_message) -> int:
    """Arm this turn's limit without changing the durable baseline."""
    base = getattr(agent, "_baseline_max_iterations", agent.max_iterations)
    if getattr(agent, "_max_iterations_explicit", False):
        resolved = base
    else:
        from agent.skill_commands import skill_invocation_name

        name = skill_invocation_name(user_message)
        limit = getattr(agent, "skill_max_turns", {}).get(name) if name else None
        resolved = (
            limit if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0 else base
        )
    agent._effective_max_iterations = resolved
    return resolved


class IterationBudget:
    """Thread-safe iteration counter; ``execute_code`` (programmatic tool calling)
    iterations are refunded via :meth:`refund` so they don't eat into the budget."""

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


__all__ = ["IterationBudget"]
