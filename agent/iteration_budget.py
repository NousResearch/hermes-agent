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
from typing import Any, Optional


CURRENT_TURN_ID_KEY = "_hermes_turn_id"


def _current_turn_user_index(
    messages: list[dict[str, Any]], turn_id: str
) -> Optional[int]:
    """Return the current user-message index across context compaction.

    Context compression rebuilds retained messages as fresh dictionaries, so
    Python object identity cannot delimit the active turn.  The private turn
    marker survives those copies and is stripped by the existing wire
    sanitizers together with every other underscore-prefixed message key.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if (
            message.get("role") == "user"
            and message.get(CURRENT_TURN_ID_KEY) == turn_id
        ):
            return index
    return None


def _latest_unseen_tool_result(
    messages: list[dict[str, Any]],
    *,
    seen_tool_call_ids: set[str],
    seen_legacy_message_ids: set[int],
    min_index: int = 0,
    require_tool_call_id: bool = False,
) -> Optional[tuple[int, Optional[str]]]:
    """Return the latest tool result absent from every prior API request.

    Provider tool-call ids are the stable identity. Object identity is only a
    compatibility fallback for legacy tool messages without an id; a repaired
    or reconstructed legacy message is deliberately treated as new rather than
    risking mutation of a known cached prefix.
    """
    first_index = max(0, int(min_index))
    for index in range(len(messages) - 1, first_index - 1, -1):
        message = messages[index]
        if message.get("role") != "tool":
            continue
        raw_call_id = message.get("tool_call_id")
        if raw_call_id:
            call_id = str(raw_call_id)
            if call_id not in seen_tool_call_ids:
                return index, call_id
        elif not require_tool_call_id and id(message) not in seen_legacy_message_ids:
            return index, None
    return None


def kanban_checkpoint_warning(
    *,
    used: int,
    max_total: int,
    ratio: float,
    is_kanban: bool,
    already_emitted: bool,
) -> Optional[str]:
    """Return one API-only pre-exhaustion warning when its threshold is met."""
    if not is_kanban or already_emitted or max_total <= 0:
        return None
    try:
        normalized_ratio = float(ratio)
    except (TypeError, ValueError):
        normalized_ratio = 0.75
    if not 0.5 <= normalized_ratio < 1.0:
        normalized_ratio = 0.75
    if used < max_total * normalized_ratio or used >= max_total:
        return None
    return (
        f"[KANBAN_BUDGET_CHECKPOINT {used}/{max_total}] "
        "The execution budget is entering its closure window. Stop expanding "
        "scope. Finish and verify the current atomic claim if feasible; "
        "otherwise preserve a concise partial handoff covering completed work, "
        "changed files, exact tests, remaining work, resume action, and safety "
        "invariants. Do not claim PASS, approval, or deployment without actual "
        "verification. Do not create duplicate review or handoff cards."
    )


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


__all__ = [
    "CURRENT_TURN_ID_KEY",
    "IterationBudget",
    "kanban_checkpoint_warning",
]
