"""Turn-bound, primary-model-authored delivery outcomes.

The model may use the existing ``todo`` tool to choose whether the completed
turn should be delivered to a messaging surface.  This module deliberately
does not inspect response text.  It only validates the small structured
protocol, binds it to the turn that authored it, and returns a receipt.

The state is agent-local and reset at every turn boundary.  A late tool worker
from an earlier turn therefore cannot suppress a newer turn.
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Mapping


DELIVERY_ACTIONS = frozenset({"deliver", "suppress"})
MAX_DELIVERY_REASON_CHARS = 2000
_DIRECTIVE_KEYS = frozenset({"action", "reason"})
_OUTCOME_KEYS = frozenset({"action", "reason", "turn_id"})


def _state_lock(agent: Any) -> threading.Lock:
    lock = getattr(agent, "_delivery_outcome_lock", None)
    if lock is None:
        lock = threading.Lock()
        agent._delivery_outcome_lock = lock
    return lock


def _validate_directive(directive: Any) -> tuple[str, str]:
    if not isinstance(directive, Mapping):
        raise ValueError("delivery_outcome must be an object")
    if frozenset(directive.keys()) != _DIRECTIVE_KEYS:
        raise ValueError("delivery_outcome requires exactly action and reason")

    action = directive.get("action")
    reason = directive.get("reason")
    if type(action) is not str or action not in DELIVERY_ACTIONS:
        raise ValueError("delivery_outcome.action must be deliver or suppress")
    if type(reason) is not str:
        raise ValueError("delivery_outcome.reason must be a string")
    if not reason.strip():
        raise ValueError("delivery_outcome.reason must not be empty")
    if len(reason) > MAX_DELIVERY_REASON_CHARS:
        raise ValueError(
            f"delivery_outcome.reason exceeds {MAX_DELIVERY_REASON_CHARS} characters"
        )
    return action, reason


def reset_delivery_outcome_turn(agent: Any, turn_id: str) -> None:
    """Start an empty delivery-outcome slot for exactly ``turn_id``."""

    bound_turn_id = str(turn_id or "")
    if not bound_turn_id:
        raise ValueError("delivery outcome requires a non-empty turn id")
    with _state_lock(agent):
        agent._delivery_outcome_state = {
            "turn_id": bound_turn_id,
            "outcome": None,
        }


def record_delivery_outcome(
    agent: Any,
    directive: Any,
    *,
    originating_turn_id: str,
) -> dict[str, Any]:
    """Validate and record the model's directive for its originating turn."""

    action, reason = _validate_directive(directive)
    origin = str(originating_turn_id or "")
    current = str(getattr(agent, "_current_turn_id", "") or "")
    if not origin or origin != current:
        raise ValueError("delivery_outcome rejected: originating turn is stale")

    with _state_lock(agent):
        state = getattr(agent, "_delivery_outcome_state", None)
        if not isinstance(state, dict) or state.get("turn_id") != origin:
            raise ValueError("delivery_outcome rejected: turn binding is unavailable")
        outcome = {
            "action": action,
            "reason": reason,
            "turn_id": origin,
        }
        state["outcome"] = outcome

    return {
        "recorded": True,
        "action": action,
        "turn_id": origin,
    }


def get_delivery_outcome(agent: Any, turn_id: str) -> dict[str, str] | None:
    """Return a copy of the valid outcome for ``turn_id``, if the model set one."""

    expected = str(turn_id or "")
    if not expected:
        return None
    with _state_lock(agent):
        state = getattr(agent, "_delivery_outcome_state", None)
        if not isinstance(state, dict) or state.get("turn_id") != expected:
            return None
        outcome = state.get("outcome")
        if not isinstance(outcome, dict):
            return None
        if frozenset(outcome.keys()) != _OUTCOME_KEYS:
            return None
        if outcome.get("turn_id") != expected:
            return None
        return copy.deepcopy(outcome)


__all__ = [
    "DELIVERY_ACTIONS",
    "MAX_DELIVERY_REASON_CHARS",
    "get_delivery_outcome",
    "record_delivery_outcome",
    "reset_delivery_outcome_turn",
]
