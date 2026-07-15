"""Mechanical validation for model-authored delivery outcomes.

The gateway never interprets response prose.  The primary model may author a
turn-bound ``delivery_outcome`` through the ``todo`` tool; this boundary only
validates the exact receipt and executes deliver/suppress.  Unknown, malformed,
stale, or failed results always remain deliverable.
"""

from __future__ import annotations

from typing import Any, Mapping

from agent.delivery_outcome import (
    DELIVERY_ACTIONS,
    MAX_DELIVERY_REASON_CHARS,
)


_OUTCOME_KEYS = frozenset({"action", "reason", "turn_id"})


def validated_delivery_outcome(
    agent_result: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    """Return an exact, same-turn outcome or ``None`` on any uncertainty."""

    if not isinstance(agent_result, Mapping):
        return None
    turn_id = agent_result.get("turn_id")
    outcome = agent_result.get("delivery_outcome")
    if type(turn_id) is not str or not turn_id:
        return None
    if not isinstance(outcome, Mapping):
        return None
    if frozenset(outcome.keys()) != _OUTCOME_KEYS:
        return None

    action = outcome.get("action")
    reason = outcome.get("reason")
    outcome_turn_id = outcome.get("turn_id")
    if type(action) is not str or action not in DELIVERY_ACTIONS:
        return None
    if type(reason) is not str or not reason.strip():
        return None
    if len(reason) > MAX_DELIVERY_REASON_CHARS:
        return None
    if type(outcome_turn_id) is not str or outcome_turn_id != turn_id:
        return None
    return {
        "action": action,
        "reason": reason,
        "turn_id": outcome_turn_id,
    }


def should_suppress_delivery(agent_result: Mapping[str, Any] | None) -> bool:
    """Execute only an exact suppress choice from a known-successful turn."""

    if not isinstance(agent_result, Mapping):
        return False
    # Fail open for delivery: failure, missing status, or a malformed status
    # must never hide the diagnostic response from the user.
    if agent_result.get("failed") is not False:
        return False
    outcome = validated_delivery_outcome(agent_result)
    return bool(outcome and outcome["action"] == "suppress")


__all__ = ["should_suppress_delivery", "validated_delivery_outcome"]
