"""Pure terminal classification for one agent turn."""

from __future__ import annotations

from typing import Literal, TypedDict


TurnOutcomeName = Literal[
    "verified",
    "completed_unverified",
    "partial",
    "blocked",
    "failed",
    "interrupted",
    "unresolved",
    "cancelled",
]
TURN_OUTCOMES = (
    "verified",
    "completed_unverified",
    "partial",
    "blocked",
    "failed",
    "interrupted",
    "unresolved",
    "cancelled",
)


class TurnOutcome(TypedDict):
    outcome: TurnOutcomeName
    reason: str


def _status_value(verification_status: object) -> str:
    if isinstance(verification_status, dict):
        verification_status = verification_status.get("status")
    return str(verification_status or "").strip().lower()


def classify_turn_outcome(
    final_response: object = None,
    failed: bool = False,
    interrupted: bool = False,
    _turn_exit_reason: object = "unknown",
    timeout: bool = False,
    unresolved: bool = False,
    verification_status: object = None,
    blocked: bool = False,
    cancelled: bool = False,
) -> TurnOutcome:
    """Classify terminal turn facts without consulting runtime state."""
    exit_reason = str(_turn_exit_reason or "").strip().lower()

    if failed:
        return {"outcome": "failed", "reason": "turn failed"}
    if interrupted:
        return {"outcome": "interrupted", "reason": "turn interrupted"}
    if unresolved or timeout or any(
        marker in exit_reason for marker in ("unresolved", "timeout", "timed_out")
    ):
        return {"outcome": "unresolved", "reason": "side effect outcome unresolved"}
    if blocked or "blocked" in exit_reason or (
        "approval" in exit_reason
        and any(marker in exit_reason for marker in ("deny", "block", "required"))
    ):
        return {"outcome": "blocked", "reason": "approval blocked"}
    if cancelled or exit_reason in {"cancelled", "canceled"}:
        return {"outcome": "cancelled", "reason": "turn cancelled"}
    if (
        exit_reason == "budget_exhausted"
        or exit_reason.startswith("max_iterations_reached")
        or "iteration" in exit_reason and "limit" in exit_reason
        or exit_reason in {
            "partial_stream_recovery",
            "fallback_prior_turn_content",
            "empty_response_exhausted",
        }
    ):
        return {"outcome": "partial", "reason": "iteration budget or turn completion was partial"}

    status = _status_value(verification_status)
    if status in {"failed", "error"}:
        return {"outcome": "failed", "reason": "verification failed"}
    if final_response:
        if status in {"passed", "verified", "success", "ok"}:
            return {"outcome": "verified", "reason": "verification passed"}
        return {
            "outcome": "completed_unverified",
            "reason": "response completed without verification",
        }
    return {"outcome": "partial", "reason": "no final response"}


__all__ = ["TURN_OUTCOMES", "TurnOutcome", "TurnOutcomeName", "classify_turn_outcome"]
