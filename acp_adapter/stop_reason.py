"""ACP ``stopReason`` for a finished Hermes turn.

ACP clients decide how to render the end of a prompt from ``stopReason`` alone, so a turn that
stopped because the model hit its output cap or Hermes hit its iteration budget must not read
as a finished answer. The turn loop already stamps the verdicts on its result dict; this maps
them onto the protocol vocabulary (``acp.schema.StopReason``).

Precedence is load-bearing: an explicit cancel wins over everything (the truncation or budget
verdict of an interrupted turn describes the attempt, not the user's stop).
"""

from __future__ import annotations

from typing import Any, Literal

StopReason = Literal["end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"]

# ``failure_reason`` values the turn loop stamps when the model's output was cut short and the
# recovery ladder gave up: ``_Trunc.end_turn`` (default failure), the repetition guards and
# ``collapse_continuation_trail``'s ceiling exit all ride ``"truncated"``; the dropped-stream
# stub rides ``"timeout"`` (a transport verdict, reported as ``end_turn`` — it is not a cap).
_TRUNCATED_FAILURE_REASONS = frozenset({"truncated"})


def acp_stop_reason(result: dict[str, Any] | None, *, cancelled: bool) -> StopReason:
    """``cancelled`` > ``max_tokens`` (output cap, retries exhausted) > ``max_turn_requests``
    (iteration budget) > ``end_turn``."""
    if cancelled:
        return "cancelled"
    if not isinstance(result, dict):
        return "end_turn"
    if str(result.get("failure_reason") or "") in _TRUNCATED_FAILURE_REASONS:
        return "max_tokens"
    exit_reason = str(result.get("turn_exit_reason") or "")
    if exit_reason.startswith("max_iterations_reached(") and result.get("completed") is False:
        return "max_turn_requests"
    return "end_turn"
