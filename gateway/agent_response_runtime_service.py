"""Shared runtime helpers for gateway agent-response normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class GatewayNormalizedResponse:
    """Normalized post-agent response state used by the gateway."""

    response: str
    suppress_reply: bool
    response_state: str


def build_failed_agent_response(
    *,
    error_detail: Any,
    history_len: int,
) -> str:
    """Build the user-visible fallback when the agent failed silently."""

    error_str = str(error_detail).lower()
    is_context_failure = any(
        pattern in error_str
        for pattern in (
            "context",
            "token",
            "too large",
            "too long",
            "exceed",
            "payload",
        )
    ) or ("400" in error_str and history_len > 50)

    if is_context_failure:
        return (
            "⚠️ Session too large for the model's context window.\n"
            "Use /compact to compress the conversation, or "
            "/reset to start fresh."
        )
    return (
        f"The request failed: {str(error_detail)[:300]}\n"
        "Try again or use /reset to start a fresh session."
    )


def normalize_gateway_agent_response(
    *,
    agent_result: dict[str, Any],
    history_len: int,
    empty_response_fallback: Callable[[str], str | None],
) -> GatewayNormalizedResponse:
    """Normalize empty/silent agent results into gateway response semantics."""

    response = str(agent_result.get("final_response") or "")
    suppress_reply = bool(agent_result.get("suppress_reply"))
    response_state = "sent"

    if response.strip() in {"(empty)", "[[NO_REPLY]]"}:
        empty_kind = "no_reply" if response.strip() == "[[NO_REPLY]]" else "empty"
        fallback = empty_response_fallback(empty_kind)
        if fallback:
            response = fallback
            suppress_reply = False
            response_state = "qq_explicit_fallback"
        else:
            suppress_reply = True
            response = ""
            response_state = "suppressed_empty"

    if response.strip() == "[[NO_REPLY]]":
        suppress_reply = True
        response = ""
        response_state = "suppressed_no_reply"
    elif not response and suppress_reply and response_state == "sent":
        response_state = "suppressed"
    elif not response and agent_result.get("failed"):
        response_state = "failed_silent"

    if not response and agent_result.get("failed"):
        response = build_failed_agent_response(
            error_detail=agent_result.get("error", "unknown error"),
            history_len=history_len,
        )

    return GatewayNormalizedResponse(
        response=response,
        suppress_reply=suppress_reply,
        response_state=response_state,
    )
