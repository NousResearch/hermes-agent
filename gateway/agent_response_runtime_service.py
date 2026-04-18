"""Shared runtime helpers for gateway agent-response normalization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable


@dataclass(slots=True)
class GatewayNormalizedResponse:
    """Normalized post-agent response state used by the gateway."""

    response: str
    suppress_reply: bool
    response_state: str
    synthetic_fallback: bool = False


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


def _build_gateway_status_hint(error: Exception) -> str:
    """Map a gateway exception to a user-facing status hint."""

    status_code = getattr(error, "status_code", None)
    if status_code == 401:
        return " Check your API key or run `claude /login` to refresh OAuth credentials."

    if status_code == 429:
        error_body = getattr(error, "response", None)
        error_json = {}
        try:
            if error_body is not None:
                error_json = error_body.json().get("error", {})
        except Exception:
            error_json = {}

        if error_json.get("type") == "usage_limit_reached":
            resets_in = error_json.get("resets_in_seconds")
            if resets_in and resets_in > 0:
                hours = math.ceil(resets_in / 3600)
                return f" Your plan's usage limit has been reached. It resets in ~{hours}h."
            return " Your plan's usage limit has been reached. Please wait until it resets."
        return " You are being rate-limited. Please wait a moment and try again."

    if status_code == 529:
        return " The API is temporarily overloaded. Please try again shortly."

    if status_code == 400:
        return " The request was rejected by the API."

    return ""


def build_gateway_exception_response(
    *,
    error: Exception,
    history_len: int,
) -> str:
    """Build the user-visible fallback for gateway exceptions."""

    status_code = getattr(error, "status_code", None)
    if status_code in (400, 500) and history_len > 50:
        return (
            "⚠️ Session too large for the model's context window.\n"
            "Use /compact to compress the conversation, or "
            "/reset to start fresh."
        )

    error_text = str(error)
    error_type = type(error).__name__
    error_detail = error_text[:300] if error_text else "no details available"
    status_hint = _build_gateway_status_hint(error)
    return (
        f"Sorry, I encountered an error ({error_type}).\n"
        f"{error_detail}\n"
        f"{status_hint}"
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
    synthetic_fallback = False

    if response.strip() in {"(empty)", "[[NO_REPLY]]"}:
        empty_kind = "no_reply" if response.strip() == "[[NO_REPLY]]" else "empty"
        fallback = empty_response_fallback(empty_kind)
        if fallback:
            response = fallback
            suppress_reply = False
            response_state = "qq_explicit_fallback"
            synthetic_fallback = True
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
        synthetic_fallback=synthetic_fallback,
    )
