"""Shared validation for OpenAI-compatible chat-completion responses."""

from __future__ import annotations

from typing import Any


ROUTER_TIMEOUT_SHIM = "Connect timeout, please try again later."


def _has_positive_completion_tokens(usage: Any) -> bool:
    """Return whether response usage proves the provider generated output."""
    for field in ("completion_tokens", "output_tokens"):
        value = (
            usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
        )
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        ):
            return True
    return False


def classify_chat_completion_response(response: Any) -> str | None:
    """Return an invalid-response reason, or ``None`` for an acceptable response.

    The router timeout shim is an HTTP-success payload emitted by some
    OpenAI-compatible gateways. It is only invalid when it is the sole text
    payload and usage does not prove the model generated output.
    """
    if response is None:
        return "missing_response"
    choices = getattr(response, "choices", None)
    if not choices:
        return "missing_choices"
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        return None
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if (
        isinstance(content, str)
        and content.strip() == ROUTER_TIMEOUT_SHIM
        and not getattr(message, "tool_calls", None)
        and not _has_positive_completion_tokens(getattr(response, "usage", None))
    ):
        return "router_timeout_shim"
    return None


def is_valid_chat_completion_response(response: Any) -> bool:
    """Return whether an OpenAI-compatible response is safe to consume."""
    return classify_chat_completion_response(response) is None
