"""Shared validation for OpenAI-compatible chat-completion responses."""

from __future__ import annotations

import math
from typing import Any


ROUTER_TIMEOUT_SHIM = "Connect timeout, please try again later."


def _has_positive_completion_tokens(usage: Any) -> bool:
    """Return whether response usage proves the provider generated output."""
    for field in ("completion_tokens", "output_tokens"):
        value = (
            usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
        )
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return True
        if isinstance(value, float) and math.isfinite(value) and value > 0:
            return True
    return False


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _responses_timeout_text(response: Any) -> tuple[bool, str | None]:
    """Return raw Responses evidence as ``(recognized, sole_text)``.

    ``sole_text`` is populated only when the response contains exactly one
    message item with exactly one text part (or only a top-level
    ``output_text`` compatibility field). Any sibling item/content part is raw
    cardinality evidence that this is not the router's one-text shim.
    """
    output = _obj_get(response, "output")
    if isinstance(output, (list, tuple)):
        if len(output) != 1:
            return True, None
        item = output[0]
        # Responses-compatible adapters often omit ``type`` while preserving
        # the same message/content shape consumed by auxiliary recovery.
        if _obj_get(item, "type") not in {"message", None}:
            return True, None
        content = _obj_get(item, "content")
        if not isinstance(content, (list, tuple)) or len(content) != 1:
            return True, None
        part = content[0]
        if _obj_get(part, "type") not in {"output_text", "text", None}:
            return True, None
        text = _obj_get(part, "text")
        return True, text if isinstance(text, str) else None

    output_text = _obj_get(response, "output_text")
    if isinstance(output_text, str):
        return True, output_text
    return False, None


def classify_chat_completion_response(response: Any) -> str | None:
    """Return an invalid-response reason, or ``None`` for an acceptable response.

    The router timeout shim is an HTTP-success payload emitted by some
    OpenAI-compatible gateways. It is only invalid when the *raw* response is
    exactly one text payload equal to the sentinel, has no tool-call evidence,
    and usage does not prove the model generated output. Whitespace, additional
    choices/items/content parts, and function calls are meaningful evidence and
    must be inspected before any consumer normalization.
    """
    if response is None:
        return "missing_response"
    choices = _obj_get(response, "choices")
    if not choices:
        recognized, responses_text = _responses_timeout_text(response)
        if not recognized:
            return "missing_choices"
        if (
            responses_text == ROUTER_TIMEOUT_SHIM
            and not _has_positive_completion_tokens(_obj_get(response, "usage"))
        ):
            return "router_timeout_shim"
        return None
    if not isinstance(choices, (list, tuple)) or len(choices) != 1:
        return None
    message = _obj_get(choices[0], "message")
    content = _obj_get(message, "content")
    if (
        content == ROUTER_TIMEOUT_SHIM
        and not _obj_get(message, "tool_calls")
        and not _has_positive_completion_tokens(_obj_get(response, "usage"))
    ):
        return "router_timeout_shim"
    return None


def is_valid_chat_completion_response(response: Any) -> bool:
    """Return whether an OpenAI-compatible response is safe to consume."""
    return classify_chat_completion_response(response) is None
