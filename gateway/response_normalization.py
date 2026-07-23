"""Outbound-only extraction of user-visible text from recognized response shapes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Tuple


_VISIBLE_TEXT_TYPES = frozenset({"text", "input_text", "output_text", "summary_text"})
_MESSAGE_ROLES = frozenset({"assistant"})
_MAX_DEPTH = 10
_MISSING = object()


def _safe_string(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return f"[Unsupported response content: {type(value).__name__}]"


def _try_extract(response: Any, *, depth: int = 0) -> Tuple[bool, str]:
    """Return ``(recognized, text)`` without guessing from generic fields."""
    if depth > _MAX_DEPTH:
        return False, ""
    if response is None:
        return True, ""
    if isinstance(response, str):
        return True, response

    if isinstance(response, (list, tuple)):
        if not response:
            return False, ""
        parts = []
        for item in response:
            # A bare list of strings/None is not a positively recognized
            # provider shape. Keep the whole unknown structure visible.
            if item is None or isinstance(item, str):
                return False, ""
            recognized, text = _try_extract(item, depth=depth + 1)
            if not recognized:
                return False, ""
            if text:
                parts.append(text)
        return True, "\n".join(parts)

    if isinstance(response, Mapping):
        item_type = _safe_string(response.get("type") or "").strip().lower()
        if item_type in _VISIBLE_TEXT_TYPES:
            text = response.get("text", _MISSING)
            if text is _MISSING or text is None:
                return False, ""
            return True, _safe_string(text)
        if item_type == "message" and "content" in response:
            return _try_extract(response["content"], depth=depth + 1)
        role = _safe_string(response.get("role") or "").strip().lower()
        if role in _MESSAGE_ROLES and "content" in response:
            return _try_extract(response["content"], depth=depth + 1)
        return False, ""

    try:
        item_type_value: Optional[Any] = getattr(response, "type", None)
        item_type = _safe_string(item_type_value or "").strip().lower()
        if item_type in _VISIBLE_TEXT_TYPES:
            text = getattr(response, "text", _MISSING)
            if text is _MISSING or text is None:
                return False, ""
            return True, _safe_string(text)
        if item_type == "message" and hasattr(response, "content"):
            return _try_extract(getattr(response, "content"), depth=depth + 1)
        role = _safe_string(getattr(response, "role", "") or "").strip().lower()
        if role in _MESSAGE_ROLES and hasattr(response, "content"):
            return _try_extract(getattr(response, "content"), depth=depth + 1)
    except Exception:
        return False, ""

    return False, ""


def extract_visible_response_text(response: Any) -> str:
    """Return outbound visible text without interpreting string contents.

    Strings pass through byte-for-byte. Structured values are unwrapped only
    when they match positively recognized provider/Hermes text blocks or
    assistant-message wrappers. Unknown structures remain visible via a safe
    string fallback instead of being silently converted to an empty response.
    """
    recognized, text = _try_extract(response)
    if recognized:
        return text
    return _safe_string(response)
