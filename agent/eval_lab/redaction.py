"""Secret redaction helpers for eval-lab artifacts.

Eval artifacts must be safe for later dataset export. This module preserves
input structure while replacing sensitive values with a stable placeholder.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "[REDACTED]"

_SENSITIVE_KEY_PARTS = (
    "api_key",
    "token",
    "secret",
    "password",
    "authorization",
    "cookie",
)

_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{10,}\b", re.IGNORECASE)


def is_sensitive_key(key: Any) -> bool:
    """Return True when a mapping key should have its value redacted."""
    normalized = str(key).lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def redact_text(text: str) -> str:
    """Redact bearer-like tokens inside free-form text."""
    return _BEARER_RE.sub(REDACTED, text)


def redact_secrets(value: Any) -> Any:
    """Recursively redact sensitive values in dict/list/string structures.

    - Mapping values are redacted when the key contains a sensitive substring.
    - Lists/tuples preserve ordering and recurse into children.
    - Strings redact bearer-like token payloads while preserving surrounding text.
    - Other scalar values are returned unchanged unless behind a sensitive key.
    """
    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, child in value.items():
            if is_sensitive_key(key):
                redacted[key] = REDACTED
            else:
                redacted[key] = redact_secrets(child)
        return redacted

    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_secrets(item) for item in value]

    if isinstance(value, str):
        return redact_text(value)

    return value
