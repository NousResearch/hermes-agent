"""Privacy helpers for workflow dashboard display."""
from __future__ import annotations

import re
from typing import Any

SENSITIVE_KEY_RE = re.compile(
    r"(secret|token|password|passwd|api[_-]?key|authorization|credential|bearer)",
    re.IGNORECASE,
)
REDACTED = "[REDACTED]"


def redact_sensitive(value: Any) -> Any:
    """Return a copy with values for sensitive-looking keys redacted."""
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if SENSITIVE_KEY_RE.search(str(key)):
                redacted[key] = REDACTED
            else:
                redacted[key] = redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value
