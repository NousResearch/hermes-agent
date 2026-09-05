"""Structure-aware, fail-closed secret redaction."""

from __future__ import annotations

import json
import re
from typing import Any

_MASK = "***"
_SENTINEL = "[redaction-unverified]"
_CREDENTIAL_FIELD = re.compile(r"(?:password|passwd|passphrase|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|credential)s?", re.I)

__all__ = ["redact_structured"]


def _redact(text: str) -> str:
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(text, force=True)


def redact_structured(payload: Any) -> Any:
    """Redact string leaves while preserving JSON-shaped structure."""
    walked = _walk(payload, key=None, hard=False)
    if _settled(walked):
        return walked

    masked = _walk(payload, key=None, hard=True)
    if _settled(masked):
        return masked

    if isinstance(payload, dict):
        return {"_redaction": _SENTINEL}
    if isinstance(payload, (list, tuple)):
        return [_SENTINEL]
    return _SENTINEL


def _walk(value: Any, key: Any, *, hard: bool) -> Any:
    # Credential containers (password: [..] or api_key: {value: ..}) carry
    # their sensitivity to every descendant, not only immediate string leaves.
    hard = hard or (isinstance(key, str) and bool(_CREDENTIAL_FIELD.fullmatch(key)))
    if isinstance(value, dict):
        return {
            item_key: _walk(item, item_key, hard=hard)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_walk(item, None, hard=hard) for item in value]
    if isinstance(value, tuple):
        return tuple(_walk(item, None, hard=hard) for item in value)
    if isinstance(value, str):
        return _MASK if hard else _redact_leaf(value, key)
    if value is None or isinstance(value, (bool, int, float)):
        return _MASK if hard else value
    return _MASK if hard else _redact_leaf(str(value), key)


def _redact_leaf(value: str, key: Any) -> str:
    if key is None:
        return _redact(value)

    prefix = json.dumps(key) + ": "
    redacted = _redact(prefix + json.dumps(value))
    if not redacted.startswith(prefix):
        return _MASK

    tail = redacted[len(prefix):]
    try:
        lifted = json.loads(tail)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _MASK
    return lifted if isinstance(lifted, str) else _MASK


def _settled(payload: Any) -> bool:
    try:
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return False
    return _redact(serialized) == serialized
