"""Structure-aware, fail-closed secret redaction."""

from __future__ import annotations

import json
from typing import Any

_MASK = "***"
_SENTINEL = "[redaction-unverified]"

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
    if isinstance(value, dict):
        # Dictionary keys can themselves contain a credential (e.g. a token-indexed
        # tool argument). Keeping a raw key makes _settled fail even after every
        # value is masked, discarding otherwise safe siblings.
        result = {}
        for item_key, item in value.items():
            safe_key = _redact_leaf(item_key, None) if isinstance(item_key, str) else item_key
            result[safe_key] = _walk(item, safe_key, hard=hard)
        return result
    if isinstance(value, list):
        return [_walk(item, None, hard=hard) for item in value]
    if isinstance(value, tuple):
        return tuple(_walk(item, None, hard=hard) for item in value)
    if isinstance(value, str):
        return _MASK if hard else _redact_leaf(value, key)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _MASK if hard else _redact_leaf(str(value), key)


def _redact_leaf(value: str, key: Any) -> str:
    # Keyless leaves (list/tuple items, a bare top-level string) are still
    # redacted as quoted JSON, the shape _settled sees: an unquoted
    # "DB_PASSWORD=..." leaf redacts to "DB_PASSWORD=***", whose closing
    # quote the env pattern swallows again on re-serialization, so the soft
    # pass never settles and the hard pass masks every sibling field.
    # ensure_ascii=False matches _settled: a \uXXXX escape ending in a letter
    # would hide a token glued to non-ASCII text from the leaf pass.
    prefix = "" if key is None else json.dumps(key, ensure_ascii=False) + ": "
    redacted = prefix + json.dumps(value, ensure_ascii=False)
    # One pass is not always a fixed point: a head/tail mask (sk-abc...wxyz) can be
    # masked again by another rule, and _settled checks with one more pass.
    for _ in range(4):
        leaf, redacted = redacted, _redact(redacted)
        if redacted == leaf:
            break
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
