"""Secrets hygiene — redact sensitive values before logging."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

_ETH_ADDRESS_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
_PRIVATE_KEY_RE = re.compile(r"\b(?:0x)?[a-fA-F0-9]{64}\b")
_REDACTED_ADDR = "0x…redacted"
_SECRET_KEYS = frozenset(
    {
        "private_key",
        "user_private_key",
        "secret",
        "api_key",
        "password",
        "token",
        "authorization",
    }
)


def redact_for_log(text: str, *, debug: bool = False) -> str:
    """Redact addresses and key-like hex at info level; full detail only when debug."""
    if debug:
        return text
    out = _PRIVATE_KEY_RE.sub("[REDACTED_KEY]", text)
    out = _ETH_ADDRESS_RE.sub(_REDACTED_ADDR, out)
    return out


def redact_mapping(data: Mapping[str, Any] | None, *, debug: bool = False) -> dict[str, Any]:
    if not data:
        return {}
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = str(key).lower()
        if key_lower in _SECRET_KEYS or "private" in key_lower:
            redacted[key] = "[REDACTED]"
        elif isinstance(value, str):
            redacted[key] = redact_for_log(value, debug=debug)
        elif isinstance(value, dict):
            redacted[key] = redact_mapping(value, debug=debug)
        else:
            redacted[key] = value
    return redacted


def hash_params(args: Mapping[str, Any] | None) -> str:
    """Stable SHA-256 of redacted params for audit trail."""
    payload = json.dumps(redact_mapping(args or {}), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]