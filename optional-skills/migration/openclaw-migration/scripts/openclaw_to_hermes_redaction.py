"""Secret redaction for migration reports.

Sharded out of ``openclaw_to_hermes.py`` (#79938, 2K-law); that module
imports ``redact_migration_value`` from here for its own report writes.
"""

from __future__ import annotations

import re
from typing import Any, Dict


# ───────────────────────────────────────────────────────────────────────
# Secret redaction for migration reports.
#
# The report JSON persists to disk inside the migration output directory and
# frequently ends up in bug reports or support channels.  Anything that looks
# like a credential — by key name or by value shape — is replaced with
# "[redacted]" before the report is written.
#
# Modelled on OpenClaw's src/plugin-sdk/migration.ts so both migration tools
# redact consistently.  Pure function — safe to call on any plain-data dict.
# ───────────────────────────────────────────────────────────────────────
REDACTED_MIGRATION_VALUE = "[redacted]"

_SECRET_KEY_MARKERS = (
    "accesstoken",
    "apikey",
    "authorization",
    "bearertoken",
    "clientsecret",
    "cookie",
    "credential",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
)

_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=\-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{16,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{8,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{12,}\b"),
)


def _normalize_secret_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _is_secret_key(key: str) -> bool:
    normalized = _normalize_secret_key(key)
    if normalized == "token" or normalized.endswith("token"):
        return True
    if normalized in {"auth", "authorization"}:
        return True
    return any(marker in normalized for marker in _SECRET_KEY_MARKERS)


def _redact_string(value: str) -> str:
    for pattern in _SECRET_VALUE_PATTERNS:
        value = pattern.sub(REDACTED_MIGRATION_VALUE, value)
    return value


def redact_migration_value(value: Any) -> Any:
    """Return a deep copy of ``value`` with secret-looking content replaced.

    Applied to every report written to disk.  Keys whose normalized form
    matches a credential marker get their value replaced wholesale.  Strings
    anywhere in the tree are scanned for common token patterns (sk-..., ghp_...,
    xox*-, AIza*, Bearer ...) and those substrings are replaced inline.
    """
    return _redact_internal(value, set())


def _redact_internal(value: Any, seen: set) -> Any:
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, (list, tuple)):
        return [_redact_internal(entry, seen) for entry in value]
    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in seen:
            return REDACTED_MIGRATION_VALUE
        seen.add(obj_id)
        out: Dict[str, Any] = {}
        for key, entry in value.items():
            if isinstance(key, str) and _is_secret_key(key):
                out[key] = REDACTED_MIGRATION_VALUE
            else:
                out[key] = _redact_internal(entry, seen)
        return out
    return value
