"""Redaction helpers for stored StoreCRM QA evidence summaries."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|access[_-]?token|auth[_-]?token|bearer|client[_-]?secret|"
    r"credential|password|private[_-]?key|secret|session[_-]?token|token)",
    re.IGNORECASE,
)
ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|bearer|client[_-]?secret|"
    r"credential|password|private[_-]?key|secret|session[_-]?token|token)"
    r"\b\s*[:=]\s*([^\s,;]+)"
)
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
LONG_SECRET_RE = re.compile(r"\b(?:sk|pk|rk|ghp|gho|xox[baprs])[-_][A-Za-z0-9._-]{8,}\b")


def _redact_string(value: str) -> str:
    value = ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=<redacted>", value)
    value = BEARER_RE.sub("Bearer <redacted>", value)
    value = JWT_RE.sub("<redacted-token>", value)
    value = LONG_SECRET_RE.sub("<redacted-token>", value)
    return value


def redact_value(value: Any, *, parent_key: str = "") -> Any:
    """Return ``value`` with credential-shaped content removed."""

    if SECRET_KEY_RE.search(parent_key):
        return "<redacted>"
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, Mapping):
        return {
            str(key): redact_value(item, parent_key=str(key))
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [redact_value(item, parent_key=parent_key) for item in value]
    return value


def redact_text(value: str) -> str:
    return _redact_string(value)


def redacted_json(value: Any) -> str:
    return json.dumps(redact_value(value), sort_keys=True, separators=(",", ":"))
