"""Fail-closed guard for classified operational content leaving Hermes."""

from __future__ import annotations

import re
from typing import Any

_RESERVED_RE = re.compile(
    r"(?:^|\n)\s*(?:access|sensitivity|classification)\s*:\s*"
    r"(?:operator(?:s)?|private|restricted|confidential|secret)\b",
    re.IGNORECASE,
)


def _contains_reserved(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_RESERVED_RE.search(value))
    if isinstance(value, dict):
        for key, item in value.items():
            if (isinstance(key, str) and key.casefold() in {"access", "sensitivity", "classification"}
                    and isinstance(item, str) and item.casefold() in {
                        "operator", "operators", "private", "restricted", "confidential", "secret"}):
                return True
            if _contains_reserved(key) or _contains_reserved(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_reserved(item) for item in value)
    return False


def assert_no_unauthorized_reserved_egress(agent: Any, payload: Any) -> None:
    """Reject classified vault content unless explicit authorization is present."""
    if not _contains_reserved(payload):
        return
    if not bool(getattr(agent, "reserved_content_authorized", False)):
        raise PermissionError(
            "Blocked outbound LLM request: reserved operational-memory content has no explicit destination authorization"
        )
