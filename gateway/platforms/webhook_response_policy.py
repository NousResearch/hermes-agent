"""Opt-in response admission for webhook routes; transcripts are never discarded."""
from __future__ import annotations

import re
from typing import Any

_URGENT_HEADER = re.compile(r"^\[(?:NEEDS-KIM|UNRESOLVED|SECURITY-INCIDENT)\][ \t]+\S.*$")
_NO_ACTION = frozenset({"none", "no action", "no action needed", "nothing", "n/a"})


def validate_response_policy(policy: Any) -> None:
    if policy not in (None, "all", "urgent_only"):
        raise ValueError(f"Unknown webhook response_policy: {policy!r}")


def urgent_response_allowed(content: str, metadata: dict | None = None) -> bool:
    """Only explicit action-required escalations may interrupt the route owner."""
    if (metadata or {}).get("_interim_send") or "[SILENT]" in content.upper():
        return False
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines or not _URGENT_HEADER.fullmatch(lines[0]):
        return False
    for line in lines[1:]:
        if line.startswith("Action required:"):
            action = line.partition(":")[2].strip()
            if action and action.rstrip(".!; ").casefold() not in _NO_ACTION:
                return True
    return False
