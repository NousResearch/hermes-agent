"""Opt-in response admission for webhook routes; transcripts are never discarded."""
from __future__ import annotations

import re
from typing import Any

# Escalation header, optionally wrapped in markdown emphasis; NBSP counts as spacing.
_URGENT_HEADER = re.compile(r"^\*{0,2}\[(?:NEEDS-APPROVAL|UNRESOLVED|SECURITY-INCIDENT)\]\*{0,2}[ \t\u00a0]+\S.*$", re.IGNORECASE)
# `Action required:` line, case-insensitive, tolerant of a bullet or bold emphasis on the label.
_ACTION_LINE = re.compile(r"^(?:[-*]\s+)?\*{0,2}action required\*{0,2}:\s*(.*)$", re.IGNORECASE)
_NO_ACTION = frozenset({
    "none", "no action", "no action needed", "no action required", "nothing",
    "nothing needed", "nothing required", "none needed", "none required",
    "n/a", "na", "not applicable", "nil",
})
# Fenced code blocks are never scanned for the action line: a quoted runbook or log excerpt
# saying "Action required: ..." must not decide urgency for the response that quotes it.
_FENCE = re.compile(r"^\s*(```|~~~)")


def validate_response_policy(policy: Any) -> None:
    if policy not in (None, "all", "urgent_only"):
        raise ValueError(f"Unknown webhook response_policy: {policy!r}")


def _strip_delivery_markers(content: str) -> str:
    """Drop a leading gateway recovery banner (``♻️ Recovered reply — ...``).

    Crash/reconnect/rate-limit redelivery prepends these markers before line 0; the
    escalation contract must be evaluated on the reply itself, or a recovered urgent
    alert would be swallowed and its ledger row marked delivered.
    """
    from gateway.delivery_ledger import FLOOD_MARKER, RECONNECTED_MARKER, RECOVERED_MARKER
    for marker in (RECOVERED_MARKER, RECONNECTED_MARKER, FLOOD_MARKER):
        if content.startswith(marker):
            return content[len(marker):].lstrip("\n")
    return content


def urgent_response_allowed(content: str, metadata: dict | None = None) -> bool:
    """Only explicit action-required escalations may interrupt the route owner."""
    if (metadata or {}).get("_interim_send") or "[SILENT]" in content.upper():
        return False
    content = _strip_delivery_markers(content)
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines or not _URGENT_HEADER.fullmatch(lines[0]):
        return False
    fenced = False
    for line in lines[1:]:
        if _FENCE.match(line):
            fenced = not fenced  # tolerate an unclosed fence at the end of a reply
            continue
        if fenced:
            continue
        if match := _ACTION_LINE.match(line):
            action = match.group(1).strip().strip("`\"'()[]*_ \u00a0")
            if action and any(ch.isalnum() for ch in action) \
                    and action.rstrip(".!; \u3002").casefold() not in _NO_ACTION:
                return True
    return False
