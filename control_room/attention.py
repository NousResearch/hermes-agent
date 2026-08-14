"""Deterministic attention ranking for Control Room.

Pure module — no I/O. Implements the CR-103 ordering contract:

1. approvals / held messages first  (``critical``)
2. errors / stalled work            (``error``)
3. blocked / review tasks           (``warning``)
4. informational running/ready      (``info``)

Tie-break by severity value, then newest actionable ``updated_at``
(ISO-8601 strings compare lexicographically when normalized UTC), then
stable ``kind:id`` identity for total determinism.
"""

from __future__ import annotations

from typing import Iterable, List

from .contract import AttentionItem, AttentionSeverity


def _timestamp_numeric(updated_at: str) -> float:
    """Parse an ISO-8601 UTC timestamp to a comparable number for sorting.

    Missing/empty timestamps return ``0.0`` so they sort oldest (last within
    their severity when sorting newest-first). Parsing failures degrade to
    ``0.0`` too — never raise inside the ranker.
    """
    if not updated_at:
        return 0.0
    try:
        from datetime import datetime, timezone

        normalized = updated_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return 0.0


def rank_attention(items: Iterable[AttentionItem]) -> List[AttentionItem]:
    """Return items sorted by the CR-103 contract ordering.

    Sort keys, in order:
    1. ``severity`` ascending (lower value = higher priority).
    2. ``updated_at`` descending — newest actionable first. Missing/empty
       timestamps sort last within their severity.
    3. ``stable_id`` ascending — total determinism for equal rows.
    """
    return sorted(
        items,
        key=lambda i: (
            i.severity.value,
            -_timestamp_numeric(i.updated_at),
            i.stable_id,
        ),
    )


def severity_for_kind(kind) -> AttentionSeverity:
    """Map an attention kind to its contract severity.

    Kept as a single decision point so providers and renderers cannot drift.
    """
    from .contract import AttentionKind

    mapping = {
        AttentionKind.approval: AttentionSeverity.critical,
        AttentionKind.held_message: AttentionSeverity.critical,
        AttentionKind.error: AttentionSeverity.error,
        AttentionKind.stalled: AttentionSeverity.error,
        AttentionKind.blocked_task: AttentionSeverity.warning,
        AttentionKind.review_task: AttentionSeverity.warning,
        AttentionKind.running: AttentionSeverity.info,
        AttentionKind.ready: AttentionSeverity.info,
        AttentionKind.system: AttentionSeverity.warning,
        AttentionKind.info: AttentionSeverity.info,
    }
    return mapping.get(kind, AttentionSeverity.info)
