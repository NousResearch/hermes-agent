"""Deterministic Self/Lineage status-line rendering.

The communication contract for project-structured profiles is deliberately
small: every Matrix/profile-facing status block should contain exactly these
canonical lines when a status is needed::

    Self status: WORKING|WAITING|BLOCKED|DORMANT — <specific detail>
    Lineage status: WORKING|WAITING|BLOCKED|DORMANT — <aggregate descendant detail>

This module is pure and dependency-free so scripts, prompts, diagnostics, and
future gateway integrations can share the same status vocabulary without asking
an LLM to recompute semantics from broad context.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class StatusDefinition:
    """Definition for one canonical live status."""

    status: str
    default_detail: str


STATUS_DEFINITIONS: Mapping[str, StatusDefinition] = {
    "working": StatusDefinition(
        "working",
        "active work is underway or immediately queued/spawnable",
    ),
    "waiting": StatusDefinition(
        "waiting",
        "active work is waiting on non-human/system continuation",
    ),
    "blocked": StatusDefinition(
        "blocked",
        "active work is prevented by required human action",
    ),
    "dormant": StatusDefinition(
        "dormant",
        "no active actionable assignment exists",
    ),
}

LIVE_STATUSES: tuple[str, ...] = ("working", "waiting", "blocked", "dormant")

# Storage/workflow aliases accepted at the renderer boundary during migration.
# ``done`` and ``archived`` are not live availability states; for lineage
# aggregation they contribute no active work, so they collapse to dormant.
LIVE_STATUS_ALIASES: Mapping[str, str] = {
    "working": "working",
    "ready": "working",
    "queue": "working",
    "running": "working",
    "in_progress": "working",
    "review": "working",
    "waiting": "waiting",
    "todo": "waiting",
    "scheduled": "waiting",
    "blocked": "blocked",
    "dormant": "dormant",
    "triage": "dormant",
    "done": "dormant",
    "archived": "dormant",
}

# Structural lineage uses the approved aggregate rule: active descendant work
# wins, then human blockers, then non-human waits, then dormant/no work.
LINEAGE_PRECEDENCE: tuple[str, ...] = ("working", "blocked", "waiting", "dormant")


class StatusLineError(ValueError):
    """Raised when status-line input is invalid."""


def normalize_status(status: str) -> str:
    """Validate and normalize a canonical live status word."""

    candidate = str(status).strip().lower()
    if candidate not in STATUS_DEFINITIONS:
        valid = ", ".join(LIVE_STATUSES)
        raise StatusLineError(f"unknown status {status!r}; expected one of: {valid}")
    return candidate


def canonical_live_status(status: str, *, fail_unknown_to_blocked: bool = False) -> tuple[str, str | None]:
    """Map storage/compatibility aliases to canonical live statuses.

    Returns ``(live_status, unknown_marker)``. Unknown lineage inputs can fail
    closed to ``blocked`` so deterministic aggregate output does not hide a
    potentially human-actionable status vocabulary drift.
    """

    candidate = str(status).strip().lower()
    if candidate in LIVE_STATUS_ALIASES:
        return LIVE_STATUS_ALIASES[candidate], None
    if fail_unknown_to_blocked:
        return "blocked", candidate or "<empty>"
    return normalize_status(candidate), None


def render_line(kind: str, status: str, detail: str = "") -> str:
    """Render one canonical status line.

    ``kind`` must be ``self`` or ``lineage``. Output status words are uppercase
    to match profile-prompt contracts; input is case-insensitive.
    """

    labels = {"self": "Self status", "lineage": "Lineage status"}
    if kind not in labels:
        raise StatusLineError("status line kind must be 'self' or 'lineage'")
    normalized = normalize_status(status)
    clean_detail = detail.strip() or STATUS_DEFINITIONS[normalized].default_detail
    return f"{labels[kind]}: {normalized.upper()} — {clean_detail}"


def lineage_counts(statuses: Iterable[str]) -> tuple[Counter[str], list[str]]:
    """Count structural descendant statuses by canonical live state."""

    counts: Counter[str] = Counter({status: 0 for status in LIVE_STATUSES})
    unknown: list[str] = []
    for status in statuses:
        live_status, unknown_marker = canonical_live_status(status, fail_unknown_to_blocked=True)
        counts[live_status] += 1
        if unknown_marker is not None:
            unknown.append(unknown_marker)
    return counts, unknown


def lineage_headline(counts: Mapping[str, int]) -> str:
    """Choose the aggregate Lineage status from descendant counts."""

    for status in LINEAGE_PRECEDENCE:
        if counts.get(status, 0) > 0:
            return status
    return "dormant"


def format_lineage_counts(counts: Mapping[str, int], unknown: Sequence[str] = ()) -> str:
    """Render concise deterministic lineage evidence in canonical order."""

    detail = "; ".join(f"{status}: {counts.get(status, 0)}" for status in LIVE_STATUSES)
    if unknown:
        detail += "; unknown→blocked: " + ", ".join(sorted(set(unknown)))
    return detail


def render_lineage_aggregate(statuses: Iterable[str]) -> str:
    """Render the aggregate Lineage status line for structural descendants."""

    counts, unknown = lineage_counts(statuses)
    headline = lineage_headline(counts)
    return render_line("lineage", headline, format_lineage_counts(counts, unknown))


def render_status_pair(
    self_status: str,
    self_detail: str,
    lineage_status: str,
    lineage_detail: str,
) -> str:
    """Render a two-line Self/Lineage status block from scalar inputs."""

    return "\n".join(
        [
            render_line("self", self_status, self_detail),
            render_line("lineage", lineage_status, lineage_detail),
        ]
    )


def render_status_with_lineage_aggregate(
    self_status: str,
    self_detail: str,
    lineage_statuses: Iterable[str],
) -> str:
    """Render Self status plus aggregate descendant Lineage status."""

    return "\n".join(
        [
            render_line("self", self_status, self_detail),
            render_lineage_aggregate(lineage_statuses),
        ]
    )
