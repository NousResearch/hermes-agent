"""Shared text projections for Wisdom security and professionalism checks."""

from __future__ import annotations

from typing import Any

from .professionalism import CHECK_LABELS

_STATUS_PRESENTATION = {
    "pass": ("✅", "Pass"),
    "advisory": ("⚠️", "Advisory"),
    "blocked": ("❌", "Blocked"),
    "pending": ("⏳", "Pending"),
    "retry": ("⏳", "Pending"),
    "running": ("⏳", "Pending"),
    "unavailable": ("➖", "Unavailable"),
}


def review_status_text(status: object) -> str:
    """Return the shared icon-and-text label for a review status."""

    icon, label = _STATUS_PRESENTATION.get(
        str(status or "unavailable").lower(),
        _STATUS_PRESENTATION["unavailable"],
    )
    return f"{icon} {label}"


def review_check_line(label: str, status: object) -> str:
    """Lead with the status icon; retain explicit labels for non-passing checks."""
    icon, state = _STATUS_PRESENTATION.get(
        str(status or "unavailable").lower(), _STATUS_PRESENTATION["unavailable"]
    )
    return f"{icon} {label}" + (f": {state}" if state != "Pass" else "")


def aggregate_review_text(
    security: dict[str, Any] | None,
    professionalism: dict[str, Any] | None,
) -> str:
    return (
        f"Security: {review_status_text((security or {}).get('status'))} · "
        "Professionalism: "
        f"{review_status_text((professionalism or {}).get('status'))}"
    )


def full_review_text(
    security: dict[str, Any] | None,
    professionalism: dict[str, Any] | None,
    *,
    status_first: bool = False,
) -> str:
    """Render both checklists with labels, statuses, counts, and bounded detail."""

    sections = [
        _checklist_text(
            "Security check",
            security,
            labels={},
            note="No known matches detected is not a security certification.",
            status_first=status_first,
        ),
        _checklist_text(
            "Professionalism check (agent-assessed, advisory)",
            professionalism,
            labels=CHECK_LABELS,
            status_first=status_first,
        ),
    ]
    return "\n\n".join(sections)


def professionalism_review_text(check: dict[str, Any] | None) -> str:
    return _checklist_text(
        "Professionalism check (agent-assessed, advisory)", check, labels=CHECK_LABELS
    )


def _checklist_text(
    title: str,
    check: dict[str, Any] | None,
    *,
    labels: dict[str, str],
    note: str | None = None,
    status_first: bool = False,
) -> str:
    value = check or {}
    lines = [review_check_line(title, value.get("status")) if status_first
             else f"{title}: {review_status_text(value.get('status'))}"]
    summary = value.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.append(summary[:512])
    rows = value.get("checks")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = str(row.get("key") or "")
            label = str(row.get("label") or labels.get(key) or key.replace("_", " ").title())
            count = int(row.get("finding_count") or 0)
            suffix = f" ({count} finding{'s' if count != 1 else ''})" if count else ""
            lines.append(
                (review_check_line(label, row.get("status")) if status_first
                 else f"{label}: {review_status_text(row.get('status'))}") + suffix
            )
            for detail in row.get("details") or []:
                lines.append(f"  {str(detail)[:256]}")
    if note:
        lines.append(note)
    return "\n".join(lines)
