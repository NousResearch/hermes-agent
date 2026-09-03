"""Shared text projections for Wisdom security and professionalism checks."""

from __future__ import annotations

from typing import Any

from .professionalism import CHECK_LABELS


def aggregate_review_text(
    security: dict[str, Any] | None,
    professionalism: dict[str, Any] | None,
) -> str:
    security_status = str((security or {}).get("status") or "unavailable")
    professionalism_status = str(
        (professionalism or {}).get("status") or "unavailable"
    )
    return (
        f"Security: {security_status.replace('_', ' ').title()} · "
        f"Professionalism: {professionalism_status.replace('_', ' ').title()}"
    )


def full_review_text(
    security: dict[str, Any] | None,
    professionalism: dict[str, Any] | None,
) -> str:
    """Render both checklists with labels, statuses, counts, and bounded detail."""

    sections = [
        _checklist_text(
            "Security check",
            security,
            labels={},
            note="No known matches detected is not a security certification.",
        ),
        _checklist_text(
            "Professionalism check (agent-assessed, advisory)",
            professionalism,
            labels=CHECK_LABELS,
        ),
    ]
    return "\n\n".join(sections)


def _checklist_text(
    title: str,
    check: dict[str, Any] | None,
    *,
    labels: dict[str, str],
    note: str | None = None,
) -> str:
    value = check or {}
    status = str(value.get("status") or "unavailable").replace("_", " ").title()
    lines = [f"{title}: {status}"]
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
            row_status = str(row.get("status") or "unavailable").replace("_", " ").title()
            count = int(row.get("finding_count") or 0)
            suffix = f" ({count} finding{'s' if count != 1 else ''})" if count else ""
            lines.append(f"- {label}: {row_status}{suffix}")
            for detail in row.get("details") or []:
                lines.append(f"  {str(detail)[:256]}")
    if note:
        lines.append(note)
    return "\n".join(lines)
