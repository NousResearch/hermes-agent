from __future__ import annotations

from html import escape
from collections.abc import Sequence

from .models import Classification


def render_report(items: Sequence[Classification]) -> str:
    urgent = sum(item.urgent for item in items)
    lines = ["Mailbox read-only report", f"Observed: {len(items)} | Urgent: {urgent}"]
    lines.extend(
        f"- [{item.category}] {escape(item.observation.sender)} — {escape(item.observation.subject)}"
        for item in items[:5]
    )
    return "\n".join(lines)
