"""Matrix Tool activity payload helpers (always-visible sticky list)."""

from __future__ import annotations

import html as _html
import json
import re
from collections import deque
from typing import Iterable, Sequence, Tuple


# Replacement events carry HTML twice. Budget escaped JSON, leaving room for event
# metadata and encryption overhead, while keeping the most recent tools visible.
_ITEMS_JSON_LIMIT = 12_000


def matrix_tool_activity_bodies(
    lines: Sequence[str] | Iterable[str],
    footer: str | None = None,
) -> Tuple[str, str]:
    """Build plain body + HTML for Matrix tool progress.

    Contract:
    - tools-only plain body: ``🛠 Tool activity (N updates)``
    - HTML: always-visible title, optional ordered list, optional footer
    - no fences, details, spoilers, or multi-line dumps
    """
    items: deque[tuple[str, int]] = deque()
    items_size = 0
    n = 0
    for line in lines:
        s = str(line or "").strip()
        if not s:
            continue
        if s in {"```", "~~~"} or set(s) <= {"`", "~", " "}:
            continue
        if s.startswith("```") or s.startswith("~~~"):
            continue
        s = s.splitlines()[0].strip()
        s = re.sub(r"\s+", " ", s)
        if len(s) > 160:
            s = s[:157] + "..."
        n += 1
        item = f"<li>{_html.escape(s)}</li>"
        size = len(json.dumps(item, ensure_ascii=True))
        items.append((item, size))
        items_size += size
        while items_size > _ITEMS_JSON_LIMIT:
            _, removed_size = items.popleft()
            items_size -= removed_size
    footer_text = str(footer or "").strip() or None
    title = (
        f"🛠 Tool activity ({n} update{'s' if n != 1 else ''})"
        if n
        else "🛠 Tool activity"
    )
    body = f"{title} · {footer_text}" if footer_text else title
    html_parts = [f"<p><strong>{_html.escape(title)}</strong></p>"]
    if items:
        recent = f"<p>Showing latest {len(items)} updates.</p>" if len(items) < n else ""
        if recent:
            html_parts.append(recent)
        html_items = "".join(item for item, _ in items)
        html_parts.append(f"<ol>{html_items}</ol>")
    if footer_text:
        html_parts.append(f"<p>{_html.escape(footer_text)}</p>")
    return body, "".join(html_parts)
