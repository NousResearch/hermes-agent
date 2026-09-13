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


def matrix_tool_activity_bodies(lines: Sequence[str] | Iterable[str]) -> Tuple[str, str]:
    """Build plain body + HTML for Matrix tool progress.

    Contract:
    - plain body: ``🛠 Tool activity (N updates)`` only
    - HTML: always-visible ``<p><strong>…</strong></p><ol><li>…</li></ol>``
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
    body = f"🛠 Tool activity ({n} update{'s' if n != 1 else ''})"
    if not items:
        return body, f"<p><strong>{_html.escape(body)}</strong></p>"
    recent = f"<p>Showing latest {len(items)} updates.</p>" if len(items) < n else ""
    html_items = "".join(item for item, _ in items)
    html_body = f"<p><strong>{_html.escape(body)}</strong></p>{recent}<ol>{html_items}</ol>"
    return body, html_body
