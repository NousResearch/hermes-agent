"""Matrix Tool activity payload helpers (always-visible sticky list)."""

from __future__ import annotations

import html as _html
import re
from typing import Iterable, List, Sequence, Tuple


def matrix_tool_activity_bodies(lines: Sequence[str] | Iterable[str]) -> Tuple[str, str]:
    """Build plain body + HTML for Matrix tool progress.

    Contract:
    - plain body: ``🛠 Tool activity (N updates)`` only
    - HTML: always-visible ``<p><strong>…</strong></p><ol><li>…</li></ol>``
    - no fences, details, spoilers, or multi-line dumps
    """
    cleaned: List[str] = []
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
        cleaned.append(s)
    n = len(cleaned)
    body = f"🛠 Tool activity ({n} update{'s' if n != 1 else ''})"
    if not cleaned:
        return body, f"<p><strong>{_html.escape(body)}</strong></p>"
    items = "".join(f"<li>{_html.escape(item)}</li>" for item in cleaned)
    html_body = f"<p><strong>{_html.escape(body)}</strong></p><ol>{items}</ol>"
    return body, html_body
