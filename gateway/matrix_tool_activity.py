"""Matrix Tool activity payload helpers (always-visible sticky list)."""

from __future__ import annotations

import html as _html
import re
from typing import Iterable, List, Sequence, Tuple


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
    footer_text = str(footer or "").strip() or None
    n = len(cleaned)
    title = (
        f"🛠 Tool activity ({n} update{'s' if n != 1 else ''})"
        if cleaned
        else "🛠 Tool activity"
    )
    body = f"{title} · {footer_text}" if footer_text else title
    html_parts = [f"<p><strong>{_html.escape(title)}</strong></p>"]
    if cleaned:
        items = "".join(f"<li>{_html.escape(item)}</li>" for item in cleaned)
        html_parts.append(f"<ol>{items}</ol>")
    if footer_text:
        html_parts.append(f"<p>{_html.escape(footer_text)}</p>")
    html_body = "".join(html_parts)
    return body, html_body
