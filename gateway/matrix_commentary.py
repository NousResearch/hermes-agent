"""Matrix interim-commentary payload helpers (always-visible sticky list)."""

from __future__ import annotations

import html as _html
import json
import re
from collections import deque
from typing import Iterable, Sequence, Tuple


_ITEMS_JSON_LIMIT = 12_000
_ITEM_CHAR_LIMIT = 600


def _item_html(text: str) -> str:
    """Escape one commentary item, keeping paragraph/line breaks visible."""
    s = str(text or "").strip()
    if not s:
        return ""
    if len(s) > _ITEM_CHAR_LIMIT:
        s = s[: _ITEM_CHAR_LIMIT - 3] + "..."
    parts = [p.strip() for p in re.split(r"\n+", s) if p.strip()]
    if not parts:
        return ""
    return "<br>".join(_html.escape(p) for p in parts)


def matrix_commentary_bodies(items: Sequence[str] | Iterable[str]) -> Tuple[str, str]:
    """Build plain body + HTML for Matrix interim commentary.

    Contract:
    - plain body: ``💬 Commentary (N updates)``
    - HTML: always-visible title + ordered list of escaped prose
    - no ``<details>``, fences, or tool dumps
    - newest items kept when the escaped-JSON budget is exceeded
    """
    kept: deque[tuple[str, int]] = deque()
    kept_size = 0
    n = 0
    for raw in items:
        inner = _item_html(raw)
        if not inner:
            continue
        n += 1
        li = f"<li>{inner}</li>"
        size = len(json.dumps(li, ensure_ascii=True))
        kept.append((li, size))
        kept_size += size
        while kept_size > _ITEMS_JSON_LIMIT:
            _, removed = kept.popleft()
            kept_size -= removed
    title = f"💬 Commentary ({n} update{'s' if n != 1 else ''})" if n else "💬 Commentary"
    body = title
    html_parts = [f"<p><strong>{_html.escape(title)}</strong></p>"]
    if kept:
        if len(kept) < n:
            html_parts.append(f"<p>Showing latest {len(kept)} updates.</p>")
        html_parts.append("<ol>" + "".join(item for item, _ in kept) + "</ol>")
    return body, "".join(html_parts)
