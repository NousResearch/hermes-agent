"""Display-width helpers for fixed-column CLI tables.

``str`` slices and format-spec padding (``{s:<32}``) count characters, but CJK characters occupy
two terminal columns, so any cell holding them shifts every column to its right (#44199). These
helpers clip and pad by terminal columns instead, using the same ``wcwidth`` approach (and the
clamp-negative-to-zero convention) as ``agent/markdown_tables.py``. Pure-ASCII text renders
byte-identically to the slice/``ljust`` it replaces.
"""

from __future__ import annotations

from wcwidth import wcswidth, wcwidth


def disp_width(text: str) -> int:
    """Terminal display width of *text*; ``wcswidth``'s ``-1`` (control chars) clamps to 0."""
    return max(wcswidth(text), 0)


def clip_to_width(text: str, width: int) -> str:
    """Truncate *text* to at most *width* display columns, never splitting a double-width char."""
    if disp_width(text) <= width:
        return text
    out, used = [], 0
    for ch in text:
        w = max(wcwidth(ch), 0)
        if used + w > width:
            break
        out.append(ch)
        used += w
    return "".join(out)


def pad_to_width(text: str, width: int) -> str:
    """Column-aware ``str.ljust``: pad *text* to *width* display columns (never truncates)."""
    return text + " " * max(0, width - disp_width(text))
