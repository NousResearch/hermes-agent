"""Terminal column formatting utilities for CLI output.

Provides CJK-aware width calculation and alignment for terminal tables.
Unlike agent.markdown_tables (which handles Markdown pipe-table syntax),
this module handles plain-text column alignment for direct terminal output.
"""

from __future__ import annotations

from wcwidth import wcswidth

__all__ = [
    "disp_width",
    "truncate_for_width",
    "pad_left",
    "pad_right",
]


def disp_width(s: str) -> int:
    """Return the display width of a string in terminal columns.

    Uses wcwidth to correctly handle CJK characters and other wide glyphs.
    Returns 0 for control characters and invalid sequences.
    """
    w = wcswidth(s)
    return w if w > 0 else 0


def truncate_for_width(s: str, max_width: int, ellipsis: str = "…") -> str:
    """Truncate string to fit within max_width display columns.

    If the string is wider than max_width, it is truncated and an ellipsis
    is appended. The result's display width will not exceed max_width.
    """
    if disp_width(s) <= max_width:
        return s
    ellipsis_w = disp_width(ellipsis)
    target = max_width - ellipsis_w
    if target <= 0:
        return ellipsis[:1] if max_width >= 1 else ""
    result = ""
    for ch in s:
        if disp_width(result + ch) > target:
            break
        result += ch
    return result + ellipsis


def pad_left(s: str, width: int) -> str:
    """Right-align string within width columns, truncating if necessary.

    Pads with spaces on the left. If the string is wider than width,
    it is truncated with an ellipsis.
    """
    s = truncate_for_width(s, width)
    return " " * max(0, width - disp_width(s)) + s


def pad_right(s: str, width: int) -> str:
    """Left-align string within width columns, truncating if necessary.

    Pads with spaces on the right. If the string is wider than width,
    it is truncated with an ellipsis.
    """
    s = truncate_for_width(s, width)
    return s + " " * max(0, width - disp_width(s))