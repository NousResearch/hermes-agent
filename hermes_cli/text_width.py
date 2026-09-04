"""Unicode-aware text width helpers for terminal output."""

from __future__ import annotations

import unicodedata


_ZERO_WIDTH_CATEGORIES = {"Mn", "Me", "Cf"}
_WIDE_EAST_ASIAN_WIDTHS = {"F", "W"}


def display_width(value: object) -> int:
    """Return the number of terminal cells needed to display *value*.

    Python's :func:`len` counts Unicode code points, but common terminals render
    full-width CJK characters as two cells and combining marks as zero cells.
    Use this for column calculations in plain-text tables and picker rows.
    """
    text = str(value)
    width = 0
    for char in text:
        if unicodedata.category(char) in _ZERO_WIDTH_CATEGORIES:
            continue
        width += 2 if unicodedata.east_asian_width(char) in _WIDE_EAST_ASIAN_WIDTHS else 1
    return width


def ljust_display(value: object, width: int) -> str:
    """Left-pad *value* to a target terminal display width."""
    text = str(value)
    padding = max(0, width - display_width(text))
    return text + (" " * padding)
