"""CJK/wide-character-aware re-alignment of model-emitted markdown tables.

Models pad markdown tables assuming each character occupies one terminal
cell. CJK glyphs and most emoji render as two cells, so the model's
spacing collapses into drift the moment a table reaches a real terminal —
header pipes line up, every body row drifts right by N cells per CJK
char.

This module rebuilds row padding using ``wcwidth.wcswidth`` (display
columns), preserving the table's pipes and dashes so it still reads as a
plain-text table in ``strip`` / unrendered display modes. Standard Rich
markdown rendering already aligns CJK correctly inside a wide enough
panel; this helper is for the paths that print the model's text more or
less verbatim.

The helper is deliberately conservative:

* Only contiguous ``| ... |`` blocks with a divider line are rewritten.
* Anything that does not look like a table is passed through unchanged.
* Single-line / mid-stream fragments are left alone — callers buffer
  table rows and flush them once the block is complete.

There is a small, intentional caveat: ``wcwidth`` returns ``-1`` for some
emoji-with-variation-selector sequences (e.g. ``⚠️``); we clamp those to
0 so they do not corrupt the column width math. The 1-cell drift on
those specific glyphs is preferable to silently widening every table
that contains one.
"""

from __future__ import annotations

import re
from typing import List

from wcwidth import wcswidth

__all__ = [
    "is_table_divider",
    "looks_like_table_row",
    "realign_markdown_tables",
    "split_table_row",
]


_DIVIDER_CELL_RE = re.compile(r"^\s*:?-{3,}:?\s*$")
_MIN_COL_WIDTH = 3  # matches the divider's minimum dash run.


def _disp_width(s: str) -> int:
    """``wcswidth`` clamped to a non-negative integer.

    ``wcswidth`` returns ``-1`` when it encounters a control char or an
    unknown sequence; treat those as zero-width rather than letting a
    negative number flow into ``max`` and break the column-width math.
    """

    w = wcswidth(s)
    return w if w > 0 else 0


def _pad_to_width(s: str, target: int) -> str:
    return s + " " * max(0, target - _disp_width(s))


def split_table_row(row: str) -> List[str]:
    """Split ``| a | b | c |`` into ``["a", "b", "c"]`` with trims."""

    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def is_table_divider(row: str) -> bool:
    """True when ``row`` is a markdown table separator line."""

    cells = split_table_row(row)
    return len(cells) > 1 and all(_DIVIDER_CELL_RE.match(c) for c in cells)


def looks_like_table_row(row: str) -> bool:
    """True when ``row`` could plausibly be a markdown table row.

    Used by streaming callers to decide whether to buffer an in-flight
    line. We are intentionally permissive here — the realigner itself
    only rewrites blocks that are accompanied by a divider, so a false
    positive here at most delays the print of one line.
    """

    if "|" not in row:
        return False
    stripped = row.strip()
    if not stripped:
        return False
    # A leading pipe is the strongest signal; without it we still allow
    # rows with at least two pipes so models that omit the leading pipe
    # don't slip past us.
    if stripped.startswith("|"):
        return True
    return stripped.count("|") >= 2


def _render_block(rows: List[List[str]]) -> List[str]:
    """Render ``rows`` (header + body, divider implied) at uniform widths."""

    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]

    widths = [
        max(_MIN_COL_WIDTH, *(_disp_width(r[c]) for r in rows))
        for c in range(ncols)
    ]

    def _row(cells: List[str]) -> str:
        return (
            "| "
            + " | ".join(_pad_to_width(c, widths[k]) for k, c in enumerate(cells))
            + " |"
        )

    out = [_row(rows[0])]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows[1:]:
        out.append(_row(r))
    return out


def realign_markdown_tables(text: str) -> str:
    """Rewrite every ``| ... |`` + divider block with wcwidth-aware padding.

    Lines that are not part of a recognised table are returned verbatim,
    so this is safe to apply to arbitrary assistant prose.
    """

    if "|" not in text:
        return text

    lines = text.split("\n")
    out: List[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        # A table starts with a header row whose next line is a divider.
        if (
            "|" in line
            and i + 1 < n
            and is_table_divider(lines[i + 1])
        ):
            header = split_table_row(line)
            body: List[List[str]] = []
            j = i + 2
            while j < n and "|" in lines[j] and lines[j].strip():
                if is_table_divider(lines[j]):
                    j += 1
                    continue
                body.append(split_table_row(lines[j]))
                j += 1

            if any(c for c in header) or body:
                out.extend(_render_block([header] + body))
                i = j
                continue
        out.append(line)
        i += 1

    return "\n".join(out)
