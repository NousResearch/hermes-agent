"""CJK/wide-character-aware re-alignment of model-emitted markdown tables.

Models pad tables assuming one cell per character; CJK glyphs and most emoji
take two, so body rows drift right on real terminals. This rebuilds padding
with ``wcwidth.wcswidth`` while preserving pipes/dashes (Rich already aligns CJK).
Deliberately conservative: only contiguous ``| ... |`` blocks with a divider are
rewritten; single-line/mid-stream fragments pass through (callers buffer rows and
flush complete blocks). ``wcswidth`` returns ``-1`` for some emoji+variation-selector
sequences (``⚠️``); those clamp to 0 — a 1-cell drift beats widening every table.
"""

from __future__ import annotations

import re
from typing import List

from wcwidth import wcswidth

__all__ = ["is_table_divider", "looks_like_table_row", "realign_markdown_tables", "split_table_row"]


_DIVIDER_CELL_RE = re.compile(r"^\s*:?-{3,}:?\s*$")
_MIN_COL_WIDTH = 3  # matches the divider's minimum dash run.


def _disp_width(s: str) -> int:
    """``wcswidth`` clamped to >= 0 (it returns -1 for control/unknown sequences)."""
    return max(wcswidth(s), 0)


def split_table_row(row: str) -> List[str]:
    """Split ``| a | b | c |`` into ``["a", "b", "c"]`` with trims."""
    return [c.strip() for c in row.strip().removeprefix("|").removesuffix("|").split("|")]


def is_table_divider(row: str) -> bool:
    """True when ``row`` is a markdown table separator line."""
    cells = split_table_row(row)
    return len(cells) > 1 and all(_DIVIDER_CELL_RE.match(c) for c in cells)


def looks_like_table_row(row: str) -> bool:
    """True when ``row`` could plausibly be a markdown table row.

    Intentionally permissive for streaming callers deciding whether to buffer a
    line (a false positive at most delays printing one line). A leading pipe is
    the strongest signal; without it we accept >= 2 pipes so models that omit
    the leading pipe still match.
    """
    stripped = row.strip()
    return bool(stripped) and (stripped.startswith("|") or stripped.count("|") >= 2)


def _render_block(rows: List[List[str]], available_width: int | None = None) -> List[str]:
    """Render ``rows`` (header + body, divider implied) at uniform widths.

    When the horizontal table would exceed ``available_width`` fall back to a
    vertical key-value rendering: terminal soft-wrap mid-cell destroys alignment
    visually even when the bytes are perfectly padded.
    """
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    widths = [max(_MIN_COL_WIDTH, *(_disp_width(r[c]) for r in rows)) for c in range(ncols)]
    # `| ` + cell + ` ` per column, plus the closing `|`.
    if available_width is not None and sum(widths) + 3 * ncols + 1 > max(available_width, 20):
        return _render_vertical(rows, ncols, available_width)

    def _row(cells: List[str]) -> str:
        return "| " + " | ".join(c + " " * max(0, widths[k] - _disp_width(c)) for k, c in enumerate(cells)) + " |"

    out = [_row(rows[0]), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out.extend(_row(r) for r in rows[1:])
    return out


def _hard_break(word: str, w: int) -> List[str]:
    """Split a single over-wide word into display-width-``w`` chunks."""
    out: List[str] = []
    buf, bw = "", 0
    for ch in word:
        cw = _disp_width(ch) or 1
        if bw + cw > w and buf:
            out.append(buf)
            buf, bw = ch, cw
        else:
            buf, bw = buf + ch, bw + cw
    return out + [buf] if buf else out


def _wrap_to_width(text: str, width: int) -> List[str]:
    """Soft-wrap ``text`` at word boundaries to ``width`` display cells.

    Words wider than ``width`` are hard-broken. Empty input yields a single
    empty string so the caller's row count stays predictable.
    """
    if width <= 0 or not text:
        return [text]
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current, current_w = "", 0

    def _start(word: str, ww: int) -> None:
        nonlocal current, current_w
        if ww <= width:
            current, current_w = word, ww
        else:
            pieces = _hard_break(word, width)
            lines.extend(pieces[:-1])
            current = pieces[-1] if pieces else ""
            current_w = _disp_width(current)

    for word in words:
        ww = _disp_width(word)
        if not current:
            _start(word, ww)
        elif current_w + 1 + ww <= width:
            current += " " + word
            current_w += 1 + ww
        else:
            lines.append(current)
            _start(word, ww)
    if current:
        lines.append(current)
    return lines or [""]


def _render_vertical(rows: List[List[str]], ncols: int, available_width: int) -> List[str]:
    """Render a too-wide table as ``Header: value`` blocks (Claude Code's narrow fallback).

    Each body row becomes one block with continuation lines indented two spaces,
    blocks separated by a thin ``─`` rule; every line stays under ``available_width``.
    """
    if not rows:
        return []
    labels = [h or f"Column {i + 1}" for i, h in enumerate(rows[0] + [""] * (ncols - len(rows[0])))]
    separator = "─" * (max(20, min(40, available_width - 2)) if available_width else 30)
    cont_budget = max(10, available_width - 2)  # continuation lines are indented two spaces
    out: List[str] = []
    for ri, row in enumerate(rows[1:]):
        if ri > 0:
            out.append(separator)
        for ci, label in enumerate(labels):
            value = row[ci] if ci < len(row) else ""
            if not value:
                out.append(f"{label}:")
                continue
            wrapped = _wrap_to_width(value, max(10, available_width - _disp_width(label) - 2))
            out.append(f"{label}: {wrapped[0]}")
            # Re-flow continuation text at the wider continuation budget.
            for cl in _wrap_to_width(" ".join(wrapped[1:]), cont_budget) if len(wrapped) > 1 else ():
                if cl.strip():
                    out.append(f"  {cl}")
    return out


def realign_markdown_tables(text: str, available_width: int | None = None) -> str:
    """Rewrite every ``| ... |`` + divider block with wcwidth-aware padding.

    Non-table lines are returned verbatim, so this is safe on arbitrary prose.
    With ``available_width`` (terminal cells), tables wider than that render as
    vertical key-value pairs instead of soft-wrapping mid-cell.
    """
    if "|" not in text:
        return text
    lines = text.split("\n")
    out: List[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        # A table starts with a header row whose next line is a divider.
        if "|" in line and i + 1 < n and is_table_divider(lines[i + 1]):
            header = split_table_row(line)
            body: List[List[str]] = []
            j = i + 2
            while j < n and "|" in lines[j] and lines[j].strip():
                if not is_table_divider(lines[j]):
                    body.append(split_table_row(lines[j]))
                j += 1
            if any(header) or body:
                out.extend(_render_block([header] + body, available_width))
                i = j
                continue
        out.append(line)
        i += 1
    return "\n".join(out)
