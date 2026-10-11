"""Markdown table formatting for Slack's mrkdwn surface."""

import re
import unicodedata
from typing import List

# Slack renders GFM pipe tables as literal pipes; fence and align them for readability.
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*){1,}\|?\s*$")


def _is_table_row(line: str) -> bool:
    """Return True if *line* could plausibly be a table data row."""
    stripped = line.strip()
    return bool(stripped) and "|" in stripped


def _disp_width(s: str) -> int:
    """Monospace display width: East-Asian Wide / Full-width chars count as 2."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(cell: str, width: int) -> str:
    """Right-pad *cell* with spaces until its display width equals *width*."""
    return cell + " " * max(width - _disp_width(cell), 0)


def _split_table_row(line: str) -> list[str]:
    """Split a ``| a | b | c |`` row into trimmed cells (outer pipes optional)."""
    s = line.strip()
    s = s.removeprefix("|")
    s = s.removesuffix("|")
    return [c.strip() for c in s.split("|")]


def _align_table(rows: list[str]) -> list[str]:
    """Re-emit a markdown table padded to per-column display width. rows[1] is the GFM separator
    (regenerated); short rows are padded to a uniform column count first."""
    if len(rows) < 2:
        return rows
    parsed = [_split_table_row(r) for r in rows]
    n_cols = max(len(r) for r in parsed)
    parsed = [r + [""] * (n_cols - len(r)) for r in parsed]
    parsed[1] = ["---"] * n_cols
    widths = [max(_disp_width(r[c]) for r in parsed) for c in range(n_cols)]
    out: list[str] = []
    for idx, row in enumerate(parsed):
        cells = ["-" * widths[c] if idx == 1 else _pad(row[c], widths[c]) for c in range(n_cols)]
        out.append("| " + " | ".join(cells) + " |")
    return out


def _wrap_markdown_tables(text: str) -> str:
    """Wrap GFM pipe tables in fences and align columns; tables already fenced are left alone."""
    if not text or "|" not in text or "-" not in text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif (
            not in_fence and "|" in line and i + 1 < len(lines)
            and _TABLE_SEPARATOR_RE.match(lines[i + 1])):
            block = [line, lines[i + 1]]
            j = i + 2
            while j < len(lines) and _is_table_row(lines[j]):
                block.append(lines[j])
                j += 1
            out.append("```")
            out.extend(_align_table(block))
            out.append("```")
            i = j
            continue
        out.append(line)
        i += 1
    return "\n".join(out)
