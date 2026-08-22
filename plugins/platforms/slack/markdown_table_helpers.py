"""GFM pipe-table preprocessing helpers for Slack mrkdwn.

Extracted verbatim from ``plugins/platforms/slack/adapter.py`` (adapter
god-file slice R1, consensus window 142-252). Slack mrkdwn does not render
GFM-style pipe tables; these helpers wrap them in ``` fences and pad cells
to per-column max display width with East-Asian Wide / CJK awareness, so
monospace code-block rendering shows readable, aligned columns.
"""

import re
import unicodedata
from typing import List

# ── GFM markdown table preprocessing ──────────────────────────────────────
# Slack mrkdwn does not render GFM-style pipe tables — they appear as literal
# pipes. Wrapping in ``` fences makes them render as monospace preformatted
# text, and padding cells to per-column max display width (with East-Asian
# Wide / CJK awareness) keeps the columns aligned for the reader.

_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*){1,}\|?\s*$"
)


def _is_table_row(line: str) -> bool:
    """Return True if *line* could plausibly be a table data row."""
    stripped = line.strip()
    return bool(stripped) and "|" in stripped


def _disp_width(s: str) -> int:
    """Monospace display width: East-Asian Wide / Full-width chars count as 2."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(cell: str, width: int) -> str:
    """Right-pad *cell* with spaces until its display width equals *width*."""
    delta = width - _disp_width(cell)
    return cell + (" " * delta if delta > 0 else "")


def _split_table_row(line: str) -> List[str]:
    """Split a ``| a | b | c |`` row into trimmed cells (outer pipes optional)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _align_table(rows: List[str]) -> List[str]:
    """Re-emit a markdown table with cells padded to per-column max display width.

    *rows[0]* is the header, *rows[1]* is the GFM separator (regenerated to
    match new column widths), and *rows[2:]* are data rows. Cells are
    normalized to a uniform column count (missing cells filled with empty
    strings) before width calculation.
    """
    if len(rows) < 2:
        return rows
    parsed = [_split_table_row(r) for r in rows]
    n_cols = max(len(r) for r in parsed)
    for r in parsed:
        while len(r) < n_cols:
            r.append("")
    sep_idx = 1
    parsed[sep_idx] = ["---"] * n_cols  # placeholder; regenerated below
    widths = [max(_disp_width(r[c]) for r in parsed) for c in range(n_cols)]
    out: List[str] = []
    for idx, row in enumerate(parsed):
        if idx == sep_idx:
            cells = ["-" * widths[c] for c in range(n_cols)]
        else:
            cells = [_pad(row[c], widths[c]) for c in range(n_cols)]
        out.append("| " + " | ".join(cells) + " |")
    return out


def _wrap_markdown_tables(text: str) -> str:
    """Wrap GFM pipe tables in ``` fences and align column widths.

    Detected by a row containing ``|`` immediately followed by a delimiter row
    matching :data:`_TABLE_SEPARATOR_RE`. Subsequent pipe-containing non-blank
    lines are consumed as the table body. Tables already inside fenced code
    blocks are left alone.
    """
    if not text or "|" not in text or "-" not in text:
        return text

    lines = text.split("\n")
    out: List[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue
        if (
            "|" in line
            and i + 1 < len(lines)
            and _TABLE_SEPARATOR_RE.match(lines[i + 1])
        ):
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
