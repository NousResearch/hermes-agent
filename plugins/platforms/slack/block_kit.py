"""Render agent markdown into Slack Block Kit blocks.

Opt-in (``slack.extra.rich_blocks: true``) alternative to the flat mrkdwn
``text`` payload produced by :meth:`SlackAdapter.format_message`.  Block Kit
gives us real structural primitives — section headers, dividers, and true
*nested* lists via ``rich_text`` — that plain mrkdwn can only approximate.

Design constraints (why this module is deliberately conservative):

* **Markdown pipe-tables render as native ``table`` blocks** — real grid
  cells with per-column alignment and inline-formatted ``rich_text`` content.
  A table that exceeds Slack's limits (100 rows / 20 cols / 10k aggregate
  cell chars) or won't parse falls back to aligned monospace
  ``rich_text_preformatted`` so a large table never breaks the message.
* **Slack caps a message at 50 blocks** and a ``section``/text object at 3000
  characters.  :func:`render_blocks` enforces both and, if the content simply
  cannot be expressed within them, returns ``None`` so the caller falls back
  to the plain-text path.  A rich render is a nice-to-have; it must never lose
  a message.
* **Every blocks payload MUST ship a ``text`` fallback.**  Slack uses it for
  notifications, screen readers, and old clients.  This module only builds the
  ``blocks`` list; the adapter pairs it with the existing mrkdwn string.

The renderer never raises: any unexpected input degrades to ``None`` (caller
uses plain text).  It is a pure function of its input — no Slack client, no
adapter state — so it is trivially unit-testable.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Slack Block Kit hard limits (https://docs.slack.dev/reference/block-kit/blocks)
MAX_BLOCKS = 50
MAX_SECTION_TEXT = 3000
MAX_HEADER_TEXT = 150
# Native table block limits (https://docs.slack.dev/reference/block-kit/blocks/table-block)
MAX_TABLE_ROWS = 100
MAX_TABLE_COLS = 20
MAX_TABLE_CHARS = 10000  # aggregate across all cells

Block = Dict[str, Any]

# ----------------------------------------------------------------------------
# Line classification
# ----------------------------------------------------------------------------

_HR_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_HEADER_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_ORDERED_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$")


def _indent_level(spaces: str) -> int:
    """Map leading whitespace to a nesting level (2 spaces or 1 tab per level)."""
    width = 0
    for ch in spaces:
        width += 4 if ch == "\t" else 1
    return min(width // 2, 5)  # Slack rich_text_list supports up to indent 5


# ----------------------------------------------------------------------------
# Inline markdown → rich_text elements
# ----------------------------------------------------------------------------

# Order matters: code first (opaque), then links, then emphasis.
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^()\s]+(?:\([^()]*\)[^()\s]*)*)\)")
_BOLD_RE = re.compile(r"(?:\*\*|__)(.+?)(?:\*\*|__)")
_ITALIC_RE = re.compile(r"(?<![\*_])(?:\*|_)(?![\*_\s])(.+?)(?<![\*_\s])(?:\*|_)(?![\*_])")
_STRIKE_RE = re.compile(r"~~(.+?)~~")


def _inline_elements(text: str) -> List[Dict[str, Any]]:
    """Parse a run of inline markdown into rich_text section child elements.

    Produces ``text`` elements (optionally styled bold/italic/strike/code) and
    ``link`` elements.  Unmatched markup is emitted verbatim as plain text, so
    this never loses characters.
    """
    elements: List[Dict[str, Any]] = []

    def emit_text(s: str, style: Optional[Dict[str, bool]] = None) -> None:
        if not s:
            return
        el: Dict[str, Any] = {"type": "text", "text": s}
        if style:
            el["style"] = style
        elements.append(el)

    # Tokenize by the highest-priority markers first using a single scan.
    # We recursively split on code, then links, then emphasis to keep spans
    # from overlapping incorrectly.
    def walk(s: str, style: Dict[str, bool]) -> None:
        pos = 0
        # inline code is opaque — no nested styling
        for m in _INLINE_CODE_RE.finditer(s):
            _walk_links(s[pos:m.start()], style)
            code_style = dict(style)
            code_style["code"] = True
            emit_text(m.group(1), code_style or None)
            pos = m.end()
        _walk_links(s[pos:], style)

    def _walk_links(s: str, style: Dict[str, bool]) -> None:
        pos = 0
        for m in _LINK_RE.finditer(s):
            _walk_emphasis(s[pos:m.start()], style)
            link_el: Dict[str, Any] = {"type": "link", "url": m.group(2), "text": m.group(1)}
            if style:
                link_el["style"] = dict(style)
            elements.append(link_el)
            pos = m.end()
        _walk_emphasis(s[pos:], style)

    def _walk_emphasis(s: str, style: Dict[str, bool]) -> None:
        if not s:
            return
        # Try bold, then strike, then italic, recursing into the inner span.
        for rx, key in ((_BOLD_RE, "bold"), (_STRIKE_RE, "strike"), (_ITALIC_RE, "italic")):
            m = rx.search(s)
            if m:
                _walk_emphasis(s[:m.start()], style)
                inner_style = dict(style)
                inner_style[key] = True
                _walk_emphasis(m.group(1), inner_style)
                _walk_emphasis(s[m.end():], style)
                return
        emit_text(s, dict(style) if style else None)

    walk(text, {})
    return elements or [{"type": "text", "text": text}]


# ----------------------------------------------------------------------------
# Structural block builders
# ----------------------------------------------------------------------------


def _header_block(text: str) -> Block:
    # header blocks are plain_text only, 150 char cap.
    clean = re.sub(r"[*_~`]", "", text).strip()
    if len(clean) > MAX_HEADER_TEXT:
        clean = clean[: MAX_HEADER_TEXT - 1] + "…"
    return {"type": "header", "text": {"type": "plain_text", "text": clean, "emoji": True}}


def _divider_block() -> Block:
    return {"type": "divider"}


def _preformatted_block(text: str) -> Block:
    # rich_text_preformatted renders monospace; used for code fences + tables.
    return {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_preformatted",
                "elements": [{"type": "text", "text": text.rstrip("\n")}],
            }
        ],
    }


def _quote_block(lines: List[str]) -> Block:
    section_children: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines):
        if i:
            section_children.append({"type": "text", "text": "\n"})
        section_children.extend(_inline_elements(ln))
    return {
        "type": "rich_text",
        "elements": [{"type": "rich_text_quote", "elements": section_children}],
    }


def _list_block(items: List[Tuple[int, bool, str]]) -> Block:
    """Build ONE rich_text block from consecutive list items.

    ``items`` is a list of ``(indent, ordered, text)``.  Each contiguous run
    sharing the same (indent, ordered) becomes a ``rich_text_list`` element;
    indentation changes start a new element, which is how Slack renders true
    nesting.
    """
    elements: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    cur_key: Optional[Tuple[int, bool]] = None
    for indent, ordered, text in items:
        key = (indent, ordered)
        if key != cur_key:
            cur = {
                "type": "rich_text_list",
                "style": "ordered" if ordered else "bullet",
                "indent": indent,
                "elements": [],
            }
            elements.append(cur)
            cur_key = key
        if cur is None:
            raise ValueError("Slack Block Kit: current element unexpectedly None during tree traversal")
        cur["elements"].append(
            {"type": "rich_text_section", "elements": _inline_elements(text)}
        )
    return {"type": "rich_text", "elements": elements}


def _section_block(text: str) -> Block:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


# ----------------------------------------------------------------------------
# Table handling — native Block Kit ``table`` block, monospace fallback
# ----------------------------------------------------------------------------


def _parse_alignment(sep_line: str) -> List[str]:
    """Parse a markdown separator row (``|:--|:-:|--:|``) into column aligns.

    Returns a list of ``"left"``/``"center"``/``"right"`` per column.
    """
    aligns: List[str] = []
    for cell in sep_line.strip().strip("|").split("|"):
        c = cell.strip()
        left = c.startswith(":")
        right = c.endswith(":")
        if left and right:
            aligns.append("center")
        elif right:
            aligns.append("right")
        else:
            aligns.append("left")
    return aligns


def _split_row(row: str) -> List[str]:
    """Split a markdown table row into trimmed cell strings.

    Respects backslash-escaped pipes (``\\|``) so they aren't treated as
    column separators.
    """
    # Temporarily protect escaped pipes, split on real ones, then restore.
    protected = row.strip().strip("|").replace(r"\|", "\x00PIPE\x00")
    return [c.strip().replace("\x00PIPE\x00", "|") for c in protected.split("|")]


def _rich_text_cell(text: str) -> Dict[str, Any]:
    """A ``rich_text`` table cell carrying inline-formatted content."""
    return {
        "type": "rich_text",
        "elements": [
            {"type": "rich_text_section", "elements": _inline_elements(text)}
        ],
    }


def _table_block(rows: List[str], sep_line: str) -> Optional[Block]:
    """Build a native Slack ``table`` block from markdown pipe-table rows.

    ``rows`` includes the header row (index 0) and body rows; ``sep_line`` is
    the ``|---|`` alignment row (already consumed by the caller).  Returns
    ``None`` when the table exceeds Slack's limits (100 rows / 20 cols /
    10,000 aggregate cell chars) or parses to nothing — the caller then falls
    back to the monospace preformatted rendering.
    """
    parsed = [_split_row(r) for r in rows if r.strip()]
    if not parsed:
        return None
    ncols = max(len(r) for r in parsed)
    # Reject rather than silently truncate beyond Slack's structural limits.
    if len(parsed) > MAX_TABLE_ROWS or ncols > MAX_TABLE_COLS:
        return None
    for r in parsed:
        r.extend([""] * (ncols - len(r)))

    total_chars = sum(len(c) for r in parsed for c in r)
    if total_chars > MAX_TABLE_CHARS:
        return None

    aligns = _parse_alignment(sep_line)
    column_settings: List[Optional[Dict[str, Any]]] = []
    for c in range(min(ncols, MAX_TABLE_COLS)):
        align = aligns[c] if c < len(aligns) else "left"
        # Only emit a setting when it differs from the default (left, no wrap);
        # use null to skip a column, per the Slack schema.
        column_settings.append({"align": align} if align != "left" else None)

    block: Block = {
        "type": "table",
        "rows": [[_rich_text_cell(cell) for cell in row] for row in parsed],
    }
    if any(cs is not None for cs in column_settings):
        block["column_settings"] = column_settings
    return block


def _render_table(rows: List[str]) -> str:
    """Render markdown pipe-table rows as aligned monospace text (fallback)."""
    parsed: List[List[str]] = []
    for r in rows:
        cells = _split_row(r)
        parsed.append(cells)
    if not parsed:
        return "\n".join(rows)
    ncols = max(len(r) for r in parsed)
    for r in parsed:
        r.extend([""] * (ncols - len(r)))
    widths = [max(len(r[c]) for r in parsed) for c in range(ncols)]
    out_lines = []
    for ri, r in enumerate(parsed):
        line = " | ".join(r[c].ljust(widths[c]) for c in range(ncols))
        out_lines.append(line.rstrip())
        if ri == 0:  # header underline
            out_lines.append("-+-".join("-" * widths[c] for c in range(ncols)))
    return "\n".join(out_lines)


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def render_blocks(
    markdown: str,
    mrkdwn_fn=None,
) -> Optional[List[Block]]:
    """Convert agent markdown to a Slack Block Kit ``blocks`` list.

    Args:
        markdown: The agent's response text (standard markdown).
        mrkdwn_fn: Optional callable converting a markdown paragraph to Slack
            mrkdwn for ``section`` blocks (the adapter passes
            ``format_message``).  When ``None``, the raw paragraph text is used.

    Returns:
        A list of Block Kit block dicts, or ``None`` when the content is empty,
        exceeds Slack's structural limits, or hits an unexpected shape — the
        caller then falls back to the flat ``text`` payload.  Never raises.
    """
    if not markdown or not markdown.strip():
        return None

    fmt = mrkdwn_fn or (lambda s: s)

    try:
        blocks: List[Block] = []
        lines = markdown.replace("\r\n", "\n").split("\n")
        i = 0
        n = len(lines)
        para: List[str] = []

        def flush_para() -> None:
            if not para:
                return
            text = "\n".join(para).strip()
            para.clear()
            if not text:
                return
            rendered = fmt(text)
            # Split oversized sections on the 3000-char limit.
            for chunk in _split_text(rendered, MAX_SECTION_TEXT):
                blocks.append(_section_block(chunk))

        while i < n:
            line = lines[i]

            # Blank line: paragraph boundary
            if not line.strip():
                flush_para()
                i += 1
                continue

            # Fenced code block
            fence = _FENCE_RE.match(line)
            if fence:
                flush_para()
                marker = fence.group(1)
                body: List[str] = []
                i += 1
                while i < n and not lines[i].lstrip().startswith(marker):
                    body.append(lines[i])
                    i += 1
                i += 1  # consume closing fence
                blocks.append(_preformatted_block("\n".join(body)))
                continue

            # Horizontal rule → divider
            if _HR_RE.match(line):
                flush_para()
                blocks.append(_divider_block())
                i += 1
                continue

            # ATX header
            hm = _HEADER_RE.match(line)
            if hm:
                flush_para()
                blocks.append(_header_block(hm.group(2)))
                i += 1
                continue

            # Pipe table: current line has a pipe AND next line is a separator
            if "|" in line and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
                flush_para()
                header_row = line
                sep_line = lines[i + 1]
                trows = [header_row]
                i += 2  # skip header + separator
                while i < n and "|" in lines[i] and lines[i].strip():
                    trows.append(lines[i])
                    i += 1
                # Prefer a native Block Kit table; fall back to aligned
                # monospace when it exceeds Slack's table limits or won't parse.
                table = _table_block(trows, sep_line)
                if table is not None:
                    blocks.append(table)
                else:
                    blocks.append(_preformatted_block(_render_table(trows)))
                continue

            # Blockquote group
            if _QUOTE_RE.match(line):
                flush_para()
                qlines: List[str] = []
                while i < n:
                    qm = _QUOTE_RE.match(lines[i])
                    if not qm:
                        break
                    qlines.append(qm.group(1))
                    i += 1
                blocks.append(_quote_block(qlines))
                continue

            # List group (bullets + ordered, with nesting)
            if _BULLET_RE.match(line) or _ORDERED_RE.match(line):
                flush_para()
                items: List[Tuple[int, bool, str]] = []
                while i < n:
                    bm = _BULLET_RE.match(lines[i])
                    om = _ORDERED_RE.match(lines[i])
                    if bm:
                        items.append((_indent_level(bm.group(1)), False, bm.group(2)))
                        i += 1
                    elif om:
                        items.append((_indent_level(om.group(1)), True, om.group(3)))
                        i += 1
                    elif lines[i].strip() and lines[i].startswith((" ", "\t")) and items:
                        # continuation line of the previous item
                        indent, ordered, txt = items[-1]
                        items[-1] = (indent, ordered, txt + " " + lines[i].strip())
                        i += 1
                    else:
                        break
                blocks.append(_list_block(items))
                continue

            # Default: accumulate into a paragraph
            para.append(line)
            i += 1

        flush_para()

        if not blocks:
            return None
        if len(blocks) > MAX_BLOCKS:
            # Too structurally complex to express safely — let the caller fall
            # back to plain text rather than truncating and losing content.
            return None
        return blocks
    except Exception:
        # Never let a rendering bug drop a message.
        return None


def _split_text(text: str, limit: int) -> List[str]:
    """Split ``text`` into <= ``limit``-char chunks on line, then hard, boundaries."""
    if len(text) <= limit:
        return [text]
    out: List[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        out.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        out.append(remaining)
    return out
