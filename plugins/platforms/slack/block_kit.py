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
import unicodedata
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


def _is_list_line(line: str) -> bool:
    """True if ``line`` is a markdown list item (bullet or ordered)."""
    return bool(_BULLET_RE.match(line) or _ORDERED_RE.match(line))


def _indent_level(spaces: str) -> int:
    """Map leading whitespace to a nesting level (2 spaces or 1 tab per level)."""
    width = 0
    for ch in spaces:
        width += 4 if ch == "\t" else 1
    return min(width // 2, 5)  # Slack rich_text_list supports up to indent 5


# ----------------------------------------------------------------------------
# Inline markdown → rich_text elements
# ----------------------------------------------------------------------------

# Order matters: code first (opaque), then Slack entities, then markdown
# links, then bare URLs, then emphasis.
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^()\s]+(?:\([^()]*\)[^()\s]*)*)\)")
_BOLD_RE = re.compile(r"(?:\*\*|__)(.+?)(?:\*\*|__)")
_ITALIC_RE = re.compile(r"(?<![\*_])(?:\*|_)(?![\*_\s])(.+?)(?<![\*_\s])(?:\*|_)(?![\*_])")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
# Manual Slack entities the agent (or upstream text) may already contain:
# user/channel/usergroup mentions, broadcasts, and <url|label> links.  In
# mrkdwn these render natively; inside rich_text ``text`` elements they would
# show up literally, so they must map to their dedicated element types.
_SLACK_ENTITY_RE = re.compile(
    r"<("
    r"@[UW][A-Z0-9]+"
    r"|#C[A-Z0-9]+(?:\|[^>]*)?"
    r"|!(?:here|channel|everyone)"
    r"|!subteam\^[A-Z0-9]+(?:\|[^>]*)?"
    r"|(?:https?|mailto|tel):[^>|]+(?:\|[^>]+)?"
    r")>"
)
_BARE_URL_RE = re.compile(r"https?://[^\s<>|]+")
# Prose punctuation commonly glued to the end of a bare URL (incl. CJK).
_URL_TRAILING_PUNCT = ").,;:!?。，；：！？）】」』\"'”’"


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
    # We recursively split on code, then Slack entities, then markdown links,
    # then bare URLs, then emphasis to keep spans from overlapping incorrectly.
    def walk(s: str, style: Dict[str, bool]) -> None:
        pos = 0
        # inline code is opaque — no nested styling
        for m in _INLINE_CODE_RE.finditer(s):
            _walk_entities(s[pos:m.start()], style)
            code_style = dict(style)
            code_style["code"] = True
            emit_text(m.group(1), code_style or None)
            pos = m.end()
        _walk_entities(s[pos:], style)

    def _walk_entities(s: str, style: Dict[str, bool]) -> None:
        pos = 0
        for m in _SLACK_ENTITY_RE.finditer(s):
            _walk_links(s[pos:m.start()], style)
            elements.append(_entity_element(m.group(1), style))
            pos = m.end()
        _walk_links(s[pos:], style)

    def _entity_element(body: str, style: Dict[str, bool]) -> Dict[str, Any]:
        el: Dict[str, Any]
        if body.startswith("@"):
            el = {"type": "user", "user_id": body[1:]}
        elif body.startswith("#"):
            el = {"type": "channel", "channel_id": body[1:].split("|", 1)[0]}
        elif body.startswith("!subteam^"):
            el = {
                "type": "usergroup",
                "usergroup_id": body[len("!subteam^"):].split("|", 1)[0],
            }
        elif body.startswith("!"):
            # broadcast takes no style property
            return {"type": "broadcast", "range": body[1:]}
        else:
            url, _, label = body.partition("|")
            el = {"type": "link", "url": url}
            if label:
                el["text"] = label
        if style and el["type"] == "link":
            el["style"] = dict(style)
        return el

    def _walk_links(s: str, style: Dict[str, bool]) -> None:
        pos = 0
        for m in _LINK_RE.finditer(s):
            _walk_bare_urls(s[pos:m.start()], style)
            link_el: Dict[str, Any] = {"type": "link", "url": m.group(2), "text": m.group(1)}
            if style:
                link_el["style"] = dict(style)
            elements.append(link_el)
            pos = m.end()
        _walk_bare_urls(s[pos:], style)

    def _walk_bare_urls(s: str, style: Dict[str, bool]) -> None:
        pos = 0
        for m in _BARE_URL_RE.finditer(s):
            url = m.group(0).rstrip(_URL_TRAILING_PUNCT)
            if not url:
                continue
            _walk_emphasis(s[pos:m.start()], style)
            link_el: Dict[str, Any] = {"type": "link", "url": url}
            if style:
                link_el["style"] = dict(style)
            elements.append(link_el)
            pos = m.start() + len(url)
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
    # Slack rejects a rich_text ``text`` element whose string is empty
    # ("must be more than 0 characters") — and one bad element (e.g. an empty
    # table cell) fails the ENTIRE blocks payload. Drop zero-length text
    # elements and guarantee at least one non-empty element in the run.
    elements = [
        el for el in elements if el.get("type") != "text" or el.get("text")
    ]
    return elements or [{"type": "text", "text": " "}]


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
    # Slack rejects a zero-length text element (an empty code fence would fail
    # the whole payload), so fall back to a single space.
    body = text.rstrip("\n") or " "
    return {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_preformatted",
                "elements": [{"type": "text", "text": body}],
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
        assert cur is not None
        cur["elements"].append(
            {"type": "rich_text_section", "elements": _inline_elements(text)}
        )
    return {"type": "rich_text", "elements": elements}


def _section_block(text: str) -> Block:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _rich_para_block(text: str) -> Block:
    """Paragraph as a rich_text block.

    Explicit style objects render reliably where mrkdwn's word-boundary
    emphasis parsing fails — e.g. ``*bold*`` glued to CJK punctuation shows
    literal asterisks in a mrkdwn section but renders bold here.
    """
    return {
        "type": "rich_text",
        "elements": [
            {"type": "rich_text_section", "elements": _inline_elements(text)}
        ],
    }


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
    column_settings: List[Dict[str, Any]] = []
    for c in range(min(ncols, MAX_TABLE_COLS)):
        align = aligns[c] if c < len(aligns) else "left"
        # Slack validates every array entry as an object; emitting ``null`` for
        # default-left columns makes the whole table payload invalid.
        column_settings.append({"align": align})

    block: Block = {
        "type": "table",
        "rows": [[_rich_text_cell(cell) for cell in row] for row in parsed],
    }
    if any(cs is not None for cs in column_settings):
        block["column_settings"] = column_settings
    return block


def _display_width(s: str, wide_ambiguous: bool = False) -> int:
    """Monospace display width of ``s`` in Slack's code font.

    East Asian Wide/Fullwidth chars occupy two columns; combining marks and
    other zero-width formatting codepoints (accent marks, ZWJ, variation
    selectors) occupy none; everything else one. Zeroing combiners keeps a
    decomposed ``e`` + U+0301 from being counted as two columns and drifting
    the table — full grapheme clustering (multi-codepoint ZWJ emoji) is still
    approximate, since the stdlib can't cluster them.

    ``wide_ambiguous`` widens East Asian *Ambiguous* ("A") codepoints —
    arrows (→), check/cross marks (✓ ✗ ×), the ellipsis (…), roman numerals
    (①), Greek/Cyrillic — to two columns. In a CJK font context Slack renders
    these at two columns, so a table that already contains CJK must count them
    as two or every row with an arrow/✓ drifts. Pure-ASCII tables keep them at
    one (the width they actually render at without a CJK font).
    """
    width = 0
    for ch in s:
        if unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Cf"):
            continue
        eaw = unicodedata.east_asian_width(ch)
        if eaw in ("W", "F") or (wide_ambiguous and eaw == "A"):
            width += 2
        else:
            width += 1
    return width


def _pad_display(s: str, width: int, wide_ambiguous: bool = False) -> str:
    """Right-pad ``s`` with spaces to ``width`` display columns."""
    return s + " " * max(0, width - _display_width(s, wide_ambiguous))


def _has_east_asian_wide(rows: List[List[str]]) -> bool:
    """True if any cell holds a Wide/Fullwidth (CJK) codepoint.

    Used to decide whether East Asian *Ambiguous* glyphs should be measured
    at two columns: a table with CJK is rendered in a CJK font, where they are
    two wide; an all-Latin table is not.
    """
    return any(
        unicodedata.east_asian_width(ch) in ("W", "F")
        for r in rows
        for cell in r
        for ch in cell
    )


def _render_table(rows: List[str]) -> str:
    """Render markdown pipe-table rows as aligned monospace text (fallback).

    Column widths are computed in display columns, not codepoints, so CJK
    cell content (two columns per glyph in the monospace font) stays aligned.
    When the table contains any CJK, East Asian *Ambiguous* glyphs (arrows,
    ✓/✗, …) are also counted as two columns so those rows don't drift.
    """
    parsed: List[List[str]] = []
    for r in rows:
        cells = _split_row(r)
        parsed.append(cells)
    if not parsed:
        return "\n".join(rows)
    ncols = max(len(r) for r in parsed)
    for r in parsed:
        r.extend([""] * (ncols - len(r)))
    wide_amb = _has_east_asian_wide(parsed)
    widths = [
        max(_display_width(r[c], wide_amb) for r in parsed) for c in range(ncols)
    ]
    out_lines = []
    for ri, r in enumerate(parsed):
        line = " | ".join(
            _pad_display(r[c], widths[c], wide_amb) for c in range(ncols)
        )
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
        mrkdwn_fn: Accepted for backwards compatibility; unused. Paragraphs
            are rendered as ``rich_text`` (explicit style objects) instead of
            mrkdwn sections, so emphasis adjacent to CJK punctuation — which
            mrkdwn's word-boundary parser refuses to style — renders reliably.

    Returns:
        A list of Block Kit block dicts, or ``None`` when the content is empty,
        exceeds Slack's structural limits, or hits an unexpected shape — the
        caller then falls back to the flat ``text`` payload.  Never raises.
    """
    if not markdown or not markdown.strip():
        return None

    del mrkdwn_fn  # kept in the signature for callers; no longer used

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
            # rich_text (not mrkdwn section): explicit style objects survive
            # CJK-adjacent emphasis that mrkdwn's word-boundary parser drops.
            # Parse inline markup ONCE, then split the resulting element list
            # on the 3000-char text-object limit — splitting the raw string
            # (old behaviour) could bisect a ``**bold**`` span and strand
            # literal asterisks at the seam. Splitting parsed elements never
            # can: each half keeps its style object.
            for section_elems in _pack_elements(_inline_elements(text), MAX_SECTION_TEXT):
                blocks.append(
                    {
                        "type": "rich_text",
                        "elements": [
                            {"type": "rich_text_section", "elements": section_elems}
                        ],
                    }
                )

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
            if _is_list_line(line):
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
                    elif not lines[i].strip() and items:
                        # Blank line inside a list run. LLM-authored ordered
                        # lists commonly separate items with a blank line; if
                        # the next non-blank line is another list item, treat
                        # the blank(s) as a soft separator and keep the run
                        # going so the items stay in one rich_text_list (Slack
                        # numbers each list independently, so splitting would
                        # restart every item at "1."). Otherwise the blank
                        # ends the list.
                        j = i + 1
                        while j < n and not lines[j].strip():
                            j += 1
                        if j < n and _is_list_line(lines[j]):
                            i = j
                        else:
                            break
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
        # NOTE: the list may exceed MAX_BLOCKS. Callers that put blocks on a
        # single message must run the list through segment_blocks() — bailing
        # out here used to flatten every long structured answer (raw pipe
        # tables, unstyled headers) back to plain mrkdwn.
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
            cut = remaining.rfind(" ", 0, limit)  # avoid cutting mid-word
        if cut <= 0:
            cut = limit
        out.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        out.append(remaining)
    return out


def _element_len(el: Dict[str, Any]) -> int:
    """Approximate character cost of a rich_text child element."""
    t = el.get("type")
    if t == "text":
        return len(el.get("text", ""))
    if t == "link":
        return len(el.get("text") or el.get("url", ""))
    return 1  # user/channel/broadcast/emoji render short — count nominally


def _pack_elements(
    elements: List[Dict[str, Any]], limit: int
) -> List[List[Dict[str, Any]]]:
    """Pack parsed inline elements into <= ``limit``-char rich_text sections.

    Splits *between* elements, and only ever splits *inside* a plain ``text``
    element (preserving its style on both halves) — never inside a link or a
    styled span, so no ``**``/`` ` `` markup can leak at a seam. One section is
    always returned so the caller can build at least one block.
    """
    out: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            out.append(cur)
            cur = []
            cur_len = 0

    for el in elements:
        elen = _element_len(el)
        # A single text run longer than the limit: split its text, keeping
        # style, into standalone sections.
        if el.get("type") == "text" and elen > limit:
            flush()
            for piece in _split_text(el.get("text", ""), limit):
                seg = dict(el)
                seg["text"] = piece
                out.append([seg])
            continue
        if cur and cur_len + elen > limit:
            flush()
        cur.append(el)
        cur_len += elen
    flush()
    return out or [[{"type": "text", "text": " "}]]


# ----------------------------------------------------------------------------
# Multi-message segmentation — Slack caps a message at MAX_BLOCKS blocks
# ----------------------------------------------------------------------------


def segment_blocks(
    blocks: List[Block], max_blocks: int = MAX_BLOCKS
) -> List[List[Block]]:
    """Split a block list into message-sized segments of <= ``max_blocks``.

    Prefers cutting before a ``header`` or ``divider`` (boundaries a reader
    already perceives as section breaks) within the trailing third of the
    window; hard-cuts at the cap when no such boundary exists.  A divider
    that would open the next segment is dropped — the message break itself
    is the visual separator.
    """
    if len(blocks) <= max_blocks:
        return [blocks]
    segments: List[List[Block]] = []
    rest = blocks
    while len(rest) > max_blocks:
        cut = max_blocks
        for i in range(max_blocks, max(1, max_blocks * 2 // 3) - 1, -1):
            if rest[i].get("type") in ("header", "divider"):
                cut = i
                break
        segments.append(rest[:cut])
        rest = rest[cut:]
        if rest and rest[0].get("type") == "divider":
            rest = rest[1:]
    if rest:
        segments.append(rest)
    return segments


def _rich_text_plain(block: Block) -> str:
    """Plain-text projection of a rich_text block (for the ``text`` fallback)."""
    parts: List[str] = []

    def walk(elements: List[Dict[str, Any]]) -> None:
        for el in elements:
            t = el.get("type")
            if t == "text":
                parts.append(el.get("text", ""))
            elif t == "link":
                parts.append(el.get("text") or el.get("url", ""))
            elif t == "user":
                parts.append(f"<@{el.get('user_id', '')}>")
            elif t == "channel":
                parts.append(f"<#{el.get('channel_id', '')}>")
            elif t == "broadcast":
                parts.append(f"@{el.get('range', '')}")
            elif t == "emoji":
                parts.append(f":{el.get('name', '')}:")
            elif isinstance(el.get("elements"), list):
                walk(el["elements"])
                if t in ("rich_text_section", "rich_text_preformatted", "rich_text_quote"):
                    parts.append("\n")

    walk(block.get("elements", []))
    return "".join(parts).strip()


_FENCE_COLLAPSE_RE = re.compile(r"```[^\n]*\n?([\s\S]*?)```")


def progress_context_blocks(content: str, max_lines: int = 10) -> Optional[List[Block]]:
    """Render a tool-progress transcript as a small-print ``context`` block.

    Long agentic runs accumulate dozens of tool-activity lines (terminal
    fences, "Reading file…" rows) that visually drown the actual answer.
    This collapses fenced commands to single-line inline code, keeps only
    the trailing ``max_lines`` lines (0 = keep all; the full transcript
    stays in the message's ``text`` fallback), and renders the result in
    Slack's muted context style.
    """
    if not content or not content.strip():
        return None

    def _collapse(m: "re.Match[str]") -> str:
        body = m.group(1).strip("\n")
        first = body.splitlines()[0].strip() if body else ""
        more = " …" if "\n" in body else ""
        return f"`{first}{more}`" if first else "`…`"

    text = _FENCE_COLLAPSE_RE.sub(_collapse, content)
    lines = [ln for ln in (raw.rstrip() for raw in text.split("\n")) if ln.strip()]
    if not lines:
        return None
    if max_lines > 0 and len(lines) > max_lines:
        elided = len(lines) - max_lines
        lines = lines[-max_lines:]
        lines.insert(0, f"_… {elided} earlier steps_")
    joined = "\n".join(lines)
    # mrkdwn control-character escaping (backticks already delimit code).
    joined = (
        joined.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    if len(joined) > MAX_SECTION_TEXT:
        joined = "…" + joined[-(MAX_SECTION_TEXT - 1):]
    return [
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": joined}],
        }
    ]


def blocks_fallback_text(segment: List[Block], limit: int = 39000) -> str:
    """Derive the notification/accessibility ``text`` fallback for a segment.

    Slack requires a non-empty ``text`` alongside ``blocks``; mobile
    notifications read ONLY this field, so it should carry the segment's
    actual content, not a placeholder.
    """
    parts: List[str] = []
    for b in segment:
        t = b.get("type")
        if t == "header":
            parts.append(b.get("text", {}).get("text", ""))
        elif t == "section":
            parts.append(b.get("text", {}).get("text", ""))
        elif t == "rich_text":
            parts.append(_rich_text_plain(b))
        elif t == "table":
            rows = b.get("rows", [])
            ncols = len(rows[0]) if rows else 0
            parts.append(f"[table {len(rows)}x{ncols}]")
    out = "\n".join(p for p in parts if p).strip()
    if len(out) > limit:
        out = out[: limit - 1] + "…"
    return out or " "


# ----------------------------------------------------------------------------
# Table-block demotion — native ``table`` block → aligned monospace
# ----------------------------------------------------------------------------


def _cell_plain(cell: Block) -> str:
    """Plain-text of a native table cell (a ``rich_text`` block), single line."""
    return " ".join(_rich_text_plain(cell).split())


def _table_block_to_preformatted(table: Block) -> Block:
    """Render a native ``table`` block as an aligned-monospace preformatted block.

    Used when a workspace/app tier rejects the native ``table`` block: instead
    of dropping the grid to raw text, re-render it as the same CJK-aligned
    monospace table :func:`_render_table` produces, so the columns still line
    up. Pure block→block transform — needs no source markdown.
    """
    grid = [[_cell_plain(c) for c in row] for row in table.get("rows", [])]
    if not grid:
        return _preformatted_block("")
    ncols = max((len(r) for r in grid), default=0)
    for r in grid:
        r.extend([""] * (ncols - len(r)))
    wide = _has_east_asian_wide(grid)
    widths = [max(_display_width(r[c], wide) for r in grid) for c in range(ncols)]
    lines: List[str] = []
    for ri, r in enumerate(grid):
        line = " | ".join(_pad_display(r[c], widths[c], wide) for c in range(ncols))
        lines.append(line.rstrip())
        if ri == 0:
            lines.append("-+-".join("-" * widths[c] for c in range(ncols)))
    return _preformatted_block("\n".join(lines))


def demote_tables(blocks: List[Block]) -> List[Block]:
    """Replace every native ``table`` block with a monospace preformatted one.

    The retry path when a ``blocks`` payload is rejected: keeps all the other
    rich structure (headers, lists, styled text) and only demotes the block
    type workspaces most often refuse, so a rejected message degrades to an
    *aligned* table rather than losing its blocks entirely (raw pipes).
    """
    return [
        _table_block_to_preformatted(b) if b.get("type") == "table" else b
        for b in blocks
    ]


def has_table_block(blocks: List[Block]) -> bool:
    """True if any block in the list is a native ``table`` block."""
    return any(b.get("type") == "table" for b in blocks)


# ----------------------------------------------------------------------------
# Send-boundary sanitizer — a single defensive pass over ANY outbound blocks
# ----------------------------------------------------------------------------

_MIN_TEXT = " "  # Slack rejects a zero-length text object/element.


def _sanitize_node(node: Any) -> Any:
    """Recursively repair the structural mistakes Slack rejects outright.

    Slack validates the WHOLE ``blocks`` array; a single bad node fails the
    entire message (``invalid_blocks``) and the rich render is lost. This
    fixes, anywhere in the tree, regardless of which builder produced it:

    * empty ``text`` strings on text / plain_text / mrkdwn objects → a space
      ("must be more than 0 characters");
    * ``null`` entries in a table's ``column_settings`` → ``{}`` (an object)
      ("must provide an object [.../column_settings/N]").

    Pure and idempotent — returns new structures, never mutates the input.
    """
    if isinstance(node, dict):
        out = {k: _sanitize_node(v) for k, v in node.items()}
        if out.get("type") in ("text", "plain_text", "mrkdwn"):
            if not (isinstance(out.get("text"), str) and out["text"]):
                out["text"] = _MIN_TEXT
        cs = out.get("column_settings")
        if isinstance(cs, list):
            out["column_settings"] = [
                c if isinstance(c, dict) else {} for c in cs
            ]
        return out
    if isinstance(node, list):
        return [_sanitize_node(x) for x in node]
    return node


def sanitize_blocks(blocks: List[Block]) -> List[Block]:
    """Send-boundary guard: repair invalid_blocks-triggering mistakes in a
    blocks payload from ANY source (render_blocks, exec-approval, slash-confirm,
    or a future builder). Never raises — returns the input unchanged on an
    unexpected shape, so it can wrap every outbound send safely."""
    if not isinstance(blocks, list):
        return blocks
    try:
        return [_sanitize_node(b) for b in blocks]
    except Exception:  # pragma: no cover - must never break a send
        return blocks
