"""Feishu markdown → native post element preprocessor.

Added 2026-08-12 (Phase 3 / L3). Implements the dual-layer rendering
strategy documented in skill `feishu-markdown-preprocess`:

  - Path A (default, prefer this): emit ``tag: md`` rows; let Feishu's
    own parser render markdown. Most cases route here.
  - Path B (only when mixing native elements): convert markdown to
    Feishu native post elements (``tag: text`` + ``style: [...]``,
    ``tag: code_block``, ``tag: a``, etc.). Triggered by callers that
    want to embed native elements alongside markdown.

CRITICAL: do NOT emit HTML strings inside ``tag: text`` elements — Feishu's
parser passes them through as literal characters. Use ``style: [...]``
arrays (Path B) or ``tag: md`` (Path A) instead.

Module surface
--------------
- ``is_complex_markdown(content) -> bool``  — should we attempt native?
- ``preprocess_to_post_payload(content) -> dict | None`` — native rows, or
  ``None`` to signal "fall back to Path A" (Path A is the correct default
  in most cases).

Pitfalls are documented in the inline comments where they apply. See
``references/feishu-post-element-truth.md`` for the full reference.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pattern catalogue
# ---------------------------------------------------------------------------

# Characters that Feishu's post parser considers "native" and would render
# literally if escaped. We strip leading backslashes before them so the
# markdown source reads correctly to Feishu's parser when it later sees
# the same string in tag: md mode.  re.escape() is REQUIRED here for
# Python 3.12+ strict escape validation (Pitfall #1 in skill SKILL.md).
_FLIGHT_NATIVE_CHARS = r"*`_{}[]()#+-|!~"
_OVER_ESCAPED_PATTERN = r"\\" + "[" + re.escape(_FLIGHT_NATIVE_CHARS) + "]"
_OVER_ESCAPED_RE = re.compile(_OVER_ESCAPED_PATTERN)

# Inline emphasis detection: **bold**, *italic*, ~~strike~~, `code`.
INLINE_EMPHASIS_RE = re.compile(
    r"\*\*"                                              # **bold** opener/closer
    r"|(^|\s|\W)\*(\S[^*]*\S)\*(\s|$|\W)"               # *italic* (non-greedy)
    r"|~~"                                                # ~~strike~~
    r"|`[^`]+`"                                          # `code`
)

# Block-level "complex" markers that justify native element rendering.
# Tables are NOT here — post+md handles GFM tables since #52786.
_COMPLEX_PATTERN = (
    r"```"                                                # fenced code block
    r"|(^|\n)\s*>\s"                                       # blockquote
    r"|(^|\n)\s*-{3,}\s*$"                                 # horizontal rule
    r"|\!\["                                               # image
    r"|(^|\n)#{1,6}\s"                                     # ATX heading
)
COMPLEX_ELEMENT_RE = re.compile(_COMPLEX_PATTERN, re.MULTILINE)

# List item detection. Used by ``_is_list_item_start``.
_BULLET_ITEM_RE = re.compile(r"^\s*[-*+]\s+")
_ORDERED_ITEM_RE = re.compile(r"^\s*\d+\.\s+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_complex_markdown(content: str) -> bool:
    """Return True when ``content`` contains block-level structures that
    benefit from native element rendering.

    Triggers on fenced code, blockquote, hr, image, ATX heading.
    Inline-only emphasis (bold / italic / strike / code) does NOT trigger
    — Path A's ``tag: md`` handles those correctly.
    """
    if not content:
        return False
    return bool(COMPLEX_ELEMENT_RE.search(content))


def preprocess_to_post_payload(content: str) -> Optional[Dict[str, Any]]:
    """Convert ``content`` to a Feishu post payload (native elements).

    Returns ``None`` when conversion fails or when content is too simple
    to justify the native path — callers should fall back to Path A's
    ``tag: md`` rows.
    """
    if not content:
        return None
    try:
        rows = _content_to_rows(content)
    except Exception:
        return None
    if not rows:
        return None
    return {
        "zh_cn": {
            "title": "",
            "content": rows,
        }
    }


# ---------------------------------------------------------------------------
# Line-level walker
# ---------------------------------------------------------------------------

def _content_to_rows(content: str) -> List[List[Dict[str, Any]]]:
    """Walk ``content`` line-by-line and produce Feishu post rows.

    Block-level parsing (paragraphs, headings, lists, code blocks, blockquote,
    hr) happens here. Inline emphasis (bold/italic/strike/code) is handled
    by ``_render_inline`` for each row's first text element.
    """
    rows: List[List[Dict[str, Any]]] = []
    lines = content.splitlines()
    i = 0
    in_code = False
    code_lang = ""
    code_buf: List[str] = []

    def flush_code() -> None:
        nonlocal code_buf
        if not code_buf and not code_lang:
            return
        body = "\n".join(code_buf)
        # Native code_block element gives language-aware highlighting.
        rows.append([{
            "tag": "code_block",
            "language": code_lang.upper() if code_lang else "",
            "text": body,
        }])
        code_buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip blank lines at the top level so the paragraph-collector
        # below can never spin (Pitfall #0 — the absence of a blank-line
        # skip at the outer loop is the canonical "paragraph that doesn't
        # advance i" hang).
        if not stripped:
            i += 1
            continue

        # --- Code fence handling ---
        if stripped.startswith("```"):
            tag = stripped[3:].strip()
            if not in_code:
                # Opening fence: capture language, start buffer.
                in_code = True
                code_lang = tag.split()[0] if tag else ""
                code_buf = []
            else:
                # Closing fence: flush buffer.
                flush_code()
                in_code = False
                code_lang = ""
                code_buf = []
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # --- Block-level patterns ---

        # Heading
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            size = {1: 22, 2: 20, 3: 18, 4: 17, 5: 16, 6: 15}.get(level, 16)
            rows.append([{
                "tag": "text",
                "text": text,
                "style": ["bold"],
                "size": size,
            }])
            i += 1
            continue

        # Horizontal rule
        if _HR_RE.match(line):
            rows.append([{"tag": "text", "text": "────────────"}])
            i += 1
            continue

        # Blockquote — collect contiguous > lines.
        if _BLOCKQUOTE_RE.match(line):
            block_lines: List[str] = []
            while i < len(lines) and _BLOCKQUOTE_RE.match(lines[i]):
                block_lines.append(_BLOCKQUOTE_RE.sub("", lines[i], count=1))
                i += 1
            quote_text = "\n".join(block_lines).strip()
            if quote_text:
                rows.append([{"tag": "text", "text": f"│ {quote_text}"}])
            continue

        # Bullet list — collect contiguous list items, recursively handle
        # nested lists (Pitfall #3 + #4 in skill SKILL.md).
        if _BULLET_ITEM_RE.match(line) or _ORDERED_ITEM_RE.match(line):
            list_block_end = _find_list_block_end(lines, i)
            list_rows = _render_list_block(lines[i:list_block_end])
            rows.extend(list_rows)
            i = list_block_end
            continue

        # --- Paragraph (default): collect until blank line / block pattern ---
        para_lines: List[str] = []
        while i < len(lines):
            cur = lines[i]
            if not cur.strip():
                break
            if (
                _HEADING_RE.match(cur)
                or _HR_RE.match(cur)
                or _BLOCKQUOTE_RE.match(cur)
                or _BULLET_ITEM_RE.match(cur)
                or _ORDERED_ITEM_RE.match(cur)
                or cur.strip().startswith("```")
            ):
                break
            para_lines.append(cur)
            i += 1

        if para_lines:
            text = "\n".join(para_lines).strip()
            # Re-detect blockquote merged into paragraph (Pitfall #5).
            if _BLOCKQUOTE_RE.match(text):
                text = _BLOCKQUOTE_RE.sub("", text, count=1)
                rows.append([{"tag": "text", "text": f"│ {text}"}])
            # Re-detect hr merged into paragraph (Pitfall #6).
            elif text.strip() == "---":
                rows.append([{"tag": "text", "text": "────────────"}])
            else:
                row = _render_inline(text)
                # Filter empty text elements (Pitfall #7).
                row = [e for e in row if e.get("text") != ""]
                if row:
                    rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# List handling
# ---------------------------------------------------------------------------

def _find_list_block_end(lines: List[str], start: int) -> int:
    """Return the index AFTER the last contiguous list-item line.

    A list "block" is contiguous lines that look like list items (possibly
    nested) or continuation-indented lines (4+ spaces or 1+ tab). Blank
    lines followed by more list items are part of the same block (loose
    list in markdown-it parlance).
    """
    i = start
    last_item = start
    saw_blank = False
    while i < len(lines):
        cur = lines[i]
        if not cur.strip():
            saw_blank = True
            i += 1
            continue
        if _BULLET_ITEM_RE.match(cur) or _ORDERED_ITEM_RE.match(cur):
            last_item = i
            saw_blank = False
            i += 1
            continue
        # Continuation line (indented) — still part of the previous item.
        if cur.startswith(("    ", "\t")):
            last_item = i
            i += 1
            continue
        # Non-blank, non-list, non-indented — block ends.
        if saw_blank:
            break
        break
    return last_item + 1


def _render_list_block(lines: List[str]) -> List[List[Dict[str, Any]]]:
    """Render a contiguous list block into rows.

    Walks each top-level item; nested lists are recursed into via
    ``_render_nested_list``. Indent depth is computed from leading whitespace
    following markdown-it convention (Pitfall #3): 2-space indent = +1 level.
    """
    rows: List[List[Dict[str, Any]]] = []
    i = 0
    # Counter persists across items in this block — initializing it
    # inside the while loop resets every iteration and produces "1. 1. 1."
    # for ordered lists (regression fix, Aug 12).
    counter = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        # Determine list type at this level.
        if _ORDERED_ITEM_RE.match(line):
            ordered = True
            prefix_re = _ORDERED_ITEM_RE
            if counter == 0:
                counter = 1
        elif _BULLET_ITEM_RE.match(line):
            ordered = False
            prefix_re = _BULLET_ITEM_RE
            # counter unused for bullets
        else:
            # Continuation or unexpected — skip.
            i += 1
            continue

        # Item body: collect lines until next list-item-start at this or
        # shallower indent, or end of block.
        item_lines: List[str] = [line]
        j = i + 1
        while j < len(lines):
            cur = lines[j]
            if not cur.strip():
                item_lines.append(cur)
                j += 1
                continue
            if _BULLET_ITEM_RE.match(cur) or _ORDERED_ITEM_RE.match(cur):
                break
            item_lines.append(cur)
            j += 1

        # Compute indent level: 2 spaces per level beyond the first.
        leading = len(line) - len(line.lstrip(" "))
        indent_level = max(0, (leading - 0) // 2)

        # Build prefix.
        if ordered:
            prefix = f"{counter}. "
            counter += 1
        else:
            prefix = "• "

        indent = "  " * indent_level

        # Item body (without the leading prefix chars).
        body = prefix_re.sub("", item_lines[0], count=1)
        # Continuation lines that are part of this item's body.
        cont = [
            l.lstrip() for l in item_lines[1:] if l.strip()
        ]
        body_text = body + ("\n" + "\n".join(cont) if cont else "")

        # First row: indent + prefix + inline-rendered body.
        first_row = [{"tag": "text", "text": indent + prefix}] + _render_inline(body_text)
        rows.append([e for e in first_row if e.get("text") != ""])

        # Nested list items (lines that started inside this item's range and
        # match a list pattern at deeper indent).
        nested_start = i + 1
        nested_end = j
        nested_lines = lines[nested_start:nested_end]
        # Filter: keep only lines that are themselves list items at deeper
        # indent OR blank separators within the nested block.
        nested_filtered: List[str] = []
        for nl in nested_lines:
            if not nl.strip():
                nested_filtered.append(nl)
                continue
            nl_leading = len(nl) - len(nl.lstrip(" "))
            if nl_leading > leading and (_BULLET_ITEM_RE.match(nl) or _ORDERED_ITEM_RE.match(nl)):
                nested_filtered.append(nl)
        if nested_filtered:
            rows.extend(_render_list_block(nested_filtered))

        i = j
    return rows


# ---------------------------------------------------------------------------
# Inline rendering
# ---------------------------------------------------------------------------

def _render_inline(text: str) -> List[Dict[str, Any]]:
    """Convert ``text`` into a list of native text elements with style flags.

    Handles **bold**, *italic*, ~~strike~~, and `code`. Other markdown
    (links, images, emoji) is passed through as plain text — Path B is only
    invoked when mixing native elements, and the typical case is Path A
    (``tag: md``), so we don't try to be exhaustive here.

    Algorithm: tokenize first into a flat stream of ``(kind, content)``
    tokens (Pitfall #2 — strong/em/s/code are token PAIRS, not containers),
    then walk the stream emitting one element per token. Style state for
    bold/italic/strike is **sticky** across plain tokens because markdown
    doesn't reset state between sibling inline runs — when you write
    ``**bold** plain *italic*`` the "plain" word inherits no style, but
    the structural parser treats each ``<em>``/``<strong>``/``<s>`` as a
    fresh context. We follow the structural model: each token gets the
    styles active AT ITS START POSITION, not accumulated.
    """
    if not text:
        return [{"tag": "text", "text": ""}]

    # Strip over-escaped backslashes so the literal "*" survives Feishu's parser.
    text = _OVER_ESCAPED_RE.sub(lambda m: m.group(0)[1:], text)

    tokens = _tokenize_inline(text)
    out: List[Dict[str, Any]] = []

    def push(text_val: str, styles: List[str]) -> None:
        if not text_val:
            return
        # Coalesce consecutive plain text with identical styles — keeps
        # the row compact and avoids spurious element fragmentation.
        if out and out[-1].get("tag") == "text" and out[-1].get("style", []) == styles:
            out[-1]["text"] = out[-1].get("text", "") + text_val
            return
        elem: Dict[str, Any] = {"tag": "text", "text": text_val}
        if styles:
            elem["style"] = styles
        out.append(elem)

    # We carry active styles as we walk the token stream. State changes
    # happen at `**`, `~~`, and (matched) `*` pairs; code (`...`) is opaque.
    bold = False
    italic = False
    strike = False
    in_code = False

    for kind, content in tokens:
        if kind == "code":
            # Inline code is opaque — emit the raw backticks-and-content as a
            # single element with the "code" style. The styles array contains
            # "code" plus whatever emphasis was active when the code started.
            push(content, ["code"] + ([ "bold"] if bold else []) + (["italic"] if italic else []) + (["strikethrough"] if strike else []))
            continue
        if kind == "bold_open":
            bold = True
            continue
        if kind == "bold_close":
            bold = False
            continue
        if kind == "strike_open":
            strike = True
            continue
        if kind == "strike_close":
            strike = False
            continue
        if kind == "italic_open":
            italic = True
            continue
        if kind == "italic_close":
            italic = False
            continue
        if kind == "plain":
            styles = []
            if bold:
                styles.append("bold")
            if italic:
                styles.append("italic")
            if strike:
                styles.append("strikethrough")
            push(content, styles)
            continue

    return out


def _tokenize_inline(text: str) -> List[Tuple[str, str]]:
    """Linear tokenizer for inline markdown into (kind, content) tokens.

    Kinds: ``plain``, ``code``, ``bold_open``, ``bold_close``, ``italic_open``,
    ``italic_close``, ``strike_open``, ``strike_close``. Unbalanced emphasis
    markers (e.g. ``*foo`` with no closing ``*``) are emitted as plain text
    rather than raising — Feishu's parser is forgiving and we don't want
    to lose user content on edge cases.
    """
    tokens: List[Tuple[str, str]] = []
    pos = 0
    n = len(text)
    plain_start = 0

    def flush_plain(end: int) -> None:
        nonlocal plain_start
        if end > plain_start:
            tokens.append(("plain", text[plain_start:end]))
        plain_start = end

    while pos < n:
        # Inline code: `...` — opaque until matching backtick.
        if text[pos] == "`":
            end = text.find("`", pos + 1)
            if end == -1:
                pos += 1
                continue
            flush_plain(pos)
            tokens.append(("code", text[pos:end + 1]))
            pos = end + 1
            plain_start = pos
            continue

        # Bold: **...**
        if text[pos:pos + 2] == "**":
            # Check whether this looks like an opener or a closer — both
            # look the same in markdown (Pitfall #2). We emit them as a
            # matching pair by tracking depth.
            flush_plain(pos)
            # Look ahead: if the next non-** chars end with ** within the
            # next 100 chars, it's a pair; otherwise emit a single marker.
            # To keep this simple and correct, we just count consecutive **
            # runs and emit alternating open/close. markdown requires balanced
            # **; an unbalanced run is rendered as literal text.
            close = text.find("**", pos + 2)
            if close == -1 or close == pos + 2:
                # ** followed by ** or end-of-text — degenerate, emit as plain.
                plain_start = pos
                pos += 2
                continue
            tokens.append(("bold_open", ""))
            tokens.append(("plain", text[pos + 2:close]))
            tokens.append(("bold_close", ""))
            pos = close + 2
            plain_start = pos
            continue

        # Strike: ~~...~~
        if text[pos:pos + 2] == "~~":
            close = text.find("~~", pos + 2)
            if close == -1 or close == pos + 2:
                pos += 2
                continue
            flush_plain(pos)
            tokens.append(("strike_open", ""))
            tokens.append(("plain", text[pos + 2:close]))
            tokens.append(("strike_close", ""))
            pos = close + 2
            plain_start = pos
            continue

        # Italic: *...* (single asterisk, NOT preceded by whitespace or **)
        if (
            text[pos] == "*"
            and pos + 1 < n
            and text[pos + 1] not in (" ", "\n", "\t", "*")
        ):
            close = text.find("*", pos + 1)
            if close == -1 or close == pos + 1:
                pos += 1
                continue
            flush_plain(pos)
            tokens.append(("italic_open", ""))
            tokens.append(("plain", text[pos + 1:close]))
            tokens.append(("italic_close", ""))
            pos = close + 1
            plain_start = pos
            continue

        pos += 1

    flush_plain(n)
    return tokens


__all__ = [
    "is_complex_markdown",
    "preprocess_to_post_payload",
    "INLINE_EMPHASIS_RE",
    "COMPLEX_ELEMENT_RE",
    "OVER_ESCAPED_RE",
]