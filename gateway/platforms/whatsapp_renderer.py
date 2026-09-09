"""CommonMark -> WhatsApp native text formatting renderer.

Uses ``markdown-it-py`` (CommonMark 0.30 + GFM strikethrough/tables) to parse
the model's markdown, then walks the token stream and emits WhatsApp's
client-side formatting syntax — every delimiter goes on *both sides* of the
wrapped span, exactly as WhatsApp expects:

    *bold*   _italic_   ~strikethrough~   `inline code`   ``` fenced code ```

Constructs WhatsApp cannot render natively are **down-rendered** (converted,
never discarded) so no information is lost and the result stays readable:

    heading          -> *bold text*
    fenced code      -> language caption line above a ``` code block ```
                       (the language tag is preserved, not dropped)
    table (GFM)      -> a ``` monospace ``` block with padded, aligned pipe
                       columns (all cell data retained)
    link             -> "text (url)"     (label and target both kept)
    image            -> "alt (url)"      (alt text and src both kept)
    blockquote       -> "> line" per line (WhatsApp native quote)
    bullet / ordered -> "- item" / "N. item"  (WhatsApp native; nested
                       lists are indented to keep structure)
    thematic break   -> a --- separator line
    html block/inline-> tags stripped, inner text kept (no markup leaked,
                       no visible content lost)
    soft/hard break  -> newline

Prose containing literal ``* _ ~ ```` is passed through unchanged (the parser
has already resolved real emphasis, so accidental flanking is not produced).
"""

from __future__ import annotations

import re
from typing import List

from markdown_it import MarkdownIt
from markdown_it.token import Token

from gateway.platforms.whatsapp_unicode import GlyphStyler, HEADING_STYLE

# WhatsApp renders a line of U+2500/em-dash chars as a clean horizontal rule.
_HR = "────────────"

# A fenced code info string may be ``python`` or ``python {.cl}`` etc. Keep the
# plain language token (the visible info) so it can be shown as a caption.
_LANG_RE = re.compile(r"^[A-Za-z0-9_+.-]+")


def _clean_lang(info: str) -> str:
    m = _LANG_RE.match(info.strip())
    return m.group(0) if m else ""


# A lone HTML *element* tag (``<b>``, ``</b>``, ``<br/>``, ``<img …>``) carries
# no visible text — drop it. Anything else HTML-ish (``<!everyone>``,
# ``<!-- … -->``, ``<!DOCTYPE …>``) is prose the model wrote literally — keep
# it so no information is lost.
_ELEMENT_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>", re.DOTALL)


def _html_to_text(raw: str) -> str:
    """Strip element tags from raw HTML, keeping the visible inner text.

    A tag-only fragment (e.g. ``<b></b>``) contributes nothing and is
    dropped; a declaration/comment (e.g. ``<!everyone>``) that survives
    stripping unchanged is kept verbatim.
    """
    stripped = re.sub(_ELEMENT_TAG_RE, "", raw).strip()
    if stripped:
        return stripped
    # After stripping there is nothing left but tags — no visible text.
    if _ELEMENT_TAG_RE.fullmatch(raw.strip()):
        return ""
    return raw.strip()


# WhatsApp has NO escape character: an ASCII `~`/`*`/`_`/backtick that
# reaches the payload verbatim is natively re-read by the phone (``~x~
# → strike, *x* → bold, _x_ → italic, `x` → monospace). WhatsApp treats a
# marker as a *valid pair* only when it satisfies the flanking rule: an open is
# preceded by a space/start-of-line AND followed by a non-space; a close is
# preceded by a non-space AND followed by a space/end. The commonmark
# parser resolves the *real* constructs into dedicated tokens
# (em/strong/strikethrough/code), so any delimiter that survives into a
# literal `text` token was NOT a valid construct in the source — but it can
# still pair up and be re-interpreted by WhatsApp (e.g. a single `~` meaning
# "approximately": ``cost ~2.00USD~ total`` → the first `~` is
# space-preceded + digit-followed = open, the second is digit-preceded +
# space-followed = close, so WhatsApp strikes "2.00USD").
#
# We do NOT substitute characters (no fullwidth `～`): we break the native
# pairing by inserting a space right after the offending OPEN delimiter,
# making it `d ` (space-followed) so it can no longer open a pair. The
# remaining token text — including the original `~` char — is preserved
# verbatim, so this is lossless. ``~5USD ~4USD`` needs no change: the 2nd
# `~` is itself space-preceded (an open, not a close), so no valid pair
# exists. Only literal `text` tokens are visited (by construction they hold
# only stray delimiters); intentional markers emitted by em/strong/s tokens
# are never touched.
_PAIR_CHARS = frozenset("*_~`")
_CLOSE_FOLLOW = frozenset(" ,.;:!?)\"'\n\t]}")


def _flank_stray_delims(text: str) -> str:
    """Break WhatsApp-native open→close pairs formed by stray delimiters.

    WhatsApp matches a delimiter pair ONLY within a single line (a
    delimiter at the end of one line can never close one opened on a
    different line). So the text is flanked line-by-line; each line is
    scanned independently so a `~` after a newline is not treated as the
    close for an `~` opened above it. Within a line, WhatsApp pairs a
    delimiter only when an *open* (space/start before, non-space after) is
    followed by a *close* (non-space before, space/end/punct after) of the
    same character. For every such link we insert one space after the
    open, making it space-closed so it can never open a pair again. Loop
    until no pair remains per line.
    """
    if not any(c in text for c in _PAIR_CHARS):
        return text
    return "\n".join(_flank_line(line) for line in text.split("\n"))


def _flank_line(s: str) -> str:
    """Flank stray delimiters within a single line (no newline boundary)."""
    while True:
        n = len(s)

        def _is_open(i: int, ch: str) -> bool:
            return (i == 0 or s[i - 1].isspace()) and (
                i + 1 < n and not s[i + 1].isspace() and s[i + 1] != ch
            )

        def _is_close(i: int) -> bool:
            if i == 0 or s[i - 1].isspace() or s[i - 1] == s[i]:
                return False
            nxt = s[i + 1] if i + 1 < n else None
            return nxt is None or nxt.isspace() or nxt in _CLOSE_FOLLOW

        opened_at = -1  # index of an open delimiter awaiting a close
        break_after = -1
        i = 0
        while i < n and break_after < 0:
            ch = s[i]
            if ch in _PAIR_CHARS:
                if opened_at < 0 and _is_open(i, ch):
                    opened_at = i
                elif opened_at >= 0 and s[opened_at] == ch and _is_close(i):
                    break_after = opened_at
            i += 1
        if break_after < 0:
            return s
        s = s[: break_after + 1] + " " + s[break_after + 1 :]


class CommonMarkToWhatsApp:
    """Render CommonMark source into WhatsApp-native formatted text."""

    def __init__(
        self,
        source: str,
        unicode_formatting: bool = True,
        table_mode: str = "flatten",
    ):
        self.source = source
        # When enabled (default), rich constructs native WhatsApp cannot
        # express (e.g. heading *levels*) are styled with Unicode fonts
        # instead of being flattened — see whatsapp_unicode module docstring.
        self.unicode = unicode_formatting
        # WhatsApp always soft-wraps long lines, so padded/aligned table
        # grids collapse the instant a row exceeds the message width
        # ("flatten"=default: alignment-free key/value lines, wrap-safe;
        # "monospace"=opt-in: the old aligned pipe fence).
        self.table_mode = table_mode if table_mode in ("flatten", "monospace") else "flatten"
        self._md = (
            MarkdownIt("commonmark", {"html": True})
            .enable("strikethrough")
            .enable("table")
        )

    # -- public ------------------------------------------------------------
    def render(self) -> str:
        tokens = self._md.parse(self.source)
        body = self._blocks(tokens, 0, len(tokens), quote=0, indent="", sep="\n\n")
        # Normalize whitespace: never more than one blank line; drop edge
        # newlines (structural) but PRESERVE spaces/tabs — a leading space
        # surviving _sanitize_outbound_text (e.g. invisible-unicode → space)
        # is meaningful content (" text", not "text").
        return re.sub(r"\n{3,}", "\n\n", body).strip("\r\n")

    # -- block-level -------------------------------------------------------
    def _blocks(
        self,
        tokens: List[Token],
        start: int,
        end: int,
        quote: int,
        indent: str,
        sep: str,
    ) -> str:
        parts: List[str] = []
        i = start
        while i < end:
            t = tokens[i]
            ty = t.type
            if ty == "blockquote_open":
                close = self._find(tokens, i, "blockquote_close")
                inner = self._blocks(
                    tokens, i + 1, close, quote=quote + 1, indent=indent, sep="\n\n"
                )
                marker = ">" * (quote + 1)
                lines = inner.split("\n")
                parts.append(
                    "\n".join(
                        f"{marker} {ln}" if ln.strip() else marker
                        for ln in lines
                    )
                )
                i = close + 1
            elif ty in ("bullet_list_open", "ordered_list_open"):
                i = self._list(tokens, i, end, quote, indent, parts)
            elif ty in ("fence", "code_block"):
                parts.append(self._code(t, indent))
                i += 1
            elif ty == "heading_open":
                inline = tokens[i + 1]
                if self.unicode:
                    # Reclaim heading *hierarchy* that native WhatsApp would
                    # flatten to one bold style: style by level with distinct
                    # Unicode fonts. Content is rendered plain first so no
                    # stray `*`/`_` markers leak into the styled heading.
                    level = int(t.tag[1:]) if t.tag[:1] == "h" and t.tag[1:].isdigit() else 1
                    text = self._inline(self._children(inline), plain=True).rstrip()
                    text = GlyphStyler.stylize(text, HEADING_STYLE.get(level, "bold"))
                    parts.append(f"{indent}{text}")
                else:
                    text = self._inline(self._children(inline)).rstrip()
                    # If the heading is already a single bold span rendered from
                    # ``**…**``/``__…__``, don't double-wrap into ``**…**`` (which
                    # WhatsApp renders as literal doubled stars).
                    if len(text) >= 2 and text.startswith("*") and text.endswith("*"):
                        text = text[1:-1].strip()
                    parts.append(f"{indent}*{text}*")
                i += 3
            elif ty == "paragraph_open":
                inline = tokens[i + 1]
                text = self._inline(self._children(inline)).rstrip()
                if text:
                    parts.append(f"{indent}{text}")
                i += 3
            elif ty in ("hr", "thematic_break"):
                parts.append(f"{indent}{_HR}")
                i += 1
            elif ty == "table_open":
                i = self._table(tokens, i, end, indent, parts)
            elif ty == "html_block":
                text = _html_to_text(t.content).strip()
                if text:
                    parts.append(f"{indent}{text}")
                i += 1
            else:
                i += 1
        return sep.join(p for p in parts if p)

    def _list(
        self,
        tokens: List[Token],
        i: int,
        end: int,
        quote: int,
        indent: str,
        parts: List[str],
    ) -> int:
        ordered = tokens[i].type == "ordered_list_open"
        close_type = "ordered_list_close" if ordered else "bullet_list_close"
        item_texts: List[str] = []
        i += 1
        while i < end and tokens[i].type != close_type:
            if tokens[i].type == "list_item_open":
                num = tokens[i].info
                marker = f"{num}. " if ordered and num.isdigit() else "- "
                li_close = self._find(tokens, i, "list_item_close")
                inner = self._blocks(
                    tokens, i + 1, li_close, quote=quote, indent=indent, sep="\n"
                )
                lines = inner.split("\n")
                buf = marker + lines[0] if lines and lines[0] else marker
                cont = " " * len(marker)
                for ln in lines[1:]:
                    buf += ("\n" + cont + ln) if ln.strip() else "\n"
                item_texts.append(buf)
                i = li_close + 1
            else:
                i += 1
        if item_texts:
            parts.append(f"{indent}" + "\n".join(item_texts))
        return i

    def _code(self, t: Token, indent: str) -> str:
        body = t.content.rstrip("\n")
        caption = ""
        if t.type == "fence":
            lang = _clean_lang(t.info)
            if lang:
                caption = f"{indent}*{lang}*\n"
        return f"{caption}{indent}```\n{body}\n{indent}```"

    def _table(
        self,
        tokens: List[Token],
        i: int,
        end: int,
        indent: str,
        parts: List[str],
    ) -> int:
        rows: List[List[str]] = []
        r = i + 1
        while r < end and tokens[r].type != "table_close":
            if tokens[r].type == "tr_open":
                r += 1
                cells: List[str] = []
                while r < end and tokens[r].type != "tr_close":
                    if tokens[r].type in ("th_open", "td_open"):
                        inline = tokens[r + 1]
                        cells.append(self._inline(self._children(inline)).strip())
                        r += 3  # *_open, inline, *_close
                    else:
                        r += 1
                rows.append(cells)
            else:
                r += 1
        if not rows:
            return r

        if self.table_mode == "monospace":
            self._table_monospace(rows, indent, parts)
        else:
            self._table_flatten(rows, indent, parts)
        return r

    def _table_monospace(
        self, rows: List[List[str]], indent: str, parts: List[str]
    ) -> None:
        """Opt-in aligned pipe fence (legacy). Wraps on WhatsApp — alignment
        only survives on devices/widths where no row exceeds the bubble."""
        ncols = max(len(row) for row in rows) or 1
        widths = [0] * ncols
        for row in rows:
            cells = list(row) + [""] * (ncols - len(row))
            for c, cell in enumerate(cells):
                if len(cell) > widths[c]:
                    widths[c] = len(cell)

        lines = []
        for row in rows:
            cells = [cell or " " for cell in row] + [" "] * (ncols - len(row))
            lines.append(
                "| " + " | ".join(cell.ljust(widths[c]) for c, cell in enumerate(cells)) + " |"
            )
        header = ["-" * widths[c] for c in range(ncols)]
        lines.insert(1, "| " + " | ".join(header) + " |")
        parts.append(f"{indent}```\n" + "\n".join(lines) + "\n```")

    def _table_flatten(
        self, rows: List[List[str]], indent: str, parts: List[str]
    ) -> None:
        """Alignment-free key/value flatten — what WhatsApp should actually
        render, because it always soft-wraps long lines (a padded grid
        collapses once a row exceeds the bubble width).

        Interpretation: the first row is the header; every data row becomes
        one ``*Header*: value`` per column on its own line, and rows are
        separated by a blank line. Each line is self-contained, so wrapping
        never destroys meaning.  A single-column table (header only, no
        data) emits nothing.
        """
        header = rows[0]
        ncols = len(header) or max(len(row) for row in rows) or 1
        blocks: List[str] = []
        for row in rows[1:]:
            if not any(c.strip() for c in row):
                continue
            lines: List[str] = []
            for c in range(ncols):
                label = (header[c] if c < len(header) else "").strip() or f"Col {c + 1}"
                value = (row[c] if c < len(row) else "").strip() or "—"
                lines.append(self._bold(label) + ": " + value)
            blocks.append("\n".join(lines))
        if blocks:
            body = "\n\n".join(blocks)
            if indent:
                lines = body.split("\n")
                parts.append("\n".join(indent + ln if ln.strip() else ln for ln in lines))
            else:
                parts.append(body)

    def _bold(self, text: str) -> str:
        """Wrap ``text`` in WhatsApp bold, guarding already-bold cells."""
        text = text.strip()
        if not text:
            return text
        if text.startswith("*") and text.endswith("*"):
            return text
        return f"*{text}*"

    # -- inline-level ------------------------------------------------------
    def _inline(self, children: List[Token], plain: bool = False) -> str:
        parts: List[str] = []
        i = 0
        n = len(children)
        while i < n:
            t = children[i]
            ty = t.type
            if ty == "text":
                # Literal text: break any WhatsApp-native open→close pair
                # formed by stray `*`/`_`/`~`/backtick delimiters (strays by
                # construction here — real constructs arrive as em/strong/s/
                # code tokens and keep their markers). See _flank_stray_delims.
                parts.append(_flank_stray_delims(t.content))
            elif ty == "strong_open":
                parts.append("" if plain else "*")
            elif ty == "strong_close":
                parts.append("" if plain else "*")
            elif ty == "em_open":
                parts.append("" if plain else "_")
            elif ty == "em_close":
                parts.append("" if plain else "_")
            elif ty == "s_open":
                parts.append("" if plain else "~")
            elif ty == "s_close":
                parts.append("" if plain else "~")
            elif ty == "code_inline":
                parts.append(f"`{t.content}`")
            elif ty in ("softbreak", "hardbreak"):
                parts.append("\n")
            elif ty == "html_inline":
                text = _html_to_text(t.content)
                if text:
                    parts.append(text)
            elif ty == "link_open":
                href = t.attrGet("href") or ""
                close = self._find(children, i, "link_close")
                label = self._inline(children[i + 1 : close]).rstrip()
                parts.append(f"{label} ({href})" if href else label)
                i = close
            elif ty == "image":
                alt = (t.attrGet("alt") or t.content or "").strip()
                src = t.attrGet("src") or ""
                if alt and src:
                    parts.append(f"{alt} ({src})")
                elif src:
                    parts.append(src)
                elif alt:
                    parts.append(alt)
            i += 1
        return "".join(parts)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _children(inline: Token) -> List[Token]:
        return inline.children or []

    @staticmethod
    def _find(tokens: List[Token], start: int, close_type: str) -> int:
        depth = 0
        for i in range(start, len(tokens)):
            ty = tokens[i].type
            if ty.endswith("_open"):
                depth += 1
            elif ty.endswith("_close"):
                depth -= 1
            if depth == 0 and ty == close_type:
                return i
        return len(tokens) - 1