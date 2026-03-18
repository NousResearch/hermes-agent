"""Lightweight streaming markdown renderer for the Hermes CLI.

Renders markdown to simple ANSI escape codes, line-by-line, compatible with
prompt_toolkit's ANSI() parser.  Designed for streaming use where text arrives
incrementally and each completed line is rendered independently.

Colors are pulled from the active skin engine so rendering adapts to whatever
theme the user has selected (default gold, ares crimson, mono, slate, sisyphus,
or custom YAML skins).  Falls back to universal SGR codes if the skin engine
is unavailable.
"""

import re
import shutil
import unicodedata

# ── ANSI SGR codes (structural, theme-neutral) ─────────────────────────
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"
_UNDERLINE = "\033[4m"
_STRIKE = "\033[9m"
_RST = "\033[0m"

# Maximum width for decorative elements (rules, fences) so they don't
# stretch absurdly on ultra-wide terminals.
_MAX_DECOR_WIDTH = 80

# Precompiled patterns for inline formatting
_RE_BOLD_ITALIC = re.compile(r"\*\*\*(.+?)\*\*\*")
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_BOLD_UND = re.compile(r"__(.+?)__")
_RE_ITALIC = re.compile(r"(?<![*\\])\*(?!\*)(.+?)(?<![*\\])\*(?!\*)")
_RE_STRIKE = re.compile(r"~~(.+?)~~")
_RE_CODE = re.compile(r"`([^`]+)`")
_RE_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_RE_ULIST = re.compile(r"^(\s*)([-*+])\s(.*)")
_RE_OLIST = re.compile(r"^(\s*)(\d+[.)])\s(.*)")
_RE_TABLE_SEP = re.compile(r"^\|[\s:]*-+[\s:]*(\|[\s:]*-+[\s:]*)*\|?\s*$")
_RE_ESCAPED = re.compile(r"\\([*_~`\[\]\\])")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_HTML_DANGEROUS = re.compile(r"<\s*(script|style|iframe)\b[^>]*>.*?</\s*\1\s*>", re.IGNORECASE)
_RE_HTML_DANGEROUS_OPEN = re.compile(r"<\s*(script|style|iframe)\b[^>]*>", re.IGNORECASE)

# Null-byte placeholder token for protecting extracted elements
_PH = "\x00"


_RE_ANSI = re.compile(r"\033\[[0-9;]*m")


def _visible_width(text: str) -> int:
    """Display width after stripping ANSI escape sequences."""
    return _display_width(_RE_ANSI.sub("", text))


def _display_width(text: str) -> int:
    """Calculate the display width of text, accounting for wide characters.

    Emoji and East Asian wide/fullwidth characters occupy 2 terminal columns
    but ``len()`` counts them as 1.  This function returns the actual number
    of columns the string will occupy.

    Also treats common symbol ranges (dingbats, miscellaneous symbols,
    geometric shapes) as 2-wide, since most modern terminals render them
    with wide emoji-style glyphs even though Unicode classifies some as
    narrow or ambiguous.
    """
    width = 0
    for ch in text:
        cp = ord(ch)
        cat = unicodedata.category(ch)
        # Zero-width: combining marks, variation selectors, ZWJ
        if cat.startswith("M") or cp in (0x200B, 0x200C, 0x200D, 0xFEFF):
            continue
        if cp in range(0xFE00, 0xFE10):  # variation selectors
            continue
        eaw = unicodedata.east_asian_width(ch)
        if eaw in ("W", "F"):
            width += 2
        elif (
            0x2500 <= cp <= 0x257F      # box drawing — always 1-wide
            or 0x2580 <= cp <= 0x259F   # block elements — always 1-wide
        ):
            width += 1
        elif (
            0x2190 <= cp <= 0x21FF      # arrows
            or 0x2200 <= cp <= 0x22FF   # mathematical operators
            or 0x2300 <= cp <= 0x23FF   # miscellaneous technical
            or 0x25A0 <= cp <= 0x25FF   # geometric shapes
            or 0x2600 <= cp <= 0x26FF   # miscellaneous symbols
            or 0x2700 <= cp <= 0x27BF   # dingbats (✓, ✗, ✂, etc.)
            or 0x2B00 <= cp <= 0x2BFF   # misc symbols and arrows
            or 0x1F000 <= cp <= 0x1FFFF # supplemental symbols, emoji
        ):
            width += 2
        else:
            width += 1
    return width


def _hex_to_ansi_fg(hex_color: str) -> str:
    """Convert ``#RRGGBB`` to a 24-bit foreground ANSI escape."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return ""
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"\033[38;2;{r};{g};{b}m"


def _decor_width() -> int:
    """Width for decorative elements (fences, rules), capped for readability."""
    return min(_MAX_DECOR_WIDTH, max(40, shutil.get_terminal_size((80, 24)).columns - 8))


def _parse_table_cells(line: str) -> list:
    """Split a pipe-delimited table row into stripped cell contents."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


class StreamingMarkdownRenderer:
    """Stateful, line-by-line markdown-to-ANSI renderer.

    Maintains minimal state (code-block tracking, table buffering) across
    lines so that fenced code blocks and tables are rendered correctly
    even when streamed.

    Colors are resolved once at construction from the active skin.

    Usage::

        renderer = StreamingMarkdownRenderer()
        for line in lines:
            rendered = renderer.render_line(line)
            if rendered is not None:
                print(rendered)
        # After all lines, flush any buffered content
        tail = renderer.flush()
        if tail is not None:
            print(tail)
    """

    def __init__(self):
        self._in_code_block = False
        self._code_lang = ""
        self._code_lexer = None     # Pygments lexer for current code block
        self._code_formatter = None # Pygments formatter (reused across lines)
        # Table state — buffer all rows so we can measure before rendering
        self._table_buf = []        # list of raw pipe-delimited lines
        self._table_sep_idx = -1    # index of separator row in buffer
        self._table_bq_depth = 0    # blockquote depth of current table
        self._code_bq_depth = 0     # blockquote depth of current code block
        self._resolve_colors()
        self._init_pygments()

    def _resolve_colors(self) -> None:
        """Pull markdown element colors from the active skin.

        Mapping:
            banner_accent  → headers          (section-header accent color)
            ui_label       → code / code spans (label color, often cyan-ish)
            banner_dim     → fences, rules, blockquote bars (muted chrome)
        """
        try:
            from hermes_cli.skin_engine import get_active_skin
            skin = get_active_skin()
            self._c_header = _BOLD + _hex_to_ansi_fg(
                skin.get_color("banner_accent", "")
            )
            self._c_code = _hex_to_ansi_fg(
                skin.get_color("ui_label", "")
            )
            self._c_dim = _hex_to_ansi_fg(
                skin.get_color("banner_dim", "")
            )
        except Exception:
            self._c_header = ""
            self._c_code = ""
            self._c_dim = ""

        # Fallback to universal SGR if skin returned empty strings
        if not self._c_header:
            self._c_header = _BOLD
        if not self._c_code:
            self._c_code = "\033[36m"  # cyan
        if not self._c_dim:
            self._c_dim = _DIM

    def _init_pygments(self) -> None:
        """Set up Pygments formatter (once). Lexer is set per code block."""
        try:
            from pygments.formatters import TerminalFormatter
            self._code_formatter = TerminalFormatter()
            self._has_pygments = True
        except ImportError:
            self._has_pygments = False

    # ── public API ──────────────────────────────────────────────────────

    def flush(self) -> str | None:
        """Flush any buffered content (e.g. a table awaiting completion).

        Call this after the last line to ensure nothing is left in the buffer.
        Returns ``None`` if the buffer is empty.
        """
        return self._flush_table_buf()

    def render_line(self, line: str) -> str | None:
        """Render a single line of markdown to ANSI-formatted text.

        Returns ``None`` when the line is buffered internally (e.g. a table
        header waiting for its separator row).  The caller should skip
        printing when ``None`` is returned.
        """

        stripped = line.strip()

        # ── Blockquote prefix stripping (early pass) ─────────────────────
        # Strip > prefixes once so code fences, tables, and other block
        # elements inside blockquotes are detected correctly.
        bq_depth = 0
        inner = stripped
        if not self._in_code_block:
            while inner.startswith(">"):
                bq_depth += 1
                inner = inner[1:].lstrip(" ")
            inner = inner.strip()
        elif self._code_bq_depth > 0:
            # Inside a code block that started in a blockquote —
            # strip the same number of > prefixes to get the code content
            tmp = inner
            for _ in range(self._code_bq_depth):
                if tmp.startswith(">"):
                    tmp = tmp[1:].lstrip(" ")
                else:
                    break
            bq_depth = self._code_bq_depth
            inner = tmp.strip() if tmp != inner else inner

        # ── HTML stripping (outside code blocks) ───────────────────────
        if not self._in_code_block and _RE_HTML_DANGEROUS_OPEN.search(inner):
            return ""
        if not self._in_code_block and "<" in inner:
            inner = _RE_HTML_TAG.sub("", inner).strip()
            if not inner:
                return ""

        # ── fenced code block toggle ────────────────────────────────────
        if inner.startswith("```"):
            result = self._flush_table_buf()
            w = _decor_width()
            bar = f"{self._c_dim}│ {_RST}" * bq_depth if bq_depth > 0 else ""
            if not self._in_code_block:
                self._in_code_block = True
                self._code_bq_depth = bq_depth
                self._code_lang = inner[3:].strip()
                # Try to get a Pygments lexer for syntax highlighting
                self._code_lexer = None
                if self._has_pygments and self._code_lang:
                    try:
                        from pygments.lexers import get_lexer_by_name
                        self._code_lexer = get_lexer_by_name(self._code_lang)
                    except Exception:
                        pass
                lang = f" {self._code_lang} " if self._code_lang else ""
                fill = max(0, w - 2 - len(lang))
                fence = f"{bar}{self._c_dim}╭──{lang}{'─' * fill}╮{_RST}"
            else:
                self._in_code_block = False
                self._code_bq_depth = 0
                self._code_lang = ""
                self._code_lexer = None
                fence = f"{bar}{self._c_dim}╰{'─' * w}╯{_RST}"
            return f"{result}\n{fence}" if result else fence

        # ── inside code block — syntax highlight or tint with code color ─
        if self._in_code_block:
            bar = f"{self._c_dim}│ {_RST}" * self._code_bq_depth if self._code_bq_depth > 0 else ""
            highlighted = self._highlight_code_line(inner)
            return f"{bar}{highlighted}"

        # ── table handling (stateful) ───────────────────────────────────
        # Buffer all pipe-delimited rows until a non-table line arrives,
        # then render the whole table with column widths derived from ALL rows.
        # Also detects tables inside blockquotes (> | A | B |).
        is_table_row = inner.startswith("|") and inner.endswith("|")

        if is_table_row:
            # If blockquote depth changed mid-table, flush the old table first
            if self._table_buf and self._table_bq_depth != bq_depth:
                prefix = self._flush_table_buf()
                # Re-enter: start a new table at the new depth
            else:
                prefix = None
            self._table_bq_depth = bq_depth
            if _RE_TABLE_SEP.match(inner) and self._table_buf:
                self._table_sep_idx = len(self._table_buf)
            self._table_buf.append(inner)
            return prefix  # None unless we flushed a previous table

        # Not a table line — flush any buffered table
        prefix = self._flush_table_buf()

        # ── headers (H1–H6) with visual weight differentiation ──────────
        for level in (6, 5, 4, 3, 2, 1):
            hdr_prefix = "#" * level + " "
            if line.startswith(hdr_prefix):
                content = line[len(hdr_prefix):]
                styled = self._style_header(content, level)
                return f"{prefix}\n{styled}" if prefix else styled

        # ── horizontal rule ─────────────────────────────────────────────
        if stripped in ("---", "***", "___"):
            w = _decor_width()
            out = f"{self._c_dim}{'─' * w}{_RST}"
            return f"{prefix}\n{out}" if prefix else out

        # ── blockquote (uses bq_depth/inner from early pass) ─────────────
        if bq_depth > 0:
            bar = f"{self._c_dim}│ {_RST}" * bq_depth
            rendered = self._render_block_content(inner)
            out = f"{bar}{rendered}"
            return f"{prefix}\n{out}" if prefix else out

        # ── unordered list ──────────────────────────────────────────────
        m = _RE_ULIST.match(line)
        if m:
            indent, _, content = m.groups()
            out = f"{indent}  • {self._render_inline(content)}"
            return f"{prefix}\n{out}" if prefix else out

        # ── ordered list ────────────────────────────────────────────────
        m = _RE_OLIST.match(line)
        if m:
            indent, num, content = m.groups()
            out = f"{indent}  {num} {self._render_inline(content)}"
            return f"{prefix}\n{out}" if prefix else out

        # ── plain text with inline formatting ───────────────────────────
        # Strip leading whitespace from plain text — models sometimes insert
        # newline + space mid-paragraph, which shows as a spurious indent.
        # (Block-level indentation is already handled above by list matchers.)
        out = self._render_inline(line.lstrip())
        return f"{prefix}\n{out}" if prefix else out

    # ── header styling helper ──────────────────────────────────────────

    def _style_header(self, content: str, level: int) -> str:
        """Style a header with visual weight appropriate to its level.

        Terminal can't change font size, so we differentiate with:
          H1: bold + underline + accent color + full-width rule below
          H2: bold + underline + accent color
          H3: bold + accent color
          H4: bold + accent color (dimmer via no underline)
          H5: accent color only (no bold)
          H6: dim + italic
        """
        accent = self._c_header  # already includes _BOLD
        if level == 1:
            w = _decor_width()
            return (f"{accent}{content}{_RST}\n"
                    f"{self._c_dim}{'━' * w}{_RST}")
        elif level == 2:
            return f"{_UNDERLINE}{accent}{content}{_RST}"
        elif level == 3:
            return f"{accent}{content}{_RST}"
        elif level == 4:
            return f"{accent}{content}{_RST}"
        elif level == 5:
            # Strip bold from accent — use color only
            accent_no_bold = accent.replace(_BOLD, "")
            return f"{accent_no_bold}{content}{_RST}"
        else:  # H6
            return f"{_DIM}{_ITALIC}{content}{_RST}"

    # ── code highlighting helper ───────────────────────────────────────

    def _highlight_code_line(self, line: str) -> str:
        """Syntax-highlight a single code line via Pygments, or fall back to code color."""
        if self._code_lexer and self._code_formatter:
            try:
                from pygments import highlight as _pyg_hl
                # Pygments adds a trailing newline; strip it
                result = _pyg_hl(line, self._code_lexer, self._code_formatter)
                return result.rstrip("\n")
            except Exception:
                pass
        return f"{self._c_code}{line}{_RST}"

    # ── block-content helper (for nested contexts like blockquotes) ────

    def _render_block_content(self, text: str) -> str:
        """Render text that may contain block-level elements (lists, headers).

        Used for content inside blockquotes where the ``>`` prefix has been
        stripped but the remainder can still be a list item or header.
        """
        # Headers
        for level in (6, 5, 4, 3, 2, 1):
            hdr_prefix = "#" * level + " "
            if text.startswith(hdr_prefix):
                return f"{self._c_header}{text[len(hdr_prefix):]}{_RST}"

        # Unordered list
        m = _RE_ULIST.match(text)
        if m:
            indent, _, content = m.groups()
            return f"{indent}  • {self._render_inline(content)}"

        # Ordered list
        m = _RE_OLIST.match(text)
        if m:
            indent, num, content = m.groups()
            return f"{indent}  {num} {self._render_inline(content)}"

        # Plain inline formatting
        return self._render_inline(text)

    # ── table rendering helpers ─────────────────────────────────────────

    def _flush_table_buf(self) -> str | None:
        """Render the buffered table with globally-calculated column widths.

        Returns the fully rendered table as a multi-line string,
        or ``None`` if the buffer is empty.
        """
        if not self._table_buf:
            return None

        buf = self._table_buf
        sep_idx = self._table_sep_idx
        self._table_buf = []
        self._table_sep_idx = -1

        # No separator found — not a real table, render as plain lines
        if sep_idx < 0:
            return "\n".join(self._render_inline(row) for row in buf)

        # Parse all rows into cell lists and pre-render inline formatting
        all_cells = [_parse_table_cells(row) for row in buf]
        all_rendered = []
        for i, cells in enumerate(all_cells):
            if i == sep_idx:
                all_rendered.append(cells)  # separator — not rendered
            else:
                all_rendered.append([self._render_inline(c) for c in cells])

        # Calculate column widths from RENDERED visible widths of ALL data rows
        n_cols = max(len(cells) for cells in all_cells)
        widths = [3] * n_cols
        for i, rendered_cells in enumerate(all_rendered):
            if i == sep_idx:
                continue
            for j, cell in enumerate(rendered_cells):
                if j < n_cols:
                    widths[j] = max(widths[j], _visible_width(cell))

        # Render each row
        lines = []
        for i, row in enumerate(buf):
            if i == sep_idx:
                segs = ["─" * w for w in widths]
                lines.append(self._c_dim + "─┼─".join(segs) + _RST)
            else:
                is_header = i < sep_idx
                rendered_cells = all_rendered[i]
                parts = []
                for j, w in enumerate(widths):
                    cell = rendered_cells[j] if j < len(rendered_cells) else ""
                    pad = w - _visible_width(cell)
                    parts.append(cell + " " * max(0, pad))
                fmt = _BOLD if is_header else ""
                rst = _RST if is_header else ""
                sep = f"{self._c_dim} │ {_RST}"
                lines.append(f"{fmt}{sep.join(parts)}{rst}")

        result = "\n".join(lines)

        # Prepend blockquote bars if this table was inside a blockquote
        bq_depth = self._table_bq_depth
        self._table_bq_depth = 0
        if bq_depth > 0:
            bar = f"{self._c_dim}│ {_RST}" * bq_depth
            result = "\n".join(f"{bar}{l}" for l in result.split("\n"))

        return result

    # ── inline formatting ───────────────────────────────────────────────

    def _render_inline(self, text: str) -> str:
        """Apply inline markdown formatting (bold, italic, code, links, etc.).

        Uses a three-phase approach to prevent regex substitutions from
        interfering with each other:
          1. Extract atomic elements (escapes, code spans, links) into
             placeholders so their content is protected.
          2. Apply emphasis (bold, italic, strikethrough) on the remaining text.
          3. Restore placeholders.
        """
        if not text:
            return text

        placeholders = []

        def _save(rendered: str) -> str:
            idx = len(placeholders)
            placeholders.append(rendered)
            return f"{_PH}{idx}{_PH}"

        # Phase 1: extract atomic elements (order matters — most specific first)

        # Backslash escapes: \* \_ \~ \` \[ \] \\  →  literal character
        text = _RE_ESCAPED.sub(lambda m: _save(m.group(1)), text)

        # Code spans — content is never further formatted
        text = _RE_CODE.sub(
            lambda m: _save(f"{self._c_code}{m.group(1)}{_RST}"), text
        )

        # Links — render as underlined text, strip URL
        text = _RE_LINK.sub(
            lambda m: _save(f"{_UNDERLINE}{m.group(1)}{_RST}"), text
        )

        # Phase 2: emphasis on the remaining (unprotected) text
        text = _RE_BOLD_ITALIC.sub(f"{_BOLD}{_ITALIC}\\1{_RST}", text)
        text = _RE_BOLD.sub(f"{_BOLD}\\1{_RST}", text)
        text = _RE_BOLD_UND.sub(f"{_BOLD}\\1{_RST}", text)
        text = _RE_ITALIC.sub(f"{_ITALIC}\\1{_RST}", text)
        text = _RE_STRIKE.sub(f"{_STRIKE}\\1{_RST}", text)

        # Phase 3: restore placeholders
        for idx, rendered in enumerate(placeholders):
            text = text.replace(f"{_PH}{idx}{_PH}", rendered)

        return text
