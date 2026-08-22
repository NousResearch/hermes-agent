"""Streaming Markdown rendering for the Hermes CLI.

Provides ``MarkdownStreamProcessor``, a stateful line-by-line
Markdown-to-ANSI transformer for the streaming display path.
"""

from __future__ import annotations

import re
import shutil
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI escape helpers
# ---------------------------------------------------------------------------

_RST = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"
_UNDERLINE = "\033[4m"
_STRIKETHROUGH = "\033[9m"
_BOLD_ITALIC = "\033[1;3m"

# Sentinel Unicode chars (private-use area) for escaped markdown characters.
# These are substituted *before* regex transforms and restored *after*.
_ESC_STAR = "\ue000"
_ESC_UNDER = "\ue001"
_ESC_TILDE = "\ue002"
_ESC_BACKTICK = "\ue003"

_ESCAPE_MAP = {
    r"\*": _ESC_STAR,
    r"\_": _ESC_UNDER,
    r"\~": _ESC_TILDE,
    r"\`": _ESC_BACKTICK,
}
_RESTORE_MAP = {v: k[1] for k, v in _ESCAPE_MAP.items()}  # sentinel -> literal char


# ---------------------------------------------------------------------------
# Inline markdown regex transforms
# ---------------------------------------------------------------------------

_RE_HEADER = re.compile(r"^(#{1,6})\s+(.+)$")
_RE_HR = re.compile(r"^[\s]*[-*_]{3,}[\s]*$")
_RE_INLINE_CODE = re.compile(r"(?<!`)`(?!`)([^`]+)`(?!`)")
_RE_BOLD_ITALIC = re.compile(r"\*{3}(\S(?:.*?\S)?)\*{3}")
_RE_BOLD_STAR = re.compile(r"\*{2}(\S(?:.*?\S)?)\*{2}")
_RE_BOLD_UNDER = re.compile(r"__(\S(?:.*?\S)?)__")
_RE_ITALIC_STAR = re.compile(r"(?<!\w)\*(\S(?:.*?\S)?)\*(?!\w)")
_RE_ITALIC_UNDER = re.compile(r"(?<!\w)_(\S(?:.*?\S)?)_(?!\w)")
_RE_STRIKETHROUGH = re.compile(r"~~(\S(?:.*?\S)?)~~")
_RE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_RE_BLOCKQUOTE = re.compile(r"^(\s*>\s?)(.*)$")
_RE_UL_MARKER = re.compile(r"^(\s*)([-*+])\s")
_RE_OL_MARKER = re.compile(r"^(\s*)(\d+\.)\s")

# Pipe-table row: starts and ends with | (after stripping), has at least one
# inner |.  Also matches separator rows like |---|---|.
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|.*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:?-]+(\|[\s:?-]+)+\|\s*$")


def _strip_inline_markers(text: str) -> str:
    """Remove inline markdown markers (**bold**, *italic*, `code`, ~~strike~~).

    Used for headers and table cells where the surrounding context already
    provides styling — we just need to remove the raw marker characters.
    """
    text = _RE_BOLD_ITALIC.sub(r"\1", text)
    text = _RE_BOLD_STAR.sub(r"\1", text)
    text = _RE_BOLD_UNDER.sub(r"\1", text)
    text = _RE_ITALIC_STAR.sub(r"\1", text)
    text = _RE_ITALIC_UNDER.sub(r"\1", text)
    text = _RE_STRIKETHROUGH.sub(r"\1", text)
    text = _RE_INLINE_CODE.sub(r"\1", text)
    return text


# ---------------------------------------------------------------------------
# MarkdownStreamProcessor
# ---------------------------------------------------------------------------

class MarkdownStreamProcessor:
    """Stateful line-by-line markdown-to-ANSI transformer for streaming output.

    Designed to sit between ``_emit_stream_text``'s line buffer and ``_cprint``.
    Processes one complete line at a time.  The caller handles line buffering
    (which ``_emit_stream_text`` already does).

    Two states:
        NORMAL     – apply inline markdown regex transforms
        CODE_BLOCK – highlight each line with Pygments
    """

    def __init__(
        self,
        base_ansi: str = "",
        pygments_theme: str = "monokai",
        inline_code_ansi: str = "",
        max_width: Optional[int] = None,
    ):
        """
        Args:
            base_ansi: Base text color ANSI escape (from skin's banner_text).
                       Applied after every reset to restore the "default" color.
            pygments_theme: Pygments style name for code blocks.
            inline_code_ansi: Complete ANSI escape to apply to inline ``code``
                       spans (e.g. bold + optional fg + optional bg resolved
                       from the skin). When empty, inline code falls back to
                       plain bold.
            max_width: Maximum visible cells available to rendered output.
                       The caller should subtract any indentation it prepends.
                       When omitted, use the terminal width for compatibility.
        """
        self._base = base_ansi
        self._rst = _RST
        self._theme = pygments_theme
        self._inline_code_ansi = inline_code_ansi
        self._max_width = max_width

        # State
        self._in_code_block = False
        self._code_lang = ""
        self._code_lines: list[str] = []
        self._lexer = None
        self._formatter = None
        self._md_fence_depth = 0  # depth of skipped ```markdown/```md fences
        self._table_rows: list[str] = []  # buffered pipe-table rows

    # -- public API ---------------------------------------------------------

    def feed_line(self, line: str) -> Optional[str]:
        """Process one complete line and return ANSI-formatted output.

        Returns ``None`` when the line is buffered (e.g. table rows) and
        should NOT be printed.  The caller must skip ``_cprint`` for ``None``.
        """
        stripped = line.lstrip()

        # -- Table buffering ---------------------------------------------------
        # Pipe-table rows are buffered until a non-table line arrives, then
        # the whole table is rendered at once (needs all rows for column widths).
        if not self._in_code_block and _TABLE_ROW_RE.match(line):
            self._table_rows.append(line)
            return None  # suppress — will be emitted when the table ends

        # If we were buffering a table and this line is NOT a table row,
        # flush the table first, then process this line normally.
        table_output = ""
        if self._table_rows:
            table_output = self._render_table()
            self._table_rows.clear()

        # -- Fence detection: ``` at start of line (after optional whitespace) --
        # Only match lines that are JUST a fence: ```lang or ``` alone.
        # Reject lines like "```text``` more stuff" or ```` (4+ backticks).
        if stripped.startswith("```") and not stripped.startswith("````"):
            _fence_rest = stripped[3:].strip()
            _is_fence = (not _fence_rest                          # bare ```
                         or re.match(r"^[a-zA-Z0-9_+-]+$", _fence_rest))  # ```lang

            if _is_fence:
                if self._in_code_block:
                    # Closing fence
                    self._exit_code_block()
                    return self._render_fence_footer()

                # Opening fence — extract language
                lang = _fence_rest or ""

                # Skip ```markdown / ```md fences — LLMs often wrap entire
                # responses in these.  Render the inner content as normal
                # markdown instead of treating it as a code block.
                if lang.lower() in ("markdown", "md"):
                    self._md_fence_depth += 1
                    return table_output or ""

                # Bare ``` in NORMAL mode: if we previously skipped a
                # ```markdown opener, this is its matching closer — skip it.
                if not lang and self._md_fence_depth > 0:
                    self._md_fence_depth -= 1
                    return table_output or ""

                self._enter_code_block(lang)
                fence = self._render_fence_header(lang)
                return f"{table_output}\n{fence}" if table_output else fence

        if self._in_code_block:
            self._code_lines.append(line)
            return self._highlight_code_line(line)

        result = self._apply_inline_markdown(line)
        return f"{table_output}\n{result}" if table_output else result

    def flush(self) -> Optional[str]:
        """Finalize state.  Returns any pending output (table, code block)."""
        parts = []
        if self._table_rows:
            parts.append(self._render_table())
            self._table_rows.clear()
        if self._in_code_block:
            self._exit_code_block()
            parts.append(self._render_fence_footer())
        return "\n".join(parts) if parts else None

    def reset(self):
        """Reset all state for a new response."""
        self._in_code_block = False
        self._code_lang = ""
        self._code_lines.clear()
        self._lexer = None
        self._formatter = None
        self._md_fence_depth = 0
        self._table_rows.clear()

    # -- code block internals -----------------------------------------------

    def _enter_code_block(self, lang: str):
        self._in_code_block = True
        self._code_lang = lang or "text"
        self._code_lines.clear()
        self._lexer = None
        self._formatter = None

    def _exit_code_block(self):
        self._in_code_block = False
        self._code_lang = ""
        self._code_lines.clear()
        self._lexer = None
        self._formatter = None

    def _get_lexer(self):
        if self._lexer is None:
            from pygments.lexers import get_lexer_by_name, TextLexer
            try:
                self._lexer = get_lexer_by_name(self._code_lang, stripall=False)
            except Exception:
                self._lexer = TextLexer()
        return self._lexer

    def _get_formatter(self):
        if self._formatter is None:
            from pygments.formatters import TerminalTrueColorFormatter
            self._formatter = TerminalTrueColorFormatter(style=self._theme)
        return self._formatter

    def _highlight_code_line(self, line: str) -> str:
        from pygments import highlight as _pyg_highlight
        lexer = self._get_lexer()
        formatter = self._get_formatter()
        # Highlight the single line; Pygments adds a trailing newline — strip it
        result = _pyg_highlight(line + "\n", lexer, formatter).rstrip("\n")
        return f"  {result}{self._rst}"

    def _render_fence_header(self, lang: str) -> str:
        w = self._max_width
        if w is None:
            w = shutil.get_terminal_size((80, 24)).columns
        w = max(int(w), 10)
        label = f" {lang} " if lang else ""
        # Keep at least three border cells after the label. Fence languages are
        # ASCII-only by parser contract, so len() is also their display width.
        label = label[:max(w - 9, 0)]
        fill = max(w - 6 - len(label), 3)
        return f"{_DIM}  ┌───{self._rst}{_DIM}{_BOLD}{label}{self._rst}{_DIM}{'─' * fill}{self._rst}"

    def _render_fence_footer(self) -> str:
        w = self._max_width
        if w is None:
            w = shutil.get_terminal_size((80, 24)).columns
        w = max(int(w), 10)
        fill = max(w - 4, 3)
        return f"{_DIM}  └{'─' * fill}{self._rst}"

    # -- table internals ----------------------------------------------------

    def _render_table(self) -> str:
        """Render buffered pipe-table rows without exceeding caller width."""
        base = self._base
        rst = self._rst

        # Reuse the CLI's established narrow-table decision before drawing
        # the richer box. Its horizontal representation is two cells narrower
        # than our indented box, so reserve those cells in the budget. If it
        # selects the vertical fallback, preserve that width-safe layout.
        if self._max_width is not None:
            from agent.markdown_tables import is_table_divider, realign_markdown_tables

            budget = max(int(self._max_width) - 2, 1)
            aligned = realign_markdown_tables("\n".join(self._table_rows), budget)
            aligned_lines = aligned.splitlines()
            stays_horizontal = len(aligned_lines) > 1 and is_table_divider(aligned_lines[1])
            if not stays_horizontal:
                rendered_lines = []
                for line in aligned_lines:
                    plain = _strip_inline_markers(line)
                    if plain and set(plain) == {"─"}:
                        rendered_lines.append(f"{_DIM}{plain}{rst}")
                    elif base:
                        rendered_lines.append(f"{base}{plain}{rst}")
                    else:
                        rendered_lines.append(plain)
                return "\n".join(rendered_lines)

        # Parse rows into cells
        parsed: list[list[str]] = []
        sep_indices: set[int] = set()
        for i, raw in enumerate(self._table_rows):
            row = raw.strip()
            # Strip leading/trailing pipes and split
            if row.startswith("|"):
                row = row[1:]
            if row.endswith("|"):
                row = row[:-1]
            cells = [_strip_inline_markers(c.strip()) for c in row.split("|")]
            parsed.append(cells)
            if _TABLE_SEP_RE.match(self._table_rows[i]):
                sep_indices.add(i)

        if not parsed:
            return ""

        # Compute column widths using terminal-aware width (handles emoji,
        # CJK characters, etc. that occupy 2 cells in the terminal).
        try:
            from wcwidth import wcswidth
            def _display_width(s: str) -> int:
                w = wcswidth(s)
                return w if w >= 0 else len(s)
        except ImportError:
            _display_width = len

        n_cols = max(len(r) for r in parsed)
        col_widths = [0] * n_cols
        for i, row in enumerate(parsed):
            if i in sep_indices:
                continue  # don't let separator dashes inflate widths
            for j, cell in enumerate(row):
                if j < n_cols:
                    col_widths[j] = max(col_widths[j], _display_width(cell))

        # Ensure minimum width of 3 per column
        col_widths = [max(w, 3) for w in col_widths]

        def _pad_cell(val: str, target_width: int) -> str:
            """Pad a cell value to target_width using display-aware spacing."""
            current = _display_width(val)
            pad = max(target_width - current, 0)
            return val + " " * pad

        def _border(left: str, mid: str, right: str, fill: str = "─") -> str:
            segs = [fill * (w + 2) for w in col_widths]
            return f"{_DIM}{left}{mid.join(segs)}{right}{rst}"

        lines = []
        lines.append(_border("  ┌", "┬", "┐"))

        for i, row in enumerate(parsed):
            if i in sep_indices:
                # Render separator as a mid-border
                lines.append(_border("  ├", "┼", "┤"))
            else:
                # Bold the header row (row 0, if followed by a separator)
                is_header = (i == 0 and 1 in sep_indices)
                cell_strs = []
                for j in range(n_cols):
                    val = row[j] if j < len(row) else ""
                    padded = _pad_cell(val, col_widths[j])
                    cell_strs.append(
                        f" {_BOLD if is_header else ''}{base}{padded}{rst} "
                    )
                inner = f"{_DIM}│{rst}".join(cell_strs)
                lines.append(f"  {_DIM}│{rst}{inner}{_DIM}│{rst}")

        lines.append(_border("  └", "┴", "┘"))
        return "\n".join(lines)

    # -- inline markdown internals ------------------------------------------

    def _apply_inline_markdown(self, line: str) -> str:
        base = self._base
        rst = self._rst

        # 1. Escape backslash-escaped markdown characters
        for esc_seq, sentinel in _ESCAPE_MAP.items():
            line = line.replace(esc_seq, sentinel)

        # 2. Full-line patterns (header, HR) — if matched, apply inline
        #    formatting to the header text (strip bold/italic/code markers).
        m = _RE_HEADER.match(line)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            text = _strip_inline_markers(text)
            if level == 1:
                line = f"{_BOLD}{_UNDERLINE}{text}{rst}{base}"
            elif level == 2:
                line = f"{_BOLD}{text}{rst}{base}"
            else:
                line = f"{_BOLD}{_DIM}{text}{rst}{base}"
            # Restore escaped chars and return early
            for sentinel, literal in _RESTORE_MAP.items():
                line = line.replace(sentinel, literal)
            return f"{base}{line}{rst}" if base else line

        if _RE_HR.match(line):
            w = shutil.get_terminal_size((80, 24)).columns
            line = f"{_DIM}{'─' * min(w - 4, 40)}{rst}{base}"
            for sentinel, literal in _RESTORE_MAP.items():
                line = line.replace(sentinel, literal)
            return f"{base}{line}{rst}" if base else line

        # 3. Protect inline code FIRST — extract `code` spans, replace with
        #    sentinels so bold/italic regexes don't eat their content.
        _code_spans: list[str] = []
        _CODE_SENTINEL = "\ue010"

        def _protect_code(m):
            idx = len(_code_spans)
            code_esc = self._inline_code_ansi or _BOLD
            _code_spans.append(f"{code_esc}{m.group(1)}{rst}{base}")
            return f"{_CODE_SENTINEL}{idx}{_CODE_SENTINEL}"

        line = _RE_INLINE_CODE.sub(_protect_code, line)

        # 4. Links
        line = _RE_LINK.sub(
            lambda m: f"{_UNDERLINE}{m.group(1)}{rst}{_DIM} ({m.group(2)}){rst}{base}",
            line,
        )

        # 5. Bold+italic, bold, italic, strikethrough
        line = _RE_BOLD_ITALIC.sub(lambda m: f"{_BOLD_ITALIC}{m.group(1)}{rst}{base}", line)
        line = _RE_BOLD_STAR.sub(lambda m: f"{_BOLD}{m.group(1)}{rst}{base}", line)
        line = _RE_BOLD_UNDER.sub(lambda m: f"{_BOLD}{m.group(1)}{rst}{base}", line)
        line = _RE_ITALIC_STAR.sub(lambda m: f"{_ITALIC}{m.group(1)}{rst}{base}", line)
        line = _RE_ITALIC_UNDER.sub(lambda m: f"{_ITALIC}{m.group(1)}{rst}{base}", line)
        line = _RE_STRIKETHROUGH.sub(lambda m: f"{_STRIKETHROUGH}{m.group(1)}{rst}{base}", line)

        # 6. Blockquote (full-line, but after inline transforms for content)
        m = _RE_BLOCKQUOTE.match(line)
        if m:
            text = m.group(2)
            line = f"{_DIM}  │ {_ITALIC}{text}{rst}{base}"

        # 7. List markers
        m = _RE_UL_MARKER.match(line)
        if m:
            indent = m.group(1)
            rest = line[m.end():]
            line = f"{indent}  • {rest}"
        else:
            m = _RE_OL_MARKER.match(line)
            if m:
                indent = m.group(1)
                num = m.group(2)
                rest = line[m.end():]
                line = f"{indent}  {_DIM}{num}{rst}{base} {rest}"

        # 8. Restore inline code sentinels
        for i, rendered in enumerate(_code_spans):
            line = line.replace(f"{_CODE_SENTINEL}{i}{_CODE_SENTINEL}", rendered)

        # 9. Restore escaped characters
        for sentinel, literal in _RESTORE_MAP.items():
            line = line.replace(sentinel, literal)

        # Apply base color wrapping
        if base:
            return f"{base}{line}{rst}"
        return line
