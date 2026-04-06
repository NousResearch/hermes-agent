"""Rich-based syntax highlighting, diff rendering, and output utilities.

Drop into hermes-agent's ``agent/`` directory.

No project-specific imports — only ``rich`` (always present in Hermes) and
``pygments`` (bundled as a rich dependency).

Public API
----------
LanguageDetector        detect language from filename / content
FilePathFormatter       per-type icons + compact relative-path display
SyntaxHighlighter       Pygments → Rich markup → ANSI string
DiffRenderer            unified diff → Rich Text with line numbers → ANSI lines
apply_inline_markdown   convert **bold** / *italic* / `code` / ~~strike~~ to ANSI
apply_block_line        convert block-level markdown (headings, hr, blockquotes,
                        lists) to ANSI on a single line
render_stateful_blocks  setext headings, blockquote continuation, tables (pass 2)
StreamingBlockBuffer    streaming-pipeline state machine for stateful blocks
clean_command_output    strip venv/stacktrace noise from command output
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import unicodedata
from difflib import SequenceMatcher
from io import StringIO
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.markup import escape as _markup_escape
from rich.style import Style
from rich.text import Text

logger = logging.getLogger(__name__)

# Diff background colours — kept in sync with agent.display._ANSI_PLUS / _ANSI_MINUS
# _ANSI_PLUS  = "\033[38;2;255;255;255;48;2;20;90;20m"   → rgb(20,90,20)
# _ANSI_MINUS = "\033[38;2;255;255;255;48;2;120;20;20m"  → rgb(120,20,20)
_DIFF_BG_ADD    = "#145a14"   # rgb(20,  90, 20)  — base addition background
_DIFF_BG_DEL    = "#781414"   # rgb(120, 20, 20)  — base deletion background
_DIFF_BG_ADD_HL = "#289428"   # rgb(40, 148, 40)  — bright add bg for changed chars
_DIFF_BG_DEL_HL = "#b43030"   # rgb(180, 48, 48)  — bright del bg for changed chars

# Minimum SequenceMatcher ratio to apply intra-line highlighting.
# Below this the lines are too dissimilar and highlighting would be noise.
_INTRA_DIFF_MIN_RATIO: float = 0.5

# ---------------------------------------------------------------------------
# Pygments availability (bundled transitively via rich, but guard anyway)
# ---------------------------------------------------------------------------

try:
    from pygments.lexers import (
        TextLexer,
        get_lexer_by_name,
        get_lexer_for_filename,
        guess_lexer,
    )
    from pygments.token import (
        Comment,
        Error,
        Generic,
        Keyword,
        Name,
        Number,
        Operator,
        String,
    )
    from pygments.util import ClassNotFound

    _PYGMENTS = True
except ImportError:
    _PYGMENTS = False


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

class LanguageDetector:
    """Detect programming language from filename extension or code content."""

    EXTENSION_MAP: dict[str, str] = {
        # Python
        ".py": "python", ".pyx": "python", ".pyi": "python", ".pyw": "python",
        # JavaScript / TypeScript
        ".js": "javascript", ".jsx": "jsx", ".mjs": "javascript", ".cjs": "javascript",
        ".ts": "typescript", ".tsx": "tsx",
        # JVM
        ".java": "java", ".scala": "scala", ".kt": "kotlin", ".groovy": "groovy",
        # C family
        ".c": "c", ".h": "c",
        ".cpp": "cpp", ".cxx": "cpp", ".cc": "cpp", ".hpp": "cpp",
        ".cs": "csharp", ".fs": "fsharp",
        # Systems
        ".rs": "rust", ".go": "go", ".swift": "swift",
        # Scripting
        ".rb": "ruby", ".php": "php", ".pl": "perl", ".lua": "lua",
        ".r": "r", ".R": "r",
        # Web
        ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
        ".sass": "sass", ".vue": "vue", ".svelte": "svelte",
        # Shell
        ".sh": "bash", ".bash": "bash", ".zsh": "zsh", ".fish": "fish",
        ".ps1": "powershell", ".bat": "batch", ".cmd": "batch",
        # Data / config
        ".json": "json", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".ini": "ini", ".cfg": "ini", ".xml": "xml",
        # Docs
        ".md": "markdown", ".rst": "rst", ".tex": "latex",
        # DB
        ".sql": "sql",
        # Containers
        ".dockerfile": "dockerfile",
        # Other
        ".dart": "dart", ".ex": "elixir", ".exs": "elixir",
        ".erl": "erlang", ".hs": "haskell", ".ml": "ocaml",
        ".elm": "elm", ".zig": "zig", ".vim": "vim",
    }

    CONTENT_PATTERNS: dict[str, list[str]] = {
        "python": [
            r"^\s*def\s+\w+\s*\(", r"^\s*class\s+\w+\s*[\(:]",
            r"^\s*import\s+\w+", r"^\s*from\s+\w+\s+import",
            r'if\s+__name__\s*==\s*[\'"]__main__[\'"]',
        ],
        "javascript": [
            r"^\s*function\s+\w+\s*\(", r"^\s*const\s+\w+\s*=",
            r"console\.log\s*\(", r'require\s*\([\'"]', r"module\.exports",
        ],
        "typescript": [
            r"^\s*interface\s+\w+", r"^\s*type\s+\w+\s*=",
            r":\s*string\s*[;,}]", r":\s*number\s*[;,}]",
        ],
        "java": [r"^\s*public\s+class\s+\w+", r"System\.out\.print"],
        "cpp": [r"#include\s*<\w+>", r"std::\w+", r"cout\s*<<"],
        "go": [r"^\s*package\s+\w+", r"^\s*func\s+\w+\s*\(", r"fmt\.Print"],
        "rust": [r"^\s*fn\s+\w+\s*\(", r"^\s*use\s+\w+", r"println!\s*\("],
        "bash": [r"#!/bin/bash", r"#!/bin/sh", r"^\s*if\s*\[", r"\$\{\w+\}"],
        "sql": [r"^\s*SELECT\s+", r"^\s*INSERT\s+INTO", r"^\s*CREATE\s+TABLE"],
    }

    def detect_from_filename(self, filename: str) -> Optional[str]:
        if not filename:
            return None
        ext = Path(filename).suffix.lower()
        if ext in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[ext]
        name = Path(filename).name.lower()
        if name in {"dockerfile", "makefile", "rakefile", "gemfile", "vagrantfile"}:
            return name
        return None

    def detect_from_content(self, content: str, max_lines: int = 50) -> Optional[str]:
        if not content.strip():
            return None
        sample = "\n".join(content.split("\n")[:max_lines])
        scores: dict[str, int] = {}
        for lang, patterns in self.CONTENT_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, sample, re.MULTILINE))
            if score:
                scores[lang] = score
        return max(scores, key=lambda k: scores[k]) if scores else None

    def detect(self, content: str, filename: Optional[str] = None) -> Optional[str]:
        return self.detect_from_filename(filename) or self.detect_from_content(content)


# ---------------------------------------------------------------------------
# File path formatting
# ---------------------------------------------------------------------------

class FilePathFormatter:
    """Per-filetype icons and compact relative-path display."""

    _ICONS: dict[str, str] = {
        ".py": "🐍", ".js": "📜", ".ts": "📘", ".tsx": "⚛️", ".jsx": "⚛️",
        ".html": "🌐", ".css": "🎨", ".scss": "🎨", ".md": "📝",
        ".json": "📋", ".yaml": "⚙️", ".yml": "⚙️", ".toml": "⚙️",
        ".txt": "📄", ".log": "📊", ".conf": "⚙️", ".cfg": "⚙️",
        ".xml": "📋", ".sql": "🗃️", ".sh": "💻", ".bash": "💻",
        ".go": "🐹", ".rs": "🦀", ".java": "☕", ".cpp": "⚙️",
        ".c": "⚙️", ".h": "📋",
    }

    @staticmethod
    def get_file_icon(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        return FilePathFormatter._ICONS.get(ext, "📄")

    @staticmethod
    def format_path(
        file_path: str,
        compact: bool = True,
        cwd: Optional[str] = None,
    ) -> str:
        if not compact:
            return file_path
        try:
            return os.path.relpath(file_path, cwd or os.getcwd())
        except (ValueError, OSError):
            return file_path

    @staticmethod
    def titled(
        file_path: str,
        compact: bool = True,
        cwd: Optional[str] = None,
    ) -> str:
        """Return ``{icon} {path}`` string."""
        icon = FilePathFormatter.get_file_icon(file_path)
        path = FilePathFormatter.format_path(file_path, compact, cwd)
        return f"{icon} {path}"


# ---------------------------------------------------------------------------
# Pygments → Rich markup formatter (internal)
# ---------------------------------------------------------------------------

class _PygmentsToRich:
    """Convert a Pygments token stream to a Rich markup string.

    By default uses a built-in colour palette that looks good on dark terminals.
    Call :func:`set_code_theme` to load colours from a named Pygments style
    (e.g. ``"monokai"``, ``"dracula"``) — typically driven by the active skin.
    """

    # Built lazily so the class-level dict isn't populated when Pygments is absent.
    # Replaced wholesale by set_code_theme() when a named Pygments style is loaded.
    _STYLES: dict = {}
    _current_theme: str = ""  # tracks which theme name is currently active

    # The built-in fallback palette — used when no Pygments style is requested
    _DEFAULT_STYLES: dict = {}

    @classmethod
    def _ensure_styles(cls) -> None:
        if cls._STYLES or not _PYGMENTS:
            return
        cls._DEFAULT_STYLES = {
            Keyword: "bold blue",
            Keyword.Type: "bold cyan",
            Name: "white",
            Name.Builtin: "cyan",
            Name.Class: "bold yellow",
            Name.Constant: "bold yellow",
            Name.Decorator: "bright_cyan",
            Name.Exception: "bold red",
            Name.Function: "bold yellow",
            Name.Function.Magic: "cyan",
            Name.Tag: "bold blue",
            Name.Variable.Magic: "cyan",
            Comment: "dim green",
            Comment.Preproc: "bold green",
            String: "green",
            String.Doc: "dim green",
            String.Escape: "bold green",
            String.Interpol: "bold green",
            String.Regex: "magenta",
            Number: "magenta",
            Operator: "white",
            Operator.Word: "bold blue",
            Generic.Deleted: "red",
            Generic.Inserted: "green",
            Generic.Error: "bold red",
            Error: "bold red",
        }
        cls._STYLES = dict(cls._DEFAULT_STYLES)

    @classmethod
    def load_theme(cls, theme_name: str) -> bool:
        """Replace the active colour palette with styles from a named Pygments theme.

        Returns True if the theme was loaded successfully, False on any error
        (unknown name, Pygments unavailable, etc.) in which case the built-in
        palette is left intact.
        """
        if not _PYGMENTS or not theme_name:
            return False
        # Don't reload the same theme
        if theme_name == cls._current_theme:
            return True
        try:
            from pygments.styles import get_style_by_name
            pygments_style = get_style_by_name(theme_name)
        except Exception:
            logger.debug("Unknown Pygments theme %r, keeping current palette", theme_name)
            return False

        # Convert Pygments style entries to Rich markup style strings.
        # A Pygments style entry has .color (hex), .bold, .italic, .underline.
        new_styles: dict = {}
        for token_type, style_dict in pygments_style:
            parts: list[str] = []
            if style_dict.get("bold"):
                parts.append("bold")
            if style_dict.get("italic"):
                parts.append("italic")
            if style_dict.get("underline"):
                parts.append("underline")
            color = style_dict.get("color")
            if color:
                parts.append(f"#{color}")
            if parts:
                new_styles[token_type] = " ".join(parts)

        cls._ensure_styles()  # make sure _DEFAULT_STYLES is populated
        cls._STYLES = new_styles
        cls._current_theme = theme_name
        logger.debug("Loaded Pygments theme %r (%d token styles)", theme_name, len(new_styles))
        return True

    @classmethod
    def reset_to_default(cls) -> None:
        """Revert to the built-in colour palette."""
        cls._ensure_styles()
        cls._STYLES = dict(cls._DEFAULT_STYLES)
        cls._current_theme = ""

    def format(self, tokens) -> str:
        self._ensure_styles()
        parts: list[str] = []
        for ttype, value in tokens:
            style = self._resolve(ttype)
            # Use rich.markup.escape() which:
            #   • doubles backslashes (\ → \\) so they render literally and
            #     can never accidentally combine with the [ of a closing tag
            #     to form \[ (Rich's escape for a literal "[")
            #   • escapes [ → \[ so bracket text is never parsed as markup
            #   • leaves ] alone — ] needs no escaping in Rich markup
            esc = _markup_escape(value)
            if style and value.strip():
                parts.append(f"[{style}]{esc}[/{style}]")
            else:
                parts.append(esc)
        return "".join(parts)

    def _resolve(self, ttype) -> Optional[str]:
        t = ttype
        while t is not None:
            if t in self._STYLES:
                return self._STYLES[t]
            t = t.parent  # type: ignore[assignment]
        return None


# ---------------------------------------------------------------------------
# Public: syntax highlighter
# ---------------------------------------------------------------------------

class SyntaxHighlighter:
    """Highlight source code using Pygments, output as Rich markup or ANSI.

    Falls back to plain green when Pygments is unavailable.
    """

    def __init__(self) -> None:
        self._fmt = _PygmentsToRich()
        self._detector = LanguageDetector()

    # -- Rich markup (for embedding in Rich Text / Panel) --------------------

    def to_markup(
        self,
        code: str,
        language: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Return a Rich markup string with syntax colours applied."""
        if not _PYGMENTS:
            return f"[green]{_markup_escape(code)}[/green]"
        try:
            lexer = self._lexer(code, language, filename)
            return self._fmt.format(list(lexer.get_tokens(code)))
        except Exception as exc:
            logger.debug("Pygments highlight failed: %s", exc)
            return f"[green]{_markup_escape(code)}[/green]"

    # -- ANSI string (for plain print / print_fn) ----------------------------

    def to_ansi(
        self,
        code: str,
        language: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Return an ANSI-escaped string suitable for plain ``print()``."""
        markup = self.to_markup(code, language, filename)
        buf = StringIO()
        Console(file=buf, highlight=False, force_terminal=True, width=220).print(markup)
        return buf.getvalue()

    # -- Helpers -------------------------------------------------------------

    def _lexer(self, code: str, language: Optional[str], filename: Optional[str]):
        try:
            if language:
                return get_lexer_by_name(language, stripnl=False)
            if filename:
                return get_lexer_for_filename(filename, stripnl=False)
            return guess_lexer(code, stripnl=False)
        except ClassNotFound:
            return TextLexer(stripnl=False)


# ---------------------------------------------------------------------------
# Public: skin-aware code theme configuration
# ---------------------------------------------------------------------------

def set_code_theme(theme_name: str) -> bool:
    """Apply a named Pygments theme to the syntax highlighter.

    Called from the CLI when the active skin provides a ``code_theme`` colour key
    (e.g. ``"monokai"``, ``"dracula"``).  Returns True on success.

    An empty *theme_name* (or ``"default"``) reverts to the built-in palette.
    """
    if not theme_name or theme_name == "default":
        _PygmentsToRich.reset_to_default()
        return True
    return _PygmentsToRich.load_theme(theme_name)


# ---------------------------------------------------------------------------
# Diff renderer helpers (module-level for testability)
# ---------------------------------------------------------------------------

def _parse_diff_filename(path: str, fallback: Optional[str] = None) -> str:
    """Return the basename from a unified-diff path string.

    Strips ``b/`` / ``a/`` prefixes produced by ``git diff``.  If the result
    is ``/dev/null`` (deleted-file diff), recurses on *fallback* (the ``---``
    path) instead.
    """
    for prefix in ("b/", "a/"):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    if path == "/dev/null":
        if fallback:
            return _parse_diff_filename(fallback)
        return "?"
    name = Path(path).name
    return name if name else path


def _count_pass(
    lines: list[str],
    explicit_filename: Optional[str] = None,
) -> list[tuple[Optional[str], int, int]]:
    """First pass over diff lines: build ``(filename, n_adds, n_dels)`` per file boundary.

    When *explicit_filename* is provided (from ``from_content()``), the
    filename is fixed and only one entry is produced.  Otherwise filenames are
    parsed from ``+++ `` lines.
    """
    entries: list[tuple[Optional[str], int, int]] = []
    current_file: Optional[str] = explicit_filename
    n_adds = n_dels = 0
    from_path: Optional[str] = None
    started = explicit_filename is not None

    for line in lines:
        if line.startswith("--- "):
            from_path = line[4:].strip()
        elif line.startswith("+++ "):
            if started:
                entries.append((current_file, n_adds, n_dels))
            to_path = line[4:].strip()
            if explicit_filename is None:
                current_file = _parse_diff_filename(to_path, from_path)
            n_adds = n_dels = 0
            started = True
        elif line.startswith("+"):
            n_adds += 1
        elif line.startswith("-"):
            n_dels += 1

    if started:
        entries.append((current_file, n_adds, n_dels))

    return entries


def _make_header(filename: Optional[str], n_adds: int, n_dels: int) -> tuple[Text, Text]:
    """Return ``(header_Text, separator_Text)`` for the diff summary line."""
    def _pl(n: int) -> str:
        return f"{n} line" if n == 1 else f"{n} lines"

    parts: list[Text] = [
        Text("● ", style="bright_white"),
        Text(filename or "?", style=Style(color="bright_white", bold=True)),
        Text("   "),
    ]
    if n_adds > 0 and n_dels == 0:
        parts.append(Text(f"Added {_pl(n_adds)}", style="green"))
    elif n_dels > 0 and n_adds == 0:
        parts.append(Text(f"Removed {_pl(n_dels)}", style="red"))
    elif n_adds > 0 and n_dels > 0:
        parts.append(Text(f"Added {_pl(n_adds)}", style="green"))
        parts.append(Text(f", removed {_pl(n_dels)}", style="red"))

    header = Text.assemble(*parts)
    separator = Text("─" * len(header.plain), style="dim")
    return header, separator


def _syntax_text(content: str, filename: Optional[str]) -> Text:
    """Return a Rich ``Text`` with Pygments syntax colours (foreground only).

    ``syntax_highlighter`` is resolved at call time so this helper can be
    defined before the module-level instance is created.
    Falls back to plain unstyled text on any error.

    Pygments always appends a trailing newline token to its output.  We strip
    it when the source *content* itself did not end with ``\\n`` so that diff
    line lengths remain accurate.
    """
    try:
        markup = syntax_highlighter.to_markup(content, filename=filename or "")
        text = Text.from_markup(markup)
        if text.plain.endswith("\n") and not content.endswith("\n"):
            text = text[: len(text.plain) - 1]
        return text
    except Exception:
        return Text(content)


def _flat_del(ln: int, content: str, filename: Optional[str] = None) -> Text:
    """Render a deletion line with syntax highlighting and diff background."""
    syn = _syntax_text(content, filename)
    syn.stylize(Style(bgcolor=_DIFF_BG_DEL))
    return Text.assemble(
        Text(f"{ln:>4} ", style=Style(dim=True, bgcolor=_DIFF_BG_DEL)),
        Text("- ", style=Style(color="white", bold=True, bgcolor=_DIFF_BG_DEL)),
        syn,
    )


def _flat_add(ln: int, content: str, filename: Optional[str] = None) -> Text:
    """Render an addition line with syntax highlighting and diff background."""
    syn = _syntax_text(content, filename)
    syn.stylize(Style(bgcolor=_DIFF_BG_ADD))
    return Text.assemble(
        Text(f"{ln:>4} ", style=Style(dim=True, bgcolor=_DIFF_BG_ADD)),
        Text("+ ", style=Style(color="white", bold=True, bgcolor=_DIFF_BG_ADD)),
        syn,
    )


def _intra_diff(
    old: str, new: str, filename: Optional[str] = None
) -> tuple[list[Text], list[Text]]:
    """Character-level diff between two line content strings.

    Returns ``([del_text], [add_text])`` — single-element lists for API
    compatibility with the ``Text.assemble(*segments)`` call sites.

    Syntax colours are applied to the foreground; diff backgrounds are applied
    as a separate layer so they never conflict with token colours:

    * Equal regions: syntax fg + dark diff background.
    * Changed regions: syntax fg + **bright** diff background (bold), which
      visually highlights the change without clobbering syntax colours.
    """
    del_text = _syntax_text(old, filename)
    add_text = _syntax_text(new, filename)

    # Base diff backgrounds across the full lines.
    del_text.stylize(Style(bgcolor=_DIFF_BG_DEL))
    add_text.stylize(Style(bgcolor=_DIFF_BG_ADD))

    # Brighter background on changed character ranges (overrides base bg).
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            del_text.stylize(Style(bgcolor=_DIFF_BG_DEL_HL, bold=True), i1, i2)
        if tag in ("replace", "insert"):
            add_text.stylize(Style(bgcolor=_DIFF_BG_ADD_HL, bold=True), j1, j2)

    return [del_text], [add_text]


# ---------------------------------------------------------------------------
# Public: diff renderer
# ---------------------------------------------------------------------------

_DIFF_MAX_LINES: int = 80  # cap for DiffRenderer.to_lines; 0 = unlimited

class DiffRenderer:
    """Render a unified diff as Rich Text objects with line numbers.

    Produces coloured ``+`` / ``-`` lines with green / red backgrounds and
    dim context lines — significantly richer than raw ANSI strings.
    """

    # -- From old/new strings ------------------------------------------------

    def from_content(
        self,
        old: str,
        new: str,
        file_path: str = "file",
        context_lines: int = 3,
    ) -> Group:
        """Generate and render a diff between *old* and *new*."""
        import difflib

        lines = list(difflib.unified_diff(
            old.splitlines(keepends=False),
            new.splitlines(keepends=False),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=context_lines,
            lineterm="",
        ))
        return self._style(lines, file_path=file_path)

    # -- From unified diff text ----------------------------------------------

    def from_unified(self, diff_text: str) -> Group:
        """Render an already-generated unified diff string."""
        return self._style(diff_text.splitlines())

    # -- ANSI lines (drop-in for _render_inline_unified_diff) ----------------

    def to_lines(self, diff_text: str, width: int = 0,
                 max_lines: int = _DIFF_MAX_LINES) -> list[str]:
        """Render *diff_text* and return a list of ANSI-escaped strings.

        Compatible with Hermes's ``print_fn`` pattern: each element maps to
        one ``print_fn(line)`` call.

        *max_lines* caps the output list and appends a dim footer for any
        omitted lines.  Pass ``max_lines=0`` to disable truncation (useful
        for callers that apply their own budget, e.g.
        ``_summarize_rendered_diff_sections``).
        """
        buf = StringIO()
        Console(file=buf, highlight=False, force_terminal=True, width=width).print(
            self.from_unified(diff_text)
        )
        # Drop the trailing empty line that Console adds
        lines = buf.getvalue().rstrip("\n").splitlines()
        if max_lines and len(lines) > max_lines:
            omitted = len(lines) - max_lines
            footer = (
                f"\033[2m   ╌╌ {omitted} more line"
                f"{'s' if omitted != 1 else ''} omitted ╌╌\033[0m"
            )
            return lines[:max_lines] + [footer]
        return lines

    # -- Internal rendering --------------------------------------------------

    def _style(self, lines: list[str], file_path: Optional[str] = None) -> Group:
        """Render *lines* (from a unified diff) as a ``Group`` of Rich ``Text``.

        *file_path* — when supplied (from ``from_content()``), its basename is
        used for the summary header instead of parsing the ``+++ `` line.
        """
        styled: list[Text] = []

        # Pass 1 — count adds/dels per file boundary for the summary header.
        explicit_filename = Path(file_path).name if file_path else None
        file_entries = iter(_count_pass(lines, explicit_filename))

        # Pass 2 — render with run-based pairing and intra-line highlighting.
        ln_old = ln_new = 0
        from_path: Optional[str] = None
        del_run: list[tuple[int, str]] = []  # (ln_old, content)
        add_run: list[tuple[int, str]] = []

        def flush_runs() -> None:
            """Pair del/add runs and emit highlighted (or flat) Text objects."""
            if not del_run and not add_run:
                return
            n_pairs = min(len(del_run), len(add_run))
            fname = explicit_filename or from_path

            # Precompute intra-diff segments for each pair.
            pair_segs: list[tuple[Optional[list[Text]], Optional[list[Text]]]] = []
            for i in range(n_pairs):
                old_content = del_run[i][1]
                new_content = add_run[i][1]
                r = SequenceMatcher(None, old_content, new_content).ratio()
                if r >= _INTRA_DIFF_MIN_RATIO:
                    d, a = _intra_diff(old_content, new_content, fname)
                    pair_segs.append((d, a))
                else:
                    pair_segs.append((None, None))

            for i, (ln_old_saved, content) in enumerate(del_run):
                # Paired deletions share the addition's new-file line number so
                # del and add lines at the same logical position show the same
                # number.  Unpaired deletions (no corresponding addition) fall
                # back to their old-file line number so the display stays
                # monotonic and correct even when context lines split a del block.
                ln = add_run[i][0] if i < n_pairs else ln_old_saved
                if i < n_pairs and pair_segs[i][0] is not None:
                    styled.append(Text.assemble(
                        Text(f"{ln:>4} ", style=Style(dim=True, bgcolor=_DIFF_BG_DEL)),
                        Text("- ", style=Style(color="white", bold=True, bgcolor=_DIFF_BG_DEL)),
                        *pair_segs[i][0],
                    ))
                else:
                    styled.append(_flat_del(ln, content, fname))

            for i, (ln, content) in enumerate(add_run):
                if i < n_pairs and pair_segs[i][1] is not None:
                    styled.append(Text.assemble(
                        Text(f"{ln:>4} ", style=Style(dim=True, bgcolor=_DIFF_BG_ADD)),
                        Text("+ ", style=Style(color="white", bold=True, bgcolor=_DIFF_BG_ADD)),
                        *pair_segs[i][1],
                    ))
                else:
                    styled.append(_flat_add(ln, content, fname))

            del_run.clear()
            add_run.clear()

        for line in lines:
            if line.startswith("--- "):
                flush_runs()
                from_path = line[4:].strip()
                continue

            if line.startswith("+++ "):
                flush_runs()
                entry = next(file_entries, None)
                if entry:
                    fname, n_adds, n_dels = entry
                    header, sep = _make_header(fname, n_adds, n_dels)
                    styled.append(header)
                    styled.append(sep)
                continue

            if line.startswith("@@"):
                flush_runs()
                m = re.search(r"@@ -(\d+),?\d* \+(\d+),?\d* @@", line)
                if m:
                    ln_old, ln_new = int(m.group(1)), int(m.group(2))
                styled.append(Text(line, style=Style(color="cyan", bold=True)))
                continue

            if line.startswith("-"):
                if add_run:
                    # -→+→- transition: flush current run and start fresh
                    flush_runs()
                del_run.append((ln_old, line[1:]))
                ln_old += 1
                continue

            if line.startswith("+"):
                add_run.append((ln_new, line[1:]))
                ln_new += 1
                continue

            # Context line — show new-file line number (matches GitHub/delta convention
            # and avoids duplicate numbers when old/new offsets diverge)
            flush_runs()
            content = line[1:] if line.startswith(" ") else line
            syn = _syntax_text(content, explicit_filename or from_path)
            syn.stylize(Style(dim=True))
            styled.append(Text.assemble(
                Text(f"{ln_new:>4} ", style="dim"),
                Text("  ", style="dim"),
                syn,
            ))
            ln_old += 1
            ln_new += 1

        flush_runs()  # end of input
        styled.append(Text(""))  # trailing blank line
        return Group(*styled)


# ---------------------------------------------------------------------------
# Public: inline markdown → ANSI rendering
# ---------------------------------------------------------------------------

_MD_CODE_RE = re.compile(r"`([^`\n]+)`")
# Bold+italic must be matched before bold/italic individually
_MD_BOLD_ITALIC_STAR_RE = re.compile(r"\*{3}(.+?)\*{3}")
_MD_BOLD_ITALIC_UNDER_RE = re.compile(r"(?<![_\w])___(.+?)___(?![_\w])")
_MD_BOLD_STAR_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_BOLD_UNDER_RE = re.compile(r"(?<![_\w])__(.+?)__(?![_\w])")
_MD_ITALIC_STAR_RE = re.compile(r"\*([^*\n]+?)\*")
_MD_ITALIC_UNDER_RE = re.compile(r"(?<![_\w])_([^_\n]+)_(?![_\w])")
_MD_STRIKE_RE = re.compile(r"~~(.+?)~~")
# Images must be matched before links (![  prefix overlaps with [)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_MD_LINK_RE = re.compile(r"(?<!\x1b)\[([^\]]+)\]\(([^)]+)\)")
# Bare URLs — negative lookbehind (?<!\() avoids matching URLs already
# formatted as "text (url)" by the link step above.
# Matches https?://, ftp?s://, file:// and bare www. domains.
_MD_BARE_URL_RE = re.compile(
    r"(?<!\()"
    r"(?:(?:https?|ftps?|file)://[^\s\x1b<>\[\]()\"]+|(?<![./\w])www\.[^\s\x1b<>\[\]()\"]+)"
)

# HTML wrapper tags (may contain inner markdown — processed with reset_suffix)
_MD_U_RE = re.compile(r"<u>(.*?)</u>", re.IGNORECASE | re.DOTALL)
_MD_INS_RE = re.compile(r"<ins>(.*?)</ins>", re.IGNORECASE | re.DOTALL)
_MD_MARK_RE = re.compile(r"<mark>(.*?)</mark>", re.IGNORECASE | re.DOTALL)
# HTML inline tags (simple — no nested markdown processing needed)
_MD_EM_RE = re.compile(r"<em>(.*?)</em>", re.IGNORECASE)
_MD_I_RE = re.compile(r"<i>(.*?)</i>", re.IGNORECASE)
_MD_STRONG_RE = re.compile(r"<strong>(.*?)</strong>", re.IGNORECASE)
_MD_B_RE = re.compile(r"<b>(.*?)</b>", re.IGNORECASE)
_MD_S_RE = re.compile(r"<s>(.*?)</s>", re.IGNORECASE)
_MD_STRIKE_TAG_RE = re.compile(r"<strike>(.*?)</strike>", re.IGNORECASE)
_MD_DEL_RE = re.compile(r"<del>(.*?)</del>", re.IGNORECASE)
_MD_CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.IGNORECASE)
_MD_KBD_RE = re.compile(r"<kbd>(.*?)</kbd>", re.IGNORECASE)
# Tags with no terminal equivalent — content is preserved, tags stripped
_MD_STRIP_TAGS_RE = re.compile(r"</?(?:sup|sub)>", re.IGNORECASE)

_MD_BOLD_ANSI = "\033[1m"
_MD_ITALIC_ANSI = "\033[3m"
_MD_BOLD_ITALIC_ANSI = "\033[1;3m"
_MD_STRIKE_ANSI = "\033[9m"
_MD_CODE_ANSI = "\033[97m"
_MD_U_ANSI = "\033[4m"
_MD_MARK_ANSI = "\033[7m"
_MD_LINK_ANSI = "\033[38;2;88;166;255m\033[4m"  # #58A6FF (GitHub dark-mode blue) + underline
_MD_RST_ANSI = "\033[0m"


def apply_inline_markdown(line: str, reset_suffix: str = "", ref_map: "dict[str, str] | None" = None) -> str:
    """Apply ANSI styling to inline markdown spans in a single text line.

    Handles ``**bold**``, ``__bold__``, ``*italic*``, ``_italic_``,
    ``**bold**``, ``__bold__``, ``*italic*``, ``_italic_``,
    ``***bold italic***``, ``___bold italic___``, ``~~strikethrough~~``,
    `` `code` ``, ``<u>``, ``<ins>``, ``<mark>``, ``<em>``, ``<i>``,
    ``<strong>``, ``<b>``, ``<s>``, ``<strike>``, ``<del>``, ``<code>``,
    ``<kbd>``.  ``<sup>``/``<sub>`` tags are stripped (no ANSI equivalent).
    Backtick spans are processed first and protected from later passes via
    placeholder tokens.

    HTML wrapper tags (``<u>``, ``<ins>``, ``<mark>``) are processed before
    markdown spans via a recursive call with the wrapper style as
    ``reset_suffix``, so inner bold/italic resets restore the outer
    underline/highlight rather than dropping it.

    ``reset_suffix`` is appended after each closing reset; pass the active
    response-text ANSI colour here so it is restored between adjacent spans
    during streaming.

    Returns *line* unchanged if it already contains ANSI escape codes.
    """
    if "\x1b" in line:
        return line

    rst = _MD_RST_ANSI + reset_suffix

    # Step 0: HTML wrapper tags — process content recursively with the wrapper
    # style as reset_suffix so inner resets restore the outer style.
    def _wrap(style: str) -> "re.Callable[[re.Match], str]":  # type: ignore[type-arg]
        def _sub(m: re.Match) -> str:  # type: ignore[type-arg]
            inner = apply_inline_markdown(m.group(1), reset_suffix=style, ref_map=ref_map)
            return f"{style}{inner}{rst}"
        return _sub

    line = _MD_U_RE.sub(_wrap(_MD_U_ANSI), line)
    line = _MD_INS_RE.sub(_wrap(_MD_U_ANSI), line)
    line = _MD_MARK_RE.sub(_wrap(_MD_MARK_ANSI), line)

    # Step 1: protect backtick code spans with index placeholders so later
    # passes cannot match * or _ inside them.
    protected: list[str] = []

    def _protect_code(m: re.Match) -> str:  # type: ignore[type-arg]
        protected.append(f"{_ANSI_INLINE_CODE_START}`{m.group(1)}`{rst}")
        return f"\x00{len(protected) - 1}\x00"

    line = _MD_CODE_RE.sub(_protect_code, line)

    # Steps 2–5 use _span() so that nested spans inside a bold/italic/strike
    # delimiter are rendered recursively.  This prevents the _MD_ITALIC_UNDER_RE
    # lookbehind ((?<![_\w])) from seeing the trailing 'm' of an enclosing ANSI
    # code (e.g. \033[1m) as a word character and silently skipping the match.
    # Guard: if the captured content already contains \x1b (ANSI from step 0),
    # pass it through unchanged to avoid double-processing.
    def _span(ansi: str) -> "Callable[[re.Match], str]":  # type: ignore[type-arg]
        def _sub(m: re.Match) -> str:  # type: ignore[type-arg]
            inner = m.group(1)
            if "\x1b" not in inner:
                inner = apply_inline_markdown(inner, reset_suffix=ansi + reset_suffix, ref_map=ref_map)
            return f"{ansi}{inner}{rst}"
        return _sub

    # Step 2: bold+italic (must precede bold and italic individually)
    line = _MD_BOLD_ITALIC_STAR_RE.sub(_span(_MD_BOLD_ITALIC_ANSI), line)
    line = _MD_BOLD_ITALIC_UNDER_RE.sub(_span(_MD_BOLD_ITALIC_ANSI), line)

    # Step 3: bold
    line = _MD_BOLD_STAR_RE.sub(_span(_MD_BOLD_ANSI), line)
    line = _MD_BOLD_UNDER_RE.sub(_span(_MD_BOLD_ANSI), line)

    # Step 4: italic (runs after bold so ** is already consumed)
    line = _MD_ITALIC_STAR_RE.sub(_span(_MD_ITALIC_ANSI), line)
    line = _MD_ITALIC_UNDER_RE.sub(_span(_MD_ITALIC_ANSI), line)

    # Step 5: strikethrough
    line = _MD_STRIKE_RE.sub(_span(_MD_STRIKE_ANSI), line)

    # Step 6a: images (before links — ![  prefix overlaps)
    line = _MD_IMAGE_RE.sub(lambda m: f"\033[2m[img: {m.group(1)}]\033[0m{reset_suffix}", line)

    # Step 6a2: reference link resolution (before inline link step)
    if ref_map:
        def _resolve_coll(m: re.Match) -> str:  # type: ignore[type-arg]
            """[text][] — use text as lookup key."""
            text_part = m.group(1)
            url = ref_map.get(text_part.lower())
            if url:
                return f"{_MD_LINK_ANSI}{text_part} ({url})\033[0m{reset_suffix}"
            return m.group(0)

        def _resolve_use(m: re.Match) -> str:  # type: ignore[type-arg]
            """[text][ref] — use ref as lookup key."""
            text_part = m.group(1)
            ref_key = m.group(2).lower()
            url = ref_map.get(ref_key)
            if url:
                return f"{_MD_LINK_ANSI}{text_part} ({url})\033[0m{reset_suffix}"
            return m.group(0)

        # [text][] collapsed ref — must run before [text][ref] to avoid partial match
        line = _MD_REF_LINK_COLL_RE.sub(_resolve_coll, line)
        line = _MD_REF_LINK_USE_RE.sub(_resolve_use, line)

    # Step 6b: links — bright-blue underline + URL for copy/ctrl+click
    line = _MD_LINK_RE.sub(lambda m: f"{_MD_LINK_ANSI}{m.group(1)} ({m.group(2)})\033[0m{reset_suffix}", line)

    # Step 6b2: bare URLs (https?://...) — style the same as markdown links.
    # Trailing punctuation characters are stripped from the URL and re-appended
    # so "See https://x.com." doesn't include the period in the styled span.
    def _bare_url(m: re.Match) -> str:  # type: ignore[type-arg]
        url = m.group(0).rstrip(".,;:!?)")
        tail = m.group(0)[len(url):]
        return f"{_MD_LINK_ANSI}{url}\033[0m{reset_suffix}{tail}"

    line = _MD_BARE_URL_RE.sub(_bare_url, line)

    # Step 6c: HTML inline tags (simple — content taken as-is)
    _h = reset_suffix  # shorthand
    line = _MD_EM_RE.sub(lambda m: f"{_MD_ITALIC_ANSI}{m.group(1)}{rst}", line)
    line = _MD_I_RE.sub(lambda m: f"{_MD_ITALIC_ANSI}{m.group(1)}{rst}", line)
    line = _MD_STRONG_RE.sub(lambda m: f"{_MD_BOLD_ANSI}{m.group(1)}{rst}", line)
    line = _MD_B_RE.sub(lambda m: f"{_MD_BOLD_ANSI}{m.group(1)}{rst}", line)
    line = _MD_S_RE.sub(lambda m: f"{_MD_STRIKE_ANSI}{m.group(1)}{rst}", line)
    line = _MD_STRIKE_TAG_RE.sub(lambda m: f"{_MD_STRIKE_ANSI}{m.group(1)}{rst}", line)
    line = _MD_DEL_RE.sub(lambda m: f"{_MD_STRIKE_ANSI}{m.group(1)}{rst}", line)
    line = _MD_CODE_TAG_RE.sub(lambda m: f"{_MD_CODE_ANSI}{m.group(1)}{rst}", line)
    line = _MD_KBD_RE.sub(lambda m: f"{_MD_CODE_ANSI}{m.group(1)}{rst}", line)

    # Step 6d: tags with no terminal equivalent — strip tags, keep content
    line = _MD_STRIP_TAGS_RE.sub("", line)

    # Step 7: restore protected code spans
    for idx, span in enumerate(protected):
        line = line.replace(f"\x00{idx}\x00", span)

    # Step 7: strip CommonMark backslash escapes (\] → ], \* → *, etc.)
    line = re.sub(r'\\([\\`*_{}\[\]()#+\-.!|~])', r'\1', line)

    return line


# ---------------------------------------------------------------------------
# Public: block-level markdown → ANSI rendering
# ---------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_MD_HR_RE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")
_MD_BLOCKQUOTE_RE = re.compile(r"^>+\s?(.*)")
_MD_BQ_LEVEL_RE = re.compile(r"^((?:>\s*)+)(.*)")
_MD_UL_RE = re.compile(r"^(\s*)([-*+])\s+(.+)")
_MD_OL_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.+)")
_MD_TASK_RE = re.compile(r"^\[( |x|X)\]\s*(.*)", re.IGNORECASE)
_MD_REF_LINK_RE = re.compile(r"^\[[^\]]+\]:\s+\S+")
_REF_DEF_RE = re.compile(r'^\[([^\]]+)\]:\s*(\S+)(?:\s+(?:"[^"]*"|\'[^\']*\'|\([^)]*\)))?\s*$')
_MD_REF_LINK_USE_RE = re.compile(r'\[([^\]]+)\]\[([^\]]*)\]')
_MD_REF_LINK_COLL_RE = re.compile(r'\[([^\]]+)\]\[\]')

_HEADING_STYLES = {
    1: "\033[1;97m",
    2: "\033[1;37m",
    3: "\033[1m",
    4: "\033[1;2m",
    5: "\033[1;2m",
    6: "\033[1;2m",
}
_BLOCKQUOTE_ANSI = "\033[2m"
_BULLETS = ["•", "◦", "▸", "·"]


def apply_block_line(line: str, reset_suffix: str = "") -> str:
    """Apply ANSI styling to block-level markdown structures in a single line.

    Handles headings (h1–h6), horizontal rules, blockquotes, unordered lists,
    and reference link suppression.  Ordered lists are passed through unchanged.

    Two early-exit guards:
    - Lines containing ``\\x1b`` are already ANSI-rendered — returned as-is.
    - Lines containing ``\\n`` are multi-line blocks from ``StreamingBlockBuffer``
      (table or setext) — returned as-is.

    ``reset_suffix`` is forwarded to every inner ``apply_inline_markdown`` call
    so that inline span resets (e.g. code-span ``\\033[0m``) restore the outer
    style (e.g. dim for reasoning blocks) instead of falling back to plain text.

    Returns *line* unchanged if no block pattern matches.
    """
    if "\x1b" in line:
        return line
    if "\n" in line:
        return line

    # Reference link definition — suppress entirely
    if _MD_REF_LINK_RE.match(line):
        return ""

    # Headings
    m = _MD_HEADING_RE.match(line)
    if m:
        level = len(m.group(1))
        text = m.group(2)
        style = _HEADING_STYLES.get(level, "\033[1;2m")
        rendered_text = apply_inline_markdown(text, reset_suffix=style)
        return f"{style}{rendered_text}{_MD_RST_ANSI}"

    # Horizontal rule
    stripped = line.rstrip()
    if _MD_HR_RE.match(stripped):
        cols = shutil.get_terminal_size((80, 24)).columns
        return f"\033[2m{'─' * cols}\033[0m"

    # Blockquote — render with depth-aware gutter
    m = _MD_BQ_LEVEL_RE.match(line)
    if m:
        raw_prefix = m.group(1)
        content = m.group(2)
        depth = raw_prefix.count('>')
        indent = "  " * (depth - 1)
        dim_prefix = "\033[2m" * min(depth - 1, 2)
        ansi = dim_prefix + _BLOCKQUOTE_ANSI
        content_rendered = apply_inline_markdown(content, reset_suffix=ansi)
        return f"{indent}{ansi}▌ {content_rendered}\033[0m"

    # Unordered list — bullet symbol by indent depth
    m = _MD_UL_RE.match(line)
    if m:
        indent, _marker, content = m.group(1), m.group(2), m.group(3)
        level = len(indent) // 2
        bullet = _BULLETS[min(level, len(_BULLETS) - 1)]
        # Task list detection
        tm = _MD_TASK_RE.match(content)
        if tm:
            checkbox_char, rest = tm.group(1), tm.group(2)
            if checkbox_char.lower() == 'x':
                checkbox_sym = f"\033[1;32m✓\033[0m{reset_suffix}"
            else:
                checkbox_sym = f"\033[2m○\033[0m{reset_suffix}"
            rest_rendered = apply_inline_markdown(rest, reset_suffix=reset_suffix)
            return f"{indent}{bullet} {checkbox_sym} {rest_rendered}"
        return f"{indent}{bullet} {apply_inline_markdown(content, reset_suffix=reset_suffix)}"

    # Ordered list — dim numeral, then content
    m = _MD_OL_RE.match(line)
    if m:
        indent, numeral, content = m.group(1), m.group(2), m.group(3)
        level = len(indent) // 2
        _ = level  # reserved for future indent-aware styling
        return f"{indent}\033[2m{numeral}.\033[0m{reset_suffix} {apply_inline_markdown(content, reset_suffix=reset_suffix)}"

    return line


# ---------------------------------------------------------------------------
# Stateful block rendering: setext headings, blockquote continuation, tables
# ---------------------------------------------------------------------------

_SETEXT_H1_RE = re.compile(r"^={2,}\s*$")
_SETEXT_H2_RE = re.compile(r"^-{2,}\s*$")
_TABLE_STRICT_ROW_RE = re.compile(r"^\|.+\|\s*$")   # pipes at both ends (strict GFM)
_TABLE_LOOSE_ROW_RE  = re.compile(r"^[^|].+\|")      # no leading pipe, contains | (loose GFM)
_TABLE_SEP_RE        = re.compile(r"^[\s:\-|]+$")     # separator row (dashes/colons/pipes)
_SEP_CELL_RE = re.compile(r"^[\s:-]+$")
_NUM_RE = re.compile(r"^-?[\d,]+\.?\d*$")
_ANSI_ESC_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visual_len(s: str) -> int:
    """Terminal column width of *s* (ANSI codes stripped, wide/emoji chars = 2 cols).

    Wide characters (east_asian_width W/F) count as 2.  U+FE0F (emoji
    presentation selector) upgrades the preceding neutral char to 2-wide,
    matching the behaviour of modern terminal emulators.
    """
    plain = _ANSI_ESC_RE.sub("", s)
    total = 0
    prev_width = 0
    for ch in plain:
        cp = ord(ch)
        if cp == 0xFE0F:  # emoji presentation selector — upgrade preceding char
            if prev_width == 1:
                total += 1
            prev_width = 0
            continue
        w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        total += w
        prev_width = w
    return total


def _split_row(raw: str) -> list[str]:
    """Split a raw pipe-row into cell strings.

    Handles both strict GFM (``| A | B |``) and loose GFM (``A | B | C``)
    formats — leading and trailing ``|`` are stripped when present.
    """
    s = raw.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return s.split("|")


def _parse_align(cell: str) -> str:
    c = cell.strip()
    if c.startswith(":") and c.endswith(":"):
        return "centre"
    if c.endswith(":"):
        return "right"
    return "left"


_MD_OL_START_RE = re.compile(r"^\s*\d+[.)]")


def _is_heading_candidate(pending: Optional[str]) -> bool:
    if pending is None or pending == "" or "\x1b" in pending:
        return False
    # Ordered-list items look like "1. text" or "1) text" — never a setext heading.
    if _MD_OL_START_RE.match(pending):
        return False
    return apply_block_line(pending) is pending


def _render_table(rows: list[list[str]], sep_idx: Optional[int], align: list[str], cols: int, framed: bool = False) -> str:
    if not rows:
        return ""
    # Apply inline markdown to every data cell so ANSI styling is accounted for
    # before measuring visual widths.  Separator rows are kept raw (replaced by
    # a divider line and never inspected for content).
    rendered_rows: list[list[str]] = []
    for i, row in enumerate(rows):
        if i == sep_idx:
            rendered_rows.append(row)
        else:
            rendered_rows.append([
                apply_inline_markdown(row[j].strip()) if j < len(row) else ""
                for j in range(cols)
            ])
    data_rows = [r for i, r in enumerate(rendered_rows) if i != sep_idx]
    widths = [
        max((_visual_len(row[i]) for row in data_rows if i < len(row)), default=0)
        for i in range(cols)
    ]
    align = list(align) + ["left"] * (cols - len(align))

    def _padded(cell: str, w: int, a: str) -> str:
        raw = _ANSI_ESC_RE.sub("", cell).strip()
        pad = w - _visual_len(cell)
        if a == "right" or _NUM_RE.match(raw):
            return " " * pad + cell
        if a == "centre":
            lpad = pad // 2
            return " " * lpad + cell + " " * (pad - lpad)
        return cell + " " * pad

    if framed:
        def _hline(l: str, m: str, r: str) -> str:
            return l + m.join("─" * (w + 2) for w in widths) + r

        content = [(i, r) for i, r in enumerate(rendered_rows) if i != sep_idx]
        out = [_hline("┌", "┬", "┐")]
        for idx, (_, row) in enumerate(content):
            cells_str = "│".join(
                f" {_padded(row[i] if i < len(row) else '', widths[i], align[i])} "
                for i in range(cols)
            )
            out.append(f"│{cells_str}│")
            if idx < len(content) - 1:
                out.append(_hline("├", "┼", "┤"))
        out.append(_hline("└", "┴", "┘"))
        return "\n".join(out)
    else:
        out = []
        for r_idx, row in enumerate(rendered_rows):
            if r_idx == sep_idx:
                out.append(" " + "  ".join("─" * w for w in widths))
                continue
            out.append(" " + "  ".join(
                _padded(row[i] if i < len(row) else "", widths[i], align[i])
                for i in range(cols)
            ))
        return "\n".join(out)


def render_stateful_blocks(text: str) -> str:
    """Pass 2: render setext headings, blockquote continuation lines, and tables.

    Runs a single left-to-right scan.  Skips lines that already contain
    ``\\x1b`` (highlighted code from pass 1).
    """
    # Pre-pass: collect reference link definitions into ref_map
    ref_map: dict[str, str] = {}
    for raw_line in text.splitlines():
        rm = _REF_DEF_RE.match(raw_line.strip())
        if rm:
            ref_map[rm.group(1).lower()] = rm.group(2)

    lines = text.splitlines()
    out: list = []

    _pending: Optional[str] = None
    _bq_depth: int = 0  # 0 = not in blockquote; >0 = current depth
    _in_ol: bool = False
    _ol_indent: int = 0
    _table_rows: list = []
    _sep_idx: Optional[int] = None
    _align: list = []
    _table_strict: bool = False

    def _emit(s: str) -> None:
        out.append(s)

    def _flush_pending() -> None:
        nonlocal _pending
        if _pending is not None:
            # If pending is a BQ line, render it with the gutter
            pm = _MD_BQ_LEVEL_RE.match(_pending)
            if pm:
                _emit(_render_bq_depth(pm.group(2), pm.group(1).count('>')))
            else:
                _emit(_pending)
            _pending = None

    def _render_bq_depth(content: str, depth: int) -> str:
        indent = "  " * (depth - 1)
        dim_prefix = "\033[2m" * min(depth - 1, 2)
        ansi = dim_prefix + _BLOCKQUOTE_ANSI
        content_rendered = apply_inline_markdown(content, reset_suffix=ansi, ref_map=ref_map)
        return f"{indent}{ansi}▌ {content_rendered}\033[0m"

    def _on_table_row(raw: str) -> None:
        nonlocal _sep_idx, _align, _table_strict
        if not _table_rows:  # first row is the header — determines strict vs loose
            _table_strict = bool(_TABLE_STRICT_ROW_RE.match(raw))
        header_cols = len(_split_row(_table_rows[0])) if _table_rows else 0
        cells = _split_row(raw)
        if _sep_idx is None and cells and all(_SEP_CELL_RE.match(c) for c in cells):
            _sep_idx = len(_table_rows)
            _align = [_parse_align(c) for c in cells]
            _align += ["left"] * (header_cols - len(_align))
        _table_rows.append(raw)

    def _flush_table_to_out() -> None:
        nonlocal _sep_idx, _align, _table_strict
        if not _table_rows:
            return
        rows = [_split_row(r) for r in _table_rows]
        cols = len(rows[0]) if rows else 0
        rendered = _render_table(rows, _sep_idx, _align, cols, framed=_table_strict)
        _table_rows.clear()
        _sep_idx = None
        _align = []
        _table_strict = False
        for tl in rendered.splitlines():
            _emit(tl)

    for line in lines:
        # Priority 1: ANSI line — flush any open table, emit immediately.
        # _pending is intentionally left untouched (spec).
        # If inside a blockquote, keep the gutter so the code block is visually
        # contained within the quote; _bq_depth stays and exits on next blank line.
        if "\x1b" in line:
            _flush_table_to_out()
            if _bq_depth:
                _emit(f"{_BLOCKQUOTE_ANSI}▌ {_MD_RST_ANSI}{line}")
            else:
                _bq_depth = 0
                _emit(line)
            continue

        # Priority 2: blockquote continuation
        if _bq_depth:
            if line == "":
                # Flush any pending BQ line before exiting
                if _pending is not None and _MD_BQ_LEVEL_RE.match(_pending):
                    pm = _MD_BQ_LEVEL_RE.match(_pending)
                    _emit(_render_bq_depth(pm.group(2), pm.group(1).count('>')))
                    _pending = None
                _bq_depth = 0
                _emit(line)
            else:
                bm = _MD_BQ_LEVEL_RE.match(line)
                if bm:
                    depth = bm.group(1).count('>')
                    inner = bm.group(2)
                    # Feature 4: setext heading inside blockquote
                    if _pending is not None and _MD_BQ_LEVEL_RE.match(_pending):
                        pm = _MD_BQ_LEVEL_RE.match(_pending)
                        pending_inner = pm.group(2)  # type: ignore[union-attr]
                        if (_SETEXT_H1_RE.match(inner) or _SETEXT_H2_RE.match(inner)) and _is_heading_candidate(pending_inner):
                            level = 1 if _SETEXT_H1_RE.match(inner) else 2
                            style = _HEADING_STYLES[level]
                            rendered_text = apply_inline_markdown(pending_inner, reset_suffix=style, ref_map=ref_map)
                            heading_out = f"{style}{rendered_text}{_MD_RST_ANSI}"
                            pending_depth = pm.group(1).count('>')  # type: ignore[union-attr]
                            pending_indent = "  " * (pending_depth - 1)
                            dim_prefix = "\033[2m" * min(pending_depth - 1, 2)
                            ansi = dim_prefix + _BLOCKQUOTE_ANSI
                            _pending = None
                            _emit(f"{pending_indent}{ansi}▌ {heading_out}\033[0m")
                            _bq_depth = depth
                            continue
                        # Not setext: flush pending BQ line, buffer new one
                        _flush_pending()
                    _bq_depth = depth
                    _pending = line  # buffer for next setext check
                else:
                    # Continuation (non-BQ line): flush any pending BQ line first
                    if _pending is not None and _MD_BQ_LEVEL_RE.match(_pending):
                        pm = _MD_BQ_LEVEL_RE.match(_pending)
                        _emit(_render_bq_depth(pm.group(2), pm.group(1).count('>')))
                        _pending = None
                    _emit(_render_bq_depth(line, _bq_depth))
            continue

        # Priority 3: table accumulation
        if _table_rows:
            # Accept strict rows always; accept loose rows (no leading pipe) once
            # the separator has been seen — after that any pipe-bearing line is a
            # data row.  Blank lines or pipe-free lines end the table.
            if _TABLE_STRICT_ROW_RE.match(line) or (_sep_idx is not None and "|" in line):
                _on_table_row(line)
                continue
            else:
                _flush_table_to_out()
                # fall through to process this non-table line normally

        # Priority 3b: OL continuation
        if _in_ol:
            if line == "":
                _in_ol = False
            elif _MD_OL_RE.match(line):
                # New OL item — check indent vs current _ol_indent
                om = _MD_OL_RE.match(line)
                item_indent = len(om.group(1))  # type: ignore[union-attr]
                if item_indent >= _ol_indent or item_indent > 0:
                    # Still part of list (same or deeper indent), pass through
                    pass
                else:
                    _in_ol = False
            elif not line.startswith(" " * max(_ol_indent, 1)):
                # Continuation lines must be indented at least to marker column
                _in_ol = False

        # Priority 4: normal mode
        bm = _MD_BQ_LEVEL_RE.match(line)
        if bm:
            _flush_pending()
            depth = bm.group(1).count('>')
            inner = bm.group(2)
            _bq_depth = depth
            # Setext-in-blockquote lookahead: store raw line as pending
            _pending = line
            continue

        if _TABLE_STRICT_ROW_RE.match(line):
            # If the pending line already contains pipes it is the loose table
            # header that preceded this strict row — rescue it instead of
            # emitting it as plain prose.
            if _pending is not None and "|" in _pending:
                _on_table_row(_pending)
                _pending = None
            else:
                _flush_pending()
            _on_table_row(line)
            continue

        # Loose table separator (no leading pipe, e.g. "---|---|---" or "--- --- ---").
        # Current line must look like a separator; pending line must be a loose header.
        if _pending is not None and "|" in _pending and "-" in line and _TABLE_SEP_RE.match(line.strip()):
            _loose_cells = _split_row(line)
            if _loose_cells and all(_SEP_CELL_RE.match(c) for c in _loose_cells):
                _on_table_row(_pending)
                _pending = None
                _on_table_row(line)
                continue

        # Setext marker check
        if _SETEXT_H1_RE.match(line) or _SETEXT_H2_RE.match(line):
            if _is_heading_candidate(_pending):
                level = 1 if _SETEXT_H1_RE.match(line) else 2
                style = _HEADING_STYLES[level]
                rendered_text = apply_inline_markdown(_pending, reset_suffix=style, ref_map=ref_map)  # type: ignore[arg-type]
                heading_out = f"{style}{rendered_text}{_MD_RST_ANSI}"
                _pending = None
                _emit(heading_out)
            else:
                _flush_pending()
                _emit(line)
            continue

        # OL start — track state
        om = _MD_OL_RE.match(line)
        if om:
            _in_ol = True
            _ol_indent = len(om.group(1))

        # Plain line — setext lookahead (one-tick delay)
        _flush_pending()
        _pending = line

    # End of input
    _flush_table_to_out()
    # Flush any pending blockquote line (was waiting for setext check)
    if _pending is not None and _bq_depth and _MD_BQ_LEVEL_RE.match(_pending):
        pm = _MD_BQ_LEVEL_RE.match(_pending)
        depth = pm.group(1).count('>')  # type: ignore[union-attr]
        inner = pm.group(2)  # type: ignore[union-attr]
        _emit(_render_bq_depth(inner, depth))
        _pending = None
    _flush_pending()

    result = "\n".join(out)
    if text.endswith("\n"):
        result += "\n"
    return result


class StreamingBlockBuffer:
    """State machine for stateful block rendering in the streaming pipeline.

    Inserted before ``StreamingCodeBlockHighlighter`` in the streaming loop.
    Handles setext headings (one-tick lookahead), multi-line blockquote
    continuation, and table buffering/rendering.
    """

    def __init__(self) -> None:
        self._pending: Optional[str] = None
        self._bq_depth: int = 0  # 0 = not in blockquote; >0 = current depth
        self._in_ol: bool = False
        self._ol_indent: int = 0
        self._table_buf: list = []
        self._sep_idx: Optional[int] = None
        self._align: list = []
        self._table_strict: bool = False
        self._emit_next: Optional[str] = None
        self._ref_map: dict[str, str] = {}

    def reset(self) -> None:
        """Reset all state for a new response turn."""
        self._pending = None
        self._bq_depth = 0
        self._in_ol = False
        self._ol_indent = 0
        self._table_buf = []
        self._sep_idx = None
        self._align = []
        self._table_strict = False
        self._emit_next = None
        self._ref_map = {}

    def process_line(self, line: str) -> Optional[str]:
        """Process one line.

        Returns the string to emit (may be multi-line ANSI for tables/setexts),
        or ``None`` while accumulating a block.  Plain lines are returned with
        the same object identity as the input so the ``out is line`` identity
        check downstream still works.
        """
        # Priority 1: _emit_next is set — pop and process it; if it resolves to
        # non-None, defer the current line so it's handled on the next call.
        if self._emit_next is not None:
            emit_line = self._emit_next
            self._emit_next = None
            result = self._handle_line(emit_line)
            if result is not None:
                self._emit_next = line
                return result
            # emit_line was buffered (e.g. a table row) — fall through to process line

        return self._handle_line(line)

    def flush(self) -> Optional[str]:
        """Flush any buffered state at end of stream."""
        parts = []
        if self._emit_next is not None:
            emit_line = self._emit_next
            self._emit_next = None
            result = self._handle_line(emit_line)
            if result is not None:
                parts.append(result)
        if self._table_buf:
            parts.append(self._flush_table_str())
        if self._pending is not None:
            # If pending is a blockquote line, render it now
            if self._bq_depth and _MD_BQ_LEVEL_RE.match(self._pending):
                pm = _MD_BQ_LEVEL_RE.match(self._pending)
                depth = pm.group(1).count('>')  # type: ignore[union-attr]
                inner = pm.group(2)  # type: ignore[union-attr]
                parts.append(self._render_bq_depth(inner, depth))
            else:
                parts.append(self._pending)
            self._pending = None
        if parts:
            return "\n".join(parts)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_line(self, line: str) -> Optional[str]:
        """Core state machine: priorities 2–4."""
        # Collect reference link definitions as they arrive (streaming pre-pass)
        rm = _REF_DEF_RE.match(line.strip())
        if rm:
            self._ref_map[rm.group(1).lower()] = rm.group(2)

        # Priority 2: blockquote continuation
        if self._bq_depth:
            if "\x1b" in line:
                # Rare: raw ANSI in stream while in blockquote — keep gutter
                # Flush any pending BQ line first
                if self._pending is not None:
                    pm = _MD_BQ_LEVEL_RE.match(self._pending)
                    if pm:
                        inner = pm.group(2)
                        depth = pm.group(1).count('>')
                        old = self._pending
                        self._pending = None
                        self._emit_next = line
                        return self._render_bq_depth(inner, depth)
                return f"{_BLOCKQUOTE_ANSI}▌ {_MD_RST_ANSI}{line}"
            if line == "":
                # Flush pending BQ line before exiting blockquote
                if self._pending is not None:
                    pm = _MD_BQ_LEVEL_RE.match(self._pending)
                    if pm:
                        inner = pm.group(2)
                        depth = pm.group(1).count('>')
                        self._pending = None
                        self._bq_depth = 0
                        self._emit_next = line
                        return self._render_bq_depth(inner, depth)
                self._bq_depth = 0
                return line
            # Code fence — exit blockquote so StreamingCodeBlockHighlighter
            # can handle it normally (gutter on the fence itself isn't possible
            # once the line passes to the code highlighter)
            if line.strip().startswith("```"):
                if self._pending is not None:
                    pm = _MD_BQ_LEVEL_RE.match(self._pending)
                    if pm:
                        inner = pm.group(2)
                        depth = pm.group(1).count('>')
                        self._pending = None
                        self._bq_depth = 0
                        self._emit_next = line
                        return self._render_bq_depth(inner, depth)
                self._bq_depth = 0
                return line
            bm = _MD_BQ_LEVEL_RE.match(line)
            if bm:
                depth = bm.group(1).count('>')
                inner = bm.group(2)
                # Feature 4: setext heading inside blockquote
                # Check if pending is a BQ line and current inner is setext
                if self._pending is not None and _MD_BQ_LEVEL_RE.match(self._pending):
                    pm = _MD_BQ_LEVEL_RE.match(self._pending)
                    pending_inner = pm.group(2)  # type: ignore[union-attr]
                    if (_SETEXT_H1_RE.match(inner) or _SETEXT_H2_RE.match(inner)) and _is_heading_candidate(pending_inner):
                        level = 1 if _SETEXT_H1_RE.match(inner) else 2
                        style = _HEADING_STYLES[level]
                        rendered_text = apply_inline_markdown(pending_inner, reset_suffix=style, ref_map=self._ref_map)
                        heading_out = f"{style}{rendered_text}{_MD_RST_ANSI}"
                        pending_depth = pm.group(1).count('>')  # type: ignore[union-attr]
                        pending_indent = "  " * (pending_depth - 1)
                        dim_prefix = "\033[2m" * min(pending_depth - 1, 2)
                        ansi = dim_prefix + _BLOCKQUOTE_ANSI
                        self._pending = None
                        self._bq_depth = depth
                        return f"{pending_indent}{ansi}▌ {heading_out}\033[0m"
                # Flush old pending BQ line, then buffer this new one for setext lookahead
                if self._pending is not None and _MD_BQ_LEVEL_RE.match(self._pending):
                    pm = _MD_BQ_LEVEL_RE.match(self._pending)
                    old_inner = pm.group(2)  # type: ignore[union-attr]
                    old_depth = pm.group(1).count('>')  # type: ignore[union-attr]
                    rendered = self._render_bq_depth(old_inner, old_depth)
                    self._pending = line
                    self._bq_depth = depth
                    return rendered
                self._bq_depth = depth
                self._pending = line
                return None  # buffered for setext lookahead
            # Continuation (non-BQ line while in blockquote)
            # Flush any pending BQ line first
            if self._pending is not None and _MD_BQ_LEVEL_RE.match(self._pending):
                pm = _MD_BQ_LEVEL_RE.match(self._pending)
                inner = pm.group(2)  # type: ignore[union-attr]
                depth = pm.group(1).count('>')  # type: ignore[union-attr]
                rendered = self._render_bq_depth(inner, depth)
                self._pending = None
                self._emit_next = line
                return rendered
            return self._render_bq_depth(line, self._bq_depth)

        # Priority 3: table accumulation
        if self._table_buf:
            if _TABLE_STRICT_ROW_RE.match(line) or (self._sep_idx is not None and "|" in line):
                self._on_table_row(line)
                return None
            else:
                rendered = self._flush_table_str()
                self._emit_next = line
                return rendered

        # Priority 3b: OL continuation tracking
        if self._in_ol:
            if line == "":
                self._in_ol = False
            elif _MD_OL_RE.match(line):
                om = _MD_OL_RE.match(line)
                item_indent = len(om.group(1))  # type: ignore[union-attr]
                if item_indent < self._ol_indent and item_indent == 0:
                    self._in_ol = False
            elif not line.startswith(" " * max(self._ol_indent, 1)):
                self._in_ol = False

        # Priority 4: normal mode
        # Blockquote start
        bm = _MD_BQ_LEVEL_RE.match(line)
        if bm:
            depth = bm.group(1).count('>')
            if self._pending is not None:
                result = self._pending
                self._pending = None
                self._emit_next = line
                self._bq_depth = depth
                return result
            self._bq_depth = depth
            # Buffer the first BQ line for setext-in-blockquote lookahead
            self._pending = line
            return None

        # Table row start
        if _TABLE_STRICT_ROW_RE.match(line):
            if self._pending is not None and "|" in self._pending:
                # Pending line is a loose table header — rescue it.
                self._on_table_row(self._pending)
                self._pending = None
                self._on_table_row(line)
                return None
            elif self._pending is not None:
                result = self._pending
                self._pending = None
                self._emit_next = line
                return result
            self._on_table_row(line)
            return None

        # Loose table separator (no leading pipe, e.g. "---|---|---" or "--- --- ---").
        if self._pending is not None and "|" in self._pending and "-" in line and _TABLE_SEP_RE.match(line.strip()):
            _loose_cells = _split_row(line)
            if _loose_cells and all(_SEP_CELL_RE.match(c) for c in _loose_cells):
                self._on_table_row(self._pending)
                self._pending = None
                self._on_table_row(line)
                return None

        # Setext marker
        if _SETEXT_H1_RE.match(line) or _SETEXT_H2_RE.match(line):
            if _is_heading_candidate(self._pending):
                level = 1 if _SETEXT_H1_RE.match(line) else 2
                style = _HEADING_STYLES[level]
                rendered_text = apply_inline_markdown(self._pending, reset_suffix=style, ref_map=self._ref_map)  # type: ignore[arg-type]
                heading = f"{style}{rendered_text}{_MD_RST_ANSI}"
                self._pending = None
                return heading
            else:
                old = self._pending
                self._pending = line
                return old  # None if nothing was pending

        # OL start — track state
        om = _MD_OL_RE.match(line)
        if om:
            self._in_ol = True
            self._ol_indent = len(om.group(1))

        # Plain line (or ANSI when _pending is None — return immediately)
        if "\x1b" in line and self._pending is None:
            return line

        old = self._pending
        self._pending = line
        return old  # None if _pending was None

    def _render_bq_depth(self, content: str, depth: int) -> str:
        indent = "  " * (depth - 1)
        dim_prefix = "\033[2m" * min(depth - 1, 2)
        ansi = dim_prefix + _BLOCKQUOTE_ANSI
        content_rendered = apply_inline_markdown(content, reset_suffix=ansi, ref_map=self._ref_map)
        return f"{indent}{ansi}▌ {content_rendered}\033[0m"

    def _render_bq(self, content: str) -> str:
        return self._render_bq_depth(content, max(self._bq_depth, 1))

    def _on_table_row(self, raw: str) -> None:
        if not self._table_buf:  # first row is the header — determines strict vs loose
            self._table_strict = bool(_TABLE_STRICT_ROW_RE.match(raw))
        header_cols = len(_split_row(self._table_buf[0])) if self._table_buf else 0
        cells = _split_row(raw)
        if self._sep_idx is None and cells and all(_SEP_CELL_RE.match(c) for c in cells):
            self._sep_idx = len(self._table_buf)
            self._align = [_parse_align(c) for c in cells]
            self._align += ["left"] * (header_cols - len(self._align))
        self._table_buf.append(raw)

    def _flush_table_str(self) -> str:
        rows = [_split_row(r) for r in self._table_buf]
        cols = len(rows[0]) if rows else 0
        rendered = _render_table(rows, self._sep_idx, self._align, cols, framed=self._table_strict)
        self._table_buf = []
        self._sep_idx = None
        self._align = []
        self._table_strict = False
        return rendered


# ---------------------------------------------------------------------------
# Code block line numbers
# ---------------------------------------------------------------------------

def _number_code_lines(highlighted: str) -> str:
    """Prepend dim line numbers to each line of a highlighted code block."""
    lines = highlighted.splitlines()
    if not lines:
        return highlighted
    width = len(str(len(lines)))
    out = []
    for i, line in enumerate(lines, 1):
        out.append(f"\033[2m{i:>{width}} \u2502\033[0m {line}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Public: fenced code block highlighting for LLM responses
# ---------------------------------------------------------------------------

# ANSI styling for inline code spans (`like this`):
# dark gray background (256-colour index 237) + bright white text.
_ANSI_INLINE_CODE_START = "\033[48;5;237m\033[97m"
_ANSI_INLINE_CODE_END = "\033[0m"

# Single backtick span: one or more non-backtick, non-newline characters.
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")


def _highlight_inline_code(text: str) -> str:
    """Apply ANSI styling to inline code spans (single backticks) in prose text.

    Preserves the backticks so the boundary is still visible; only applies
    a background/foreground colour change to distinguish code from prose.
    Does NOT touch triple-backtick fenced blocks — callers must ensure
    this is only called on prose segments, not on code block content.
    """
    return _INLINE_CODE_RE.sub(
        lambda m: f"{_ANSI_INLINE_CODE_START}`{m.group(1)}`{_ANSI_INLINE_CODE_END}",
        text,
    )


def _number_code_lines(highlighted: str) -> str:
    """Prepend dim right-justified line numbers to each line of a highlighted code block."""
    lines = highlighted.splitlines()
    if not lines:
        return highlighted
    width = len(str(len(lines)))
    out = []
    for i, line in enumerate(lines, 1):
        out.append(f"\033[2m{i:>{width}} \u2502\033[0m {line}")
    return "\n".join(out)


# Fast-path regex: matches common markdown syntax characters.
# If none are found in the first 500 chars, the text is almost certainly plain
# and we can skip the full markdown parsing pipeline for a faster display.
_MD_SYNTAX_RE = re.compile(
    r'[#*`|>\[~]|\n\n|^\d+\.\s|\n\d+\.\s',
    re.MULTILINE,
)


def has_markdown_syntax(text: str) -> bool:
    """Return True if *text* likely contains markdown that should be rendered.

    Scans only the first 500 characters to keep the check fast. Used as a
    fast-path gate before ``format_response()`` so plain-text responses skip
    the full parsing pipeline.
    """
    sample = text[:500] if len(text) > 500 else text
    return bool(_MD_SYNTAX_RE.search(sample))


def format_response(text: str) -> str:
    """Apply syntax highlighting and markdown rendering to a complete response string.

    Pass 1: replaces each fenced code block with an ANSI-highlighted version.
    Pass 2: ``render_stateful_blocks`` — setext headings, blockquote
    continuation, and tables.
    Pass 3: per non-ANSI line — ``apply_block_line`` then
    ``apply_inline_markdown`` (headings, hr, blockquotes, lists, bold, italic,
    code spans, etc.).


    Suitable for the non-streaming Rich Panel display path.
    """
    _hl = SyntaxHighlighter()
    _det = LanguageDetector()

    _RST = "\033[0m"  # reset — transparent no-op; marks lines as code for pass 2

    def _highlight(m: "re.Match") -> str:
        lang = m.group(2).strip() or None
        code = m.group(3)
        if not lang:
            lang = _det.detect_from_content(code)
        highlighted = _hl.to_ansi(code, language=lang).rstrip("\n")
        # _number_code_lines prepends dim line-number prefix ensuring every
        # line has at least one ANSI escape — safe for pass-2 \x1b detection.
        return _number_code_lines(highlighted)

    # Pre-pass: collect reference link definitions for inline resolution
    ref_map: dict[str, str] = {}
    for raw_line in text.splitlines():
        rm = _REF_DEF_RE.match(raw_line.strip())
        if rm:
            ref_map[rm.group(1).lower()] = rm.group(2)

    # Match fenced code blocks of any depth (3+ backticks); \1 backreference
    # ensures the closing fence uses the same backtick sequence as the opener.
    text = re.sub(r"(?m)^(`{3,})(\w*)\n(.*?)\1", _highlight, text, flags=re.DOTALL)
    # Pass 2: stateful block elements (setext headings, blockquote continuation, tables)
    text = render_stateful_blocks(text)
    # Pass 3: per non-ANSI line — block + inline markdown.
    # Use splitlines() (no keepends) so apply_block_line never receives a trailing
    # \n that its capture groups would silently drop.  Rejoin manually and restore
    # the final newline if the original text ended with one.
    lines = text.splitlines()
    result = "\n".join(
        l if "\x1b" in l else apply_inline_markdown(apply_block_line(l), ref_map=ref_map)
        for l in lines
    )
    if text.endswith("\n"):
        result += "\n"
    return result


class StreamingCodeBlockHighlighter:
    """State machine that syntax-highlights fenced code blocks during streaming.

    Feed lines one at a time with ``process_line()``.  Regular lines are
    returned immediately; lines inside a code block are buffered and the
    entire highlighted block is returned when the closing fence arrives.

    Example usage in a line-emission loop::

        hl = StreamingCodeBlockHighlighter()
        for line in stream_lines:
            out = hl.process_line(line)
            if out is not None:
                emit(out)
        # End of stream — flush any unclosed block
        tail = hl.flush()
        if tail is not None:
            emit(tail)
    """

    # Matches an opening fence: 3+ backticks, optional language hint (word chars)
    _FENCE_OPEN_RE = re.compile(r"^(`{3,})\s*(\w*)$")
    # Matches a closing fence: 3+ backticks, optional trailing whitespace only
    _FENCE_CLOSE_RE = re.compile(r"^(`+)\s*$")

    def __init__(self) -> None:
        self._in_block: bool = False
        self._lang: Optional[str] = None
        self._fence_depth: int = 3  # backtick count of the opening fence
        self._buf: list[str] = []
        self._hl = SyntaxHighlighter()
        self._det = LanguageDetector()

    def process_line(self, line: str) -> Optional[str]:
        """Process one line.

        Returns the string to emit (may be multi-line for a highlighted block),
        or ``None`` to suppress the line (still accumulating a code block).
        """
        stripped = line.strip()

        if not self._in_block:
            m = self._FENCE_OPEN_RE.match(stripped)
            if m:
                self._in_block = True
                self._fence_depth = len(m.group(1))
                self._lang = m.group(2) or None
                self._buf = []
                return None  # suppress opening fence — will re-emit with block
            return line  # prose: caller applies full block + inline markdown pipeline

        # Inside a code block — closing fence: >= fence_depth backticks, nothing else
        m = self._FENCE_CLOSE_RE.match(stripped)
        if m and len(m.group(1)) >= self._fence_depth:
            return self._flush_block()
        self._buf.append(line)
        return None  # still accumulating

    def flush(self) -> Optional[str]:
        """Flush any open (unclosed) code block at end of stream."""
        if self._in_block and self._buf:
            return self._flush_block()
        return None

    def reset(self) -> None:
        """Reset state for a new response turn."""
        self._in_block = False
        self._lang = None
        self._fence_depth = 3
        self._buf = []

    def _flush_block(self) -> str:
        code = "\n".join(self._buf)
        lang = self._lang or self._det.detect_from_content(code)
        highlighted = self._hl.to_ansi(code, language=lang).rstrip("\n")
        self._in_block = False
        self._lang = None
        self._buf = []
        return _number_code_lines(highlighted)


# ---------------------------------------------------------------------------
# Public: output noise cleaning
# ---------------------------------------------------------------------------

_NOISE_SUBSTRINGS = frozenset({
    "/venv/lib/python", "/site-packages/", "langsmith/", "langchain/",
    "__pycache__", "venv/lib/", "site-packages",
    "Traceback (most recent call last)", '  File "/',
})


def clean_command_output(content: str) -> str:
    """Strip venv paths, stacktrace boilerplate, and excessive blank lines.

    Useful for cleaning up ``terminal`` tool results before display.
    """
    out: list[str] = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(s in line for s in _NOISE_SUBSTRINGS):
            continue
        if len(line) > 80 and line.count("/") > 5:
            continue
        if line.startswith("from ") and "import" in line and len(line) > 60:
            continue
        line = re.sub(r"\\n\./([^/]+/)*", "", line)
        line = re.sub(r"\\n/[^/]+/[^/]+/([^/]+)", r" \1", line)
        line = line.replace("\\n", "\n").replace("\n\n\n", "\n\n")
        if line and len(line) > 3:
            out.append(line)

    result = "\n".join(out)
    return re.sub(r"\n\s*\n\s*\n", "\n\n", result).strip()


# ---------------------------------------------------------------------------
# Module-level convenience singletons
# ---------------------------------------------------------------------------

lang_detector = LanguageDetector()
syntax_highlighter = SyntaxHighlighter()
diff_renderer = DiffRenderer()
