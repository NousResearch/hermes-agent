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
clean_command_output    strip venv/stacktrace noise from command output

Internal helpers (module-level, exposed for testing)
-----------------------------------------------------
_intra_diff             character-level segment diff between two line strings
_parse_diff_filename    strip a/ b/ prefixes from unified-diff path headers
"""

from __future__ import annotations

import logging
import os
import re
import shutil
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
    """Convert a Pygments token stream to a Rich markup string."""

    # Built lazily so the class-level dict isn't populated when Pygments is absent
    _STYLES: dict = {}

    @classmethod
    def _ensure_styles(cls) -> None:
        if cls._STYLES or not _PYGMENTS:
            return
        cls._STYLES = {
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
            t = t.parent  # type: ignore[assignment]  # pygments Token hierarchy isn't typed
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
        width = shutil.get_terminal_size((220, 50)).columns
        Console(file=buf, highlight=False, force_terminal=True, width=width).print(markup)
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
# Diff renderer helpers (module-level for testability)
# ---------------------------------------------------------------------------

def _parse_diff_filename(path: str, fallback: Optional[str] = None) -> str:
    """Return a displayable path from a unified-diff path string.

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
    return path or "?"


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
        import shutil
        render_width = width or shutil.get_terminal_size((220, 24)).columns
        buf = StringIO()
        Console(file=buf, highlight=False, force_terminal=True, width=render_width).print(
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
        del_run: list[tuple[int, str]] = []  # (line_number, content)
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

            for i, (ln, content) in enumerate(del_run):
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


def format_response(text: str) -> str:
    """Apply syntax highlighting to fenced code blocks in a complete response string.

    Replaces each `` ```lang\\ncode\\n``` `` block with an ANSI-highlighted
    version.  Inline code spans (single backticks) in prose segments are also
    styled.  Blocks with no language hint use content-based detection.
    Suitable for the non-streaming Rich Panel display path.
    """
    _hl = SyntaxHighlighter()
    _det = LanguageDetector()

    def _highlight_block(m: "re.Match") -> str:
        lang = m.group(2).strip() or None
        code = m.group(3)
        if not lang:
            lang = _det.detect_from_content(code)
        highlighted = _hl.to_ansi(code, language=lang).rstrip("\n")
        return _number_code_lines(highlighted)

    # Match fenced code blocks of any depth (3+ backticks); \1 backreference
    # ensures the closing fence uses the same backtick sequence as the opener.
    # Apply inline-code highlighting only to prose segments between/around blocks.
    fence_re = re.compile(r"(`{3,})\s*([^\s`]*)\n(.*?)\1", re.DOTALL)
    parts: list[str] = []
    last_end = 0
    for m in fence_re.finditer(text):
        parts.append(_highlight_inline_code(text[last_end:m.start()]))
        parts.append(_highlight_block(m))
        last_end = m.end()
    parts.append(_highlight_inline_code(text[last_end:]))
    return "".join(parts)


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

    # Matches an opening fence: 3+ backticks, optional language hint supporting
    # common Markdown info-string punctuation like c++, f#, or shell-session.
    _FENCE_OPEN_RE = re.compile(r"^(`{3,})\s*([^\s`]*)$")
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
            return _highlight_inline_code(line)  # prose: style any inline code spans

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
