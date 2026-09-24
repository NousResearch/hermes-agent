"""Classic-CLI rendering helpers: reasoning-tag stripping, ANSI/skin colours, light-mode detection, markdown/final-content rendering, output-history replay, ``_cprint`` and the panel box/wrap helpers.

Split out of ``cli.py``; ``cli`` re-exports every public name and moved bodies late-bind
cli-level names through ``from cli import ...`` at call time so facade monkeypatch seams hold.
"""

from __future__ import annotations

import functools
import itertools
import os
import re
import shutil
import sys
import textwrap
import threading
import time
from contextlib import contextmanager, suppress
from agent.think_scrubber import THINK_TAG_NAMES
from hermes_cli.banner import format_banner_version_label
from rich.console import Console
from rich.text import Text as _RichText
from typing import Any

def _cli():
    """Late import of the ``cli`` facade: mutable CLI module state (and its test seams) lives there."""
    import cli

    return cli


_REASONING_TAGS = THINK_TAG_NAMES


_TOOL_CALL_TAGS = ("tool_call", "tool_calls", "tool_result", "function_call", "function_calls")


def _strip_reasoning_tags(text: str) -> str:
    """Strip reasoning blocks (closed, unterminated, orphan-close) and leaked tool-call XML from display text.

    Keep in sync with ``agent.agent_runtime_helpers.strip_think_blocks`` and the stream consumer's think-tag sets.

    Also strips tool-call XML blocks some open models leak into visible content (``<tool_call>``,
    ``<function_calls>``, Gemma-style ``<function name="…">…</function>``). Ported from
    openclaw/openclaw#67318.
    """
    from cli import _REASONING_TAGS, _TOOL_CALL_TAGS
    cleaned = text
    for tag in _REASONING_TAGS:
        cleaned = re.sub(rf"<{tag}>.*?</{tag}>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(rf"<{tag}>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(rf"</{tag}>\s*", "", cleaned, flags=re.IGNORECASE)
    for tc_tag in _TOOL_CALL_TAGS:
        cleaned = re.sub(
            rf"<(?:[\w.-]+:)?{tc_tag}\b[^>]*>.*?</(?:[\w.-]+:)?{tc_tag}>\s*",
            "", cleaned, flags=re.DOTALL | re.IGNORECASE,
        )
    # <function name="..."> — boundary + attribute gated to avoid prose false positives.
    cleaned = re.sub(
        r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*<function\b[^>]*\bname\s*=[^>]*>(?:(?:(?!</function>).)*)</function>\s*',
        '', cleaned, flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r'</(?:(?:[\w.-]+:)?(?:tool_call|tool_calls|tool_result|function_call|function_calls|function))>\s*', '', cleaned,
        flags=re.IGNORECASE,
    )
    # Unterminated opener / stray <arg_key>/<arg_value> markup = stream cut
    # mid tool-call serialization (#101899); strip to end of text.
    cleaned = re.sub(
        r'(?:^|\n)[ \t]*<(?:[\w.-]+:)?(?:tool_call|tool_calls|tool_result|function_call|function_calls)\b[^>]*>.*$'
        r'|(?:^|\n)[^\n<]*</?arg_(?:key|value)\b.*$',
        '',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def _assistant_content_as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"]
        return "\n".join(p for p in parts if p)
    return str(content)


def _assistant_copy_text(content: Any) -> str:
    from cli import _assistant_content_as_text, _strip_reasoning_tags
    return _strip_reasoning_tags(_assistant_content_as_text(content))


_ACCENT_ANSI_DEFAULT = "\033[1;38;2;255;215;0m"  # #FFD700 bold fallback


_BOLD = "\033[1m"


_RST = "\033[0m"


_STREAM_PAD = ""  # no indent: leading whitespace pollutes copy/paste


_STREAM_PARTIAL_PREVIEW_LEN = 60  # tail of an unfinished line mirrored into the spinner


def _hex_to_ansi(hex_color: str, *, bold: bool = False) -> str:
    """Convert '#RRGGBB' to a true-color ANSI escape, remapping dark-tuned colors in light mode."""
    from cli import _ACCENT_ANSI_DEFAULT, _maybe_remap_for_light_mode
    hex_color = _maybe_remap_for_light_mode(hex_color)
    try:
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        return f"\033[{'1;' if bold else ''}38;2;{r};{g};{b}m"
    except (ValueError, IndexError):
        return _ACCENT_ANSI_DEFAULT if bold else "\033[38;2;184;134;11m"


_TRUE_RE = re.compile(r"^(1|true|on|yes|y)$")


_FALSE_RE = re.compile(r"^(0|false|off|no|n)$")


_LIGHT_DEFAULT_TERM_PROGRAMS = frozenset()  # Apple_Terminal isn't reliable; require explicit config


def _luminance_from_hex(hex_str: str) -> float | None:
    """Rec.709 luma in [0, 1] for '#RGB'/'#RRGGBB', or None when malformed."""
    s = (hex_str or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6 or not all(c in "0123456789abcdefABCDEF" for c in s):
        return None
    try:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


_DA1_REPLY_RE = re.compile(rb"\x1b\[\?[0-9;]*c")


def _query_osc11_background() -> str | None:
    """Terminal background via OSC 11 as "#RRGGBB", or None.

    Fenced with a DA1 sentinel (``ESC[c``): terminals answer in order and virtually all
    answer DA1, so its reply proves our OSC 11 was processed — otherwise a late reply
    leaks into prompt_toolkit's stdin as typed text. Skipped over SSH (round-trip too
    slow; a late BEL reads as Ctrl+G). A 50 ms drain after TCSAFLUSH catches stragglers.

    After the main read + TCSAFLUSH, a short drain window (50 ms) catches late-arriving bytes that slipped
    past the flush — a race observed on VPS and container terminals under load (#40250).
    """
    from cli import _DA1_REPLY_RE
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    if any(os.environ.get(v) for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return None
    try:
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except Exception:
        return None
    try:
        try:
            tty.setcbreak(fd)
        except Exception:
            return None
        try:
            # One write so the OSC 11 query and DA1 fence cannot reorder.
            sys.stdout.write("\x1b]11;?\x1b\\\x1b[c")
            sys.stdout.flush()
        except Exception:
            return None
        # Read until the DA1 fence closes; the 1s deadline only covers terminals ignoring DA1.
        deadline = time.monotonic() + 1.0
        buf = b""
        while time.monotonic() < deadline:
            r, _, _ = select.select([fd], [], [], deadline - time.monotonic())
            if not r:
                continue
            try:
                chunk = os.read(fd, 64)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            if _DA1_REPLY_RE.search(buf):
                break
        # Reply: \x1b]11;rgb:RRRR/GGGG/BBBB\x1b\\ — components are 1-4 hex digits.
        m = re.search(rb"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", buf)
        if not m:
            return None

        def norm(h: bytes) -> int:
            v = int(h, 16)
            bits = len(h) * 4
            return (v * 255) // ((1 << bits) - 1) if bits else 0
        r, g, b = norm(m.group(1)), norm(m.group(2)), norm(m.group(3))
        return f"#{r:02X}{g:02X}{b:02X}"
    finally:
        # TCSAFLUSH discards unread input, scrubbing a partial reply before prompt_toolkit reads it.
        with suppress(Exception):
            termios.tcsetattr(fd, termios.TCSAFLUSH, old)
        try:
            drain_deadline = time.monotonic() + 0.05
            while time.monotonic() < drain_deadline:
                r, _, _ = select.select([fd], [], [], drain_deadline - time.monotonic())
                if not r or not os.read(fd, 64):
                    break
        except Exception:
            pass


def _heal_cooked_mode_drift(fd: int) -> bool:
    """Re-apply raw mode on *fd* when termios drifted back to cooked (POSIX only).

    A lost ``run_in_terminal`` cooked_mode() restore makes the kernel line-buffer every
    keystroke and the CLI looks dead. Mirrors prompt_toolkit's raw_mode flag surgery in
    place. Returns True when healed; False when already raw or not inspectable.
    """
    try:
        import termios
        attrs = termios.tcgetattr(fd)
    except Exception:
        return False
    lflag = attrs[3]
    if not (lflag & (termios.ICANON | termios.ECHO)):
        return False  # still raw — nothing to do
    attrs[3] = lflag & ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)
    attrs[0] = attrs[0] & ~(termios.IXON | termios.IXOFF | termios.ICRNL | termios.INLCR | termios.IGNCR)
    attrs[6][termios.VMIN] = 1
    try:
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:
        return False
    return True


def _detect_light_mode_uncached() -> bool:
    """The detection ladder documented above; may raise (caller maps errors to dark)."""
    from cli import _FALSE_RE, _LIGHT_DEFAULT_TERM_PROGRAMS, _TRUE_RE, _luminance_from_hex, _query_osc11_background
    for var in ("HERMES_LIGHT", "HERMES_TUI_LIGHT"):
        v = (os.environ.get(var) or "").strip().lower()
        if _TRUE_RE.match(v):
            return True
        if _FALSE_RE.match(v):
            return False
    theme = (os.environ.get("HERMES_TUI_THEME") or "").strip().lower()
    if theme == "light":
        return True
    if theme == "dark":
        return False
    bg_lum = _luminance_from_hex(os.environ.get("HERMES_TUI_BACKGROUND") or "")
    if bg_lum is not None:
        return bg_lum >= 0.5
    last = (os.environ.get("COLORFGBG") or "").strip().split(";")[-1]
    if last.isdigit() and 0 <= int(last) < 16:
        return int(last) in {7, 15}
    bg_color = _query_osc11_background()
    if bg_color:
        lum = _luminance_from_hex(bg_color)
        if lum is not None:
            return lum >= 0.5
    return (os.environ.get("TERM_PROGRAM") or "").strip() in _LIGHT_DEFAULT_TERM_PROGRAMS


# Light-mode equivalents of skin colors unreadable on cream backgrounds. Only colors used
# as STANDALONE foregrounds: ones paired with a dark bg (status bar text on #1a1a2e) would
# become invisible the other direction, hence #C0C0C0/#888888/#555555/#8B8682 are skipped.
_LIGHT_MODE_REMAP: dict[str, str] = {
    "#FFF8DC": "#1A1A1A", "#FFD700": "#9A6B00", "#FFBF00": "#8A5A00", "#B8860B": "#5C4500",
    "#DAA520": "#6B4F00", "#F1E6CF": "#1A1A1A", "#c9d1d9": "#24292F", "#EAF7FF": "#0F1B26",
    "#F5F5F5": "#1A1A1A", "#FFF0D4": "#1A1A1A", "#CD7F32": "#8A4F1A", "#FFEFB5": "#3A2A00",
}


_LIGHT_MODE_REMAP_UPPER = {k.upper(): v for k, v in _LIGHT_MODE_REMAP.items()}


def _maybe_remap_for_light_mode(hex_color: str) -> str:
    """In light mode, remap a dark-tuned color to its higher-contrast equivalent."""
    from cli import _LIGHT_MODE_REMAP_UPPER, _detect_light_mode
    if not _detect_light_mode():
        return hex_color
    if not hex_color or not hex_color.startswith("#"):
        return hex_color
    return _LIGHT_MODE_REMAP_UPPER.get(hex_color.upper(), hex_color)


def _install_skin_light_mode_hook() -> None:
    """Wrap SkinConfig.get_color so EVERY skin color read goes through the light-mode remap. Idempotent."""
    from cli import _maybe_remap_for_light_mode
    try:
        from hermes_cli.skin_engine import SkinConfig  # type: ignore[import]
    except Exception:
        return
    if getattr(SkinConfig, "_hermes_light_mode_hook_installed", False):
        return
    _orig_get_color = SkinConfig.get_color

    def _wrapped_get_color(self, key, fallback=""):
        value = _orig_get_color(self, key, fallback)
        try:
            return _maybe_remap_for_light_mode(value)
        except Exception:
            return value

    SkinConfig.get_color = _wrapped_get_color  # type: ignore[method-assign]
    SkinConfig._hermes_light_mode_hook_installed = True  # type: ignore[attr-defined]


class _SkinAwareAnsi:
    """Lazy ANSI escape resolved from the skin on first use; ``.reset()`` after a ``/skin`` switch."""

    def __init__(self, skin_key: str, fallback_hex: str = "#FFD700", *, bold: bool = False):
        self._skin_key = skin_key
        self._fallback_hex = fallback_hex
        self._bold = bold
        self._cached: str | None = None

    def __str__(self) -> str:
        from cli import _hex_to_ansi
        if self._cached is None:
            try:
                from hermes_cli.skin_engine import get_active_skin
                self._cached = _hex_to_ansi(
                    get_active_skin().get_color(self._skin_key, self._fallback_hex),
                    bold=self._bold,
                )
            except Exception:
                self._cached = _hex_to_ansi(self._fallback_hex, bold=self._bold)
        return self._cached

    def __add__(self, other: str) -> str:
        return str(self) + other

    def __radd__(self, other: str) -> str:
        return other + str(self)

    def reset(self) -> None:
        """Clear cache so the next access re-reads the skin."""
        self._cached = None


_ACCENT = _SkinAwareAnsi("response_border", "#FFD700", bold=True)


# dim+italic attributes (not a hex) so dim text inherits the terminal foreground in both modes.
_DIM = "\x1b[2;3m"


def _tty_wrap(s: str, sgr: str) -> str:
    """Wrap *s* in an SGR attribute when stdout is a real TTY; plain text otherwise."""
    try:
        return f"{sgr}{s}\x1b[0m" if sys.stdout.isatty() else str(s)
    except Exception:
        return str(s)


_b = functools.partial(_tty_wrap, sgr="\x1b[1m")  # bold when stdout is a real TTY


_d = functools.partial(_tty_wrap, sgr="\x1b[2;3m")  # dim-italic when stdout is a real TTY


def _accent_hex() -> str:
    """Return the active skin accent color for legacy CLI output lines."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        return get_active_skin().get_color("ui_accent", "#FFBF00")
    except Exception:
        return "#FFBF00"


def _rich_text_from_ansi(text: str) -> _RichText:
    """Rich Text from ANSI output; literal ``[brackets]`` are not treated as markup."""
    return _RichText.from_ansi(text or "")


def _strip_markdown_syntax(text: str) -> str:
    """Best-effort markdown marker removal for plain-text display."""
    from cli import _rich_text_from_ansi
    plain = _rich_text_from_ansi(text or "").plain
    # HR markers: "-"/"_" runs of 3+, but "*" only when exactly 3 (cron schedules "* * * * *").
    plain = re.sub(r"^\s{0,3}(?:[-_]\s*){3,}$", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s{0,3}(?:\*\s*){3}\s*$", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s{0,3}#{1,6}\s+", "", plain, flags=re.MULTILINE)
    # Blockquotes, lists, and checkboxes are preserved because they carry structure.
    plain = re.sub(r"(```+|~~~+)", "", plain)
    plain = re.sub(r"`([^`]*)`", r"\1", plain)
    plain = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", plain)
    plain = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", plain)
    plain = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)___([^_]+)___(?!\w)", r"\1", plain)
    plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)__([^_]+)__(?!\w)", r"\1", plain)
    # `*emphasis*` only when the inner text is non-whitespace (cron expressions again).
    plain = re.sub(r"\*([^\s*][^*]*?[^\s*])\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", plain)
    plain = re.sub(r"~~([^~]+)~~", r"\1", plain)
    plain = re.sub(r"\n{3,}", "\n\n", plain)
    return plain.strip("\n")


_WINDOWS_PATH_WITH_DOT_SEGMENT_RE = re.compile(r"(?i)(?:\b[a-z]:\\|\\\\)[^\s`]*\\\.[^\s`]*")


def _preserve_windows_dot_segments_for_markdown(text: str) -> str:
    r"""Double the ``\`` before hidden dirs in Windows paths: CommonMark reads ``\.`` as an escaped dot."""
    from cli import _WINDOWS_PATH_WITH_DOT_SEGMENT_RE
    if "\\." not in text:
        return text

    def _protect(match: re.Match[str]) -> str:
        return re.sub(r"(?<!\\)\\(?=\.)", r"\\\\", match.group(0))

    return _WINDOWS_PATH_WITH_DOT_SEGMENT_RE.sub(_protect, text)


def _terminal_columns() -> int:
    try:
        return shutil.get_terminal_size((80, 24)).columns
    except Exception:
        return 80


def _terminal_width_for_streaming() -> int:
    """Display cells inside the streamed response box (small margin for resize races)."""
    from cli import _STREAM_PAD, _terminal_columns
    return max(20, _terminal_columns() - len(_STREAM_PAD) - 2)


def _render_final_assistant_content(text: str, mode: str = "render"):
    """Render final assistant content as markdown, stripped text, or raw text."""
    from cli import _preserve_windows_dot_segments_for_markdown, _rich_text_from_ansi, _strip_markdown_syntax, _terminal_columns, realign_markdown_tables
    from rich.markdown import Markdown

    # 1 border cell each side + margin so resize races don't push a borderline table into soft-wrap.
    panel_width = max(20, _terminal_columns() - 4)

    normalized_mode = str(mode or "render").strip().lower()
    if normalized_mode == "strip":
        # Strip first (inline markdown changes cell width), then re-align padding.
        return _RichText(realign_markdown_tables(_strip_markdown_syntax(text), panel_width))
    if normalized_mode == "raw":
        return _rich_text_from_ansi(text or "")

    # Normalising under-padded tables up front gives narrow-panel fallbacks consistent input.
    plain = _rich_text_from_ansi(text or "").plain
    plain = _preserve_windows_dot_segments_for_markdown(plain)
    plain = realign_markdown_tables(plain, panel_width)
    return Markdown(plain)


def _post_stream_transform_output(response: str, result: dict | None) -> str:
    """Text still to display after a streamed response transform: the suffix, or the whole response when replaced."""
    if not result or not result.get("response_transformed"):
        return ""

    original = result.get("pre_transform_response") or ""
    if original and response.startswith(original):
        return response[len(original):]

    return f"\n[Response transformed after streaming]\n{response}"


def _coerce_output_history_limit(value) -> int:
    try:
        return max(10, int(value))
    except (TypeError, ValueError):
        return 200


def _clear_output_history() -> None:
    _cli()._OUTPUT_HISTORY.clear()
    _set_chrome_floor(None)  # the screen is cleared with it


def _output_history_recording() -> bool:
    return _cli()._OUTPUT_HISTORY_ENABLED and not _cli()._OUTPUT_HISTORY_REPLAYING and not _cli()._OUTPUT_HISTORY_SUPPRESSED


def _record_output_history_entry(entry) -> None:
    from cli import _output_history_recording
    if _output_history_recording():
        _cli()._OUTPUT_HISTORY.append(entry)


class _PaintedLine(str):
    """A recorded output line tagged with the terminal ``width`` it was painted at: a terminal
    that does not reflow keeps the rows it wrapped into then, whatever the width is now."""
    width = None


def _painted_columns():
    """The width the terminal soft-wraps a print at right now, or ``None``."""
    from prompt_toolkit.application import get_app_or_none
    app = get_app_or_none()
    try:
        if app is not None:
            return app.output.get_size().columns
        return os.get_terminal_size(sys.__stdout__.fileno()).columns
    except (AttributeError, OSError, ValueError):
        return None


def _record_output_history(text: str, *, force: bool = False) -> None:
    """Record ``text`` as painted now. ``force`` skips the recording check when the caller
    made it at print-request time (the print itself was deferred to the app loop)."""
    from cli import _output_history_recording
    if force or _output_history_recording():
        width = _painted_columns()
        lines = []
        # One entry per printed line: ``_pt_print`` ends every text with a newline, so "" and a
        # trailing "\n" are blank rows on screen and must count in the replay's row budget.
        for line in str(text).replace("\r", "").split("\n"):
            line = _PaintedLine(line)
            line.width = width
            lines.append(line)
        _cli()._OUTPUT_HISTORY.extend(lines)


_ANSI_SEQUENCE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")


def _ansi_drop_cells(line: str, cells: int) -> str:
    """``line`` without its first ``cells`` visible cells; escape sequences are kept for their styling."""
    from prompt_toolkit.utils import get_cwidth
    out, pos = [], 0
    for match in [*_ANSI_SEQUENCE_RE.finditer(line), None]:
        end = match.start() if match else len(line)
        for ch in line[pos:end]:
            if cells > 0:
                cells -= get_cwidth(ch)
            else:
                out.append(ch)
        if match:
            out.append(match.group())
            pos = match.end()
    return "".join(out)


# TERM names the terminal the CLI talks to, a multiplexer included (tmux, GNU screen), and is set
# by that terminal for its children — decisive whenever it names one. Everything else is inherited
# from whatever started the shell (TMUX, KITTY_WINDOW_ID or TERM_PROGRAM=vscode leak into an st
# or urxvt launched from there), so it only speaks for a generic TERM such as xterm-256color;
# TMUX/STY first there: tmux with `default-terminal xterm-256color` (a common setup) hands its
# panes a generic TERM plus TMUX, while XTERM_VERSION is only inherited from the outer xterm. Then
# XTERM_VERSION: an xterm started from a VTE shell inherits VTE_VERSION but sets XTERM_VERSION.
_REFLOW_TERM_PREFIXES = ("tmux", "screen", "xterm-kitty", "alacritty", "foot", "xterm-ghostty", "wezterm",
                         "contour", "vte")
# TERM is all that survives ssh; these prefixes name terminals that truncate rows in place.
_NO_REFLOW_TERM_PREFIXES = ("linux", "st", "mosh", "vt", "cons", "rxvt")
_MULTIPLEXER_ENV = ("TMUX", "STY")
_NO_REFLOW_ENV = ("XTERM_VERSION",)
_REFLOW_ENV = ("VTE_VERSION", "KITTY_WINDOW_ID", "WT_SESSION", "KONSOLE_VERSION",
               "ALACRITTY_WINDOW_ID", "WEZTERM_PANE", "GHOSTTY_RESOURCES_DIR")
_REFLOW_TERM_PROGRAMS = ("iTerm.app", "Apple_Terminal", "WezTerm", "vscode", "ghostty", "Tabby", "Hyper")


def _terminal_reflows() -> bool | None:
    """Whether the terminal re-wraps the rows it shows when its width changes: ``True``
    (tmux, GNU screen, VTE, kitty, iTerm2, Terminal.app, WezTerm, Alacritty, Windows Terminal —
    a shrink pushes the rows that grew into scrollback), ``False`` (xterm, st, urxvt, mosh, the
    Linux console keep every row in place, truncated) or ``None`` when nothing says (xterm or
    iTerm2 over ssh both look like ``TERM=xterm-256color``)."""
    env = os.environ
    term = env.get("TERM", "").lower()
    if term.startswith(_REFLOW_TERM_PREFIXES):  # before "vt": TERM=vte-256color
        return True
    if term.startswith(_NO_REFLOW_TERM_PREFIXES):
        return False
    if any(env.get(name) for name in _MULTIPLEXER_ENV):
        return True
    if any(env.get(name) for name in _NO_REFLOW_ENV):
        return False
    if any(env.get(name) for name in _REFLOW_ENV) or env.get("TERM_PROGRAM") in _REFLOW_TERM_PROGRAMS:
        return True
    if env.get("LC_TERMINAL") == "iTerm2":  # iTerm2 sets it so that ssh forwards it (LC_*)
        return True
    return None


def _line_rows(line: str, columns: int) -> int:
    """Rows ``line`` fills when the terminal soft-wraps it at ``columns``."""
    from prompt_toolkit.formatted_text import ANSI, fragment_list_width, to_formatted_text
    width = fragment_list_width(to_formatted_text(ANSI(line)))
    return max(1, -(-width // columns)) if columns and columns > 0 else 1


def _output_tail_fitting(lines: list[str], max_rows: int, columns: int, painted: bool = True) -> list[str]:
    """Newest ``lines`` filling ``max_rows`` rows, soft-wrapped at ``columns`` — or, with
    ``painted``, at the width each was painted at, as a terminal that does not reflow still
    shows it. The oldest one may only partly fit: its bottom rows are kept, the rows above
    them are already in scrollback."""
    kept, used = [], 0
    for line in reversed(lines):
        if used >= max_rows:
            break
        cols = (getattr(line, "width", None) if painted else None) or columns
        height = _line_rows(line, cols)
        if used + height > max_rows:
            kept.append(_ansi_drop_cells(line, (height - (max_rows - used)) * cols))
            break
        used += height
        kept.append(line)
    kept.reverse()
    return kept


def _output_history_lines() -> list[str]:
    """The recorded output as the lines a replay paints (callable entries render now)."""
    rendered_lines = []
    for entry in tuple(_cli()._OUTPUT_HISTORY):
        lines = [entry]
        if callable(entry):
            try:
                lines = entry()
            except Exception:
                continue
            if isinstance(lines, str):
                lines = lines.splitlines()
        rendered_lines.extend(line if isinstance(line, str) else str(line) for line in lines)
    return rendered_lines


def _output_history_rows(limit: int, columns: int, painted: bool):
    """Rows the whole recorded output fills (counted as ``_output_tail_fitting`` does), or
    ``None`` when that is ``limit`` rows or more."""
    if not _cli()._OUTPUT_HISTORY_ENABLED:
        return None
    total = 0
    for line in reversed(_output_history_lines()):
        total += _line_rows(line, (getattr(line, "width", None) if painted else None) or columns)
        if total >= limit:
            return None
    return total


def _pt_print_ansi(text: str) -> None:
    """``_pt_print(ANSI(text))``, falling back to ``print`` when stdout is not a real console."""
    from cli import _PT_ANSI, _pt_print
    try:
        _pt_print(_PT_ANSI(text))
    except Exception:
        # NoConsoleScreenBufferError (Windows) / OSError when stdout is e.g. a worker log file.
        with suppress(Exception):
            print(text)


_HELD_PAINTS: list | None = None
_HELD_PAINTS_LOCK = threading.Lock()
_PAINT_SEQ = itertools.count()
# ``(app, gate)`` while the CLI's app runs: ``gate()`` is True when output must wait for a resize
# recovery (see ``CLITerminalMixin._output_waits_for_resize``).
_PAINT_GATE = None


def _set_paint_gate(app, gate) -> None:
    global _PAINT_GATE
    _PAINT_GATE = (app, gate) if gate is not None else None


# Rows from the top of the prompt chrome down to the bottom of the screen, once a refill left
# the chrome's top at a known row (``None``: unknown). prompt_toolkit learns that only through
# CPR, which the CLI leaves off; drawn at least this tall, the chrome keeps reaching the bottom
# row, where ``_transcript_room`` counts the transcript from, also when it shrinks (a narrower
# status bar, a closed modal) — rows left blank below it would be counted as transcript (#95375).
_CHROME_FLOOR = None


def _set_chrome_floor(rows) -> None:
    global _CHROME_FLOOR
    _CHROME_FLOOR = rows


def _chrome_floor():
    return _CHROME_FLOOR


# Rows output scrolled into scrollback while the terminal's width changed under it: a terminal
# that does not reflow may have truncated them first, if it resized before reading all of that
# output. The next refill repaints them too — twice rather than truncated for good (#95375).
_SUSPECT_ROWS = 0


def _add_suspect_rows(rows: int) -> None:
    global _SUSPECT_ROWS
    _SUSPECT_ROWS += max(0, rows)


def _take_suspect_rows() -> int:
    global _SUSPECT_ROWS
    rows, _SUSPECT_ROWS = _SUSPECT_ROWS, 0
    return rows


def _hold_paints() -> None:
    """Hold ``_cprint`` paints until ``_release_paints``, while a resize awaits its recovery.

    A paint erases the prompt chrome from prompt_toolkit's cursor, which is stale once the
    terminal re-wrapped the chrome to its new width: rows of the old chrome would stay in the
    transcript, where the replay cannot account for them (#95375). On a terminal that does not
    reflow, a paint would scroll rows it truncated into scrollback before the refill restores them.
    """
    global _HELD_PAINTS
    with _HELD_PAINTS_LOCK:
        if _HELD_PAINTS is None:
            _HELD_PAINTS = []


def _release_paints() -> None:
    """Paint, in the order they were requested, what ``_hold_paints`` held."""
    global _HELD_PAINTS
    with _HELD_PAINTS_LOCK:
        held, _HELD_PAINTS = _HELD_PAINTS or [], None
    for _seq, paint in sorted(held, key=lambda item: item[0]):
        with suppress(Exception):
            paint()


def _paint_held(seq: int, paint, app=None) -> bool:
    """Queue ``paint`` (requested ``seq``-th) when paints are held — or must be, because the
    terminal's width changed under ``app`` and its recovery has not run yet."""
    gate = _PAINT_GATE
    if gate is not None and gate[0] is app and gate[1]():
        _hold_paints()
    with _HELD_PAINTS_LOCK:
        if _HELD_PAINTS is None:
            return False
        _HELD_PAINTS.append((seq, paint))
        return True


def _cprint(text: str):
    """Print ANSI text through prompt_toolkit's renderer (patch_stdout swallows raw ANSI).

    From a background thread while an Application runs, a direct print races the input
    redraw and gets buried, so those are painted on the app's loop via ``call_soon_threadsafe``.
    """
    from cli import _PT_ANSI, _output_history_recording, _pt_print, _pt_print_ansi, _record_output_history
    recording = _output_history_recording()
    seq = next(_PAINT_SEQ)

    def _painted(paint):
        # Recorded when painted, not when requested: a redraw replaying the history must
        # neither print rows still queued for the loop nor size them at a stale width.
        def _paint():
            if recording:
                _record_output_history(text, force=True)
            paint()
        return _paint
    paint_pt = _painted(lambda: _pt_print(_PT_ANSI(text)))
    paint_fallback = _painted(lambda: _pt_print_ansi(text))

    try:
        from prompt_toolkit.application import get_app_or_none, run_in_terminal
    except Exception:
        paint_pt()
        return

    try:
        app = get_app_or_none()
    except Exception:
        app = None

    if app is None or not getattr(app, "_is_running", False):
        _release_paints()
        paint_fallback()
        return

    import asyncio as _asyncio

    try:
        loop = app.loop  # type: ignore[attr-defined]
    except Exception:
        loop = None
    try:
        # get_running_loop(): get_event_loop() warns from threads with no current loop.
        # Use get_running_loop() instead of get_event_loop() to avoid the DeprecationWarning /
        # RuntimeWarning emitted by Python 3.10+ when get_event_loop() is called from a thread that has no
        # current event loop set (e.g. the process_loop background thread). Fixes #19285.
        current_loop = _asyncio.get_running_loop()
    except Exception:
        current_loop = None
    if loop is None or (current_loop is loop and loop.is_running()):
        if not _paint_held(seq, paint_pt, app):
            paint_pt()
        return

    def _print_now():
        from prompt_toolkit.formatted_text import to_formatted_text
        from prompt_toolkit.renderer import print_formatted_text as _paint_formatted_text
        from prompt_toolkit.styles import Style
        _paint_formatted_text(app.output, to_formatted_text(_PT_ANSI(text)) + [("", "\n")], Style([]))
    paint_now = _painted(_print_now)

    def _schedule():
        if not getattr(app, "_is_running", False):
            paint_fallback()
            return
        if _paint_held(seq, _schedule, app):
            return
        with suppress(Exception):
            pending = getattr(app, "_running_in_terminal_f", None)
            if getattr(app, "_running_in_terminal", False) or (pending is not None and not pending.done()):
                # Another run_in_terminal body owns the terminal: paint after it. Never fall back
                # to a bare print on error: run_in_terminal may already have painted.
                import inspect as _inspect
                coro = run_in_terminal(lambda: _paint_held(seq, _schedule, app) or paint_now())
                if coro is not None and (_inspect.isawaitable(coro) or _inspect.iscoroutine(coro)):
                    _asyncio.ensure_future(coro)
                return
            # What run_in_terminal does, but now: its erase and print would run a loop pass
            # later, when a resize may have landed after the check above (#95375).
            renderer = app.renderer
            floor = _CHROME_FLOOR
            if floor is not None:  # the chrome, as tall as drawn, moves down by the rows printed
                screen = renderer._last_screen
                floor = max(floor, renderer._min_available_height, screen.height if screen else 0)
            columns = app.output.get_size().columns
            renderer.erase()
            paint_now()
            renderer.reset()
            printed = sum(_line_rows(line, columns) for line in text.split("\n"))
            if app.output.get_size().columns != columns:
                _add_suspect_rows(printed)
            if floor is not None:
                _set_chrome_floor(max(0, floor - printed))
            app._request_absolute_cursor_position()
            app._redraw()

    try:
        loop.call_soon_threadsafe(_schedule)
    except Exception:
        paint_fallback()


def _prepend_note_to_message(message, note: str):
    """Prepend a one-shot note to a user message (str, or content-part list when an image is attached).

    For lists the note is folded into the first text part or inserted as a leading one.
    Unknown shapes are returned unchanged.
    """
    note = str(note or "").strip()
    if not note:
        return message
    if isinstance(message, str):
        return f"{note}\n\n{message}" if message else note
    if isinstance(message, list):
        parts = list(message)
        for i, part in enumerate(parts):
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                parts[i] = {**part, "text": f"{note}\n\n{text}" if text else note}
                return parts
        return [{"type": "text", "text": note}, *parts]
    return message


def _pt_app_is_running() -> bool:
    """Whether a prompt_toolkit Application currently owns the live terminal."""
    try:
        from prompt_toolkit.application import get_app_or_none
        app = get_app_or_none()
    except Exception:
        return False
    return app is not None and bool(getattr(app, "_is_running", False))


def _cli_visible_print(text: str = "") -> None:
    """``print`` unless a prompt_toolkit Application owns the terminal (patch_stdout swallows bare prints)."""
    from cli import _cprint, _pt_app_is_running
    if _pt_app_is_running():
        _cprint(text)
    else:
        print(text)


class ChatConsole:
    """Rich Console drop-in routing rendered ANSI through ``_cprint`` so colors survive patch_stdout."""

    def __init__(self):
        from io import StringIO
        self._buffer = StringIO()
        self._inner = Console(file=self._buffer, force_terminal=True, color_system="truecolor", highlight=False)

    def print(self, *args, **kwargs):
        from cli import _OSC_ESCAPE_RE, _cprint
        self._buffer.seek(0)
        self._buffer.truncate()
        self._inner.width = shutil.get_terminal_size((80, 24)).columns
        self._inner.print(*args, **kwargs)
        for line in _OSC_ESCAPE_RE.sub("", self._buffer.getvalue()).rstrip("\n").split("\n"):
            _cprint(line)

    @contextmanager
    def status(self, *_args, **_kwargs):
        """No-op ``console.status`` so slash helpers don't duplicate ``_busy_command()``'s indicator."""
        yield self


def _build_compact_banner() -> str:
    """Build a compact banner that fits the current terminal width."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        _skin = get_active_skin()
    except Exception:
        _skin = None

    def _color(key, default):
        return _skin.get_color(key, default) if _skin else default

    border_color = _color("banner_border", "#FFD700")
    title_color = _color("banner_title", "#FFBF00")
    dim_color = _color("banner_dim", "#B8860B")

    if (getattr(_skin, "name", "default") if _skin else "default") == "default":
        tiny_line = "☤ NOUS HERMES"
    else:
        tiny_line = _skin.get_branding("agent_name", "Hermes Agent") if _skin else "Hermes Agent"
    line1 = f"{tiny_line} - AI Agent Framework"

    if os.environ.get("HERMES_FAST_STARTUP_BANNER") == "1":
        from hermes_cli import __release_date__ as _release_date
        from hermes_cli import __version__ as _version

        version_line = f"Hermes Agent v{_version} ({_release_date})"
    else:
        version_line = format_banner_version_label()

    w = min(shutil.get_terminal_size().columns - 2, 88)
    if w < 30:
        return f"\n[{title_color}]{tiny_line}[/] [dim {dim_color}]- Nous Research[/]\n"

    inner = w - 2  # inside the box border
    bar = "═" * w
    content_width = inner - 2

    line1 = line1[:content_width].ljust(content_width)
    line2 = version_line[:content_width].ljust(content_width)

    return (
        f"\n[bold {border_color}]╔{bar}╗[/]\n"
        f"[bold {border_color}]║[/] [{title_color}]{line1}[/] [bold {border_color}]║[/]\n"
        f"[bold {border_color}]║[/] [dim {dim_color}]{line2}[/] [bold {border_color}]║[/]\n"
        f"[bold {border_color}]╚{bar}╝[/]\n"
    )


def _panel_box_width(title: str, content_lines: list[str], min_width: int = 46, max_width: int = 76) -> int:
    """Stable TUI panel width wide enough for the title and content (incl. borders)."""
    term_cols = shutil.get_terminal_size((100, 20)).columns
    longest = max([len(title)] + [len(line) for line in content_lines] + [min_width - 4])
    inner = min(max(longest + 4, min_width - 2), max_width - 2, max(24, term_cols - 6))
    return inner + 2  # leading/trailing space inside the borders


def _wrap_panel_text(text: str, width: int, subsequent_indent: str = "", *, keep_ws: bool = False) -> list[str]:
    """Wrap panel text; ``keep_ws`` preserves whitespace (command/detail previews)."""
    kw = dict(replace_whitespace=False, drop_whitespace=False) if keep_ws else dict(break_long_words=False, break_on_hyphens=False)
    wrapped = textwrap.wrap(text, width=max(8, width), subsequent_indent=subsequent_indent, **kw)
    return wrapped or [""]


_wrap_panel_text_keep_ws = functools.partial(_wrap_panel_text, keep_ws=True)


def _append_panel_line(lines, border_style: str, content_style: str, text: str, box_width: int) -> None:
    lines.extend(((border_style, "│ "), (content_style, text.ljust(max(0, box_width - 2))), (border_style, " │\n")))


def _append_blank_panel_line(lines, border_style: str, box_width: int) -> None:
    lines.append((border_style, "│" + (" " * box_width) + "│\n"))
