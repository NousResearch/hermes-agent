"""Classic-CLI terminal input helpers: file-drop/attachment path resolution, bracketed-paste patching, extended Enter-key sequences, CPR leak guards and TUI input-height estimation.

Split out of ``cli.py``; ``cli`` re-exports every public name and moved bodies late-bind
cli-level names through ``from cli import ...`` at call time so facade monkeypatch seams hold.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Dict, Any, Optional, Mapping
from urllib.parse import unquote, urlparse

# Log-record parity with the origin module.
logger = logging.getLogger("cli")


def _cli():
    """Late import of the ``cli`` facade: mutable CLI module state (and its test seams) lives there."""
    import cli

    return cli


_IMAGE_EXTENSIONS = frozenset({
    '.png', '.jpg', '.jpeg', '.gif', '.webp',
    '.bmp', '.tiff', '.tif', '.svg', '.ico',
})


def _termux_example_image_path(filename: str = "cat.png") -> str:
    """Return a realistic example media path for the current Termux setup."""
    candidates = [
        os.path.expanduser("~/storage/shared"),
        "/sdcard",
        "/storage/emulated/0",
        "/storage/self/primary",
    ]
    # Literal "/" so the Android hint is right even on Windows.
    for root in candidates:
        if os.path.isdir(root):
            return f"{root}/Pictures/{filename}"
    return f"~/storage/shared/Pictures/{filename}"


def _split_path_input(raw: str) -> tuple[str, str]:
    r"""Split a leading path token (quoted or with ``\ `` escapes) from trailing free-form text."""
    raw = str(raw or "").strip()
    if not raw:
        return "", ""

    if raw[0] in {'"', "'"}:
        quote = raw[0]
        pos = 1
        while pos < len(raw):
            ch = raw[pos]
            if ch == '\\' and pos + 1 < len(raw):
                pos += 2
                continue
            if ch == quote:
                return raw[1:pos], raw[pos + 1 :].strip()
            pos += 1
        return raw[1:], ""

    pos = 0
    while pos < len(raw):
        ch = raw[pos]
        if ch == '\\' and pos + 1 < len(raw) and raw[pos + 1] == ' ':
            pos += 2
        elif ch == ' ':
            break
        else:
            pos += 1

    return raw[:pos].replace('\\ ', ' '), raw[pos:].strip()


def _resolve_attachment_path(raw_path: str) -> Path | None:
    """Resolve a user-supplied attachment path (quotes, ``~``, env vars, ``file://``; relative to TERMINAL_CWD).

    Returns ``None`` unless it resolves to an existing file.
    """
    token = str(raw_path or "").strip()
    if not token:
        return None

    if token[0] == token[-1] and token[0] in {'"', "'"}:
        token = token[1:-1].strip()
    token = token.replace('\\ ', ' ')
    if not token:
        return None

    expanded = token
    if token.startswith("file://"):
        try:
            parsed = urlparse(token)
            if parsed.scheme == "file":
                expanded = unquote(parsed.path or "")
                if parsed.netloc and os.name == "nt":
                    expanded = f"//{parsed.netloc}{expanded}"
                elif os.name == "nt" and len(expanded) >= 3 and expanded[0] == "/" and expanded[1].isalpha() and expanded[2] == ":":
                    # file:///C:/... parses to path "/C:/..." — drop the leading slash
                    # so it resolves as a drive-letter path.
                    expanded = expanded[1:]
        except Exception:
            expanded = token
    expanded = os.path.expandvars(os.path.expanduser(expanded))
    if os.name != "nt":
        normalized = expanded.replace("\\", "/")
        if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/" and normalized[0].isalpha():
            expanded = f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    path = Path(expanded)
    if not path.is_absolute():
        base_dir = Path(os.getenv("TERMINAL_CWD", os.getcwd()))
        path = base_dir / path

    try:
        resolved = path.resolve()
    except Exception:
        resolved = path

    # ENAMETOOLONG for a pasted `/goal <long prose>` that passed the `/` prefilter
    # would otherwise reach process_loop and silently lose the input.
    try:
        if not resolved.exists() or not resolved.is_file():
            return None
    except OSError:
        return None
    return resolved


def _file_drop_result(path: Path, remainder: str) -> dict:
    from cli import _IMAGE_EXTENSIONS
    return {"path": path, "is_image": path.suffix.lower() in _IMAGE_EXTENSIONS, "remainder": remainder}


def _detect_file_drop(user_input: str) -> "dict | None":
    """Detect a dragged/pasted file path at the start of *user_input* -> ``{path, is_image, remainder}`` or None."""
    from cli import _file_drop_result, _resolve_attachment_path, _split_path_input
    if not isinstance(user_input, str):
        return None

    stripped = user_input.strip()
    if not stripped:
        return None

    # Optionally quoted; then /, ~, ./, ../, a Windows drive prefix, or (unquoted) file://.
    quoted = stripped[:1] in {"'", '"'}
    unquoted = stripped[1:] if quoted else stripped
    starts_like_path = (
        unquoted.startswith(("/", "~", "./", "../"))
        or (not quoted and unquoted.startswith("file://"))
        or (len(unquoted) >= 3 and unquoted[1] == ":" and unquoted[2] in {"\\", "/"} and unquoted[0].isalpha())
    )
    if not starts_like_path:
        return None

    direct_path = _resolve_attachment_path(stripped)
    if direct_path is not None:
        return _file_drop_result(direct_path, "")

    first_token, remainder = _split_path_input(stripped)
    drop_path = _resolve_attachment_path(first_token)
    if drop_path is None and " " in stripped and not quoted:
        for pos in reversed([idx for idx, ch in enumerate(stripped) if ch == " "]):
            drop_path = _resolve_attachment_path(stripped[:pos].rstrip())
            if drop_path is not None:
                remainder = stripped[pos + 1 :].strip()
                break
    if drop_path is None:
        return None
    return _file_drop_result(drop_path, remainder)


def _format_image_attachment_badges(attached_images: list[Path], image_counter: int, width: int | None = None) -> str:
    """Attached-image badge row: compact summary on narrow terminals, per-image badges otherwise."""
    if not attached_images:
        return ""

    width = width or shutil.get_terminal_size((80, 24)).columns

    def _trunc(name: str, limit: int) -> str:
        return name if len(name) <= limit else name[: max(1, limit - 3)] + "..."

    if width < 52:
        if len(attached_images) == 1:
            return f"[📎 {_trunc(attached_images[0].name, 20)}]"
        return f"[📎 {len(attached_images)} images attached]"

    if width < 80:
        if len(attached_images) == 1:
            return f"[📎 {_trunc(attached_images[0].name, 32)}]"
        return f"[📎 {_trunc(attached_images[0].name, 20)}] [+{len(attached_images) - 1}]"

    base = image_counter - len(attached_images) + 1
    return " ".join(f"[📎 Image #{base + i}]" for i in range(len(attached_images)))


def _should_auto_attach_clipboard_image_on_paste(pasted_text: str) -> bool:
    """Auto-attach clipboard images only for image-only paste gestures."""
    return not pasted_text.strip()


def _hermes_call_output_screen_diff(
    orig_osd, app, output, screen, current_pos, color_depth, previous_screen, last_style, is_done, full_screen,
    attrs_for_style_string, style_string_has_style, size, previous_width,
):
    """prompt_toolkit ``_output_screen_diff`` with resize guards.

    Inflates ``previous_screen.height`` when the new screen is taller so pt skips the
    cursor move that stamps chrome into scrollback; on a corrupt previous paint buffer
    (tmux re-attach) retries once as a first paint instead of crashing the loop.

    1. 2. On AttributeError/TypeError from a corrupt previous paint buffer (classic after tmux attach with
    same width), retry once with ``previous_screen=None`` so pt first-paints cleanly instead of crashing the
    event loop with ``'cell' object has no attribute 'char'``. See #26137.
    """
    try:
        if previous_screen is not None and hasattr(previous_screen, "height") and previous_screen.height < screen.height:
            previous_screen.height = screen.height
    except Exception:
        pass

    common = (app, output, screen, current_pos, color_depth)
    tail = (is_done, full_screen, attrs_for_style_string, style_string_has_style, size)
    try:
        return orig_osd(*common, previous_screen, last_style, *tail, previous_width)
    except (AttributeError, TypeError):
        # Corrupt previous_screen / row cells after client reattach: previous_screen=None
        # takes the first-paint erase path, previous_width=0 treats the width as changed.
        return orig_osd(*common, None, None, *tail, 0)


def _apply_bracketed_paste_timeout_patch() -> None:
    """Patch ``Vt100Parser.feed`` to flush a bracketed paste whose ESC[201~ end mark never arrives.

    Without it a dropped end mark (SSH glitch, sleep/wake) freezes input forever. Idempotent.
    """
    try:
        import prompt_toolkit.input.vt100_parser as _vt100_mod
        from prompt_toolkit.keys import Keys as _PtKeys
        from prompt_toolkit.key_binding.key_processor import KeyPress as _PtKeyPress

        if getattr(_vt100_mod, "_hermes_bp_timeout_patched", False):
            return

        _BP_TIMEOUT_S = 2.0

        def _patched_vt100_feed(self_parser, data: str) -> None:
            if self_parser._in_bracketed_paste:
                self_parser._paste_buffer += data
                end_mark = "\x1b[201~"

                if end_mark in self_parser._paste_buffer:
                    end_index = self_parser._paste_buffer.index(end_mark)
                    paste_content = self_parser._paste_buffer[:end_index]
                    self_parser.feed_key_callback(_PtKeyPress(_PtKeys.BracketedPaste, paste_content))
                    self_parser._in_bracketed_paste = False
                    remaining = self_parser._paste_buffer[end_index + len(end_mark):]
                    self_parser._paste_buffer = ""
                    self_parser._hermes_bp_start = None
                    if remaining:
                        _patched_vt100_feed(self_parser, remaining)
                else:
                    bp_start = getattr(self_parser, "_hermes_bp_start", None)
                    now = time.monotonic()
                    if bp_start is None:
                        self_parser._hermes_bp_start = now
                    elif now - bp_start > _BP_TIMEOUT_S:
                        paste_content = self_parser._paste_buffer
                        self_parser._in_bracketed_paste = False
                        self_parser._paste_buffer = ""
                        self_parser._hermes_bp_start = None
                        if paste_content:
                            self_parser.feed_key_callback(_PtKeyPress(_PtKeys.BracketedPaste, paste_content))
                            logger.warning(
                                "Bracketed-paste timeout (%.1fs) — flushed %d bytes "
                                "without end mark. Terminal may have dropped ESC[201~ "
                                "(see #16263).",
                                now - bp_start, len(paste_content),
                            )
            else:
                # Re-inlined: calling the original would double-buffer after entering paste mode.
                for i, c in enumerate(data):
                    if self_parser._in_bracketed_paste:
                        _patched_vt100_feed(self_parser, data[i:])
                        break
                    self_parser._input_parser.send(c)

        _vt100_mod.Vt100Parser.feed = _patched_vt100_feed
        _vt100_mod._hermes_bp_timeout_patched = True
        logger.debug("Applied Vt100Parser bracketed-paste timeout patch (#16263)")
    except Exception as exc:  # noqa: BLE001 — defensive: never break startup
        logger.debug("Bracketed-paste timeout patch skipped: %s", exc)


# CPR replies (``ESC[<row>;<col>R``) can race past the input parser under resize storms
# and land as literal text; the ``^[[...R`` form appears when a filter stripped the ESC.
# Cursor Position Report (CPR / DSR) response, format ``ESC[<row>;<col>R``. prompt_toolkit's _on_resize() +
# renderer send ``ESC[6n`` queries to the terminal; under resize storms or tab switches the terminal's reply
# can race past the input parser and end up in the input buffer as literal text (see issue #14692). Also
# matches the visible-form ``^[[<row>;<col>R`` that appears when the ESC byte was stripped by a prior
# filter.
_DSR_CPR_ESC_RE = re.compile(r"\x1b\[\d+;\d+R")


_DSR_CPR_VISIBLE_RE = re.compile(r"\^\[\[\d+;\d+R")


_SGR_MOUSE_ESC_RE = re.compile(r"\x1b\[<\d+;\d+;\d+[Mm]")


_SGR_MOUSE_VISIBLE_RE = re.compile(r"\^\[\[<\d+;\d+;\d+[Mm]")


# Bare "<btn;col;rowM" fragments; deliberately broad, they are almost never intentional input.
_SGR_MOUSE_BARE_RE = re.compile(r"<\d+;\d+;\d+[Mm]")


_TERMINAL_INPUT_MODE_RESET_SEQ = (
    "\x1b[?1006l\x1b[?1003l\x1b[?1002l\x1b[?1000l"  # mouse: SGR, any-motion, button-motion, click
    "\x1b[?1004l"  # focus events
    "\x1b[?2004l"  # bracketed paste
    "\x1b[?1049l"  # leave alt screen
    "\x1b[<u"  # pop kitty keyboard mode
    "\x1b[>4m"  # reset modifyOtherKeys
    "\x1b[0m\x1b[?25h"  # reset attributes, show cursor
)


_KITTY_KEYBOARD_PUSH_SEQ = "\x1b[>1u"


_MODIFY_OTHER_KEYS_SEQ = "\x1b[>4;2m"


_EXTENDED_ENTER_KEYS_SEQ = _KITTY_KEYBOARD_PUSH_SEQ + _MODIFY_OTHER_KEYS_SEQ


_BACKSLASH_LINE_CONTINUATION_RE = re.compile(r"\\[ \t]*$")


def _is_ghostty_terminal(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the terminal is Ghostty.

    Ghostty gets ONLY modifyOtherKeys: its Kitty disambiguate mode strips Alt from
    Backspace (upstream bug), breaking backward-kill-word.

    Ghostty implements modifyOtherKeys correctly (it then emits ``\\x1b[27;3;127~``, which the alias table
    also maps). See #87630.
    """
    env = os.environ if env is None else env
    return (env.get("TERM_PROGRAM") or "").strip() == "ghostty" or (env.get("TERM") or "").strip().lower() == "xterm-ghostty"


def _terminal_supports_extended_enter_keys(env: Optional[Mapping[str, str]] = None) -> bool:
    """Allowlist of terminals where requesting modified-Enter reporting is safe (aligned with the Ink TUI)."""
    env = os.environ if env is None else env
    term_program = (env.get("TERM_PROGRAM") or "").strip()
    term = (env.get("TERM") or "").strip().lower()
    return bool(
        env.get("WT_SESSION")
        or term_program in {"iTerm.app", "WezTerm", "ghostty", "vscode"}
        or env.get("KITTY_WINDOW_ID") or "kitty" in term
        or term == "xterm-ghostty"
        or term.startswith("tmux") or term_program.lower() == "tmux"
    )


def _enable_extended_enter_keys(output=None, env: Optional[Mapping[str, str]] = None) -> bool:
    """Ask allowlisted terminals to report modified keys distinctly.

    Pushes BOTH kitty keyboard protocol and xterm modifyOtherKeys (kitty dropped the
    latter; tmux/VS Code only accept it). Both re-encode modified keys as sequences
    stock prompt_toolkit barely maps (Ctrl+C once arrived as ``ESC[99;5u``), so
    ``install_modify_other_keys_aliases()`` must have run first. Ghostty gets only
    modifyOtherKeys. The exit reset pops both modes.

    Under either protocol the terminal re-encodes modified keys as escape sequences — Kitty disambiguate
    mode as ``ESC[<codepoint>;<mod>u`` (plus the Esc key as ``ESC[27u``), modifyOtherKeys=2 as
    ``ESC[27;<mod>;<codepoint>~``. Stock prompt_toolkit 3.x maps almost none of these, which is why the CSI
    >1u push was temporarily removed in 87074 (Ctrl+C arrived as ``ESC[99;5u`` and died, #56684).
    ``install_modify_other_keys_aliases()`` (called at CLI startup from ``hermes_cli.pt_input_extras``) now
    populates ``ANSI_SEQUENCES`` with the full Ctrl/Alt/Shift/multi-modifier and functional-key tables under
    BOTH formats, so every existing key binding continues to fire — including Ctrl+C, which is handled by
    prompt_toolkit's ``c-c`` binding (raw mode clears ISIG, so the kernel INTR path was never in play for
    the CLI).
    See #87630.
    """
    from cli import _EXTENDED_ENTER_KEYS_SEQ, _MODIFY_OTHER_KEYS_SEQ, _is_ghostty_terminal, _terminal_supports_extended_enter_keys
    if not _terminal_supports_extended_enter_keys(env):
        return False
    seq = _MODIFY_OTHER_KEYS_SEQ if _is_ghostty_terminal(env) else _EXTENDED_ENTER_KEYS_SEQ
    try:
        if output is not None and hasattr(output, "write_raw"):
            output.write_raw(seq)
            output.flush()
            return True
        if sys.stdout is not None and sys.stdout.isatty():
            sys.stdout.write(seq)
            sys.stdout.flush()
            return True
    except Exception:
        pass
    return False


def _cli_multiline_shortcuts_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """``display.cli_multiline_shortcuts`` (default on: Ctrl+J = newline; off restores the legacy c-j submit)."""
    if config is None:
        config = _cli().CLI_CONFIG
    display = config.get("display") if isinstance(config, dict) else None
    value = display.get("cli_multiline_shortcuts", True) if isinstance(display, dict) else True
    if isinstance(value, bool):
        return value
    return not (isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "off", "disabled"})


def _is_backslash_line_continuation(text: str) -> bool:
    """True when Enter should turn a trailing backslash into a newline."""
    from cli import _BACKSLASH_LINE_CONTINUATION_RE
    return bool(_BACKSLASH_LINE_CONTINUATION_RE.search(text or ""))


def _apply_backslash_line_continuation(text: str) -> str:
    """Replace a trailing ``\\`` marker with an actual newline."""
    from cli import _BACKSLASH_LINE_CONTINUATION_RE
    return _BACKSLASH_LINE_CONTINUATION_RE.sub("", text or "") + "\n"


def _preserve_ctrl_enter_newline() -> bool:
    """Environments delivering Ctrl+Enter as bare LF (Windows Terminal, WSL, SSH, Ghostty): c-j must stay newline.

    See issue #22379.
    """
    env = os.environ
    if (
        sys.platform == "win32"
        or any(env.get(v) for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "WT_SESSION",
                                    "GHOSTTY_RESOURCES_DIR", "GHOSTTY_BIN_DIR"))
        or env.get("TERM", "").lower() == "xterm-ghostty" or env.get("TERM_PROGRAM", "").lower() == "ghostty"
        or "microsoft" in env.get("WSL_DISTRO_NAME", "").lower()
    ):
        return True
    # WSL env vars can be scrubbed under sudo; also peek /proc.
    for p in ("/proc/version", "/proc/sys/kernel/osrelease"):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                if "microsoft" in f.read().lower():
                    return True
        except OSError:
            continue
    return False


def _bind_prompt_submit_keys(kb, handler, *, multiline_shortcuts_enabled: Optional[bool] = None) -> None:
    """Enter always submits; c-j submits only with multiline shortcuts off AND where Ctrl+Enter isn't c-j.

    Even when the setting is disabled, environments where Ctrl+Enter is known to arrive as c-j (Windows,
    WSL, SSH, Windows Terminal, Ghostty) keep c-j reserved for newline; otherwise Ctrl+Enter submits instead
    of composing. See _preserve_ctrl_enter_newline() and issue #22379.
    """
    from cli import _cli_multiline_shortcuts_enabled, _preserve_ctrl_enter_newline
    if multiline_shortcuts_enabled is None:
        multiline_shortcuts_enabled = _cli_multiline_shortcuts_enabled()
    kb.add("enter")(handler)
    if sys.platform != "win32" and not multiline_shortcuts_enabled and not _preserve_ctrl_enter_newline():
        kb.add("c-j")(handler)


def _disable_prompt_toolkit_cpr_warning(app) -> None:
    """Let prompt_toolkit fall back from CPR without printing into the prompt."""
    with suppress(Exception):
        app.renderer.cpr_not_supported_callback = None


def _terminal_may_leak_cpr() -> bool:
    """Suppress prompt_toolkit CPR queries (delayed replies leak into input); Windows keeps pt's default.

    Delayed CPR replies (``ESC[<row>;<col>R`` / visible ``^[[<row>;<col>R``) leak into the status line and
    can freeze input when the reply is slow (#13870 on SSH/slow PTYs). The same race hits local POSIX TTYs
    under heavy subagent / status-line load — see ``tests/hermes_cli/test_cpr_local_leak.py``.
    """
    return os.environ.get("PROMPT_TOOLKIT_NO_CPR", "") == "1" or sys.platform != "win32"


def _build_cpr_disabled_output(stdout):
    """Vt100_Output with ``enable_cpr=False`` (``from_pty()`` doesn't expose it), or None on failure.

    prompt_toolkit's renderer sends ``ESC[6n`` (Device Status Report) to learn the cursor row before
    painting in non-fullscreen mode; the terminal replies ``ESC[<row>;<col>R``. When that reply is delayed
    it races into the display as raw ``^[[39;1R`` and can stall the renderer's pending-CPR future (#13870;
    also local POSIX under heavy subagent load).
    """
    try:
        import io as _io
        from prompt_toolkit.output.vt100 import Vt100_Output, _get_size
        from prompt_toolkit.data_structures import Size

        def _get_term_size():
            rows = columns = None
            try:
                rows, columns = _get_size(stdout.fileno())
            except (OSError, _io.UnsupportedOperation, AttributeError, ValueError):
                pass
            return Size(rows=rows or 24, columns=columns or 80)

        return Vt100_Output(stdout, _get_term_size, enable_cpr=False)
    except Exception:
        return None


def _select_classic_cli_pt_output(stdout):
    """CPR-disabled ``Vt100_Output`` when CPR may leak, else None (Application keeps pt's default)."""
    from cli import _build_cpr_disabled_output, _terminal_may_leak_cpr
    return _build_cpr_disabled_output(stdout) if _terminal_may_leak_cpr() else None


def _strip_leaked_terminal_responses_with_meta(text: str) -> tuple[str, bool]:
    """Strip leaked CPR replies and mouse-report fragments -> ``(cleaned, had_mouse_reports)``."""
    from cli import _DSR_CPR_ESC_RE, _DSR_CPR_VISIBLE_RE, _SGR_MOUSE_BARE_RE, _SGR_MOUSE_ESC_RE, _SGR_MOUSE_VISIBLE_RE
    if not text:
        return text, False

    had_mouse_reports = False
    for present, cpr_re, mouse_re in (
        ("\x1b[" in text, _DSR_CPR_ESC_RE, _SGR_MOUSE_ESC_RE),
        ("^[" in text, _DSR_CPR_VISIBLE_RE, _SGR_MOUSE_VISIBLE_RE),
        ("<" in text and ";" in text and ("M" in text or "m" in text), None, _SGR_MOUSE_BARE_RE),
    ):
        if not present:
            continue
        if cpr_re is not None:
            text = cpr_re.sub("", text)
        text, count = mouse_re.subn("", text)
        had_mouse_reports = had_mouse_reports or count > 0
    return text, had_mouse_reports


def _estimate_tui_input_height(
    lines: list[str] | tuple[str, ...], prompt_text: str, terminal_columns: int, *, max_height: int = 8,
) -> int:
    """Input rows from live terminal cells; the BeforeInput prompt consumes cells only on line 0.

    Never substitute a fake wide fallback: a mis-sized TextArea leaves stale cells at the bottom.
    """
    from cli import _int_or
    try:
        from prompt_toolkit.utils import get_cwidth
    except Exception:
        get_cwidth = lambda value: len(value or "")  # type: ignore[assignment]

    columns = max(1, _int_or(terminal_columns or 0, 0))
    prompt_width = max(0, get_cwidth(prompt_text or ""))

    visual_lines = 0
    for index, line in enumerate(lines or [""]):
        display_width = get_cwidth(line or "") + (prompt_width if index == 0 else 0)
        visual_lines += max(1, -(-display_width // columns))

    return min(max(visual_lines, 1), max(1, int(max_height or 1)))


def _status_bar_visible_from_display_config(display_config: object) -> bool:
    """Initial status-bar visibility; both YAML ``off`` (False) and strings like ``"hidden"`` mean off."""
    if not isinstance(display_config, dict):
        display_config = {}
    # None is "unset" (including the seeded default), so it must not shadow tui_statusbar.
    statusbar_config = display_config.get("statusbar", None)
    if statusbar_config is None:
        statusbar_config = display_config.get("tui_statusbar", "top")
    if isinstance(statusbar_config, str):
        return statusbar_config.strip().lower() not in {"0", "false", "hidden", "no", "off"}
    # int 0 is what `hermes config set display.statusbar 0` stores. False == 0, so
    # compare with == after the string branch; True is not 0.
    return not (statusbar_config is False or statusbar_config == 0)


def _collect_query_images(query: str | None, image_arg: str | None = None) -> tuple[str, list[Path]]:
    """Collect local image attachments for single-query CLI flows."""
    from cli import _IMAGE_EXTENSIONS, _detect_file_drop, _resolve_attachment_path
    message = query or ""
    images: list[Path] = []

    if isinstance(message, str):
        dropped = _detect_file_drop(message)
        if dropped and dropped.get("is_image"):
            images.append(dropped["path"])
            message = dropped["remainder"] or f"[User attached image: {dropped['path'].name}]"

    if image_arg:
        explicit_path = _resolve_attachment_path(image_arg)
        if explicit_path is None:
            raise ValueError(f"Image file not found: {image_arg}")
        if explicit_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            raise ValueError(f"Not a supported image file: {explicit_path}")
        images.append(explicit_path)

    return message, list(dict.fromkeys(images))
