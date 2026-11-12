"""Terminal tab activity indicators (OSC 0 title + OSC 9;4 progress) for the interactive CLI.

Claude Code keeps its tab labelled with the session it is working in, and drives the terminal's
own progress bar while a turn runs; the CLI REPL did neither, so a Ghostty tab running three
Hermes sessions looked exactly like a plain shell. Two display settings, both config-only:

  ``display.terminal_title``    (default on) — ``Hermes · <session title>``, ◐◑ while a turn runs
  ``display.terminal_progress`` (default on) — OSC 9;4 indeterminate bar while a turn runs

Writes take the same route as the bell (``hermes_cli.terminal_notify``): through the
prompt_toolkit output ON THE APP LOOP when the app is live, else straight to ``/dev/tty`` — a
second writer on the tty splices into an in-flight kitty pet frame. Non-TTY runs (``hermes -z``,
piped output, cron) never emit anything.
"""

from __future__ import annotations

import os
import re
import sys
import threading

_TITLE_NAME = "Hermes"
# Busy glyph right before the label while a turn runs — the same half-circles Claude Code spins,
# so the tab reads the same way it does there. Static terminals just show the first frame.
SPINNER_FRAMES = ("\u25d0", "\u25d1")
# Claude repaints its tab title at roughly this rate; the payload is 20-odd bytes and the write is
# serialized on the app loop, so the cost is a rounding error next to the CLI's own spinner.
SPINNER_INTERVAL = 0.15
_MAX_LEN = 60
# Control chars would terminate the OSC early or inject a new one; the title is also on the
# CR/LF boundary of every terminal that shows it.
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
# True once an indicator sequence reached the tty: the exit path only clears what we actually set.
_EMITTED = False

# OSC 9;4 progress states (ConEmu protocol, which iTerm2/Ghostty/WezTerm also render). We only
# ever emit two: an indeterminate bar for the length of a turn, then remove.
_PROGRESS_INDETERMINATE = 3
_PROGRESS_REMOVE = 0


def _display_flag(config, key: str, default: bool = True) -> bool:
    """One boolean ``display.<key>``; truthy strings count as on, so ``"false"`` reads as off."""
    display = config.get("display") if isinstance(config, dict) else None
    value = display.get(key, default) if isinstance(display, dict) else default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _config_flag(key: str, default: bool = True) -> bool:
    """``display.<key>`` from the read-only loader (importing ``cli`` here would be a cycle)."""
    try:
        from hermes_cli.config import load_config_readonly
        return _display_flag(load_config_readonly(), key, default)
    except Exception:
        return False


def title_enabled(config=None) -> bool:
    """``display.terminal_title`` — on unless explicitly disabled (``config`` injectable for tests)."""
    return _config_flag("terminal_title") if config is None else _display_flag(config, "terminal_title")


def progress_enabled(config=None) -> bool:
    """``display.terminal_progress`` — on unless explicitly disabled."""
    return _config_flag("terminal_progress") if config is None else _display_flag(config, "terminal_progress")


def sanitize_title(text: str) -> str:
    """Single-line, control-free title (semicolons are fine: OSC 0 takes the rest of the string)."""
    return _CTRL.sub("", str(text or "").replace("\n", " ")).strip()


def title_text(label: str, *, busy: bool = False, frame: int = 0) -> str:
    """``Hermes · <label>``, spinner glyph first while a turn runs (``label`` already resolved)."""
    title = f"{_TITLE_NAME} \u00b7 {label}" if label else _TITLE_NAME
    if busy:
        title = f"{SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]} {title}"
    return title[:_MAX_LEN]


def compose_title(*, session_title: str = "", cwd: str = "", busy: bool = False, frame: int = 0) -> str:
    """``title_text`` from raw parts; falls back to basename(cwd) for a still-untitled session."""
    label = sanitize_title(session_title) or os.path.basename(sanitize_title(cwd).rstrip("/"))
    return title_text(label, busy=busy, frame=frame)


def title_sequence(title: str) -> str:
    """OSC 0 (SET_TITLE_AND_ICON: tab + window title) — Ghostty, iTerm2, kitty, WezTerm, VS Code."""
    return f"\x1b]0;{sanitize_title(title)}\x07"


def progress_sequence(*, running: bool) -> str:
    """OSC 9;4: an indeterminate bar while a turn runs, remove when it ends.

    Progress is a turn-shaped fact, not a percentage: the agent cannot know how many tool calls
    remain, so the bar animates until the turn is over (Claude Code does the same). The percent
    field is optional and meaningless for both states, so it is omitted when removing.
    """
    return (f"\x1b]9;4;{_PROGRESS_INDETERMINATE};0\x07" if running
            else f"\x1b]9;4;{_PROGRESS_REMOVE}\x07")


def _app_running(cli) -> bool:
    app = getattr(cli, "_app", None)
    return bool(app is not None and getattr(app, "_is_running", False))


def _tty_available(cli) -> bool:
    """The app's output, or a real stdout. Gates piped/one-shot runs out of the indicator path."""
    if _app_running(cli):
        return True
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _write(cli, seq: str) -> None:
    """App loop when the prompt_toolkit app is live, else /dev/tty (the bell's contract)."""
    from hermes_cli.cli_terminal_mixin import _run_on_app_loop, _write_terminal_sequence
    from hermes_cli.terminal_notify import write_tty

    app = getattr(cli, "_app", None)
    if not _app_running(cli):
        write_tty(seq)
        return
    _run_on_app_loop(app, lambda: _write_terminal_sequence(app, seq))


def session_label(cli) -> str:
    """The status bar's session title (polls state.db at most every 1.5s), '' when unavailable."""
    getter = getattr(cli, "_get_status_bar_session_title", None)
    if not callable(getter):
        return ""
    try:
        return str(getter() or "")
    except Exception:
        return ""


def resolve_label(cli) -> str:
    """Tab label for this session: the (possibly not-yet-generated) session title, else basename(cwd)."""
    return (sanitize_title(session_label(cli))
            or os.path.basename(sanitize_title(os.getcwd()).rstrip("/"))
            or _TITLE_NAME)


class _Spinner:
    """Repaints the busy title with alternating glyphs while a turn runs.

    Owns the label captured at turn start: the session title can change mid-turn (auto-title),
    but re-reading state.db every 150ms for a decorative glyph is not worth it — the turn-end
    write picks the fresh label up. One spinner per process (one interactive CLI).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, cli, *, label: str) -> None:
        self.stop()
        self._stop = threading.Event()
        thread = threading.Thread(target=self._run, args=(cli, label, self._stop),
                                  name="hermes-tab-title", daemon=True)
        self._thread = thread
        thread.start()

    def _run(self, cli, label: str, stop: threading.Event) -> None:
        frame = 1  # frame 0 was written by the caller's set_activity
        while not stop.wait(SPINNER_INTERVAL):
            try:
                _write(cli, title_sequence(title_text(label, busy=True, frame=frame)))
            except Exception:
                return  # dead tty / broken app: stop spinning, the idle write still runs
            frame += 1

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
            self._stop.set()
        if thread is not None and thread.is_alive():
            # Not the daemon's 0.15s sleep: join briefly so the last spin frame cannot land
            # after the idle title (a stale ◐ over an idle tab).
            thread.join(timeout=SPINNER_INTERVAL * 2)

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._thread is not None and self._thread.is_alive())


_SPINNER = _Spinner()


def stop_spinner() -> None:
    """Halt the busy-title repaint (turn end, exit). Safe to call from any thread."""
    _SPINNER.stop()


def _flag(cli, attr: str, key: str, config) -> bool:
    """Resolve one display flag once per process and remember it on the CLI (config-only knob)."""
    if config is not None:
        return _display_flag(config, key)
    cached = getattr(cli, attr, None)
    if cached is None:
        cached = _config_flag(key)
        setattr(cli, attr, cached)
    return bool(cached)


def set_activity(cli, *, busy: bool, config=None) -> bool:
    """Label the tab and drive its progress bar for the current state; True when anything was written.

    Never raises: indicators are cosmetic and must not take a turn down with them.
    """
    global _EMITTED
    try:
        if getattr(cli, "_terminal_io_broken", False) or not _tty_available(cli):
            stop_spinner()
            return False
        title_on = _flag(cli, "_terminal_title_cfg_enabled", "terminal_title", config)
        progress_on = _flag(cli, "_terminal_progress_cfg_enabled", "terminal_progress", config)
        if not (title_on or progress_on):
            stop_spinner()
            return False
        label = resolve_label(cli)
    except Exception:
        stop_spinner()
        return False
    written = False
    cli._terminal_title_busy = bool(busy)
    if title_on:
        _write(cli, title_sequence(title_text(label, busy=busy)))
        written = True
    if progress_on:
        _write(cli, progress_sequence(running=busy))
        written = True
    if written:
        _EMITTED = True
    if busy and title_on:
        _SPINNER.start(cli, label=label)
    else:
        stop_spinner()
    return written


def reset_on_exit() -> None:
    """Drop the title/bar we set so the shell's own label comes back (exit path). Never raises.

    Claude Code clears its title on the way out; without this the tab keeps saying "Hermes · …"
    (or keeps an animated bar) after the process is gone.
    """
    global _EMITTED
    stop_spinner()
    if not _EMITTED:
        return
    _EMITTED = False
    try:
        from hermes_cli.terminal_notify import write_tty
        write_tty("\x1b]0;\x07" + progress_sequence(running=False))
    except Exception:
        pass
