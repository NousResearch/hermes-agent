"""Best-effort terminal tab and window title updates for the classic CLI."""

from __future__ import annotations

import os
import re
import sys
import threading


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_MAX_TITLE_LENGTH = 200
_WRITE_LOCK = threading.Lock()


def sanitize_terminal_title(value: object) -> str:
    """Return a printable, bounded title that cannot inject terminal escapes."""
    text = _CONTROL_CHARS.sub("", str(value or ""))
    return " ".join(text.split())[:_MAX_TITLE_LENGTH]


def terminal_title_symbol(response_label: object, fallback: str = "⚕") -> str:
    """Extract the skin's leading symbol from its response-panel label."""
    label = sanitize_terminal_title(response_label)
    return label.split(maxsplit=1)[0] if label else fallback


def compose_terminal_title(
    response_label: object,
    session_title: object = "",
    *,
    busy: bool = False,
) -> str:
    """Compose the short tab title for an idle or active classic CLI session."""
    parts = [terminal_title_symbol(response_label)]
    title = sanitize_terminal_title(session_title)
    if title:
        parts.append(title)
    if busy:
        parts.append("⏳")
    return " ".join(parts)


def _set_windows_console_title(title: str) -> bool:
    """Set the native Windows console title without relying on OSC support."""
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.SetConsoleTitleW(title))
    except Exception:
        return False


def _is_interactive_output(output: object) -> bool:
    """Check the underlying stream for prompt_toolkit Output instances."""
    stream = getattr(output, "stdout", output)
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _write_osc_terminal_title(output: object, title: str) -> bool:
    """Write OSC title sequences, returning whether the terminal accepted them."""
    try:
        with _WRITE_LOCK:
            sequence = f"\033]1;{title}\a\033]2;{title}\a"
            write_raw = getattr(output, "write_raw", None)
            if callable(write_raw):
                write_raw(sequence)
            else:
                output.write(sequence)
            output.flush()
        return True
    except Exception:
        return False


def write_terminal_title(title: object, output: object | None = None) -> bool:
    """Set an interactive terminal's tab and window title.

    On Windows, this uses ``SetConsoleTitleW`` for classic conhost and also
    writes OSC 1/2 for terminal emulators such as mintty and VS Code. Elsewhere,
    OSC 1 updates a terminal icon/tab label and OSC 2 updates the window title.
    Prompt_toolkit ``Output`` objects are supported so callers can bypass
    ``patch_stdout`` safely. The writer deliberately avoids logging failures
    because it may be called from an agent callback.
    """
    if os.environ.get("TERM", "").lower() == "dumb":
        return False

    try:
        output = output if output is not None else sys.stdout
        if output is None or not _is_interactive_output(output):
            return False
        clean_title = sanitize_terminal_title(title)
        if not clean_title:
            return False
        if sys.platform == "win32":
            native_updated = _set_windows_console_title(clean_title)
            osc_updated = _write_osc_terminal_title(output, clean_title)
            return native_updated or osc_updated
        return _write_osc_terminal_title(output, clean_title)
    except Exception:
        return False
