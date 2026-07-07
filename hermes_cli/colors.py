"""Shared ANSI color utilities for Hermes CLI modules."""

import os
import sys


def _enable_windows_ansi() -> None:
    """Enable Windows Virtual Terminal Processing for ANSI escape sequences.

    On Windows, the console does not interpret ANSI escape sequences by
    default.  This function enables the ``ENABLE_VIRTUAL_TERMINAL_PROCESSING``
    flag on the stdout (and stderr) handles so that ``\\033[...`` codes
    render as colors instead of being printed as raw text.

    Safe to call on non-Windows platforms (no-op).  Never raises — if the
    API call fails, we silently fall back to no-color output.
    """
    if sys.platform != "win32":
        return

    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        # STD_OUTPUT_HANDLE = -11, STD_ERROR_HANDLE = -12
        for handle_id in (-11, -12):
            handle = kernel32.GetStdHandle(handle_id)
            if handle is None or handle == wintypes.HANDLE(-1).value:
                continue

            # Read current mode
            mode = wintypes.DWORD()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                continue

            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(handle, new_mode)
    except Exception:
        # Silently ignore — worst case, no colors on Windows.
        pass


# Enable Windows VT processing once at import time.
_enable_windows_ansi()


def should_use_color() -> bool:
    """Return True when colored output is appropriate.

    Respects the NO_COLOR environment variable (https://no-color.org/)
    and TERM=dumb, in addition to the existing TTY check.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if not sys.stdout.isatty():
        return False
    return True


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def color(text: str, *codes) -> str:
    """Apply color codes to text (only when color output is appropriate)."""
    if not should_use_color():
        return text
    return "".join(codes) + text + Colors.RESET
