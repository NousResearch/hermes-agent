"""Shared ANSI color utilities for Hermes CLI modules."""

import os
import sys


def _enable_windows_vt_processing() -> bool:
    """Enable Virtual Terminal Processing on Windows console handles.

    Modern Windows 10+ terminals support ANSI escape sequences, but they
    must be explicitly opted-in via ``SetConsoleMode`` with the
    ``ENABLE_VIRTUAL_TERMINAL_PROCESSING`` flag (0x0004).  Without this
    the kernel passes raw escape codes through as literal text — exactly
    the symptom reported in #59397.

    Returns True when VT processing was successfully enabled or when the
    call is not applicable (non-Windows).
    """
    if sys.platform != "win32":
        return True

    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        STD_OUTPUT_HANDLE = -11
        STD_ERROR_HANDLE = -12
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

        def _try_enable(handle_id: int) -> bool:
            handle = kernel32.GetStdHandle(handle_id)
            if handle is None or handle == -1:
                return False
            mode = wintypes.DWORD()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            if mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING:
                return True  # already enabled
            new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            if kernel32.SetConsoleMode(handle, new_mode):
                return True
            return False

        # Enable on both stdout and stderr; at least one must succeed.
        return _try_enable(STD_OUTPUT_HANDLE) or _try_enable(STD_ERROR_HANDLE)
    except Exception:
        # ctypes or Windows API unavailable — fall through to disable colors.
        return False


# Cache the result so we only call into the Windows API once.
_vt_enabled: bool = _enable_windows_vt_processing()


def should_use_color() -> bool:
    """Return True when colored output is appropriate.

    Respects the NO_COLOR environment variable (https://no-color.org/),
    TERM=dumb, and the existing TTY check.  On Windows, also verifies
    that Virtual Terminal Processing is active so ANSI escape codes are
    rendered as colors instead of printed literally (see #59397).
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32" and not _vt_enabled:
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
