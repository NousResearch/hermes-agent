"""Shared ANSI color utilities for Hermes CLI modules."""

import os
import sys


def enable_windows_ansi(stream) -> bool | None:
    """Enable ANSI on a Windows console; None means the stream has no console."""
    import ctypes
    from ctypes import wintypes
    import msvcrt

    try:
        handle = msvcrt.get_osfhandle(stream.fileno())
    except (OSError, ValueError, AttributeError):
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetConsoleMode.restype = wintypes.BOOL
    kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.SetConsoleMode.restype = wintypes.BOOL
    mode = wintypes.DWORD()
    if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
        return None
    virtual_terminal_processing = 0x0004
    if mode.value & virtual_terminal_processing:
        return True
    return bool(kernel.SetConsoleMode(handle, mode.value | virtual_terminal_processing))


def should_use_color() -> bool:
    """Return True when colored output is appropriate."""
    if os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
        return False
    if not sys.stdout.isatty():
        return False
    return bool(enable_windows_ansi(sys.stdout)) if sys.platform == "win32" else True


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
