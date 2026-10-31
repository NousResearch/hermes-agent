"""Prevent macOS tests and their descendants from executing host launchctl.

Apply a narrow Seatbelt restriction before test collection. Unlike shell text
inspection, it follows forks/execs and allows arbitrary harmless shell syntax
and test-owned launchctl fakes. Reading the installed controller is denied too,
so a child cannot copy it to a different executable path.

This is a test-process restriction, not a general hostile-code sandbox. macOS's
sandbox_init API is deprecated; if it is unavailable or rejects the profile,
collection must fail explicitly rather than continue without protection.
"""
from __future__ import annotations

import ctypes
import sys

_PROFILE = b'''(version 1)
(allow default)
; macOS rejects set-ID ps under Seatbelt even with allow default. Only the
; protected system diagnostic binary is exempt, never shells/interpreters.
(allow process-exec (literal "/bin/ps") (with no-sandbox))
(deny process-exec file-read-data
    (literal "/bin/launchctl")
    (literal "/usr/bin/launchctl"))
'''
_INSTALLED = False


def install_launchctl_guard() -> None:
    """Constrain this Darwin process and its children; leave other OSes alone."""
    global _INSTALLED
    if sys.platform != "darwin" or _INSTALLED:
        return
    try:
        library = ctypes.CDLL("/usr/lib/libsandbox.dylib", use_errno=True)
        initialize = library.sandbox_init
        initialize.argtypes = [
            ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p),
        ]
        initialize.restype = ctypes.c_int
        release = library.sandbox_free_error
        release.argtypes = [ctypes.c_char_p]
        release.restype = None
    except (OSError, AttributeError) as error:
        raise RuntimeError(
            "macOS launchctl test guard unavailable: cannot load sandbox_init"
        ) from error

    error_buffer = ctypes.c_char_p()
    try:
        result = initialize(_PROFILE, 0, ctypes.byref(error_buffer))
        if result != 0:
            detail = error_buffer.value.decode(errors="replace") if error_buffer.value else "profile rejected"
            raise RuntimeError(f"macOS launchctl test guard failed: {detail}")
    finally:
        if error_buffer:
            release(error_buffer)
    _INSTALLED = True
