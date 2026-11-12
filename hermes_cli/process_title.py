"""Cosmetic process title (``ps``/``top`` shows ``hermes`` rather than ``python3.xx``).

Extracted from ``hermes_cli/main.py`` so the CLI facade can carry new dispatch-time
behaviour without growing past its line cap (see scripts/code_health).
"""

import sys


def set_process_title() -> None:
    """Cosmetic: show 'hermes' instead of 'python3.xx' in ps/top/htop.

    Order: opt-in ``setproctitle`` dep; ctypes ``prctl(PR_SET_NAME)`` (Linux,
    15-char limit); ``pthread_setname_np`` (macOS — lldb/top only, not ``ps
    aux``); no-op on Windows (the .exe is already ``hermes.exe``). Never fatal.
    """
    try:
        import setproctitle  # type: ignore[import-untyped]

        setproctitle.setproctitle("hermes")
        return
    except ImportError:
        pass

    import ctypes
    import platform

    try:
        system = platform.system()
        if system == "Linux":
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(15, b"hermes", 0, 0, 0)  # PR_SET_NAME = 15
        elif system == "Darwin":
            libc = ctypes.CDLL("libc.dylib", use_errno=True)
            libc.pthread_setname_np(b"hermes")
    except Exception:
        pass
