"""Cosmetic process title for ``hermes`` startup, split out of ``hermes_cli.main``."""

import logging

logger = logging.getLogger(__name__)


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
    except UnicodeDecodeError:
        # setproctitle can raise this while importing, and an unhandled one here aborts
        # startup entirely (#98620). Degrade to the ctypes fallback, but say so: a silent
        # permanent no-op is how this was found.
        logger.debug("setproctitle failed to decode on import; using the ctypes fallback", exc_info=True)

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
    except Exception:  # cosmetic only: no libc / symbol must never stop startup
        logger.debug("ctypes process title fallback failed", exc_info=True)
