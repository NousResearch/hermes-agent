"""Turn termination signals into an exception for the duration of a block.

A ``finally`` block or an ``ExitStack`` never runs when the process dies from the default SIGTERM
disposition, so the cleanup those blocks perform (unlink a hidden ``.partial``, remove a staging
directory) silently does not happen: that is how an aborted ``hermes backup`` left
``.<name>.<pid>-<tid>.partial`` and its staged SQLite copy behind, and how an interrupted
``hermes plugins doctor`` stranded its ``hermes-plugin-doctor-*`` home in the temp dir.

Installing ``unwind_on_termination()`` around the region that owns the temporary artifact makes
SIGTERM/SIGHUP raise ``SystemExit(128 + signum)`` in the main thread, so the ordinary unwinding
path runs and the process still reports a signal-style exit status to its parent. The previous
handler is restored on exit; nesting is safe (LIFO restore). A process that cannot install
handlers (not the main thread, unsupported platform) keeps its current disposition and is not
held back — the caller's cleanup then only covers the exception path.
"""

from __future__ import annotations

import logging
import signal
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# SIGHUP is absent on Windows; SIGINT is deliberately excluded because CPython already raises
# KeyboardInterrupt for it, which unwinds without help.
TERMINATION_SIGNALS: tuple[int, ...] = tuple(
    signum for signum in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None))
    if signum is not None
)

_Handler = Any


def _exit_on_signal(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


@contextmanager
def unwind_on_termination() -> Iterator[None]:
    """Unwind the enclosing ``with``/``finally`` blocks when SIGTERM or SIGHUP arrives."""
    installed: list[tuple[int, _Handler]] = []
    for signum in TERMINATION_SIGNALS:
        try:
            previous = signal.getsignal(signum)
            signal.signal(signum, _exit_on_signal)
        except (ValueError, OSError, RuntimeError):  # off-main-thread or unsupported platform
            continue
        installed.append((signum, previous))
    if not installed:
        logger.debug("termination guard inactive: signals cannot be installed in this process")
    try:
        yield
    finally:
        for signum, previous in reversed(installed):
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError, RuntimeError, TypeError):  # pragma: no cover - restore race
                pass
