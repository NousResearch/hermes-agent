"""Turn termination signals into an exception for the duration of a block.

A ``finally`` block or an ``ExitStack`` never runs when the process dies from the default SIGTERM
disposition, so the cleanup those blocks perform (unlink a hidden ``.partial``, remove a staging
directory) silently does not happen after a default-disposition termination.

Installing ``unwind_on_termination()`` around the region that owns the temporary artifact makes
SIGTERM/SIGHUP raise ``SystemExit(128 + signum)`` in the main thread, so the ordinary unwinding
path runs and the process still reports a signal-style exit status to its parent. The previous
handler is restored on exit. Custom handlers and ignored signals retain their existing
semantics, and worker threads leave process-wide handlers alone. Uncatchable exits still
require the caller's abandoned-staging recovery protocol.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from typing import Any, Generator

# SIGHUP is absent on Windows; SIGINT is deliberately excluded because CPython already raises
# KeyboardInterrupt for it, which unwinds without help.
TERMINATION_SIGNALS: tuple[int, ...] = tuple(
    signum
    for signum in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None))
    if signum is not None
)

_Handler = Any


def _exit_on_signal(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


@contextmanager
def unwind_on_termination() -> Generator[None]:
    """Unwind the enclosing ``with``/``finally`` blocks when SIGTERM or SIGHUP arrives."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    installed: list[tuple[int, _Handler]] = []
    for signum in TERMINATION_SIGNALS:
        previous = signal.getsignal(signum)
        # Embedded callers retain their own shutdown contract. Nested guards
        # already have the outer handler and need no second installation.
        if previous != signal.SIG_DFL:
            continue
        signal.signal(signum, _exit_on_signal)
        installed.append((signum, previous))
    try:
        yield
    finally:
        for signum, previous in reversed(installed):
            signal.signal(signum, previous)
