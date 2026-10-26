"""Deadline-aware acquisition helpers for SessionDB's in-process writer mutex."""

import sqlite3
import time
from contextlib import contextmanager


class WriterMutexTimeout(sqlite3.OperationalError):
    """The in-process writer mutex consumed a write's full patience budget."""


def write_lock_timeout_diagnostic(exc: sqlite3.OperationalError, patience_s: float) -> tuple[str, bool]:
    """Return the timeout text and whether a cross-process holder report is useful."""
    if isinstance(exc, WriterMutexTimeout):
        return (
            f"database is locked (the in-process state.db writer mutex was held "
            f"for over {patience_s:g}s)",
            False,
        )
    return (
        f"database is locked (another Hermes process held the state.db write lock for over "
        f"{patience_s:.0f}s — likely a long maintenance operation such as VACUUM, "
        "a large WAL checkpoint, or an older pre-update process; the database itself is healthy)",
        True,
    )


@contextmanager
def lock_until_deadline(lock, deadline: float, patience_s: float):
    """Acquire a lock within the current write budget; preserve Lock/RLock semantics."""
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not lock.acquire(timeout=remaining):
        raise WriterMutexTimeout(
            f"database is locked (the in-process state.db writer mutex was held "
            f"for over {patience_s:g}s)"
        )
    try:
        yield
    finally:
        lock.release()
