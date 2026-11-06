"""Bounded retry policy for *settled, rolled-back* conversation storage writes.

No provider/tool retries, reconnect, WAL deletion, checkpoint or database repair.
The optional host reserve belongs to the external storage guard, not Hermes data.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path
import random
import sqlite3
import stat
import time
from contextlib import contextmanager
from contextvars import ContextVar

_active_budget = ContextVar("conversation_save_recovery", default=None)


def current_save_recovery():
    return _active_budget.get()


@contextmanager
def conversation_save_recovery(db_path):
    existing = _active_budget.get()
    if existing is not None and existing.db_path == db_path:
        yield
        return
    token = _active_budget.set(SaveRecoveryBudget(db_path))
    try:
        yield
    finally:
        _active_budget.reset(token)

RECOVERY_PATIENCE_S = 30.0
RECOVERY_MAX_RETRIES = 32


def recoverable_storage_cause(exc: BaseException) -> str | None:
    """Result codes outrank prose; never treat general disk/IO errors as full."""
    if isinstance(exc, OSError):
        return "full" if exc.errno == errno.ENOSPC else None
    if not isinstance(exc, sqlite3.OperationalError):
        return None
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None:
        primary = code & 0xff
        if primary == sqlite3.SQLITE_FULL:
            return "full"
        if primary in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            return "busy"
        return None
    text = str(exc).lower()
    if text in ("database or disk is full", "disk full"):
        return "full"
    if text in ("database is locked", "database table is locked", "database is busy"):
        return "busy"
    return None


def release_guard_reserve(db_path: Path) -> bool:
    """Release ONLY the installed guard's private reserve under real low space.

    Use its existing nonblocking lock; never create/rearm it or run cleanup code.
    Directory fd and no-follow opens fence symlinks. Unknown/unsafe paths decline.
    """
    if os.name != "posix":
        return False
    import fcntl

    directory = Path.home() / ".local/state/hermes-storage-guard"
    dfd = lockfd = None
    try:
        if directory.resolve() != directory:
            return False
        dfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        ds = os.fstat(dfd)
        if ds.st_uid != os.getuid() or ds.st_mode & 0o077:
            return False
        if os.stat(db_path).st_dev != ds.st_dev:
            return False
        space = os.statvfs(db_path)
        if space.f_bavail * space.f_frsize >= 4 * 1024 ** 3:
            return False
        lockfd = os.open("guard.lock", os.O_RDWR | os.O_NOFOLLOW, dir_fd=dfd)
        ls = os.fstat(lockfd)
        if not stat.S_ISREG(ls.st_mode) or ls.st_uid != os.getuid() or ls.st_nlink != 1 or ls.st_mode & 0o077:
            return False
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reserve = os.stat("emergency.reserve", dir_fd=dfd, follow_symlinks=False)
        if not stat.S_ISREG(reserve.st_mode) or reserve.st_uid != os.getuid() or reserve.st_nlink != 1 or reserve.st_mode & 0o077:
            return False
        os.unlink("emergency.reserve", dir_fd=dfd)
        return True
    except OSError:
        return False
    finally:
        if lockfd is not None:
            os.close(lockfd)
        if dfd is not None:
            os.close(dfd)


class SaveRecoveryBudget:
    """One additional bounded recovery window, not another full 60-second wait."""
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.deadline = None
        self.retries = 0
        self.pressure_checked = False

    @contextmanager
    def bounded_busy_timeout(self, conn):
        """Keep SQLite's own lock wait inside an already-started recovery window."""
        if self.deadline is None:
            yield
            return
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise sqlite3.OperationalError("database is locked")
        previous_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        conn.execute(f"PRAGMA busy_timeout={min(previous_ms, int(remaining * 1000))}")
        try:
            yield
        finally:
            conn.execute(f"PRAGMA busy_timeout={int(previous_ms)}")

    def wait(self, cause: str) -> bool:
        if self.deadline is None:
            self.deadline = time.monotonic() + RECOVERY_PATIENCE_S
        if self.retries >= RECOVERY_MAX_RETRIES or time.monotonic() >= self.deadline:
            return False
        if cause == "full" and not self.pressure_checked:
            self.pressure_checked = True
            release_guard_reserve(self.db_path)
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(remaining, random.uniform(0.05, min(1.0, 0.1 * 2 ** min(self.retries, 4)))))
        self.retries += 1
        return time.monotonic() < self.deadline
