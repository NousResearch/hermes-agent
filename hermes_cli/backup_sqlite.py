"""WAL-safe SQLite snapshots. Direct execution needs only the standard library.

Desktop invokes this file before stopping its backend, even when application
imports cannot load. Full and quick backups use the same SQLite copy operation.
"""
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from contextlib import suppress
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class _SQLiteBackupTimeout(RuntimeError):
    """Raised when a SQLite snapshot remains busy past its deadline."""


def _close_quietly(conn: Optional[sqlite3.Connection]) -> None:
    if conn is not None:
        with suppress(Exception):
            conn.close()


class _SQLiteBackupRestarted(RuntimeError):
    """Raised when a foreign write restarted a stepped snapshot from page one."""


# A rollback-journal reader blocks every writer's commit while it holds the read
# lock, so that copy releases it after each step; a waiting writer's PENDING lock
# refuses the next step, so it waits for one step at most.  Activity writes give
# up after 0.5s (SessionDB._ACTIVITY_WRITE_PATIENCE_S); every hold stays under it.
_ROLLBACK_STEP_SECONDS = 0.1
_ROLLBACK_FIRST_STEP_PAGES = 256
_ROLLBACK_BUSY_SLEEP_SECONDS = 0.01
# Each foreign write restarts a stepped copy, so a continuous writer gets a bounded
# number of attempts, then one whole-copy hold if it fits the budget, else a refusal.
_ROLLBACK_MAX_RESTARTS = 20
_ROLLBACK_LOCK_HOLD_SECONDS = 0.25


def _uses_wal(src: Path) -> bool:
    # Header byte 18 (file format write version) is 2 for WAL; reading it takes no lock.
    with open(src, "rb") as handle:
        header = handle.read(19)
    return len(header) == 19 and header[18] == 2


class _CopyRate:
    def __init__(self) -> None:
        self.seconds = 0.0
        self.pages = 0
        self.total = 0

    def pages_per_second(self) -> float:
        return self.pages / max(self.seconds, 1e-6)


def _stepped_attempt(conn: sqlite3.Connection, backup_conn: sqlite3.Connection, pages: int,
                     check_busy, rate: _CopyRate) -> bool:
    """One stepped backup; False when a foreign write restarted it."""
    last_remaining: Optional[int] = None
    last_tick, last_busy = time.monotonic(), False

    def _progress(status: int, remaining: int, total: int) -> None:
        nonlocal last_remaining, last_tick, last_busy
        check_busy(status, remaining, total)
        now = time.monotonic()
        # Copy time only: drop the sleep sqlite3 takes after a busy step.
        step_seconds = max(0.0, now - last_tick - (_ROLLBACK_BUSY_SLEEP_SECONDS if last_busy else 0.0))
        last_tick, last_busy = now, status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
        if status not in (sqlite3.SQLITE_OK, sqlite3.SQLITE_DONE):
            return
        restarted = last_remaining is not None and remaining >= last_remaining
        rate.seconds += step_seconds
        rate.pages += total - remaining if last_remaining is None or restarted else last_remaining - remaining
        rate.total = total
        if restarted:
            raise _SQLiteBackupRestarted()
        last_remaining = remaining

    try:
        conn.backup(backup_conn, pages=pages, progress=_progress, sleep=_ROLLBACK_BUSY_SLEEP_SECONDS)
    except _SQLiteBackupRestarted:
        return False
    return True


def _stepped_rollback_journal_copy(conn: sqlite3.Connection, backup_conn: sqlite3.Connection,
                                   check_busy) -> None:
    rate = _CopyRate()
    pages = _ROLLBACK_FIRST_STEP_PAGES
    for _attempt in range(_ROLLBACK_MAX_RESTARTS + 1):
        if _stepped_attempt(conn, backup_conn, pages, check_busy, rate):
            return
        pages = max(_ROLLBACK_FIRST_STEP_PAGES, int(rate.pages_per_second() * _ROLLBACK_STEP_SECONDS))
    expected_hold = rate.total / rate.pages_per_second()
    if expected_hold > _ROLLBACK_LOCK_HOLD_SECONDS:
        raise RuntimeError(
            "state.db uses a rollback journal and is written continuously; a consistent snapshot "
            f"would block those writes for about {expected_hold:.1f}s, so it was not taken. "
            "Retry when no agent turn is running.")
    conn.backup(backup_conn, pages=-1, progress=check_busy, sleep=_ROLLBACK_BUSY_SLEEP_SECONDS)


def _safe_copy_db(src: Path, dst: Path, *, timeout_seconds: float = 10.0) -> bool:
    """Copy a SQLite database with the backup() API (WAL-safe consistent snapshot).

    Fails closed when no consistent snapshot can be made: copying only the main file loses WAL data.
    """
    conn = backup_conn = None
    try:
        # sqlite3.connect() creates a missing destination with the process
        # umask, which is commonly 0022 (0644).  Snapshot databases contain
        # session and tool state, so create the inode owner-only before SQLite
        # writes its first byte.  O_NOFOLLOW also refuses a planted symlink on
        # platforms that support it.  Tighten an existing internal staging
        # file as well (NamedTemporaryFile callers already create it 0600).
        if os.name != "nt":
            open_flags = os.O_WRONLY | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                open_flags |= os.O_NOFOLLOW
            secure_fd = os.open(dst, open_flags, 0o600)
            try:
                os.fchmod(secure_fd, 0o600)
            finally:
                os.close(secure_fd)
        # timeout=0.0 disables sqlite3's implicit busy wait so the progress callback owns the
        # full locked-source deadline instead of adding the default timeout before each callback.
        conn = sqlite3.connect(f"{src.resolve().as_uri()}?mode=ro", uri=True, timeout=0.0)
        backup_conn = sqlite3.connect(str(dst))
        busy_deadline = time.monotonic() + max(0.0, timeout_seconds)

        def _check_backup_progress(status: int, _remaining: int, _total: int) -> None:
            nonlocal busy_deadline
            now = time.monotonic()
            if status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                if now >= busy_deadline:
                    raise _SQLiteBackupTimeout(f"database remained locked for {timeout_seconds:g} seconds")
            else:
                busy_deadline = now + max(0.0, timeout_seconds)

        if _uses_wal(src):
            # One step: SQLite restarts a multi-step backup whenever another connection writes the
            # source, and a WAL reader never blocks a writer however long the copy takes.
            conn.backup(backup_conn, pages=-1, progress=_check_backup_progress, sleep=0.1)
        else:
            _stepped_rollback_journal_copy(conn, backup_conn, _check_backup_progress)
        return True
    except Exception as exc:
        logger.warning("SQLite safe copy failed for %s: %s", src, exc)
        # Windows won't remove the partial destination while SQLite still has it open.
        _close_quietly(backup_conn)
        backup_conn = None
        with suppress(OSError):
            dst.unlink(missing_ok=True)
        return False
    finally:
        _close_quietly(backup_conn)
        _close_quietly(conn)


# Longer than the Desktop's 30-minute pre-flight cap: a staging file this old
# has no live writer. A hard-killed pre-flight (Windows TerminateProcess)
# cannot unlink its own staging copy, and each one is a full-size database.
_ABANDONED_STAGING_SECONDS = 2 * 3600


def _reclaim_abandoned_staging(home: Path, prefix: str) -> None:
    horizon = time.time() - _ABANDONED_STAGING_SECONDS
    for staged in home.glob(f"{prefix}*.partial*"):  # + the copy's -journal
        with suppress(OSError):
            if staged.stat().st_mtime < horizon:
                staged.unlink()


def preflight_state_db(home: Path) -> dict:
    """Publish an emergency snapshot; do not prune recovery files on failure."""
    source = home / "state.db"
    if not source.exists():
        return {"path": None, "message": "state.db not found (fresh install?)"}
    prefix = "state.db.pre-update-emergency-"
    _reclaim_abandoned_staging(home, prefix)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    destination = home / f"{prefix}{stamp}-{os.getpid()}.bak"
    # Named before it exists, so a SIGTERM at any point after creation reaches the
    # cleanup below; _safe_copy_db creates it owner-only and refuses a symlink.
    staged = home / f"{prefix}{stamp}-{os.getpid()}.partial"
    try:
        if not _safe_copy_db(source, staged):
            raise RuntimeError("SQLite safe copy failed; previous emergency snapshots were retained")
        connection = sqlite3.connect(str(staged))
        try:
            result = connection.execute("PRAGMA quick_check").fetchall()
            if result != [("ok",)]:
                raise RuntimeError(f"SQLite snapshot integrity check failed: {result}")
        finally:
            connection.close()
        size = staged.stat().st_size
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
    for old in sorted(home.glob(f"{prefix}*.bak"), reverse=True)[2:]:
        try:
            old.unlink()
        except OSError as exc:
            logger.warning("Could not prune emergency snapshot %s: %s", old, exc)
    return {"path": str(destination), "bytes": size}


if __name__ == "__main__":
    # The Desktop stops a pre-flight past its deadline with SIGTERM; exit through
    # the staging cleanup instead of dying with a full-size .partial left behind.
    signal.signal(signal.SIGTERM, lambda signum, _frame: sys.exit(128 + signum))
    print(json.dumps(preflight_state_db(Path(sys.argv[1]))))
