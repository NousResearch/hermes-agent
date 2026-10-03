"""WAL-safe SQLite snapshots. Direct execution needs only the standard library.

Desktop invokes this file before stopping its backend, even when application
imports cannot load. Full and quick backups use the same SQLite copy operation.
"""
import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class _SQLiteBackupTimeout(RuntimeError):
    """Raised when a SQLite snapshot remains busy past its deadline."""


# How many emergency backups to keep, newest first, counting the one just written.
# This is the module's existing effective retention (the previous sweep kept the
# two newest after a successful snapshot); it only becomes a named constant so
# the sweep and its tests agree on the budget. #91298's keep-3 reading applied to
# the Electron-side copy, whose filter excluded the just-written backup before
# slicing; this Python writer never excluded it, so its cap stays 2.
EMERGENCY_BACKUP_RETENTION = 2

# Filename pieces of the emergency copies written below; kept as constants so a
# test can assert the sweep and the writer agree on the shape.
EMERGENCY_BACKUP_PREFIX = "state.db.pre-update-emergency-"
EMERGENCY_BACKUP_SUFFIX = ".bak"


def prune_emergency_backups(home: Path) -> int:
    """Delete emergency backups past the retention budget; return how many.

    Runs unconditionally, before the snapshot attempt, so reclaiming happens on
    exactly the paths that used to skip it (#91229): a locked source database
    (EBUSY — the ordinary state during a failed Windows self-update), a full
    disk (ENOSPC — full partly because these files were never reclaimed), and a
    missing/too-small ``state.db`` that returns before the copy. Never raises:
    this is disk hygiene on the update path and a failure to reclaim must not
    abort the update. ``win-unpacked.bak`` is deliberately not touched — that is
    a single rollback copy maintained by the desktop pack step, not an
    accumulating artifact of this module.
    """
    try:
        backups = sorted(
            p for p in home.glob(f"{EMERGENCY_BACKUP_PREFIX}*{EMERGENCY_BACKUP_SUFFIX}")
        )
        deleted = 0
        for old in backups[:-EMERGENCY_BACKUP_RETENTION] if EMERGENCY_BACKUP_RETENTION > 0 else backups:
            try:
                old.unlink()
                deleted += 1
            except OSError as exc:
                logger.warning("Could not prune emergency snapshot %s: %s", old, exc)
        return deleted
    except OSError as exc:
        logger.warning("Could not list emergency snapshots in %s: %s", home, exc)
        return 0


def _close_quietly(conn: Optional[sqlite3.Connection]) -> None:
    if conn is not None:
        with suppress(Exception):
            conn.close()


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

        conn.backup(backup_conn, pages=256, progress=_check_backup_progress, sleep=0.1)
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


def preflight_state_db(home: Path) -> dict:
    """Publish an emergency snapshot; do not prune recovery files on failure."""
    # Unconditionally, and BEFORE any early return or snapshot attempt: the sweep
    # used to live at the end of the success path, so every reason this function
    # gives up (state.db missing — fresh install) and every reason the snapshot
    # fails (locked source on a failed Windows self-update, a full disk) skipped
    # the cleanup, and the backups accumulated precisely when reclaiming
    # mattered most (#91229).
    prune_emergency_backups(home)
    source = home / "state.db"
    if not source.exists():
        return {"path": None, "message": "state.db not found (fresh install?)"}
    prefix = EMERGENCY_BACKUP_PREFIX
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    destination = home / f"{prefix}{stamp}-{os.getpid()}.bak"
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".partial", dir=home)
    os.close(fd)
    staged = Path(name)
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
    # Reclaim now that a new backup exists, so the newest one counts toward the
    # retention budget (the unconditional sweep above is what guarantees the
    # budget holds at all; this second pass keeps it from being exceeded by one
    # until the next update runs).
    prune_emergency_backups(home)
    return {"path": str(destination), "bytes": size}


if __name__ == "__main__":
    print(json.dumps(preflight_state_db(Path(sys.argv[1]))))
