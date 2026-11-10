"""BackupAdmin operations of the SQLite store admin: WAL-safe snapshots and restore, file verification, the session
row counts, the runtime-ledger probe and offline transcript recovery."""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from contextlib import closing, contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

from persistence.sqlite._state.holders import read_only_db_uri

from .._backend import translated

from . import _recovery
from ._base import AdminFile

logger = logging.getLogger("hermes_cli.backup_restore")
_backup_sqlite_logger = logging.getLogger("hermes_cli.backup_sqlite")

_SQLITE_HEADER = b"SQLite format 3\0"

# Above this size ``PRAGMA integrity_check`` (walks every b-tree page: minutes of pegged CPU on a
# 30 GB state.db, reading as a hung ``hermes update``) is replaced by the O(1) header+schema probe.
# Sessions databases in the tens of GB are normal for heavy users, so the size-unbounded check is
# never an acceptable default on the update path.
DEFAULT_INTEGRITY_CHECK_MAX_BYTES = 2 << 30  # 2 GiB

_LEDGER_PROBE = ("SELECT EXISTS(SELECT 1 FROM session_admissions) "
                 "OR EXISTS(SELECT 1 FROM worker_executions)")


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
# up after 0.5s (the store's activity-write patience); every hold stays under it.
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
    # Floored: a run where no step completed has no measured rate, and the estimate must not divide by 0.
    expected_hold = rate.total / max(rate.pages_per_second(), 1.0)
    if expected_hold > _ROLLBACK_LOCK_HOLD_SECONDS:
        raise RuntimeError(
            "state.db uses a rollback journal and is written continuously; a consistent snapshot "
            f"would block those writes for about {expected_hold:.1f}s, so it was not taken. "
            "Retry when no agent turn is running.")
    conn.backup(backup_conn, pages=-1, progress=check_busy, sleep=_ROLLBACK_BUSY_SLEEP_SECONDS)


class _OffThreadCopy:
    """One long sqlite3 call run off the main thread. Python runs signal handlers only on the main
    thread and only between bytecodes, so a one-step backup (a single C ``sqlite3_backup_step``,
    GIL released) would hold a SIGTERM until the whole copy finished; the main thread waits here in
    short joins instead, so a deadline signal interrupts the snapshot promptly."""

    def __init__(self, copy) -> None:
        self._copy, self._error = copy, None
        # health: allow HX012 -- the copy thread runs one sqlite3 backup call and reads no profile state
        self.thread = threading.Thread(target=self._run, name="sqlite-snapshot", daemon=True)

    def _run(self) -> None:
        try:
            self._copy()
        except BaseException as exc:  # health: allow BLE001 -- handed to the waiting main thread, which re-raises it
            self._error = exc

    def run(self) -> None:
        self.thread.start()
        while self.thread.is_alive():
            self.thread.join(0.05)
        if self._error is not None:
            raise self._error

    def running(self) -> bool:
        return self.thread.is_alive()


def _safe_copy_db(src: Path, dst: Path, *, timeout_seconds: float = 10.0) -> bool:
    """Copy a SQLite database with the backup() API (WAL-safe consistent snapshot).

    Fails closed when no consistent snapshot can be made: copying only the main file loses WAL data.
    """
    conn = backup_conn = None
    copier: Optional[_OffThreadCopy] = None
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
        conn = sqlite3.connect(f"{src.resolve().as_uri()}?mode=ro", uri=True, timeout=0.0,
                               check_same_thread=False)
        backup_conn = sqlite3.connect(str(dst), check_same_thread=False)
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
            # Run off the main thread so a SIGTERM is serviced during the copy (_OffThreadCopy).
            copier = _OffThreadCopy(lambda: conn.backup(
                backup_conn, pages=-1, progress=_check_backup_progress, sleep=0.1))
            copier.run()
        else:
            _stepped_rollback_journal_copy(conn, backup_conn, _check_backup_progress)
        return True
    except Exception as exc:  # health: allow BLE001 -- any failure fails the copy closed and removes the partial file
        _backup_sqlite_logger.warning("SQLite safe copy failed for %s: %s", src, exc)
        # Windows won't remove the partial destination while SQLite still has it open.
        _close_quietly(backup_conn)
        backup_conn = None
        with suppress(OSError):
            dst.unlink(missing_ok=True)
        return False
    finally:
        # Interrupted mid-copy (SIGTERM -> SystemExit): the worker is still inside the C copy with
        # both connections, and close() would wait on their mutex until the copy ends. The process
        # is exiting; leave them to it, and the caller unlinks the staging file.
        if copier is None or not copier.running():
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


def _query_ro_sqlite(path: Path, fn):
    """Run ``fn(conn)`` on a read-only connection to *path*; return ``(value, None)`` or ``(None, exc)``."""
    conn = None
    try:
        conn = sqlite3.connect(read_only_db_uri(path), uri=True, timeout=1.0)
        return fn(conn), None
    except Exception as exc:  # health: allow BLE001 -- the error is returned to the caller
        return None, exc
    finally:
        _close_quietly(conn)


def verify_sqlite_integrity(
    path: Path, *, check_header: bool = True, run_pragma: bool = True,
    max_bytes: int = DEFAULT_INTEGRITY_CHECK_MAX_BYTES) -> dict:
    """Verify a SQLite database: existence + minimum size, header magic, then a read-only
    ``PRAGMA integrity_check`` (or a cheap structural probe above ``max_bytes``)."""
    def _done(message: str, valid: bool = False, size: Optional[int] = None) -> dict:
        return {"valid": valid, "message": message, "size": size}
    try:
        st = path.stat()
    except FileNotFoundError:
        return _done(f"not found: {path}")
    except OSError as exc:
        return _done(f"cannot stat: {exc}")
    size = st.st_size
    if size < 100:  # SQLite minimum viable size (header + 1 page)
        return _done(f"too small ({size} bytes) to be a valid SQLite database", size=size)
    if check_header:
        # Refused when a live connection exists: close() would cancel this process's POSIX locks
        # (see sqlite_safe_read); verification targets offline snapshots/backup artifacts anyway.
        from hermes_cli.sqlite_safe_read import read_header_bytes_preopen
        head = read_header_bytes_preopen(path, length=len(_SQLITE_HEADER))
        if head is None:
            return _done("cannot read header", size=size)
        if head != _SQLITE_HEADER:
            return _done(f"missing SQLite header magic (got {head[:16].hex()!r})", size=size)
    if max_bytes > 0 and size > max_bytes:
        # O(1) probe: the header check caught the zeroed signature; reading sqlite_master + page
        # geometry catches malformed-schema and truncated-header-page classes without a data walk.
        _, exc = _query_ro_sqlite(path, lambda c: (
            c.execute("PRAGMA schema_version").fetchone(),
            c.execute("SELECT count(*) FROM sqlite_master").fetchone()))
        if exc is not None:
            kind = "failed" if isinstance(exc, sqlite3.DatabaseError) else "error"
            return _done(f"schema probe {kind}: {exc}", size=size)
        return _done(
            f"size {size:,} bytes exceeds max_bytes {max_bytes:,}; "
            "skipped PRAGMA integrity_check (header + schema probe passed)",
            valid=True, size=size)
    if run_pragma:
        rows, exc = _query_ro_sqlite(
            path, lambda c: [str(r[0]) for r in c.execute("PRAGMA integrity_check")])
        if exc is not None:
            kind = "cannot open database" if isinstance(exc, sqlite3.DatabaseError) else "integrity check error"
            return _done(f"{kind}: {exc}", size=size)
        if rows == ["ok"]:
            return _done("integrity check passed", valid=True, size=size)
        return _done(f"integrity check failed: {'; '.join(rows[:5])}", size=size)
    return _done("header check passed", valid=True, size=size)


def _foreign_db_holder_pids(db_path: Path) -> Optional[list[int]]:
    """PIDs of OTHER processes holding *db_path* or its WAL/SHM open.

    Linux-only ``/proc/<pid>/fd`` scan (no psutil dependency), preserving the
    kernel's ``(deleted)`` suffix so an already-unlinked sidecar generation
    (the split-brain fingerprint) still counts as held. Returns
    ``None`` when the scan is unavailable (non-Linux, or /proc unreadable);
    callers must treat ``None`` as "unknown", not as "no holders".
    """
    if not sys.platform.startswith("linux"):
        return None

    def _canonical(path: str) -> str:
        return os.path.normcase(
            os.path.abspath(path.removesuffix(" (deleted)"))
        )

    canonical_db = _canonical(os.fspath(db_path))
    watched = {canonical_db, canonical_db + "-wal", canonical_db + "-shm"}
    pids: list[int] = []
    try:
        own_pid = os.getpid()
        for pid_str in os.listdir("/proc"):
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            if pid == own_pid:
                continue
            fd_dir = f"/proc/{pid}/fd"
            try:
                fds = os.listdir(fd_dir)
            except OSError:
                continue
            for fd in fds:
                try:
                    target = os.readlink(f"{fd_dir}/{fd}")
                except OSError:
                    continue
                if _canonical(target) in watched:
                    pids.append(pid)
                    break
    except OSError:
        return None
    return pids


def _restore_epoch(path: Path) -> int:
    if not path.exists():
        return 0
    with closing(sqlite3.connect(read_only_db_uri(path), uri=True)) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='runtime_epoch'").fetchone():
            return 0
        row = conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton=1").fetchone()
        return int(row[0]) if row else 0


def _restore_destination_epoch(dst: Path) -> int:
    try:
        return _restore_epoch(dst)
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise OSError(
            f"Cannot read the canonical runtime epoch at {dst}; refusing in-place restore. "
            "Use `hermes sessions recover --source <snapshot> --output <separate-path>` "
            "to salvage transcripts into a separate output."
        ) from exc


@contextmanager
def _restore_epoch_source(src: Path, dst: Path):
    """Stage an epoch floor into the image BEFORE publishing it atomically.

    Updating the destination after backup() would leave a crash window that
    reuses an older worker epoch. Never modify the user's source snapshot.
    Full restores retain all admissions/worker receipts; normal startup advances
    this floor and reconciles started work to unknown, not replayable queued work.
    An unreadable destination cannot prove its epoch floor: use transcript salvage
    into a separate output rather than silently restoring with a recycled epoch.
    """
    if dst.name != 'state.db' and dst.resolve().name != 'state.db':
        yield src
        return
    floor = _restore_destination_epoch(dst)
    if floor <= _restore_epoch(src):
        yield src
        return
    with tempfile.TemporaryDirectory(prefix='.restore-epoch-', dir=dst.parent) as work:
        prepared = Path(work) / 'state.db'
        if not _safe_copy_db(src, prepared):
            raise OSError('Unable to stage a restore with a safe runtime epoch')
        with closing(sqlite3.connect(prepared)) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS runtime_epoch (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                epoch INTEGER NOT NULL CHECK (epoch > 0), instance_id TEXT NOT NULL)""")
            conn.execute("""INSERT INTO runtime_epoch VALUES(1,?, 'offline-restore')
                ON CONFLICT(singleton) DO UPDATE SET epoch=excluded.epoch,
                instance_id=excluded.instance_id""", (floor,))
            conn.commit()
        prepared.chmod(src.stat().st_mode)
        yield prepared


def _restore_db_pages(src: Path, dst: Path) -> bool:
    """Restore a SQLite database from snapshot *src* into live *dst*.

    Uses SQLite's backup() API to write snapshot pages into the live
    database file, preserving the file's inode and WAL state so that
    any other process still holding the DB open (gateway, dashboard,
    another CLI session) sees the restored data on the next read,
    instead of continuing to serve stale cached pages from a replaced
    inode. Replacing the file (``unlink() + move()``) under a live
    connection leaves that connection serving its cached pre-restore
    pages while new connections see the snapshot, a partial and
    inconsistent state.

    Falls back to the unlink+move approach on failure ONLY when no other
    process or in-process connection holds the file: replacing the inode
    under a live holder is the split-brain, so that branch fails
    closed (returns ``False``) and the caller reports the file as skipped.
    Callers go through ``restore_from``, which validates *src* and holds
    the maintenance reservation and epoch floor around this publication.
    """
    dst_conn: Optional[sqlite3.Connection] = None
    try:
        dst_conn = sqlite3.connect(str(dst))
        try:
            # Force a WAL checkpoint so the backup starts from a clean
            # state rather than writing on top of a deep WAL.
            dst_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as exc:
            # Optional: backup() still publishes over a deep WAL, just less tidily.
            logger.debug("Pre-restore WAL checkpoint of %s skipped: %s", dst, exc)
        src_conn = sqlite3.connect(read_only_db_uri(src), uri=True)
        try:
            src_conn.backup(dst_conn)
        finally:
            src_conn.close()
        dst_conn.close()
        # Restore original file permissions from the snapshot
        try:
            mode = src.stat().st_mode
            dst.chmod(mode)
        except OSError as exc:
            # The data is restored; only the snapshot's file mode could not be carried over.
            logger.warning("Restored %s but could not apply the snapshot's file mode: %s", dst, exc)
        return True
    except (sqlite3.Error, OSError, ValueError) as exc:  # ValueError: an unrepresentable path
        logger.warning("SQLite safe restore failed for %s -> %s: %s", src, dst, exc)
        # Release our own handle on *dst* before the fallback: on Windows an
        # open connection blocks unlink() with WinError 32, which would make
        # the fallback below dead code exactly when it is needed.
        if dst_conn is not None:
            try:
                dst_conn.close()
            except sqlite3.Error as close_exc:
                logger.debug("Closing %s before the fallback restore failed: %s", dst, close_exc)
        # Fallback: unlink+move.  This still works for
        # the common case where no other process holds the DB open.
        from hermes_cli.sqlite_safe_read import (
            LiveConnectionError,
            offline_file_access,
        )

        try:
            holders = _foreign_db_holder_pids(dst)
            if holders:
                # Replacing the inode under a live holder is the split-brain
                # corruption class: the holder keeps writing through a
                # deleted-inode fd, and removing its sidecars
                # detaches the WAL index it is checkpointing through. The
                # backup-API path above is the live-safe route; if it failed,
                # fail closed rather than corrupt.
                logger.error(
                    "Refusing unlink+move restore of %s: process(es) %s still "
                    "hold the database or its WAL open. Stop them and retry.",
                    dst, holders,
                )
                return False
            # The foreign-pid scan above deliberately excludes THIS process,
            # but an in-process store handle (the agent's own handle during
            # /snapshot restore, a second store handle, a read pool)
            # is exactly as much of a live holder: unlinking the DB and its
            # sidecars under it leaves this process on deleted-inode fds,
            # the same split brain, produced first-party
            # (``/proc/self/fd`` shows ``state.db-wal (deleted)`` right
            # after this fallback runs under a tracked connection).
            # ``offline_file_access`` fails CLOSED when any tracked
            # connection to *dst* is live and holds the connection-lifecycle
            # lock across the whole swap so no new connection can appear
            # mid-replace.
            with offline_file_access(dst, what="unlink+move restore of"):
                tmp = dst.parent / f".{dst.name}.snap_restore"
                shutil.copy2(src, tmp)
                dst.unlink(missing_ok=True)
                # Drop the destination's sidecars before installing the
                # snapshot. The snapshot is a checkpointed ``sqlite3.backup()``
                # image (see ``_safe_copy_db``) that owns no WAL, so any
                # ``-wal``/``-shm`` still sitting here describes the database we
                # just unlinked: an ungracefully killed gateway leaves them
                # behind, which is exactly when a restore gets run. SQLite
                # replays that foreign WAL over the restored file on the next
                # open and the database comes up "malformed" (or silently
                # resurrects post-snapshot rows). The backup excludes the same
                # sidecars for the same reason.
                for _sidecar_suffix in ("-wal", "-shm", "-journal"):
                    dst.with_name(dst.name + _sidecar_suffix).unlink(missing_ok=True)
                shutil.move(str(tmp), str(dst))
            return True
        except LiveConnectionError as exc2:
            logger.error(
                "Refusing unlink+move restore of %s: %s Close the in-process "
                "database handles (or restart Hermes) and retry.",
                dst, exc2,
            )
            return False
        except (OSError, ValueError) as exc2:  # ValueError: a path the OS cannot represent
            logger.error("Fallback restore also failed for %s -> %s: %s", src, dst, exc2)
            return False


class BackupAdmin(AdminFile):
    def backup_to(self, dst: Path, *, timeout_seconds: float = 10.0) -> bool:
        """WAL-safe snapshot of the file into ``dst``; False, with ``dst`` removed, when none can be made."""
        return _safe_copy_db(self.db_path, dst, timeout_seconds=timeout_seconds)

    def preflight_backup(self) -> dict:
        """Publish an emergency snapshot beside the file; do not prune recovery files on failure."""
        source = self.db_path
        home = source.parent
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
            # A copy stopped mid-write leaves the staging copy's rollback journal behind too.
            Path(f"{staged}-journal").unlink(missing_ok=True)
        for old in sorted(home.glob(f"{prefix}*.bak"), reverse=True)[2:]:
            try:
                old.unlink()
            except OSError as exc:
                _backup_sqlite_logger.warning("Could not prune emergency snapshot %s: %s", old, exc)
        return {"path": str(destination), "bytes": size}

    @staticmethod
    def verify_file(path: Path, *, max_bytes: int = DEFAULT_INTEGRITY_CHECK_MAX_BYTES) -> Any:
        """``{valid, message, size}`` for the SQLite file at ``path``: size, header magic, then integrity_check, or a
        cheap structural probe for a file over ``max_bytes``."""
        return verify_sqlite_integrity(path, max_bytes=max_bytes)

    def restore_from(self, snapshot: Path) -> bool:
        """Restore only while holding the destination authority's maintenance reservation.

        Returns ``False``, without touching the file, when *snapshot* fails the SQLite
        integrity check, when the runtime owns the profile, or when a canonical
        ``state.db`` epoch floor cannot be proven.
        """
        self._require_writable("restore_from")
        from gateway.runtime_ownership import OwnershipConflict, exclusive_maintenance

        src, dst = snapshot, self.db_path
        # backup() copies pages without validating their contents; its fallback
        # copies bytes even when SQLite rejected the source. Neither may touch the
        # destination until the snapshot passes the existing bounded integrity policy.
        source_check = verify_sqlite_integrity(src)
        if not source_check["valid"]:
            logger.error("Refusing SQLite restore from %s: %s", src, source_check["message"])
            return False
        try:
            with exclusive_maintenance([dst.absolute().parent, dst.resolve().parent]):
                with _restore_epoch_source(src, dst) as prepared:
                    return _restore_db_pages(prepared, dst)
        except (OwnershipConflict, OSError, sqlite3.Error) as exc:
            logger.error("%s", exc)
            return False

    def restore_epoch_floor(self) -> int:
        """The file's ``runtime_epoch`` on a ``mode=ro`` connection, 0 when the file or the row is absent; an unreadable
        epoch raises ``OSError`` pointing at transcript salvage."""
        return _restore_destination_epoch(self.db_path)

    def session_row_counts(self) -> Optional[tuple[int, int]]:
        """Return ``(sessions, messages)`` stored in the file.

        Read-only and best effort.  ``None`` means "unknown": a missing file, a
        database that is not a Hermes session store, or one that cannot be read.
        Callers must never read ``None`` as "zero rows": acting on an unreadable
        database would mask the very loss this count exists to surface.
        """
        path = self.db_path
        if not path.is_file():
            return None
        try:
            conn = sqlite3.connect(read_only_db_uri(path), uri=True)
        except sqlite3.Error:
            return None
        try:
            sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            return int(sessions), int(messages)
        except (sqlite3.Error, TypeError, ValueError):
            return None
        finally:
            conn.close()

    @staticmethod
    def has_runtime_ledger_rows(path: Path) -> bool:
        """Whether the file at ``path`` holds any runtime ledger row; False when it is missing or unreadable."""
        db_path = path
        if not db_path.is_file():
            return False
        try:
            with closing(sqlite3.connect(read_only_db_uri(db_path), uri=True, timeout=5.0)) as conn:
                return bool(conn.execute(_LEDGER_PROBE).fetchone()[0])
        except sqlite3.Error:
            return False  # no ledger tables yet, or unreadable: nothing an older release would trip on

    @staticmethod
    def has_runtime_ledger_table(path: Path) -> bool:
        """Whether the file at ``path`` holds a ``session_admissions`` table, read on one ``mode=ro`` connection; an
        unreadable file raises the store error."""
        with translated(), closing(sqlite3.connect(read_only_db_uri(path), uri=True)) as conn:
            return conn.execute("SELECT 1 FROM sqlite_master WHERE name='session_admissions'").fetchone() is not None

    @staticmethod
    def recover_transcripts(
        source: Path, output: Path, *, inspect_only: bool = False, work_dir: Optional[Path] = None,
        chunk_size: int = 1_000, progress_cb: Optional[Callable[[dict[str, Any]], None]] = None,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        """Salvage the canonical rows of ``source`` into the new file ``output``, never opening ``source`` itself:
        the snapshot lives under ``work_dir`` (a temporary folder by default), rows are copied ``chunk_size`` at a
        time with ``progress_cb`` called between chunks, and ``allow_partial`` keeps what can be salvaged (row-range
        salvage of damaged tables, the page-level salvage when a canonical schema is unreadable) instead of refusing
        a source that is not fully recoverable.

        ``inspect_only`` reports which canonical tables are readable instead, and writes nothing at ``output``.
        """
        if inspect_only:
            return _recovery.inspect_session_database(source, work_dir=work_dir)
        return _recovery.recover_session_database(
            source, output, work_dir=work_dir, chunk_size=chunk_size, progress_cb=progress_cb,
            allow_partial=allow_partial,
        )
