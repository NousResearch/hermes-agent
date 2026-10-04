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
import threading
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Progress heartbeats for the desktop updater (state-db-preflight.ts): one
# stderr line per throttle interval while a phase is alive, so the updater can
# tell a slow-but-progressing snapshot from a wedged one and only kill the
# latter (#124972, #124983).
_HEARTBEAT_PREFIX = "PRFL-HB"
_HEARTBEAT_INTERVAL_SECONDS = 5.0
# Exit code for "the watchdog killed us": distinguishable from a quick_check
# corruption failure without the caller parsing stderr.
_NO_PROGRESS_EXIT = 75
# No page of the copy/quick_check advanced for this long -> the process is
# wedged (the backup API cannot be interrupted from another thread, so the
# watchdog must terminate the whole process).
_DEFAULT_STALL_SECONDS = 45.0
# Staging files written before ownership markers existed. Age is the only
# proof their owner is gone; a fresh markerless partial may still belong to a
# copy an older runtime is running right now.
_STALE_PARTIAL_HORIZON_SECONDS = 2 * 3600.0
# A healthy 14.40 GB WAL state.db measured ~124 s to copy and ~46 s to
# quick_check on Windows (#124972). Budgets scale with the source size so a
# fixed cap can never again sit ~6% above a healthy run: the floor keeps small
# stores at the old fixed-cap behavior, the per-GiB slope buys large ones the
# headroom the measured rate implies (~9 s/GiB copy, ~3 s/GiB check).
_COPY_MIN_SECONDS = 180.0
_COPY_PER_GIB_SECONDS = 15.0
_QUICK_CHECK_MIN_SECONDS = 60.0
_QUICK_CHECK_PER_GIB_SECONDS = 8.0
_GIB = 1024 ** 3

_STAGING_PREFIX = "state.db.pre-update-emergency-"
_STAGING_GLOB = _STAGING_PREFIX + "*.partial"


class _SQLiteBackupTimeout(RuntimeError):
    """Raised when a SQLite snapshot remains busy past its deadline."""


class _PreflightNoProgress(RuntimeError):
    """Internal marker for a watchdog termination; never crosses a boundary."""


def _close_quietly(conn: Optional[sqlite3.Connection]) -> None:
    if conn is not None:
        with suppress(Exception):
            conn.close()


def _copy_budget_seconds(source_bytes: int) -> float:
    """Wall-clock ceiling for the snapshot copy, scaled by the source size."""
    return _COPY_MIN_SECONDS + max(0, source_bytes) / _GIB * _COPY_PER_GIB_SECONDS


def _quick_check_budget_seconds(source_bytes: int) -> float:
    """Wall-clock ceiling for the snapshot's PRAGMA quick_check."""
    return _QUICK_CHECK_MIN_SECONDS + max(0, source_bytes) / _GIB * _QUICK_CHECK_PER_GIB_SECONDS


def _heartbeat(message: str) -> None:
    with suppress(Exception):
        sys.stderr.write(f"{_HEARTBEAT_PREFIX} {message}\n")
        sys.stderr.flush()


class _StallWatchdog:
    """Terminate the process when a phase stops making progress.

    ``sqlite3.Connection.backup()`` cannot be interrupted from another thread,
    so a wedged copy would otherwise hang until the updater's external kill —
    which, on Windows, never runs Python cleanup. The watchdog hard-exits
    instead, leaving the staging file for the next run's reclaim.

    Two deadlines: a re-arming idle window (no callback/loop ping for
    ``stall_seconds``) and a phase budget (``budget_seconds`` of wall time from
    ``start()``). A slow copy that keeps advancing outlives any fixed cap; a
    stalled one dies in ``stall_seconds``.
    """

    def __init__(self, *, phase: str, stall_seconds: float, budget_seconds: float) -> None:
        self._phase = phase
        self._stall_seconds = max(0.05, float(stall_seconds))
        self._budget_seconds = max(self._stall_seconds, float(budget_seconds))
        self._lock = threading.Lock()
        self._started: Optional[float] = None
        self._last_ping: Optional[float] = None
        self._last_heartbeat: Optional[float] = None
        self._timers: list = []

    def start(self) -> "_StallWatchdog":
        with self._lock:
            if self._started is not None:
                return self
            self._started = self._last_ping = self._last_heartbeat = time.monotonic()
            self._spawn(self._stall_seconds, self._idle_expired)
            self._spawn(self._budget_seconds, self._budget_expired)
        return self

    def ping(self, report: Optional[str] = None) -> None:
        """Record progress; optionally emit a throttled heartbeat line."""
        with self._lock:
            now = time.monotonic()
            self._last_ping = now
            if report is not None and now - (self._last_heartbeat or 0.0) >= _HEARTBEAT_INTERVAL_SECONDS:
                self._last_heartbeat = now
                _heartbeat(report)

    def stop(self) -> None:
        with self._lock:
            self._started = None
            for timer in self._timers:
                timer.cancel()
            self._timers = []

    def _spawn(self, delay: float, callback: Callable[[], None]) -> None:
        timer = threading.Timer(delay, callback)
        timer.daemon = True
        timer.start()
        self._timers.append(timer)
        # Re-armed idle checks outlive their predecessors; keep only live timers.
        self._timers = [candidate for candidate in self._timers if candidate.is_alive() or candidate is timer]

    def _idle_expired(self) -> None:
        with self._lock:
            if self._started is None:
                return
            idle = time.monotonic() - (self._last_ping or self._started)
            if idle < self._stall_seconds:
                self._spawn(self._stall_seconds - idle, self._idle_expired)
                return
            self._terminate(
                f"the {self._phase} made no progress for {idle:.0f} s and was aborted "
                f"(this is not evidence of corruption; staging was left for the next run to reclaim)"
            )

    def _budget_expired(self) -> None:
        with self._lock:
            if self._started is None:
                return
            elapsed = time.monotonic() - self._started
            self._terminate(
                f"the {self._phase} exceeded its {self._budget_seconds:.0f} s budget after "
                f"{elapsed:.0f} s and was aborted (this is not evidence of corruption; "
                f"staging was left for the next run to reclaim)"
            )

    def _terminate(self, reason: str) -> None:
        with suppress(Exception):
            sys.stderr.write(f"state.db pre-flight aborted: {reason}\n")
            sys.stderr.flush()
        os._exit(_NO_PROGRESS_EXIT)


def _acquire_owner_marker(path: Path):
    """Lock *path* as the ownership marker of a staging file; None if taken.

    The lock is the same primitive as ``backup.py``'s cross-process
    ``.backup.lock`` (flock / msvcrt byte lock). The kernel releases it when the
    owning process dies by ANY means — SIGKILL, Windows TerminateProcess, a
    power cut — which is exactly how a timed-out pre-flight dies. A PID written
    into the file would go stale on PID reuse; a held lock cannot.
    """
    handle = None
    try:
        handle = path.open("a+b")
        if os.name == "nt":
            import msvcrt

            if path.stat().st_size == 0:
                handle.write(b" ")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except OSError:
        _close_quietly(None)
        if handle is not None:
            with suppress(Exception):
                handle.close()
        return None


def _reclaim_abandoned_staging(home: Path) -> None:
    """Delete staging files whose owner is provably gone.

    A timed-out or hard-killed pre-flight cannot clean up after itself (no
    Python runs on TerminateProcess; the interpreter dies on default SIGTERM),
    so this runs first on every pre-flight. A lockable marker next to the
    staging file proves the owner is dead; a markerless partial predates
    ownership markers and is only trusted gone once hours old.
    """
    candidates = set(home.glob(_STAGING_GLOB))
    # Successful Windows runs used to leave only the marker behind; discover
    # those too, but still require the ownership lock before reclaiming them.
    candidates.update(Path(str(marker)[:-len(".owner")])
                      for marker in home.glob(_STAGING_GLOB + ".owner"))
    for staged in candidates:
        marker = Path(str(staged) + ".owner")
        owner = None
        if marker.exists():
            owner = _acquire_owner_marker(marker)
            if owner is None:
                continue  # a live pre-flight owns it right now
        elif not staged.exists() or time.time() - staged.stat().st_mtime <= _STALE_PARTIAL_HORIZON_SECONDS:
            continue  # markerless but young: an old-runtime copy may still be writing it
        with suppress(OSError):
            staged.unlink()
            logger.warning("Reclaimed abandoned state.db pre-flight staging file %s", staged)
        for suffix in ("-journal", "-wal", "-shm"):
            with suppress(OSError):
                Path(str(staged) + suffix).unlink()
        if owner is not None:
            owner.close()
        with suppress(OSError):
            marker.unlink()


def _safe_copy_db(src: Path, dst: Path, *, timeout_seconds: float = 10.0, watchdog: Optional[_StallWatchdog] = None) -> bool:
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

        def _check_backup_progress(status: int, remaining: int, total: int) -> None:
            nonlocal busy_deadline
            now = time.monotonic()
            if status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                if now >= busy_deadline:
                    raise _SQLiteBackupTimeout(f"database remained locked for {timeout_seconds:g} seconds")
            else:
                busy_deadline = now + max(0.0, timeout_seconds)
            if watchdog is not None:
                watchdog.ping(f"copying state.db: {total - remaining} of {total} pages")

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


def _quick_check(path: Path, *, watchdog: Optional[_StallWatchdog] = None, poll_seconds: float = 2.0):
    """Run PRAGMA quick_check with progress pings.

    SQLite offers no cursor-level progress for quick_check, so it runs on a
    worker thread while this loop pings the watchdog: a live check keeps the
    idle window armed, a wedged one gets aborted like a wedged copy.
    """
    result: dict = {}
    failure: list = []

    def _run() -> None:
        try:
            connection = sqlite3.connect(str(path))
            try:
                result["rows"] = connection.execute("PRAGMA quick_check").fetchall()
            finally:
                connection.close()
        except BaseException as exc:  # surfaced on the main thread below
            failure.append(exc)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(poll_seconds)
        if watchdog is not None:
            watchdog.ping(f"quick_check on {path.name} ({path.stat().st_size} bytes)")
    if failure:
        raise failure[0]
    return result.get("rows")


def preflight_state_db(
    home: Path,
    *,
    stall_seconds: float = _DEFAULT_STALL_SECONDS,
    copy_budget_seconds: Optional[float] = None,
    quick_check_budget_seconds: Optional[float] = None,
) -> dict:
    """Publish an emergency snapshot; reclaim staging files a killed run left.

    Budgets scale with the source size (see ``_copy_budget_seconds``); the
    default ``None`` derives them from ``state.db``. A wedged phase aborts the
    process via the watchdog, leaving its staging file for this function's
    next run to reclaim.
    """
    source = home / "state.db"
    if not source.exists():
        return {"path": None, "message": "state.db not found (fresh install?)"}
    _reclaim_abandoned_staging(home)
    source_bytes = source.stat().st_size
    if copy_budget_seconds is None:
        copy_budget_seconds = _copy_budget_seconds(source_bytes)
    if quick_check_budget_seconds is None:
        quick_check_budget_seconds = _quick_check_budget_seconds(source_bytes)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    destination = home / f"{_STAGING_PREFIX}{stamp}-{os.getpid()}.bak"
    fd, name = tempfile.mkstemp(prefix=_STAGING_PREFIX, suffix=".partial", dir=home)
    os.close(fd)
    staged = Path(name)
    owner = _acquire_owner_marker(Path(str(staged) + ".owner"))
    try:
        copy_watchdog = _StallWatchdog(
            phase="state.db snapshot copy",
            stall_seconds=stall_seconds,
            budget_seconds=copy_budget_seconds,
        ).start()
        try:
            if not _safe_copy_db(source, staged, watchdog=copy_watchdog):
                raise RuntimeError("SQLite safe copy failed; previous emergency snapshots were retained")
        finally:
            copy_watchdog.stop()
        check_watchdog = _StallWatchdog(
            phase="snapshot integrity check",
            stall_seconds=stall_seconds,
            budget_seconds=quick_check_budget_seconds,
        ).start()
        try:
            result = _quick_check(staged, watchdog=check_watchdog)
        finally:
            check_watchdog.stop()
        if result != [("ok",)]:
            raise RuntimeError(f"SQLite snapshot integrity check failed: {result}")
        size = staged.stat().st_size
        os.replace(staged, destination)
    finally:
        try:
            staged.unlink(missing_ok=True)
        finally:
            # Windows forbids unlinking the marker while our handle is open.
            # Keep ownership until staging cleanup completes, then release it.
            if owner is not None:
                owner.close()
            with suppress(OSError):
                Path(str(staged) + ".owner").unlink()
    for old in sorted(home.glob(f"{_STAGING_PREFIX}*.bak"), reverse=True)[2:]:
        try:
            old.unlink()
        except OSError as exc:
            logger.warning("Could not prune emergency snapshot %s: %s", old, exc)
    return {"path": str(destination), "bytes": size}


if __name__ == "__main__":
    print(json.dumps(preflight_state_db(Path(sys.argv[1]))))
