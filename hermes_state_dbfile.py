"""state.db file-level health helpers, split out of ``hermes_state.py``.

Header probes (application_id / zeroed-file detection), deleted-WAL-sidecar
holder scans, quarantine of zeroed databases, ``collect_state_db_stats`` and
holder-process classification.  Helpers that hermes_state itself imports and
calls (``_connect_tracked_db`` & co) are looked up lazily from ``hermes_state`` at
call time, so tests that monkeypatch ``hermes_state.<name>`` keep intercepting.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_state_common import (
    FTS_REBUILD_DEFERRAL_KEY, stat_db_file_identity as _stat_db_file_identity
)

# Log-record parity with the origin module (caplog tests pin "hermes_state").
logger = logging.getLogger("hermes_state")

# _read_sqlite_application_id runs on EVERY write (_raise_if_db_replaced) against the LIVE
# state.db.  A bare open()/read()/close() there is the howtocorrupt §2.2 bug: close() cancels
# every POSIX advisory lock this process holds on the file, dropping the writer's WAL-mode DMS
# shared lock (see hermes_cli/sqlite_safe_read.py) so another process can treat this writer as
# dead and rerun WAL-index recovery underneath it.  So the probe preads through a per-path fd
# cached for the life of the process (opening never cancels locks).  When the path is re-pointed
# at a new inode (the very replacement this probe detects) the stale fd is RETIRED, never closed
# — closing it would cancel the live connection's locks.  Replacements are rare and halt writes.
_HEADER_PROBE_LOCK = threading.Lock()
_HEADER_PROBE_FDS: "dict[str, tuple[int, int, int]]" = {}  # key -> (fd, dev, ino)
_RETIRED_HEADER_PROBE_FDS: "list[int]" = []  # intentionally never closed
_FTS_TABLE_NAMES = ("messages_fts", "messages_fts_trigram", "messages_fts_cjk")


def _pread_db_header(db_path: Path, length: int) -> "Optional[bytes]":
    """Lock-safe raw header read of a possibly-live SQLite database: POSIX preads from a cached,
    never-closed fd (rebound when the path names a new inode); Windows reads plainly, since
    advisory-lock cancellation is a POSIX-only hazard."""
    from hermes_state import _IS_WINDOWS
    if _IS_WINDOWS:
        with contextlib.suppress(OSError), db_path.open("rb") as handle:
            return handle.read(length)
        return None
    key = str(db_path)
    try:
        st = os.stat(db_path)
    except OSError:
        return None
    with _HEADER_PROBE_LOCK:
        cached = _HEADER_PROBE_FDS.get(key)
        if cached is not None and (cached[1], cached[2]) != (st.st_dev, st.st_ino):
            # Path re-pointed at a new file. Retire (never close) the old fd.
            _RETIRED_HEADER_PROBE_FDS.append(_HEADER_PROBE_FDS.pop(key)[0])
            cached = None
        if cached is None:
            try:
                fd = os.open(db_path, os.O_RDONLY)
            except OSError:
                return None
            try:
                fst = os.fstat(fd)
            except OSError:
                _RETIRED_HEADER_PROBE_FDS.append(fd)
                return None
            cached = _HEADER_PROBE_FDS[key] = (fd, fst.st_dev, fst.st_ino)
        with contextlib.suppress(OSError):
            return os.pread(cached[0], length, 0)
    return None


def _read_sqlite_application_id(db_path: Path) -> "Optional[int]":
    """application_id from the SQLite header, via the lock-safe :func:`_pread_db_header`."""
    from hermes_state_errors import _STATE_DB_APPLICATION_ID_OFFSET
    end = _STATE_DB_APPLICATION_ID_OFFSET + 4
    header = _pread_db_header(db_path, end)
    if header is None or len(header) < end or header[:16] != b"SQLite format 3\x00":
        return None
    return int(struct.unpack(">I", header[_STATE_DB_APPLICATION_ID_OFFSET:end])[0])


def _stat_sqlite_sidecar_identity(db_path: Path) -> Dict[str, tuple]:
    """Snapshot ``(st_dev, st_ino)`` for existing WAL/SHM sidecars."""
    base = os.fspath(db_path)
    idents = {suffix: _stat_db_file_identity(Path(base + suffix)) for suffix in ("-wal", "-shm")}
    return {suffix: ident for suffix, ident in idents.items() if ident is not None}


def _canonical_sqlite_path(path: str) -> str:
    """Normalize a /proc fd target, stripping the Linux `` (deleted)`` suffix."""
    return os.path.normcase(os.path.abspath(path.removesuffix(" (deleted)")))


def _watched_sqlite_sidecar_paths(db_path) -> Set[str]:
    base = os.path.abspath(os.fspath(db_path))
    return {_canonical_sqlite_path(base + "-wal"), _canonical_sqlite_path(base + "-shm")}


def _iter_proc_fd_targets():
    """Yield ``(pid, readlink target)`` for every readable ``/proc/<pid>/fd`` entry."""
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        fd_dir = f"/proc/{pid_str}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue  # process gone or not ours
        for fd in fds:
            with contextlib.suppress(OSError):
                yield int(pid_str), os.readlink(f"{fd_dir}/{fd}")


def iter_deleted_sqlite_sidecar_holders(db_path) -> List[Tuple[int, str]]:
    """Return processes holding an unlinked ``state.db-wal`` / ``-shm``.  Linux-only; ``[]``
    elsewhere (Windows cannot unlink a held sidecar, macOS has no `` (deleted)`` suffix).
    Includes this process: on the open/write refuse path the in-process writer holding the orphan
    inode must not mint a replacement WAL (``_foreign_state_db_holders`` skips this PID)."""
    if not sys.platform.startswith("linux"):
        return []
    holders: List[Tuple[int, str]] = []
    watched = _watched_sqlite_sidecar_paths(db_path)
    try:
        for pid, target in _iter_proc_fd_targets():
            if " (deleted)" in target and _canonical_sqlite_path(target) in watched:
                holders.append((pid, target))
    except Exception as exc:
        logger.debug("deleted-WAL holder scan failed for %s: %s", db_path, exc)
    return holders


def refuse_deleted_wal_generation(db_path) -> None:
    """Raise if any process holds a deleted WAL/SHM generation for *db_path*; called
    *before* ``sqlite3.connect`` so a second opener cannot mint a replacement WAL inode."""
    from hermes_state import DeletedWalGenerationError
    from hermes_state_errors import _DELETED_WAL_GENERATION_MSG
    if not iter_deleted_sqlite_sidecar_holders(db_path):
        return
    logger.error(_DELETED_WAL_GENERATION_MSG)
    raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)


def _connect_tracked_db(path, tracking_path=None, **kwargs):
    """``sqlite3.connect`` that registers the open fd so byte-level probes of a live file are
    refused (an ``open()``/``close()`` would cancel every POSIX lock, even a running VACUUM's
    EXCLUSIVE).  The ONLY tolerated fallback is the helper being absent (scaffold/embed installs
    without hermes_cli); a real connection failure must propagate — a silent untracked retry
    would disable the guard for that connection."""
    try:
        from hermes_cli.sqlite_safe_read import connect_tracked
    except ImportError:
        logger.debug("hermes_cli.sqlite_safe_read unavailable; opening %s untracked "
                     "(byte-probe guard inactive in this install)", path)
        return sqlite3.connect(str(path), **kwargs)
    # Open through THIS module's sqlite3.connect so tests patching hermes_state.sqlite3.connect keep control.
    return connect_tracked(path, tracking_path=tracking_path, connect_fn=sqlite3.connect, **kwargs)


def is_zeroed_state_db(path: Path, *, probe_bytes: int = 100, force: bool = False) -> bool:
    """Detect the zeroed state.db signature (0-byte or NUL header).  Byte-level probe, so only
    safe BEFORE any connection to *path* exists in this process (``close()`` cancels every POSIX
    lock, even a running VACUUM's EXCLUSIVE); ``read_header_bytes_preopen`` refuses (-> False)
    once a connection is live.  Pass ``force=True`` only for offline files (quarantined copies,
    snapshots).  Prefers ``hermes_cli.backup.is_zeroed_sqlite_file``; this copy keeps SessionDB
    openable without the CLI package in constrained embed paths.

    See #97568.
    """
    with contextlib.suppress(Exception):
        from hermes_cli.backup import is_zeroed_sqlite_file
        return is_zeroed_sqlite_file(path, probe_bytes=probe_bytes, force=force)
    try:
        # Special files (FIFO, device, socket) are never "zeroed", and probing
        # a FIFO would block until a writer appears.
        if not path.is_file():
            return False
        path.stat()
    except OSError:
        return False
    from hermes_cli.sqlite_safe_read import has_live_connection, read_header_bytes_preopen
    if not force and has_live_connection(path):
        return False
    head = read_header_bytes_preopen(path, length=max(16, probe_bytes), force=force)
    # b"" (0-byte file) is zeroed; all() over an empty header is True.
    return head is not None and not head.startswith(b"SQLite format 3") and all(b == 0 for b in head)


@contextlib.contextmanager
def quarantine_cross_process_lock(path: Path, timeout: float = 5.0):
    """Acquire the cross-process lock for path.quarantine.lock."""
    import platform
    lock_path = path.with_name(path.name + ".quarantine.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        if platform.system() == "Windows":
            import msvcrt
            def _lock(mode):  # msvcrt locks a byte range from the current position
                handle.seek(0)
                msvcrt.locking(handle.fileno(), mode, 1)

            _try_lock = lambda: _lock(msvcrt.LK_NBLCK)  # noqa: E731
            _unlock = lambda: _lock(msvcrt.LK_UNLCK)  # noqa: E731
        else:
            import fcntl
            _try_lock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # noqa: E731
            _unlock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # noqa: E731
        deadline = time.monotonic() + timeout
        while not acquired:
            try:
                _try_lock()
                acquired = True
            except OSError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.020)
        yield acquired
    finally:
        try:
            if acquired:
                _unlock()
        except (OSError, AttributeError):
            pass
        finally:
            handle.close()


def quarantine_zeroed_state_db(path: Path, *, already_locked: bool = False) -> Optional[Path]:
    """Move a zeroed state.db aside (preserve bytes) and return quarantine path.  A cross-process
    lock stops two concurrent startups racing: the second re-checks under the lock and finds the
    file gone (or fresh) instead of clobbering the quarantine."""
    def _do_quarantine():
        if not path.exists():
            logger.info("quarantine_zeroed_state_db: %s already moved by another process", path)
            return None
        if not is_zeroed_state_db(path):
            logger.info("quarantine_zeroed_state_db: %s is no longer zeroed (another "
                        "process quarantined it and a fresh DB was created)", path)
            return None
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
        except Exception:
            ts = "unknown"
        stem = f"{path.name}.zeroed-{ts}-{os.getpid()}"
        dest = path.with_name(f"{stem}.bak")
        n = 0
        while dest.exists():
            n += 1
            dest = path.with_name(f"{stem}-{n}.bak")
        try:
            path.rename(dest)
        except OSError as exc:
            logger.error("Failed to quarantine zeroed %s: %s", path, exc)
            return None
        for suffix in ("-wal", "-shm"):
            side = Path(str(path) + suffix)
            if side.exists():
                with contextlib.suppress(OSError):
                    side.rename(Path(str(dest) + suffix))
        return dest

    if already_locked:
        return _do_quarantine()
    with quarantine_cross_process_lock(path) as acquired:
        if not acquired:
            logger.error("quarantine lock for %s not acquired within 5s — refusing to "
                         "quarantine without the cross-process lock. The zeroed file "
                         "is left in place. If sessions fail to load, restore from "
                         "state-snapshots via `hermes snapshot list` / `hermes snapshot restore <id>`.",
                         path)
            return None
        return _do_quarantine()


def collect_state_db_stats(db_path: Path) -> Dict[str, Any]:
    """Best-effort, strictly read-only stats snapshot of a state.db file: ``mode=ro`` with a short
    timeout so it can run against a *live* database without taking a write lock.  Every field is
    collected independently (a failed pragma/SELECT yields ``None`` for it); never raises.
    Deliberately does NOT instantiate :class:`SessionDB` — its constructor runs DDL.
    ``wal_size_bytes`` is 0 when the sidecar is absent; ``fts_storage_version`` None means the
    legacy inline layout; ``fts_rebuild_deferral`` is the durable blocked-repair diagnostic."""
    from hermes_state import _connect_tracked_db
    stats: Dict[str, Any] = dict.fromkeys((
        "page_count", "page_size", "freelist_count", "logical_size_bytes", "wal_size_bytes", "journal_mode",
        "messages", "sessions", "fts_tables", "fts_storage_version", "fts_rebuild_pending",
        "fts_rebuild_high_water", "fts_rebuild_progress", "fts_rebuild_deferral"))
    # WAL sidecar size needs no connection at all.
    with contextlib.suppress(OSError):
        wal_path = Path(str(db_path) + "-wal")
        stats["wal_size_bytes"] = wal_path.stat().st_size if wal_path.exists() else 0
    try:
        # A short timeout keeps doctor snappy when a writer holds the lock.  The tracked connect
        # lets byte-probe helpers see this connection and refuse raw opens that would cancel locks.
        conn = _connect_tracked_db(f"file:{Path(db_path)}?mode=ro", tracking_path=Path(db_path),
                                   uri=True, timeout=2.0)
    except Exception as exc:
        logger.debug("collect_state_db_stats: cannot open %s read-only: %s", db_path, exc)
        return stats
    def _scalar(sql: str, params=()) -> Any:
        try:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else None
        except Exception:
            return None
    def _int(sql: str, params=()) -> Optional[int]:
        value = _scalar(sql, params)
        return int(value) if value is not None else None
    def _meta_int(key: str) -> Optional[int]:
        try:  # a non-numeric meta value must yield None, not fail the snapshot
            return _int("SELECT value FROM state_meta WHERE key = ?", (key,))
        except Exception:
            return None

    try:
        stats["page_count"] = _int("PRAGMA page_count")
        stats["page_size"] = _int("PRAGMA page_size")
        if stats["page_count"] is not None and stats["page_size"] is not None:
            stats["logical_size_bytes"] = stats["page_count"] * stats["page_size"]
        stats["freelist_count"] = _int("PRAGMA freelist_count")
        jm = _scalar("PRAGMA journal_mode")
        stats["journal_mode"] = str(jm) if jm is not None else None
        stats["messages"] = _int("SELECT COUNT(*) FROM messages")
        stats["sessions"] = _int("SELECT COUNT(*) FROM sessions")
        # FTS table presence via sqlite_master (never SELECTs from the
        # virtual tables themselves — a corrupt index must not fail stats).
        with contextlib.suppress(Exception):
            names = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?, ?)",
                _FTS_TABLE_NAMES).fetchall()}
            stats["fts_tables"] = {t: (t in names) for t in _FTS_TABLE_NAMES}
        # Raw state_meta reads — cheap, and independent of SessionDB.
        stats["fts_storage_version"] = _meta_int("fts_storage_version")
        stats["fts_rebuild_high_water"] = high_water = _meta_int("fts_rebuild_high_water")
        stats["fts_rebuild_progress"] = progress = _meta_int("fts_rebuild_progress")
        stats["fts_rebuild_pending"] = False if high_water is None else (progress or 0) < high_water
        with contextlib.suppress(Exception):
            row = conn.execute("SELECT value FROM state_meta WHERE key = ? LIMIT 1",
                               (FTS_REBUILD_DEFERRAL_KEY,)).fetchone()
            parsed = json.loads(row[0]) if row else None
            if isinstance(parsed, dict):
                stats["fts_rebuild_deferral"] = parsed
    finally:
        with contextlib.suppress(Exception):
            conn.close()
    return stats


def count_db_holders(db_path: Path) -> Optional[int]:
    """Best-effort count of distinct PIDs holding ``db_path`` open (``/proc/*/fd`` scan); ``None``
    on any error or non-Linux host, never raises.  Unreadable fd dirs (other users' processes
    without root) are skipped, so this is a lower bound."""
    try:
        if not sys.platform.startswith("linux"):
            return None
        target = os.path.realpath(str(db_path))
        return len({pid for pid, link in _iter_proc_fd_targets() if link == target})
    except Exception:
        return None


def _is_inactive_orphan_desktop_holder(*, ppid: int, age_seconds: float, min_age_seconds: float,
                                       ephemeral_backend: bool, connection_statuses: List[str]) -> bool:
    """Pure safety predicate for the narrow Desktop holder reap."""
    return (ppid in (0, 1) and age_seconds >= min_age_seconds and ephemeral_backend
            and "ESTABLISHED" not in connection_statuses)


def _concrete_state_db_holder_pids(db_path: Path, holders: List[Tuple[int, str]]) -> List[int]:
    """Return unique PIDs proven to hold this DB or one of its sidecars."""
    canonical_db = os.path.normcase(os.path.abspath(os.fspath(db_path)))
    watched = {canonical_db, canonical_db + "-wal", canonical_db + "-shm"}
    return list(dict.fromkeys(
        pid for pid, path in holders if pid > 0 and _canonical_sqlite_path(path) in watched))
