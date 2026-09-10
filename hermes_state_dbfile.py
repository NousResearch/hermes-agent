"""state.db file-level health helpers, split out of ``hermes_state.py``.

Header probes (application_id / zeroed-file detection), deleted-WAL-sidecar
holder scans, quarantine of zeroed databases, ``collect_state_db_stats`` and
holder-process classification.  Helpers that hermes_state itself imports and
calls (``_connect_tracked_db`` & co) are looked up lazily from ``hermes_state`` at
call time, so tests that monkeypatch ``hermes_state.<name>`` keep intercepting.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from hermes_state_common import (
    FTS_REBUILD_DEFERRAL_KEY, stat_db_file_identity as _stat_db_file_identity
)

# Log-record parity with the origin module (caplog tests pin "hermes_state").
logger = logging.getLogger("hermes_state")


def _prepare_connection_retirement():
    """Bind a non-finalizing reference before opening a writable SQLite handle.

    A Python container is cleared during interpreter shutdown. An unmatched
    CPython C reference keeps the exact connection alive through that cleanup,
    so SQLite cannot checkpoint its lost WAL generation from a finalizer. This
    deliberately retains the connection and its descriptors until process exit;
    it does not make already-unlinked WAL data durable after the last fd closes.
    """
    message = (
        "Writable SessionDB on a Python without sqlite3 setconfig (< 3.12) requires CPython with "
        "ctypes support to retain a quarantined SQLite connection through interpreter shutdown."
    )
    if sys.implementation.name != "cpython":
        raise RuntimeError(message)
    try:
        import ctypes

        # A private function object avoids changing another caller's signature.
        retain = ctypes.pythonapi["Py_IncRef"]
        retain.argtypes = (ctypes.py_object,)
        retain.restype = None
    except (ImportError, AttributeError, OSError) as exc:
        raise RuntimeError(message) from exc
    return retain


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


def _pread_db_range(db_path: Path, offset: int, length: int) -> "Optional[bytes]":
    """Lock-safe raw read of a possibly-live SQLite database: POSIX preads from a cached,
    never-closed fd (rebound when the path names a new inode); Windows reads plainly, since
    advisory-lock cancellation is a POSIX-only hazard."""
    from hermes_state import _IS_WINDOWS
    if _IS_WINDOWS:
        with contextlib.suppress(OSError), db_path.open("rb") as handle:
            handle.seek(offset)
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
            return os.pread(cached[0], length, offset)
    return None


def _pread_db_header(db_path: Path, length: int) -> "Optional[bytes]":
    """Lock-safe raw header read of a possibly-live SQLite database (see :func:`_pread_db_range`)."""
    return _pread_db_range(db_path, 0, length)


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


def _fd_is_truly_unlinked(fd_path: str) -> bool:
    """Confirm a `` (deleted)`` /proc fd target really lost its last name.

    The suffix alone is not proof: on OpenZFS a live, still-linked file whose
    dentry was unhashed is reported as deleted while ``st_nlink`` is still 1 and
    the path resolves to the very same inode. Only ``st_nlink == 0`` means the
    inode is an orphan generation. An unstattable descriptor counts as deleted
    so the guard keeps failing closed."""
    try:
        return os.stat(fd_path).st_nlink == 0
    except OSError:
        return True


def _iter_proc_fd_targets():
    """Yield ``(pid, readlink target, fd path)`` for every readable ``/proc/<pid>/fd`` entry."""
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
                fd_path = f"{fd_dir}/{fd}"
                yield int(pid_str), os.readlink(fd_path), fd_path


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
        for pid, target, fd_path in _iter_proc_fd_targets():
            if (" (deleted)" in target and _canonical_sqlite_path(target) in watched
                    and _fd_is_truly_unlinked(fd_path)):
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


# ── Retired WAL generation capture ──────────────────────────────────────────────────────────────
#
# When a writer's -wal/-shm generation is deleted or replaced underneath it, the frames committed only in
# that WAL survive exactly as long as this process keeps the unlinked inode open. The quarantine (#105670)
# stops them from being checkpointed under wrong page numbers, but the canonical remediation -- "stop the
# writers, then reopen" -- lets the kernel drop the inode with them. So the exact generation is captured
# durably first: located by the recorded (st_dev, st_ino) among this process's OWN descriptors (never by
# pathname, so a sibling's deleted WAL or a newer sidecar minted at the same path cannot be mistaken for
# it), read with pread (no descriptor is closed, moved or truncated) and written with fsync.

RETIRED_GENERATION_DIR_SUFFIX = ".retired-wal-"
RETIRED_GENERATION_MANIFEST = "manifest.json"
RETIRED_GENERATION_MANIFEST_VERSION = 1
# Up to this size the main image is copied whole, so the artifact is a self-contained state.db + -wal
# pair that `hermes sessions recover --source <dir>/state.db` can open. Above it only the 100-byte
# header is kept (the manifest says so): unlike the WAL inode, the main file survives process exit at
# its path, and a multi-GB copy inside a shutdown path is a worse failure than a header-only artifact.
RETIRED_GENERATION_MAIN_IMAGE_MAX_BYTES = 512 * 1024 * 1024
_CAPTURE_CHUNK_BYTES = 8 * 1024 * 1024
_SQLITE_HEADER_BYTES = 100


class RetiredGenerationCaptureError(RuntimeError):
    """The retired WAL generation could not be captured durably; nothing was mutated."""


def _fsync_path(path: Path) -> None:
    """fsync a file or directory we own. Never used on the live database: opening and closing a
    descriptor on a file SQLite has locked would cancel this process's POSIX advisory locks."""
    from hermes_state import _IS_WINDOWS
    if _IS_WINDOWS:
        return  # directories cannot be opened; file writes fsync their own handle
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _own_descriptor_for_identity(identity) -> "Optional[int]":
    """This process's open descriptor for the inode ``(st_dev, st_ino)``, or None.

    Exact-generation qualified: pathnames are never consulted. The descriptor stays owned by
    SQLite; callers only pread from it."""
    if not identity:
        return None
    wanted = tuple(identity)
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            names = os.listdir(fd_dir)
        except OSError:
            continue
        for name in names:
            if not name.isdigit():
                continue
            try:
                st = os.fstat(int(name))
            except OSError:
                continue
            if (st.st_dev, st.st_ino) == wanted:
                return int(name)
        return None  # the table was readable: a definitive miss
    return None


def _copy_range(read: Callable[[int, int], Optional[bytes]], dest: Path, *, size: int) -> Dict[str, Any]:
    """Stream ``size`` bytes via ``read(offset, length)`` into ``dest`` (temp file, fsync, rename).

    A short read raises ``RetiredGenerationCaptureError``: a truncated copy with a valid-looking
    manifest would let the caller settle the handle while the unlinked inode still dies at exit."""
    digest = hashlib.sha256()
    part = dest.with_name(dest.name + ".part")
    offset = 0
    try:
        with open(part, "wb") as out:
            while offset < size:
                chunk = read(offset, min(_CAPTURE_CHUNK_BYTES, size - offset))
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                offset += len(chunk)
            out.flush()
            os.fsync(out.fileno())
    except Exception:
        part.unlink(missing_ok=True)
        raise
    if offset != size:
        part.unlink(missing_ok=True)
        raise RetiredGenerationCaptureError(
            f"short read copying {dest.name}: {offset} of {size} bytes — the retired generation "
            "is NOT fully captured; the handle stays open for a retry"
        )
    os.replace(part, dest)
    return {"file": dest.name, "bytes": offset, "sha256": digest.hexdigest()}


def _copy_descriptor(fd: int, dest: Path, *, size: int) -> Dict[str, Any]:
    """pread ``fd`` (a descriptor SQLite owns; never closed here) into ``dest``."""
    return _copy_range(lambda offset, length: os.pread(fd, length, offset), dest, size=size)


def _copy_main_image(db_path: Path, dest: Path, *, size: int) -> Dict[str, Any]:
    """Copy the live main file through the lock-safe cached descriptor (see ``_pread_db_range``)."""
    return _copy_range(lambda offset, length: _pread_db_range(db_path, offset, length), dest, size=size)


def _parse_sqlite_header(header: bytes) -> Dict[str, Any]:
    if len(header) < _SQLITE_HEADER_BYTES or header[:16] != b"SQLite format 3\x00":
        return {"valid": False}
    raw_page_size = struct.unpack(">H", header[16:18])[0]
    from hermes_state_errors import _STATE_DB_APPLICATION_ID_OFFSET
    fields = {"change_counter": 24, "page_count": 28, "user_version": 60,
              "application_id": _STATE_DB_APPLICATION_ID_OFFSET, "version_valid_for": 92}
    parsed = {name: struct.unpack(">I", header[off:off + 4])[0] for name, off in fields.items()}
    return {"valid": True, "page_size": 65536 if raw_page_size == 1 else raw_page_size, **parsed}


def capture_retired_wal_generation(
    db_path, *, sidecar_identity: Dict[str, tuple], trigger: str,
) -> Path:
    """Durably capture the lost WAL generation this process still holds open; return the artifact dir.

    The artifact sits next to the database as ``<name>.retired-wal-<utc ts>-<pid>/`` and holds the
    retired ``<name>-wal`` (and ``-shm`` when still ours), the main image (or its header past the size
    cap) and ``manifest.json`` with identities, sizes, digests and the sidecar generation found at the
    path at capture time. Nothing is merged: whether those frames belong on top of the main file is
    the operator's decision. Raises :class:`RetiredGenerationCaptureError` when the exact generation
    cannot be located or written; no descriptor is ever closed, moved or truncated.
    """
    db_path = Path(db_path)
    wal_identity = tuple(sidecar_identity.get("-wal") or ())
    if not wal_identity:
        raise RetiredGenerationCaptureError(
            f"no recorded WAL generation identity for {db_path}; refusing to guess it by pathname")
    wal_fd = _own_descriptor_for_identity(wal_identity)
    if wal_fd is None:
        raise RetiredGenerationCaptureError(
            f"this process no longer holds the retired WAL inode {wal_identity} of {db_path}")
    try:
        wal_size = os.fstat(wal_fd).st_size
    except OSError as exc:
        raise RetiredGenerationCaptureError(f"cannot stat the retired WAL of {db_path}: {exc}") from exc

    stem = f"{db_path.name}{RETIRED_GENERATION_DIR_SUFFIX}{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{os.getpid()}"
    final = db_path.with_name(stem)
    n = 0
    while final.exists() or final.with_name(final.name + ".partial").exists():
        n += 1
        final = db_path.with_name(f"{stem}-{n}")
    staging = final.with_name(final.name + ".partial")
    try:
        staging.mkdir(parents=True, exist_ok=False)
        manifest: Dict[str, Any] = {
            "version": RETIRED_GENERATION_MANIFEST_VERSION,
            "database": str(db_path),
            "trigger": trigger,
            "pid": os.getpid(),
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "sqlite": sqlite3.sqlite_version,
            "wal": {"identity": list(wal_identity),
                    **_copy_descriptor(wal_fd, staging / (db_path.name + "-wal"), size=wal_size)},
            "shm": None,
            # The generation living at the path when we captured: a newer writer's, if one was minted.
            "path_generation_at_capture": {
                suffix: list(ident) for suffix, ident in _stat_sqlite_sidecar_identity(db_path).items()},
            "note": ("Frames in the captured WAL were committed by the retired generation. Whether they "
                     "belong on top of the main file now at the path is an operator decision; inspect "
                     "the copied image with `hermes sessions recover --inspect-only` first."),
        }
        shm_identity = tuple(sidecar_identity.get("-shm") or ())
        shm_fd = _own_descriptor_for_identity(shm_identity) if shm_identity else None
        if shm_fd is not None:
            with contextlib.suppress(OSError):  # SQLite rebuilds the index; the WAL is what matters
                manifest["shm"] = {"identity": list(shm_identity), **_copy_descriptor(
                    shm_fd, staging / (db_path.name + "-shm"), size=os.fstat(shm_fd).st_size)}
        header = _pread_db_range(db_path, 0, _SQLITE_HEADER_BYTES)
        if header is None:
            raise RetiredGenerationCaptureError(f"cannot read the main image header of {db_path}")
        main_size = os.stat(db_path).st_size
        main: Dict[str, Any] = {"identity": list(_stat_db_file_identity(db_path) or ()) or None,
                                "size": main_size, "header": _parse_sqlite_header(header)}
        if main_size <= RETIRED_GENERATION_MAIN_IMAGE_MAX_BYTES:
            main.update(mode="copied", **_copy_main_image(db_path, staging / db_path.name, size=main_size))
        else:
            header_file = staging / (db_path.name + ".header")
            header_file.write_bytes(header)
            _fsync_path(header_file)
            main.update(mode="header_only", file=header_file.name, bytes=len(header))
        manifest["main"] = main
        # Late import: utils pulls agent-side deps; the capture must stay dependency-light.
        from utils import atomic_json_write
        atomic_json_write(staging / RETIRED_GENERATION_MANIFEST, manifest, sort_keys=True)
        _fsync_path(staging)
        os.replace(staging, final)
        _fsync_path(final.parent)
    except RetiredGenerationCaptureError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise RetiredGenerationCaptureError(
            f"could not write the retired WAL generation of {db_path} under {staging}: {exc}") from exc
    return final


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


def _preopen_header(path: Path, probe_bytes: int, force: bool) -> Optional[bytes]:
    """First ``probe_bytes`` of *path* read WITHOUT opening a SQLite connection, or None when the probe
    must not run: special files (a FIFO would block), unreadable paths, or — unless ``force`` — a path
    this process already has a connection to (``close()`` cancels every POSIX lock, even a running
    VACUUM's EXCLUSIVE; see #97568).  ``force=True`` only for offline files (quarantined copies, snapshots)."""
    try:
        if not path.is_file():
            return None
        path.stat()
        from hermes_cli.sqlite_safe_read import has_live_connection, read_header_bytes_preopen
        if not force and has_live_connection(path):
            return None
        return read_header_bytes_preopen(path, length=max(16, probe_bytes), force=force)
    except Exception:
        return None


def is_zeroed_state_db(path: Path, *, probe_bytes: int = 100, force: bool = False) -> bool:
    """Detect the zeroed state.db signature (0-byte or NUL header).  Prefers
    ``hermes_cli.backup.is_zeroed_sqlite_file``; this copy keeps SessionDB openable without the CLI
    package in constrained embed paths."""
    with contextlib.suppress(Exception):
        from hermes_cli.backup import is_zeroed_sqlite_file
        return is_zeroed_sqlite_file(path, probe_bytes=probe_bytes, force=force)
    head = _preopen_header(path, probe_bytes, force)
    # b"" (0-byte file) is zeroed; all() over an empty header is True.
    return head is not None and not head.startswith(b"SQLite format 3") and all(b == 0 for b in head)


def has_invalid_sqlite_header_preopen(path: Path, *, probe_bytes: int = 100, force: bool = False) -> bool:
    """A pre-existing state.db whose first page is not SQLite: 0-byte, NUL, or clobbered page 0
    (#102198).  Zeroed files are the subset :func:`is_zeroed_state_db` names."""
    head = _preopen_header(path, probe_bytes, force)
    return head is not None and not head.startswith(b"SQLite format 3")


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


def quarantine_invalid_state_db(path: Path, *, already_locked: bool = False) -> Optional[Path]:
    """Move a non-SQLite state.db (zeroed or clobbered page 0) aside with its -wal/-shm sidecars,
    preserving bytes; return the quarantine path.  A cross-process lock stops two concurrent
    startups racing: the second re-checks under the lock and finds the file gone (or fresh)."""
    def _do_quarantine():
        if not path.exists():
            logger.info("quarantine_invalid_state_db: %s already moved by another process", path)
            return None
        if not has_invalid_sqlite_header_preopen(path):
            logger.info("quarantine_invalid_state_db: %s is no longer invalid (another "
                        "process quarantined it and a fresh DB was created)", path)
            return None
        kind = "zeroed" if is_zeroed_state_db(path) else "notadb"
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
        except Exception:
            ts = "unknown"
        stem = f"{path.name}.{kind}-{ts}-{os.getpid()}"
        dest = path.with_name(f"{stem}.bak")
        n = 0
        while dest.exists():
            n += 1
            dest = path.with_name(f"{stem}-{n}.bak")
        try:
            path.rename(dest)
        except OSError as exc:
            logger.error("Failed to quarantine invalid %s: %s", path, exc)
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
                         "quarantine without the cross-process lock. The invalid file "
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
        return len({pid for pid, link, _fd_path in _iter_proc_fd_targets() if link == target})
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
