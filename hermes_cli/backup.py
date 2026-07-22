"""
Backup and import commands for hermes CLI.

`hermes backup` creates a zip archive of the entire ~/.hermes/ directory
(excluding the hermes-agent repo and transient files).

`hermes import` restores from a backup zip, overlaying onto the current
HERMES_HOME root.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_default_hermes_root, get_hermes_home, display_hermes_home

logger = logging.getLogger(__name__)


# Backup creation and retention are shared by CLI, gateway, and migration
# processes.  Keep a small in-process lock for threads and a byte-range lock for
# separate processes; both are deliberately stdlib-only so backup safety does
# not depend on an optional package being installed.
try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None
    try:
        import msvcrt
    except ImportError:  # pragma: no cover - unsupported Python platform
        msvcrt = None
else:
    msvcrt = None

_BACKUP_LOCKS: dict[str, threading.RLock] = {}
_BACKUP_LOCKS_GUARD = threading.Lock()
_BACKUP_LOCK_FILENAME = ".maintenance.lock"


def _opened_windows_handle_path(handle: int) -> Optional[Path]:
    """Return the kernel-resolved path for an opened Windows handle."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        get_final_path = ctypes.windll.kernel32.GetFinalPathNameByHandleW
        get_final_path.argtypes = (
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        )
        get_final_path.restype = ctypes.c_uint32
        needed = get_final_path(handle, None, 0, 0)
        if not needed:
            return None
        buffer = ctypes.create_unicode_buffer(needed + 1)
        written = get_final_path(handle, buffer, len(buffer), 0)
        if not written or written >= len(buffer):
            return None
        actual = buffer.value
        if actual.startswith("\\\\?\\UNC\\"):
            actual = "\\\\" + actual[8:]
        elif actual.startswith("\\\\?\\"):
            actual = actual[4:]
        return Path(actual)
    except (AttributeError, OSError, ValueError):
        return None


def _opened_fd_path(fd: int) -> Optional[Path]:
    """Return the kernel-resolved Windows path for an opened descriptor."""
    if os.name != "nt":
        return None
    try:
        return _opened_windows_handle_path(msvcrt.get_osfhandle(fd))
    except (AttributeError, OSError, ValueError):
        return None


def _opened_fd_matches_path(fd: int, expected_path: Path) -> bool:
    """On Windows, prove the handle did not traverse a swapped junction."""
    if os.name != "nt":
        return True
    actual = _opened_fd_path(fd)
    if actual is None:
        return False
    expected = os.path.abspath(os.fspath(expected_path))
    return os.path.normcase(os.path.normpath(os.fspath(actual))) == os.path.normcase(
        os.path.normpath(expected)
    )


def _windows_handle_matches_stat(handle: int, expected: os.stat_result) -> bool:
    """Compare a raw Windows handle without weakening its rename guard."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        class _FileTime(ctypes.Structure):
            _fields_ = [
                ("low", ctypes.c_uint32),
                ("high", ctypes.c_uint32),
            ]

        class _ByHandleFileInformation(ctypes.Structure):
            _fields_ = [
                ("attributes", ctypes.c_uint32),
                ("creation_time", _FileTime),
                ("access_time", _FileTime),
                ("write_time", _FileTime),
                ("volume_serial", ctypes.c_uint32),
                ("size_high", ctypes.c_uint32),
                ("size_low", ctypes.c_uint32),
                ("links", ctypes.c_uint32),
                ("index_high", ctypes.c_uint32),
                ("index_low", ctypes.c_uint32),
            ]

        information = _ByHandleFileInformation()
        get_information = ctypes.windll.kernel32.GetFileInformationByHandle
        get_information.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ByHandleFileInformation),
        )
        get_information.restype = ctypes.c_int
        if not get_information(handle, ctypes.byref(information)):
            return False
        file_index = (information.index_high << 32) | information.index_low
        return bool(
            information.volume_serial == expected.st_dev
            and file_index == expected.st_ino
            and not (information.attributes & 0x400)
        )
    except (AttributeError, OSError, ValueError):
        return False


@contextmanager
def _backup_maintenance_lock(root: Path) -> Iterator[None]:
    """Serialize backup publication and family-specific pruning.

    The lock file is intentionally inside the managed backup/snapshot root and
    is excluded from full archives.  Lock acquisition errors propagate so a
    caller can fail closed rather than pruning without ownership of the
    maintenance operation.
    """
    root.mkdir(parents=True, exist_ok=True)
    root_identity = root.lstat()
    if (
        not stat.S_ISDIR(root_identity.st_mode)
        or int(getattr(root_identity, "st_file_attributes", 0)) & 0x400
    ):
        raise OSError(f"backup maintenance root is not a regular directory: {root}")
    key = str(root.resolve())
    with _BACKUP_LOCKS_GUARD:
        thread_lock = _BACKUP_LOCKS.setdefault(key, threading.RLock())

    with thread_lock:
        lock_path = root / _BACKUP_LOCK_FILENAME
        flags = os.O_RDWR
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        created_lock = False
        try:
            lock_fd = os.open(lock_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
            created_lock = True
        except FileExistsError:
            selected = lock_path.lstat()
            lock_fd = os.open(lock_path, flags)
            opened = os.fstat(lock_fd)
            current = lock_path.lstat()
            if (
                not stat.S_ISREG(selected.st_mode)
                or int(getattr(selected, "st_file_attributes", 0)) & 0x400
                or selected.st_nlink != 1
                or opened.st_nlink != 1
                or not os.path.samestat(selected, opened)
                or not os.path.samestat(opened, current)
            ):
                os.close(lock_fd)
                raise OSError(f"backup maintenance lock is not exclusively owned: {lock_path}")
        lock_file = os.fdopen(lock_fd, "r+b", closefd=True)
        locked = False
        provenance_valid = False
        opened_lock: Optional[os.stat_result] = None
        opened_lock_path: Optional[Path] = None
        try:
            current_root = root.lstat()
            current_lock = lock_path.lstat()
            opened_lock = os.fstat(lock_file.fileno())
            opened_lock_path = _opened_fd_path(lock_file.fileno())
            if (
                not os.path.samestat(root_identity, current_root)
                or not stat.S_ISREG(opened_lock.st_mode)
                or int(getattr(opened_lock, "st_file_attributes", 0)) & 0x400
                or opened_lock.st_nlink != 1
                or not os.path.samestat(opened_lock, current_lock)
                or not _opened_fd_matches_path(lock_file.fileno(), lock_path)
            ):
                raise OSError("backup maintenance lock provenance changed")
            provenance_valid = True
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                locked = True
            elif msvcrt is not None:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                locked = True
            else:  # pragma: no cover - a platform without either primitive
                raise OSError("no supported inter-process file-lock primitive")
            yield
        finally:
            if locked:
                try:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    elif msvcrt is not None:
                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                finally:
                    lock_file.close()
            else:
                lock_file.close()
            if created_lock and not provenance_valid and opened_lock is not None:
                try:
                    cleanup_path = opened_lock_path or lock_path
                    current_created = cleanup_path.lstat()
                    if os.path.samestat(opened_lock, current_created):
                        _remove_owned_snapshot_file(cleanup_path, current_created)
                except OSError:
                    pass


def _unique_archive_path(backup_dir: Path, prefix: str) -> Path:
    """Return a deterministic unused archive path for one backup family."""
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S-%f")
    candidate = backup_dir / f"{prefix}{stamp}.zip"
    suffix = 1
    while candidate.exists():
        candidate = backup_dir / f"{prefix}{stamp}-{suffix}.zip"
        suffix += 1
    return candidate


# ---------------------------------------------------------------------------
# Exclusion rules
# ---------------------------------------------------------------------------

# Directory names to skip entirely (matched against each path component)
# ``hermes-agent`` is special-cased to root level only in ``_should_exclude``
# so that skill directories like ``skills/autonomous-ai-agents/hermes-agent/``
# are not accidentally excluded.
#
# The dependency/cache entries below matter for more than tidiness: without
# them a single plugin venv, MCP-server install, or pip/uv cache living under
# HERMES_HOME gets walked file-by-file, ballooning a backup to hundreds of
# thousands of entries that crawl for hours — the exact "backup stuck for
# days / 426543 files" symptom users hit. The dependency/test-env names mostly
# mirror ``agent.skill_utils.EXCLUDED_SKILL_DIRS`` (the project's canonical
# "regeneratable dir" set); ``.cache`` is an additional backup-only entry, as
# it names a broad regeneratable cache convention (pip/uv/etc.) that the skill
# scanner doesn't need to prune but a backup walk does. We deliberately do NOT
# exclude ``.archive`` here because the curator's ``skills/.archive/`` holds
# restorable user skills that must survive a backup.
_EXCLUDED_DIRS = {
    "hermes-agent",     # the codebase repo — re-clone instead
    "__pycache__",      # bytecode caches — regenerated on import
    ".git",             # nested git dirs (profiles shouldn't have these, but safety)
    "node_modules",     # js deps — reinstalled on demand
    "backups",          # prior auto-backups — don't nest backups exponentially
    "checkpoints",      # session-local trajectory caches — regenerated per-session,
                        # session-hash-keyed so they don't port to another machine anyway
    # Python dependency trees (plugin / MCP-server venvs under HERMES_HOME) —
    # regenerated by reinstalling; never irreplaceable state.
    ".venv",
    "venv",
    "site-packages",
    # Tool / build caches — all regeneratable.
    ".cache",
    ".tox",
    ".nox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

# File-name suffixes to skip
_EXCLUDED_SUFFIXES = (
    ".pyc",
    ".pyo",
    # SQLite sidecar files — the backup takes a consistent snapshot of ``*.db``
    # via ``sqlite3.backup()``, so shipping the live WAL / shared-memory /
    # rollback-journal alongside would pair a fresh snapshot with stale sidecar
    # state and produce a torn restore on the next open. They're transient and
    # regenerated on first connection anyway.
    ".db-wal",
    ".db-shm",
    ".db-journal",
)

# File names to skip (runtime state that's meaningless on another machine)
_EXCLUDED_NAMES = {
    "gateway.pid",
    "cron.pid",
}

# File names that ``hermes import`` must never overwrite, matched by basename so
# they're caught for the root profile (``gateway_state.json``) and for named
# profiles alike (``profiles/<name>/gateway_state.json``).
#
# These hold *volatile gateway/process runtime state that is namespaced to the
# machine or container the backup was taken on* — PIDs in a dead process
# namespace, a runtime lock, the process registry, and the gateway's last
# recorded run/desired state. Restoring them onto a different host (or a hosted
# container) is at best meaningless and at worst actively harmful:
#
#   - ``gateway_state.json`` drives the container-boot reconciler
#     (``container_boot._read_desired_state``), which only auto-starts a
#     gateway whose recorded state is ``running``. A backup taken from a
#     machine where the gateway was stopped (or carrying a stale/foreign
#     value) overwrites the container's own state and leaves the gateway
#     stuck "starting"/"cooking", disconnecting it from the Nous portal
#     (NS-508 / the second half of NS-501).
#   - ``gateway.pid`` / ``cron.pid`` / ``gateway.lock`` / ``processes.json``
#     reference PIDs and locks in the *source* machine's process namespace; a
#     numerically-equal PID in the new environment is a different process.
#     These mirror exactly what ``container_boot._STALE_RUNTIME_FILES`` already
#     sweeps on every container boot.
#
# Older backups predate the backup-side exclusions, so we filter on import too
# rather than trusting the archive's contents.
_IMPORT_SKIP_NAMES = {
    "gateway_state.json",
    "gateway.pid",
    "cron.pid",
    "gateway.lock",
    "processes.json",
}

# zipfile.open() drops Unix mode bits on extract; restore tightens these to 0600.
_SECRET_FILE_NAMES = {".env", "auth.json", "state.db"}

# Reserved archive subtree for provider state that lives OUTSIDE HERMES_HOME
# (e.g. ~/.honcho, ~/.hindsight). The active memory provider declares these via
# MemoryProvider.backup_paths(); they're stored under this prefix encoded
# relative to the user's home directory, and restored to their original
# home-relative location on import. Anything not under home is skipped.
_EXTERNAL_PREFIX = "_external/"


def _collect_memory_provider_external_paths() -> List[Path]:
    """Return existing absolute paths the active memory provider stores
    outside HERMES_HOME, resolved from config only (no network, no init).

    Reads ``memory.provider`` from config, loads just that provider, and asks
    it for ``backup_paths()``. Returns an empty list when no external provider
    is active or the provider can't be loaded — backup must never fail because
    of a flaky plugin.
    """
    try:
        from plugins.memory import _get_active_memory_provider, load_memory_provider
    except Exception:
        return []

    try:
        active = _get_active_memory_provider()
    except Exception:
        active = None
    if not active:
        return []

    try:
        provider = load_memory_provider(active)
    except Exception:
        provider = None
    if provider is None:
        return []

    try:
        declared = provider.backup_paths() or []
    except Exception as exc:
        logger.warning("backup_paths() failed for memory provider %r: %s", active, exc)
        return []

    out: List[Path] = []
    seen: set = set()
    for raw in declared:
        try:
            p = Path(raw).expanduser()
        except Exception:
            continue
        if not p.exists():
            continue
        try:
            resolved = p.resolve()
        except (OSError, ValueError):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(p)
    return out


def _iter_external_files(base: Path) -> List[Path]:
    """Yield regular files under *base* (a file or a directory), skipping
    symlinks, caches, and pyc files. *base* itself may be a file."""
    files: List[Path] = []
    if base.is_file() and not base.is_symlink():
        files.append(base)
        return files
    if not base.is_dir():
        return files
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dp = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]
        for fname in filenames:
            fpath = dp / fname
            if fpath.is_symlink():
                continue
            if fpath.name in _EXCLUDED_NAMES or fpath.name.endswith(_EXCLUDED_SUFFIXES):
                continue
            files.append(fpath)
    return files


def _should_exclude(rel_path: Path) -> bool:
    """Return True if *rel_path* (relative to hermes root) should be skipped."""
    parts = rel_path.parts

    for part in parts:
        if part not in _EXCLUDED_DIRS:
            continue
        # ``hermes-agent`` only matches at the root level (first component).
        # Nested directories with the same name — e.g.
        # ``skills/autonomous-ai-agents/hermes-agent/`` — must be preserved.
        if part == "hermes-agent" and part != parts[0]:
            continue
        return True

    name = rel_path.name

    if name in _EXCLUDED_NAMES:
        return True

    if name.endswith(_EXCLUDED_SUFFIXES):
        return True

    return False


def _should_skip_backup_file(abs_path: Path, rel_path: Path, out_path: Path) -> bool:
    """Return True when a candidate file should not be written to a backup zip."""
    if _should_exclude(rel_path):
        return True

    # zipfile.write() follows file symlinks, so skip links before any archive
    # write can copy data from outside HERMES_HOME.
    if abs_path.is_symlink():
        return True

    try:
        return abs_path.resolve() == out_path.resolve()
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# SQLite safe copy
# ---------------------------------------------------------------------------

def _safe_copy_db(
    src: Path,
    dst: Path,
    *,
    expected_dst_identity: Optional[os.stat_result] = None,
    expected_src_identity: Optional[os.stat_result] = None,
    result_dst_identity: Optional[list[os.stat_result]] = None,
) -> bool:
    """Copy a SQLite database safely using the backup() API.

    Handles WAL mode — produces a consistent snapshot even while
    the DB is being written to. Fail closed if a consistent snapshot cannot
    be created: copying only the live main file can omit committed WAL data.
    When provided, ``result_dst_identity`` receives the final owned pathname
    identity so callers that pre-created ``dst`` can clean it up safely.
    """
    conn = None
    initializer_conn = None
    backup_conn = None
    owner_fd: Optional[int] = None
    source_fd: Optional[int] = None
    source_parent_fd: Optional[int] = None
    cleanup_identity = expected_dst_identity
    published_destination: Optional[os.stat_result] = None
    created_here = False
    success = False
    if result_dst_identity is not None:
        result_dst_identity.clear()
    try:
        if expected_dst_identity is None:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            owner_fd = os.open(dst, flags, 0o600)
            expected_dst_identity = os.fstat(owner_fd)
            cleanup_identity = expected_dst_identity
            created_here = True
            if not _opened_fd_matches_path(owner_fd, dst):
                actual_created = _opened_fd_path(owner_fd)
                os.close(owner_fd)
                owner_fd = None
                if actual_created is not None:
                    try:
                        actual_identity = actual_created.lstat()
                        if os.path.samestat(expected_dst_identity, actual_identity):
                            _remove_owned_snapshot_file(
                                actual_created, actual_identity
                            )
                    except OSError:
                        pass
                raise OSError(f"SQLite destination escaped its parent: {dst}")
        else:
            selected = dst.lstat()
            if (
                not stat.S_ISREG(selected.st_mode)
                or not _same_snapshot_object(selected, expected_dst_identity)
            ):
                logger.warning(
                    "SQLite safe copy refused changed destination %s", dst
                )
                return False
            flags = os.O_RDWR
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            owner_fd = os.open(dst, flags)
            if (
                not _same_snapshot_object(os.fstat(owner_fd), expected_dst_identity)
                or not _opened_fd_matches_path(owner_fd, dst)
            ):
                logger.warning(
                    "SQLite safe copy refused replaced destination %s", dst
                )
                return False

        source_parent = src.parent
        parent_selected = source_parent.lstat()
        if not stat.S_ISDIR(parent_selected.st_mode):
            raise OSError(f"SQLite source parent is not a directory: {source_parent}")
        if os.name != "nt":
            parent_flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                parent_flags |= os.O_DIRECTORY
            if hasattr(os, "O_NOFOLLOW"):
                parent_flags |= os.O_NOFOLLOW
            source_parent_fd = os.open(source_parent, parent_flags)
            if not os.path.samestat(parent_selected, os.fstat(source_parent_fd)):
                raise OSError(f"SQLite source parent changed: {source_parent}")
            parent_monitor = os.fstat(source_parent_fd)
        else:
            parent_monitor = source_parent.lstat()

        source_selected = src.lstat()
        if (
            expected_src_identity is not None
            and not _same_snapshot_object(expected_src_identity, source_selected)
        ):
            raise OSError(f"SQLite source changed after enumeration: {src}")
        source_flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            source_flags |= os.O_BINARY
        if hasattr(os, "O_NOFOLLOW"):
            source_flags |= os.O_NOFOLLOW
        source_fd = os.open(src, source_flags)
        if (
            not stat.S_ISREG(source_selected.st_mode)
            or int(getattr(source_selected, "st_file_attributes", 0)) & 0x400
            or not os.path.samestat(source_selected, os.fstat(source_fd))
            or not _opened_fd_matches_path(source_fd, src)
        ):
            raise OSError(f"SQLite source changed before backup: {src}")

        conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        # The parent/path guard is established before SQLite receives the
        # replaceable pathname. Baselining only after connect would leave a
        # rename/open/restore ABA window in which SQLite could bind another DB.
        conn.execute("PRAGMA schema_version").fetchone()
        parent_after_open = (
            os.fstat(source_parent_fd)
            if source_parent_fd is not None
            else source_parent.lstat()
        )
        parent_path_after_open = source_parent.lstat()
        parent_changed_while_opening = (
            not os.path.samestat(parent_monitor, parent_after_open)
            or not os.path.samestat(parent_after_open, parent_path_after_open)
            or parent_after_open.st_mtime_ns != parent_monitor.st_mtime_ns
            or parent_after_open.st_ctime_ns != parent_monitor.st_ctime_ns
        )
        if not os.path.samestat(source_selected, src.lstat()):
            raise OSError(f"SQLite source changed while opening backup: {src}")
        if parent_changed_while_opening:
            # A WAL database with no live readers may create its shared-memory
            # sidecar on first open. Keep that connection alive only to hold
            # those sidecars, establish a fresh baseline, then open the actual
            # backup connection under a second strict monitor. Any repeated
            # mutation (including rename/open/restore ABA) fails closed.
            initializer_conn = conn
            conn = None
            parent_monitor = parent_after_open
            conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
            conn.execute("PRAGMA schema_version").fetchone()
            second_parent = (
                os.fstat(source_parent_fd)
                if source_parent_fd is not None
                else source_parent.lstat()
            )
            second_parent_path = source_parent.lstat()
            if (
                not os.path.samestat(parent_monitor, second_parent)
                or not os.path.samestat(second_parent, second_parent_path)
                or second_parent.st_mtime_ns != parent_monitor.st_mtime_ns
                or second_parent.st_ctime_ns != parent_monitor.st_ctime_ns
                or not os.path.samestat(source_selected, src.lstat())
            ):
                raise OSError(f"SQLite source changed while opening backup: {src}")

        # Destination provenance is exact because SQLite never receives the
        # caller's pathname: its consistent WAL snapshot is serialized from
        # memory and copied only through the retained destination descriptor.
        backup_conn = sqlite3.connect(":memory:")
        conn.backup(backup_conn)
        serialized = backup_conn.serialize()

        if not os.path.samestat(source_selected, src.lstat()):
            raise OSError(f"SQLite source changed during backup: {src}")
        if source_parent_fd is not None:
            parent_after = os.fstat(source_parent_fd)
            path_parent_after = source_parent.lstat()
            if (
                not os.path.samestat(parent_monitor, parent_after)
                or not os.path.samestat(parent_after, path_parent_after)
                or parent_after.st_mtime_ns != parent_monitor.st_mtime_ns
                or parent_after.st_ctime_ns != parent_monitor.st_ctime_ns
            ):
                raise OSError(
                    f"SQLite source pathname changed during backup: {src}"
                )
        else:
            path_parent_after = source_parent.lstat()
            if (
                not os.path.samestat(parent_monitor, path_parent_after)
                or path_parent_after.st_mtime_ns != parent_monitor.st_mtime_ns
                or path_parent_after.st_ctime_ns != parent_monitor.st_ctime_ns
            ):
                raise OSError(
                    f"SQLite source pathname changed during backup: {src}"
                )

        os.lseek(owner_fd, 0, os.SEEK_SET)
        os.ftruncate(owner_fd, 0)
        serialized_view = memoryview(serialized)
        offset = 0
        while offset < len(serialized_view):
            written = os.write(
                owner_fd, serialized_view[offset : offset + 1024 * 1024]
            )
            if written <= 0:
                raise OSError("short write while copying SQLite snapshot")
            offset += written
        os.fsync(owner_fd)
        success = True
    except Exception as exc:
        logger.warning("SQLite safe copy failed for %s: %s", src, exc)
    finally:
        for connection in (backup_conn, conn, initializer_conn):
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
        if owner_fd is not None:
            try:
                opened_destination = os.fstat(owner_fd)
                path_destination = dst.lstat()
                still_bound = bool(
                    expected_dst_identity is not None
                    and stat.S_ISREG(opened_destination.st_mode)
                    and not (
                        int(getattr(opened_destination, "st_file_attributes", 0))
                        & 0x400
                    )
                    and os.path.samestat(expected_dst_identity, opened_destination)
                    and os.path.samestat(opened_destination, path_destination)
                )
                success = bool(success and still_bound)
                if still_bound:
                    published_destination = opened_destination
            except OSError:
                success = False
            try:
                os.close(owner_fd)
            except OSError:
                pass
        if source_fd is not None:
            try:
                os.close(source_fd)
            except OSError:
                pass
        if source_parent_fd is not None:
            try:
                os.close(source_parent_fd)
            except OSError:
                pass

    try:
        current = dst.lstat()
        still_owned_after_close = bool(
            published_destination is not None
            and stat.S_ISREG(current.st_mode)
            and not (int(getattr(current, "st_file_attributes", 0)) & 0x400)
            and os.path.samestat(published_destination, current)
        )
        success = bool(success and still_owned_after_close)
        if still_owned_after_close:
            # CloseHandle can legitimately publish a final Windows ctime.
            # Bind by the retained fd evidence, then take the exact deletion
            # token from the pathname only after the handle has closed.
            cleanup_identity = current
            if result_dst_identity is not None:
                result_dst_identity.append(current)
    except OSError:
        success = False

    if not success and created_here and cleanup_identity is not None:
        try:
            _remove_owned_snapshot_file(dst, cleanup_identity)
        except OSError:
            pass
    return success


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def _format_size(nbytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def run_backup(args) -> None:
    """Create a zip backup of the Hermes home directory."""
    hermes_root = get_default_hermes_root()

    if not hermes_root.is_dir():
        print(f"Error: Hermes home directory not found at {hermes_root}")
        sys.exit(1)

    # Determine output path
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        # If user gave a directory, put the zip inside it
        if out_path.is_dir():
            stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            out_path = out_path / f"hermes-backup-{stamp}.zip"
    else:
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        out_path = Path.home() / f"hermes-backup-{stamp}.zip"

    # Ensure the suffix is .zip
    if out_path.suffix.lower() != ".zip":
        out_path = out_path.with_suffix(out_path.suffix + ".zip")

    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect files
    print(f"Scanning {display_hermes_home()} ...")
    files_to_add: list[tuple[Path, Path, os.stat_result]] = []
    skipped_dirs = set()
    root_identity = hermes_root.lstat()
    if (
        not stat.S_ISDIR(root_identity.st_mode)
        or int(getattr(root_identity, "st_file_attributes", 0)) & 0x400
    ):
        print(f"Error: Hermes home is not a regular directory: {hermes_root}")
        return

    for dirpath, dirnames, filenames in os.walk(hermes_root, followlinks=False):
        dp = Path(dirpath)
        rel_dir = dp.relative_to(hermes_root)
        directory_identity = dp.lstat()
        if (
            not stat.S_ISDIR(directory_identity.st_mode)
            or int(getattr(directory_identity, "st_file_attributes", 0)) & 0x400
            or (
                dp == hermes_root
                and not _same_snapshot_object(root_identity, directory_identity)
            )
        ):
            dirnames[:] = []
            continue

        # Prune excluded directories in-place so os.walk doesn't descend
        # ``hermes-agent`` is only pruned at the root level; nested dirs
        # with the same name (e.g. in skills/) must be preserved.
        is_root = rel_dir == Path(".")
        orig_dirnames = dirnames[:]
        safe_directories = []
        for dirname in dirnames:
            if dirname in _EXCLUDED_DIRS and not (
                dirname == "hermes-agent" and not is_root
            ):
                continue
            child = dp / dirname
            try:
                child_identity = child.lstat()
            except OSError:
                continue
            if (
                stat.S_ISDIR(child_identity.st_mode)
                and not (
                    int(getattr(child_identity, "st_file_attributes", 0)) & 0x400
                )
            ):
                safe_directories.append(dirname)
        dirnames[:] = safe_directories
        for removed in set(orig_dirnames) - set(dirnames):
            skipped_dirs.add(str(rel_dir / removed))

        for fname in filenames:
            fpath = dp / fname
            rel = fpath.relative_to(hermes_root)

            if _should_skip_backup_file(fpath, rel, out_path):
                continue

            try:
                selected = fpath.lstat()
            except OSError:
                continue
            if (
                not stat.S_ISREG(selected.st_mode)
                or int(getattr(selected, "st_file_attributes", 0)) & 0x400
            ):
                continue
            files_to_add.append((fpath, rel, selected))

    # External memory-provider state (e.g. ~/.honcho, ~/.hindsight) lives
    # outside HERMES_HOME, so the walk above never sees it. Ask the active
    # provider for its declared paths and stage them under the reserved
    # ``_external/`` arc prefix, encoded relative to the user's home dir.
    # Only paths under home are captured (security + portability); anything
    # else is skipped with a note.
    home_dir = Path.home().resolve()
    external_to_add: list[tuple[Path, str, os.stat_result]] = []
    skipped_external: list[str] = []
    for base in _collect_memory_provider_external_paths():
        try:
            base_resolved = base.resolve()
            base_resolved.relative_to(home_dir)
        except (ValueError, OSError):
            skipped_external.append(str(base))
            continue
        for fpath in _iter_external_files(base):
            try:
                rel_to_home = fpath.resolve().relative_to(home_dir)
            except (ValueError, OSError):
                continue
            arcname = _EXTERNAL_PREFIX + rel_to_home.as_posix()
            try:
                selected = fpath.lstat()
            except OSError:
                continue
            if (
                not stat.S_ISREG(selected.st_mode)
                or int(getattr(selected, "st_file_attributes", 0)) & 0x400
            ):
                continue
            external_to_add.append((fpath, arcname, selected))

    if not files_to_add and not external_to_add:
        print("No files to back up.")
        return

    # Create the zip
    file_count = len(files_to_add) + len(external_to_add)
    print(f"Backing up {file_count} files ...")

    total_bytes = 0
    errors = []
    t0 = time.monotonic()

    temp_path: Optional[Path] = None
    temp_identity: Optional[os.stat_result] = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".zip", delete=False, dir=str(out_path.parent)
        ) as tmp_archive:
            temp_path = Path(tmp_archive.name)
        temp_identity = temp_path.lstat()

        with zipfile.ZipFile(
            temp_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            for i, (abs_path, rel_path, source_identity) in enumerate(
                files_to_add, 1
            ):
                try:
                    if time.localtime(source_identity.st_mtime).tm_year < 1980:
                        raise ValueError("ZIP does not support timestamps before 1980")
                    if abs_path.suffix == ".db":
                        with tempfile.NamedTemporaryFile(
                            suffix=".db", delete=False, dir=str(out_path.parent)
                        ) as tmp:
                            tmp_db = Path(tmp.name)
                        tmp_db_identity = tmp_db.lstat()
                        final_tmp_db_identity: list[os.stat_result] = []
                        try:
                            copied = _safe_copy_db(
                                abs_path,
                                tmp_db,
                                expected_dst_identity=tmp_db_identity,
                                expected_src_identity=source_identity,
                                result_dst_identity=final_tmp_db_identity,
                            )
                            if final_tmp_db_identity:
                                tmp_db_identity = final_tmp_db_identity[0]
                            if not copied:
                                errors.append(
                                    f"  {rel_path}: SQLite safe copy failed"
                                )
                                continue
                            total_bytes += _write_bound_file_to_zip(
                                zf,
                                tmp_db,
                                rel_path.as_posix(),
                                expected_identity=tmp_db_identity,
                            )
                        finally:
                            _remove_owned_snapshot_file(tmp_db, tmp_db_identity)
                    else:
                        total_bytes += _write_bound_file_to_zip(
                            zf,
                            abs_path,
                            rel_path.as_posix(),
                            expected_identity=source_identity,
                        )
                except (PermissionError, OSError, ValueError) as exc:
                    errors.append(f"  {rel_path}: {exc}")
                    continue

                if i % 500 == 0:
                    print(f"  {i}/{file_count} files ...")

            for abs_path, arcname, source_identity in external_to_add:
                try:
                    if time.localtime(source_identity.st_mtime).tm_year < 1980:
                        raise ValueError("ZIP does not support timestamps before 1980")
                    total_bytes += _write_bound_file_to_zip(
                        zf,
                        abs_path,
                        arcname,
                        expected_identity=source_identity,
                    )
                except (PermissionError, OSError, ValueError) as exc:
                    errors.append(f"  {arcname}: {exc}")
                    continue
            zf.comment = _archive_ownership_comment(out_path)

        with zipfile.ZipFile(temp_path, "r") as completed:
            if not completed.namelist() or completed.testzip() is not None:
                raise OSError("backup archive integrity check failed")

        candidate = out_path
        for attempt in range(16):
            if attempt:
                candidate = out_path.with_name(
                    f"{out_path.stem}-{uuid.uuid4().hex[:12]}{out_path.suffix}"
                )
                with zipfile.ZipFile(temp_path, "a") as completed:
                    completed.comment = _archive_ownership_comment(candidate)
            with open(temp_path, "r+b") as archive_file:
                os.fsync(archive_file.fileno())
                temp_identity = os.fstat(archive_file.fileno())
            try:
                if os.name == "nt":
                    os.rename(temp_path, candidate)
                    published_via_link = False
                else:
                    os.link(temp_path, candidate, follow_symlinks=False)
                    published_via_link = True
            except FileExistsError:
                continue
            published = candidate.lstat()
            if (
                not stat.S_ISREG(published.st_mode)
                or not os.path.samestat(temp_identity, published)
            ):
                raise OSError("published backup identity changed")
            if published_via_link:
                if _remove_owned_snapshot_file(temp_path, temp_identity):
                    temp_path = None
                    temp_identity = None
            else:
                temp_path = None
                temp_identity = None
            out_path = candidate
            break
        else:
            raise OSError("backup destination kept colliding")
    finally:
        if temp_path is not None and temp_identity is not None:
            try:
                current_temp = temp_path.lstat()
                if os.path.samestat(temp_identity, current_temp):
                    temp_identity = current_temp
            except OSError:
                pass
            _remove_owned_snapshot_file(temp_path, temp_identity)

    elapsed = time.monotonic() - t0
    zip_size = out_path.stat().st_size

    # Summary
    print()
    if errors:
        print(f"Backup incomplete: {out_path}")
    else:
        print(f"Backup complete: {out_path}")
    print(f"  Files:       {file_count}")
    print(f"  Original:    {_format_size(total_bytes)}")
    print(f"  Compressed:  {_format_size(zip_size)}")
    print(f"  Time:        {elapsed:.1f}s")

    if external_to_add:
        print(
            f"\n  Included {len(external_to_add)} memory-provider file(s) "
            f"stored outside {display_hermes_home()}."
        )

    if skipped_external:
        print(
            f"\n  Skipped {len(skipped_external)} memory-provider path(s) "
            f"outside your home directory (not portable):"
        )
        for p in sorted(skipped_external)[:10]:
            print(f"    {p}")

    if skipped_dirs:
        print("\n  Excluded directories:")
        for d in sorted(skipped_dirs):
            print(f"    {d}/")

    if errors:
        print(f"\n  Warnings ({len(errors)} files skipped):")
        for e in errors[:10]:
            print(e)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    if not errors:
        print(f"\nRestore with: hermes import {out_path.name}")


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def _validate_backup_zip(zf: zipfile.ZipFile) -> tuple[bool, str]:
    """Check that a zip looks like a Hermes backup.

    Returns (ok, reason).
    """
    names = zf.namelist()
    if not names:
        return False, "zip archive is empty"

    # Look for telltale files that a hermes home would have
    markers = {"config.yaml", ".env", "state.db"}
    found = set()
    for n in names:
        # Could be at the root or one level deep (if someone zipped the directory)
        basename = Path(n).name
        if basename in markers:
            found.add(basename)

    if not found:
        return False, (
            "zip does not appear to be a Hermes backup "
            "(no config.yaml, .env, or state databases found)"
        )

    return True, ""


def _detect_prefix(zf: zipfile.ZipFile) -> str:
    """Detect if the zip has a common directory prefix wrapping all entries.

    Some tools zip as `.hermes/config.yaml` instead of `config.yaml`.
    Returns the prefix to strip (empty string if none).
    """
    names = [n for n in zf.namelist() if not n.endswith("/")]
    if not names:
        return ""

    # Find common prefix
    parts_list = [Path(n).parts for n in names]

    # Check if all entries share a common first directory
    first_parts = {p[0] for p in parts_list if len(p) > 1}
    if len(first_parts) == 1:
        prefix = first_parts.pop()
        # Only strip if it looks like a hermes dir name
        if prefix in {".hermes", "hermes"}:
            return prefix + "/"

    return ""


def run_import(args) -> None:
    """Restore a Hermes backup from a zip file."""
    zip_path = Path(args.zipfile).expanduser().resolve()

    if not zip_path.is_file():
        print(f"Error: File not found: {zip_path}")
        sys.exit(1)

    if not zipfile.is_zipfile(zip_path):
        print(f"Error: Not a valid zip file: {zip_path}")
        sys.exit(1)

    hermes_root = get_default_hermes_root()

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Validate
        ok, reason = _validate_backup_zip(zf)
        if not ok:
            print(f"Error: {reason}")
            sys.exit(1)

        prefix = _detect_prefix(zf)
        members = [n for n in zf.namelist() if not n.endswith("/")]
        file_count = len(members)

        print(f"Backup contains {file_count} files")
        print(f"Target: {display_hermes_home()}")

        if prefix:
            print(f"Detected archive prefix: {prefix!r} (will be stripped)")

        # Check for existing installation
        has_config = (hermes_root / "config.yaml").exists()
        has_env = (hermes_root / ".env").exists()

        if (has_config or has_env) and not args.force:
            print()
            print("Warning: Target directory already has Hermes configuration.")
            print("Importing will overwrite existing files with backup contents.")
            print()
            try:
                answer = input("Continue? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(1)
            if answer not in {"y", "yes"}:
                print("Aborted.")
                return

        # Extract
        print(f"\nImporting {file_count} files ...")
        hermes_root.mkdir(parents=True, exist_ok=True)

        errors = []
        restored = 0
        restored_external = 0
        skipped_runtime: list[str] = []
        home_dir = Path.home().resolve()
        t0 = time.monotonic()

        for member in members:
            # External memory-provider state captured under the reserved
            # ``_external/`` arc prefix restores to its original home-relative
            # location (e.g. ~/.honcho/config.json), NOT under HERMES_HOME.
            if member.startswith(_EXTERNAL_PREFIX):
                ext_rel = member[len(_EXTERNAL_PREFIX):]
                if not ext_rel:
                    continue
                target = home_dir / ext_rel
                # Security: the resolved target must stay under the home dir.
                try:
                    target.resolve().relative_to(home_dir)
                except ValueError:
                    errors.append(f"  {member}: path traversal blocked")
                    continue
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    # External provider configs commonly hold credentials.
                    if target.suffix in {".json", ".env", ".conf"} or target.name in _SECRET_FILE_NAMES:
                        try:
                            os.chmod(target, 0o600)
                        except OSError:
                            pass
                    restored += 1
                    restored_external += 1
                except (PermissionError, OSError) as exc:
                    errors.append(f"  {member}: {exc}")
                if restored % 500 == 0:
                    print(f"  {restored}/{file_count} files ...")
                continue

            # Strip prefix if detected
            if prefix and member.startswith(prefix):
                rel = member[len(prefix):]
            else:
                rel = member

            if not rel:
                continue

            # Never overwrite volatile gateway/process runtime state. These are
            # namespaced to the machine/container the backup was taken on;
            # clobbering them (especially gateway_state.json) breaks the gateway
            # reconciler on the target and disconnects hosted instances from the
            # Nous portal. Matched by basename so both the root profile and
            # named profiles (profiles/<name>/gateway_state.json) are covered.
            if Path(rel).name in _IMPORT_SKIP_NAMES:
                skipped_runtime.append(rel)
                continue

            target = hermes_root / rel

            # Security: reject absolute paths and traversals
            try:
                target.resolve().relative_to(hermes_root.resolve())
            except ValueError:
                errors.append(f"  {rel}: path traversal blocked")
                continue

            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                if target.name in _SECRET_FILE_NAMES:
                    os.chmod(target, 0o600)
                restored += 1
            except (PermissionError, OSError) as exc:
                errors.append(f"  {rel}: {exc}")

            if restored % 500 == 0:
                print(f"  {restored}/{file_count} files ...")

        elapsed = time.monotonic() - t0

        # Summary
        print()
        print(f"Import complete: {restored} files restored in {elapsed:.1f}s")
        print(f"  Target: {display_hermes_home()}")

        if restored_external:
            print(
                f"\n  Restored {restored_external} memory-provider file(s) to "
                f"their original location(s) outside {display_hermes_home()}."
            )

        if errors:
            print(f"\n  Warnings ({len(errors)} files skipped):")
            for e in errors[:10]:
                print(e)
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")

        if skipped_runtime:
            print(
                f"\n  Preserved {len(skipped_runtime)} runtime state "
                f"file(s) (kept this machine's, not the backup's):"
            )
            for rel in sorted(skipped_runtime)[:10]:
                print(f"    {rel}")
            if len(skipped_runtime) > 10:
                print(f"    ... and {len(skipped_runtime) - 10} more")

        # Post-import: restore profile wrapper scripts
        profiles_dir = hermes_root / "profiles"
        restored_profiles = []
        if profiles_dir.is_dir():
            try:
                from hermes_cli.profiles import (
                    create_wrapper_script, check_alias_collision,
                    _is_wrapper_dir_in_path, _get_wrapper_dir,
                )
                for entry in sorted(profiles_dir.iterdir()):
                    if not entry.is_dir():
                        continue
                    profile_name = entry.name
                    # Only create wrappers for directories with config
                    if not (entry / "config.yaml").exists() and not (entry / ".env").exists():
                        continue
                    collision = check_alias_collision(profile_name)
                    if collision:
                        print(f"  Skipped alias '{profile_name}': {collision}")
                        restored_profiles.append((profile_name, False))
                    else:
                        wrapper = create_wrapper_script(profile_name)
                        restored_profiles.append((profile_name, wrapper is not None))

                if restored_profiles:
                    created = [n for n, ok in restored_profiles if ok]
                    skipped = [n for n, ok in restored_profiles if not ok]
                    if created:
                        print(f"\n  Profile aliases restored: {', '.join(created)}")
                    if skipped:
                        print(f"  Profile aliases skipped:  {', '.join(skipped)}")
                    if not _is_wrapper_dir_in_path():
                        print(f"\n  Note: {_get_wrapper_dir()} is not in your PATH.")
                        print('  Add to your shell config (~/.bashrc or ~/.zshrc):')
                        print('    export PATH="$HOME/.local/bin:$PATH"')
            except ImportError:
                # hermes_cli.profiles might not be available (fresh install)
                if any(profiles_dir.iterdir()):
                    print("\n  Profiles detected but aliases could not be created.")
                    print("  Run: hermes profile list  (after installing hermes)")

        # Guidance
        print()
        if not (hermes_root / "hermes-agent").is_dir():
            print("Note: The hermes-agent codebase was not included in the backup.")
            print("  If this is a fresh install, run: hermes update")

        if restored_profiles:
            gw_profiles = [n for n, _ in restored_profiles]
            print("\nTo re-enable gateway services for profiles:")
            for pname in gw_profiles:
                print(f"  hermes -p {pname} gateway install")

        print("Done. Your Hermes configuration has been restored.")


# ---------------------------------------------------------------------------
# Quick state snapshots (used by /snapshot slash command and hermes backup --quick)
# ---------------------------------------------------------------------------

# Critical state files to include in quick snapshots (relative to HERMES_HOME).
# Everything else is either regeneratable (logs, cache) or managed separately
# (skills, repo, sessions/).
#
# Entries may be individual files OR directories.  Directories are captured
# recursively; missing entries are silently skipped.  Pairing data lives in
# platform-specific JSON blobs outside state.db, so it's listed here explicitly
# — `hermes update` snapshots this set before pulling so approved-user lists
# are recoverable if anything goes wrong (issue #15733).
_QUICK_STATE_FILES = (
    "state.db",
    "config.yaml",
    ".env",
    "auth.json",
    "cron/jobs.json",
    "cron/executions.db",
    "gateway_state.json",
    "channel_directory.json",
    "channel_aliases.json",
    "processes.json",
    "gateway/discord_message_recovery.db",  # Discord reconnect replay ledger
    # Per-profile user-created stores that live outside the git checkout and
    # are therefore destroyed if the update flow removes/replaces the file and
    # the post-update schema-init re-creates an empty one (issue #52889). All
    # are at $HERMES_HOME/<name> for the default/root profile; on non-root
    # profiles the real path is outside HERMES_HOME and the entry is silently
    # skipped (best-effort, same as the pairing stores). SQLite DBs are copied
    # WAL-safely via _safe_copy_db.
    "projects.db",                      # per-profile project store
    "response_store.db",                # gateway conversation history / tool payloads
    "memory_store.db",                  # holographic memory facts/entities
    "verification_evidence.db",         # agent verification audit trail
    "kanban.db",                        # default board (back-compat <root>/kanban.db)
    "kanban/boards",                    # non-default boards: each <slug>/kanban.db + board metadata (workspaces/ + attachments/ are skipped as regenerable)
    # Pairing stores (generic + per-platform JSONs outside state.db)
    "pairing",                          # legacy location (gateway/pairing.py)
    "platforms/pairing",                # new location (gateway/pairing.py)
    "feishu_comment_pairing.json",      # Feishu comment subscription pairings
)

_QUICK_SNAPSHOTS_DIR = "state-snapshots"
_QUICK_DEFAULT_KEEP = 20
_QUICK_IN_PROGRESS_MAX_AGE_SECONDS = 24 * 60 * 60
_QUICK_IN_PROGRESS_MARKER = ".in-progress"
_QUICK_OWNERSHIP_MARKER = ".hermes-managed"


def _quick_snapshot_label_segment(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    raw = str(label)
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in raw)
    safe = safe.strip("_-") or "snapshot"
    if safe != raw or len(safe) > 48:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        safe = f"{safe[:32]}-{digest}"
    return safe


def _move_owned_snapshot_path(path: Path, expected: os.stat_result) -> Optional[Path]:
    """Move the exact expected object to a nonce sibling, or preserve it there."""
    quarantine = path.with_name(f".{path.name}.hermes-delete-{uuid.uuid4().hex}")
    try:
        current = path.lstat()
        if not _same_snapshot_object(expected, current):
            return None
        os.replace(path, quarantine)
        moved = quarantine.lstat()
        if _same_snapshot_object(expected, moved):
            return quarantine
    except OSError:
        return None
    return None


def _same_snapshot_object(
    expected: os.stat_result,
    current: os.stat_result,
) -> bool:
    """Compare stable object evidence, including Windows creation time."""
    return bool(
        os.path.samestat(expected, current)
        and (os.name != "nt" or expected.st_ctime_ns == current.st_ctime_ns)
    )


def _remove_owned_snapshot_file(path: Path, expected: os.stat_result) -> bool:
    quarantine = _move_owned_snapshot_path(path, expected)
    if quarantine is None:
        return False
    try:
        quarantine.unlink()
        return True
    except OSError:
        return False


def _remove_owned_snapshot_dir(
    path: Path,
    expected: os.stat_result,
    *,
    marker_name: str,
    marker_identity: os.stat_result,
    marker_payload: Dict[str, Any],
) -> bool:
    quarantine = _move_owned_snapshot_path(path, expected)
    if quarantine is None:
        return False
    try:
        moved_marker = quarantine / marker_name
        if (
            not os.path.samestat(marker_identity, moved_marker.lstat())
            or json.loads(moved_marker.read_text(encoding="utf-8"))
            != marker_payload
            or (
                marker_name == _QUICK_OWNERSHIP_MARKER
                and os.path.lexists(quarantine / _QUICK_IN_PROGRESS_MARKER)
            )
        ):
            return False
        shutil.rmtree(quarantine)
        return True
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def _snapshot_path_is_link_like(path: Path) -> bool:
    try:
        st = path.lstat()
        reparse = int(getattr(st, "st_file_attributes", 0)) & 0x400
        return path.is_symlink() or bool(reparse)
    except OSError:
        return True


def _quick_snapshot_marker_evidence(
    snapshot_dir: Path, marker_name: str
) -> Optional[tuple[Dict[str, Any], os.stat_result, os.stat_result]]:
    marker = snapshot_dir / marker_name
    fd: Optional[int] = None
    try:
        directory_stat = snapshot_dir.lstat()
        marker_stat = marker.lstat()
        directory_reparse = int(
            getattr(directory_stat, "st_file_attributes", 0)
        ) & 0x400
        marker_reparse = int(getattr(marker_stat, "st_file_attributes", 0)) & 0x400
        if (
            directory_reparse
            or marker_reparse
            or not stat.S_ISDIR(directory_stat.st_mode)
            or not stat.S_ISREG(marker_stat.st_mode)
        ):
            return None
        flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(marker, flags)
        opened_marker_stat = os.fstat(fd)
        if (
            not stat.S_ISREG(opened_marker_stat.st_mode)
            or not os.path.samestat(marker_stat, opened_marker_stat)
        ):
            return None
        with os.fdopen(fd, "r", encoding="utf-8") as marker_file:
            fd = None
            payload = json.load(marker_file)
        token = payload.get("owner_token") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or payload.get("snapshot_id") != snapshot_dir.name
            or not isinstance(payload.get("pid"), int)
            or not isinstance(token, str)
            or len(token) != 32
            or any(char not in "0123456789abcdef" for char in token)
        ):
            return None
        if (
            not os.path.samestat(directory_stat, snapshot_dir.lstat())
            or not os.path.samestat(marker_stat, marker.lstat())
        ):
            return None
        return payload, directory_stat, marker_stat
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    finally:
        if fd is not None:
            os.close(fd)


def _quick_snapshot_marker_payload(
    snapshot_dir: Path, marker_name: str
) -> Optional[Dict[str, Any]]:
    evidence = _quick_snapshot_marker_evidence(snapshot_dir, marker_name)
    return evidence[0] if evidence is not None else None


def _snapshot_process_is_running(pid: int) -> bool:
    """Fail closed when an in-progress snapshot owner may still be alive."""
    if pid <= 0:
        return True
    if os.name == "nt":
        import ctypes

        open_process = ctypes.windll.kernel32.OpenProcess
        open_process.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        open_process.restype = ctypes.c_void_p
        handle = open_process(0x00100000, False, pid)
        if not handle:
            return ctypes.windll.kernel32.GetLastError() != 87
        wait_for_single_object = ctypes.windll.kernel32.WaitForSingleObject
        wait_for_single_object.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        wait_for_single_object.restype = ctypes.c_uint32
        close_handle = ctypes.windll.kernel32.CloseHandle
        close_handle.argtypes = (ctypes.c_void_p,)
        close_handle.restype = ctypes.c_int
        try:
            return wait_for_single_object(handle, 0) == 258
        finally:
            close_handle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (OSError, PermissionError):
        return True


def _quick_snapshot_root(hermes_home: Optional[Path] = None) -> Path:
    home = hermes_home or get_hermes_home()
    return home / _QUICK_SNAPSHOTS_DIR


def _snapshot_directory_is_bound(
    snapshot_dir: Path,
    expected: os.stat_result,
) -> bool:
    try:
        current = snapshot_dir.lstat()
        return bool(
            stat.S_ISDIR(current.st_mode)
            and not (int(getattr(current, "st_file_attributes", 0)) & 0x400)
            and _same_snapshot_object(expected, current)
        )
    except OSError:
        return False


def _snapshot_parent_is_bound(
    snapshot_dir: Path,
    expected: os.stat_result,
    relative_parent: PurePosixPath,
) -> bool:
    if not _snapshot_directory_is_bound(snapshot_dir, expected):
        return False
    current = snapshot_dir
    for part in relative_parent.parts:
        if part in ("", ".", ".."):
            return False
        current = current / part
        try:
            selected = current.lstat()
        except OSError:
            return False
        if (
            not stat.S_ISDIR(selected.st_mode)
            or int(getattr(selected, "st_file_attributes", 0)) & 0x400
        ):
            return False
    return _snapshot_directory_is_bound(snapshot_dir, expected)


def _snapshot_parent_chain(
    snapshot_dir: Path,
    expected: os.stat_result,
    relative_parent: PurePosixPath,
) -> Optional[list[tuple[Path, os.stat_result]]]:
    if not _snapshot_directory_is_bound(snapshot_dir, expected):
        return None
    chain: list[tuple[Path, os.stat_result]] = [(snapshot_dir, expected)]
    current = snapshot_dir
    for part in relative_parent.parts:
        if part in ("", ".", ".."):
            return None
        current = current / part
        try:
            selected = current.lstat()
        except OSError:
            return None
        if (
            not stat.S_ISDIR(selected.st_mode)
            or int(getattr(selected, "st_file_attributes", 0)) & 0x400
        ):
            return None
        chain.append((current, selected))
    return chain


def _snapshot_parent_chain_is_bound(
    chain: list[tuple[Path, os.stat_result]],
) -> bool:
    try:
        for current, expected in chain:
            selected = current.lstat()
            if (
                not stat.S_ISDIR(selected.st_mode)
                or int(getattr(selected, "st_file_attributes", 0)) & 0x400
                or not _same_snapshot_object(expected, selected)
            ):
                return False
        return True
    except OSError:
        return False


def _open_bound_snapshot_directory(path: Path) -> int:
    """Open a non-reparse directory, denying replacement on Windows."""
    if os.name != "nt":
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        return os.open(path, flags)

    import ctypes

    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file.restype = ctypes.c_void_p
    handle = create_file(
        str(path),
        0x00000080,  # FILE_READ_ATTRIBUTES
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,  # OPEN_EXISTING
        0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError()
    # Keep this as a raw handle. It remains bound to the selected directory
    # even when another process renames that directory and replaces its path.
    return handle


@contextmanager
def _hold_snapshot_parent_chain(
    chain: list[tuple[Path, os.stat_result]],
) -> Iterator[int]:
    """Hold the exact restore parent chain through atomic publication."""
    opened: list[int] = []
    try:
        for path, expected in chain:
            fd = _open_bound_snapshot_directory(path)
            opened.append(fd)
            selected = path.lstat()
            if os.name == "nt":
                if (
                    int(getattr(selected, "st_file_attributes", 0)) & 0x400
                    or not stat.S_ISDIR(selected.st_mode)
                    or not _same_snapshot_object(expected, selected)
                    or not _windows_handle_matches_stat(fd, expected)
                ):
                    raise OSError(
                        "snapshot parent changed while binding publication"
                    )
            else:
                actual = os.fstat(fd)
                if (
                    not stat.S_ISDIR(actual.st_mode)
                    or not _same_snapshot_object(expected, selected)
                    or not _same_snapshot_object(expected, actual)
                ):
                    raise OSError(
                        "snapshot parent changed while binding publication"
                    )
        if not opened or not _snapshot_parent_chain_is_bound(chain):
            raise OSError("snapshot parent changed after binding publication")
        yield opened[-1]
    finally:
        if os.name == "nt":
            import ctypes

            for handle in reversed(opened):
                ctypes.windll.kernel32.CloseHandle(handle)
        else:
            for fd in reversed(opened):
                os.close(fd)


def _bound_parent_stat(parent_fd: int, parent: Path, name: str) -> os.stat_result:
    if os.name == "nt":
        return (parent / name).lstat()
    return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)


def _remove_bound_snapshot_member(
    parent_fd: int,
    parent: Path,
    name: str,
    expected: os.stat_result,
) -> bool:
    """Remove only the exact staged member from an already-bound parent."""
    if os.name == "nt":
        return _remove_owned_snapshot_file(parent / name, expected)
    quarantine = f".{name}.hermes-delete-{uuid.uuid4().hex}"
    try:
        current = _bound_parent_stat(parent_fd, parent, name)
        if not _same_snapshot_object(expected, current):
            return False
        os.replace(
            name,
            quarantine,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        moved = _bound_parent_stat(parent_fd, parent, quarantine)
        if not _same_snapshot_object(expected, moved):
            return False
        os.unlink(quarantine, dir_fd=parent_fd)
        return True
    except OSError:
        return False


def _bound_snapshot_member_sha256(
    parent_fd: int,
    parent: Path,
    name: str,
    expected: os.stat_result,
) -> str:
    """Hash one exact staged member through its already-bound parent."""
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if os.name == "nt":
        fd = os.open(parent / name, flags)
    else:
        fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(fd)
        selected = _bound_parent_stat(parent_fd, parent, name)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not _same_snapshot_object(expected, opened)
            or not _same_snapshot_object(opened, selected)
            or not _opened_fd_matches_path(fd, parent / name)
        ):
            raise OSError("restore staging changed before verification")
        digest = hashlib.sha256()
        copied = 0
        while chunk := os.read(fd, 1024 * 1024):
            copied += len(chunk)
            digest.update(chunk)
        after = os.fstat(fd)
        selected_after = _bound_parent_stat(parent_fd, parent, name)
        if (
            copied != opened.st_size
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
            or not _same_snapshot_object(opened, selected_after)
        ):
            raise OSError("restore staging changed during verification")
        return digest.hexdigest()
    finally:
        os.close(fd)


def _nt_replace_windows_file(
    source_handle: int,
    parent_handle: int,
    destination_name: str,
) -> None:
    """Rename an opened file relative to an opened directory handle."""
    import ctypes

    class _FileRenameInformation(ctypes.Structure):
        _fields_ = [
            ("flags", ctypes.c_uint32),
            ("root_directory", ctypes.c_void_p),
            ("file_name_length", ctypes.c_uint32),
            ("file_name", ctypes.c_wchar * len(destination_name)),
        ]

    class _IoStatusBlock(ctypes.Structure):
        _fields_ = [
            ("status_or_pointer", ctypes.c_void_p),
            ("information", ctypes.c_size_t),
        ]

    rename = _FileRenameInformation()
    rename.flags = 1  # ReplaceIfExists
    rename.root_directory = parent_handle
    rename.file_name_length = len(destination_name.encode("utf-16-le"))
    rename.file_name = destination_name
    io_status = _IoStatusBlock()
    set_information = ctypes.windll.ntdll.NtSetInformationFile
    set_information.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_IoStatusBlock),
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
    )
    set_information.restype = ctypes.c_long
    status = set_information(
        source_handle,
        ctypes.byref(io_status),
        ctypes.byref(rename),
        ctypes.sizeof(rename),
        10,  # FileRenameInformation
    )
    if status < 0:
        raise OSError(
            f"handle-relative restore rename failed with NTSTATUS "
            f"0x{status & 0xFFFFFFFF:08x}"
        )


def _replace_windows_snapshot_member_by_handle(
    parent_fd: int,
    source: Path,
    destination_name: str,
    expected: os.stat_result,
) -> os.stat_result:
    """Rename an exact opened stage relative to the exact parent handle."""
    import ctypes

    create_file = ctypes.windll.kernel32.CreateFileW
    create_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file.restype = ctypes.c_void_p
    source_handle = create_file(
        str(source),
        0x00010000 | 0x00000080,  # DELETE | FILE_READ_ATTRIBUTES
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,  # OPEN_EXISTING
        0x00200000,  # FILE_FLAG_OPEN_REPARSE_POINT
        None,
    )
    if source_handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError()
    try:
        if not _windows_handle_matches_stat(source_handle, expected):
            raise OSError("restore staging changed before handle-relative rename")
        _nt_replace_windows_file(source_handle, parent_fd, destination_name)
        if not _windows_handle_matches_stat(source_handle, expected):
            raise OSError("restore staging changed during handle-relative rename")
        return expected
    finally:
        ctypes.windll.kernel32.CloseHandle(source_handle)


def _replace_bound_snapshot_member(
    parent_fd: int,
    parent: Path,
    stage_name: str,
    destination_name: str,
    stage_identity: os.stat_result,
    parent_chain: list[tuple[Path, os.stat_result]],
) -> os.stat_result:
    """Publish a staged restore without resolving a replaceable parent path."""
    selected = _bound_parent_stat(parent_fd, parent, stage_name)
    if (
        not _same_snapshot_object(stage_identity, selected)
        or not _snapshot_parent_chain_is_bound(parent_chain)
    ):
        raise OSError("restore staging or destination parent changed")
    if os.name == "nt":
        published = _replace_windows_snapshot_member_by_handle(
            parent_fd,
            parent / stage_name,
            destination_name,
            stage_identity,
        )
    else:
        os.replace(
            stage_name,
            destination_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        published = _bound_parent_stat(parent_fd, parent, destination_name)
    if (
        not _same_snapshot_object(stage_identity, published)
        or not _snapshot_parent_chain_is_bound(parent_chain)
    ):
        raise OSError("restored destination identity changed")
    return published


def _ensure_snapshot_parent(
    snapshot_dir: Path,
    expected: os.stat_result,
    relative_parent: PurePosixPath,
) -> bool:
    current = snapshot_dir
    for part in relative_parent.parts:
        if part in ("", ".", "..") or not _snapshot_directory_is_bound(
            snapshot_dir, expected
        ):
            return False
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        except OSError:
            return False
        try:
            selected = current.lstat()
        except OSError:
            return False
        if (
            not stat.S_ISDIR(selected.st_mode)
            or int(getattr(selected, "st_file_attributes", 0)) & 0x400
        ):
            return False
    return _snapshot_parent_is_bound(snapshot_dir, expected, relative_parent)


def _write_exclusive_snapshot_file(
    snapshot_dir: Path,
    snapshot_identity: os.stat_result,
    relative_name: str,
    chunks: Iterator[bytes],
    *,
    bound_parent_fd: Optional[int] = None,
) -> tuple[Path, os.stat_result, int]:
    """Write a new snapshot member without trusting a replaceable directory."""
    relative = PurePosixPath(relative_name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise OSError(f"invalid snapshot member path: {relative_name}")
    parent_rel = relative.parent
    if not _ensure_snapshot_parent(snapshot_dir, snapshot_identity, parent_rel):
        raise OSError("snapshot directory changed before member creation")
    parent_chain = _snapshot_parent_chain(
        snapshot_dir, snapshot_identity, parent_rel
    )
    if parent_chain is None:
        raise OSError("snapshot parent changed before member creation")
    destination = snapshot_dir.joinpath(*relative.parts)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    use_bound_parent = bound_parent_fd is not None and os.name != "nt"
    if use_bound_parent:
        fd = os.open(relative.name, flags, 0o600, dir_fd=bound_parent_fd)
    else:
        fd = os.open(destination, flags, 0o600)
    opened: Optional[os.stat_result] = None
    opened_actual_path: Optional[Path] = None
    success = False
    copied = 0
    try:
        opened = os.fstat(fd)
        opened_actual_path = _opened_fd_path(fd)
        path_opened = (
            _bound_parent_stat(bound_parent_fd, destination.parent, relative.name)
            if use_bound_parent
            else destination.lstat()
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or not os.path.samestat(opened, path_opened)
            or not _opened_fd_matches_path(fd, destination)
            or not _snapshot_parent_chain_is_bound(parent_chain)
        ):
            raise OSError("snapshot directory changed during member creation")
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                chunk = bytes(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write while creating snapshot member")
                copied += written
                view = view[written:]
        os.fsync(fd)
        after = os.fstat(fd)
        path_after = (
            _bound_parent_stat(bound_parent_fd, destination.parent, relative.name)
            if use_bound_parent
            else destination.lstat()
        )
        if (
            not os.path.samestat(opened, after)
            or not os.path.samestat(after, path_after)
            or not _opened_fd_matches_path(fd, destination)
            or not _snapshot_parent_chain_is_bound(parent_chain)
        ):
            raise OSError("snapshot member changed during write")
        success = True
    finally:
        os.close(fd)
        if not success and opened is not None:
            try:
                if use_bound_parent:
                    _remove_bound_snapshot_member(
                        bound_parent_fd,
                        destination.parent,
                        relative.name,
                        opened,
                    )
                else:
                    cleanup_path = opened_actual_path or destination
                    current = cleanup_path.lstat()
                    if os.path.samestat(opened, current):
                        _remove_owned_snapshot_file(cleanup_path, current)
            except OSError:
                pass
    final = (
        _bound_parent_stat(bound_parent_fd, destination.parent, relative.name)
        if use_bound_parent
        else destination.lstat()
    )
    if (
        opened is None
        or not os.path.samestat(opened, final)
        or not _snapshot_parent_chain_is_bound(parent_chain)
    ):
        raise OSError("snapshot member changed after close")
    return destination, final, copied


def _opened_file_chunks(path: Path, expected: os.stat_result) -> Iterator[bytes]:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not _same_snapshot_object(expected, path.lstat())
            or not os.path.samestat(expected, opened)
            or not _opened_fd_matches_path(fd, path)
        ):
            raise OSError(f"snapshot source changed before read: {path}")
        copied = 0
        while chunk := os.read(fd, 1024 * 1024):
            copied += len(chunk)
            yield chunk
        after = os.fstat(fd)
        if (
            copied != opened.st_size
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
            or not os.path.samestat(opened, path.lstat())
        ):
            raise OSError(f"snapshot source changed during read: {path}")
    finally:
        os.close(fd)


def _opened_file_sha256(path: Path, expected: os.stat_result) -> str:
    digest = hashlib.sha256()
    for chunk in _opened_file_chunks(path, expected):
        digest.update(chunk)
    return digest.hexdigest()


def create_quick_snapshot(
    label: Optional[str] = None,
    hermes_home: Optional[Path] = None,
    keep: Optional[int] = None,
    max_file_size: Optional[int] = None,
) -> Optional[str]:
    """Create a quick state snapshot of critical files.

    Copies STATE_FILES to a timestamped directory under state-snapshots/.
    Auto-prunes old snapshots beyond the keep limit.

    Args:
        max_file_size: When set, individual files larger than this many bytes
            are skipped (with a printed warning) instead of copied. Used by
            the pre-update safety snapshot so a multi-GB ``state.db`` can
            never stall ``hermes update`` or silently eat disk — the small
            pairing/cron/config files the snapshot exists to protect are
            always captured. ``None`` (default) copies everything, which
            preserves manual ``/snapshot`` and ``hermes backup --quick``
            behavior.

    Returns:
        Snapshot ID (timestamp-based), or None if no files found.
    """
    home = hermes_home or get_hermes_home()
    root = _quick_snapshot_root(home)

    def _too_large(path: Path, rel_name: str) -> bool:
        """True (and warn) when ``path`` exceeds the max_file_size cap."""
        if max_file_size is None:
            return False
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size <= max_file_size:
            return False
        print(
            f"  ⚠ Snapshot: skipping {rel_name} "
            f"({_format_size(size)} exceeds {_format_size(max_file_size)} limit)"
        )
        logger.warning(
            "Quick snapshot skipped %s: %d bytes exceeds %d byte limit",
            rel_name,
            size,
            max_file_size,
        )
        return True

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    safe_label = _quick_snapshot_label_segment(label)
    base_id = f"{ts}-{safe_label}" if safe_label else ts
    with _backup_maintenance_lock(root):
        snapshot_root_identity = root.lstat()
        snap_id = base_id
        suffix = 1
        while True:
            snap_dir = root / snap_id
            try:
                snap_dir.mkdir(parents=True, exist_ok=False)
                snapshot_identity = snap_dir.lstat()
                if (
                    not stat.S_ISDIR(snapshot_identity.st_mode)
                    or int(
                        getattr(snapshot_identity, "st_file_attributes", 0)
                    )
                    & 0x400
                ):
                    raise OSError("new snapshot path is not a regular directory")
                break
            except FileExistsError:
                snap_id = f"{base_id}-{suffix}"
                suffix += 1
        marker_payload = {
            "version": 1,
            "snapshot_id": snap_id,
            "pid": os.getpid(),
            "owner_token": uuid.uuid4().hex,
        }
        marker_bytes = json.dumps(marker_payload, separators=(",", ":")).encode(
            "utf-8"
        )
        marker, marker_identity, _ = _write_exclusive_snapshot_file(
            snap_dir,
            snapshot_identity,
            _QUICK_IN_PROGRESS_MARKER,
            iter((marker_bytes,)),
        )

    manifest: Dict[str, int] = {}  # rel_path -> file size
    manifest_hashes: Dict[str, str] = {}

    def _capture_member(source: Path, relative_name: str) -> tuple[int, str]:
        selected = source.lstat()
        if (
            not stat.S_ISREG(selected.st_mode)
            or int(getattr(selected, "st_file_attributes", 0)) & 0x400
        ):
            raise OSError(f"snapshot source is not a regular file: {source}")
        if source.suffix != ".db":
            stored, stored_identity, copied = _write_exclusive_snapshot_file(
                snap_dir,
                snapshot_identity,
                relative_name,
                _opened_file_chunks(source, selected),
            )
            return copied, _opened_file_sha256(stored, stored_identity)

        temp_db: Optional[Path] = None
        temp_identity: Optional[os.stat_result] = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".db", delete=False, dir=str(root)
            ) as temp_file:
                temp_db = Path(temp_file.name)
                temp_opened = os.fstat(temp_file.fileno())
                if (
                    not _same_snapshot_object(
                        snapshot_root_identity, root.lstat()
                    )
                    or not _opened_fd_matches_path(temp_file.fileno(), temp_db)
                ):
                    actual_temp = _opened_fd_path(temp_file.fileno())
                    if actual_temp is not None:
                        temp_db = actual_temp
                    temp_identity = temp_opened
                    raise OSError("SQLite staging path escaped snapshot root")
            temp_identity = temp_db.lstat()
            final_identity: list[os.stat_result] = []
            if not _safe_copy_db(
                source,
                temp_db,
                expected_dst_identity=temp_identity,
                expected_src_identity=selected,
                result_dst_identity=final_identity,
            ):
                raise OSError(f"SQLite snapshot failed for {source}")
            if final_identity:
                temp_identity = final_identity[0]
            stored, stored_identity, copied = _write_exclusive_snapshot_file(
                snap_dir,
                snapshot_identity,
                relative_name,
                _opened_file_chunks(temp_db, temp_identity),
            )
            return copied, _opened_file_sha256(stored, stored_identity)
        finally:
            if temp_db is not None and temp_identity is not None:
                _remove_owned_snapshot_file(temp_db, temp_identity)

    for rel in _QUICK_STATE_FILES:
        src = home / rel
        if not src.exists():
            continue

        try:
            source_root_identity = src.lstat()
        except OSError:
            continue
        if stat.S_ISDIR(source_root_identity.st_mode) and not (
            int(getattr(source_root_identity, "st_file_attributes", 0)) & 0x400
        ):
            # Walk the directory and record each file individually in the
            # manifest so restore can treat them uniformly.  Empty dirs are
            # skipped (nothing to snapshot).
            for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
                directory = Path(dirpath)
                try:
                    directory_identity = directory.lstat()
                except OSError:
                    dirnames[:] = []
                    continue
                if (
                    not stat.S_ISDIR(directory_identity.st_mode)
                    or int(
                        getattr(directory_identity, "st_file_attributes", 0)
                    )
                    & 0x400
                ):
                    dirnames[:] = []
                    continue
                safe_directories = []
                for dirname in dirnames:
                    child = directory / dirname
                    try:
                        child_identity = child.lstat()
                    except OSError:
                        continue
                    if (
                        stat.S_ISDIR(child_identity.st_mode)
                        and not (
                            int(
                                getattr(
                                    child_identity, "st_file_attributes", 0
                                )
                            )
                            & 0x400
                        )
                        and dirname not in {"workspaces", "attachments"}
                    ):
                        safe_directories.append(dirname)
                dirnames[:] = safe_directories
                for filename in filenames:
                    sub = directory / filename
                    sub_rel = sub.relative_to(home).as_posix()
                    if _too_large(sub, sub_rel):
                        continue
                    try:
                        member_size, member_hash = _capture_member(sub, sub_rel)
                        manifest[sub_rel] = member_size
                        manifest_hashes[sub_rel] = member_hash
                    except (OSError, PermissionError) as exc:
                        logger.warning("Could not snapshot %s: %s", sub_rel, exc)
            continue

        if not stat.S_ISREG(source_root_identity.st_mode) or int(
            getattr(source_root_identity, "st_file_attributes", 0)
        ) & 0x400:
            continue

        if _too_large(src, rel):
            continue

        try:
            member_size, member_hash = _capture_member(src, rel)
            manifest[rel] = member_size
            manifest_hashes[rel] = member_hash
        except (OSError, PermissionError) as exc:
            logger.warning("Could not snapshot %s: %s", rel, exc)

    if not manifest:
        try:
            _remove_owned_snapshot_dir(
                snap_dir,
                snapshot_identity,
                marker_name=_QUICK_IN_PROGRESS_MARKER,
                marker_identity=marker_identity,
                marker_payload=marker_payload,
            )
        except OSError:
            pass
        return None

    meta = {
        "id": snap_id,
        "timestamp": ts,
        "label": label,
        "file_count": len(manifest),
        "total_size": sum(manifest.values()),
        "files": manifest,
        "sha256": manifest_hashes,
    }
    try:
        _write_exclusive_snapshot_file(
            snap_dir,
            snapshot_identity,
            "manifest.json",
            iter((json.dumps(meta, indent=2).encode("utf-8"),)),
        )
        _write_exclusive_snapshot_file(
            snap_dir,
            snapshot_identity,
            _QUICK_OWNERSHIP_MARKER,
            iter((marker_bytes,)),
        )
    except OSError as exc:
        logger.warning("State snapshot publication failed for %s: %s", snap_id, exc)
        return None
    if not _remove_owned_snapshot_file(marker, marker_identity):
        logger.warning("State snapshot retained in-progress marker evidence: %s", snap_id)
    if os.path.lexists(marker):
        return None

    # Auto-prune. Defaults preserve historical manual /snapshot behavior; callers
    # with known high-churn safety snapshots (for example pre-update) can pass a
    # smaller keep value so large state.db copies do not accumulate indefinitely.
    with _backup_maintenance_lock(root):
        _prune_quick_snapshots(root, keep=_QUICK_DEFAULT_KEEP if keep is None else keep)

    logger.info("State snapshot created: %s (%d files)", snap_id, len(manifest))
    return snap_id


def _legacy_quick_snapshot_manifest(snapshot_dir: Path) -> Optional[Dict[str, Any]]:
    """Validate a pre-marker snapshot for read-only list/restore compatibility."""
    if (
        os.path.lexists(snapshot_dir / _QUICK_OWNERSHIP_MARKER)
        or os.path.lexists(snapshot_dir / _QUICK_IN_PROGRESS_MARKER)
        or re.fullmatch(r"\d{8}-\d{6}(?:-[^/\\\x00-\x1f]+)?", snapshot_dir.name)
        is None
    ):
        return None
    manifest_path = snapshot_dir / "manifest.json"
    fd: Optional[int] = None
    try:
        directory_identity = snapshot_dir.lstat()
        manifest_identity = manifest_path.lstat()
        if (
            not stat.S_ISDIR(directory_identity.st_mode)
            or _snapshot_path_is_link_like(snapshot_dir)
            or not stat.S_ISREG(manifest_identity.st_mode)
            or _snapshot_path_is_link_like(manifest_path)
        ):
            return None
        flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(manifest_path, flags)
        if not os.path.samestat(manifest_identity, os.fstat(fd)):
            return None
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = None
            metadata = json.load(stream)
        if not isinstance(metadata, dict) or metadata.get("id") != snapshot_dir.name:
            return None
        timestamp = metadata.get("timestamp")
        label = metadata.get("label")
        if (
            not isinstance(timestamp, str)
            or re.fullmatch(r"\d{8}-\d{6}", timestamp) is None
            or not (
                (label is None and snapshot_dir.name == timestamp)
                or (
                    isinstance(label, str)
                    and label
                    and "/" not in label
                    and "\\" not in label
                    and all(ord(char) >= 32 for char in label)
                    and snapshot_dir.name == f"{timestamp}-{label}"
                )
            )
        ):
            return None
        files = metadata.get("files")
        if not isinstance(files, dict) or not files:
            return None
        allowed_roots = tuple(PurePosixPath(item) for item in _QUICK_STATE_FILES)
        total_size = 0
        for raw_rel, declared_size in files.items():
            if (
                not isinstance(raw_rel, str)
                or not isinstance(declared_size, int)
                or declared_size < 0
            ):
                return None
            rel = PurePosixPath(raw_rel)
            if rel.is_absolute() or not rel.parts or ".." in rel.parts or "." in rel.parts:
                return None
            if not any(rel == root or root in rel.parents for root in allowed_roots):
                return None
            source = snapshot_dir.joinpath(*rel.parts)
            selected = source.lstat()
            if (
                not stat.S_ISREG(selected.st_mode)
                or _snapshot_path_is_link_like(source)
                or selected.st_size != declared_size
                or not source.resolve(strict=True).is_relative_to(
                    snapshot_dir.resolve(strict=True)
                )
            ):
                return None
            total_size += declared_size
        if (
            metadata.get("file_count") != len(files)
            or metadata.get("total_size") != total_size
            or not os.path.samestat(directory_identity, snapshot_dir.lstat())
            or not os.path.samestat(manifest_identity, manifest_path.lstat())
        ):
            return None
        return metadata
    except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    finally:
        if fd is not None:
            os.close(fd)


def list_quick_snapshots(
    limit: int = 20,
    hermes_home: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List existing quick state snapshots, most recent first."""
    root = _quick_snapshot_root(hermes_home)
    if not root.exists():
        return []

    results = []
    for d in sorted(root.iterdir(), reverse=True):
        owned = _quick_snapshot_marker_payload(d, _QUICK_OWNERSHIP_MARKER)
        legacy_metadata = None if owned is not None else _legacy_quick_snapshot_manifest(d)
        if (owned is None and legacy_metadata is None) or (
            d / _QUICK_IN_PROGRESS_MARKER
        ).exists():
            continue
        if legacy_metadata is not None:
            results.append(legacy_metadata)
            if len(results) >= limit:
                break
            continue
        manifest_path = d / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                results.append({"id": d.name, "file_count": 0, "total_size": 0})
        if len(results) >= limit:
            break

    return results


def restore_quick_snapshot(
    snapshot_id: str,
    hermes_home: Optional[Path] = None,
) -> bool:
    """Restore state from a quick snapshot.

    Overwrites current state files with the snapshot's copies.
    Returns True if at least one file was restored.
    """
    home = hermes_home or get_hermes_home()
    root = _quick_snapshot_root(home)

    # Security: reject snapshot_id values that contain path separators or
    # traversal sequences so that `root / snapshot_id` stays inside root.
    if not snapshot_id or "/" in snapshot_id or "\\" in snapshot_id or snapshot_id in (".", ".."):
        logger.error("Invalid snapshot_id: %s", snapshot_id)
        return False

    snap_dir = root / snapshot_id

    # Confirm the resolved path is still inside root (handles symlinks etc.)
    try:
        snap_dir.resolve().relative_to(root.resolve())
    except ValueError:
        logger.error("Snapshot path traversal blocked for id: %s", snapshot_id)
        return False

    owned = _quick_snapshot_marker_payload(snap_dir, _QUICK_OWNERSHIP_MARKER)
    legacy_metadata = None if owned is not None else _legacy_quick_snapshot_manifest(snap_dir)
    if (owned is None and legacy_metadata is None) or (
        snap_dir / _QUICK_IN_PROGRESS_MARKER
    ).exists():
        return False
    try:
        snapshot_identity = snap_dir.lstat()
        home_identity = home.lstat()
        if (
            not stat.S_ISDIR(snapshot_identity.st_mode)
            or int(getattr(snapshot_identity, "st_file_attributes", 0)) & 0x400
            or not stat.S_ISDIR(home_identity.st_mode)
            or int(getattr(home_identity, "st_file_attributes", 0)) & 0x400
        ):
            return False
    except OSError:
        return False

    manifest_path = snap_dir / "manifest.json"
    try:
        if legacy_metadata is not None:
            meta = legacy_metadata
        else:
            manifest_identity = manifest_path.lstat()
            manifest_bytes = b"".join(
                _opened_file_chunks(manifest_path, manifest_identity)
            )
            if not _snapshot_directory_is_bound(snap_dir, snapshot_identity):
                return False
            meta = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False

    if not isinstance(meta, dict):
        return False
    files = meta.get("files")
    hashes = meta.get("sha256", {})
    if not isinstance(files, dict) or not files or not isinstance(hashes, dict):
        return False
    allowed_roots = tuple(PurePosixPath(item) for item in _QUICK_STATE_FILES)
    total_size = 0
    members: list[
        tuple[str, PurePosixPath, Path, os.stat_result, Optional[str]]
    ] = []
    for rel, declared_size in files.items():
        if (
            not isinstance(rel, str)
            or not isinstance(declared_size, int)
            or isinstance(declared_size, bool)
            or declared_size < 0
        ):
            return False
        relative = PurePosixPath(rel)
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or not any(
                relative == allowed or allowed in relative.parents
                for allowed in allowed_roots
            )
        ):
            return False
        declared_hash = hashes.get(rel)
        if declared_hash is not None and (
            not isinstance(declared_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", declared_hash) is None
        ):
            return False
        src = snap_dir.joinpath(*relative.parts)
        try:
            source_identity = src.lstat()
        except OSError:
            return False
        if (
            not stat.S_ISREG(source_identity.st_mode)
            or int(getattr(source_identity, "st_file_attributes", 0)) & 0x400
            or source_identity.st_size != declared_size
            or not _snapshot_directory_is_bound(snap_dir, snapshot_identity)
        ):
            return False
        if declared_hash is not None and _opened_file_sha256(
            src, source_identity
        ) != declared_hash:
            return False
        members.append(
            (rel, relative, src, source_identity, declared_hash)
        )
        total_size += declared_size
    if (
        meta.get("file_count") != len(files)
        or meta.get("total_size") != total_size
    ):
        return False

    restored = 0
    for rel, relative, src, source_identity, declared_hash in members:
        dst = home.joinpath(*relative.parts)

        stage_rel = relative.parent / (
            f".{relative.name}.snap-restore-{uuid.uuid4().hex}.tmp"
        )
        stage: Optional[Path] = None
        stage_identity: Optional[os.stat_result] = None
        try:
            if not _ensure_snapshot_parent(
                home, home_identity, relative.parent
            ):
                raise OSError("restore destination parent is unavailable")
            restore_parent_chain = _snapshot_parent_chain(
                home, home_identity, relative.parent
            )
            if restore_parent_chain is None:
                raise OSError("restore destination parent changed")
            with _hold_snapshot_parent_chain(restore_parent_chain) as parent_fd:
                try:
                    stage, stage_identity, _ = _write_exclusive_snapshot_file(
                        home,
                        home_identity,
                        stage_rel.as_posix(),
                        _opened_file_chunks(src, source_identity),
                        bound_parent_fd=parent_fd,
                    )
                    if (
                        not _snapshot_directory_is_bound(
                            snap_dir, snapshot_identity
                        )
                        or not _snapshot_parent_chain_is_bound(
                            restore_parent_chain
                        )
                    ):
                        raise OSError("snapshot or restore destination changed")
                    try:
                        if os.name == "nt":
                            os.chmod(
                                stage, stat.S_IMODE(source_identity.st_mode)
                            )
                        else:
                            os.chmod(
                                stage.name,
                                stat.S_IMODE(source_identity.st_mode),
                                dir_fd=parent_fd,
                                follow_symlinks=False,
                            )
                    except OSError:
                        pass
                    current_stage = _bound_parent_stat(
                        parent_fd, dst.parent, stage.name
                    )
                    if not _same_snapshot_object(
                        stage_identity, current_stage
                    ):
                        raise OSError("restore staging identity changed")
                    stage_identity = current_stage
                    if (
                        declared_hash is not None
                        and _bound_snapshot_member_sha256(
                            parent_fd,
                            dst.parent,
                            stage.name,
                            stage_identity,
                        )
                        != declared_hash
                    ):
                        raise OSError("restore staging hash changed")
                    _replace_bound_snapshot_member(
                        parent_fd,
                        dst.parent,
                        stage.name,
                        dst.name,
                        stage_identity,
                        restore_parent_chain,
                    )
                    stage = None
                    stage_identity = None
                    restored += 1
                finally:
                    if stage is not None and stage_identity is not None:
                        _remove_bound_snapshot_member(
                            parent_fd,
                            dst.parent,
                            stage.name,
                            stage_identity,
                        )
        except (OSError, PermissionError) as exc:
            logger.error("Failed to restore %s: %s", rel, exc)

    logger.info("Restored %d files from snapshot %s", restored, snapshot_id)
    return restored > 0


# Relative path of the cron job database inside HERMES_HOME. Kept in sync with
# the entry in ``_QUICK_STATE_FILES`` and with ``cron/jobs.py``'s ``JOBS_FILE``.
_CRON_JOBS_REL = "cron/jobs.json"


def _count_cron_jobs(path: Path) -> Optional[int]:
    """Return the number of cron jobs stored in ``path``.

    The canonical on-disk shape is ``{"jobs": [...]}`` (see ``cron/jobs.py``).
    A legacy bare-list shape (``[...]``) is also honoured.

    Returns:
        The job count for any *valid, readable* JSON document, or ``None`` if
        the file is missing or cannot be parsed. ``None`` means "unknown" —
        callers must not treat it as "zero jobs", because acting on an
        unreadable file could mask a real corruption the user needs to see.
    """
    if not path.is_file():
        return None
    try:
        # utf-8-sig: same dialect as cron/jobs.load_jobs — Windows editors
        # may leave a UTF-8 BOM that plain utf-8 json.load rejects. Without
        # it a BOM'd jobs.json counts as "unreadable" (None) and the
        # post-update cron-loss auto-restore safety net silently disables.
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        jobs = data.get("jobs", [])
        return len(jobs) if isinstance(jobs, list) else None
    if isinstance(data, list):
        return len(data)
    return None


def restore_cron_jobs_if_emptied(
    snapshot_id: str,
    hermes_home: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Safety net for silent cron-job loss across ``hermes update``.

    Config-version migrations have been observed to leave ``cron/jobs.json``
    valid-but-empty after an update, silently dropping every scheduled job
    (issue #34600). The desktop scheduler can also overwrite the file with its
    own small set of internally-tracked crons, causing partial loss (issue
    #52144).

    This compares the *current* job count against the pre-update snapshot. If
    the live file now has **fewer** jobs than the snapshot, the snapshot copy
    of ``cron/jobs.json`` is restored in place.

    The check is deliberately conservative — it only ever restores when there
    is unambiguous evidence of loss (snapshot had more jobs than live file),
    so a user who genuinely deleted jobs during/after the update is never
    second-guessed, and an unreadable live file (count ``None``) is left
    untouched so real corruption still surfaces.

    Args:
        snapshot_id: The pre-update quick-snapshot id (from
            :func:`create_quick_snapshot`).
        hermes_home: Override for the Hermes home directory (tests).

    Returns:
        ``None`` when no action was taken (the common, healthy path). On a
        successful restore, a dict ``{"restored": True, "job_count": N,
        "snapshot_id": ...}`` so the caller can warn the user.
    """
    if not snapshot_id:
        return None

    home = hermes_home or get_hermes_home()
    live_path = home / _CRON_JOBS_REL

    live_count = _count_cron_jobs(live_path)
    # ``None`` (missing or unparseable) is intentionally left alone — that's a
    # different failure mode the user should see rather than have papered over.
    if live_count is None:
        return None

    snap_path = _quick_snapshot_root(home) / snapshot_id / _CRON_JOBS_REL
    snap_count = _count_cron_jobs(snap_path)
    if not snap_count:  # None or 0 — nothing worth restoring
        return None

    # Restore when live has FEWER jobs than the pre-update snapshot.
    # Catches both total loss (0 vs N) and partial loss (1 vs 19) — the
    # desktop scheduler can overwrite jobs.json with its own small set of
    # internally-tracked crons after an update/restart.
    if live_count >= snap_count:
        return None

    try:
        live_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snap_path, live_path)
    except (OSError, PermissionError) as exc:
        logger.error(
            "Cron jobs were emptied during update but auto-restore failed: %s", exc
        )
        return None

    logger.warning(
        "Restored %d cron job(s) from pre-update snapshot %s "
        "(live file had %d job(s), snapshot had %d — jobs were lost during migration)",
        snap_count,
        snapshot_id,
        live_count,
        snap_count,
    )
    return {"restored": True, "job_count": snap_count, "snapshot_id": snapshot_id}


def _prune_quick_snapshots(root: Path, keep: int = _QUICK_DEFAULT_KEEP) -> int:
    """Remove oldest quick snapshots beyond the keep limit. Returns count deleted."""
    if not root.exists():
        return 0

    deleted = 0
    now = time.time()
    for candidate in root.iterdir():
        try:
            evidence = _quick_snapshot_marker_evidence(
                candidate, _QUICK_IN_PROGRESS_MARKER
            )
            if evidence is None:
                continue
            payload, directory_stat, marker_stat = evidence
            if (
                now - marker_stat.st_mtime <= _QUICK_IN_PROGRESS_MAX_AGE_SECONDS
                or _snapshot_process_is_running(payload["pid"])
            ):
                continue
            if _remove_owned_snapshot_dir(
                candidate,
                directory_stat,
                marker_name=_QUICK_IN_PROGRESS_MARKER,
                marker_identity=marker_stat,
                marker_payload=payload,
            ):
                deleted += 1
        except (OSError, ValueError, json.JSONDecodeError):
            continue

    dirs = []
    for candidate in root.iterdir():
        evidence = _quick_snapshot_marker_evidence(
            candidate, _QUICK_OWNERSHIP_MARKER
        )
        if evidence is not None and not os.path.lexists(
            candidate / _QUICK_IN_PROGRESS_MARKER
        ):
            dirs.append((candidate, evidence))
    dirs.sort(key=lambda item: item[0].name, reverse=True)

    for d, evidence in dirs[keep:]:
        try:
            payload, directory_stat, marker_stat = evidence
            if _remove_owned_snapshot_dir(
                d,
                directory_stat,
                marker_name=_QUICK_OWNERSHIP_MARKER,
                marker_identity=marker_stat,
                marker_payload=payload,
            ):
                deleted += 1
        except OSError as exc:
            logger.warning("Failed to prune snapshot %s: %s", d.name, exc)

    return deleted


def prune_quick_snapshots(
    keep: int = _QUICK_DEFAULT_KEEP,
    hermes_home: Optional[Path] = None,
) -> int:
    """Manually prune quick snapshots. Returns count deleted."""
    root = _quick_snapshot_root(hermes_home)
    with _backup_maintenance_lock(root):
        return _prune_quick_snapshots(root, keep=keep)


def run_quick_backup(args) -> None:
    """CLI entry point for hermes backup --quick."""
    label = getattr(args, "label", None)
    snap_id = create_quick_snapshot(label=label)
    if snap_id:
        print(f"State snapshot created: {snap_id}")
        snaps = list_quick_snapshots()
        print(f"  {len(snaps)} snapshot(s) stored in {display_hermes_home()}/state-snapshots/")
        print(f"  Restore with: /snapshot restore {snap_id}")
    else:
        print("No state files found to snapshot.")


# ---------------------------------------------------------------------------
# Shared full-zip backup helper
# ---------------------------------------------------------------------------

_ARCHIVE_OWNERSHIP_PREFIX = b"hermes-managed-backup:v1:"


def _archive_ownership_comment(path: Path) -> bytes:
    digest = hashlib.sha256(path.name.encode("utf-8")).hexdigest().encode("ascii")
    return _ARCHIVE_OWNERSHIP_PREFIX + digest


def _owned_archive_identity(path: Path) -> Optional[os.stat_result]:
    """Return identity only for an exact archive published by this helper."""
    fd: Optional[int] = None
    try:
        selected = path.lstat()
        if not stat.S_ISREG(selected.st_mode) or _snapshot_path_is_link_like(path):
            return None
        flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        opened = os.fstat(fd)
        if not os.path.samestat(selected, opened) or not stat.S_ISREG(opened.st_mode):
            return None
        with os.fdopen(fd, "rb") as archive_file:
            fd = None
            with zipfile.ZipFile(archive_file, "r") as archive:
                if archive.comment != _archive_ownership_comment(path):
                    return None
        if not os.path.samestat(selected, path.lstat()):
            return None
        return selected
    except (OSError, ValueError, zipfile.BadZipFile):
        return None
    finally:
        if fd is not None:
            os.close(fd)


def _write_bound_file_to_zip(
    archive: zipfile.ZipFile,
    source: Path,
    arcname: str,
    *,
    expected_identity: Optional[os.stat_result] = None,
) -> int:
    """Archive one opened regular file only while its pathname stays bound."""
    selected = source.lstat()
    if expected_identity is not None and not _same_snapshot_object(
        expected_identity, selected
    ):
        raise OSError(f"backup source changed after enumeration: {source}")
    reparse = int(getattr(selected, "st_file_attributes", 0)) & 0x400
    if not stat.S_ISREG(selected.st_mode) or reparse:
        raise OSError(f"backup source is not a regular file: {source}")

    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(source, flags)
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not os.path.samestat(selected, opened)
            or not _opened_fd_matches_path(fd, source)
        ):
            raise OSError(f"backup source changed before open: {source}")

        timestamp = list(time.localtime(opened.st_mtime)[:6])
        timestamp[0] = min(max(timestamp[0], 1980), 2107)
        info = zipfile.ZipInfo(arcname.replace(os.sep, "/"), tuple(timestamp))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = (opened.st_mode & 0xFFFF) << 16
        copied = 0
        with os.fdopen(fd, "rb", closefd=False) as source_file, archive.open(
            info, "w", force_zip64=True
        ) as destination:
            while chunk := source_file.read(1024 * 1024):
                destination.write(chunk)
                copied += len(chunk)

        after = os.fstat(fd)
        if (
            copied != opened.st_size
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
            or not os.path.samestat(opened, source.lstat())
        ):
            raise OSError(f"backup source changed while reading: {source}")
        return copied
    finally:
        os.close(fd)


def _write_full_zip_backup(out_path: Path, hermes_root: Path) -> Optional[Path]:
    """Write a full zip snapshot of ``hermes_root`` to ``out_path``.

    Uses the same exclusion rules and SQLite safe-copy as :func:`run_backup`.
    Returns the output path on success, None on failure (nothing to back up,
    or write error — caller should surface the outcome but not raise).
    """
    files_to_add: list[tuple[Path, Path, os.stat_result]] = []
    try:
        root_identity = hermes_root.lstat()
        if (
            not stat.S_ISDIR(root_identity.st_mode)
            or int(getattr(root_identity, "st_file_attributes", 0)) & 0x400
        ):
            return None
        for dirpath, dirnames, filenames in os.walk(hermes_root, followlinks=False):
            dp = Path(dirpath)
            directory_identity = dp.lstat()
            directory_reparse = int(
                getattr(directory_identity, "st_file_attributes", 0)
            ) & 0x400
            if not stat.S_ISDIR(directory_identity.st_mode) or directory_reparse:
                dirnames[:] = []
                continue
            if dp == hermes_root and not _same_snapshot_object(
                root_identity, directory_identity
            ):
                return None

            # followlinks=False does not stop Windows directory junctions on
            # supported Python versions. Inspect every child with lstat and
            # prune symlinks/reparse points before os.walk can descend.
            safe_directories = []
            for dirname in dirnames:
                if dirname in _EXCLUDED_DIRS:
                    continue
                child = dp / dirname
                try:
                    child_identity = child.lstat()
                except OSError:
                    continue
                child_reparse = int(
                    getattr(child_identity, "st_file_attributes", 0)
                ) & 0x400
                if stat.S_ISDIR(child_identity.st_mode) and not child_reparse:
                    safe_directories.append(dirname)
            dirnames[:] = safe_directories

            for fname in filenames:
                fpath = dp / fname
                try:
                    rel = fpath.relative_to(hermes_root)
                except ValueError:
                    continue

                if _should_skip_backup_file(fpath, rel, out_path):
                    continue
                selected = fpath.lstat()
                reparse = int(
                    getattr(selected, "st_file_attributes", 0)
                ) & 0x400
                if not stat.S_ISREG(selected.st_mode) or reparse:
                    continue

                files_to_add.append((fpath, rel, selected))
    except OSError as exc:
        logger.warning("Full-zip backup: walk failed: %s", exc)
        return None

    if not files_to_add:
        return None

    temp_path: Optional[Path] = None
    temp_identity: Optional[os.stat_result] = None
    written_members = 0
    try:
        # Publish through a sibling temporary file.  The destination is never
        # opened until the complete archive has passed integrity checks, so a
        # failed/crashed write cannot destroy an older recovery point.
        with tempfile.NamedTemporaryFile(
            suffix=".zip", delete=False, dir=str(out_path.parent)
        ) as tmp:
            temp_path = Path(tmp.name)
            temp_identity = os.fstat(tmp.fileno())

        with zipfile.ZipFile(
            temp_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            for abs_path, rel_path, source_identity in files_to_add:
                if abs_path.suffix == ".db":
                    # Stage the snapshot alongside the output zip so that the
                    # temp file lives on the same filesystem.  The system
                    # default (/tmp) may be a small tmpfs that cannot hold
                    # large databases, causing silent backup incompleteness.
                    with tempfile.NamedTemporaryFile(
                        suffix=".db", delete=False, dir=str(out_path.parent)
                    ) as tmp_db_file:
                        tmp_db = Path(tmp_db_file.name)
                    tmp_db_identity = tmp_db.lstat()
                    final_tmp_db_identity: list[os.stat_result] = []
                    try:
                        copied = _safe_copy_db(
                            abs_path,
                            tmp_db,
                            expected_dst_identity=tmp_db_identity,
                            expected_src_identity=source_identity,
                            result_dst_identity=final_tmp_db_identity,
                        )
                        if final_tmp_db_identity:
                            tmp_db_identity = final_tmp_db_identity[0]
                        if not copied:
                            logger.warning(
                                "Full-zip backup aborted: SQLite snapshot failed for %s",
                                rel_path,
                            )
                            return None
                        _write_bound_file_to_zip(
                            zf,
                            tmp_db,
                            rel_path.as_posix(),
                            expected_identity=tmp_db_identity,
                        )
                        written_members += 1
                    finally:
                        _remove_owned_snapshot_file(tmp_db, tmp_db_identity)
                else:
                    _write_bound_file_to_zip(
                        zf,
                        abs_path,
                        rel_path.as_posix(),
                        expected_identity=source_identity,
                    )
                    written_members += 1
            zf.comment = _archive_ownership_comment(out_path)

        if not written_members:
            logger.warning("Full-zip backup aborted: no files could be written")
            return None

        # Re-open the closed archive before publication.  testzip() catches
        # corrupt member data and namelist() ensures a non-empty archive.
        with zipfile.ZipFile(temp_path, "r") as zf:
            if not zf.namelist() or zf.testzip() is not None:
                logger.warning("Full-zip backup aborted: archive integrity check failed")
                return None

        candidate = out_path
        for attempt in range(16):
            if attempt:
                candidate = out_path.with_name(
                    f"{out_path.stem}-{uuid.uuid4().hex[:12]}{out_path.suffix}"
                )
                with zipfile.ZipFile(temp_path, "a") as zf:
                    zf.comment = _archive_ownership_comment(candidate)

            with open(temp_path, "r+b") as archive_file:
                os.fsync(archive_file.fileno())
                temp_identity = os.fstat(archive_file.fileno())

            try:
                if os.name == "nt":
                    # Unlike os.replace(), Windows os.rename() fails when the
                    # destination already exists.
                    os.rename(temp_path, candidate)
                    published_via_link = False
                else:
                    # link() is the portable no-replace publication primitive
                    # on POSIX. The sibling temp and destination share a
                    # filesystem by construction.
                    os.link(temp_path, candidate, follow_symlinks=False)
                    published_via_link = True
            except FileExistsError:
                continue

            published = candidate.lstat()
            if (
                not stat.S_ISREG(published.st_mode)
                or not os.path.samestat(published, temp_identity)
            ):
                logger.warning(
                    "Full-zip backup: published archive identity changed: %s",
                    candidate,
                )
                return None

            if published_via_link:
                if _remove_owned_snapshot_file(temp_path, temp_identity):
                    temp_path = None
                    temp_identity = None
            else:
                temp_path = None
                temp_identity = None
            return candidate

        logger.warning("Full-zip backup: destination kept colliding: %s", out_path)
        return None
    except (PermissionError, OSError, ValueError, zipfile.BadZipFile) as exc:
        logger.warning("Full-zip backup: zip write failed: %s", exc)
        return None
    finally:
        if temp_path is not None and temp_identity is not None:
            try:
                _remove_owned_snapshot_file(temp_path, temp_identity)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Pre-update auto-backup
# ---------------------------------------------------------------------------

_PRE_UPDATE_BACKUPS_DIR = "backups"
_PRE_UPDATE_PREFIX = "pre-update-"
_PRE_UPDATE_DEFAULT_KEEP = 5


def _pre_update_backup_dir(hermes_home: Optional[Path] = None) -> Path:
    home = hermes_home or get_hermes_home()
    return home / _PRE_UPDATE_BACKUPS_DIR


def _prune_pre_update_backups(backup_dir: Path, keep: int) -> int:
    """Remove oldest pre-update backups beyond the keep limit.

    Returns the number of files deleted.  Only touches files matching
    ``pre-update-*.zip`` so hand-made zips dropped in the same directory
    are never touched.

    ``keep`` is floored to 1 because this helper is only called immediately
    after a fresh backup is written: deleting that backup right after the
    user paid the disk/CPU cost to create it would leave them worse off
    than no backup at all (and the wrapper in ``main.py`` would still print
    a misleading ``Saved: <path>`` line for a file that no longer exists).
    Operators who genuinely don't want a backup should set
    ``updates.pre_update_backup: off`` in config — that gates creation.
    """
    keep = max(keep, 1)
    if not backup_dir.exists():
        return 0

    backups = []
    for path in backup_dir.iterdir():
        selected = _owned_archive_identity(path)
        if (
            selected is not None
            and path.name.startswith(_PRE_UPDATE_PREFIX)
            and path.suffix.lower() == ".zip"
        ):
            backups.append((path, selected))
    backups.sort(
        key=lambda item: item[0].name,
        reverse=True,
    )

    deleted = 0
    for p, selected in backups[keep:]:
        try:
            if _remove_owned_snapshot_file(p, selected):
                deleted += 1
        except OSError as exc:
            logger.warning("Failed to prune backup %s: %s", p.name, exc)

    return deleted


def create_pre_update_backup(
    hermes_home: Optional[Path] = None,
    keep: int = _PRE_UPDATE_DEFAULT_KEEP,
) -> Optional[Path]:
    """Create a full zip backup of HERMES_HOME under ``backups/``.

    Mirrors :func:`run_backup` (same exclusion rules, same SQLite safe-copy)
    but writes to ``<HERMES_HOME>/backups/pre-update-<timestamp>.zip`` and
    auto-prunes old pre-update backups.

    Returns the path to the created zip, or ``None`` if no files were
    found or the backup could not be created.  Never raises — the caller
    (``hermes update``) should continue even if the backup fails.
    """
    hermes_root = hermes_home or get_default_hermes_root()
    if not hermes_root.is_dir():
        return None

    backup_dir = _pre_update_backup_dir(hermes_root)
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create pre-update backup dir %s: %s", backup_dir, exc)
        return None

    try:
        with _backup_maintenance_lock(backup_dir):
            out_path = _unique_archive_path(backup_dir, _PRE_UPDATE_PREFIX)
            result = _write_full_zip_backup(out_path, hermes_root)
            if result is None:
                return None
            _prune_pre_update_backups(backup_dir, keep=keep)
            return result
    except OSError as exc:
        logger.warning("Could not publish pre-update backup: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Pre-migration auto-backup (used by `hermes claw migrate`)
# ---------------------------------------------------------------------------

_PRE_MIGRATION_PREFIX = "pre-migration-"
_PRE_MIGRATION_DEFAULT_KEEP = 5


def _prune_pre_migration_backups(backup_dir: Path, keep: int) -> int:
    """Remove oldest pre-migration backups beyond the keep limit.

    Only touches files matching ``pre-migration-*.zip`` so other backups in
    the same directory are never touched.
    """
    keep = max(keep, 1)
    if not backup_dir.exists():
        return 0

    backups = []
    for path in backup_dir.iterdir():
        selected = _owned_archive_identity(path)
        if (
            selected is not None
            and path.name.startswith(_PRE_MIGRATION_PREFIX)
            and path.suffix.lower() == ".zip"
        ):
            backups.append((path, selected))
    backups.sort(
        key=lambda item: item[0].name,
        reverse=True,
    )

    deleted = 0
    for p, selected in backups[keep:]:
        try:
            if _remove_owned_snapshot_file(p, selected):
                deleted += 1
        except OSError as exc:
            logger.warning("Failed to prune pre-migration backup %s: %s", p.name, exc)

    return deleted


def create_pre_migration_backup(
    hermes_home: Optional[Path] = None,
    keep: int = _PRE_MIGRATION_DEFAULT_KEEP,
) -> Optional[Path]:
    """Create a full zip backup of HERMES_HOME under ``backups/`` before a
    ``hermes claw migrate`` apply.

    Shares implementation with :func:`create_pre_update_backup` via
    ``_write_full_zip_backup`` — same exclusions, same SQLite safe-copy,
    restorable with ``hermes import <archive>``.  Writes to
    ``<HERMES_HOME>/backups/pre-migration-<timestamp>.zip`` and auto-prunes
    old pre-migration backups.

    Returns the path to the created zip, or ``None`` if nothing was found
    to back up (fresh install) or the write failed.  Never raises — the
    caller decides whether to abort or proceed.
    """
    hermes_root = hermes_home or get_default_hermes_root()
    if not hermes_root.is_dir():
        return None

    # Reuses the shared backups/ directory so `hermes import` and the
    # update-backup listing pick up pre-migration archives too.
    backup_dir = _pre_update_backup_dir(hermes_root)
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create pre-migration backup dir %s: %s", backup_dir, exc)
        return None

    try:
        with _backup_maintenance_lock(backup_dir):
            out_path = _unique_archive_path(backup_dir, _PRE_MIGRATION_PREFIX)
            result = _write_full_zip_backup(out_path, hermes_root)
            if result is None:
                return None
            _prune_pre_migration_backups(backup_dir, keep=keep)
            return result
    except OSError as exc:
        logger.warning("Could not publish pre-migration backup: %s", exc)
        return None
