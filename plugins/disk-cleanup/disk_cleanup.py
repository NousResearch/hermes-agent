"""disk_cleanup — ephemeral file cleanup for Hermes Agent.

Library module wrapping the deterministic cleanup rules written by
@LVT382009 in PR #12212. The plugin ``__init__.py`` wires these
functions into ``post_tool_call`` and ``on_session_end`` hooks so
tracking and cleanup happen automatically — the agent never needs to
call a tool or remember a skill.

Rules:
  - test files    → delete immediately at task end (age >= 0)
  - temp files    → delete after 7 days
  - cron-output   → delete after 14 days
  - empty dirs    → delete only inside explicitly marked Hermes-owned roots
  - research      → keep 10 newest, prompt for older (deep only)
  - chrome-profile→ prompt after 14 days (deep only)
  - >500 MB files → prompt always (deep only)

Scope: strictly HERMES_HOME and /tmp/hermes-*
Never touches: ~/.hermes/logs/ or any system directory.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover — plugin may load before constants resolves
    import os

    def get_hermes_home() -> Path:  # type: ignore[no-redef]
        val = (os.environ.get("HERMES_HOME") or "").strip()
        return Path(val).resolve() if val else (Path.home() / ".hermes").resolve()


logger = logging.getLogger(__name__)
_TRACKED_THREAD_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_state_dir() -> Path:
    """State dir — separate from ``$HERMES_HOME/logs/``."""
    return get_hermes_home() / "disk-cleanup"


def get_tracked_file() -> Path:
    return get_state_dir() / "tracked.json"


def get_log_file() -> Path:
    """Audit log — intentionally NOT under ``$HERMES_HOME/logs/``."""
    return get_state_dir() / "cleanup.log"


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def is_safe_path(path: Path) -> bool:
    """Accept only paths under HERMES_HOME or ``/tmp/hermes-*``.

    Rejects Windows mounts (``/mnt/c`` etc.) and any system directory.
    """
    try:
        candidate = Path(path)
        hermes_home = Path(get_hermes_home()).resolve()
        resolved = candidate.resolve()
    except (TypeError, ValueError, OSError):
        return False

    try:
        resolved.relative_to(hermes_home)
        return True
    except (ValueError, OSError):
        pass

    # Allow the explicitly temporary POSIX-style ``/tmp/hermes-*`` scope.
    # On Windows, ``Path('/tmp/...').resolve()`` becomes ``C:\\tmp\\...``;
    # retain that compatibility while rejecting an explicit drive path such
    # as ``C:\\tmp\\hermes-user``.  The latter is not the approved temporary
    # namespace and could otherwise be mistaken for it by ``parts`` alone.
    if os.name == "nt" and candidate.anchor not in {"", "\\", "/"}:
        return False
    parts = resolved.parts
    if (
        len(parts) >= 3
        and (
            parts[0] in {"/", "\\"}
            or (os.name == "nt" and candidate.anchor in {"\\", "/"})
        )
        and parts[1].casefold() == "tmp"
        and parts[2].casefold().startswith("hermes-")
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def _log(message: str) -> None:
    try:
        log_file = get_log_file()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except OSError:
        # Never let the audit log break the agent loop.
        pass


# ---------------------------------------------------------------------------
# tracked.json — atomic read/write, backup scoped to tracked.json only
# ---------------------------------------------------------------------------

@contextmanager
def _tracked_registry_lock():
    """Serialize tracked.json read-modify-write cycles across processes."""
    lock_path = get_state_dir() / "tracked.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    with _TRACKED_THREAD_LOCK:
        try:
            lock_fd = os.open(lock_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
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
                raise OSError("tracked registry lock is not independently owned")
        lock_file = os.fdopen(lock_fd, "r+b", closefd=True)
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            acquired = True
            yield
        finally:
            try:
                if acquired:
                    lock_file.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()


def _load_tracked_unlocked() -> List[Dict[str, Any]]:
    """Load tracked.json.  Restores from ``.bak`` on corruption."""
    tf = get_tracked_file()
    tf.parent.mkdir(parents=True, exist_ok=True)

    if not tf.exists():
        return []

    try:
        data = json.loads(tf.read_text())
        if not isinstance(data, list):
            raise ValueError("tracked.json must contain a list")
        return data
    except (json.JSONDecodeError, ValueError):
        bak = tf.with_suffix(".json.bak")
        if bak.exists():
            try:
                data = json.loads(bak.read_text())
                if not isinstance(data, list):
                    raise ValueError("tracked.json backup must contain a list")
                _log("WARN: tracked.json corrupted — restored from .bak")
                return data
            except Exception:
                pass
        _log("WARN: tracked.json corrupted, no backup — starting fresh")
        return []


def load_tracked() -> List[Dict[str, Any]]:
    with _tracked_registry_lock():
        return _load_tracked_unlocked()


def _save_tracked_unlocked(tracked: List[Dict[str, Any]]) -> None:
    """Atomic write: ``.tmp`` → backup old → rename."""
    tf = get_tracked_file()
    tf.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=tf.parent,
            prefix="tracked-",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(tracked, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        if tf.exists():
            shutil.copy2(tf, tf.with_suffix(".json.bak"))
        tmp_path.replace(tf)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def save_tracked(tracked: List[Dict[str, Any]]) -> None:
    with _tracked_registry_lock():
        _save_tracked_unlocked(tracked)


def _commit_tracked_snapshot(
    original: List[Dict[str, Any]], desired: List[Dict[str, Any]]
) -> None:
    """Apply removals from one snapshot without losing concurrent additions."""
    desired_ids = {id(item) for item in desired}
    removed = [item for item in original if id(item) not in desired_ids]
    with _tracked_registry_lock():
        current = _load_tracked_unlocked()
        for old_item in removed:
            try:
                current.remove(old_item)
            except ValueError:
                pass
        _save_tracked_unlocked(current)


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

ALLOWED_CATEGORIES = {
    "temp", "test", "research", "download",
    "chrome-profile", "cron-output", "other",
}

_EMPTY_DIR_PROTECTED_TOP_LEVEL = frozenset({
    "logs", "memories", "sessions", "cron", "cronjobs",
    "cache", "skills", "plugins", "disk-cleanup", "optional-skills",
    "hermes-agent", "backups", "profiles", ".worktrees",
})

_EMPTY_DIR_SWEEP_PRUNE_DIRS = frozenset({
    ".git", "node_modules", "venv", ".venv",
    "site-packages", "__pycache__",
})

# Empty-directory pruning is limited to roots that this plugin explicitly
# owns.  An arbitrary empty directory under HERMES_HOME may be a user's
# project output or a control-plane staging area; emptiness is not ownership
# evidence.  Keep persistent Singularity/cache subtrees out of the sweep even
# when they live below the opt-in scratch root.
_EMPTY_DIR_OWNED_TOP_LEVEL = frozenset({
    "scratch", "tmp", "temp", "artifacts", "downloads",
})
_EMPTY_DIR_SWEEP_PROTECTED_NAMES = frozenset({"hermes-overlays", ".apptainer"})
_EMPTY_DIR_OWNERSHIP_MARKER = ".hermes-managed"

_MANAGED_ARTIFACT_RETENTION_DAYS = {
    "hook_outputs": 14,
    "spawn-trees": 30,
}
_MANAGED_ARTIFACT_SKIP_NAMES = frozenset({
    ".hermes-managed", ".in-progress", ".active", ".lock",
})
_DELETE_QUARANTINE_TOKEN = ".hermes-delete-"


# Paths under $HERMES_HOME that must NEVER be deleted by quick(),
# regardless of what the stored category says.  This is a defense-in-depth
# guard against stale tracked.json entries from before #34840.
_PROTECTED_CRON_PATHS: set[str] = set()

_PROTECTED_TRACKED_TOP_LEVEL = frozenset({
    ".worktrees",
    "backups",
    "disk-cleanup",
    "gateway",
    "hermes-agent",
    "kanban",
    "logs",
    "memories",
    "optional-skills",
    "pairing",
    "platforms",
    "plugins",
    "profiles",
    "sessions",
    "skills",
})
_PROTECTED_TRACKED_TOP_LEVEL_FILES = frozenset({
    ".env",
    "MEMORY.md",
    "SOUL.md",
    "USER.md",
    "auth.json",
    "channel_aliases.json",
    "channel_directory.json",
    "config.yaml",
    "feishu_comment_pairing.json",
    "gateway_state.json",
    "kanban.db",
    "memory_store.db",
    "processes.json",
    "projects.db",
    "response_store.db",
    "state.db",
    "verification_evidence.db",
})
_PROTECTED_TRACKED_TOP_LEVEL_CASEFOLD = frozenset(
    name.casefold() for name in _PROTECTED_TRACKED_TOP_LEVEL
)
_PROTECTED_TRACKED_TOP_LEVEL_FILES_CASEFOLD = frozenset(
    name.casefold() for name in _PROTECTED_TRACKED_TOP_LEVEL_FILES
)
_DISPOSABLE_TOP_LEVEL_PREFIXES = ("test_", "tmp_", "temp_")
_DISPOSABLE_TRACKED_ROOTS = frozenset({
    "artifacts",
    "cache",
    "downloads",
    "scratch",
    "temp",
    "tmp",
})


def _is_narrow_top_level_artifact(name: str) -> bool:
    """Allow only unmistakably disposable names at HERMES_HOME's root."""
    folded = name.casefold()
    return (
        folded.startswith(_DISPOSABLE_TOP_LEVEL_PREFIXES)
        or ".test." in folded
        or folded.endswith(".tmp")
    )


def _is_delete_quarantine_path(path: Path) -> bool:
    """Return whether *path* is preserved evidence from an uncertain delete."""
    return any(_DELETE_QUARANTINE_TOKEN in part for part in path.parts)


def _is_protected_cron_path(p: Path) -> bool:
    """Return True if *p* is a cron control-plane file/directory that must
    never be deleted.

    This matches, by EXACT path only, the ``cron/`` directory itself, known
    control-plane files (``jobs.json``, ``.tick.lock``), and the ``output/``
    root directory. It does NOT (and must not be "simplified" to) blanket-match
    everything under ``cron/output/`` — those run artifacts are disposable and
    are cleaned by retention policy; only the ``output/`` root itself is
    protected, because deleting it wholesale erases every job's retained run
    history at once.
    """
    try:
        rel = p.resolve().relative_to(get_hermes_home().resolve())
    except (OSError, ValueError):
        rel = None
    if rel is not None and rel.parts:
        parts = tuple(part.casefold() for part in rel.parts)
        if parts[0] in {"cron", "cronjobs"}:
            # Only regular artifacts below cron/output/<job>/ are disposable.
            # Every other descendant is scheduler control-plane state.
            return not (
                len(parts) >= 3
                and parts[1] == "output"
                and p.is_file()
                and not _is_link_like(p)
            )

    # Lazily build the set once per process so HERMES_HOME is resolved
    # exactly once.
    if not _PROTECTED_CRON_PATHS:
        hermes_home = get_hermes_home()
        for parent in ("cron", "cronjobs"):
            base = hermes_home / parent
            for protected in (base, base / "output", base / "jobs.json", base / ".tick.lock"):
                _PROTECTED_CRON_PATHS.add(
                    os.path.normcase(os.path.normpath(str(protected.resolve())))
                )
    resolved = os.path.normcase(os.path.normpath(str(p.resolve())))
    return resolved in _PROTECTED_CRON_PATHS


def _is_protected_tracked_path(path: Path) -> bool:
    """Protect durable Hermes state from manual or stale tracking records."""
    if _is_delete_quarantine_path(path):
        return True
    try:
        rel = path.resolve().relative_to(get_hermes_home().resolve())
    except (OSError, ValueError):
        return False
    if not rel.parts:
        return True
    top = rel.parts[0].casefold()
    if len(rel.parts) == 1:
        return (
            not _is_narrow_top_level_artifact(top)
            or top in _PROTECTED_TRACKED_TOP_LEVEL_FILES_CASEFOLD
        )
    if top in {"cron", "cronjobs"}:
        return False  # _is_protected_cron_path applies the narrower exception.
    return top not in _DISPOSABLE_TRACKED_ROOTS


class _OwnedEmptyDirRootEvidence(NamedTuple):
    """Filesystem objects that authorized one empty-directory sweep."""

    path: Path
    root_identity: os.stat_result
    marker_identity: os.stat_result


def _same_regular_file_object(path: Path, expected: os.stat_result) -> bool:
    """Revalidate a selected non-reparse regular file by filesystem ID."""
    try:
        current = path.lstat()
        return (
            stat.S_ISREG(current.st_mode)
            and not _is_link_like(path)
            and os.path.samestat(expected, current)
        )
    except OSError:
        return False


def _owned_empty_dir_root_matches(evidence: _OwnedEmptyDirRootEvidence) -> bool:
    """Return whether both objects that granted sweep ownership are unchanged."""
    return _same_directory_object(
        evidence.path, evidence.root_identity
    ) and _same_regular_file_object(
        evidence.path / _EMPTY_DIR_OWNERSHIP_MARKER,
        evidence.marker_identity,
    )


def _is_owned_empty_dir_root(evidence: _OwnedEmptyDirRootEvidence) -> bool:
    """Return whether *path* carries explicit ownership evidence.

    Names such as ``scratch`` and ``tmp`` are common in user projects and
    cannot authorize deletion by themselves.  The creator of a disposable
    Hermes root must leave a regular ``.hermes-managed`` marker containing
    the expected root name.  Missing, symlinked, or malformed markers fail
    closed and keep the entire subtree untouched.
    """
    path = evidence.path
    if path.name not in _EMPTY_DIR_OWNED_TOP_LEVEL:
        return False
    marker = path / _EMPTY_DIR_OWNERSHIP_MARKER
    try:
        if not _owned_empty_dir_root_matches(evidence):
            return False
        owned = marker.read_text(encoding="utf-8").strip() == path.name
        if not owned:
            return False
        # A lock/active marker is evidence that this namespace may still be
        # live.  Treat malformed or inaccessible markers as active too.
        if _has_active_maintenance_marker(path):
            return False
        # Bind the content and active-marker checks to the same root and
        # ownership-marker objects selected before validation.  A replacement
        # root carrying a copied marker must not inherit deletion authority.
        return _owned_empty_dir_root_matches(evidence)
    except (OSError, UnicodeError):
        return False


def _capture_owned_empty_dir_root(
    path: Path,
) -> Optional[_OwnedEmptyDirRootEvidence]:
    """Capture, validate, and bind a root plus its ownership marker."""
    try:
        evidence = _OwnedEmptyDirRootEvidence(
            path=path,
            root_identity=path.lstat(),
            marker_identity=(path / _EMPTY_DIR_OWNERSHIP_MARKER).lstat(),
        )
    except OSError:
        return None
    if not _is_owned_empty_dir_root(evidence):
        return None
    # The validator itself is a Python boundary and can be delayed or wrapped;
    # revalidate after it returns before exposing the evidence to traversal.
    return evidence if _owned_empty_dir_root_matches(evidence) else None


def _has_active_maintenance_marker(path: Path) -> bool:
    """Return whether *path* has a live or unverifiable maintenance marker."""
    for name in (".in-progress", ".active", ".lock"):
        marker = path / name
        try:
            marker.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            return True
        # A symlink or any existing marker is intentionally conservative:
        # stale ownership cannot be inferred from its name or contents.
        return True
    return False


def _capture_identity(path: Path) -> Optional[Dict[str, int]]:
    """Capture the filesystem object identity required for later deletion."""
    try:
        stat_result = path.lstat()
    except (OSError, ValueError):
        return None
    return {
        "version": 1,
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
        "mode": int(stat_result.st_mode),
        "size": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
        "ctime_ns": int(stat_result.st_ctime_ns),
    }


def _is_link_like(path: Path) -> bool:
    """Reject symlinks and Windows reparse points such as junctions."""
    try:
        if path.is_symlink():
            return True
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
        reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        return bool(attributes & reparse_flag)
    except OSError:
        return True


def _identity_matches(path: Path, expected: Dict[str, int]) -> bool:
    """Return whether *path* is still the exact object that was tracked."""
    current = _capture_identity(path)
    return current is not None and current == expected


def _same_directory_object(path: Path, expected: os.stat_result) -> bool:
    """Revalidate a selected non-reparse directory by stable filesystem ID."""
    try:
        current = path.lstat()
        return (
            stat.S_ISDIR(current.st_mode)
            and not _is_link_like(path)
            and os.path.samestat(expected, current)
        )
    except OSError:
        return False


def _validated_tracked_item(item: Any):
    """Return validated tracking fields, or ``None`` for untrusted evidence."""
    if not isinstance(item, dict):
        return None
    path_value = item.get("path")
    category = item.get("category")
    timestamp = item.get("timestamp")
    size = item.get("size")
    identity = item.get("identity")
    if (
        not isinstance(path_value, str)
        or not path_value
        or category not in ALLOWED_CATEGORIES
        or not isinstance(timestamp, str)
        or not timestamp
        or not isinstance(size, (int, float))
        or isinstance(size, bool)
        or (isinstance(size, float) and not size.is_integer())
        or not isinstance(identity, dict)
    ):
        return None
    try:
        path = Path(path_value)
        parsed_timestamp = datetime.fromisoformat(timestamp)
        if parsed_timestamp.tzinfo is None:
            parsed_timestamp = parsed_timestamp.replace(tzinfo=timezone.utc)
        parsed_size = int(size)
        parsed_identity = {
            key: int(identity[key])
            for key in (
                "version",
                "device",
                "inode",
                "mode",
                "size",
                "mtime_ns",
                "ctime_ns",
            )
        }
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        return None
    if parsed_size < 0 or parsed_identity["version"] != 1:
        return None
    return path, category, parsed_timestamp, parsed_size, parsed_identity


def _managed_marker_matches(path: Path, expected: str) -> bool:
    """Return true only for a regular, readable ownership marker."""
    marker = path / _EMPTY_DIR_OWNERSHIP_MARKER
    try:
        if not marker.is_file() or _is_link_like(marker):
            return False
        value = marker.read_text(encoding="utf-8").strip()
        if expected == "hook_outputs":
            if value == expected:
                return True
            prefix = "hook_outputs:v2:"
            bound_name = value.removeprefix(prefix)
            if not value.startswith(prefix) or bound_name != path.name:
                return False
            parts = bound_name.split("-")
            if len(parts) not in (2, 3) or parts[0] != "s":
                return False
            digest = parts[1]
            if len(digest) != 24 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                return False
            if len(parts) == 2:
                return True
            suffix = parts[2]
            return suffix == str(int(suffix)) and 1 <= int(suffix) <= 31
        if expected != "spawn-trees":
            return value == expected
        prefix = "spawn-trees:"
        digest = value.removeprefix(prefix)
        return (
            value.startswith(prefix)
            and len(digest) == 24
            and all(char in "0123456789abcdef" for char in digest)
            and path.name == f"s-{digest}"
        )
    except (OSError, UnicodeError):
        return False


class _ManagedArtifactLease(NamedTuple):
    """Opened directory objects that authorize one retention deletion."""

    root_path: Path
    root_identity: os.stat_result
    root_resolved: Path
    root_handle: int
    session_path: Path
    session_identity: os.stat_result
    session_resolved: Path
    session_handle: int
    marker_identity: os.stat_result


def _windows_opened_handle_path(handle: int) -> Path:
    """Return the kernel-resolved path for an opened Windows handle."""
    import ctypes

    get_path = ctypes.WinDLL("kernel32", use_last_error=True).GetFinalPathNameByHandleW
    get_path.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    get_path.restype = ctypes.c_uint32
    size = 512
    while True:
        buffer = ctypes.create_unicode_buffer(size)
        length = get_path(handle, buffer, size, 0)
        if length == 0:
            raise ctypes.WinError(ctypes.get_last_error())
        if length < size:
            value = buffer.value
            break
        size = length + 1
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)


def _windows_directory_handle_matches_stat(
    handle: int,
    expected: os.stat_result,
) -> bool:
    """Compare a raw handle with Python stat using Python's own encoding.

    Windows exposes a 32-bit volume serial through
    ``BY_HANDLE_FILE_INFORMATION``, while some Python builds encode ``st_dev``
    as a wider device identifier. Re-stat the kernel-resolved handle path so
    both sides use the same representation, then compare stable file IDs.
    """
    try:
        actual = _windows_opened_handle_path(handle).lstat()
        return bool(
            stat.S_ISDIR(actual.st_mode)
            and not int(getattr(actual, "st_file_attributes", 0)) & 0x400
            and os.path.samestat(expected, actual)
        )
    except OSError:
        return False


def _same_artifact_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(
        os.path.normpath(str(right))
    )


def _open_managed_artifact_directory(path: Path, *, parent_fd: Optional[int] = None) -> int:
    """Open a directory object, denying replacement while it is retained."""
    if os.name != "nt":
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if parent_fd is None:
            return os.open(path, flags)
        return os.open(path.name, flags, dir_fd=parent_fd)

    import ctypes

    create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    handle = create_file(
        str(path),
        0x00000080,  # FILE_READ_ATTRIBUTES
        0x00000001 | 0x00000002,  # SHARE_READ | SHARE_WRITE; deny SHARE_DELETE
        None,
        3,  # OPEN_EXISTING
        0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(handle)


def _close_managed_artifact_directory(handle: int) -> None:
    if os.name == "nt":
        import ctypes

        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
    else:
        os.close(handle)


def _managed_artifact_directory_handle_matches(
    handle: int,
    path: Path,
    expected: os.stat_result,
    resolved: Path,
) -> bool:
    """Verify an opened directory still names the exact selected object."""
    try:
        current = path.lstat()
        if (
            not stat.S_ISDIR(current.st_mode)
            or _is_link_like(path)
            or not os.path.samestat(expected, current)
            or path.resolve(strict=True) != resolved
        ):
            return False
        if os.name != "nt":
            opened = os.fstat(handle)
            return stat.S_ISDIR(opened.st_mode) and os.path.samestat(expected, opened)
        return bool(
            _windows_directory_handle_matches_stat(handle, expected)
            and _same_artifact_path(_windows_opened_handle_path(handle), resolved)
        )
    except OSError:
        return False


@contextmanager
def _hold_managed_artifact_session(
    artifact_root: Path,
    artifact_root_identity: os.stat_result,
    session_dir: Path,
    session_identity: os.stat_result,
    root_name: str,
):
    """Bind the root and its exact session child through retention work."""
    root_handle: Optional[int] = None
    session_handle: Optional[int] = None
    try:
        root_resolved = artifact_root.resolve(strict=True)
        session_resolved = session_dir.resolve(strict=True)
        if session_resolved.parent != root_resolved:
            raise OSError("managed artifact session is not an exact root child")
        root_handle = _open_managed_artifact_directory(artifact_root)
        if not _managed_artifact_directory_handle_matches(
            root_handle,
            artifact_root,
            artifact_root_identity,
            root_resolved,
        ):
            raise OSError("managed artifact root changed while binding")
        session_handle = _open_managed_artifact_directory(
            session_dir,
            parent_fd=root_handle if os.name != "nt" else None,
        )
        if not _managed_artifact_directory_handle_matches(
            session_handle,
            session_dir,
            session_identity,
            session_resolved,
        ):
            raise OSError("managed artifact session changed while binding")
        marker = session_dir / _EMPTY_DIR_OWNERSHIP_MARKER
        marker_identity = marker.lstat()
        if (
            not _same_regular_file_object(marker, marker_identity)
            or not _managed_marker_matches(session_dir, root_name)
            or not _same_regular_file_object(marker, marker_identity)
        ):
            raise OSError("managed artifact ownership marker changed while binding")
        yield _ManagedArtifactLease(
            artifact_root,
            artifact_root_identity,
            root_resolved,
            root_handle,
            session_dir,
            session_identity,
            session_resolved,
            session_handle,
            marker_identity,
        )
    finally:
        if session_handle is not None:
            _close_managed_artifact_directory(session_handle)
        if root_handle is not None:
            _close_managed_artifact_directory(root_handle)


def _managed_artifact_lease_matches(
    lease: _ManagedArtifactLease,
    root_name: str,
) -> bool:
    """Revalidate directory handles plus marker/active-maintenance policy."""
    return bool(
        _managed_artifact_directory_handle_matches(
            lease.root_handle,
            lease.root_path,
            lease.root_identity,
            lease.root_resolved,
        )
        and _managed_artifact_directory_handle_matches(
            lease.session_handle,
            lease.session_path,
            lease.session_identity,
            lease.session_resolved,
        )
        and _same_regular_file_object(
            lease.session_path / _EMPTY_DIR_OWNERSHIP_MARKER,
            lease.marker_identity,
        )
        and _managed_marker_matches(lease.session_path, root_name)
        and _same_regular_file_object(
            lease.session_path / _EMPTY_DIR_OWNERSHIP_MARKER,
            lease.marker_identity,
        )
        and not _has_active_maintenance_marker(lease.session_path)
    )


def _artifact_identity_from_stat(value: os.stat_result) -> Dict[str, int]:
    return {
        "version": 1,
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _bound_managed_artifact_stat(
    lease: _ManagedArtifactLease,
    name: str,
) -> os.stat_result:
    if os.name == "nt":
        return (lease.session_path / name).lstat()
    return os.stat(name, dir_fd=lease.session_handle, follow_symlinks=False)


def _atomic_unlink_managed_artifact(
    lease: _ManagedArtifactLease,
    root_name: str,
    name: str,
    expected_identity: Dict[str, int],
) -> Optional[str]:
    """Quarantine/delete one exact session child through the bound directory."""
    if (
        name in {"", ".", ".."}
        or Path(name).name != name
        or _DELETE_QUARANTINE_TOKEN in name
    ):
        return "invalid managed artifact name"
    quarantine_name = f".{name}{_DELETE_QUARANTINE_TOKEN}{uuid.uuid4().hex}"
    fd: Optional[int] = None
    try:
        if not _managed_artifact_lease_matches(lease, root_name):
            return "managed artifact container identity changed"
        flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if os.name == "nt":
            fd = _open_delete_candidate(lease.session_path / name)
        else:
            fd = os.open(name, flags, dir_fd=lease.session_handle)
        opened = os.fstat(fd)
        selected = _bound_managed_artifact_stat(lease, name)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(getattr(opened, "st_file_attributes", 0)) & 0x400
            or not os.path.samestat(opened, selected)
            or _artifact_identity_from_stat(opened) != expected_identity
        ):
            return "managed artifact filesystem identity changed"
        if os.name == "nt":
            import msvcrt

            actual = _windows_opened_handle_path(msvcrt.get_osfhandle(fd))
            expected = lease.session_resolved / name
            if not _same_artifact_path(actual, expected):
                return "managed artifact opened outside its bound session"
        if not _managed_artifact_lease_matches(lease, root_name):
            return "managed artifact container identity changed"

        if os.name == "nt":
            os.replace(
                lease.session_path / name,
                lease.session_path / quarantine_name,
            )
        else:
            os.replace(
                name,
                quarantine_name,
                src_dir_fd=lease.session_handle,
                dst_dir_fd=lease.session_handle,
            )
        moved = _bound_managed_artifact_stat(lease, quarantine_name)
        if not os.path.samestat(opened, moved):
            return _restore_quarantined_path(
                lease.session_path / quarantine_name,
                lease.session_path / name,
                moved,
            )
        if not _managed_artifact_lease_matches(lease, root_name):
            return "managed artifact container changed; preserved quarantine"
        if os.name == "nt":
            (lease.session_path / quarantine_name).unlink()
        else:
            os.unlink(quarantine_name, dir_fd=lease.session_handle)
        return None
    except OSError as exc:
        return str(exc)
    finally:
        if fd is not None:
            os.close(fd)


def _remove_path(path: Path) -> Optional[str]:
    """Remove one regular file or directory, returning an error if unsure."""
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            return "not a regular file or directory"
    except OSError as exc:
        return str(exc)
    return None


def _restore_quarantined_path(
    quarantine: Path,
    original: Path,
    moved: os.stat_result,
) -> str:
    """No-clobber restore a pathname replacement moved during cleanup."""
    try:
        quarantined = quarantine.lstat()
        if (
            not stat.S_ISREG(quarantined.st_mode)
            or not os.path.samestat(moved, quarantined)
            or os.path.lexists(original)
        ):
            raise OSError("restore destination is occupied or quarantine changed")
        if os.name == "nt":
            # Windows rename is no-replace, so a newer race winner survives.
            os.rename(quarantine, original)
        else:
            # link() atomically publishes only when the original name is free.
            os.link(quarantine, original, follow_symlinks=False)
            restored = original.lstat()
            quarantined = quarantine.lstat()
            if not (
                stat.S_ISREG(restored.st_mode)
                and stat.S_ISREG(quarantined.st_mode)
                and os.path.samestat(moved, restored)
                and os.path.samestat(moved, quarantined)
            ):
                raise OSError("restored object identity changed")
            quarantine.unlink()
        restored = original.lstat()
        if stat.S_ISREG(restored.st_mode) and os.path.samestat(moved, restored):
            return (
                "filesystem identity changed during delete; restored the "
                f"pathname replacement to {original}"
            )
    except OSError:
        pass
    return (
        f"filesystem identity changed during delete; preserved at {quarantine} "
        f"instead of overwriting {original}"
    )


def _open_delete_candidate(path: Path) -> int:
    """Open *path* while allowing an atomic rename on Windows."""
    if os.name != "nt":
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        return os.open(path, flags)

    import ctypes
    import msvcrt

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
        0x80000000,  # GENERIC_READ
        0x00000001 | 0x00000002 | 0x00000004,  # SHARE_READ|WRITE|DELETE
        None,
        3,  # OPEN_EXISTING
        0x00200000,  # FILE_FLAG_OPEN_REPARSE_POINT
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError()
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDONLY)
    except BaseException:
        ctypes.windll.kernel32.CloseHandle(handle)
        raise


def _atomic_unlink_regular(
    path: Path, expected_identity: Optional[Dict[str, int]] = None
) -> Optional[str]:
    """Move an opened object to a nonce path before unlinking it."""
    if _is_delete_quarantine_path(path):
        return "preserved delete quarantine"
    fd: Optional[int] = None
    quarantine = path.with_name(f".{path.name}.hermes-delete-{uuid.uuid4().hex}")
    try:
        fd = _open_delete_candidate(path)
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _is_link_like(path)
            or not path.is_file()
        ):
            return "path is not a regular non-link file"
        if expected_identity is not None and _capture_identity(path) != expected_identity:
            return "tracked filesystem identity changed"
        os.replace(path, quarantine)
        moved = quarantine.lstat()
        if not os.path.samestat(opened, moved):
            return _restore_quarantined_path(quarantine, path, moved)
        try:
            quarantine.unlink()
        except OSError as exc:
            return f"delete failed; preserved at {quarantine}: {exc}"
        return None
    except OSError as exc:
        return str(exc)
    finally:
        if fd is not None:
            os.close(fd)


def _remove_tracked_path(path: Path, identity: Dict[str, int]) -> Optional[str]:
    """Remove only the unchanged filesystem object authorized by tracking."""
    if _is_protected_tracked_path(path) or _is_protected_cron_path(path):
        return "protected durable path"
    if not stat.S_ISREG(identity.get("mode", 0)):
        return "tracked path is not a regular file"
    return _atomic_unlink_regular(path, identity)


def _atomic_rmdir_empty(
    path: Path,
    expected: os.stat_result,
    root_evidence: Optional[_OwnedEmptyDirRootEvidence] = None,
) -> bool:
    """Remove the exact selected empty directory, preserving uncertainty."""
    if _is_delete_quarantine_path(path):
        return False
    quarantine = path.with_name(
        f".{path.name}{_DELETE_QUARANTINE_TOKEN}{uuid.uuid4().hex}"
    )
    try:
        if (
            root_evidence is not None
            and not _is_owned_empty_dir_root(root_evidence)
        ):
            return False
        current = path.lstat()
        if (
            not stat.S_ISDIR(current.st_mode)
            or _is_link_like(path)
            or not os.path.samestat(expected, current)
        ):
            return False
        os.replace(path, quarantine)
        moved = quarantine.lstat()
        if (
            not stat.S_ISDIR(moved.st_mode)
            or _is_link_like(quarantine)
            or not os.path.samestat(expected, moved)
        ):
            return False
        if (
            root_evidence is not None
            and not _is_owned_empty_dir_root(root_evidence)
        ):
            return False
        try:
            quarantine.rmdir()
        except OSError:
            return False
        return True
    except OSError:
        return False


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Track / forget
# ---------------------------------------------------------------------------

def track(
    path_str: str,
    category: str,
    silent: bool = False,
    expected_identity: Optional[Dict[str, int]] = None,
) -> bool:
    """Register a file for tracking. Returns True if newly tracked."""
    if category not in ALLOWED_CATEGORIES:
        _log(f"WARN: unknown category '{category}', using 'other'")
        category = "other"

    source_path = Path(path_str).expanduser()

    try:
        if _is_link_like(source_path) or not source_path.is_file():
            _log(f"REJECT: {source_path} (not a regular file)")
            return False
        path = source_path.resolve()
    except (OSError, ValueError):
        _log(f"REJECT: {source_path} (filesystem identity unavailable)")
        return False

    if not path.exists():
        _log(f"SKIP: {path} (does not exist)")
        return False

    if not is_safe_path(path):
        _log(f"REJECT: {path} (outside HERMES_HOME)")
        return False

    if _is_protected_tracked_path(path) or _is_protected_cron_path(path):
        _log(f"REJECT: {path} (protected durable path)")
        return False

    identity = _capture_identity(path)
    if identity is None:
        _log(f"SKIP: {path} (filesystem identity unavailable)")
        return False
    if expected_identity is not None and identity != expected_identity:
        _log(f"REJECT: {path} (creation identity changed before tracking)")
        return False
    size = identity["size"] if path.is_file() else 0
    with _tracked_registry_lock():
        tracked = _load_tracked_unlocked()
        if any(
            isinstance(item, dict) and item.get("path") == str(path)
            for item in tracked
        ):
            return False
        tracked.append({
            "path": str(path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "size": size,
            "identity": identity,
        })
        _save_tracked_unlocked(tracked)
    _log(f"TRACKED: {path} ({category}, {fmt_size(size)})")
    if not silent:
        print(f"Tracked: {path} ({category}, {fmt_size(size)})")
    return True


def forget(path_str: str) -> int:
    """Remove a path from tracking without deleting the file."""
    p = Path(path_str).resolve()
    with _tracked_registry_lock():
        tracked = _load_tracked_unlocked()
        before = len(tracked)
        tracked = [i for i in tracked if Path(i["path"]).resolve() != p]
        removed = before - len(tracked)
        if removed:
            _save_tracked_unlocked(tracked)
            _log(f"FORGOT: {p} ({removed} entries)")
    return removed


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run() -> Tuple[List[Dict], List[Dict]]:
    """Return (auto_delete_list, needs_prompt_list) without touching files."""
    tracked = load_tracked()
    now = datetime.now(timezone.utc)

    auto: List[Dict] = []
    prompt: List[Dict] = []

    for item in tracked:
        validated = _validated_tracked_item(item)
        if validated is None:
            # A malformed record is not deletion authorization.  Omit it
            # from the preview rather than aborting the whole report.
            continue
        p, cat, timestamp, size, identity = validated
        if (
            not p.exists()
            or not is_safe_path(p)
            or _is_protected_tracked_path(p)
            or _is_protected_cron_path(p)
            or not stat.S_ISREG(identity.get("mode", 0))
            or not _identity_matches(p, identity)
        ):
            continue
        age = (now - timestamp).days

        # Re-validate stale "cron-output" entries (fixes #37721).
        if cat == "cron-output":
            re_cat = guess_category(p)
            if re_cat != "cron-output":
                # Stale entry — would be skipped by quick(); omit from
                # dry-run output too.
                continue

        if cat == "test":
            auto.append(item)
        elif cat == "temp" and age > 7:
            auto.append(item)
        elif cat == "cron-output" and age > 14:
            auto.append(item)
        elif cat == "research" and age > 30:
            prompt.append(item)
        elif cat == "chrome-profile" and age > 14:
            prompt.append(item)
        elif size > 500 * 1024 * 1024:
            prompt.append(item)

    return auto, prompt


# ---------------------------------------------------------------------------
# Quick cleanup
# ---------------------------------------------------------------------------

def quick(paths: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Safe deterministic cleanup — no prompts.

    Returns: ``{"deleted": N, "artifacts": N, "empty_dirs": N,
               "freed": bytes, "errors": [str, ...]}``.
    """
    tracked = load_tracked()
    scoped_paths = None
    if paths is not None:
        scoped_paths = set()
        for path in paths:
            try:
                scoped_paths.add(str(Path(path).resolve()))
            except (OSError, TypeError, ValueError):
                continue
    now = datetime.now(timezone.utc)
    deleted = 0
    freed = 0
    new_tracked: List[Any] = []
    errors: List[str] = []
    artifacts = 0

    for item in tracked:
        validated = _validated_tracked_item(item)
        if validated is None:
            # Preserve malformed records and continue.  Hand-edited or
            # partially-written tracking state is never proof of ownership.
            errors.append(f"invalid tracked record: {item!r}")
            new_tracked.append(item)
            continue
        p, cat, timestamp, size, identity = validated

        if scoped_paths is not None and str(p.resolve()) not in scoped_paths:
            new_tracked.append(item)
            continue

        try:
            p.stat()
        except FileNotFoundError:
            _log(f"STALE: {p} (removed from tracking)")
            continue
        except (OSError, ValueError) as exc:
            _log(f"ERROR inspecting {p}: {exc}")
            errors.append(f"{p}: {exc}")
            new_tracked.append(item)
            continue

        # tracked.json is durable state, not proof that the current target is
        # still owned by Hermes.  A stale or hand-edited record must never make
        # quick() delete a path outside the active HERMES_HOME (or approved
        # /tmp/hermes-* scope).
        if not is_safe_path(p):
            _log(f"SKIP unsafe tracked path: {p}")
            errors.append(f"{p}: unsafe path")
            new_tracked.append(item)
            continue

        if _is_protected_tracked_path(p) or _is_protected_cron_path(p):
            _log(f"SKIP protected tracked path: {p}")
            errors.append(f"{p}: protected durable path")
            new_tracked.append(item)
            continue

        if not stat.S_ISREG(identity.get("mode", 0)):
            _log(f"SKIP non-file tracked path: {p}")
            errors.append(f"{p}: tracked path is not a regular file")
            new_tracked.append(item)
            continue

        if not _identity_matches(p, identity):
            _log(f"SKIP changed tracked identity: {p}")
            errors.append(f"{p}: tracked filesystem identity changed")
            new_tracked.append(item)
            continue

        age = (now - timestamp).days

        # ---- stale-state migration (fixes #37721) ----
        # Old tracked.json entries may carry a "cron-output" category for
        # paths that are NOT under cron/output/ (e.g. cron/jobs.json).
        # guess_category() was fixed in #34840, but existing entries are
        # never re-validated.  Re-classify here so stale entries for cron
        # control-plane state are not deleted.
        if cat == "cron-output":
            re_cat = guess_category(p)
            if re_cat != "cron-output":
                _log(
                    f"SKIP stale cron-output entry: {p} "
                    f"(re-classified as {re_cat!r})"
                )
                # Drop the stale entry — it was misclassified.
                continue

        # Hard safety net: never delete cron control-plane state even if
        # the category somehow slipped through re-validation above.
        if _is_protected_cron_path(p):
            _log(f"SKIP protected cron path: {p}")
            continue

        should_delete = (
            cat == "test"
            or (cat == "temp" and age > 7)
            or (cat == "cron-output" and age > 14)
        )

        if should_delete:
            error = _remove_tracked_path(p, identity)
            if error is None:
                freed += size
                deleted += 1
                _log(f"DELETED: {p} ({cat}, {fmt_size(size)})")
            else:
                _log(f"ERROR deleting {p}: {error}")
                errors.append(f"{p}: {error}")
                new_tracked.append(item)
        else:
            new_tracked.append(item)

    # Scoped session cleanup is limited to the exact newly tracked paths. The
    # manual/global quick command additionally performs retention and empty-dir
    # sweeps under their separately marked ownership roots.
    if scoped_paths is not None:
        _commit_tracked_snapshot(tracked, new_tracked)
        _log(
            f"QUICK_SUMMARY: {deleted} files, 0 dirs, "
            f"{fmt_size(freed)}"
        )
        return {
            "deleted": deleted,
            "artifacts": 0,
            "empty_dirs": 0,
            "freed": freed,
            "errors": errors,
        }

    # Prune artifact files only inside explicitly marked session directories.
    # The marker is the durable ownership proof; directory names alone are not.
    hermes_home = get_hermes_home()
    for root_name, retention_days in _MANAGED_ARTIFACT_RETENTION_DAYS.items():
        artifact_root = hermes_home / root_name
        try:
            artifact_root_identity = artifact_root.lstat()
            if (
                _is_link_like(artifact_root)
                or not stat.S_ISDIR(artifact_root_identity.st_mode)
            ):
                continue
            session_dirs = [
                (p, p.lstat())
                for p in artifact_root.iterdir()
                if p.is_dir() and not _is_link_like(p)
            ]
            if not _same_directory_object(artifact_root, artifact_root_identity):
                continue
        except OSError:
            continue
        for session_dir, session_identity in session_dirs:
            try:
                if (
                    not _same_directory_object(
                        artifact_root, artifact_root_identity
                    )
                    or not _same_directory_object(session_dir, session_identity)
                    or not _managed_marker_matches(session_dir, root_name)
                    or _has_active_maintenance_marker(session_dir)
                ):
                    continue
                with _hold_managed_artifact_session(
                    artifact_root,
                    artifact_root_identity,
                    session_dir,
                    session_identity,
                    root_name,
                ) as lease:
                    if not _managed_artifact_lease_matches(lease, root_name):
                        continue
                    names = (
                        [entry.name for entry in session_dir.iterdir()]
                        if os.name == "nt"
                        else os.listdir(lease.session_handle)
                    )
                    for name in names:
                        artifact = session_dir / name
                        if not _managed_artifact_lease_matches(lease, root_name):
                            break
                        if (
                            name in _MANAGED_ARTIFACT_SKIP_NAMES
                            or _is_delete_quarantine_path(Path(name))
                        ):
                            continue
                        try:
                            selected = _bound_managed_artifact_stat(lease, name)
                            if (
                                not stat.S_ISREG(selected.st_mode)
                                or int(
                                    getattr(selected, "st_file_attributes", 0)
                                )
                                & 0x400
                            ):
                                continue
                            identity = _artifact_identity_from_stat(selected)
                            age_seconds = (
                                now.timestamp()
                                - identity["mtime_ns"] / 1_000_000_000
                            )
                            if age_seconds <= retention_days * 24 * 60 * 60:
                                continue
                            size = identity["size"]
                            error = _atomic_unlink_managed_artifact(
                                lease,
                                root_name,
                                name,
                                identity,
                            )
                            if error is not None:
                                errors.append(f"{artifact}: {error}")
                                continue
                            artifacts += 1
                            freed += size
                            _log(
                                f"DELETED: {artifact} "
                                "(managed artifact retention)"
                            )
                        except OSError as exc:
                            errors.append(f"{artifact}: {exc}")
            except OSError as exc:
                errors.append(f"{session_dir}: {exc}")

    # Remove empty dirs under HERMES_HOME, but never recurse into known
    # durable state trees.  Some installs place the Hermes checkout, venv,
    # and desktop build under HERMES_HOME; a full rglob over that tree can
    # stall the gateway event loop for minutes.
    empty_removed = 0
    sweep_stack: List[
        Tuple[Path, bool, os.stat_result, _OwnedEmptyDirRootEvidence]
    ] = []
    try:
        for top in hermes_home.iterdir():
            root_evidence = _capture_owned_empty_dir_root(top)
            if (
                root_evidence is not None
                and not _is_delete_quarantine_path(top)
            ):
                sweep_stack.append(
                    (top, False, root_evidence.root_identity, root_evidence)
                )
    except OSError:
        sweep_stack = []

    while sweep_stack:
        dirpath, visited, selected_identity, root_evidence = sweep_stack.pop()
        if not _is_owned_empty_dir_root(root_evidence):
            continue
        if visited:
            try:
                if (
                    not any(dirpath.iterdir())
                    and _is_owned_empty_dir_root(root_evidence)
                    and _atomic_rmdir_empty(
                        dirpath,
                        selected_identity,
                        root_evidence,
                    )
                ):
                    empty_removed += 1
                    _log(f"DELETED: {dirpath} (empty dir)")
            except OSError:
                pass
            continue

        sweep_stack.append((dirpath, True, selected_identity, root_evidence))
        try:
            for child in dirpath.iterdir():
                child_identity = child.lstat()
                if (
                    stat.S_ISDIR(child_identity.st_mode)
                    and not _is_link_like(child)
                    and not _is_delete_quarantine_path(child)
                    and child.name not in _EMPTY_DIR_SWEEP_PRUNE_DIRS
                    and child.name not in _EMPTY_DIR_SWEEP_PROTECTED_NAMES
                    and _same_directory_object(child, child_identity)
                    and _is_owned_empty_dir_root(root_evidence)
                ):
                    sweep_stack.append(
                        (child, False, child_identity, root_evidence)
                    )
        except OSError:
            pass

    _commit_tracked_snapshot(tracked, new_tracked)
    _log(
        f"QUICK_SUMMARY: {deleted} files, {empty_removed} dirs, "
        f"{fmt_size(freed)}"
    )
    return {
        "deleted": deleted,
        "artifacts": artifacts,
        "empty_dirs": empty_removed,
        "freed": freed,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Deep cleanup (interactive — not called from plugin hooks)
# ---------------------------------------------------------------------------

def deep(
    confirm: Optional[callable] = None,
) -> Dict[str, Any]:
    """Deep cleanup.

    Runs :func:`quick` first, then asks the *confirm* callable for each
    risky item (research > 30d beyond 10 newest, chrome-profile > 14d,
    any file > 500 MB).  *confirm(item)* must return True to delete.

    Returns: ``{"quick": {...}, "deep_deleted": N, "deep_freed": bytes}``.
    """
    quick_result = quick()

    if confirm is None:
        # No interactive confirmer — deep stops after the quick pass.
        return {"quick": quick_result, "deep_deleted": 0, "deep_freed": 0}

    tracked = load_tracked()
    now = datetime.now(timezone.utc)
    research, chrome, large = [], [], []

    for item in tracked:
        validated = _validated_tracked_item(item)
        if validated is None:
            continue
        p, cat, timestamp, size, identity = validated
        try:
            p.stat()
        except (FileNotFoundError, OSError, ValueError):
            continue
        if (
            not is_safe_path(p)
            or _is_protected_cron_path(p)
            or _is_protected_tracked_path(p)
            or not stat.S_ISREG(identity.get("mode", 0))
        ):
            continue
        age = (now - timestamp).days

        if cat == "research" and age > 30:
            research.append(item)
        elif cat == "chrome-profile" and age > 14:
            chrome.append(item)
        elif size > 500 * 1024 * 1024:
            large.append(item)

    research.sort(key=lambda x: x["timestamp"], reverse=True)
    old_research = research[10:]

    freed, count = 0, 0
    to_remove: List[Dict] = []

    for group in (old_research, chrome, large):
        for item in group:
            if confirm(item):
                validated = _validated_tracked_item(item)
                if validated is None:
                    continue
                p, _cat, _timestamp, size, identity = validated
                if (
                    not is_safe_path(p)
                    or _is_protected_cron_path(p)
                    or _is_protected_tracked_path(p)
                    or not stat.S_ISREG(identity.get("mode", 0))
                ):
                    continue
                error = _remove_tracked_path(p, identity)
                if error is None:
                    to_remove.append(item)
                    freed += size
                    count += 1
                    _log(
                        f"DELETED: {p} ({item['category']}, "
                        f"{fmt_size(size)})"
                    )
                else:
                    _log(f"ERROR deleting {p}: {error}")

    if to_remove:
        remove_paths = {i["path"] for i in to_remove}
        _commit_tracked_snapshot(
            tracked, [i for i in tracked if i["path"] not in remove_paths]
        )

    return {"quick": quick_result, "deep_deleted": count, "deep_freed": freed}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def status() -> Dict[str, Any]:
    """Return per-category breakdown and top 10 largest tracked files."""
    tracked = load_tracked()
    cats: Dict[str, Dict] = {}
    for item in tracked:
        c = item["category"]
        cats.setdefault(c, {"count": 0, "size": 0})
        cats[c]["count"] += 1
        cats[c]["size"] += item["size"]

    existing = [
        (i["path"], i["size"], i["category"])
        for i in tracked if Path(i["path"]).exists()
    ]
    existing.sort(key=lambda x: x[1], reverse=True)

    return {
        "categories": cats,
        "top10": existing[:10],
        "total_tracked": len(tracked),
    }


def format_status(s: Dict[str, Any]) -> str:
    """Human-readable status string (for slash command output)."""
    lines = [f"{'Category':<20} {'Files':>6}  {'Size':>10}", "-" * 40]
    cats = s["categories"]
    for cat, d in sorted(cats.items(), key=lambda x: x[1]["size"], reverse=True):
        lines.append(f"{cat:<20} {d['count']:>6}  {fmt_size(d['size']):>10}")

    if not cats:
        lines.append("(nothing tracked yet)")

    lines.append("")
    lines.append("Top 10 largest tracked files:")
    if not s["top10"]:
        lines.append("  (none)")
    else:
        for rank, (path, size, cat) in enumerate(s["top10"], 1):
            lines.append(f"  {rank:>2}. {fmt_size(size):>8}  [{cat}]  {path}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-categorisation from tool-call inspection
# ---------------------------------------------------------------------------

_TEST_PATTERNS = ("test_", "tmp_")
_TEST_SUFFIXES = (".test.py", ".test.js", ".test.ts", ".test.md")


def guess_category(path: Path) -> Optional[str]:
    """Return a category label for *path*, or None if we shouldn't track it.

    Used by the ``post_tool_call`` hook to auto-track ephemeral files.
    """
    if not is_safe_path(path):
        return None

    # Skip the state dir itself, logs, memory files, sessions, config.
    hermes_home = get_hermes_home()
    try:
        rel = path.resolve().relative_to(hermes_home)
        top = rel.parts[0] if rel.parts else ""
        if top in {
            "disk-cleanup", "logs", "memories", "sessions", "config.yaml",
            "skills", "plugins", ".env", "USER.md", "MEMORY.md", "SOUL.md",
            "auth.json", "hermes-agent",
        }:
            return None
        if top == "cron" or top == "cronjobs":
            # Only files under the disposable ``output/`` subtree are
            # cleanup candidates. Top-level cron control-plane state
            # (e.g. ``jobs.json``, ``.tick.lock``) must never be
            # auto-tracked — deleting it wipes the live scheduler
            # registry. See issue #32164.
            if len(rel.parts) >= 3 and rel.parts[1] == "output":
                return "cron-output"
            return None
        if top == "cache":
            return "temp"
    except ValueError:
        # Path isn't under HERMES_HOME (e.g. /tmp/hermes-*) — fall through.
        pass

    name = path.name
    if name.startswith(_TEST_PATTERNS):
        return "test"
    if any(name.endswith(sfx) for sfx in _TEST_SUFFIXES):
        return "test"
    return None
