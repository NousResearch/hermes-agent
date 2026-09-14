"""disk_cleanup — ephemeral file cleanup library behind the disk-cleanup plugin.

Rules: explicitly owned cache artifacts after 7 days; cron-output after 14 days. Prompt-only: research
(keep 10 newest, > 30 days), chrome-profile > 14 days, any file > 500 MB.
Arbitrary workspace and platform-temp paths are never inferred to be disposable by name.
"""

from __future__ import annotations

import contextlib
import errno
import functools
import json
import logging
import os
import shutil
import stat
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_LARGE_FILE_BYTES = 500 * 1024 * 1024
_DELETE_LOCK = threading.Lock()
_STATE_LOCK = threading.RLock()
_STATE_LOCK_TIMEOUT_SECONDS = 5.0
_LOCK_CONTENTION_ERRNOS = {
    errno.EACCES,
    errno.EAGAIN,
    errno.EDEADLK,
    errno.EWOULDBLOCK,
}


def _state_file(name: str) -> Path:
    """``$HERMES_HOME/disk-cleanup/<name>`` — deliberately outside ``$HERMES_HOME/logs/``."""
    return get_hermes_home() / "disk-cleanup" / name


def _try_lock_state(handle) -> bool:
    """Take the profile-state lock once without blocking."""
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno not in _LOCK_CONTENTION_ERRNOS:
                raise
            return False
    else:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in _LOCK_CONTENTION_ERRNOS:
                raise
            return False
    return True


def _unlock_state(handle) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _open_state_lock(path: Path):
    """Open the lock as a real regular file; POSIX refuses symlink traversal."""
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if os.name == "nt":  # no O_NOFOLLOW; reject an already-present reparse link
        if path.is_symlink():  # pragma: no cover - exercised on Windows CI
            raise RuntimeError("disk-cleanup state lock must not be a symlink")
    else:
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("disk-cleanup state lock must be a regular file")
        return os.fdopen(descriptor, "r+b", buffering=0)
    except Exception:
        os.close(descriptor)
        raise


@contextlib.contextmanager
def _state_transaction() -> Iterator[None]:
    """Serialize one tracking transaction across gateway and worker processes."""
    with _STATE_LOCK:
        lock_path = _state_file("tracked.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with _open_state_lock(lock_path) as handle:
            if os.name == "nt" and os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"\0")
                handle.flush()
            deadline = time.monotonic() + _STATE_LOCK_TIMEOUT_SECONDS
            while not _try_lock_state(handle):
                if time.monotonic() >= deadline:
                    raise RuntimeError("disk-cleanup state lock timed out")
                time.sleep(0.05)
            try:
                yield
            finally:
                with contextlib.suppress(OSError):
                    _unlock_state(handle)


def is_safe_path(path: Path) -> bool:
    """Accept paths that stay lexically and physically inside this profile."""
    try:
        lexical = _absolute_without_symlinks(path)
        resolved = path.resolve()
        lexical_home = _absolute_without_symlinks(get_hermes_home())
        resolved_home = get_hermes_home().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    if _is_descendant(lexical, lexical_home):
        return _is_descendant(resolved, resolved_home)
    return _is_descendant(resolved, resolved_home)


def _log(message: str) -> None:
    """Append to the audit log; never let it break the agent loop."""
    with contextlib.suppress(OSError):
        log_file = _state_file("cleanup.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")


def _load_tracked_unlocked() -> List[Dict[str, Any]]:
    tf = _state_file("tracked.json")
    tf.parent.mkdir(parents=True, exist_ok=True)
    if not tf.exists():
        return []
    with contextlib.suppress(ValueError, OSError):
        data = json.loads(tf.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    bak = tf.with_suffix(".json.bak")
    if bak.exists():
        with contextlib.suppress(Exception):
            data = json.loads(bak.read_text(encoding="utf-8"))
            if isinstance(data, list):
                _log("WARN: tracked.json corrupted — restored from .bak")
                return data
    _log("WARN: tracked.json corrupted, no backup — starting fresh")
    return []


def load_tracked() -> List[Dict[str, Any]]:
    """Load tracked.json under the shared profile-state lock."""
    with _state_transaction():
        return _load_tracked_unlocked()


def _save_tracked_unlocked(tracked: List[Dict[str, Any]]) -> None:
    tf = _state_file("tracked.json")
    tf.parent.mkdir(parents=True, exist_ok=True)
    tmp = tf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(tracked, indent=2), encoding="utf-8")
    if tf.exists():
        shutil.copy2(tf, tf.with_suffix(".json.bak"))
    tmp.replace(tf)


def save_tracked(tracked: List[Dict[str, Any]]) -> None:
    """Atomically replace tracked.json under the shared profile-state lock."""
    with _state_transaction():
        _save_tracked_unlocked(tracked)


ALLOWED_CATEGORIES = {
    "temp", "test", "research", "download", "chrome-profile", "cron-output", "other"}

_EMPTY_DIR_SWEEP_PRUNE_DIRS = frozenset({
    ".git", "node_modules", "venv", ".venv", "site-packages", "__pycache__"})

_MANAGED_HERMES_ROOTS = {
    ("cache", "vision", "temp_vision_images"): "temp",
    ("cache", "video", "temp_video_files"): "temp",
    ("cron", "output"): "cron-output",
    ("cronjobs", "output"): "cron-output",
}


def _managed_hermes_roots() -> Dict[Path, str]:
    home = get_hermes_home().resolve()
    roots = {}
    for parts, category in _MANAGED_HERMES_ROOTS.items():
        root = home.joinpath(*parts)
        try:
            # The owned boundary itself and each existing ancestor beneath HOME
            # must be real directories, never a symlink to external storage.
            if root.resolve(strict=False) != root:
                continue
        except (OSError, RuntimeError):
            continue
        roots[root] = category
    return roots


def _absolute_without_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _is_descendant(path: Path, root: Path) -> bool:
    with contextlib.suppress(ValueError):
        return bool(path.relative_to(root).parts)
    return False


def _managed_category(path: Path) -> Optional[str]:
    try:
        lexical = _absolute_without_symlinks(path)
        resolved = path.resolve()
        lexical_home = _absolute_without_symlinks(get_hermes_home())
        resolved_home = get_hermes_home().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if _is_descendant(lexical, lexical_home) and not _is_descendant(resolved, resolved_home):
        return None
    for root, category in _managed_hermes_roots().items():
        with contextlib.suppress(ValueError):
            if resolved.relative_to(root).parts:
                return category
    with contextlib.suppress(ValueError):
        resolved.relative_to(resolved_home)
        return None
    return None


def _managed_sweep_root(path: Path) -> Optional[Path]:
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    for root in _managed_hermes_roots():
        with contextlib.suppress(ValueError):
            if resolved.relative_to(root).parts:
                return root
    return None

@functools.lru_cache(maxsize=8)  # keyed by home: a multiplexed process serves several profiles
def _protected_cron_paths(home: Path) -> frozenset:
    """Defense-in-depth for quick(): EXACT cron control-plane paths (``cron/``, ``output/`` root,
    ``jobs.json``, ``.tick.lock``) never deleted regardless of stored category (stale tracked.json).
    Never widen to everything under ``cron/output/``: run artifacts there are disposable; only
    wholesale deletion of ``output/`` is fatal."""
    home = home.resolve()
    return frozenset(str(x.resolve()) for parent in ("cron", "cronjobs")
                     for base in (home / parent,)
                     for x in (base, base / "output", base / "jobs.json", base / ".tick.lock"))


# Paths under $HERMES_HOME that must NEVER be deleted by quick(), regardless of what the stored category
# says. This is a defense-in-depth guard against stale tracked.json entries from before #34840.
def _is_protected_cron_path(p: Path) -> bool:
    return str(p.resolve()) in _protected_cron_paths(get_hermes_home())


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _managed_root(path: Path, category: str) -> Optional[Path]:
    """Return the exact owned root for a path/category pair."""
    try:
        canonical = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    for root, owned_category in _managed_hermes_roots().items():
        if category != owned_category:
            continue
        with contextlib.suppress(ValueError):
            if canonical.relative_to(root).parts:
                return root
    return None


def _generation_receipt(path: Path, category: str) -> Optional[Dict[str, Any]]:
    """Bind a tracked row to one file generation and its owned root generation."""
    try:
        file_stat = path.lstat()
        profile = str(get_hermes_home().resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    if not stat.S_ISREG(file_stat.st_mode):
        return None
    receipt = {
        "profile": profile,
        "device": file_stat.st_dev,
        "inode": file_stat.st_ino,
        "generation_size": file_stat.st_size,
        "mtime_ns": file_stat.st_mtime_ns,
        "ctime_ns": file_stat.st_ctime_ns,
    }
    root = _managed_root(path, category)
    if root is None:
        return receipt
    try:
        root_stat = root.lstat()
    except OSError:
        return None
    if not stat.S_ISDIR(root_stat.st_mode):
        return None
    return {
        **receipt,
        "root": str(root),
        "root_device": root_stat.st_dev,
        "root_inode": root_stat.st_ino,
    }


def track(
    path_str: str,
    category: str,
    silent: bool = False,
    *,
    owner: Optional[str] = None,
) -> bool:
    """Register one immutable file generation for tracking."""
    if category not in ALLOWED_CATEGORIES:
        _log(f"WARN: unknown category '{category}', using 'other'")
        category = "other"
    try:
        lexical_path = Path(path_str).expanduser()
        path = lexical_path.resolve()
        exists = path.exists()
    except (OSError, RuntimeError, ValueError):
        _log(f"SKIP: invalid path {path_str!r}")
        return False
    if not exists:
        _log(f"SKIP: {path} (does not exist)")
        return False
    if not is_safe_path(path):
        _log(f"REJECT: {path} (outside HERMES_HOME)")
        return False
    if category in {"test", "temp", "cron-output"} and guess_category(path) != category:
        _log(f"REJECT: {path} ({category} is not owned by disk-cleanup)")
        return False
    receipt = _generation_receipt(path, category)
    if receipt is None:
        _log(f"REJECT: {path} (only regular files can be tracked)")
        return False
    if category in {"test", "temp", "cron-output"} and "root" not in receipt:
        _log(f"REJECT: {path} ({category} has no owned generation receipt)")
        return False
    size = receipt["generation_size"]
    with _state_transaction():
        tracked = _load_tracked_unlocked()
        if any(
            isinstance(item, dict)
            and item.get("path") == str(path)
            and all(item.get(key) == receipt[key] for key in (
                "device", "inode", "generation_size", "mtime_ns", "ctime_ns"
            ))
            for item in tracked
        ):
            return False
        tracked.append({
            "path": str(path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "size": size,
            **receipt,
            **({"owner": owner} if owner else {}),
        })
        _save_tracked_unlocked(tracked)
    _log(f"TRACKED: {path} ({category}, {fmt_size(size)})")
    if not silent:
        print(f"Tracked: {path} ({category}, {fmt_size(size)})")
    return True


def forget(path_str: str) -> int:
    """Remove a path from tracking without deleting the file."""
    p = Path(path_str).resolve()
    with _state_transaction():
        tracked = _load_tracked_unlocked()
        kept = []
        for item in tracked:
            try:
                matches = Path(item["path"]).resolve() == p
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                matches = False
            if not matches:
                kept.append(item)
        removed = len(tracked) - len(kept)
        if removed:
            _save_tracked_unlocked(kept)
            _log(f"FORGOT: {p} ({removed} entries)")
    return removed


def _live_items(tracked: List[Dict], now: datetime, *, log_stale: bool = False) -> Iterator[Tuple[Dict, Path, int]]:
    """Yield ``(item, path, age_days)`` for entries whose path still exists."""
    for item in tracked:
        try:
            if not isinstance(item, dict):
                raise TypeError("entry is not an object")
            p = Path(item["path"])
            age = (now - datetime.fromisoformat(item["timestamp"])).days
            category = item["category"]
            size = item["size"]
            if category not in ALLOWED_CATEGORIES or not isinstance(size, (int, float)) or size < 0:
                raise ValueError("invalid category or size")
            exists = p.exists()
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if log_stale:
                _log(f"SKIP malformed tracking entry: {exc}")
            continue
        if exists:
            yield item, p, age
        elif log_stale:
            _log(f"STALE: {p} (removed from tracking)")


def _is_auto_delete(cat: str, age: int) -> bool:
    return cat == "test" or (cat == "temp" and age > 7) or (cat == "cron-output" and age > 14)


def _prompt_group(item: Dict, age: int) -> Optional[str]:
    """Prompt-only bucket: ``research`` / ``chrome`` / ``large`` or None."""
    cat = item["category"]
    if cat == "research" and age > 30:
        return "research"
    if cat == "chrome-profile" and age > 14:
        return "chrome"
    return "large" if item["size"] > _LARGE_FILE_BYTES else None


def _is_current_owned_file(item: Dict, path: Path) -> bool:
    """Revalidate the complete automatic-deletion boundary for one file."""
    if not _secure_unlink_supported():
        return False
    try:
        file_stat = path.lstat()
        canonical = path.resolve(strict=True)
        category = item.get("category")
        root = _managed_root(path, category)
        root_stat = root.lstat() if root is not None else None
    except (OSError, RuntimeError, ValueError):
        return False
    try:
        return (
            path.is_absolute()
            and canonical == path
            and stat.S_ISREG(file_stat.st_mode)
            and item.get("profile") == str(get_hermes_home().resolve())
            and root is not None
            and root_stat is not None
            and stat.S_ISDIR(root_stat.st_mode)
            and _same_file_generation(item, file_stat)
            and item.get("root") == str(root)
            and item.get("root_device") == root_stat.st_dev
            and item.get("root_inode") == root_stat.st_ino
            and is_safe_path(path)
            and guess_category(path) == category
            and not _is_protected_cron_path(path)
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _secure_unlink_supported() -> bool:
    """Whether this host can bind traversal and unlink to opened directory handles."""
    return (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
        and os.unlink in os.supports_dir_fd
    )


def _same_generation(item: Dict, current: os.stat_result, prefix: str = "") -> bool:
    return (
        item.get(f"{prefix}device") == current.st_dev
        and item.get(f"{prefix}inode") == current.st_ino
    )


def _same_file_generation(item: Dict, current: os.stat_result) -> bool:
    """Match both filesystem identity and in-place rewrite metadata."""
    return (
        _same_generation(item, current)
        and item.get("generation_size") == current.st_size
        and item.get("mtime_ns") == current.st_mtime_ns
        and item.get("ctime_ns") == current.st_ctime_ns
    )


def _unlink_at(filename: str, parent_fd: int) -> None:
    os.unlink(filename, dir_fd=parent_fd)


def _unlink_owned_generation(item: Dict, path: Path) -> bool:
    """Unlink through verified directory handles; never re-traverse a mutable pathname."""
    if not _secure_unlink_supported():
        return False
    category = item.get("category")
    root = _managed_root(path, category)
    if root is None or item.get("root") != str(root):
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if not relative.parts:
        return False

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: List[int] = []
    try:
        current_fd = os.open(root, directory_flags)
        descriptors.append(current_fd)
        root_stat = os.fstat(current_fd)
        if not stat.S_ISDIR(root_stat.st_mode) or not _same_generation(
            item, root_stat, "root_"
        ):
            return False
        for component in relative.parts[:-1]:
            current_fd = os.open(component, directory_flags, dir_fd=current_fd)
            descriptors.append(current_fd)
        filename = relative.parts[-1]
        file_stat = os.stat(filename, dir_fd=current_fd, follow_symlinks=False)
        if not stat.S_ISREG(file_stat.st_mode) or not _same_file_generation(
            item, file_stat
        ):
            return False
        _unlink_at(filename, current_fd)
        return True
    except (NotImplementedError, OSError, TypeError, ValueError):
        return False
    finally:
        for descriptor in reversed(descriptors):
            with contextlib.suppress(OSError):
                os.close(descriptor)


def _delete_item(item: Dict) -> Tuple[bool, Optional[str]]:
    """Revalidate and delete one owned item. Returns ``(deleted, error)``."""
    p = Path(str(item.get("path", "")))
    try:
        with _DELETE_LOCK:
            deleted = _unlink_owned_generation(item, p)
        if not deleted:
            _log(f"SKIP unmanaged path before delete: {p}")
            return False, None
    except (OSError, RuntimeError, ValueError) as e:
        _log(f"ERROR deleting {p}: {e}")
        return False, f"{p}: {e}"
    _log(f"DELETED: {p} ({item['category']}, {fmt_size(item['size'])})")
    return True, None


def dry_run() -> Tuple[List[Dict], List[Dict]]:
    """Return (auto_delete_list, needs_prompt_list) without touching files."""
    auto, prompt = [], []
    for item, p, age in _live_items(load_tracked(), datetime.now(timezone.utc)):
        cat = item.get("category")
        if _is_auto_delete(cat, age):
            if _is_current_owned_file(item, p):
                auto.append(item)
        elif _prompt_group(item, age) and is_safe_path(p):
            prompt.append(item)
    return auto, prompt


def quick(
    only_paths: Optional[Set[str]] = None,
    *,
    immediate_owner: Optional[str] = None,
) -> Dict[str, Any]:
    """Safe cleanup; ``immediate_owner`` scopes immediate files, not aged retention."""
    with _state_transaction():
        return _quick_locked(only_paths, immediate_owner=immediate_owner)


def _quick_locked(
    only_paths: Optional[Set[str]] = None,
    *,
    immediate_owner: Optional[str] = None,
) -> Dict[str, Any]:
    if only_paths is not None and immediate_owner is not None:
        raise ValueError("Select cleanup by path or owner, not both.")
    if not _secure_unlink_supported():
        _log("SKIP quick cleanup: secure directory-handle unlink is unavailable")
        return {"deleted": 0, "empty_dirs": 0, "freed": 0, "errors": []}
    deleted = freed = 0
    new_tracked: List[Dict] = []
    errors: List[str] = []
    selected = None if only_paths is None else {str(Path(path).resolve()) for path in only_paths}
    sweep_roots = (
        set(_managed_hermes_roots())
        if selected is None and immediate_owner is None
        else set()
    )
    for item, p, age in _live_items(
        _load_tracked_unlocked(), datetime.now(timezone.utc), log_stale=True
    ):
        try:
            item_path = str(p.resolve())
        except (OSError, RuntimeError):
            continue
        if selected is not None and item_path not in selected:
            new_tracked.append(item)
            continue
        cat = item.get("category")
        if (
            immediate_owner is not None
            and cat == "test"
            and item.get("owner") != immediate_owner
        ):
            new_tracked.append(item)
            continue
        if not _is_auto_delete(cat, age):
            new_tracked.append(item)
            continue
        if not _is_current_owned_file(item, p):
            _log(f"SKIP stale {cat} entry: {p} (not a current owned regular file)")
            continue
        if root := _managed_sweep_root(p):
            sweep_roots.add(root)
        was_deleted, err = _delete_item(item)
        if was_deleted:
            freed += item.get("size", 0)
            deleted += 1
        elif err is not None:
            errors.append(err)
            new_tracked.append(item)
    empty_removed = sum(_sweep_empty_dirs(root) for root in sweep_roots)
    _save_tracked_unlocked(new_tracked)
    _log(f"QUICK_SUMMARY: {deleted} files, {empty_removed} dirs, {fmt_size(freed)}")
    return {"deleted": deleted, "empty_dirs": empty_removed, "freed": freed, "errors": errors}


def _is_real_descendant_dir(path: Path, root: Path) -> bool:
    try:
        return path.resolve(strict=True) == path and _is_descendant(path, root)
    except (OSError, RuntimeError, ValueError):
        return False


def _subdirs(dirpath: Path, root: Path, exclude: frozenset) -> List[Path]:
    try:
        return [
            child for child in dirpath.iterdir()
            if child.name not in exclude and _is_real_descendant_dir(child, root)
        ]
    except OSError:
        return []


def _sweep_empty_dirs(root: Path) -> int:
    """Remove empty descendants of one explicitly owned ephemeral root."""
    if root not in _managed_hermes_roots() or root.is_symlink():
        return 0
    removed = 0
    stack: List[Tuple[Path, bool]] = [
        (top, False) for top in _subdirs(root, root, _EMPTY_DIR_SWEEP_PRUNE_DIRS)]
    while stack:
        dirpath, visited = stack.pop()
        if visited:
            with contextlib.suppress(OSError):
                if _is_real_descendant_dir(dirpath, root) and not any(dirpath.iterdir()):
                    dirpath.rmdir()
                    removed += 1
                    _log(f"DELETED: {dirpath} (empty dir)")
            continue
        stack.append((dirpath, True))
        stack.extend(
            (child, False) for child in _subdirs(dirpath, root, _EMPTY_DIR_SWEEP_PRUNE_DIRS)
        )
    return removed


def status() -> Dict[str, Any]:
    """Return per-category breakdown and top 10 largest tracked files."""
    tracked = load_tracked()
    cats: Dict[str, Dict] = {}
    existing = []
    valid = list(_live_items(tracked, datetime.now(timezone.utc), log_stale=True))
    for item, p, _age in valid:
        c = cats.setdefault(item["category"], {"count": 0, "size": 0})
        c["count"] += 1
        c["size"] += item["size"]
        existing.append((str(p), item["size"], item["category"]))
    existing.sort(key=lambda x: x[1], reverse=True)
    return {"categories": cats, "top10": existing[:10], "total_tracked": len(valid)}


def format_status(s: Dict[str, Any]) -> str:
    """Human-readable status string (for slash command output)."""
    lines = [f"{'Category':<20} {'Files':>6}  {'Size':>10}", "-" * 40]
    cats = s["categories"]
    for cat, d in sorted(cats.items(), key=lambda x: x[1]["size"], reverse=True):
        lines.append(f"{cat:<20} {d['count']:>6}  {fmt_size(d['size']):>10}")
    if not cats:
        lines.append("(nothing tracked yet)")
    lines += ["", "Top 10 largest tracked files:"]
    if not s["top10"]:
        lines.append("  (none)")
    for rank, (path, size, cat) in enumerate(s["top10"], 1):
        lines.append(f"  {rank:>2}. {fmt_size(size):>8}  [{cat}]  {path}")
    return "\n".join(lines)


def guess_category(path: Path) -> Optional[str]:
    """Return a category only when *path* belongs to an explicit ephemeral owner root."""
    return _managed_category(path)
