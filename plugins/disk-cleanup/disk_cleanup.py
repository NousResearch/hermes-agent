"""disk_cleanup — ephemeral file cleanup library behind the disk-cleanup plugin.

Rules: files in Hermes-owned temporary roots delete at task end; managed cache
artifacts after 7 days; cron-output after 14 days. Prompt-only: research
(keep 10 newest, > 30 days), chrome-profile > 14 days, any file > 500 MB.
Arbitrary paths under HERMES_HOME are never inferred to be disposable by name.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_LARGE_FILE_BYTES = 500 * 1024 * 1024
_REGISTERED_SYSTEM_TEMP_ROOTS: Dict[str, Tuple[int, int]] = {}


def _state_file(name: str) -> Path:
    """``$HERMES_HOME/disk-cleanup/<name>`` — deliberately outside ``$HERMES_HOME/logs/``."""
    return get_hermes_home() / "disk-cleanup" / name


def is_safe_path(path: Path) -> bool:
    """Accept paths inside the profile or a process-registered Hermes temp root."""
    try:
        lexical = _absolute_without_symlinks(path)
        resolved = path.resolve()
        lexical_home = _absolute_without_symlinks(get_hermes_home())
        resolved_home = get_hermes_home().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    if _is_descendant(lexical, lexical_home):
        return _is_descendant(resolved, resolved_home)
    if _is_descendant(resolved, resolved_home):
        return True
    return _system_temp_owner_root(path) is not None


def _log(message: str) -> None:
    """Append to the audit log; never let it break the agent loop."""
    with contextlib.suppress(OSError):
        log_file = _state_file("cleanup.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")


def load_tracked() -> List[Dict[str, Any]]:
    """Load tracked.json.  Restores from ``.bak`` on corruption."""
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


def save_tracked(tracked: List[Dict[str, Any]]) -> None:
    """Atomic write: ``.tmp`` → backup old → rename."""
    tf = _state_file("tracked.json")
    tf.parent.mkdir(parents=True, exist_ok=True)
    tmp = tf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(tracked, indent=2), encoding="utf-8")
    if tf.exists():
        shutil.copy2(tf, tf.with_suffix(".json.bak"))
    tmp.replace(tf)


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


def _temp_roots() -> Set[Tuple[Path, Path]]:
    roots = set()
    for candidate in (Path(tempfile.gettempdir()), Path("/tmp")):
        with contextlib.suppress(OSError):
            lexical = _absolute_without_symlinks(candidate)
            resolved = candidate.resolve()
            roots.add((lexical, resolved))
            roots.add((resolved, resolved))
    return roots


def _candidate_system_temp_owner_root(path: Path) -> Optional[Path]:
    """Find a real direct ``hermes-*`` child of a platform temp directory."""
    try:
        lexical = _absolute_without_symlinks(path)
        resolved = path.resolve()
        lexical_home = _absolute_without_symlinks(get_hermes_home())
        resolved_home = get_hermes_home().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    # A lexical path that starts inside HOME but resolves outside is a symlink
    # escape, not a system-temp candidate.
    if _is_descendant(lexical, lexical_home):
        return None
    for lexical_temp, resolved_temp in _temp_roots():
        with contextlib.suppress(ValueError, OSError):
            rel = lexical.relative_to(lexical_temp)
            if len(rel.parts) < 2 or not rel.parts[0].startswith("hermes-"):
                continue
            lexical_owner = lexical_temp / rel.parts[0]
            if lexical_owner.is_symlink() or not lexical_owner.is_dir():
                continue
            owner = lexical_owner.resolve()
            owner_rel = owner.relative_to(resolved_temp)
            if len(owner_rel.parts) != 1 or not _is_descendant(resolved, owner):
                continue
            # A test/profile HOME may itself be named hermes-*; never grant its
            # outer directory ownership over sibling paths.
            if resolved_home == owner or _is_descendant(resolved_home, owner):
                continue
            return owner
    return None


def register_system_temp_root(path: Path) -> bool:
    """Register a temp root observed in this process from an explicit file track."""
    owner = _candidate_system_temp_owner_root(path)
    if owner is None:
        return False
    try:
        stat = owner.stat()
    except (OSError, RuntimeError, ValueError):
        return False
    _REGISTERED_SYSTEM_TEMP_ROOTS[str(owner)] = (stat.st_dev, stat.st_ino)
    return True


def _system_temp_owner_root(path: Path) -> Optional[Path]:
    """Return *path*'s still-identical temp root when this process registered it."""
    owner = _candidate_system_temp_owner_root(path)
    if owner is None:
        return None
    expected = _REGISTERED_SYSTEM_TEMP_ROOTS.get(str(owner))
    if expected is None:
        return None
    try:
        current = owner.stat()
    except (OSError, RuntimeError, ValueError):
        return None
    return owner if (current.st_dev, current.st_ino) == expected else None


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
    owner = _system_temp_owner_root(resolved)
    return "test" if owner is not None and resolved != owner else None


def _managed_sweep_root(path: Path) -> Optional[Path]:
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    for root in _managed_hermes_roots():
        with contextlib.suppress(ValueError):
            if resolved.relative_to(root).parts:
                return root
    with contextlib.suppress(ValueError):
        resolved.relative_to(get_hermes_home().resolve())
        return None
    owner = _system_temp_owner_root(resolved)
    return owner if owner is not None and resolved != owner else None

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


def track(path_str: str, category: str, silent: bool = False) -> bool:
    """Register a file for tracking. Returns True if newly tracked."""
    if category not in ALLOWED_CATEGORIES:
        _log(f"WARN: unknown category '{category}', using 'other'")
        category = "other"
    try:
        lexical_path = Path(path_str).expanduser()
        if category == "test" and guess_category(lexical_path) is None:
            register_system_temp_root(lexical_path)
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
    try:
        size = path.stat().st_size if path.is_file() else 0
    except OSError:
        _log(f"SKIP: {path} (cannot stat)")
        return False
    tracked = load_tracked()
    if any(isinstance(item, dict) and item.get("path") == str(path) for item in tracked):
        return False
    tracked.append({"path": str(path), "timestamp": datetime.now(timezone.utc).isoformat(),
                    "category": category, "size": size})
    save_tracked(tracked)
    _log(f"TRACKED: {path} ({category}, {fmt_size(size)})")
    if not silent:
        print(f"Tracked: {path} ({category}, {fmt_size(size)})")
    return True


def forget(path_str: str) -> int:
    """Remove a path from tracking without deleting the file."""
    p = Path(path_str).resolve()
    tracked = load_tracked()
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
        save_tracked(kept)
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


def _delete_item(item: Dict) -> Tuple[bool, Optional[str]]:
    """Revalidate and delete one owned item. Returns ``(deleted, error)``."""
    p = Path(str(item.get("path", "")))
    try:
        p = p.resolve()
        if guess_category(p) != item.get("category"):
            _log(f"SKIP unmanaged path before delete: {p}")
            return False, None
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
        else:
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
        if _is_auto_delete(cat, age) and guess_category(p) != cat:
            continue
        if _is_auto_delete(cat, age):
            auto.append(item)
        elif _prompt_group(item, age):
            prompt.append(item)
    return auto, prompt


def quick(only_paths: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Safe deterministic cleanup — no prompts. Returns ``{deleted, empty_dirs, freed, errors}``."""
    deleted = freed = 0
    new_tracked: List[Dict] = []
    errors: List[str] = []
    selected = None if only_paths is None else {str(Path(path).resolve()) for path in only_paths}
    sweep_roots = set(_managed_hermes_roots()) if selected is None else set()
    for item, p, age in _live_items(load_tracked(), datetime.now(timezone.utc), log_stale=True):
        try:
            item_path = str(p.resolve())
        except (OSError, RuntimeError):
            continue
        if selected is not None and item_path not in selected:
            new_tracked.append(item)
            continue
        cat = item.get("category")
        if _is_auto_delete(cat, age) and guess_category(p) != cat:
            _log(f"SKIP stale {cat} entry: {p} (outside an owned ephemeral root)")
            continue
        if _is_protected_cron_path(p):
            _log(f"SKIP protected cron path: {p}")
            continue
        if not _is_auto_delete(cat, age):
            new_tracked.append(item)
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
    save_tracked(new_tracked)
    _log(f"QUICK_SUMMARY: {deleted} files, {empty_removed} dirs, {fmt_size(freed)}")
    return {"deleted": deleted, "empty_dirs": empty_removed, "freed": freed, "errors": errors}


def _subdirs(dirpath: Path, exclude: frozenset) -> List[Path]:
    try:
        return [c for c in dirpath.iterdir() if c.is_dir() and not c.is_symlink() and c.name not in exclude]
    except OSError:
        return []


def _sweep_empty_dirs(root: Path) -> int:
    """Remove empty descendants of one explicitly owned ephemeral root."""
    removed = 0
    stack: List[Tuple[Path, bool]] = [
        (top, False) for top in _subdirs(root, _EMPTY_DIR_SWEEP_PRUNE_DIRS)]
    while stack:
        dirpath, visited = stack.pop()
        if visited:
            with contextlib.suppress(OSError):
                if not any(dirpath.iterdir()):
                    dirpath.rmdir()
                    removed += 1
                    _log(f"DELETED: {dirpath} (empty dir)")
            continue
        stack.append((dirpath, True))
        stack.extend((child, False) for child in _subdirs(dirpath, _EMPTY_DIR_SWEEP_PRUNE_DIRS))
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
