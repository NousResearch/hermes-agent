"""disk_cleanup — ephemeral file cleanup library behind the disk-cleanup plugin.

Rules: test files delete at task end (age >= 0); temp after 7 days; cron-output
after 14 days; empty dirs under HERMES_HOME always. Prompt-only: research
(keep 10 newest, > 30 days), chrome-profile > 14 days, any file > 500 MB.
Scope: strictly HERMES_HOME and /tmp/hermes-*; never ~/.hermes/logs/ or system dirs.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_LARGE_FILE_BYTES = 500 * 1024 * 1024


def _state_file(name: str) -> Path:
    """``$HERMES_HOME/disk-cleanup/<name>`` — deliberately outside ``$HERMES_HOME/logs/``."""
    return get_hermes_home() / "disk-cleanup" / name


def is_safe_path(path: Path) -> bool:
    """Accept only paths under HERMES_HOME or ``/tmp/hermes-*`` (rejects /mnt/c etc.)."""
    with contextlib.suppress(ValueError, OSError):
        path.resolve().relative_to(get_hermes_home())
        return True
    parts = path.parts
    return len(parts) >= 3 and parts[1] == "tmp" and parts[2].startswith("hermes-")


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
    with contextlib.suppress(ValueError):
        return json.loads(tf.read_text(encoding="utf-8"))
    bak = tf.with_suffix(".json.bak")
    if bak.exists():
        with contextlib.suppress(Exception):
            data = json.loads(bak.read_text(encoding="utf-8"))
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

# Top-level HERMES_HOME dirs whose empty subdirs are never swept (last row: user project trees).
_EMPTY_DIR_PROTECTED_TOP_LEVEL = frozenset({
    "logs", "memories", "sessions", "cron", "cronjobs",
    "cache", "skills", "plugins", "disk-cleanup", "optional-skills",
    "hermes-agent", "backups", "profiles", ".worktrees",
    "patches", "projects", "skins", "themes", "contributors"})

_EMPTY_DIR_SWEEP_PRUNE_DIRS = frozenset({
    ".git", "node_modules", "venv", ".venv", "site-packages", "__pycache__"})

# Top-level HERMES_HOME entries guess_category() never auto-tracks: state, logs, memory,
# sessions, config/secrets, and user project trees (test_* inside projects/ is not disposable).
_NEVER_TRACK_TOP_LEVEL = frozenset({
    "disk-cleanup", "logs", "memories", "sessions", "config.yaml",
    "skills", "plugins", ".env", "USER.md", "MEMORY.md", "SOUL.md",
    "auth.json", "hermes-agent",
    # User-authored project trees — never sweep empty directories inside these (#75403).
    # User-authored and project trees — never auto-delete files inside these just because they happen to be
    # named test_* or tmp_* (#75403, also #32164, #37721).
    "patches", "projects", "skins", "themes", "contributors",
    "profiles", "backups", "optional-skills"})

@functools.lru_cache(maxsize=1)  # built lazily so HERMES_HOME resolves once
def _protected_cron_paths() -> frozenset:
    """Defense-in-depth for quick(): EXACT cron control-plane paths (``cron/``, ``output/`` root,
    ``jobs.json``, ``.tick.lock``) never deleted regardless of stored category (stale tracked.json).
    Never widen to everything under ``cron/output/``: run artifacts there are disposable; only
    wholesale deletion of ``output/`` is fatal."""
    return frozenset(str(x) for parent in ("cron", "cronjobs") for base in (get_hermes_home() / parent,)
                     for x in (base, base / "output", base / "jobs.json", base / ".tick.lock"))


# Paths under $HERMES_HOME that must NEVER be deleted by quick(), regardless of what the stored category
# says. This is a defense-in-depth guard against stale tracked.json entries from before #34840.
def _is_protected_cron_path(p: Path) -> bool:
    return str(p.resolve()) in _protected_cron_paths()


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
    path = Path(path_str).resolve()
    if not path.exists():
        _log(f"SKIP: {path} (does not exist)")
        return False
    if not is_safe_path(path):
        _log(f"REJECT: {path} (outside HERMES_HOME)")
        return False
    size = path.stat().st_size if path.is_file() else 0
    tracked = load_tracked()
    if any(item["path"] == str(path) for item in tracked):
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
    kept = [i for i in tracked if Path(i["path"]).resolve() != p]
    removed = len(tracked) - len(kept)
    if removed:
        save_tracked(kept)
        _log(f"FORGOT: {p} ({removed} entries)")
    return removed


def _live_items(tracked: List[Dict], now: datetime, *, log_stale: bool = False) -> Iterator[Tuple[Dict, Path, int]]:
    """Yield ``(item, path, age_days)`` for entries whose path still exists."""
    for item in tracked:
        p = Path(item["path"])
        if p.exists():
            yield item, p, (now - datetime.fromisoformat(item["timestamp"])).days
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


def _delete_item(item: Dict) -> Optional[str]:
    """Delete a tracked file/dir and audit-log it. Returns an error string on OSError, else None."""
    p = Path(item["path"])
    try:
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    except OSError as e:
        _log(f"ERROR deleting {p}: {e}")
        return f"{p}: {e}"
    _log(f"DELETED: {p} ({item['category']}, {fmt_size(item['size'])})")
    return None


# Stored categories re-validated against guess_category() before use: old tracked.json entries
# may carry "cron-output" for control-plane files or "test" for files under protected trees.
_STALE_SKIP_NOTE = {"cron-output": "", "test": " — under protected tree"}


def dry_run() -> Tuple[List[Dict], List[Dict]]:
    """Return (auto_delete_list, needs_prompt_list) without touching files."""
    auto, prompt = [], []
    for item, p, age in _live_items(load_tracked(), datetime.now(timezone.utc)):
        cat = item["category"]
        # Stale cron-output entries are skipped by quick(); omit them here too.
        if cat == "cron-output" and guess_category(p) != "cron-output":
            continue
        if _is_auto_delete(cat, age):
            auto.append(item)
        elif _prompt_group(item, age):
            prompt.append(item)
    return auto, prompt


def quick() -> Dict[str, Any]:
    """Safe deterministic cleanup — no prompts. Returns ``{deleted, empty_dirs, freed, errors}``."""
    deleted = freed = 0
    new_tracked: List[Dict] = []
    errors: List[str] = []
    for item, p, age in _live_items(load_tracked(), datetime.now(timezone.utc), log_stale=True):
        cat = item["category"]
        if cat in _STALE_SKIP_NOTE and (re_cat := guess_category(p)) != cat:
            # Misclassified stale entry — drop it rather than delete the file.
            _log(f"SKIP stale {cat} entry: {p} (re-classified as {re_cat!r}{_STALE_SKIP_NOTE[cat]})")
            continue
        # Hard safety net even if re-validation above somehow let it through.
        if _is_protected_cron_path(p):
            _log(f"SKIP protected cron path: {p}")
            continue
        if not _is_auto_delete(cat, age):
            new_tracked.append(item)
            continue
        err = _delete_item(item)
        if err is None:
            freed += item["size"]
            deleted += 1
        else:
            errors.append(err)
            new_tracked.append(item)
    empty_removed = _sweep_empty_dirs(get_hermes_home())
    save_tracked(new_tracked)
    _log(f"QUICK_SUMMARY: {deleted} files, {empty_removed} dirs, {fmt_size(freed)}")
    return {"deleted": deleted, "empty_dirs": empty_removed, "freed": freed, "errors": errors}


def _subdirs(dirpath: Path, exclude: frozenset) -> List[Path]:
    try:
        return [c for c in dirpath.iterdir() if c.is_dir() and not c.is_symlink() and c.name not in exclude]
    except OSError:
        return []


def _sweep_empty_dirs(hermes_home: Path) -> int:
    """Remove empty dirs under HERMES_HOME without recursing into durable/heavy trees (a full
    rglob over a checkout+venv under HERMES_HOME can stall the gateway loop for minutes).
    Iterative post-order so parents emptied by child removal are caught."""
    removed = 0
    stack: List[Tuple[Path, bool]] = [
        (top, False) for top in _subdirs(hermes_home, _EMPTY_DIR_PROTECTED_TOP_LEVEL | _EMPTY_DIR_SWEEP_PRUNE_DIRS)]
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
    for item in tracked:
        c = cats.setdefault(item["category"], {"count": 0, "size": 0})
        c["count"] += 1
        c["size"] += item["size"]
    existing = sorted(((i["path"], i["size"], i["category"]) for i in tracked
                       if Path(i["path"]).exists()), key=lambda x: x[1], reverse=True)
    return {"categories": cats, "top10": existing[:10], "total_tracked": len(tracked)}


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


_TEST_PATTERNS = ("test_", "tmp_")
_TEST_SUFFIXES = (".test.py", ".test.js", ".test.ts", ".test.md")


def guess_category(path: Path) -> Optional[str]:
    """Category label for *path*, or None if we shouldn't track it (``post_tool_call`` hook)."""
    if not is_safe_path(path):
        return None
    with contextlib.suppress(ValueError):  # not under HERMES_HOME (/tmp/hermes-*) — name rules only
        rel = path.resolve().relative_to(get_hermes_home())
        top = rel.parts[0] if rel.parts else ""
        if top in _NEVER_TRACK_TOP_LEVEL:
            return None
        if top in ("cron", "cronjobs"):
            # Only the disposable ``output/`` subtree; control-plane state (jobs.json,
            # .tick.lock) must never be tracked — deleting it wipes the scheduler registry.
            return "cron-output" if len(rel.parts) >= 3 and rel.parts[1] == "output" else None
        if top == "cache":
            return "temp"
    name = path.name
    return "test" if name.startswith(_TEST_PATTERNS) or name.endswith(_TEST_SUFFIXES) else None
