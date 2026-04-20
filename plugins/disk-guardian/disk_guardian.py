"""disk_guardian — ephemeral file cleanup for Hermes Agent.

Library module wrapping the deterministic cleanup rules written by
@LVT382009 in PR #12212. The plugin ``__init__.py`` wires these
functions into ``post_tool_call`` and ``on_session_end`` hooks so
tracking and cleanup happen automatically — the agent never needs to
call a tool or remember a skill.

Rules:
  - test files    → delete immediately at task end (age >= 0)
  - temp files    → delete after 7 days
  - cron-output   → delete after 14 days
  - empty dirs    → always delete (under HERMES_HOME)
  - research      → keep 10 newest, prompt for older (deep only)
  - chrome-profile→ prompt after 14 days (deep only)
  - >500 MB files → prompt always (deep only)

Scope: strictly HERMES_HOME and /tmp/hermes-*
Never touches: ~/.hermes/logs/ or any system directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover — plugin may load before constants resolves
    import os

    def get_hermes_home() -> Path:  # type: ignore[no-redef]
        val = (os.environ.get("HERMES_HOME") or "").strip()
        return Path(val).resolve() if val else (Path.home() / ".hermes").resolve()


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_state_dir() -> Path:
    """State dir — separate from ``$HERMES_HOME/logs/``."""
    return get_hermes_home() / "disk-guardian"


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
    hermes_home = get_hermes_home()
    try:
        path.resolve().relative_to(hermes_home)
        return True
    except (ValueError, OSError):
        pass
    # Allow /tmp/hermes-* explicitly
    parts = path.parts
    if len(parts) >= 3 and parts[1] == "tmp" and parts[2].startswith("hermes-"):
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
        with open(log_file, "a") as f:
            f.write(f"[{ts}] {message}\n")
    except OSError:
        # Never let the audit log break the agent loop.
        pass


# ---------------------------------------------------------------------------
# tracked.json — atomic read/write, backup scoped to tracked.json only
# ---------------------------------------------------------------------------

def load_tracked() -> List[Dict[str, Any]]:
    """Load tracked.json.  Restores from ``.bak`` on corruption."""
    tf = get_tracked_file()
    tf.parent.mkdir(parents=True, exist_ok=True)

    if not tf.exists():
        return []

    try:
        return json.loads(tf.read_text())
    except (json.JSONDecodeError, ValueError):
        bak = tf.with_suffix(".json.bak")
        if bak.exists():
            try:
                data = json.loads(bak.read_text())
                _log("WARN: tracked.json corrupted — restored from .bak")
                return data
            except Exception:
                pass
        _log("WARN: tracked.json corrupted, no backup — starting fresh")
        return []


def save_tracked(tracked: List[Dict[str, Any]]) -> None:
    """Atomic write: ``.tmp`` → backup old → rename."""
    tf = get_tracked_file()
    tf.parent.mkdir(parents=True, exist_ok=True)
    tmp = tf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(tracked, indent=2))
    if tf.exists():
        shutil.copy2(tf, tf.with_suffix(".json.bak"))
    tmp.replace(tf)


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

ALLOWED_CATEGORIES = {
    "temp", "test", "research", "download",
    "chrome-profile", "cron-output", "other",
}


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Track / forget
# ---------------------------------------------------------------------------

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

    # Deduplicate
    if any(item["path"] == str(path) for item in tracked):
        return False

    tracked.append({
        "path": str(path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "size": size,
    })
    save_tracked(tracked)
    _log(f"TRACKED: {path} ({category}, {fmt_size(size)})")
    if not silent:
        print(f"Tracked: {path} ({category}, {fmt_size(size)})")
    return True


def forget(path_str: str) -> int:
    """Remove a path from tracking without deleting the file."""
    p = Path(path_str).resolve()
    tracked = load_tracked()
    before = len(tracked)
    tracked = [i for i in tracked if Path(i["path"]).resolve() != p]
    removed = before - len(tracked)
    if removed:
        save_tracked(tracked)
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
        p = Path(item["path"])
        if not p.exists():
            continue
        age = (now - datetime.fromisoformat(item["timestamp"])).days
        cat = item["category"]
        size = item["size"]

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

def quick() -> Dict[str, Any]:
    """Safe deterministic cleanup — no prompts.

    Returns: ``{"deleted": N, "empty_dirs": N, "freed": bytes,
               "errors": [str, ...]}``.
    """
    tracked = load_tracked()
    now = datetime.now(timezone.utc)
    deleted = 0
    freed = 0
    new_tracked: List[Dict] = []
    errors: List[str] = []

    for item in tracked:
        p = Path(item["path"])
        cat = item["category"]

        if not p.exists():
            _log(f"STALE: {p} (removed from tracking)")
            continue

        age = (now - datetime.fromisoformat(item["timestamp"])).days

        should_delete = (
            cat == "test"
            or (cat == "temp" and age > 7)
            or (cat == "cron-output" and age > 14)
        )

        if should_delete:
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
                freed += item["size"]
                deleted += 1
                _log(f"DELETED: {p} ({cat}, {fmt_size(item['size'])})")
            except OSError as e:
                _log(f"ERROR deleting {p}: {e}")
                errors.append(f"{p}: {e}")
                new_tracked.append(item)
        else:
            new_tracked.append(item)

    # Remove empty dirs under HERMES_HOME (but leave HERMES_HOME itself and
    # a short list of well-known top-level state dirs alone — a fresh install
    # has these empty, and deleting them would surprise the user).
    hermes_home = get_hermes_home()
    _PROTECTED_TOP_LEVEL = {
        "logs", "memories", "sessions", "cron", "cronjobs",
        "cache", "skills", "plugins", "disk-guardian", "optional-skills",
        "hermes-agent", "backups", "profiles", ".worktrees",
    }
    empty_removed = 0
    try:
        for dirpath in sorted(hermes_home.rglob("*"), reverse=True):
            if not dirpath.is_dir() or dirpath == hermes_home:
                continue
            try:
                rel_parts = dirpath.relative_to(hermes_home).parts
            except ValueError:
                continue
            # Skip the well-known top-level state dirs themselves.
            if len(rel_parts) == 1 and rel_parts[0] in _PROTECTED_TOP_LEVEL:
                continue
            try:
                if not any(dirpath.iterdir()):
                    dirpath.rmdir()
                    empty_removed += 1
                    _log(f"DELETED: {dirpath} (empty dir)")
            except OSError:
                pass
    except OSError:
        pass

    save_tracked(new_tracked)
    _log(
        f"QUICK_SUMMARY: {deleted} files, {empty_removed} dirs, "
        f"{fmt_size(freed)}"
    )
    return {
        "deleted": deleted,
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
        p = Path(item["path"])
        if not p.exists():
            continue
        age = (now - datetime.fromisoformat(item["timestamp"])).days
        cat = item["category"]

        if cat == "research" and age > 30:
            research.append(item)
        elif cat == "chrome-profile" and age > 14:
            chrome.append(item)
        elif item["size"] > 500 * 1024 * 1024:
            large.append(item)

    research.sort(key=lambda x: x["timestamp"], reverse=True)
    old_research = research[10:]

    freed, count = 0, 0
    to_remove: List[Dict] = []

    for group in (old_research, chrome, large):
        for item in group:
            if confirm(item):
                try:
                    p = Path(item["path"])
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
                    to_remove.append(item)
                    freed += item["size"]
                    count += 1
                    _log(
                        f"DELETED: {p} ({item['category']}, "
                        f"{fmt_size(item['size'])})"
                    )
                except OSError as e:
                    _log(f"ERROR deleting {item['path']}: {e}")

    if to_remove:
        remove_paths = {i["path"] for i in to_remove}
        save_tracked([i for i in tracked if i["path"] not in remove_paths])

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
            "disk-guardian", "logs", "memories", "sessions", "config.yaml",
            "skills", "plugins", ".env", "USER.md", "MEMORY.md", "SOUL.md",
            "auth.json", "hermes-agent",
        }:
            return None
        if top == "cron" or top == "cronjobs":
            return "cron-output"
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
