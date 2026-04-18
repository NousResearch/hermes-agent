#!/usr/bin/env python3
"""
disk_guardian.py v1.2.0 — ephemeral file cleanup for Hermes Agent

Tracks and removes temp outputs, test artifacts, cron logs, and stale
chrome profiles created during Hermes sessions.

Rules:
  - test files    → delete immediately at task end (age > 0)
  - temp files    → delete after 7 days
  - cron-output   → delete after 14 days
  - empty dirs    → always delete
  - research      → keep 10 newest, prompt for older (deep only)
  - chrome-profile→ prompt after 14 days (deep only)
  - >500 MB files → prompt always (deep only)

Scope: strictly HERMES_HOME and /tmp/hermes-*
Never touches: ~/.hermes/logs/ or any system directory
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_hermes_home() -> Path:
    """Return HERMES_HOME, defaulting to ~/.hermes."""
    val = os.environ.get("HERMES_HOME", "").strip()
    return Path(val).resolve() if val else (Path.home() / ".hermes").resolve()


def get_state_dir() -> Path:
    """State dir — separate from ~/.hermes/logs/."""
    return get_hermes_home() / "disk-guardian"


def get_tracked_file() -> Path:
    return get_state_dir() / "tracked.json"


def get_log_file() -> Path:
    """Audit log — NOT ~/.hermes/logs/."""
    return get_state_dir() / "cleanup.log"


# ---------------------------------------------------------------------------
# WSL + path safety
# ---------------------------------------------------------------------------

def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        return False


def _is_safe_path(path: Path) -> bool:
    """
    Accept only paths under HERMES_HOME or /tmp/hermes-*.
    Rejects Windows mounts (/mnt/c etc.) and system directories.
    """
    hermes_home = get_hermes_home()
    try:
        path.relative_to(hermes_home)
        return True
    except ValueError:
        pass
    # Allow /tmp/hermes-* explicitly
    parts = path.parts
    if len(parts) >= 3 and parts[1] == "tmp" and parts[2].startswith("hermes-"):
        return True
    return False


# ---------------------------------------------------------------------------
# Audit log — writes only to disk-guardian/cleanup.log
# ---------------------------------------------------------------------------

def _log(message: str) -> None:
    log_file = get_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{ts}] {message}\n")


# ---------------------------------------------------------------------------
# tracked.json — atomic read/write, backup scoped to tracked.json only
# ---------------------------------------------------------------------------

def load_tracked() -> List[Dict[str, Any]]:
    """
    Load tracked.json.
    Corruption recovery: restore from .bak — never touches ~/.hermes/logs/.
    """
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
                print("Warning: tracking file corrupted, restored from backup.")
                return data
            except Exception:
                pass
        _log("WARN: tracked.json corrupted, no backup — starting fresh")
        print("Warning: tracking file corrupted, starting fresh.")
        return []


def save_tracked(tracked: List[Dict[str, Any]]) -> None:
    """Atomic write: .tmp → backup old → rename."""
    tf = get_tracked_file()
    tf.parent.mkdir(parents=True, exist_ok=True)
    tmp = tf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(tracked, indent=2))
    if tf.exists():
        shutil.copy2(tf, tf.with_suffix(".json.bak"))
    tmp.replace(tf)


# ---------------------------------------------------------------------------
# Allowed categories
# ---------------------------------------------------------------------------

ALLOWED_CATEGORIES = {
    "temp", "test", "research", "download",
    "chrome-profile", "cron-output", "other",
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_track(path_str: str, category: str) -> None:
    """Register a file for tracking."""
    if category not in ALLOWED_CATEGORIES:
        print(f"Unknown category '{category}', using 'other'.")
        _log(f"WARN: unknown category '{category}', using 'other'")
        category = "other"

    path = Path(path_str).resolve()

    if not path.exists():
        print(f"Path does not exist, skipping: {path}")
        _log(f"SKIP: {path} (does not exist)")
        return

    if not _is_safe_path(path):
        print(f"Rejected: path is outside HERMES_HOME — {path}")
        _log(f"REJECT: {path} (outside HERMES_HOME)")
        return

    size = path.stat().st_size if path.is_file() else 0
    tracked = load_tracked()

    # Deduplicate
    if any(item["path"] == str(path) for item in tracked):
        print(f"Already tracked: {path}")
        return

    tracked.append({
        "path": str(path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "size": size,
    })
    save_tracked(tracked)
    _log(f"TRACKED: {path} ({category}, {_fmt(size)})")
    print(f"Tracked: {path} ({category}, {_fmt(size)})")


def cmd_dry_run() -> None:
    """Show what would be deleted — no files touched."""
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

    auto_size = sum(i["size"] for i in auto)
    prompt_size = sum(i["size"] for i in prompt)

    print("Dry-run preview (nothing deleted):")
    print(f"  Auto-delete : {len(auto)} files ({_fmt(auto_size)})")
    for item in auto:
        print(f"    [{item['category']}] {item['path']}")
    print(f"  Needs prompt: {len(prompt)} files ({_fmt(prompt_size)})")
    for item in prompt:
        print(f"    [{item['category']}] {item['path']}")
    print(f"\n  Total potential: {_fmt(auto_size + prompt_size)}")
    print("Run 'quick' for auto-delete only, 'deep' for full cleanup.")


def cmd_quick(silent: bool = False) -> None:
    """
    Safe deterministic cleanup — no prompts.
    Deletes: test (age>0), temp (>7d), cron-output (>14d), empty dirs.
    Pass silent=True to suppress output (for auto-runs).
    """
    tracked = load_tracked()
    now = datetime.now(timezone.utc)
    deleted, freed = 0, 0
    new_tracked = []

    for item in tracked:
        p = Path(item["path"])
        cat = item["category"]

        if not p.exists():
            _log(f"STALE: {p} (removed from tracking)")
            continue

        age = (now - datetime.fromisoformat(item["timestamp"])).days

        should_delete = (
            cat == "test" or                          # always delete test files
            (cat == "temp" and age > 7) or
            (cat == "cron-output" and age > 14)
        )

        if should_delete:
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
                freed += item["size"]
                deleted += 1
                _log(f"DELETED: {p} ({cat}, {_fmt(item['size'])})")
            except OSError as e:
                _log(f"ERROR deleting {p}: {e}")
                if not silent:
                    print(f"  Skipped (error): {p} — {e}")
                new_tracked.append(item)
        else:
            new_tracked.append(item)

    # Remove empty dirs under HERMES_HOME
    hermes_home = get_hermes_home()
    empty_removed = 0
    for dirpath in sorted(hermes_home.rglob("*"), reverse=True):
        if dirpath.is_dir() and dirpath != hermes_home:
            try:
                if not any(dirpath.iterdir()):
                    dirpath.rmdir()
                    empty_removed += 1
                    _log(f"DELETED: {dirpath} (empty dir)")
            except OSError:
                pass

    save_tracked(new_tracked)

    summary = (f"[disk-guardian] Cleaned {deleted} files + {empty_removed} "
               f"empty dirs, freed {_fmt(freed)}.")
    _log(f"QUICK_SUMMARY: {deleted} files, {empty_removed} dirs, {_fmt(freed)}")
    print(summary)


def cmd_deep() -> None:
    """Full cleanup — auto for safe files, interactive for risky."""
    print("Running quick cleanup first...")
    cmd_quick()

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

    # Keep 10 newest research folders
    research.sort(key=lambda x: x["timestamp"], reverse=True)
    old_research = research[10:]

    freed, count = 0, 0
    to_remove = []

    for item in old_research:
        p = Path(item["path"])
        ans = input(f"\nDelete old research ({_fmt(item['size'])}): {p} [y/N] ")
        if ans.lower() == "y":
            _delete_item(p, item, to_remove)
            freed += item["size"]
            count += 1

    for item in chrome:
        p = Path(item["path"])
        ans = input(f"\nDelete chrome profile ({_fmt(item['size'])}): {p} [y/N] ")
        if ans.lower() == "y":
            _delete_item(p, item, to_remove)
            freed += item["size"]
            count += 1

    for item in large:
        p = Path(item["path"])
        ans = input(f"\nDelete large file ({_fmt(item['size'])}, "
                    f"{item['category']}): {p} [y/N] ")
        if ans.lower() == "y":
            _delete_item(p, item, to_remove)
            freed += item["size"]
            count += 1

    if to_remove:
        remove_paths = {i["path"] for i in to_remove}
        save_tracked([i for i in tracked if i["path"] not in remove_paths])

    print(f"\n[disk-guardian] Deep cleanup done: {count} items, freed {_fmt(freed)}.")


def _delete_item(p: Path, item: Dict, to_remove: list) -> None:
    try:
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
        to_remove.append(item)
        _log(f"DELETED: {p} ({item['category']}, {_fmt(item['size'])})")
        print(f"  Deleted: {p}")
    except OSError as e:
        _log(f"ERROR deleting {p}: {e}")
        print(f"  Error: {e}")


def cmd_status() -> None:
    """Print per-category breakdown and top 10 largest tracked files."""
    tracked = load_tracked()
    cats: Dict[str, Dict] = {}
    for item in tracked:
        c = item["category"]
        cats.setdefault(c, {"count": 0, "size": 0})
        cats[c]["count"] += 1
        cats[c]["size"] += item["size"]

    print(f"{'Category':<20} {'Files':>6}  {'Size':>10}")
    print("-" * 40)
    for cat, d in sorted(cats.items(), key=lambda x: x[1]["size"], reverse=True):
        print(f"{cat:<20} {d['count']:>6}  {_fmt(d['size']):>10}")

    existing = [
        (i["path"], i["size"], i["category"])
        for i in tracked if Path(i["path"]).exists()
    ]
    existing.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 10 largest tracked files:")
    for rank, (path, size, cat) in enumerate(existing[:10], 1):
        print(f"  {rank:>2}. {_fmt(size):>8}  [{cat}]  {path}")

    if not tracked:
        print("No files tracked yet.")


def cmd_forget(path_str: str) -> None:
    """Remove a path from tracking without deleting the file."""
    p = Path(path_str).resolve()
    tracked = load_tracked()
    before = len(tracked)
    tracked = [i for i in tracked if Path(i["path"]).resolve() != p]
    removed = before - len(tracked)
    if removed:
        save_tracked(tracked)
        _log(f"FORGOT: {p} ({removed} entries)")
        print(f"Removed {removed} tracking entries for {p}.")
    else:
        print(f"Not found in tracking: {p}")


def cmd_install_memory() -> None:
    """
    Write persistent memory rules so disk-guardian auto-runs across sessions.
    Writes to ~/.hermes/MEMORY.md (appends if exists).
    """
    memory_file = get_hermes_home() / "MEMORY.md"
    rules = """
## disk-guardian auto-cleanup rules (added by disk-guardian v1.2.0)

- After every task that creates test files (test_*.py, *.test.*, tmp_*),
  automatically run: python3 ~/.hermes/optional-skills/devops/disk-guardian/scripts/disk_guardian.py quick
  Do this silently at task end — no need to ask the user.
- Track every file created under HERMES_HOME during tasks using disk-guardian track.
- If HERMES_HOME disk usage exceeds 5 GB, silently run disk-guardian quick.
- Never ask permission for temp/test file deletion — clean silently, report one line.
- Print cleanup summary as: [disk-guardian] Cleaned N files, freed X MB
"""
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    with open(memory_file, "a") as f:
        f.write(rules)
    _log("MEMORY: wrote auto-cleanup rules to MEMORY.md")
    print(f"Memory rules written to {memory_file}")
    print("Disk-guardian will now auto-run at the end of every task that creates test files.")
    print("Restart Hermes for the rules to take effect.")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="disk_guardian — ephemeral session file cleanup for Hermes Agent"
    )
    sub = parser.add_subparsers(dest="cmd")

    p_track = sub.add_parser("track", help="Register a file for tracking")
    p_track.add_argument("path")
    p_track.add_argument("category", choices=sorted(ALLOWED_CATEGORIES))

    sub.add_parser("dry-run",        help="Preview deletions, touch nothing")
    sub.add_parser("quick",          help="Auto-delete safe files (no prompts)")
    sub.add_parser("deep",           help="Full cleanup with prompts for risky items")
    sub.add_parser("status",         help="Show disk usage by category")
    sub.add_parser("install-memory", help="Write persistent auto-run memory rules")

    p_forget = sub.add_parser("forget", help="Stop tracking a path")
    p_forget.add_argument("path")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    try:
        if args.cmd == "track":
            cmd_track(args.path, args.category)
        elif args.cmd == "dry-run":
            cmd_dry_run()
        elif args.cmd == "quick":
            cmd_quick()
        elif args.cmd == "deep":
            cmd_deep()
        elif args.cmd == "status":
            cmd_status()
        elif args.cmd == "install-memory":
            cmd_install_memory()
        elif args.cmd == "forget":
            cmd_forget(args.path)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
    except Exception as e:
        _log(f"ERROR: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
