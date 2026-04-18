#!/usr/bin/env python3
"""
Disk Guardian - Autonomous disk cleanup for Hermes Agent

Tracks files created by Hermes and safely removes stale ones.
"""

import argparse
import json
import os
import sys
import subprocess
import shlex
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import fcntl
import shutil


def get_hermes_home() -> Path:
    """Return the Hermes home directory (default: ~/.hermes)."""
    val = os.environ.get("HERMES_HOME", "").strip()
    return Path(val) if val else Path.home() / ".hermes"


def get_disk_guardian_dir() -> Path:
    """Return the disk-guardian directory."""
    return get_hermes_home() / "disk-guardian"


def get_tracked_file() -> Path:
    """Return the tracked.json file path."""
    return get_disk_guardian_dir() / "tracked.json"


def get_log_file() -> Path:
    """Return the cleanup.log file path."""
    return get_disk_guardian_dir() / "cleanup.log"


def is_wsl() -> bool:
    """Check if running in WSL."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def log_message(message: str) -> None:
    """Write a message to the cleanup log."""
    log_file = get_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


def load_tracked() -> List[Dict[str, Any]]:
    """Load tracked.json with error handling."""
    tracked_file = get_tracked_file()
    tracked_file.parent.mkdir(parents=True, exist_ok=True)

    if not tracked_file.exists():
        return []

    try:
        with open(tracked_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Try to restore from backup
        backup_file = tracked_file.with_suffix(".json.bak")
        if backup_file.exists():
            log_message("Tracking file corrupted, restoring from backup")
            with open(backup_file, "r") as f:
                return json.load(f)
        log_message("Tracking file corrupted, starting fresh")
        return []


def save_tracked(tracked: List[Dict[str, Any]]) -> None:
    """Save tracked.json with atomic write and backup."""
    tracked_file = get_tracked_file()
    tracked_file.parent.mkdir(parents=True, exist_ok=True)

    # Create backup
    if tracked_file.exists():
        backup_file = tracked_file.with_suffix(".json.bak")
        shutil.copy2(tracked_file, backup_file)

    # Atomic write
    temp_file = tracked_file.with_suffix(".json.tmp")
    with open(temp_file, "w") as f:
        json.dump(tracked, f, indent=2)
    temp_file.replace(tracked_file)


def track_path(path: str, category: str) -> None:
    """Add a path to tracking."""
    allowed_categories = ["temp", "test", "research", "download", "chrome-profile", "cron-output", "other"]
    if category not in allowed_categories:
        log_message(f"Unknown category '{category}', using 'other'")
        category = "other"

    path_obj = Path(path).resolve()
    if not path_obj.exists():
        log_message(f"Path {path} does not exist, skipping")
        return

    # Check if path is under Hermes home
    hermes_home = get_hermes_home().resolve()
    try:
        path_obj.relative_to(hermes_home)
    except ValueError:
        log_message(f"Path {path} is outside Hermes home, skipping")
        return

    # Get file size
    size = path_obj.stat().st_size if path_obj.is_file() else 0

    tracked = load_tracked()
    tracked.append({
        "path": str(path_obj),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "size": size
    })
    save_tracked(tracked)
    log_message(f"TRACKED: {path} ({category}, {size} bytes)")
    print(f"Tracked: {path} ({category}, {size} bytes)")


def scan_files() -> None:
    """Discover temp/test files by pattern and add to tracking."""
    hermes_home = get_hermes_home()
    tracked = load_tracked()
    tracked_paths = {item["path"] for item in tracked}

    # Scan for temp files
    temp_patterns = [
        hermes_home / "cache" / "hermes" / "*",
        Path("/tmp") / "hermes-*"
    ]

    for pattern in temp_patterns:
        for path in pattern.parent.glob(pattern.name):
            if str(path.resolve()) not in tracked_paths and path.exists():
                track_path(str(path), "temp")

    # Scan for test files
    test_patterns = [
        hermes_home / "test_*.py",
        hermes_home / "*.test.log",
        hermes_home / "tmp_*.json"
    ]

    for pattern in test_patterns:
        for path in hermes_home.glob(pattern.name):
            if str(path.resolve()) not in tracked_paths and path.exists():
                track_path(str(path), "test")

    print(f"Scan complete. Total tracked files: {len(load_tracked())}")


def dry_run() -> None:
    """Preview what would be deleted without touching anything."""
    hermes_home = get_hermes_home()
    tracked = load_tracked()

    # Categorize files by age
    now = datetime.now(timezone.utc)
    temp_files = []
    test_files = []
    research_folders = []
    large_files = []
    chrome_profiles = []

    for item in tracked:
        path = Path(item["path"])
        if not path.exists():
            continue

        timestamp = datetime.fromisoformat(item["timestamp"])
        age_days = (now - timestamp).days

        if item["category"] == "temp" and age_days > 7:
            temp_files.append(item)
        elif item["category"] == "test" and age_days > 3:
            test_files.append(item)
        elif item["category"] == "research" and age_days > 30:
            research_folders.append(item)
        elif item["size"] > 500 * 1024 * 1024:  # > 500 MB
            large_files.append(item)
        elif item["category"] == "chrome-profile" and age_days > 14:
            chrome_profiles.append(item)

    # Calculate sizes
    temp_size = sum(item["size"] for item in temp_files)
    test_size = sum(item["size"] for item in test_files)
    research_size = sum(item["size"] for item in research_folders)
    large_size = sum(item["size"] for item in large_files)
    chrome_size = sum(item["size"] for item in chrome_profiles)

    print("Dry-run results:")
    print(f"Would delete {len(temp_files)} temp files ({format_size(temp_size)})")
    print(f"Would delete {len(test_files)} test files ({format_size(test_size)})")
    print(f"Would prompt for {len(research_folders)} research folders ({format_size(research_size)})")
    print(f"Would prompt for {len(large_files)} large files ({format_size(large_size)})")
    print(f"Would prompt for {len(chrome_profiles)} chrome profiles ({format_size(chrome_size)})")

    total_size = temp_size + test_size + research_size + large_size + chrome_size
    print(f"\nTotal potential cleanup: {format_size(total_size)}")
    print("Run 'quick' for safe auto-cleanup")
    print("Run 'deep' for full cleanup with confirmation")


def quick_cleanup() -> None:
    """Safe fast clean, no confirmation needed."""
    hermes_home = get_hermes_home()
    tracked = load_tracked()
    now = datetime.now(timezone.utc)

    deleted_files = []
    total_freed = 0

    # Delete temp files > 7 days
    for item in tracked[:]:
        if item["category"] == "temp":
            path = Path(item["path"])
            if not path.exists():
                tracked.remove(item)
                continue

            timestamp = datetime.fromisoformat(item["timestamp"])
            age_days = (now - timestamp).days

            if age_days > 7:
                try:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
                    deleted_files.append(item)
                    total_freed += item["size"]
                    tracked.remove(item)
                    log_message(f"DELETED: {item['path']} (temp, {item['size']} bytes)")
                except Exception as e:
                    log_message(f"ERROR deleting {item['path']}: {e}")

    # Delete test files > 3 days
    for item in tracked[:]:
        if item["category"] == "test":
            path = Path(item["path"])
            if not path.exists():
                tracked.remove(item)
                continue

            timestamp = datetime.fromisoformat(item["timestamp"])
            age_days = (now - timestamp).days

            if age_days > 3:
                try:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)
                    deleted_files.append(item)
                    total_freed += item["size"]
                    tracked.remove(item)
                    log_message(f"DELETED: {item['path']} (test, {item['size']} bytes)")
                except Exception as e:
                    log_message(f"ERROR deleting {item['path']}: {e}")

    # Delete empty directories
    for root, dirs, files in os.walk(hermes_home, topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            try:
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    log_message(f"DELETED: {dir_path} (empty directory)")
            except Exception as e:
                pass  # Directory not empty or permission denied

    save_tracked(tracked)
    print(f"Deleted {len(deleted_files)} files, freed {format_size(total_freed)}")


def deep_cleanup() -> None:
    """Full scan with confirmation for risky items."""
    hermes_home = get_hermes_home()
    tracked = load_tracked()
    now = datetime.now(timezone.utc)

    # First, do quick cleanup
    print("Running quick cleanup first...")
    quick_cleanup()
    tracked = load_tracked()

    # Now handle risky items
    research_folders = []
    large_files = []
    chrome_profiles = []

    for item in tracked:
        path = Path(item["path"])
        if not path.exists():
            continue

        timestamp = datetime.fromisoformat(item["timestamp"])
        age_days = (now - timestamp).days

        if item["category"] == "research" and age_days > 30:
            research_folders.append(item)
        elif item["size"] > 500 * 1024 * 1024:  # > 500 MB
            large_files.append(item)
        elif item["category"] == "chrome-profile" and age_days > 14:
            chrome_profiles.append(item)

    # Keep last 10 research folders
    research_folders.sort(key=lambda x: x["timestamp"], reverse=True)
    old_research = research_folders[10:]

    total_freed = 0
    deleted_count = 0

    # Prompt for old research folders
    for item in old_research:
        path = Path(item["path"])
        response = input(f"Delete old research folder: {path}? [y/N] ")
        if response.lower() == "y":
            try:
                if path.exists():
                    shutil.rmtree(path)
                    total_freed += item["size"]
                    deleted_count += 1
                    tracked.remove(item)
                    log_message(f"DELETED: {item['path']} (research, {item['size']} bytes)")
                    print(f"Deleted: {path} ({format_size(item['size'])})")
            except Exception as e:
                log_message(f"ERROR deleting {item['path']}: {e}")
                print(f"Error deleting {path}: {e}")

    # Prompt for large files
    for item in large_files:
        path = Path(item["path"])
        print(f"\nLarge file: {path} ({format_size(item['size'])})")
        print("Category:", item["category"])
        response = input("Delete this file? [y/N] ")
        if response.lower() == "y":
            try:
                if path.exists():
                    path.unlink()
                    total_freed += item["size"]
                    deleted_count += 1
                    tracked.remove(item)
                    log_message(f"DELETED: {item['path']} (large file, {item['size']} bytes)")
                    print(f"Deleted: {path}")
            except Exception as e:
                log_message(f"ERROR deleting {item['path']}: {e}")
                print(f"Error deleting {path}: {e}")

    # Prompt for chrome profiles
    for item in chrome_profiles:
        path = Path(item["path"])
        print(f"\nChrome profile: {path} ({format_size(item['size'])})")
        response = input("Delete this chrome profile? [y/N] ")
        if response.lower() == "y":
            try:
                if path.exists():
                    shutil.rmtree(path)
                    total_freed += item["size"]
                    deleted_count += 1
                    tracked.remove(item)
                    log_message(f"DELETED: {item['path']} (chrome-profile, {item['size']} bytes)")
                    print(f"Deleted: {path}")
            except Exception as e:
                log_message(f"ERROR deleting {item['path']}: {e}")
                print(f"Error deleting {path}: {e}")

    save_tracked(tracked)
    print(f"\nSummary: Deleted {deleted_count} items, freed {format_size(total_freed)}")


def show_status() -> None:
    """Show disk usage breakdown by category + top 10 largest files."""
    tracked = load_tracked()

    # Calculate usage by category
    categories = {}
    for item in tracked:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "size": 0}
        categories[cat]["count"] += 1
        categories[cat]["size"] += item["size"]

    print("Disk usage by category:")
    print(f"{'Category':<20} {'Files':<10} {'Size':<15}")
    print("-" * 45)
    for cat, data in sorted(categories.items(), key=lambda x: x[1]["size"], reverse=True):
        print(f"{cat:<20} {data['count']:<10} {format_size(data['size']):<15}")

    # Find top 10 largest files
    all_files = [(item["path"], item["size"], item["category"]) for item in tracked if Path(item["path"]).exists()]
    all_files.sort(key=lambda x: x[1], reverse=True)
    top_10 = all_files[:10]

    print("\nTop 10 largest files:")
    for i, (path, size, cat) in enumerate(top_10, 1):
        print(f"{i}. {path} ({format_size(size)}, {cat})")


def forget_path(path: str) -> None:
    """Remove a path from tracking permanently."""
    path_obj = Path(path).resolve()
    tracked = load_tracked()

    original_count = len(tracked)
    tracked = [item for item in tracked if Path(item["path"]).resolve() != path_obj]
    removed = original_count - len(tracked)

    if removed > 0:
        save_tracked(tracked)
        log_message(f"FORGOT: {path} ({removed} entries)")
        print(f"Removed {removed} entries from tracking")
    else:
        print(f"Path {path} not found in tracking")


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def main():
    parser = argparse.ArgumentParser(description="Disk Guardian - Autonomous disk cleanup for Hermes Agent")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Track command
    track_parser = subparsers.add_parser("track", help="Track a path")
    track_parser.add_argument("path", help="Path to track")
    track_parser.add_argument("category", help="Category (temp, test, research, download, chrome-profile, cron-output, other)")

    # Scan command
    subparsers.add_parser("scan", help="Discover temp/test files by pattern")

    # Dry-run command
    subparsers.add_parser("dry-run", help="Preview what would be deleted")

    # Quick command
    subparsers.add_parser("quick", help="Safe fast clean")

    # Deep command
    subparsers.add_parser("deep", help="Full scan with confirmation")

    # Status command
    subparsers.add_parser("status", help="Show disk usage breakdown")

    # Forget command
    forget_parser = subparsers.add_parser("forget", help="Remove a path from tracking")
    forget_parser.add_argument("path", help="Path to forget")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "track":
            track_path(args.path, args.category)
        elif args.command == "scan":
            scan_files()
        elif args.command == "dry-run":
            dry_run()
        elif args.command == "quick":
            quick_cleanup()
        elif args.command == "deep":
            deep_cleanup()
        elif args.command == "status":
            show_status()
        elif args.command == "forget":
            forget_path(args.path)
    except Exception as e:
        log_message(f"ERROR: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
