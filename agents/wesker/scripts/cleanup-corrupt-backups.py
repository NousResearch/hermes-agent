#!/usr/bin/env python3
"""
Cleanup corrupt kanban DB backups older than 7 days.
Searches in all kanban boards for *.bak and *.corrupt.* files.
"""
import os
import time
from pathlib import Path

# Base directory for kanban boards
BASE_DIR = Path("/home/kensei/.hermes/kanban/boards")
# Also check the global kanban directory for any stray backups? Not necessary but safe.
# We'll also check the logs? The task mentions logs but they are not corrupt backups.

# Patterns to match
PATTERNS = ["*.bak", "*.corrupt.*"]

# Age threshold in seconds (7 days)
MAX_AGE = 7 * 24 * 60 * 60

def main():
    now = time.time()
    deleted_count = 0
    freed_bytes = 0

    if not BASE_DIR.is_dir():
        print(f"Base directory {BASE_DIR} does not exist.")
        return

    for board_dir in BASE_DIR.iterdir():
        if not board_dir.is_dir():
            continue
        for pattern in PATTERNS:
            for file_path in board_dir.rglob(pattern):
                try:
                    # Check if it's a file
                    if file_path.is_file():
                        # Check age
                        file_age = now - file_path.stat().st_mtime
                        if file_age > MAX_AGE:
                            size = file_path.stat().st_size
                            file_path.unlink()
                            deleted_count += 1
                            freed_bytes += size
                            print(f"Deleted: {file_path} (size: {size} bytes, age: {file_age/86400:.1f} days)")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    print(f"Cleanup complete. Deleted {deleted_count} files, freed {freed_bytes} bytes ({freed_bytes/1024/1024:.2f} MB).")

if __name__ == "__main__":
    main()