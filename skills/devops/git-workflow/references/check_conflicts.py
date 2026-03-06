#!/usr/bin/env python3
"""
check_conflicts.py — Scan the working tree for unresolved Git conflict markers.

Usage:
    python scripts/check_conflicts.py [path]

Returns exit code 0 if no conflict markers found, 1 otherwise.
The agent runs this after resolving conflicts to verify nothing was missed.
"""

import sys
import os

CONFLICT_MARKERS = ["<<<<<<<", "=======", ">>>>>>>"]

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".whl", ".pyc",
    ".bin", ".exe", ".so", ".dylib",
}

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}


def scan(root: str) -> list[tuple[str, int, str]]:
    """Return list of (filepath, line_number, line_content) for each conflict marker found."""
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune directories we never want to scan
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in SKIP_EXTENSIONS:
                continue

            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, start=1):
                        for marker in CONFLICT_MARKERS:
                            if line.startswith(marker):
                                hits.append((filepath, lineno, line.rstrip()))
                                break
            except (OSError, PermissionError):
                continue
    return hits


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    root = os.path.abspath(root)

    hits = scan(root)

    if not hits:
        print("✓ No unresolved conflict markers found.")
        sys.exit(0)

    print(f"✗ Found {len(hits)} unresolved conflict marker(s):\n")
    for filepath, lineno, content in hits:
        rel = os.path.relpath(filepath, root)
        print(f"  {rel}:{lineno}  →  {content}")

    print(
        "\nResolve each conflict, remove ALL <<<<<<<, =======, >>>>>>> markers, "
        "then run this script again."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
