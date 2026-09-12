#!/usr/bin/env python3
"""Create a real-dir + file-level-symlink copy of a skill tree.

Hermes (Python 3.11) Path.rglob("SKILL.md") does not descend directory
symlinks. File-level links inside a real directory are found.

Usage:
  python3 link_skill_tree.py CANONICAL_DIR DEST_DIR
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SKIP_NAMES = {".DS_Store", ".git", "__pycache__"}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: link_skill_tree.py CANONICAL_DIR DEST_DIR", file=sys.stderr)
        return 2

    src = Path(sys.argv[1]).expanduser().resolve()
    dest = Path(sys.argv[2]).expanduser()

    if not src.is_dir():
        print(f"canonical is not a directory: {src}", file=sys.stderr)
        return 1
    if dest.exists() and dest.is_symlink():
        print(f"refusing: dest is a directory symlink: {dest}", file=sys.stderr)
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    dest = dest.resolve()

    linked = 0
    for root, dirs, files in os.walk(src, followlinks=False):
        root_p = Path(root)
        rel = root_p.relative_to(src)
        dirs[:] = [
            d
            for d in dirs
            if d not in SKIP_NAMES and not (root_p / d).is_symlink()
        ]
        dest_root = dest if rel == Path(".") else dest / rel
        dest_root.mkdir(parents=True, exist_ok=True)

        for name in files:
            if name in SKIP_NAMES:
                continue
            s = root_p / name
            if not s.is_file():
                continue
            resolved = s.resolve()
            d = dest_root / name
            target = Path(os.path.relpath(resolved, start=d.parent))
            if d.exists() or d.is_symlink():
                if d.is_symlink() and d.resolve() == resolved:
                    linked += 1
                    continue
                print(f"refusing to clobber: {d}", file=sys.stderr)
                return 1
            d.symlink_to(target)
            linked += 1

    print(f"linked {linked} files → {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
