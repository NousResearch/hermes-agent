#!/usr/bin/env python3
"""Reject profile export archives before publication.

``.gitignore`` and ``.dockerignore`` are useful first-line filters, but both
can be bypassed (for example with ``git add -f`` or a non-standard build
context).  This check is the blocking, executable policy at the CI and image
publication boundaries.  It intentionally checks the filesystem rather than
Git's index so a generated archive cannot enter a build after checkout.

What it deliberately does NOT judge: an archive inside a gitignored
**directory** (``.cache/``, ``.treehouse/``, ``apps/desktop/release/``).  A
whole ignored tree is another tool's local state - the desktop packager unpacks
its own ``*.tar.gz`` build inputs under ``.cache/`` - not a profile export
sitting in the checkout.  Nothing else is skipped: a gitignored **file** is
still judged, because ``.gitignore`` ignores ``*.tar.gz`` repo-wide and
honouring that would switch this check off entirely, and a tracked archive is
always judged, because ``git add -f`` is the bypass this check exists for.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# `scripts/` is a namespace package, so a guard run directly has only `scripts/ci/` on the path;
# the checkout root has to be added to reach the helpers shared with the other disk-walking guards.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.ci.gitignored import ignored_paths

_PROFILE_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz")

# Never descended: version-control internals, dependency trees and byte-compiled output hold no
# first-party archive, and walking them is pure cost.
_UNSEARCHED_DIR_NAMES = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__"})


def find_forbidden_profile_archives(root: Path) -> list[Path]:
    """Return profile archive paths anywhere in the checkout's own content."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"repository root is not a directory: {root}")

    ignored = ignored_paths(root)
    offenders: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(directory)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in _UNSEARCHED_DIR_NAMES
            # A gitignored subtree is local state. Pruned by directory because git reports an
            # embedded repository as one bare directory entry and never as its files.
            and (here / name) not in ignored.directories
        ]
        for name in (*dirnames, *filenames):
            if name.casefold().endswith(_PROFILE_ARCHIVE_SUFFIXES):
                offenders.append((here / name).relative_to(root))

    return sorted(offenders, key=lambda path: path.as_posix().casefold())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject profile export archives in the checkout."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root to inspect (default: current directory)",
    )
    args = parser.parse_args(argv)

    try:
        offenders = find_forbidden_profile_archives(args.root)
    except ValueError as exc:
        parser.error(str(exc))

    if not offenders:
        print("No profile export archives detected in the checkout.")
        return 0

    print(
        "::error::profile export archives are forbidden "
        "in source and Docker build contexts"
    )
    for path in offenders:
        print(f"  {path.as_posix()}")
    print(
        "Move the archive outside the checkout or pass an explicit external "
        "output path to the profile export command."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
