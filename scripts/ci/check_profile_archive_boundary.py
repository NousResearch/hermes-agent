#!/usr/bin/env python3
"""Reject profile export archives before publication.

``.gitignore`` and ``.dockerignore`` are useful first-line filters, but both
can be bypassed (for example with ``git add -f`` or a non-standard build
context).  This check is the blocking, executable policy at the CI and image
publication boundaries.  It intentionally checks the filesystem rather than
Git's index so a generated archive cannot enter a build after checkout.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_PROFILE_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz")


def find_forbidden_profile_archives(root: Path) -> list[Path]:
    """Return profile archive paths anywhere in the checkout."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"repository root is not a directory: {root}")

    offenders: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {".git", ".venv", "venv", "node_modules", "__pycache__"}
        ]
        for name in (*dirnames, *filenames):
            if name.casefold().endswith(_PROFILE_ARCHIVE_SUFFIXES):
                offenders.append((Path(directory) / name).relative_to(root))

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
