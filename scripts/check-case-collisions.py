#!/usr/bin/env python3
"""
Blocking check for tracked files whose paths collide when case is ignored.

Linux is case-sensitive; Windows and macOS (default) are not. Two tracked
paths that differ only by case — ``README.md`` and ``readme.md``, or
``src/Foo.py`` and ``SRC/foo.py`` — coexist happily in a Linux checkout and
silently break every clone on a case-insensitive host: the filesystem can
hold only one of them, so checkout either refuses or whichever file is
written last wins and clobbers the other. Git itself won't stop the pair
from landing — it only warns at checkout time, on a case-insensitive FS,
for whichever client happens to do the checkout, and the collision is
invisible on Linux. This check is the enforcement point: scan the index,
fail the build, name the offenders.

Usage:
    # Check the checkout this script lives in (CI + the common local case)
    python scripts/check-case-collisions.py

    # Check an arbitrary git checkout (tests, other worktrees)
    python scripts/check-case-collisions.py /path/to/other/repo

Exit status:
    0 — no case-colliding tracked paths
    1 — at least one collision group (paths printed to stdout)
    2 — not in a git repository / git failed

Comparison key: the casefolded FULL path (``str.casefold``), not the
basename — on a case-insensitive filesystem the entire path is
case-insensitive, so ``dir/Foo.txt`` and ``DIR/foo.txt`` collide just like
same-directory pairs. ``casefold`` (not ``lower``) is used because it
matches how the OSes fold case for non-ASCII text (straße vs strasse,
sigma variants); a pair it flags is a genuine collision on macOS/Windows
even when Linux disagrees.

Deliberately out of scope: Unicode NFC/NFD normalization collisions (macOS
stores NFD, Linux NFC). git already handles those at checkout via
``core.precomposeunicode``; this check is strictly about case.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=str(REPO_ROOT),
        help="git checkout to scan (default: the repo this script lives in)",
    )
    args = parser.parse_args()

    try:
        os.chdir(args.root)
    except OSError as exc:
        print(f"::error::cannot enter {args.root}: {exc}")
        return 2

    proc = subprocess.run(["git", "ls-files", "-z"], capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", errors="replace").strip()
        print(f"::error::git ls-files failed in {args.root}: {msg}")
        return 2

    paths = [
        p.decode("utf-8", errors="surrogateescape")
        for p in proc.stdout.split(b"\0")
        if p
    ]

    by_casefold: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        by_casefold[path.casefold()].append(path)

    collisions = {key: group for key, group in by_casefold.items() if len(group) > 1}

    if not collisions:
        print(f"::notice::{len(paths)} tracked files, no case-colliding paths.")
        return 0

    print(
        f"::error::Found {len(collisions)} case-collision group(s) among "
        f"{len(paths)} tracked files."
    )
    print(
        "Paths that differ only by case are ONE file on Windows/macOS but "
        "several on Linux - the pair breaks every clone on a case-insensitive "
        "host. Rename one member of each group so the paths differ beyond case."
    )
    print()
    for key, group in sorted(collisions.items()):
        for path in sorted(group):
            print(f"  {path}")
        print()
    print(
        "Fix: `git mv` one path in each group to a name that doesn't collide. "
        "On Windows/macOS you may need two steps (`git mv a.txt tmp && git mv "
        "tmp A.txt`) because the filesystem can't hold both spellings at once."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
