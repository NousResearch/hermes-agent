#!/usr/bin/env python3
"""Fail when a tracked text file is stored with Windows (CRLF) line endings.

``.gitattributes`` asks git to store text with LF, but only a client that honours it does so:
a file uploaded through GitHub's web editor or created through the API lands exactly as sent.
One such blob then reads as modified in every fresh checkout (git renormalizes it on the way
out), and any CI step that demands a clean tree afterwards fails on every PR, whatever the PR
touched. ``plugin-catalog/board-insights.yaml`` did exactly that to every catalog PR.

The check reads git's own view of the index (``git ls-files --eol``): a path whose staged blob is
``crlf`` or ``mixed`` fails, unless its attributes mark it ``-text`` (binary) or ask for CRLF
(``eol=crlf``, e.g. ``*.ps1``, whose index copy is still LF but whose checkout is CRLF).

    python scripts/ci/check_line_endings.py          # check the index
    python scripts/ci/check_line_endings.py --fix    # rewrite offenders with LF and stage them

Exit codes: 0 = clean (or fixed), 1 = offenders found, 2 = git error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_BAD_INDEX = {"i/crlf", "i/mixed"}


def offenders(root: Path) -> list[str]:
    proc = subprocess.run(["git", "ls-files", "--eol", "-z"], cwd=root, capture_output=True,
                          stdin=subprocess.DEVNULL, timeout=120, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace").strip())
    found = []
    for record in proc.stdout.split(b"\0"):
        if not record:
            continue
        info, _, path = record.decode("utf-8", errors="surrogateescape").partition("\t")
        index_eol, _worktree_eol, attr = (info.split() + ["", "", ""])[:3]
        if index_eol in _BAD_INDEX and "-text" not in attr and "eol=crlf" not in attr:
            found.append(path)
    return found


def fix(root: Path, paths: list[str]) -> None:
    for rel in paths:
        target = root / rel
        target.write_bytes(target.read_bytes().replace(b"\r\n", b"\n"))
    subprocess.run(["git", "add", "--", *paths], cwd=root, stdin=subprocess.DEVNULL, timeout=120,
                   check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail when a tracked text file is stored with CRLF.")
    parser.add_argument("--fix", action="store_true", help="rewrite offenders with LF and stage them")
    parser.add_argument("--root", type=Path, default=REPO, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        found = offenders(args.root)
    except (RuntimeError, OSError) as exc:
        print(f"::error::git ls-files --eol failed: {exc}")
        return 2
    if not found:
        print("::notice::no tracked text file is stored with CRLF line endings.")
        return 0
    if args.fix:
        fix(args.root, found)
        print(f"Rewrote {len(found)} file(s) with LF and staged them; commit the result:")
        print("\n".join(f"  {p}" for p in found))
        return 0
    print(f"::error::{len(found)} tracked text file(s) are stored with CRLF line endings.")
    print("Every fresh checkout shows them as modified, which fails clean-tree CI steps on every PR.")
    print("Fix: python scripts/ci/check_line_endings.py --fix, then commit.")
    print("\n".join(f"  {p}" for p in found))
    return 1


if __name__ == "__main__":
    sys.exit(main())
