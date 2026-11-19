#!/usr/bin/env python3
"""Fail when a tracked Dockerfile pipes a download straight into an extractor or interpreter.

``curl URL | tar -x`` (or ``| sh``) turns whatever bytes the server returns into files or code
with no point at which they can be checked. Every image the repo builds fetches to a file,
verifies it against a committed digest (``sha256sum -c``), and only then extracts it; the root
``Dockerfile`` does this for SQLite and s6-overlay. ``docker/sandbox-desktop.Dockerfile`` once
streamed the cua-driver release asset into ``tar`` and published the result as the default
terminal sandbox (#126942). This guard keeps that shape out of every tracked Dockerfile.

Lines are joined the way BuildKit parses them (backslash continuations, dropping comment lines
and blank lines inside a continuation), so a ``curl ... \\`` / ``| tar`` split over several
physical lines is still caught; ``| /bin/tar`` counts the same as ``| tar``.

    python scripts/ci/check_dockerfile_downloads.py [--root DIR]

Exit codes: 0 = clean, 1 = a streamed download found (file:line printed), 2 = git error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_DOCKERFILES = ("Dockerfile", "*.Dockerfile", "**/Dockerfile", "**/*.Dockerfile")
_STREAMED = re.compile(
    r"\b(?:curl|wget)\b[^|;&]*\|\s*(?:sudo\s+)?(?:\S*/)?"
    r"(?:tar|bsdtar|unzip|gunzip|gzip|xz|zstd|bzip2|sh|bash|dash|zsh|python3?|perl|node)\b"
)


def _tracked_dockerfiles(root: Path) -> list[str]:
    proc = subprocess.run(["git", "ls-files", "-z", "--", *_DOCKERFILES], cwd=root,
                          capture_output=True, stdin=subprocess.DEVNULL, timeout=120, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace").strip())
    return sorted({p for p in proc.stdout.decode("utf-8", errors="surrogateescape").split("\0") if p})


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """(first physical line number, joined text) for each continuation-joined line."""
    joined: list[tuple[int, str]] = []
    start, parts = 0, []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("#") or (parts and not line.strip()):
            continue  # BuildKit drops comment lines, and blank lines inside a continuation
        if not parts:
            start = lineno
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            parts.append(stripped[:-1])
            continue
        parts.append(line)
        joined.append((start, " ".join(parts)))
        parts = []
    if parts:
        joined.append((start, " ".join(parts)))
    return joined


def offenders(root: Path) -> list[str]:
    found = []
    for rel in _tracked_dockerfiles(root):
        path = root / rel
        if not path.is_file():
            continue
        for lineno, line in _logical_lines(path.read_text(encoding="utf-8-sig", errors="replace")):
            if _STREAMED.search(line):
                found.append(f"{rel}:{lineno}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject Dockerfile downloads piped into tar/sh.")
    parser.add_argument("--root", type=Path, default=REPO, help="repository root (default: this repo)")
    args = parser.parse_args(argv)
    try:
        found = offenders(args.root)
    except RuntimeError as exc:
        print(f"check_dockerfile_downloads: git failed: {exc}", file=sys.stderr)
        return 2
    if not found:
        print("No Dockerfile pipes a download into an extractor or interpreter.")
        return 0
    print("Downloads piped straight into an extractor or interpreter (no integrity check):")
    for where in found:
        print(f"  {where}")
    print("Fetch to a file, verify it against a committed sha256 (sha256sum -c), then extract.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
