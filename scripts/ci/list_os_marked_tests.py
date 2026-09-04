#!/usr/bin/env python3
"""List the test files that carry a platforms() gate for a given platform.

Used by the marked-OS lane of ``.github/workflows/tests.yml`` to scope what
the macOS lane imports.

Why scope at all, when ``pytest -m platforms`` already selects correctly?
Because ``-m`` filters AFTER collection, and collection IMPORTS every test
module under ``tests/``. On the Linux lane that is fine (it runs them all
anyway), but on the macOS lane it would drag ~900 unrelated modules through
import on a host they were never expected to import on — one unrelated
ImportError would fail a job whose actual subject passed. Narrowing the
paths keeps each lane's failure signal about its own tests.

``-m platforms`` (plus the conftest's per-test host skips) remains the
authoritative selector: this script only decides which files get imported,
never which tests run. Over-selecting here is harmless (the skips drop the
extras); the failure mode to care about is UNDER-selecting, which is why
the workflow fails the job when zero tests end up selected.

A file matches when a quoted ``platforms("...")`` spec names the platform —
including negated (``"not macos"``) and any-of lists. The match is anchored
inside the string literal so bare identifiers (a variable named ``windows``)
don't produce false positives.

Usage:
    python scripts/ci/list_os_marked_tests.py macos [tests_root]

Prints one path per line (POSIX separators, repo-relative), sorted. Exits
non-zero when no file matches (a renamed spec or broken selection would
otherwise report a green lane that ran nothing).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_VALID_PLATFORMS = ("linux", "macos", "windows")


def find_marked_files(platform: str, root: Path) -> list[Path]:
    """Return every ``test_*.py`` under *root* gating on *platform*."""
    pattern = re.compile(
        rf'platforms\(\s*[^)]*?"[^")]*\b{re.escape(platform)}\b[^")]*"'
    )
    hits: list[Path] = []
    for path in sorted(root.rglob("test_*.py")):
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        if pattern.search(text):
            hits.append(path)
    return hits


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    platform = argv[1]
    if platform not in _VALID_PLATFORMS:
        print(
            f"unknown platform {platform!r}; valid: {', '.join(_VALID_PLATFORMS)}",
            file=sys.stderr,
        )
        return 2
    tests_root = Path(argv[2]) if len(argv) > 2 else Path("tests")
    if not tests_root.is_dir():
        print(f"no such directory: {tests_root}", file=sys.stderr)
        return 2
    hits = find_marked_files(platform, tests_root)
    for path in hits:
        print(path.as_posix())
    if not hits:
        print(
            f"no test files gate on {platform!r} under {tests_root} — "
            "either the spec vocabulary changed or selection is broken",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
