#!/usr/bin/env python3
"""List the test files that carry a given OS marker.

Used by ``.github/workflows/tests-os.yml`` to scope what the macOS and
Windows lanes import.

Why scope at all, when ``pytest -m macos_only`` already selects correctly?
Because ``-m`` filters AFTER collection, and collection IMPORTS every test
module under ``tests/``. On the Linux lane that is fine (it runs them all
anyway), but on the macOS/Windows lanes it would drag ~900 unrelated modules
through import on a host they were never expected to import on — one
unrelated ImportError would fail a job whose actual subject passed. Narrowing
the paths keeps each lane's failure signal about its own tests.

``-m`` is still passed by the workflow and remains the authoritative
selector: this script only decides which files get imported, never which
tests run. Over-selecting here is harmless (``-m`` drops the extras); the
failure mode to care about is UNDER-selecting, which is why the workflow
fails the job when zero tests end up selected.

Usage:
    python scripts/ci/list_os_marked_tests.py macos_only [tests_root]

Prints one path per line (POSIX separators, repo-relative), sorted.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_VALID_MARKERS = ("linux_only", "macos_only", "windows_only")


def find_marked_files(marker: str, root: Path) -> list[Path]:
    """Return every ``test_*.py`` under *root* that references *marker*.

    Matches the marker as a whole word so ``macos_only`` doesn't pick up a
    hypothetical ``macos_only_extra``. Catches both the decorator form
    (``@pytest.mark.macos_only``, on a function or a class) and the
    module-level ``pytestmark`` form.
    """
    pattern = re.compile(rf"\b{re.escape(marker)}\b")
    hits: list[Path] = []
    for path in sorted(root.rglob("test_*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if pattern.search(text):
            hits.append(path)
    return hits


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    marker = argv[1]
    if marker not in _VALID_MARKERS:
        print(
            f"error: unknown marker {marker!r} (expected one of "
            f"{', '.join(_VALID_MARKERS)})",
            file=sys.stderr,
        )
        return 2

    repo_root = Path(__file__).resolve().parents[2]
    root = Path(argv[2]) if len(argv) > 2 else repo_root / "tests"
    if not root.exists():
        print(f"error: no such directory: {root}", file=sys.stderr)
        return 2

    files = find_marked_files(marker, root)
    if not files:
        print(
            f"error: no test file references @pytest.mark.{marker} — the marker "
            "was probably renamed or dropped. Refusing to emit an empty list, "
            "which would let the OS lane pass without running anything.",
            file=sys.stderr,
        )
        return 1

    lines: list[str] = []
    for path in files:
        # POSIX separators so the output is safe to paste into a bash
        # command line on the Windows runner (Git Bash accepts them).
        #
        # Relative to the repo root when the path is inside it (the CI case —
        # pytest is invoked from the repo root). A root outside the repo is a
        # test/manual invocation; emit it as-is rather than raising, since
        # ``relative_to`` refuses non-descendant paths.
        try:
            rel = path.resolve().relative_to(repo_root)
        except ValueError:
            lines.append(path.as_posix())
        else:
            lines.append(rel.as_posix())

    # Write bytes with explicit LF rather than print(), which on Windows
    # translates "\n" to "\r\n" in text mode. The consumer reads this list with
    # ``$(cat ...)`` in bash, and word splitting uses IFS (space/tab/newline) —
    # a CR is NOT a separator, so it stays glued to each path and pytest then
    # fails with "file or directory not found: tests/...py" for a path that
    # looks correct in the log because the CR is invisible. Emitting bytes makes
    # the output identical on every host instead of depending on the platform's
    # newline translation.
    sys.stdout.buffer.write(b"".join(line.encode("utf-8") + b"\n" for line in lines))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
