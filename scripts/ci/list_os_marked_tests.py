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

A file matches when a quoted spec inside a ``mark.platforms(...)`` call
COVERS the platform, resolved the way the conftest gate resolves it:
``"posix"`` covers linux and macOS, ``"any"`` covers every lane, and a
negated spec (``"not macos"``) lists on every lane since it admits all the
others (and the named one keeps its import, harmlessly). Matching the
literal word only would drop every ``platforms("posix")`` file from the
macOS lane. The match is anchored inside the string literal so bare
identifiers (a variable named ``windows``) don't produce false positives.

Usage:
    python scripts/ci/list_os_marked_tests.py macos [tests_root]

Prints one path per line (POSIX separators, repo-relative), sorted. Exits
non-zero when no file matches (a renamed spec or broken selection would
otherwise report a green lane that ran nothing).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_VALID_PLATFORMS = ("linux", "macos", "windows")

# Mirrors tests/_fixtures/platform_gating.py::_PLATFORM_ALIASES, keyed by lane name rather than
# sys.platform value: this side never runs on the host it is asking about.
_SPEC_HOSTS = {
    "linux": frozenset({"linux"}),
    "macos": frozenset({"macos"}),
    "windows": frozenset({"windows"}),
    "posix": frozenset({"linux", "macos"}),
    "any": frozenset(_VALID_PLATFORMS),
}
_PLATFORMS_CALL = re.compile(r"mark\.platforms\(([^)]*)\)")
_QUOTED = re.compile(r"""["']([^"']*)["']""")


def gated_specs(text: str) -> set[str]:
    """Every quoted spec inside the ``mark.platforms(...)`` calls of a test file.

    Keyword values (``arch="arm64"``) come along; they resolve to no host below,
    so they can never select a lane.
    """
    return {
        spec.strip().lower()
        for call in _PLATFORMS_CALL.finditer(text)
        for spec in _QUOTED.findall(call.group(1))
    }


def spec_hosts(spec: str) -> frozenset[str]:
    """Lane names a spec admits; ``not X`` admits every lane but X's; unknown → none."""
    leaf = spec.removeprefix("not ").strip()
    hosts = _SPEC_HOSTS.get(leaf, frozenset())
    return _SPEC_HOSTS["any"] - hosts if spec.startswith("not ") else hosts


def file_gates_on(text: str, platform: str) -> bool:
    """True when some spec covers *platform* — or is negated (see module docstring)."""
    return any(
        platform in spec_hosts(spec) or spec.startswith("not ") for spec in gated_specs(text)
    )


def find_marked_files(platform: str, root: Path) -> list[Path]:
    """Return every ``test_*.py`` under *root* gating on *platform*."""
    hits: list[Path] = []
    # os.walk, not Path.rglob: rglob raises FileNotFoundError when a directory
    # (a sibling job's __pycache__) vanishes mid-scan; os.walk skips it.
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if not (fname.startswith("test_") and fname.endswith(".py")):
                continue
            path = Path(dirpath) / fname
            try:
                text = path.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            if file_gates_on(text, platform):
                hits.append(path)
    return sorted(hits)


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
