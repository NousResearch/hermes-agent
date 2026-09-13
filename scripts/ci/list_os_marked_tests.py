#!/usr/bin/env python3
"""List the test files that carry a platforms() gate for a given platform.

Used by the marked-OS lane of ``.github/workflows/tests-os.yml`` to scope
what the macOS/Windows lanes import: ``-m platforms`` filters AFTER
collection, and collection imports every test module under ``tests/``, so
narrowing the imported paths keeps each lane's failure signal about its
own tests instead of failing on ~900 unrelated imports.

This script only decides which files are IMPORTED, never which tests run
(``-m platforms`` plus the conftest's per-test host skips stay
authoritative). Over-selection is harmless — the skips drop the extras —
so whenever a file's gating cannot be evaluated statically (non-literal
spec, unknown leaf) OR the file itself cannot be read or parsed, the file
is conservatively listed; the lane, not this selector, then reports the
problem. UNDER-selection, which silently drops coverage, is the failure
mode this tool exists to prevent.

Spec recognition is AST-based over the conftest gate's vocabulary:
decorator and module-level ``pytestmark`` forms, quoted string specs,
negation (``"not macos"``), the grouping specs (``"posix"`` = linux+macos
hosts, ``"any"`` = all), and ``arch=``/``arch_negate=`` keyword filters.
A file is listed for platform P when any spec could run a test on a P
host.

Usage:
    python scripts/ci/list_os_marked_tests.py macos [tests_root]

Prints one path per line (POSIX separators, repo-relative), sorted. Exits
non-zero when no file matches — a renamed spec or broken selection would
otherwise report a green lane that ran nothing.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_VALID_PLATFORMS = ("linux", "macos", "windows")

# Host sets each positive leaf spec can run on (mirrors the conftest's
# _PLATFORM_ALIASES evaluation, expressed over lane names).
_SPEC_HOSTS = {
    "linux": {"linux"},
    "macos": {"macos"},
    "windows": {"windows"},
    "posix": {"linux", "macos"},
    "any": {"linux", "macos", "windows"},
}


def _platforms_call_matches_p(call: ast.Call, platform: str) -> bool | None:
    """Could this platforms(...) call gate a test onto a *platform* host?

    Returns True/False, or None when the call cannot be evaluated
    statically (non-literal spec) — the caller then over-selects.
    """
    literal_specs: list[str] = []
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            literal_specs.append(arg.value)
        else:
            return None  # non-literal spec: cannot evaluate → over-select
    if not literal_specs:
        return None  # no specs at all: conftest treats it as matches-everything
    matched = False
    for spec in literal_specs:
        text = spec.strip().lower()
        negate = text.startswith("not ")
        leaf = text[4:].strip() if negate else text
        if negate:
            # Negation excludes hosts, so the gate can pass on any host
            # outside the leaf's set — including this lane's.
            return True
        if leaf not in _SPEC_HOSTS:
            return None  # unknown leaf: conftest hard-errors; over-select
        if platform in _SPEC_HOSTS[leaf]:
            matched = True
    return matched


def _file_matches_platform(tree: ast.AST, platform: str) -> bool | None:
    """True/False/None(unevaluable) from one parsed test module."""
    verdict: bool | None = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_platforms_call = (
            isinstance(func, ast.Name) and func.id == "platforms"
        ) or (isinstance(func, ast.Attribute) and func.attr == "platforms")
        if not is_platforms_call:
            continue
        # arch=/arch_negate= narrow the per-test skip, not file selection —
        # the lane's machine isn't statically known, so ignore them.
        hit = _platforms_call_matches_p(node, platform)
        if hit is None:
            verdict = None
        elif hit:
            return True
    return verdict


def find_marked_files(platform: str, root: Path) -> list[Path]:
    """Return every ``test_*.py`` under *root* gating on *platform*."""
    hits: list[Path] = []
    for path in sorted(root.rglob("test_*.py")):
        try:
            source = path.read_bytes()
        except OSError as exc:
            # Fail closed: an unreadable file is conservatively SELECTED so
            # the lane fails visibly there instead of silently dropping
            # whatever coverage the file carries.
            print(f"warning: unreadable {path}: {exc}", file=sys.stderr)
            hits.append(path)
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            print(f"warning: unparseable {path}: {exc}", file=sys.stderr)
            hits.append(path)
            continue
        match = _file_matches_platform(tree, platform)
        if match is None or match:
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
