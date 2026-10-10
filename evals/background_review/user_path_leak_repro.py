#!/usr/bin/env python3
"""Reproduce / verify the pytest -> ``HKCU\\Environment\\Path`` leak.

Runs the given pytest targets and asserts the operator's *persisted* User PATH
is byte-identical -- raw value plus registry type, no ``%VAR%`` expansion --
before and after.

    python evals/background_review/user_path_leak_repro.py [pytest args...]

Exit codes: 0 = the registry value did not move, 1 = it moved (leak), 2 = the
key could not be read. Re-runnable; it never writes the registry itself.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = [
    "tests/hermes_cli/test_windows_user_path_isolation.py",
    "tests/hermes_cli/test_post_update_expose_cli.py",
]


def _snapshot() -> tuple[str, int] | None:
    if sys.platform != "win32":
        return None
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ
    ) as key:
        try:
            value, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            return None
        return str(value), int(kind)


def _describe(snap: tuple[str, int] | None) -> str:
    if snap is None:
        return "NO VALUE (or non-Windows host)"
    value, kind = snap
    parts = [p for p in value.split(";") if p]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"type={kind} entries={len(parts)} chars={len(value)} sha256={digest[:16]}"


def _entries(snap: tuple[str, int] | None) -> list[str]:
    if snap is None:
        return []
    return [p for p in snap[0].split(";") if p]


def main(argv: list[str]) -> int:
    targets = argv or DEFAULT_TARGETS
    before = _snapshot()
    if before is None:
        print("user_path_leak_repro: HKCU\\Environment\\Path is not readable here")
        return 2
    print("before:", _describe(before))
    print("running:", " ".join(targets))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *targets, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT,
        env=dict(os.environ),
        # Generous, but bounded: a hung suite must not hang the repro forever.
        timeout=3600,
    )
    after = _snapshot()
    print("after: ", _describe(after))
    print(f"pytest exit: {proc.returncode}")

    if after == before:
        print("OK: the operator's persisted User PATH is byte-identical")
        return 0

    new = [p for p in _entries(after) if p not in set(_entries(before))]
    gone = [p for p in _entries(before) if p not in set(_entries(after))]
    print("LEAK: HKCU\\Environment\\Path moved during the test run")
    for p in new[:10]:
        print("  +", p)
    for p in gone[:10]:
        print("  -", p)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
