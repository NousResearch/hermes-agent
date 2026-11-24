"""Shared helpers for the facade's contract tests. Stdlib `re` is the reference here (tests only)."""
import os
import re as std
import subprocess
import sys
from pathlib import Path

import hermes_re as hre

REPO_ROOT = Path(__file__).resolve().parents[2]

FIELDS = "group(0), span(), groups(), groupdict(), lastindex, lastgroup"


def view(m):
    """Everything a caller can read from a match, as a comparable value."""
    if m is None:
        return None
    return (m.group(0), m.span(), m.groups(), m.groupdict(), m.lastindex, m.lastgroup)


def outcome(fn):
    """The value of ``fn()``, or the NAME of the exception class it raises (``error`` for both engines)."""
    try:
        return fn()
    except Exception as exc:
        return "raises " + type(exc).__name__.replace("PatternError", "error")


def same_everywhere(pattern, flags, subjects):
    """Assert that every operation gives stdlib's result for every subject. Returns the number of checks."""
    mine, ref = hre.compile(pattern, flags), std.compile(pattern, flags)
    template = rb"<\g<0>>" if isinstance(pattern, bytes) else r"<\g<0>>"
    operations = (
        ("search", lambda p, s: view(p.search(s))),
        ("match", lambda p, s: view(p.match(s))),
        ("fullmatch", lambda p, s: view(p.fullmatch(s))),
        ("findall", lambda p, s: p.findall(s)),
        ("finditer", lambda p, s: [view(m) for m in p.finditer(s)]),
        ("sub", lambda p, s: p.sub(template, s)),
        ("subn", lambda p, s: p.subn(template, s)),
        ("split", lambda p, s: p.split(s)),
    )
    checks = 0
    for subject in subjects:
        for name, call in operations:
            expected, got = outcome(lambda: call(ref, subject)), outcome(lambda: call(mine, subject))
            assert got == expected, f"{name}({pattern!r}, flags={int(flags)}, {subject!r}): stdlib {expected!r}, facade {got!r}"
            checks += 1
    return checks


def fresh_interpreter(code, timeout=300):
    """Run ``code`` in a new interpreter that imports the facade from the repository root (never
    from ``tests/``, where ``hermes_re`` is this test package); returns its output as words."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO_ROOT)}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=timeout,
                          cwd=REPO_ROOT, env=env)
    assert done.returncode == 0, done.stderr[-1500:]
    return done.stdout.split()
