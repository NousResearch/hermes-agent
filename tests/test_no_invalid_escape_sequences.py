"""Source-hygiene invariant: no in-tree Python file may carry escape sequences
Python does not recognise.

Two distinct failure modes hide behind the same ``SyntaxWarning``:

- *Invalid* escapes (``\\W``, ``\\S``, ``\\c`` ...) compile but warn on every
  Python 3.12+ interpreter, and will eventually become errors.
- *Valid-but-unintended* escapes (``\\b``, ``\\t``, ``\\n`` inside docstrings
  that document Windows paths or regexes) warn about nothing — the backslash
  is silently reinterpreted, e.g. ``C:\\Windows\\System32\\bash.exe`` rendered
  as ``C:\\Windows\\System32\\x08ash.exe`` in ``pm/shell.py``'s docstring
  (#123484).

Docstrings and embedded script strings that mean their backslashes literally
must be raw strings. This test compiles every tracked Python source file and
asserts the warning class stays empty, so the bug class cannot creep back.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fallback directory filter for checkouts without git metadata: skip anything
# that is not authored in-tree source — virtualenvs, caches, vendored or built
# artifacts, scratch worktrees. ``git ls-files`` needs none of this: tracked
# files are exactly the intended domain (``scripts/build`` is authored source
# even though its parent dir is named ``build``; an ignored ``.venv`` never is).
_SKIP_PARTS = {
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    ".tox",
    ".nox",
}


def _iter_source_files():
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
        tracked = [p for p in listing.split(b"\0") if p]
    except (OSError, subprocess.SubprocessError):
        tracked = None

    if tracked is not None:
        for rel in tracked:
            yield REPO_ROOT / rel.decode("utf-8", "surrogateescape")
        return

    for path in sorted(REPO_ROOT.rglob("*.py")):
        parts = path.relative_to(REPO_ROOT).parts
        if parts[0] == "build" or any(part in _SKIP_PARTS or part.startswith(".") for part in parts):
            continue
        yield path


def test_sources_compile_without_invalid_escape_warnings():
    offenders: dict[str, list[str]] = {}
    for path in _iter_source_files():
        try:
            source = path.read_bytes()
        except OSError:
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                compile(source, str(path), "exec")
            except SyntaxError:
                # A file that does not parse fails loudly elsewhere; this test
                # only owns the escape-sequence warning class.
                continue
        messages = [str(w.message) for w in caught if "invalid escape sequence" in str(w.message)]
        if messages:
            offenders[str(path.relative_to(REPO_ROOT))] = messages
    assert not offenders, (
        "in-tree sources emit SyntaxWarning for invalid escape sequences "
        "(make the offending string raw or double the backslashes): "
        f"{offenders}"
    )
