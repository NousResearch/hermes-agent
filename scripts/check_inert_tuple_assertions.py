#!/usr/bin/env python3
"""Fail when a statement is a bare tuple whose first element is an assertion.

Dropping the ``assert`` keyword in front of an assertion that carries a message
turns the whole statement into a tuple *expression*::

    sync.assert_not_awaited(), "a reconnect inside the backoff window must not sync"

Python builds a 2-tuple and discards it. A ``Mock.assert_*()`` element does still
execute (so a mismatch raises), but the statement is broken as a *contract* in
three ways that all shipped in this tree:

  * a non-call element never runs at all — ``x == y, "msg"`` and
    ``m.assert_called_once, "msg"`` assert nothing whatsoever;
  * the message is discarded, so a failure reports the mock's generic text
    instead of the reason the author wrote down;
  * it reads as an assertion to every reviewer, which is how four of these
    survived review. The one at
    ``tests/gateway/test_discord_command_sync_recovery.py`` was caught by a
    human (@Enough1122 on #117299), never by CI.

Why a checker and not a lint rule: **ruff/flake8 B018 (useless-expression) does
not catch this.** B018 deliberately exempts any expression containing a ``Call``
because calls may have side effects — and the common shapes here are calls.
Measured on this tree with ruff 0.15.12: B018 flags ``x, 2`` and is silent on
``m.assert_called_once(), "msg"``. A regex is wrong in both directions: it
misses the non-mock shapes and over-matches the thousands of legitimate bare
``m.assert_called()`` statements that are not tuples. The tuple is the
discriminator, so the detector needs an AST.

The rule is narrow on purpose: only a *statement-level* ``ast.Expr`` whose value
is an ``ast.Tuple``, and only when its FIRST element is an assertion-looking
shape. A bare ``m.assert_called_once()`` on its own line is the correct idiom
and is untouched, as is a deliberate side-effect tuple such as
``(a.clear(), b.clear())``.

Opt out of one line with ``# inert-tuple: ok — <why>`` on that line or the line
directly above it.

Run: python scripts/check_inert_tuple_assertions.py [paths...]
Exit 1 on any violation, 0 when clean.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MARKER = "inert-tuple: ok"

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "site-packages",
}


def assertion_shape(node: ast.expr) -> str | None:
    """Name the assertion-looking shape of ``node``, else ``None``.

    These are the expression shapes that are meaningful as the subject of an
    ``assert`` and meaningless as a discarded tuple element.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr.startswith("assert"):
            return f"mock assertion {func.attr}()"
        if isinstance(func, ast.Name) and func.id.startswith("assert"):
            return f"assertion helper {func.id}()"
        return None
    if isinstance(node, ast.Attribute) and node.attr.startswith("assert"):
        return f"uncalled mock assertion {node.attr}"
    if isinstance(node, ast.Compare):
        return "comparison"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return "`not` expression"
    return None


def _suppressed(lines: list[str], lineno: int) -> bool:
    """True when the statement's line, or the line above it, carries MARKER."""
    for idx in (lineno - 1, lineno - 2):
        if 0 <= idx < len(lines) and MARKER in lines[idx]:
            return True
    return False


def inert_tuple_assertions(source: str, rel: str) -> list[str]:
    """Statement-level tuples whose first element looks like an assertion."""
    tree = ast.parse(source)
    lines = source.splitlines()
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Tuple):
            continue
        elts = node.value.elts
        if not elts:
            continue
        shape = assertion_shape(elts[0])
        if shape is None or _suppressed(lines, node.lineno):
            continue
        problems.append(
            f"{rel}:{node.lineno}: discards a {shape} inside a tuple expression "
            f"— the `assert` keyword is missing, so the message is dropped and "
            f"the statement asserts nothing it appears to"
        )
    return problems


def iter_python_files(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for base in paths:
        if base.is_file():
            if base.suffix == ".py":
                found.append(base)
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            found.append(path)
    return sorted(set(found))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="files or directories (default: repo root)")
    args = parser.parse_args(argv)

    targets = args.paths or [ROOT]
    files = iter_python_files(targets)
    if not files:
        print("check_inert_tuple_assertions: scanned ZERO Python files — the scan is vacuous", file=sys.stderr)
        return 1

    problems: list[str] = []
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        try:
            problems += inert_tuple_assertions(source, rel)
        except SyntaxError:
            # Not our job to police syntax; ruff/pytest collection report it.
            continue

    if problems:
        print("A tuple expression is not an assert. Add the missing `assert` keyword")
        print("(or drop the message and leave the mock call on its own line):")
        print()
        for problem in problems:
            print(f"  {problem}")
        print()
        print(f"{len(problems)} inert tuple assertion(s) in {len(files)} scanned file(s).")
        return 1

    print(f"check_inert_tuple_assertions: clean ({len(files)} files scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
