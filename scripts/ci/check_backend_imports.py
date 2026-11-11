#!/usr/bin/env python3
"""Fail when code outside the storage layer imports the SQLite backend.

Callers reach storage through the contracts in ``persistence/`` and the doors in
``persistence_assembly.py`` (``open_persistence``, ``open_persistence_reader``,
``open_store_admin``). Only those two places name ``persistence.sqlite``; tests may reach
into the backend they test.

Flags, in every Python file outside ``persistence/``, ``persistence_assembly.py`` and any
``tests`` directory: ``import persistence.sqlite...``, ``from persistence.sqlite... import``,
``from persistence import sqlite`` (alone or among other names, in any layout) and
``importlib.import_module("persistence.sqlite...")`` (or ``__import__``) with a literal name.
A name built at run time is not caught.

``_ALLOWED`` maps a file to the reason it may import the backend. An entry whose file is gone
is reported so it can be dropped.

Inside the backend (``persistence/sqlite/``) a store reads only beside its own profile folder, so
the guard also flags each import or attribute read of a process or task home reader
(``_HOME_READERS``) that ``_HOME_READS`` does not list for that file with its reason, and every
import of the ``hermes_state`` stand-in. A listed read no longer present is reported so its entry
can be dropped.

Run: python scripts/ci/check_backend_imports.py [--root DIR]
Exit 1 on any violation, 0 when clean.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

_BACKEND = "persistence.sqlite"
_OWNERS = ("persistence/", "persistence_assembly.py")
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "tests"}
_ALLOWED: dict[str, str] = {}
_BACKEND_DIR = "persistence/sqlite"
_HOME_READERS = frozenset({
    "get_hermes_home", "get_hermes_home_override", "get_process_hermes_home", "profile_cli_selector",
    "display_hermes_home", "get_default_hermes_root", "load_config_readonly",
})
_HOME_READS: dict[tuple[str, str], str] = {
    ("persistence/sqlite/conversation/_store/sessions.py", "get_default_hermes_root"):
        "the owner name the store stamps: default for the install root, the profile name under <root>/profiles",
    ("persistence/sqlite/admin/_reconcile.py", "get_default_hermes_root"):
        "the repair command the reconciler prints targets the install the operator's shell resolves",
    ("persistence/sqlite/_state/holders.py", "profile_cli_selector"):
        "the held-store refusal's -p guidance names the process's active profile",
    ("persistence/sqlite/_state/repair.py", "profile_cli_selector"):
        "the exhausted-repair error's -p guidance names the process's active profile",
    ("persistence/sqlite/_state/repair.py", "get_hermes_home"): "the writability preflight checks the process home",
    ("persistence/sqlite/_backend.py", "get_hermes_home"):
        "a store's open binds its own profile folder as the home only when it differs from the current one",
}


def _names_backend(name: str | None) -> bool:
    return bool(name) and (name == _BACKEND or name.startswith(_BACKEND + "."))


def backend_import_lines(source: str) -> list[int]:
    """Line numbers of the backend imports in ``source``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_names_backend(alias.name) for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if _names_backend(node.module) or (
                    node.module == "persistence" and any(alias.name == "sqlite" for alias in node.names)):
                lines.append(node.lineno)
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            first = node.args[0]
            if (called in ("import_module", "__import__") and isinstance(first, ast.Constant)
                    and isinstance(first.value, str) and _names_backend(first.value)):
                lines.append(node.lineno)
    return sorted(set(lines))


def find_backend_imports(root: Path) -> list[tuple[str, int]]:
    """``(path, line)`` of every backend import outside the storage layer, allowed files excepted."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"repository root is not a directory: {root}")
    offenders = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIRS)
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if rel.startswith(_OWNERS) or rel in _ALLOWED:
                continue
            try:
                source = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError):
                continue
            if "sqlite" not in source:
                continue
            offenders.extend((rel, line) for line in backend_import_lines(source))
    return offenders


def home_read_lines(source: str) -> list[tuple[int, str]]:
    """``(line, name)`` of each home reader ``source`` imports or reads as an attribute, and ``(line, "hermes_state")``
    of each import of the stand-in."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {(node.lineno, "hermes_state") for alias in node.names if alias.name.startswith("hermes_state")}
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and (node.module or "").startswith("hermes_state"):
                found.add((node.lineno, "hermes_state"))
            found |= {(node.lineno, alias.name) for alias in node.names if alias.name in _HOME_READERS}
        elif isinstance(node, ast.Attribute) and node.attr in _HOME_READERS:
            found.add((node.lineno, node.attr))
    return sorted(found)


def find_backend_home_reads(root: Path) -> tuple[list[tuple[str, int, str]], set[tuple[str, str]]]:
    """``(path, line, name)`` of every unlisted home read and stand-in import in the backend, and the listed reads
    present."""
    root = root.resolve()
    offenders, listed = [], set()
    for path in sorted((root / _BACKEND_DIR).rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            continue
        for line, name in home_read_lines(source):
            if (rel, name) in _HOME_READS:
                listed.add((rel, name))
            else:
                offenders.append((rel, line, name))
    return offenders, listed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject imports of the SQLite backend outside the storage layer.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="repository root to inspect (default: this checkout)")
    args = parser.parse_args(argv)
    try:
        offenders = find_backend_imports(args.root)
    except ValueError as exc:
        parser.error(str(exc))
    home_reads, listed = find_backend_home_reads(args.root)
    for rel in sorted(_ALLOWED):
        if not (args.root / rel).exists():
            print(f"note: {rel} is gone; drop its entry from _ALLOWED in scripts/ci/check_backend_imports.py")
    if (args.root / _BACKEND_DIR).is_dir():
        for rel, name in sorted(set(_HOME_READS) - listed):
            print(f"note: {rel} no longer reads {name}; drop its entry from _HOME_READS in "
                  "scripts/ci/check_backend_imports.py")
    if offenders:
        print(f"::error::{_BACKEND} is imported outside the storage layer")
        for rel, line in offenders:
            print(f"  {rel}:{line}")
        print("Open storage through persistence_assembly (open_persistence, open_persistence_reader,"
              " open_store_admin) and use the contracts in persistence/.")
    if home_reads:
        print("::error::the SQLite backend reads the process or task home, or imports hermes_state, where "
              "_HOME_READS lists no reason")
        for rel, line, name in home_reads:
            print(f"  {rel}:{line} {name}")
        print("A store reads beside its own profile folder (its ProfileScope); a process-home read the backend"
              " needs goes in _HOME_READS with its reason.")
    if offenders or home_reads:
        return 1
    print(f"No {_BACKEND} imports outside persistence/ and persistence_assembly.py, and no unlisted home read in it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
