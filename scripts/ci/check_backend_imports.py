#!/usr/bin/env python3
"""Fail when code outside the storage layer imports the SQLite backend.

Callers reach storage through the contracts in ``persistence/`` and the doors in
``persistence_assembly.py`` (``open_persistence``, ``open_persistence_reader``,
``open_store_admin``). Only those two places name ``persistence.sqlite``; tests may reach
into the backend they test.

Flags, in every Python file outside ``persistence/``, ``persistence_assembly.py`` and any
``tests`` directory: ``import persistence.sqlite...``, ``from persistence.sqlite... import``,
``from persistence import sqlite`` and ``importlib.import_module("persistence.sqlite...")`` (or
``__import__``) with a literal name. A name built at run time is not caught.

``_ALLOWED`` maps a file to the reason it may import the backend. An entry whose file is gone
is reported so it can be dropped.

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
_ALLOWED = {
    "hermes_state.py": "the legacy SessionDB is served by the backend's store holder until it is removed",
    "hermes_state_holders.py": "the legacy store's generation check reads the backend's holder registry",
    "hermes_state_registry.py": "the legacy store's registry is the backend's holder registry",
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
            if _BACKEND not in source and "import sqlite" not in source:
                continue
            offenders.extend((rel, line) for line in backend_import_lines(source))
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject imports of the SQLite backend outside the storage layer.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="repository root to inspect (default: this checkout)")
    args = parser.parse_args(argv)
    try:
        offenders = find_backend_imports(args.root)
    except ValueError as exc:
        parser.error(str(exc))
    for rel in sorted(_ALLOWED):
        if not (args.root / rel).exists():
            print(f"note: {rel} is gone; drop its entry from _ALLOWED in scripts/ci/check_backend_imports.py")
    if not offenders:
        print(f"No {_BACKEND} imports outside persistence/ and persistence_assembly.py.")
        return 0
    print(f"::error::{_BACKEND} is imported outside the storage layer")
    for rel, line in offenders:
        print(f"  {rel}:{line}")
    print("Open storage through persistence_assembly (open_persistence, open_persistence_reader,"
          " open_store_admin) and use the contracts in persistence/.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
