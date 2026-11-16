#!/usr/bin/env python3
"""Fail when packaged code imports a regular package that the wheel would not ship.

Root single-file modules reach the wheel through ``setup.py::_root_py_modules``; packages reach
it only through ``[tool.setuptools.packages.find] include`` in ``pyproject.toml``. A module
turned into a folder (``toolsets.py`` -> ``toolsets/``, ``acp_adapter/server.py`` ->
``acp_adapter/server/``) leaves the first list, and unless the include patterns match the new
package, the installed wheel raises ``ModuleNotFoundError`` while a checkout keeps working.

Usage: python3 scripts/check_packaged_imports.py
"""
from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent


def _tracked_py(root: Path) -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=root, capture_output=True, text=True, check=True, timeout=60,
    ).stdout
    return [rel for rel in listed.split("\0") if rel]


def shipped_packages(root: Path, tracked: list[str]) -> set[str]:
    """Dotted names pyproject's ``packages.find`` ships: it defaults to ``namespaces = true``, so
    every directory holding tracked Python (``__init__.py`` or not) that an include pattern
    matches. Git paths are POSIX on every host."""
    include = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8-sig"))[
        "tool"]["setuptools"]["packages"]["find"]["include"]
    dirs = {PurePosixPath(rel).parent.parts for rel in tracked} - {()}
    packages = {".".join(parts[:i]) for parts in dirs for i in range(1, len(parts) + 1)}
    return {p for p in packages if any(fnmatchcase(p, pattern) for pattern in include)}


def imported_names(path: Path, package: tuple[str, ...]) -> set[str]:
    """Absolute module names a file imports. ``from x import y`` yields ``x`` and ``x.y``, since
    ``y`` may be a submodule; relative imports resolve against ``package``."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else ()
            module = ".".join((*base, *(node.module.split(".") if node.module else ()))).strip(".")
            if module:
                names.add(module)
                names.update(f"{module}.{alias.name}" for alias in node.names if alias.name != "*")
    return names


def unshipped_imports(root: Path) -> list[str]:
    tracked = _tracked_py(root)
    shipped = shipped_packages(root, tracked)
    shipped_files = [rel for rel in tracked
                     if ("/" not in rel and rel != "setup.py")
                     or ".".join(PurePosixPath(rel).parent.parts) in shipped]
    initialised = {PurePosixPath(rel).parent.as_posix() for rel in tracked
                   if PurePosixPath(rel).name == "__init__.py"}
    missing: set[str] = set()
    for rel in shipped_files:
        for name in imported_names(root / rel, PurePosixPath(rel).parent.parts):
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                pkg, parent = ".".join(parts[:i]), ".".join(parts[: i - 1])
                # A regular package at the root, or inside a shipped one, that the wheel lacks.
                if (parent == "" or parent in shipped) and pkg not in shipped \
                        and "/".join(parts[:i]) in initialised:
                    missing.add(pkg)
    return sorted(missing)


def main() -> int:
    missing = unshipped_imports(ROOT)
    if missing:
        print("Packaged code imports packages the wheel would not ship; add each to"
              " [tool.setuptools.packages.find] include in pyproject.toml:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1
    print("check_packaged_imports: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
