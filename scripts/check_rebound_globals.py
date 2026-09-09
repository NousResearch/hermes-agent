#!/usr/bin/env python3
"""Fail when a module from-imports a name that its defining module reassigns via ``global``.

``from M import N`` copies the binding at import time; a later ``global N; N = ...`` in ``M`` (a
token applied from a CLI flag, a lazily probed SDK) never reaches the copy. One such copy, taken
inside ``mount_spa()`` at import, served Desktop-over-SSH a token the validator rejected (#102930).
Read such names through their module instead: ``import M; M.N`` at the use site.

Production sources only: tests patch module attributes on purpose. Exit 1 with a
``file:line  from M import N`` list on any hit. Run: python scripts/check_rebound_globals.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Importable first-party packages plus the root-level modules (cli.py, run_agent.py, ...).
FIRST_PARTY = ("acp_adapter", "agent", "cron", "gateway", "hermes_cli", "plugins", "tools", "tui_gateway")
SKIP_DIRS = {"__pycache__", "node_modules", "MagicMock", "tests"}


def _py_files():
    yield from ROOT.glob("*.py")
    for pkg in FIRST_PARTY:
        for p in (ROOT / pkg).rglob("*.py"):
            if not (set(p.relative_to(ROOT).parts) & SKIP_DIRS):
                yield p


def _module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _assigned_names(node: ast.AST) -> set[str]:
    """Names bound by assignment statements anywhere under ``node``."""
    names: set[str] = set()
    for sub in ast.walk(node):
        targets = []
        if isinstance(sub, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
        for t in targets:
            for leaf in ast.walk(t):
                if isinstance(leaf, ast.Name):
                    names.add(leaf.id)
    return names


def _rebound_names(tree: ast.AST) -> set[str]:
    """Names a module declares ``global`` in a function AND assigns there (read-only ``global`` is legal)."""
    out: set[str] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        declared = {n for node in fn.body for sub in ast.walk(node) if isinstance(sub, ast.Global) for n in sub.names}
        if declared:
            out |= declared & _assigned_names(fn)
    return out


def _resolve(importer: str, node: ast.ImportFrom, is_package: bool) -> str:
    """Absolute dotted target of ``node`` as seen from ``importer`` (handles ``from . import``)."""
    if not node.level:
        return node.module or ""
    pkg = importer.split(".") if is_package else importer.split(".")[:-1]
    pkg = pkg[: len(pkg) - (node.level - 1)]
    return ".".join(pkg + ([node.module] if node.module else []))


def main() -> int:
    trees: dict[Path, ast.AST] = {}
    for path in _py_files():
        try:
            trees[path] = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue

    rebound = {_module_name(p): names for p, t in trees.items() if (names := _rebound_names(t))}

    hits: list[str] = []
    for path, tree in trees.items():
        me = _module_name(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _resolve(me, node, path.name == "__init__.py")
            if target not in rebound or target == me:
                continue
            bad = sorted(a.name for a in node.names if a.name in rebound[target])
            if bad:
                hits.append(f"{path.relative_to(ROOT)}:{node.lineno}  from {target} import {', '.join(bad)}")

    if hits:
        print("Names reassigned via `global` must be read through their module (import M; M.N),")
        print("never copied with `from M import N` (the copy never sees the rebind):\n")
        print("\n".join(sorted(hits)))
        return 1
    print("check_rebound_globals: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
