#!/usr/bin/env python3
"""Fail when in-tree code depends on a plugin-compat pointer.

``compat_manifest.json`` lists every name the Sep 2026 decomposition kept importable from its OLD
module purely for external plugins (the `PLUGIN-COMPAT` blocks). Those blocks are removed on a
schedule by reverting the commit that added them, so nothing inside this repository may depend on
them — otherwise the revert breaks the tree. This check walks every first-party Python file (source
AND tests) and flags:

  from <facade> import <compat_name>          # direct import through the old path
  import <facade>; <facade>.<compat_name>     # attribute access through the old path
  patch("<facade>.<compat_name>") / monkeypatch.setattr(<facade>, "<compat_name>")

Excluded trees and virtualenvs are pruned from the walk *before* they are entered (see
``_prune_reason``): a dependency tree — ``node_modules`` at the root or nested three packages deep —
and an environment (pyvenv.cfg / conda-meta / *-packages / conventional name) cost no directory
reads and can never be reported as in-tree code.

Exit 1 with a file:line list on any hit. Run: python scripts/check_compat_pointers.py
"""
from __future__ import annotations

import ast
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "compat_manifest.json"
# Never first-party source, at any depth in the tree.
EXCLUDED_DIR_NAMES = {".git", "node_modules", "build", "MagicMock", ".worktrees", "__pycache__"}
# First-party trees that are out of scope for this check, but only AT THE REPO ROOT: a nested
# `tests/skills/` or `tests/evals/` is test source and must stay scanned (the old root-component-only
# filter did that by accident; excluding these names at any depth silently dropped 59 test files).
ROOT_EXCLUDED_DIR_NAMES = {"website", "skills", "optional-skills", "apps", "evals"}

# Environments are not first-party code. Scanning one is pure cost, and the environment's own
# packages get reported as in-tree violations the moment an installed package happens to touch a
# pointer name (issue #112584: a `.venv` in the checkout failed the build on `pip/__init__.py`).
# `pyvenv.cfg` (PEP 405) is the marker `python -m venv` always writes, whatever the directory was
# named; `conda-meta` is conda's; a `*-packages` store only ever exists inside an environment or the
# system interpreter. The conventional names below also cover environments without those markers,
# but `env` is a plausible first-party package name, so it needs marker evidence too.
VENV_DIR_NAMES = {"venv", ".venv", "virtualenv", ".virtualenv", "env"}
AMBIGUOUS_VENV_DIR_NAMES = {"env"}
PACKAGE_STORE_DIR_NAMES = {"site-packages", "dist-packages"}

# `fnmatch` patterns for exclusions keyed on location rather than bare name, matched against a
# directory's name AND its slash-separated path relative to the scan root (fnmatch's `*` crosses
# `/`, so `pkg/*/vendor` prunes a vendored tree at any depth under `pkg`). Empty today: the two
# name sets above cover the repo's own layout.
SKIP_PATH_PATTERNS: tuple[str, ...] = ()
# Skipped wherever they appear: this scanner itself, and the compat layer's own contract test
# (which uses the pointers on purpose; it is deleted with them).
SKIP_FILES = {"check_compat_pointers.py", "test_compat_manifest_targets.py"}


def _venv_reason(path: Path) -> str | None:
    """Why `path` is an environment / package store that must not be scanned, else None."""
    name = path.name
    if name in PACKAGE_STORE_DIR_NAMES:
        return name
    if (path / "pyvenv.cfg").is_file():
        return "pyvenv.cfg"
    if (path / "conda-meta").is_dir():
        return "conda-meta"
    if name not in VENV_DIR_NAMES:
        return None
    if name not in AMBIGUOUS_VENV_DIR_NAMES:
        return f"virtualenv dir {name}"
    if (path / "bin" / "activate").is_file() or (path / "Scripts" / "activate").is_file():
        return f"{name}/ with an activate script"
    if (path / "Scripts" / "Lib" / "site-packages").is_dir() or next((path / "lib").glob("python*/site-packages"), None):
        return f"{name}/ with a package store"
    return None


def _prune_reason(path: Path, base: Path | None = None) -> str | None:
    """Why directory `path` must not be walked at all, else None.

    `base` is the scan root used for the root-only layout rule; it defaults to the module ``ROOT``.
    """
    root = Path(base) if base is not None else ROOT
    name = path.name
    if name in EXCLUDED_DIR_NAMES:
        return f"excluded dir {name}"
    if path.parent == root and name in ROOT_EXCLUDED_DIR_NAMES:
        return f"excluded root dir {name}"
    venv = _venv_reason(path)
    if venv is not None:
        return venv
    rel = path.relative_to(root).as_posix()
    return next(
        (f"path pattern {pat}" for pat in SKIP_PATH_PATTERNS
         if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat)),
        None,
    )


def _excluded_dir(rel_path: Path) -> bool:
    """True when the directory at ``rel_path`` (relative to the scan root) is out of scope."""
    return _prune_reason(ROOT / rel_path) is not None


def _py_files(root: Path | None = None):
    """Yield every first-party ``.py`` file under ``root`` (default: this repo).

    The walk is top-down and an excluded directory is dropped from the pending list *before* it
    would be opened, so an excluded dependency tree or virtualenv costs no ``scandir`` calls and
    contributes no file: nothing under it is enumerated, read or parsed. The same rule applies at
    every depth, which is what fixes a `node_modules` nested inside a first-party package. Symlinks
    are not followed, so a linked tree (or a loop) is never entered.
    """
    base = Path(root) if root is not None else ROOT
    for dirpath, dirnames, filenames in os.walk(base, topdown=True, followlinks=False):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if _prune_reason(here / d, base) is None)
        for name in sorted(filenames):
            # ``.lower()`` because pathlib's glob matched case-insensitively on Windows,
            # where a ``FOO.PY`` module is importable.
            if name.lower().endswith(".py") and name not in SKIP_FILES:
                yield here / name


def main() -> int:
    if not MANIFEST.exists():
        print("compat_manifest.json missing — nothing to check (compat layer already reverted?)")
        return 0
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["entries"]
    compat: dict[str, set[str]] = {}
    for e in entries:
        compat.setdefault(e["facade"], set()).add(e["name"])
    facades = set(compat)
    hits: list[str] = []
    str_pat = re.compile(r"""["']((?:[A-Za-z_][\w]*\.)+[A-Za-z_]\w*)["']""")
    for path in _py_files():
        rel = path.relative_to(ROOT)
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except SyntaxError:
            continue
        # module-level facade import aliases in this file: alias -> facade
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in facades and node.level == 0:
                bad = [a.name for a in node.names if a.name in compat[node.module]]
                for b in bad:
                    hits.append(f"{rel}:{node.lineno}: from {node.module} import {b}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in facades:
                        aliases[a.asname or a.name] = a.name
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for a in node.names:
                    full = f"{node.module}.{a.name}"
                    if full in facades:
                        aliases[a.asname or a.name] = full
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                fac = aliases.get(node.value.id)
                if fac and node.attr in compat[fac]:
                    hits.append(f"{rel}:{node.lineno}: {node.value.id}.{node.attr} (via {fac})")
            elif isinstance(node, ast.Call):
                # monkeypatch.setattr(<facade alias>, "<name>", ...) / patch.object(<facade alias>, "<name>")
                fn = node.func
                is_setattr = (isinstance(fn, ast.Attribute) and fn.attr in ("setattr", "delattr", "object")) or (
                    isinstance(fn, ast.Name) and fn.id in ("setattr", "delattr", "getattr", "hasattr"))
                if is_setattr and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                    fac = aliases.get(node.args[0].id)
                    if fac and node.args[1].value in compat[fac]:
                        hits.append(f"{rel}:{node.lineno}: setattr/patch({node.args[0].id}, \"{node.args[1].value}\") (via {fac})")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                m = str_pat.fullmatch(node.value.strip())
                if m:
                    dotted = m.group(1); fac, _, name = dotted.rpartition(".")
                    if fac in facades and name in compat[fac]:
                        hits.append(f"{rel}:{node.lineno}: \"{dotted}\" (string patch target)")
    if hits:
        print("❌ in-tree code depends on plugin-compat pointers (scheduled for removal):")
        for h in sorted(set(hits)):
            print("  " + h)
        print(f"\n{len(set(hits))} site(s). Import from the defining module instead (see COMPAT_MANIFEST.md).")
        return 1
    print(f"✅ no in-tree dependency on the {len(entries)} plugin-compat pointers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
