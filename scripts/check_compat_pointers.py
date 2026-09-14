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

Exit 1 with a file:line list on any hit. Run: python scripts/check_compat_pointers.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "compat_manifest.json"
SKIP_DIRS = {".git", "node_modules", "website", "skills", "optional-skills", "apps", "evals", "build", "MagicMock", ".worktrees", "__pycache__"}


def _py_files():
    for p in ROOT.rglob("*.py"):
        parts = p.relative_to(ROOT).parts
        if parts[0] in SKIP_DIRS or p.name == "check_compat_pointers.py":
            continue
        # The compat layer's own contract test uses the pointers on purpose; it is deleted with them.
        if p.name == "test_compat_manifest_targets.py":
            continue
        yield p


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
