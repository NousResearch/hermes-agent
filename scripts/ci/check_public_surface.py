#!/usr/bin/env python3
"""Public-surface diff against a base ref: dropped public names and dropped test functions.

The Sep 2026 whole-codebase refactor (PR #102117) opened with 1,703 public top-level names dropped
across 341 modules, 1,000 public methods in 166, and 130 ``def test_`` deleted in 54 files.
Reviewers found ~30 of the names by hand; the rest surfaced as post-merge fixes (10 commits
restoring symbols/re-exports, 6 restoring tests, plus a qwen OAuth break that went past import
smoke because the caller used ``module.attr``). Every one of those was catchable in seconds with
this script; nothing ran it because it did not exist.

Rules (deliberately narrow, no allowlist file to maintain):
- A PUBLIC top-level name (function, class, module-level assignment, re-exported import) or a
  public/dunder method of a module present on BOTH sides may be moved and re-exported or
  deprecated, never silently removed. A deleted MODULE is not a drop here (that is a visible
  decision; ``check_doc_paths`` territory).
- A ``tests/**`` file present on both sides must not end with fewer ``def test_`` than it started
  with. Deleting a whole test file is, again, a visible decision and not flagged.

Advisory by default (prints the report, exit 0). ``--strict`` exits 1 on any finding so a
refactor brief or a CI lane can make it a hard gate. Runs in ~30 s over the whole tree.

Usage:
    python scripts/ci/check_public_surface.py [--base origin/main] [--head HEAD] [--strict] [--json out.json]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys

SRC_DIRS = ("agent", "gateway", "hermes_cli", "tools", "tui_gateway", "cron", "acp_adapter", "plugins")
_TEST_DEF_RE = re.compile(r"^\s*(?:async\s+)?def test_", re.M)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def _changed_py(base: str, head: str) -> list[str]:
    out = _git("diff", "--name-only", "--diff-filter=M", base, head)
    return [f for f in out.split() if f.endswith(".py")]


def _show(rev: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else None


def public_toplevel(src: str) -> set[str] | None:
    """Public top-level names: defs, classes, assigned names, and imported/re-exported names."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(x.id for x in ast.walk(target) if isinstance(x, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return {n for n in names if not n.startswith("_")}


def _is_public_method(name: str) -> bool:
    return not name.startswith("_") or name.startswith("__")


def public_methods(src: str) -> set[str]:
    """``Class.method`` for public and dunder methods REACHABLE on each top-level class: defined on it
    or inherited from a base class defined in the same module. Extracting a method into a mixin/base
    that the class still derives from is not a removal (the attribute still resolves)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

    def own(cls: ast.ClassDef) -> set[str]:
        return {m.name for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_method(m.name)}

    def reachable(cls: ast.ClassDef, seen: set[str]) -> set[str]:
        names = own(cls)
        for base in cls.bases:
            base_name = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else None)
            if base_name in classes and base_name not in seen:
                names |= reachable(classes[base_name], seen | {base_name})
        return names

    out: set[str] = set()
    for name, cls in classes.items():
        out.update(f"{name}.{m}" for m in reachable(cls, {name}))
    return out


def _is_source(path: str) -> bool:
    return "/" not in path or path.split("/", 1)[0] in SRC_DIRS


def diff_surface(base: str, head: str) -> dict:
    dropped_names: dict[str, list[str]] = {}
    dropped_methods: dict[str, list[str]] = {}
    test_drops: dict[str, tuple[int, int]] = {}
    for path in _changed_py(base, head):
        before, after = _show(base, path), _show(head, path)
        if before is None or after is None:
            continue
        if path.startswith("tests/"):
            nb, na = len(_TEST_DEF_RE.findall(before)), len(_TEST_DEF_RE.findall(after))
            if na < nb:
                test_drops[path] = (nb, na)
            continue
        if not _is_source(path) or path.startswith("tests"):
            continue
        pb, pa = public_toplevel(before), public_toplevel(after)
        if pb is not None and pa is not None and (gone := sorted(pb - pa)):
            dropped_names[path] = gone
        if gone_m := sorted(public_methods(before) - public_methods(after)):
            dropped_methods[path] = gone_m
    return {"dropped_names": dropped_names, "dropped_methods": dropped_methods, "test_drops": test_drops}


def render(report: dict) -> str:
    lines: list[str] = []
    n_names = sum(len(v) for v in report["dropped_names"].values())
    n_meth = sum(len(v) for v in report["dropped_methods"].values())
    n_tests = sum(b - a for b, a in report["test_drops"].values())
    lines.append(
        f"public-surface: {n_names} public top-level name(s) dropped in {len(report['dropped_names'])} module(s); "
        f"{n_meth} public/dunder method(s) dropped in {len(report['dropped_methods'])}; "
        f"{n_tests} test def(s) dropped in {len(report['test_drops'])} file(s)"
    )
    for path, names in sorted(report["dropped_names"].items()):
        lines.append(f"  {path}: -{', '.join(names[:12])}{' …' if len(names) > 12 else ''}")
    for path, names in sorted(report["dropped_methods"].items()):
        lines.append(f"  {path}: -{', '.join(names[:8])}{' …' if len(names) > 8 else ''}")
    for path, (b, a) in sorted(report["test_drops"].items()):
        lines.append(f"  {path}: {b} -> {a} test defs")
    if n_names or n_meth or n_tests:
        lines.append(
            "  A public name may be moved and re-exported or deprecated, never silently removed (in-tree refs say "
            "nothing about plugins); a test file must not lose test defs unless the commit names the private symbol "
            "they pinned."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any finding")
    ap.add_argument("--json", dest="json_out", help="also write the report as JSON")
    args = ap.parse_args(argv)
    for ref in (args.base, args.head):
        if subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], capture_output=True).returncode != 0:
            print(f"public-surface: cannot resolve ref {ref!r} (fetch it first); refusing to report a clean diff", file=sys.stderr)
            return 2
    base = _git("merge-base", args.base, args.head).strip()
    if not base:
        print(f"public-surface: no merge-base between {args.base!r} and {args.head!r}", file=sys.stderr)
        return 2
    report = diff_surface(base, args.head)
    print(render(report))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    findings = any(report[k] for k in ("dropped_names", "dropped_methods", "test_drops"))
    return 1 if (args.strict and findings) else 0


if __name__ == "__main__":
    sys.exit(main())
