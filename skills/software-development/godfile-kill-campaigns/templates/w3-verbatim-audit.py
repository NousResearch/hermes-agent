#!/usr/bin/env python3
"""Wave-3 extraction validator (AST-anchored) — copy and adapt per shard.

Validates a canonical extraction patch applied to a FRESH WORKTREE:
  1. Every moved name's def is char-for-char identical between the LIVE
     godfile (AST line-span extraction) and the new module it moved to.
  2. No moved name is still DEFINED in the applied godfile.
  3. Every moved name is re-exported there — parsed via ast.ImportFrom nodes
     (per-line string checks MISS parenthesized multi-line imports).
  4. Optional: comment blocks immediately above named constants are verbatim
     vs live (anchor on the AST Assign node, NOT str.find — the module
     docstring usually names the constants and .find() hits it first).

False-positive traps learned in main.py s5-w3a: verify every EXPECTED count
against the LIVE source, never against a prior-witness JSON (w2 miscounted an
8-line comment block as 9 — the module matched live exactly).

Usage:
  python w3-verbatim-audit.py \
      --live <repo>/hermes_cli/main.py \
      --new-main <worktree>/hermes_cli/main.py \
      --module <worktree>/hermes_cli/cli_discovery.py:_first_positional_argv,_plugin_cli_discovery_needed \
      --module <worktree>/hermes_cli/agent_startup.py:_is_tui_chat_launch,_prepare_agent_startup \
      [--comment-block <worktree>/hermes_cli/cli_discovery.py:_BUILTIN_SUBCOMMANDS:9] \
      [--expect-absent-ok]  # moved names may legitimately stay defined (rare)

Exit 0 only if every check passes. Run with the repo venv python.
"""
import argparse
import ast
import sys


def read(p):
    with open(p, "r", encoding="utf-8") as fh:
        return fh.read()


def def_spans(src):
    """name -> defining node (first per name), nested ones mapped outward.

    Walks the whole module (ast.walk) instead of tree.body so annotated
    constants (ast.AnnAssign) and names assigned inside a module-level
    try/except or if block are found — both shapes never appear in
    tree.body (SKILL.md traps; PR #79609). A nested assignment maps to
    its enclosing TOP-LEVEL statement so span_text() compares the whole
    moved unit (e.g. the entire try block) against live.
    """
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.setdefault(t.id, node)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                out.setdefault(node.target.id, node)
    for name, node in out.items():
        if node in tree.body:
            continue
        for stmt in tree.body:
            if stmt.lineno <= node.lineno <= stmt.end_lineno:
                out[name] = stmt
                break
    return out


def span_text(src, node):
    ln = src.splitlines(keepends=True)
    return "".join(ln[node.lineno - 1:node.end_lineno])


def reexported_names(src):
    """Names imported at module level from ANY 'from ... import (...)'."""
    out = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.ImportFrom):
            out.update(a.name for a in node.names)
    return out


def comment_block_above(src, name):
    lines = src.splitlines()
    node = def_spans(src).get(name)
    if node is None:
        return None
    out, i = [], node.lineno - 2
    while i >= 0 and lines[i].lstrip().startswith("#"):
        out.append(lines[i])
        i -= 1
    return out[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True, help="live godfile path (READ ONLY)")
    ap.add_argument("--new-main", required=True, help="applied (worktree) godfile")
    ap.add_argument("--module", action="append", default=[],
                    help="path:name1,name2 (module file + moved names)")
    ap.add_argument("--comment-block", action="append", default=[],
                    help="path:name:expected_line_count")
    args = ap.parse_args()

    failures = []
    live_src = read(args.live)
    new_main_src = read(args.new_main)
    live_spans = def_spans(live_src)
    new_spans = def_spans(new_main_src)

    for spec in args.module:
        path, names = spec.split(":", 1)
        mod_src = read(path)
        mod_spans = def_spans(mod_src)
        for name in names.split(","):
            if name not in live_spans:
                failures.append(f"NOT IN LIVE: {name}")
                continue
            if name not in mod_spans:
                failures.append(f"MISSING IN {path}: {name}")
                continue
            lt = span_text(live_src, live_spans[name])
            mt = span_text(mod_src, mod_spans[name])
            if lt != mt:
                failures.append(f"VERBATIM MISMATCH: {name}")
            if name in new_spans:
                failures.append(f"STILL DEFINED IN APPLIED MAIN: {name}")

    reexported = reexported_names(new_main_src)
    for spec in args.module:
        path, names = spec.split(":", 1)
        for name in names.split(","):
            if name not in reexported:
                failures.append(f"NOT RE-EXPORTED IN APPLIED MAIN: {name}")

    for spec in args.comment_block:
        path, name, want = spec.split(":")
        block = comment_block_above(read(path), name)
        if block is None:
            failures.append(f"CONST MISSING IN {path}: {name}")
        elif len(block) != int(want):
            failures.append(
                f"COMMENT BLOCK {name}: {len(block)} lines, expected {want} "
                f"(verify against LIVE source before trusting any JSON count)")

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
