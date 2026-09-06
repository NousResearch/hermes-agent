#!/usr/bin/env python3
"""Agent-navigability benchmark: what does it cost an LLM agent to find and read a symbol?

Workload (not hand-picked): every ``from <first-party module> import <Name>`` in tests/ on the tree under
test, resolved to the module that DEFINES the name on that tree. Each is one "locate and read X" task.

Per task, three costs are measured in real tokenizer tokens (tiktoken o200k_base, the GPT-4o/o-series
tokenizer; other tokenizers differ by a roughly constant factor):

  file_tokens     tokens to read the whole file that defines X (the naive "read_file" cost)
  symbol_tokens   tokens of X's own definition (irreducible: you must read this)
  overhead        file_tokens - symbol_tokens  (what the file's SIZE costs you beyond the answer)
  fits_128k / fits_32k   can the defining file be loaded whole into that context?
  windows_2k      how many 2,000-line read_file windows the file spans (how many reads to scan it)

Also: for each defining file, the number of OTHER top-level symbols it contains (how much unrelated
code sits next to the answer), and the depth/complexity of the symbol itself.

Usage: python evals/codebase_navigability/bench.py <tree> <label> [--out DIR]
Compare two labels with: python evals/codebase_navigability/compare.py base.json head.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SKIP_TOP = {".git", "node_modules", "apps", "website", "build", ".venv", "venv", "MagicMock", "__pycache__",
            ".worktrees", "dist", "evals", "skills", "optional-skills", "docs", "tests"}
FIRST_PARTY_HINT = None  # resolved from the tree: any top-level .py or package dir


def _tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return lambda s: len(enc.encode(s, disallowed_special=()))
    except Exception:
        sys.stderr.write("tiktoken unavailable; falling back to bytes/4 (label will say so)\n")
        return lambda s: len(s.encode("utf-8")) // 4


def source_modules(tree: Path) -> dict[str, Path]:
    mods = {}
    for dp, dns, fns in os.walk(tree):
        rel = os.path.relpath(dp, tree)
        top = rel.split(os.sep)[0]
        if rel != "." and top in SKIP_TOP:
            dns[:] = []
            continue
        for f in fns:
            if f.endswith(".py"):
                p = Path(dp) / f
                m = os.path.relpath(p, tree)[:-3].replace(os.sep, ".")
                if m.endswith(".__init__"):
                    m = m[:-9]
                mods[m] = p
    return mods


def top_level_defs(src: str) -> dict[str, tuple[int, int, ast.AST]]:
    out = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for n in tree.body:
        names = []
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [n.name]
        elif isinstance(n, ast.Assign):
            names = [t.id for t in n.targets if isinstance(t, ast.Name)]
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names = [n.target.id]
        start = n.lineno
        if getattr(n, "decorator_list", None):
            start = min(d.lineno for d in n.decorator_list)
        for nm in names:
            out[nm] = (start, n.end_lineno, n)
    return out


def test_imports(tree: Path, mods: dict[str, Path]) -> list[tuple[str, str, str]]:
    """(test_file, module, name) for every first-party from-import in tests/."""
    tasks = []
    for dp, _, fns in os.walk(tree / "tests"):
        for f in fns:
            if not f.endswith(".py"):
                continue
            p = Path(dp) / f
            try:
                t = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for n in ast.walk(t):
                if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and n.module in mods:
                    for a in n.names:
                        if a.name != "*":
                            tasks.append((os.path.relpath(p, tree), n.module, a.name))
    return tasks


def resolve_definer(mods, name, start_mod, cache, depth=0):
    """Follow re-exports (from X import name / lazy tables) to the module whose top level DEFINES name."""
    if depth > 6 or start_mod not in mods:
        return None
    if start_mod not in cache:
        src = mods[start_mod].read_text(encoding="utf-8", errors="replace")
        cache[start_mod] = (src, top_level_defs(src), _reexports(src))
    src, defs, rex = cache[start_mod]
    if name in defs and not _is_alias(defs[name][2]):
        return start_mod                       # defined here (def/class/real assignment)
    if name in rex:                            # re-export: follow to the origin
        return resolve_definer(mods, rex[name][1], rex[name][0], cache, depth + 1)
    return start_mod if name in defs else None


def _is_alias(node) -> bool:
    return isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(getattr(node, "value", None), ast.Name)


def _reexports(src: str) -> dict[str, tuple[str, str]]:
    """name -> (module, original) for top-level `from m import orig as name`, plus PLUGIN-COMPAT lazy tables."""
    out = {}
    try:
        t = ast.parse(src)
    except SyntaxError:
        return out
    for n in t.body:
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            for a in n.names:
                out[a.asname or a.name] = (n.module, a.name)
        elif isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "_PLUGIN_COMPAT_LAZY" for x in n.targets) and isinstance(n.value, ast.Dict):
            for k, v in zip(n.value.keys, n.value.values):
                try:
                    out[ast.literal_eval(k)] = tuple(ast.literal_eval(v))
                except Exception:
                    pass
    return out


def cyclomatic(node) -> int:
    c = 1
    for n in ast.walk(node):
        if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.Assert, ast.IfExp, ast.comprehension, ast.AsyncFor, ast.AsyncWith)):
            c += 1
        elif isinstance(n, ast.BoolOp):
            c += len(n.values) - 1
        elif isinstance(n, ast.Match):
            c += len(n.cases)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("label")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    tree = Path(args.tree).resolve()
    tok = _tokenizer()
    mods = source_modules(tree)
    tasks = test_imports(tree, mods)
    cache: dict = {}
    file_tok_cache: dict[str, int] = {}
    rows = []
    unresolved = 0
    for test_file, mod, name in tasks:
        definer = resolve_definer(mods, name, mod, cache)
        if definer is None:
            unresolved += 1
            continue
        src, defs, _ = cache[definer]
        if name not in defs:
            unresolved += 1
            continue
        s, e, node = defs[name]
        if definer not in file_tok_cache:
            file_tok_cache[definer] = tok(src)
        lines = src.split("\n")
        seg = "\n".join(lines[s - 1:e])
        rows.append({
            "test": test_file, "imported_from": mod, "name": name, "definer": definer,
            "file_lines": len(lines), "file_tokens": file_tok_cache[definer],
            "symbol_lines": e - s + 1, "symbol_tokens": tok(seg),
            "siblings": len(defs) - 1, "cc": cyclomatic(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else 0,
            "reexported": definer != mod,
        })
    # aggregate
    ft = [r["file_tokens"] for r in rows]
    ov = [r["file_tokens"] - r["symbol_tokens"] for r in rows]
    st = [r["symbol_tokens"] for r in rows]
    q = lambda a, p: (sorted(a)[min(len(a) - 1, int(len(a) * p))] if a else 0)
    summary = {
        "label": args.label, "tree": str(tree), "tokenizer": "o200k_base" if "tiktoken" in sys.modules else "bytes/4",
        "tasks": len(rows), "unresolved": unresolved, "distinct_symbols": len({(r["definer"], r["name"]) for r in rows}),
        "distinct_defining_files": len({r["definer"] for r in rows}),
        "file_tokens_mean": round(statistics.mean(ft)), "file_tokens_p50": q(ft, .5), "file_tokens_p90": q(ft, .9), "file_tokens_p99": q(ft, .99), "file_tokens_max": max(ft),
        "symbol_tokens_mean": round(statistics.mean(st)), "symbol_tokens_p50": q(st, .5),
        "overhead_tokens_mean": round(statistics.mean(ov)), "overhead_tokens_p50": q(ov, .5), "overhead_tokens_p90": q(ov, .9),
        "overhead_ratio_mean": round(statistics.mean(r["file_tokens"] / max(1, r["symbol_tokens"]) for r in rows), 1),
        "total_file_tokens_if_read_whole": sum(ft), "total_symbol_tokens": sum(st),
        "tasks_file_over_128k": sum(1 for x in ft if x > 128_000), "tasks_file_over_32k": sum(1 for x in ft if x > 32_000), "tasks_file_over_8k": sum(1 for x in ft if x > 8_000),
        "tasks_file_over_2000_lines": sum(1 for r in rows if r["file_lines"] > 2000),
        "read_windows_2k_mean": round(statistics.mean(-(-r["file_lines"] // 2000) for r in rows), 2),
        "siblings_mean": round(statistics.mean(r["siblings"] for r in rows), 1), "siblings_p90": q([r["siblings"] for r in rows], .9),
        "symbol_cc_mean": round(statistics.mean(r["cc"] for r in rows if r["cc"]), 2), "symbol_cc_p90": q([r["cc"] for r in rows if r["cc"]], .9),
        "symbol_lines_p50": q([r["symbol_lines"] for r in rows], .5), "symbol_lines_p90": q([r["symbol_lines"] for r in rows], .9),
        "reexport_hops": sum(1 for r in rows if r["reexported"]),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.navigability.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
