#!/usr/bin/env python3
"""Simulated agent lookups: how many tool calls / tokens to answer "show me the definition of X"?

Deterministic agent policy (the one a careful model actually follows, given read_file's 2,000-line window):
  1. grep -n "def X\\b|class X\\b|^X\\s*=" across the tree          -> 1 call, returns (file, line) hits
  2. read_file(file, offset=hit_line-20, limit=W) for W in (200,)   -> 1 call; if the symbol's end is past the
     window, read the next 2,000-line window until it is           -> +1 call each
  3. done when the whole definition is in context

Costs charged per task:
  tool_calls   = 1 grep + N reads
  tokens       = tokens of grep output (all hits, one line each) + tokens of every read window returned
Both trees get the same symbol set: the intersection of symbols that exist (by name) on both, so a symbol moving
to a smaller file counts as a WIN and a symbol that was deleted on one side is excluded, not scored.

Usage: python evals/codebase_navigability/lookup_sim.py <base_tree> <head_tree> [--sample N] [--seed S]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import statistics
import sys
from pathlib import Path

SKIP_TOP = {".git", "node_modules", "apps", "website", "build", ".venv", "venv", "MagicMock", "__pycache__", ".worktrees", "dist", "evals", "skills", "optional-skills", "docs", "tests"}
WINDOW = 2000          # read_file's default max lines per call
FIRST_READ = 200       # a careful agent's first targeted read around the grep hit (override: --first-read)
MARGIN = 5             # lines of context above the grep hit


def tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return lambda s: len(enc.encode(s, disallowed_special=()))
    except Exception:
        return lambda s: len(s.encode()) // 4


def index(tree: Path):
    """name -> list of (relpath, start_line, end_line, lines_of_file)"""
    idx: dict[str, list] = {}
    files: dict[str, list[str]] = {}
    for dp, dns, fns in os.walk(tree):
        rel = os.path.relpath(dp, tree)
        if rel != "." and rel.split(os.sep)[0] in SKIP_TOP:
            dns[:] = []
            continue
        for f in fns:
            if not f.endswith(".py"):
                continue
            p = Path(dp) / f
            src = p.read_text(encoding="utf-8", errors="replace")
            lines = src.split("\n")
            r = os.path.relpath(p, tree)
            files[r] = lines
            try:
                t = ast.parse(src)
            except SyntaxError:
                continue
            for n in ast.walk(t):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    s = min([d.lineno for d in n.decorator_list] + [n.lineno])
                    idx.setdefault(n.name, []).append((r, s, n.end_lineno, len(lines)))
    return idx, files


def grep_hits(name, idx):
    """Simulate `grep -rn` for the symbol: every def/class of that name, one line each."""
    return idx.get(name, [])


def simulate(name, idx, files, tok):
    hits = grep_hits(name, idx)
    if not hits:
        return None
    grep_out = "\n".join(f"{r}:{s}: def {name}(...)" for r, s, _, _ in hits)
    tokens = tok(grep_out)
    calls = 1
    # The agent opens the FIRST hit (deterministic; ambiguity costs are the same policy on both trees)
    r, s, e, n = hits[0]
    lines = files[r]
    start = max(1, s - MARGIN)
    # The agent doesn't know where the definition ends until it reads it: it asks for a window
    # sized to the typical definition (FIRST_READ) and pages forward only if the def keeps going.
    end = min(n, start + FIRST_READ - 1)
    tokens += tok("\n".join(lines[start - 1:end]))
    calls += 1
    while end < e:                     # definition continues past the window: page forward
        start = end + 1
        end = min(n, start + WINDOW - 1)
        tokens += tok("\n".join(lines[start - 1:end]))
        calls += 1
    exact = tok("\n".join(lines[s - 1:e]))
    return {"name": name, "file": r, "file_lines": n, "def_lines": e - s + 1, "hits": len(hits), "calls": calls, "tokens": tokens, "exact_tokens": exact}


def main():
    global FIRST_READ
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("head")
    ap.add_argument("--sample", type=int, default=3000); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--first-read", type=int, default=FIRST_READ, help="lines in the first targeted read")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    FIRST_READ = a.first_read
    tok = tokenizer()
    bi, bf = index(Path(a.base)); hi, hf = index(Path(a.head))
    common = sorted(set(bi) & set(hi))
    # weight the sample toward symbols that are actually looked up: public names, exclude dunders/tests
    common = [n for n in common if not n.startswith("__") and not n.startswith("test_")]
    random.Random(a.seed).shuffle(common)
    sample = common[: a.sample]
    B = [simulate(n, bi, bf, tok) for n in sample]
    H = [simulate(n, hi, hf, tok) for n in sample]
    pairs = [(b, h) for b, h in zip(B, H) if b and h]
    def agg(rows):
        c = [r["calls"] for r in rows]; t = [r["tokens"] for r in rows]
        return {"tasks": len(rows), "calls_mean": round(statistics.mean(c), 3), "calls_total": sum(c), "multi_window_tasks": sum(1 for x in c if x > 2),
                "tokens_mean": round(statistics.mean(t)), "tokens_p50": int(statistics.median(t)), "tokens_p90": sorted(t)[int(len(t) * .9)], "tokens_total": sum(t),
                "file_lines_p50": int(statistics.median(r["file_lines"] for r in rows)), "def_lines_p50": int(statistics.median(r["def_lines"] for r in rows)),
                "ambiguous_hits_mean": round(statistics.mean(r["hits"] for r in rows), 2),
                "exact_def_tokens_mean": round(statistics.mean(r["exact_tokens"] for r in rows)), "exact_def_tokens_total": sum(r["exact_tokens"] for r in rows)}
    res = {"common_symbols": len(common), "sampled": len(pairs), "seed": a.seed, "window": WINDOW, "first_read": FIRST_READ,
           "base": agg([b for b, _ in pairs]), "head": agg([h for _, h in pairs]),
           "head_cheaper": sum(1 for b, h in pairs if h["tokens"] < b["tokens"]), "head_costlier": sum(1 for b, h in pairs if h["tokens"] > b["tokens"]), "equal": sum(1 for b, h in pairs if h["tokens"] == b["tokens"]),
           "worst_regressions": sorted(({"name": h["name"], "base_tok": b["tokens"], "head_tok": h["tokens"], "head_file": h["file"]} for b, h in pairs), key=lambda x: x["base_tok"] - x["head_tok"])[:8],
           "best_wins": sorted(({"name": h["name"], "base_tok": b["tokens"], "head_tok": h["tokens"], "base_file": b["file"], "head_file": h["file"]} for b, h in pairs), key=lambda x: x["head_tok"] - x["base_tok"])[:8]}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"lookup_sim_w{FIRST_READ}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("worst_regressions", "best_wins")}, indent=1))
    print("best wins:", json.dumps(res["best_wins"][:4], indent=1)); print("worst regressions:", json.dumps(res["worst_regressions"][:4], indent=1))


if __name__ == "__main__":
    main()
