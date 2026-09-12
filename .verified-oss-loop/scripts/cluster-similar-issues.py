#!/usr/bin/env python3
"""Cluster similar issues from a local JSON fixture.

Reads at most MAX_ISSUES records. Never scrapes a live tracker.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MAX_ISSUES = 64
STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "after",
    "when",
    "plus",
    "own",
}

TOKEN = re.compile(r"[a-z0-9]+")


def tokens(title: str) -> frozenset[str]:
    return frozenset(t for t in TOKEN.findall(title.lower()) if t not in STOP and len(t) > 1)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def load_issues(path: Path, cap: int) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("fixture must be a JSON array of issues")
    if len(data) > cap:
        raise SystemExit(f"fixture has {len(data)} issues; cap is {cap}")
    out = []
    for item in data:
        if not isinstance(item, dict) or "number" not in item or "title" not in item:
            raise SystemExit("each issue needs number and title")
        out.append({"number": int(item["number"]), "title": str(item["title"])})
    return out


def cluster(issues: list[dict], threshold: float) -> list[list[int]]:
    n = len(issues)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    toks = [tokens(it["title"]) for it in issues]
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(toks[i], toks[j]) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i, it in enumerate(issues):
        groups.setdefault(find(i), []).append(it["number"])
    clusters = [sorted(v) for v in groups.values()]
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cluster similar issues from a local fixture")
    p.add_argument("fixture", type=Path, help="JSON array of {number, title}")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--cap", type=int, default=MAX_ISSUES)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    if args.cap > MAX_ISSUES:
        print(f"cap cannot exceed {MAX_ISSUES}", file=sys.stderr)
        return 2
    issues = load_issues(args.fixture, args.cap)
    clusters = cluster(issues, args.threshold)
    if args.json:
        json.dump({"clusters": clusters}, sys.stdout)
        sys.stdout.write("\n")
    else:
        for c in clusters:
            print(" ".join(str(n) for n in c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
