#!/usr/bin/env python3
"""Rank a git repo's files by churn and size: the tech-debt audit's priority corner.

  python3 churn_rank.py [--repo .] [--since "12 months ago"] [--top 30] [--path src] [--json]

Counts the commits touching each file (``git log --name-only``), joins the
current line count of files that still exist, and prints the top files by
churn plus the largest tracked files. Stdlib only; needs ``git`` on PATH.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Generated files whose size says nothing about debt.
SKIP_SUFFIXES = {".lock", ".svg", ".map"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                          stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace", timeout=300).stdout


def _lines(path: Path) -> int | None:
    """Line count of a text file; None for binaries (a NUL byte in the first 8 KiB) and lockfiles."""
    if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
        return None
    with path.open("rb") as fh:
        if b"\0" in fh.read(8192):
            return None
        fh.seek(0)
        return sum(1 for _ in fh)


def _paths(output: str) -> list[str]:
    """Split ``-z`` output: raw paths, unlike the C-quoted display lines core.quotePath produces."""
    return [p for p in output.split("\0") if p]


def churn(repo: Path, since: str | None, scope: str | None) -> Counter:
    args = ["log", "-z", "--format=", "--name-only", "--no-renames"]
    if since:
        args.append(f"--since={since}")
    if scope:
        args += ["--", scope]
    return Counter(_paths(_git(repo, *args)))


def rank(repo: Path, since: str | None, top: int, scope: str | None) -> dict:
    counts = churn(repo, since, scope)
    tracked = _paths(_git(repo, "ls-files", "-z", *(["--", scope] if scope else [])))
    loc = {p: _lines(repo / p) for p in {*counts, *tracked}}  # each file read once for both rankings
    live = [(p, n, loc[p]) for p, n in counts.most_common() if (repo / p).is_file()]
    sizes = sorted(((p, loc[p]) for p in tracked), key=lambda t: t[1] or 0, reverse=True)
    return {
        "repo": str(repo),
        "since": since,
        "files_changed": len(counts),
        "by_churn": [{"path": p, "commits": n, "lines": loc} for p, n, loc in live[:top]],
        "by_size": [{"path": p, "lines": loc, "commits": counts.get(p, 0)} for p, loc in sizes[:top] if loc],
    }


def _table(rows: list[dict], first: str, second: str) -> str:
    out = [f"{first:>8} {second:>8}  path"]
    out += [f"{r[first]!s:>8} {r[second]!s:>8}  {r['path']}" for r in rows]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--since", default=None, help='git date, e.g. "12 months ago"')
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--path", default=None, help="limit to a path inside the repo")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        result = rank(Path(args.repo).resolve(), args.since, args.top, args.path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"git failed: {getattr(exc, 'stderr', None) or exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"# churn ({result['files_changed']} files changed{', since ' + args.since if args.since else ''})")
    print(_table(result["by_churn"], "commits", "lines"))
    print("\n# largest tracked files")
    print(_table(result["by_size"], "lines", "commits"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
