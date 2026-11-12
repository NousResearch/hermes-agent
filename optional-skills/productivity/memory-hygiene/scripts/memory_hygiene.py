#!/usr/bin/env python3
"""Deterministic dry-run hygiene scan of the Hermes memory stores.

Scans the injected-memory files (MEMORY.md / USER.md) and reports:

* entry count and character-budget pressure per store;
* exact and near-duplicate entry pairs;
* status-log-like entries that probably do not belong in memory.

The script NEVER edits or deletes anything. Removals are the agent's call via
the `memory` tool and must be quarantined first: append one line to
`<HERMES_HOME>/mem-quarantine.log` per content-losing operation, before the
change happens:

    <date> | <memory|user> | <full text lost> | <reason>

Usage:
    python memory_hygiene.py [--mem PATH] [--user PATH]
                            [--budget-mem 2200] [--budget-user 1375]
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
from pathlib import Path


def default_home() -> Path:
    """Hermes home: $HERMES_HOME, else %LOCALAPPDATA%/hermes (Windows), else ~/.hermes."""
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    if os.name != "posix":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "hermes"
    return Path.home() / ".hermes"


def entries(text: str) -> list[str]:
    """Split a memory store into entries (a line containing only '§')."""
    out: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.strip() == "§":
            out.append("\n".join(cur).strip())
            cur = []
        else:
            cur.append(line)
    out.append("\n".join(cur).strip())
    return [e for e in out if e]


# Heuristic: status/log lines that belong in session history, not in memory.
# Tune for your vocabulary.
STATUS_RE = re.compile(
    r"(?i)(\d+\s*/\s*\d+\s*(done|ок|ok)|cycle (done|complete)|status.?log"
    r"|\bпрогонён\w*|\bвыполнено\b|\bготово\b|\bзаверш\w*|\bOK\s+\d+)"
)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def scan(label: str, path: Path, budget: int) -> None:
    print(f"== {label}: {path}")
    if not path.exists():
        print("   (not found — skip)\n")
        return
    text = path.read_text(encoding="utf-8")
    es = entries(text)
    used = len(text)
    pct = round(100 * used / budget) if budget else 0
    print(f"   {len(es)} entries, {used}/{budget} chars ({pct}%)")
    if pct > 80:
        print(f"   BUDGET PRESSURE: {pct}% — consolidate before adding")
    for i in range(len(es)):
        for j in range(i + 1, len(es)):
            a, b = norm(es[i]), norm(es[j])
            if a == b:
                print(f"   EXACT DUP  #{i + 1} == #{j + 1}\n       {a[:100]}")
            elif len(a) > 40 and len(b) > 40:
                r = difflib.SequenceMatcher(None, a, b).ratio()
                if r >= 0.85:
                    print(f"   NEAR DUP   #{i + 1}~#{j + 1} (ratio {r:.2f})\n"
                          f"       A: {a[:100]}\n       B: {b[:100]}")
    for i, e in enumerate(es):
        if STATUS_RE.search(e):
            print(f"   STATUS?    #{i + 1}: {norm(e)[:100]}")
    print()


def main() -> None:
    home = default_home() / "memories"
    ap = argparse.ArgumentParser(
        description="Deterministic dry-run scan of the Hermes memory stores (read-only).")
    ap.add_argument("--mem", default=str(home / "MEMORY.md"))
    ap.add_argument("--user", default=str(home / "USER.md"))
    ap.add_argument("--budget-mem", type=int, default=2200,
                    help="char budget of the memory store (default 2200)")
    ap.add_argument("--budget-user", type=int, default=1375,
                    help="char budget of the user store (default 1375)")
    args = ap.parse_args()
    scan("MEMORY.md", Path(args.mem), args.budget_mem)
    scan("USER.md", Path(args.user), args.budget_user)
    print("Dry-run only. To remove an entry: use the `memory` tool, and append a")
    print("quarantine line to <HERMES_HOME>/mem-quarantine.log BEFORE the change:")
    print("  <date> | <memory|user> | <full text lost> | <reason>")


if __name__ == "__main__":
    main()
