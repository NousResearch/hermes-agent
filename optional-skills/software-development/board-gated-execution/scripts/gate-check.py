#!/usr/bin/env python3
"""
Prove the execution gate refuses.

A gate that only ever says "go" is decoration. The behaviour that matters is the
refusal: when two issues are in progress, the gate must stop and say so rather
than helpfully picking one — because helpfully picking one is exactly how a
half-built feature gets abandoned for a fresher, more interesting task.

This mutates the board (adds `in-progress` to one issue), asserts the gate
refuses, and restores the label in a `finally` block so the board is left as
found even if an assertion fails.

    python3 gate-check.py                      # repo inferred, victim auto-picked
    python3 gate-check.py owner/repo 42        # explicit repo and victim issue
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

GATE = Path(__file__).with_name("next-task.py")


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=180)


def detect_repo() -> str:
    p = run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"])
    if p.returncode != 0 or not p.stdout.strip():
        print("Cannot determine the repository.", file=sys.stderr)
        sys.exit(2)
    return p.stdout.strip()


def gate(repo: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(GATE), repo],
                          capture_output=True, text=True, timeout=300)


def pick_victim(repo: str) -> str | None:
    """An open issue that is neither in-progress nor blocked — safe to relabel."""
    p = run(["gh", "issue", "list", "--repo", repo, "--state", "open",
             "--limit", "100", "--json", "number,labels"])
    if p.returncode != 0:
        return None
    for issue in json.loads(p.stdout or "[]"):
        names = {l["name"] for l in issue.get("labels", [])}
        if "in-progress" not in names and "blocked:human" not in names:
            return str(issue["number"])
    return None


def main(argv: list[str]) -> int:
    repo = argv[1] if len(argv) > 1 else detect_repo()
    victim = argv[2] if len(argv) > 2 else pick_victim(repo)
    if not victim:
        print("No safe issue to use for the refusal test. Seed the board first.")
        return 2

    checks: list[tuple[str, bool]] = []

    before = gate(repo)
    checks.append(("gate runs cleanly before the mutation", before.returncode == 0))
    checks.append(("gate names exactly one task or reports none actionable",
                   "WORK THIS" in before.stdout or "Nothing actionable" in before.stdout))
    if "BLOCKED ON A HUMAN" in before.stdout:
        head = before.stdout.split("BLOCKED ON A HUMAN")[0]
        checks.append(("gate never selects a blocked issue as the task",
                       "blocked:human" not in head))

    try:
        run(["gh", "issue", "edit", victim, "--repo", repo, "--add-label", "in-progress"])
        during = gate(repo)
        checks.append(("gate REFUSES with two in-progress issues", during.returncode == 1))
        checks.append(("refusal explains why",
                       "more than one issue is in progress" in during.stdout))
        checks.append((f"refusal names the offender #{victim}", f"#{victim}" in during.stdout))
        checks.append(("refusal declines to name a task",
                       "WORK THIS" not in during.stdout))
    finally:
        run(["gh", "issue", "edit", victim, "--repo", repo, "--remove-label", "in-progress"])

    after = gate(repo)
    checks.append(("gate recovers once the WIP limit is restored", after.returncode == 0))

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
