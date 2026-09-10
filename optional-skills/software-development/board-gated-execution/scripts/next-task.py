#!/usr/bin/env python3
"""
The execution gate: what am I allowed to work on right now?

WHY THIS EXISTS

A board that is not consulted is just a nicer place to lose work. The failure it
prevents is concrete: a feature reaches "service running, route written, UI
missing, nothing committed" while attention moves to an adjacent question, and
the half-built state stays invisible until someone asks "did you finish that?".

So the rule is mechanical rather than a good intention:

    Before starting anything, run this. Work the issue it names. If what you
    are about to do is not that issue, either it is not the next thing, or it
    is not on the board — and unboarded work is how drift starts.

SELECTION ORDER

  1. WIP LIMIT FIRST. More than one unblocked `in-progress` issue is itself the
     defect; the gate refuses to name new work until the count is back to one.
  2. Anything `in-progress` that is not `blocked:human` — finish before starting.
  3. Otherwise the highest priority (P0 > P1 > P2 > P3) unblocked issue.

`blocked:human` issues are never selected: they need a decision or a credential
the agent cannot supply, so "working" them would mean waiting, and waiting looks
identical to drift. Report them to the user instead.

USAGE

    python3 next-task.py                 # repo inferred from the git remote
    python3 next-task.py owner/repo      # explicit

EXIT CODES
  0  an issue was selected, or the board holds no actionable work
  1  the gate refuses: WIP limit breached, resolve that first
  2  the board could not be read (auth, network, no remote)
"""
from __future__ import annotations

import json
import subprocess
import sys

PRIORITY = ["P0", "P1", "P2", "P3"]


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=180)


def detect_repo() -> str:
    """`owner/repo` from the current checkout, so the script is not repo-specific."""
    p = run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"])
    if p.returncode != 0 or not p.stdout.strip():
        print("Cannot determine the repository. Pass it explicitly: "
              "python3 next-task.py owner/repo", file=sys.stderr)
        sys.exit(2)
    return p.stdout.strip()


def fetch_issues(repo: str) -> list[dict]:
    p = run(["gh", "issue", "list", "--repo", repo, "--state", "open",
             "--limit", "100", "--json", "number,title,labels"])
    if p.returncode != 0:
        print(f"Could not read the board: {p.stderr.strip()[:200]}", file=sys.stderr)
        sys.exit(2)
    return json.loads(p.stdout or "[]")


def labels_of(issue: dict) -> set[str]:
    return {l["name"] for l in issue.get("labels", [])}


def rank(issue: dict) -> int:
    names = labels_of(issue)
    for i, p in enumerate(PRIORITY):
        if p in names:
            return i
    return len(PRIORITY)  # unprioritised sorts last


def describe(issue: dict, indent: str = "") -> str:
    names = ", ".join(sorted(labels_of(issue)))
    return f"{indent}#{issue['number']} {issue['title']}\n{indent}  labels: {names}"


def main(argv: list[str]) -> int:
    repo = argv[1] if len(argv) > 1 else detect_repo()
    issues = fetch_issues(repo)
    if not issues:
        print(f"{repo}: the board is empty. Nothing is tracked — that is itself "
              f"the problem. Seed it before starting work.")
        return 0

    blocked = [i for i in issues if "blocked:human" in labels_of(i)]
    active = [i for i in issues
              if "in-progress" in labels_of(i) and "blocked:human" not in labels_of(i)]

    # 1. WIP limit. Two things in flight means one of them is drifting.
    if len(active) > 1:
        print("GATE: REFUSED — more than one issue is in progress.\n")
        for i in active:
            print(describe(i, "  "))
        print("\nFinish or explicitly park one before starting anything new.")
        return 1

    # 2. Finish what is already started.
    if active:
        chosen = active[0]
        print("WORK THIS (already in progress — finish before starting anything new):\n")
        print(describe(chosen))
    else:
        candidates = [i for i in issues
                      if "blocked:human" not in labels_of(i)
                      and "in-progress" not in labels_of(i)]
        if not candidates:
            print("Nothing actionable: every open issue is blocked on a human decision.\n")
            for i in blocked:
                print(describe(i, "  "))
            print("\nSay so plainly rather than inventing adjacent work.")
            return 0
        candidates.sort(key=lambda i: (rank(i), i["number"]))
        chosen = candidates[0]
        print("WORK THIS (highest priority unblocked):\n")
        print(describe(chosen))
        print(f"\n  Mark it started:  gh issue edit {chosen['number']} "
              f"--repo {repo} --add-label in-progress")

    if blocked:
        print(f"\nBLOCKED ON A HUMAN ({len(blocked)}) — surface these, do not work them:")
        for i in blocked:
            print(describe(i, "  "))

    remaining = [i for i in issues
                 if i["number"] != chosen["number"]
                 and "blocked:human" not in labels_of(i)]
    if remaining:
        remaining.sort(key=lambda i: (rank(i), i["number"]))
        print(f"\nQUEUED ({len(remaining)}):")
        for i in remaining[:5]:
            pri = next((p for p in PRIORITY if p in labels_of(i)), "--")
            print(f"  {pri:3} #{i['number']} {i['title'][:60]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
