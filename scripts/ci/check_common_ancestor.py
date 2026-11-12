#!/usr/bin/env python3
"""Fail a PR whose branch shares no ancestor with origin/main.

Run by the ``Reject PRs with no common ancestor on main`` step of
``.github/workflows/repo-guards.yml``. PR #25045 was merged from a branch disconnected from
main's history; GitHub's merge UI accepted it and ~1500 files' ``git blame`` collapsed onto its
parent-less root commit. ``git merge-base`` exits non-zero AND prints nothing when the two
commits share no ancestor; both are checked so the message is clear whichever signal fires.
Writes the ``history-check`` review status (scripts/ci/review_status.py).

Exit codes: 0 = common ancestor found, 1 = unrelated histories.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci.review_status import action_required, publish

ARTIFACT = "history-check"
GIT_TIMEOUT = 120
REBASE_STEPS = """\
git fetch origin main
git checkout -b fix-branch origin/main
# re-apply your changes (cherry-pick, copy files, etc.)
git push -f origin fix-branch"""

EXPLANATION = """\
Your branch's history is disconnected from main.  Common causes:
  - the branch was created with 'git checkout --orphan'
  - '.git/' was re-initialized at some point during the work
  - the branch was force-pushed from an unrelated repository

Merging an unrelated-history PR grafts a parent-less root commit
into main and collapses git blame for every file in that snapshot.
Reference: PR #25045 caused this and re-rooted blame on ~1500
files to a single orphan commit.

To fix, rebase your changes onto current main:
""" + "\n".join(f"  {line}" for line in REBASE_STEPS.splitlines())


def merge_base() -> str:
    """The merge base of origin/main and HEAD, or "" when there is none."""
    result = subprocess.run(("git", "merge-base", "origin/main", "HEAD"), capture_output=True,
                            text=True, stdin=subprocess.DEVNULL, timeout=GIT_TIMEOUT, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> int:
    base = merge_base()
    if base:
        print(f"::notice::Common ancestor with main: {base}")
        publish([], ARTIFACT)
        return 0
    publish(action_required("unrelated histories", "Unrelated histories",
                            "This PR has no common ancestor with main.",
                            f"Rebase your changes onto current main:\n```\n{REBASE_STEPS}\n```\n", ""),
            ARTIFACT)
    print()
    print("::error::This PR has no common ancestor with main.")
    print()
    print(EXPLANATION)
    return 1


if __name__ == "__main__":
    sys.exit(main())
