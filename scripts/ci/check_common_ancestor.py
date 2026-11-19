#!/usr/bin/env python3
"""Fail a PR whose branch shares no ancestor with origin/main, or carries a foreign root.

Run by the ``Reject PRs with no common ancestor on main`` step of
``.github/workflows/repo-guards.yml``. PR #25045 was merged from a branch disconnected from
main's history; GitHub's merge UI accepted it and ~1500 files' ``git blame`` collapsed onto its
parent-less root commit. ``git merge-base`` exits non-zero AND prints nothing when the two
commits share no ancestor; both are checked so the message is clear whichever signal fires.
Writes the ``history-check`` review status (scripts/ci/review_status.py).

The PR head is ``$_HERMES_PR_HEAD_SHA`` (``github.event.pull_request.head.sha``), never ``HEAD``:
a pull_request checkout is the synthetic ``refs/pull/N/merge`` commit, whose first parent is
main's tip, so ``merge-base origin/main HEAD`` always succeeds there (#128930). A head that merged
an unrelated branch still shares an ancestor with main, so every root commit of the head must
also already be in main's history. Only when the variable is unset (a local run) is ``HEAD``
the head; set but empty or unresolvable fails closed.

Exit codes: 0 = common ancestor found and no foreign root, 1 = unrelated histories or the PR
head could not be resolved.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
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


FETCH_ATTEMPTS = 3


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("git", *args), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", stdin=subprocess.DEVNULL, timeout=GIT_TIMEOUT, check=False)


def _has_commit(sha: str) -> bool:
    return _git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def pr_head() -> str:
    """The PR head commit, fetched when the checkout lacks it; "" when it cannot be resolved."""
    sha = os.environ.get("_HERMES_PR_HEAD_SHA")
    if sha is None:
        sha = "HEAD"
    sha = sha.strip()
    if not sha or sha.startswith("-"):
        return ""
    for attempt in range(FETCH_ATTEMPTS):
        if _has_commit(sha):
            break
        if attempt:
            time.sleep(2)
        _git("fetch", "--no-tags", "origin", sha)
    result = _git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}")
    return result.stdout.strip() if result.returncode == 0 else ""


def merge_base(head: str) -> str:
    """The merge base of origin/main and the PR head, or "" when there is none."""
    result = _git("merge-base", "origin/main", head)
    return result.stdout.strip() if result.returncode == 0 else ""


def foreign_roots(head: str) -> list[str] | None:
    """Root commits of the head that main's history lacks; None when git cannot list them."""
    result = _git("rev-list", "--max-parents=0", head)
    if result.returncode != 0:
        return None
    return [root for root in result.stdout.split()
            if _git("merge-base", "--is-ancestor", root, "origin/main").returncode != 0]


def _fail(title: str, summary: str, detail: str = "") -> int:
    publish(action_required("unrelated histories", title, summary,
                            f"Rebase your changes onto current main:\n```\n{REBASE_STEPS}\n```\n", detail),
            ARTIFACT)
    print()
    print(f"::error::{summary}")
    print()
    print(EXPLANATION)
    return 1


def main() -> int:
    head = pr_head()
    if not head:
        sha = os.environ.get("_HERMES_PR_HEAD_SHA", "HEAD")
        publish(action_required("unrelated histories", "PR head unavailable",
                                f"The history check could not resolve the PR head {sha!r}.",
                                "Re-run the job; the check cannot pass without the PR head.", ""),
                ARTIFACT)
        print(f"::error::Could not resolve the PR head {sha!r}; the history check cannot run.")
        return 1
    base = merge_base(head)
    if not base:
        return _fail("Unrelated histories", "This PR has no common ancestor with main.")
    roots = foreign_roots(head)
    if roots is None:
        return _fail("Unrelated histories", "Could not list the root commits of this PR.")
    if roots:
        return _fail("Unrelated history merged into this PR",
                     "This PR brings in a root commit that is not part of main's history.",
                     "\n".join(roots))
    print(f"::notice::Common ancestor with main: {base}")
    publish([], ARTIFACT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
