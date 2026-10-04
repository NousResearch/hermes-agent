#!/usr/bin/env python3
"""Detect a branch that silently reverts landed work on the target.

"This merge undid a landed change" is a property of the HISTORY, not the tree, so no green
test suite can catch it: a file reverted to an older version is still a valid file whose
older tests still pass. The failure mode is common with parallel agents and stale branches:
a branch was cut before commit X landed, edits a file X also touched, and its version of the
file is byte-for-byte the pre-X blob. Git sees an ordinary edit (often with the landing
merge as a direct parent, so nothing to three-way against); the reviewer sees a plausible
diff; commit X is gone. AGENTS.md § "Commits, Merges, PRs" describes the same class for
squash merges from stale branches.

Rule, per file the branch changes: if the branch's blob for that file equals an OLDER blob
of that file on the target — one the target has already moved PAST (the target's current
blob appears more recently in the file's history) — then merging would undo that landed
change. Ordinary forward work never satisfies this. Ported from gastownhall/gastown#4840
(refinery silent-revert check).

Stated limit: this catches a branch whose file content matches a superseded version byte
for byte. A stale copy that also picked up unrelated edits matches no historical blob and
slips through. Disposition is halt-and-explain, not reject: a deliberate revert is legitimate
work, so this is advisory by default and ``--strict`` makes it a hard gate.

Usage:
    python scripts/ci/check_silent_revert.py [--base origin/main] [--head HEAD] [--depth 80] [--strict]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass

DEFAULT_DEPTH = 80
MAX_WALK = 5000  # commits; main gains ~170/day, so this is roughly a month of first-parent history


@dataclass
class RevertFinding:
    path: str
    branch_blob: str
    target_blob: str
    superseded_commit: str
    superseded_subject: str

    def __str__(self) -> str:
        return (
            f"{self.path}: branch blob {self.branch_blob[:12]} is the superseded version from "
            f"{self.superseded_commit[:12]} ({self.superseded_subject}); target is now at "
            f"{self.target_blob[:12]}, so merging would undo that"
        )


def _git(*args: str, cwd: str | None = None) -> str | None:
    r = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd)
    return r.stdout.strip() if r.returncode == 0 else None


def check_for_reverts(target: str, branch: str, depth: int = DEFAULT_DEPTH, cwd: str | None = None) -> list[RevertFinding]:
    """Files changed on ``branch`` (vs merge-base with ``target``) whose branch blob is a superseded target blob."""
    depth = depth if depth > 0 else DEFAULT_DEPTH
    changed = _git("diff", "--name-only", "--diff-filter=ACMR", f"{target}...{branch}", cwd=cwd) or ""
    paths = [p.strip() for p in changed.splitlines() if p.strip()]
    branch_blobs, target_blobs = _blobs(branch, paths, cwd), _blobs(target, paths, cwd)
    blobs = {p: (branch_blobs.get(p), target_blobs.get(p)) for p in paths}
    # Absent on one side (a new file cannot revert anything) or identical to the target (not a change at all).
    candidates = sorted(p for p, (b, t) in blobs.items() if b and t and b != t)
    if not candidates:
        return []
    histories = _file_histories(target, candidates, depth, cwd)
    findings: list[RevertFinding] = []
    for path in candidates:
        branch_blob, target_blob = blobs[path]
        seen_current = False
        for commit, subject, hist_blob in histories.get(path, []):
            if hist_blob == target_blob:
                seen_current = True
                continue
            if seen_current and hist_blob == branch_blob:
                findings.append(RevertFinding(path, branch_blob, target_blob, commit, subject))
                break
    return findings


def _blobs(rev: str, paths: list[str], cwd: str | None) -> dict[str, str]:
    """``{path: blob}`` at ``rev`` for the given paths in one ``ls-tree`` (a rev-parse per path is a process each)."""
    out = _git("ls-tree", "-z", rev, "--", *paths, cwd=cwd) or ""
    blobs: dict[str, str] = {}
    for entry in out.split("\0"):
        if not entry:
            continue
        meta, _, path = entry.partition("\t")
        blobs[path] = meta.split()[2]
    return blobs


def _file_histories(target: str, paths: list[str], depth: int, cwd: str | None) -> dict[str, list[tuple[str, str, str]]]:
    """Per path, ``(commit, subject, blob-after-commit)`` for up to ``depth`` commits touching it on ``target``,
    newest first. ONE ``git log --raw`` walk over all paths (a per-file walk costs ~0.5 s each on this repo, so a
    300-file refactor would take minutes); ``--first-parent -m`` makes a non-squash merge that lands a blob show it.
    The walk is bounded at ``depth * len(paths)`` commits (capped at ``MAX_WALK``), so a path's own history may be
    shorter than ``depth`` when other paths churn far more; the check stays sound (fewer commits inspected), never
    wrong."""
    out = _git("log", f"-n{min(depth * len(paths), MAX_WALK)}", "--format=%x00%H %s", "--raw", "--no-abbrev", "--no-renames",
               "--first-parent", "-m", target, "--", *paths, cwd=cwd) or ""
    wanted = set(paths)
    histories: dict[str, list[tuple[str, str, str]]] = {}
    for block in out.split("\x00")[1:]:
        lines = block.strip().splitlines()
        if not lines:
            continue
        commit, _, subject = lines[0].partition(" ")
        for raw in lines[1:]:
            if not raw.startswith(":"):
                continue
            path = raw.split("\t", 1)[-1]
            if path in wanted and len(hist := histories.setdefault(path, [])) < depth:
                hist.append((commit, subject, raw.split()[3]))
    return histories


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--base", default="origin/main", help="target ref the branch would merge into")
    ap.add_argument("--head", default="HEAD", help="branch ref to check")
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="commits of per-file target history to scan")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any finding")
    args = ap.parse_args(argv)
    for ref in (args.base, args.head):
        if _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}") is None:
            print(f"silent-revert: cannot resolve ref {ref!r} (fetch it first); refusing to report a clean diff", file=sys.stderr)
            return 2
    if not _git("merge-base", args.base, args.head):
        print(f"silent-revert: no merge-base between {args.base!r} and {args.head!r}", file=sys.stderr)
        return 2
    findings = check_for_reverts(args.base, args.head, args.depth)
    print(f"silent-revert: {len(findings)} file(s) on {args.head} match a version {args.base} has already moved past")
    for f in findings:
        print(f"  {f}")
    if findings:
        print(
            "  A stale branch carrying the pre-landing copy of a file undoes the landed commit without a conflict "
            "and without a red test. Rebase onto the target and re-apply the intended edit; a deliberate revert "
            "should say so in the commit message."
        )
    return 1 if (args.strict and findings) else 0


if __name__ == "__main__":
    sys.exit(main())
