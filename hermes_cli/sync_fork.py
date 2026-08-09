"""Merge upstream into a *diverged* fork.

``hermes update`` refuses this case on purpose. ``_sync_upstream`` compares
``origin/main`` with ``upstream/main`` and bails out when the fork carries
commits upstream does not have::

    if origin_ahead > 0:
        print(f"ℹ Your fork has {origin_ahead} commit(s) not on upstream.")
        print("  Skipping upstream sync to preserve your changes.")
        return

That refusal is right for what it does — the sync is a ``pull --ff-only``, and a
fast-forward cannot express a merge — but it leaves every fork that has ever
committed with no supported update path, and the gap is silent: the skipped
fast-forward is the only upstream check ``update`` performs, so the run still
reports success while the fork drifts arbitrarily far behind.

This module merges instead of fast-forwarding, and does it **incrementally**.
Merging a large backlog in one jump produces a single conflict spanning the
whole range, where the three-way merge base is so old that git can drop
definitions the merged text still references — a runtime ``NameError`` rather
than a conflict marker. Stopping at each upstream commit that touches a file the
fork also modified keeps every conflict attributable to one commit.

Every failure path restores the pre-merge ``HEAD`` and leaves the worktree
clean: a half-merged checkout is worse than no update, because the next run
cannot tell it apart from local work.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_UPSTREAM_REF = "upstream/main"

# A merge writes a commit, so git demands a committer identity. On a host with
# no global git identity — a fresh container, a service account, CI — the merge
# fails with "Committer identity unknown" and *no* conflicted paths. Reporting
# that as a conflict sends the user hunting for a merge conflict that does not
# exist, so supply an identity for the merge commit only, without writing to the
# user's config. A real configured identity always wins: -c is overridden by the
# repo/global config when one is set.
_IDENTITY: tuple[str, ...] = (
    "-c",
    "user.name=hermes sync-fork",
    "-c",
    "user.email=sync-fork@hermes.local",
)


@dataclass
class ForkState:
    """How the fork sits relative to upstream."""

    behind: int
    ahead: int
    diverged: bool
    dirty: bool


@dataclass
class SyncPlan:
    """Ordered refs to merge, ending at the upstream tip."""

    steps: list[str] = field(default_factory=list)
    conflict_prone: list[str] = field(default_factory=list)


@dataclass
class SyncResult:
    ok: bool
    reason: str = ""
    merged: list[str] = field(default_factory=list)
    conflicted_paths: list[str] = field(default_factory=list)
    failed_at: str = ""


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _out(cwd: Path, *args: str) -> str:
    try:
        return _git(cwd, *args).stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def _count(cwd: Path, base: str, head: str) -> int:
    raw = _out(cwd, "rev-list", "--count", f"{base}..{head}")
    try:
        return int(raw)
    except ValueError:
        return 0


def inspect(cwd: Path, upstream_ref: str = DEFAULT_UPSTREAM_REF) -> ForkState:
    """Report the fork's position without touching anything."""
    behind = _count(cwd, "HEAD", upstream_ref)
    ahead = _count(cwd, upstream_ref, "HEAD")
    return ForkState(
        behind=behind,
        ahead=ahead,
        diverged=bool(behind and ahead),
        dirty=bool(_out(cwd, "status", "--porcelain")),
    )


def _local_files(cwd: Path, upstream_ref: str) -> set[str]:
    """Files the fork changed relative to the merge base.

    These are the only files where an upstream commit can conflict, so they
    decide where the plan stops.
    """
    base = _out(cwd, "merge-base", "HEAD", upstream_ref)
    if not base:
        return set()
    listing = _out(cwd, "diff", "--name-only", f"{base}..HEAD")
    return {line for line in listing.splitlines() if line}


def plan(cwd: Path, upstream_ref: str = DEFAULT_UPSTREAM_REF) -> SyncPlan:
    """Build the incremental merge sequence.

    One step per upstream commit that touches a file the fork also modified,
    plus a final step at the tip. A fork that shares no touched files gets a
    single step and behaves exactly like a plain merge.
    """
    if _count(cwd, "HEAD", upstream_ref) == 0:
        return SyncPlan(steps=[], conflict_prone=[])

    touched = _local_files(cwd, upstream_ref)
    steps: list[str] = []
    prone: list[str] = []

    if touched:
        base = _out(cwd, "merge-base", "HEAD", upstream_ref)
        revs = _out(cwd, "rev-list", "--reverse", f"{base}..{upstream_ref}")
        for sha in (r for r in revs.splitlines() if r):
            files = _out(cwd, "show", "--name-only", "--format=", sha).splitlines()
            if touched.intersection(f for f in files if f):
                steps.append(sha)
                prone.append(sha)

    steps.append(upstream_ref)
    return SyncPlan(steps=steps, conflict_prone=prone)


def _abort_merge(cwd: Path, original_head: str) -> None:
    """Return the repo to ``original_head`` with a clean worktree.

    ``git merge --abort`` alone is not enough: if anything rewrote a conflicted
    file outside the index, abort fails with ``Entry '<path>' not uptodate``.
    Refreshing the index stat cache first clears that, and the hard reset is the
    backstop so this function cannot leave a half-merged tree behind.
    """
    _git(cwd, "update-index", "-q", "--refresh", check=False)
    _git(cwd, "merge", "--abort", check=False)
    if (cwd / ".git" / "MERGE_HEAD").exists():
        _git(cwd, "reset", "--hard", original_head, check=False)
    if _out(cwd, "rev-parse", "HEAD") != original_head:
        _git(cwd, "reset", "--hard", original_head, check=False)


def sync(
    cwd: Path,
    upstream_ref: str = DEFAULT_UPSTREAM_REF,
    dry_run: bool = False,
) -> SyncResult:
    """Merge ``upstream_ref`` into the current branch, one planned step at a time.

    Returns without touching the repo when the worktree is dirty: stashing
    someone else's uncommitted work as a side effect of an update is how work
    gets lost.
    """
    state = inspect(cwd, upstream_ref)
    if state.dirty:
        return SyncResult(
            ok=False,
            reason=(
                "Refusing to sync with uncommitted changes. Commit or stash "
                "them first — an update must never decide what happens to "
                "unsaved work."
            ),
        )
    if state.behind == 0:
        return SyncResult(ok=True, reason="Already up to date with upstream.")

    steps = plan(cwd, upstream_ref).steps
    if dry_run:
        return SyncResult(
            ok=True,
            reason=f"{state.behind} commit(s) behind; would merge in {len(steps)} step(s).",
            merged=steps,
        )

    original_head = _out(cwd, "rev-parse", "HEAD")
    merged: list[str] = []

    for step in steps:
        proc = _git(
            cwd,
            *_IDENTITY,
            "merge",
            "--no-edit",
            "-m",
            f"merge: upstream {step}",
            step,
            check=False,
        )
        if proc.returncode != 0:
            conflicted = [
                line
                for line in _out(cwd, "diff", "--name-only", "--diff-filter=U").splitlines()
                if line
            ]
            _abort_merge(cwd, original_head)
            if conflicted:
                reason = (
                    f"Conflict merging {step}. The fork was restored to "
                    f"{original_head[:9]} with a clean worktree; resolve this "
                    "step by hand."
                )
            else:
                # No unmerged paths means git refused for some other reason
                # (bad ref, unrelated histories, hook rejection). Surface its
                # own words rather than calling it a conflict.
                detail = (proc.stderr or proc.stdout or "").strip().splitlines()
                reason = (
                    f"git could not merge {step}: "
                    f"{detail[0] if detail else 'unknown error'}. "
                    f"The fork was restored to {original_head[:9]}."
                )
            return SyncResult(
                ok=False,
                reason=reason,
                merged=merged,
                conflicted_paths=conflicted,
                failed_at=step,
            )
        merged.append(step)

    return SyncResult(ok=True, reason=f"Merged upstream in {len(merged)} step(s).", merged=merged)
