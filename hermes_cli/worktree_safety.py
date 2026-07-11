"""Read-only git safety helpers for task-worktree lifecycle decisions.

REQ-027/REQ-028 (saga decision 0004 — surface-don't-merge, preserve-don't-
stash, ancestry-gated prune). Every helper in this module is strictly
read-only: nothing here mutates a repository, moves refs, or touches the
working tree. Mutation (WIP commits, worktree removal, branch deletion)
lives with the callers so the safety predicates stay trivially auditable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = [
    "is_branch_unmerged",
    "detect_default_branch",
    "is_worktree_dirty",
    "branch_ahead_count",
]

_GIT_TIMEOUT = 30


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
        check=False,
    )


def _local_branch_exists(repo: Path, name: str) -> bool:
    """True iff ``name`` resolves to a commit in ``repo`` (read-only)."""
    result = _run_git(repo, "rev-parse", "--verify", "--quiet", f"{name}^{{commit}}")
    return result.returncode == 0


def is_branch_unmerged(repo: Path, branch: str, base: str = "main") -> bool:
    """True iff ``branch`` has at least one commit not reachable from ``base``.

    - ``False`` when every commit of ``branch`` is an ancestor of (or equal
      to) ``base``; ``branch == base`` is trivially ``False``.
    - Unknown ``branch`` or ``base`` raises :class:`ValueError` naming the
      missing ref.
    - Ancestry is checked via ``git merge-base --is-ancestor`` — pure
      read-only plumbing; the repository is never mutated.
    """
    for ref in (branch, base):
        if not _local_branch_exists(repo, ref):
            raise ValueError(f"unknown ref {ref!r} in repo {repo}")
    if branch == base:
        return False
    result = _run_git(repo, "merge-base", "--is-ancestor", branch, base)
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    stderr = (result.stderr or result.stdout or "").strip()
    raise RuntimeError(
        f"git merge-base --is-ancestor {branch} {base} failed in {repo}: {stderr}"
    )


def detect_default_branch(repo: Path) -> str | None:
    """Return ``"main"`` if it exists, else ``"master"``, else ``None``."""
    for name in ("main", "master"):
        result = _run_git(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{name}")
        if result.returncode == 0:
            return name
    return None


def is_worktree_dirty(wt: Path) -> bool:
    """True iff ``git status --porcelain`` reports anything (incl. untracked)."""
    result = _run_git(wt, "status", "--porcelain")
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"git status failed in {wt}: {stderr}")
    return bool((result.stdout or "").strip())


def branch_ahead_count(repo: Path, branch: str, base: str) -> int:
    """Number of commits on ``branch`` not reachable from ``base``."""
    result = _run_git(repo, "rev-list", "--count", f"{base}..{branch}")
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise ValueError(
            f"git rev-list --count {base}..{branch} failed in {repo}: {stderr}"
        )
    return int((result.stdout or "0").strip())
