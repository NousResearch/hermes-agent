"""The one place a git worktree is removed.

Before this module, five call sites each ran their own ``git worktree remove``
with their own preconditions. On Windows every one of them was capable of the
same data loss: ``git worktree remove`` follows a **junction** as if it were an
ordinary directory and deletes THROUGH it, and on 2026-09-06 that emptied two
sibling source repositories reached via a ``node_modules`` junction chain (see
``hermes_cli/reparse_guard.py`` for the incident and why ``os.path.islink``
cannot see it).

Fixing five copies of a deletion policy leaves five copies to drift. This is the
single owner instead: callers state their *policy* — may I remove this, and may
I force — and the physical removal is performed once, here, by a walker that
classifies every edge before traversing it.

The removal has two stages and the order is the point:

1. ``reparse_guard.remove_tree`` deletes the tree itself, unlinking any reparse
   point rather than descending through it, and refusing outright if any edge
   cannot be classified. Nothing is followed.
2. ``git worktree prune`` then drops the now-dangling admin entry. ``git
   worktree remove`` is never asked to do the deleting, because its recursive
   delete is precisely the unsafe traversal.

That ordering also removes the check/use gap the old shape had: there is no
window in which a scan says "clean" and a *different* deleter then walks the
tree. The traversal that classifies an edge is the traversal that deletes it.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from hermes_cli.reparse_guard import UnknownEdge, remove_tree

logger = logging.getLogger("cli")


@dataclass(frozen=True)
class RemovalOutcome:
    """What happened, in terms a caller can log or show a user."""

    removed: bool
    reason: str = ""
    """Empty when removed. Otherwise names what stopped it, including the path."""

    def __bool__(self) -> bool:  # `if remove_worktree(...):`
        return self.removed


def _git(args: Sequence[str], cwd: str, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, cwd=cwd, check=False,
    )


def remove_worktree(
    repo_root: str,
    worktree_path: str | Path,
    branch: Optional[str] = None,
    *,
    unlock: bool = True,
) -> RemovalOutcome:
    """Remove *worktree_path* and, when given, delete *branch*.

    The caller owns the *policy* (dirty checks, age tiers, whether the branch is
    theirs to delete) and has already decided the tree is disposable. This
    function owns the *mechanics*, and refuses whenever the tree cannot be
    deleted without traversing something it did not classify.

    Never raises: a failure is a ``RemovalOutcome`` whose ``reason`` names the
    offending path. Callers preserve the worktree on a refusal — a stale
    worktree costs disk, and deleting through a junction costs a source tree.
    """
    wt = Path(worktree_path)

    if unlock:
        # A lock left by creation (or a dead worktree's own lock) blocks the
        # admin-entry prune below; the physical delete does not care.
        try:
            _git(["worktree", "unlock", str(wt)], repo_root, timeout=10)
        except Exception as exc:
            logger.debug("worktree unlock failed (non-fatal) for %s: %s", wt, exc)

    if wt.exists():
        try:
            remove_tree(wt)
        except UnknownEdge as exc:
            # The whole point: an edge we could not classify might be the very
            # junction this guard exists to stop, so nothing is deleted.
            return RemovalOutcome(
                False,
                f"could not classify {exc.path} ({exc.cause}) — refusing to "
                f"delete, because traversing an unclassified entry can destroy "
                f"whatever it points at",
            )
        except OSError as exc:
            return RemovalOutcome(False, f"{type(exc).__name__} removing {wt}: {exc}")

    # The directory is gone; drop the admin entry that still names it. `prune`
    # is the dirless-case counterpart of `remove` and does no filesystem walk.
    try:
        _git(["worktree", "prune"], repo_root, timeout=15)
    except Exception as exc:
        logger.debug("worktree prune failed (non-fatal) after removing %s: %s", wt, exc)

    if branch:
        try:
            _git(["branch", "-D", branch], repo_root, timeout=15)
        except Exception as exc:
            logger.debug("failed to delete branch %s: %s", branch, exc)

    return RemovalOutcome(True)
