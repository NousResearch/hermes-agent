"""A linked git worktree's activity lives OUTSIDE its own subtree.

Commits, resets and checkouts inside a linked worktree write to
``<repo>/.git/worktrees/<name>/`` — the entry itself only gains files when the
checkout is created. So a worktree nobody has *checked out* into for half an
hour looks idle to a subtree walk even while a session is committing in it,
and the pruner then deletes a live tree (and prunes its registration, which
makes the loss invisible to ``git worktree list``).

These cases pin both directions: recent gitdir activity must keep the entry,
and a worktree with an old gitdir must still be pruned (otherwise the fix
would be an exemption, not an idle clock).
"""

import os
import time
from pathlib import Path

from hermes_constants_scratch import prune_idle_entries

IDLE_HOURS = 0.5
OLD = 3 * 3600


def _age(path: Path, seconds: int = OLD) -> None:
    stamp = time.time() - seconds
    for root, dirs, files in os.walk(path):
        for name in files + dirs:
            try:
                os.utime(os.path.join(root, name), (stamp, stamp))
            except OSError:
                pass
    os.utime(path, (stamp, stamp))


def _make_linked_worktree(tmp_path: Path, gitdir_recently_touched: bool) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    gitdir = repo / ".git" / "worktrees" / "wt-1"
    gitdir.mkdir(parents=True)
    root = tmp_path / "scratch"
    entry = root / "wt-1"
    (entry / "pkg").mkdir(parents=True)
    (entry / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (entry / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    _age(entry)
    (gitdir / "index").write_text("idx\n", encoding="utf-8")
    if not gitdir_recently_touched:
        os.utime(gitdir / "index", (time.time() - OLD, time.time() - OLD))
    # The gitdir directory itself must not leak activity either: a fresh
    # directory mtime would mask whether the branch under test is the assert.
    os.utime(gitdir, ((time.time() - OLD if not gitdir_recently_touched else time.time()),) * 2)
    return root, entry


def test_recent_worktree_activity_keeps_entry(tmp_path):
    root, entry = _make_linked_worktree(tmp_path, gitdir_recently_touched=True)
    removed = prune_idle_entries(root, IDLE_HOURS, frozenset())
    assert removed == 0
    assert entry.exists(), "a worktree whose gitdir was just written was pruned"


def test_idle_worktree_entry_still_pruned(tmp_path):
    root, entry = _make_linked_worktree(tmp_path, gitdir_recently_touched=False)
    removed = prune_idle_entries(root, IDLE_HOURS, frozenset())
    assert removed == 1
    assert not entry.exists(), "an idle worktree must still be reclaimed"
