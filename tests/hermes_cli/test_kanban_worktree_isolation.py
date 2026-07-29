"""Per-task worktree isolation for decompose siblings.

Decompose children used to inherit the root's literal ``workspace_path``,
so every sibling of a worktree-kind root pointed at the SAME checkout —
and ``_resolve_worktree_workspace``'s existing-checkout shortcut reused it
on whatever branch was there, letting sibling workers run concurrently in
one directory on one branch (cross-task provenance corruption, no lock).

Two-part fix under test:
- ``decompose_triage_task`` leaves worktree children's ``workspace_path``
  unset so each child materializes its own ``<repo>/.worktrees/<child-id>``.
- ``_resolve_worktree_workspace`` falls back to a fresh per-task worktree
  when the requested path is occupied by another task's branch (heals
  pre-existing rows that still carry a shared path).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _add_worktree(repo: Path, target: Path, branch: str) -> Path:
    _git(repo, "worktree", "add", str(target), "-b", branch, "HEAD")
    return target


def test_decompose_worktree_children_get_own_workspace(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="build the feature", triage=True)
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', "
            "workspace_path='/repo/.worktrees/root' WHERE id = ?",
            (root,),
        )
        conn.commit()

        child_ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[
                {"title": "spec it", "assignee": "alice", "parents": []},
                {"title": "implement it", "assignee": "bob", "parents": [0]},
            ],
            author="decomposer",
        )
        assert child_ids is not None and len(child_ids) == 2

        for cid in child_ids:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row["workspace_kind"] == "worktree"
            # Each child resolves its own <repo>/.worktrees/<child-id> at
            # dispatch; the root's literal path must never be shared.
            assert row["workspace_path"] is None


def test_decompose_dir_children_still_inherit_path(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="ops sweep", triage=True)
        conn.execute(
            "UPDATE tasks SET workspace_kind='dir', "
            "workspace_path='/srv/ops' WHERE id = ?",
            (root,),
        )
        conn.commit()

        child_ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[{"title": "child", "assignee": "alice", "parents": []}],
            author="decomposer",
        )
        assert child_ids is not None
        row = conn.execute(
            "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
            (child_ids[0],),
        ).fetchone()
        assert row["workspace_kind"] == "dir"
        assert row["workspace_path"] == "/srv/ops"


def test_resolve_worktree_falls_back_when_path_occupied(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="second sibling",
            workspace_kind="worktree",
            workspace_path=str(occupied),  # inherited shared/stale path
        )
        task = kb.get_task(conn, tid)

    workspace, branch = kb._resolve_worktree_workspace(task)
    assert workspace == (repo / ".worktrees" / tid).resolve()
    assert branch == f"wt/{tid}"
    # The sibling's checkout is untouched, still on its own branch.
    assert (occupied / "README.md").exists()
    head = subprocess.run(
        ["git", "-C", str(occupied), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == "wt/sibling"


def test_resolve_worktree_same_branch_still_reuses(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="returning task",
            workspace_kind="worktree",
        )
        own = _add_worktree(repo, repo / ".worktrees" / tid, f"wt/{tid}")
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?",
            (str(own), tid),
        )
        conn.commit()
        task = kb.get_task(conn, tid)

    workspace, branch = kb._resolve_worktree_workspace(task)
    assert workspace == own.resolve()
    assert branch == f"wt/{tid}"


def test_resolve_worktree_own_path_on_foreign_branch_keeps_legacy_reuse(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="foreign-branch checkout",
            workspace_kind="worktree",
        )
        own = _add_worktree(repo, repo / ".worktrees" / tid, "wt/foreign")
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?",
            (str(own), tid),
        )
        conn.commit()
        task = kb.get_task(conn, tid)

    # The fallback target would be the occupied path itself, so the
    # legacy reuse applies rather than failing dispatch.
    workspace, branch = kb._resolve_worktree_workspace(task)
    assert workspace == own.resolve()
    assert branch == "wt/foreign"


# --- _ensure_git_worktree: stale-branch realign on reuse ----------------------
#
# ``_ensure_git_worktree`` reuses ``target`` whenever it is a linked worktree of
# the same repo -- WITHOUT checking which branch it is on. A previous failed or
# interrupted dispatch of the same task leaves the canonical
# ``<repo>/.worktrees/<id>`` path on a stale branch, so the next dispatch of
# that task silently runs on the wrong branch. The per-task fallback in
# ``_resolve_worktree_workspace`` routes right back to this same canonical path,
# so it cannot heal the case either -- this is the last gap.


def test_ensure_git_worktree_realigns_a_stale_reused_branch(tmp_path):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "t_abc"
    # A previous dispatch left the canonical path on some other branch.
    _add_worktree(repo, target, "wt/stale-previous-run")
    assert kb._git_current_branch(target) == "wt/stale-previous-run"

    kb._ensure_git_worktree(repo, target, "wt/t_abc")

    assert kb._git_current_branch(target) == "wt/t_abc", (
        "a reused worktree left on a stale branch must be realigned to the "
        "task's own branch, not silently run on the previous run's branch"
    )


def test_ensure_git_worktree_same_branch_reuse_is_a_noop(tmp_path):
    """Control: the common path (already on the right branch) must not churn."""
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "t_abc"
    _add_worktree(repo, target, "wt/t_abc")
    (target / "work-in-progress.txt").write_text("uncommitted\n", encoding="utf-8")

    kb._ensure_git_worktree(repo, target, "wt/t_abc")

    assert kb._git_current_branch(target) == "wt/t_abc"
    # No checkout was issued, so uncommitted work is untouched.
    assert (target / "work-in-progress.txt").exists()


def test_ensure_git_worktree_realign_reuses_an_existing_branch(tmp_path):
    """Realigning onto a branch that already exists must check it out, not
    fail trying to create a duplicate."""
    repo = _make_repo(tmp_path)
    other = repo / ".worktrees" / "other"
    _add_worktree(repo, other, "wt/t_target")   # branch exists, checked out elsewhere
    _git(other, "checkout", "-b", "wt/parked")  # free the branch again

    target = repo / ".worktrees" / "t_target"
    _add_worktree(repo, target, "wt/stale")

    kb._ensure_git_worktree(repo, target, "wt/t_target")
    assert kb._git_current_branch(target) == "wt/t_target"


def test_resolve_no_path_worktree_task_realigns_a_stale_canonical_checkout(
    kanban_home, tmp_path, monkeypatch
):
    """E2E through the real resolver: a no-``workspace_path`` worktree task
    whose canonical ``<repo>/.worktrees/<id>`` was left on a stale branch by an
    interrupted run must come back checked out on ITS OWN branch.

    This is the user-visible contract -- a worker cd's into the returned path
    and commits there. Pre-fix, ``_ensure_git_worktree`` saw a linked worktree
    of the right repo and returned immediately, so the worker committed to the
    previous run's branch.
    """
    repo = _make_repo(tmp_path)
    kb.write_board_metadata("default", default_workdir=str(repo))

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="resume me", assignee="w", workspace_kind="worktree",
        )
        conn.execute(
            "UPDATE tasks SET workspace_path = NULL WHERE id = ?", (tid,)
        )
        conn.commit()
        task = kb.get_task(conn, tid)

    # A previous interrupted dispatch of THIS task left its canonical path on
    # an unrelated branch.
    target = _add_worktree(repo, repo / ".worktrees" / tid, "wt/interrupted-run")
    assert kb._git_current_branch(target) == "wt/interrupted-run"

    resolved, branch = kb._resolve_worktree_workspace(task, board="default")

    assert Path(resolved).resolve() == target.resolve()
    assert branch == f"wt/{tid}"
    assert kb._git_current_branch(Path(resolved)) == f"wt/{tid}", (
        f"resolver returned {resolved} on branch "
        f"{kb._git_current_branch(Path(resolved))!r}; a worker would commit to "
        "the previous run's branch"
    )
