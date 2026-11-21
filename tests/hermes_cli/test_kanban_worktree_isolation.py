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
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


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
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="build the feature", triage=True)
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', "
            "workspace_path='/repo/.worktrees/root' WHERE id = ?",
            (root,),
        )
        conn.commit()

        child_ids = decompose_triage_task(
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




def test_resolve_worktree_falls_back_when_path_occupied(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="second sibling",
            workspace_kind="worktree",
            workspace_path=str(occupied),  # inherited shared/stale path
        )
        task = kb.get_task(conn, tid)

    workspace, branch = kbw._resolve_worktree_workspace(task)
    assert workspace == (repo / ".worktrees" / tid).resolve()
    assert branch == f"wt/{tid}"
    # The sibling's checkout is untouched, still on its own branch.
    assert (occupied / "README.md").exists()
    head = subprocess.run(
        ["git", "-C", str(occupied), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == "wt/sibling"


@pytest.mark.parametrize("entry", ["helper", "persisted"])
def test_stale_canonical_worktree_refuses_without_mutation(
    kanban_home, tmp_path, monkeypatch, entry
):
    repo = _make_repo(tmp_path)
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="retry", workspace_kind="worktree", workspace_path=str(repo)
        )
        task = kb.get_task(conn, tid)
        target, branch = kbw._resolve_worktree_workspace(task)
        kbw.set_workspace_path(conn, tid, target)
        kbw.set_branch_name(conn, tid, branch)
        task = kb.get_task(conn, tid)
        # The retry sees exactly the dispatcher-persisted canonical path.
        _git(target, "checkout", "-b", "wt/stale")
        (target / "README.md").write_bytes(b"dirty tracked\n")
        (target / "untracked").write_bytes(b"dirty untracked\n")
        before_head = (kbw._git_dir(target) / "HEAD").read_bytes()
        before_task = (task.workspace_path, task.branch_name)
        real_run = subprocess.run
        commands = []

        def record_run(args, *a, **kw):
            commands.append(args)
            return real_run(args, *a, **kw)

        monkeypatch.setattr(subprocess, "run", record_run)
        with pytest.raises(RuntimeError, match="branch") as exc:
            if entry == "helper":
                kbw._ensure_git_worktree(repo, target, branch)
            else:
                kbw._resolve_worktree_workspace(task)
        assert str(target) in str(exc.value)
        assert branch in str(exc.value) and "wt/stale" in str(exc.value)
        assert "manually" in str(exc.value)
        assert not any("checkout" in cmd or "switch" in cmd for cmd in commands)
        assert (kbw._git_dir(target) / "HEAD").read_bytes() == before_head
        assert (target / "README.md").read_bytes() == b"dirty tracked\n"
        assert (target / "untracked").read_bytes() == b"dirty untracked\n"
        persisted = kb.get_task(conn, tid)
        assert (persisted.workspace_path, persisted.branch_name) == before_task
        assert (task.workspace_path, task.branch_name) == before_task


def test_linked_anchor_same_branch_reuse_and_foreign_sibling(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    anchor = _add_worktree(repo, tmp_path / "anchor", "wt/anchor")
    target = _add_worktree(repo, anchor / ".worktrees" / "same", "wt/same")
    (target / "README.md").write_bytes(b"keep dirty\n")
    kbw._ensure_git_worktree(anchor, target, "wt/same")
    assert kbw._git_common_dir(anchor) == kbw._git_common_dir(target)
    assert (target / "README.md").read_bytes() == b"keep dirty\n"
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="sibling", workspace_kind="worktree", workspace_path=str(target)
        )
        task = kb.get_task(conn, tid)
        resolved, branch = kbw._resolve_worktree_workspace(task)
        assert resolved == (anchor / ".worktrees" / tid).resolve()
        assert kbw._git_current_branch(resolved) == branch
        assert kbw._git_current_branch(target) == "wt/same"
        assert (target / "README.md").read_bytes() == b"keep dirty\n"
        kbw.set_workspace_path(conn, tid, resolved)
        kbw.set_branch_name(conn, tid, branch)
        assert kbw._resolve_worktree_workspace(kb.get_task(conn, tid)) == (resolved, branch)




