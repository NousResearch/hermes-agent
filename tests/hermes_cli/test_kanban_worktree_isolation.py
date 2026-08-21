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


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_first_dispatch_spawn_receives_resolved_worktree_branch(
    kanban_home, tmp_path, monkeypatch, lane
):
    """The first worker sees the same fallback branch persisted for its task."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    workspace = tmp_path / "worktree"
    workspace.mkdir()
    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(
        kb,
        "_resolve_worktree_workspace",
        lambda task, **_kwargs: (workspace, f"wt/{task.id}"),
    )
    monkeypatch.setattr(
        cfgmod,
        "load_config",
        lambda *args, **kwargs: {"kanban": {"review_dispatch": True}},
    )
    captured = {}

    def spawn(task, workspace):
        captured["branch_name"] = task.branch_name
        captured["workspace"] = workspace

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title=f"{lane} worktree",
            assignee="worker",
            workspace_kind="worktree",
            workspace_path=str(workspace),
        )
        if lane == "review":
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
            conn.commit()

        result = kb.dispatch_once(conn, spawn_fn=spawn)
        persisted = kb.get_task(conn, tid)

    expected_branch = f"wt/{tid}"
    assert result.spawned == [(tid, "worker", captured["workspace"])]
    assert persisted is not None
    assert persisted.branch_name == expected_branch
    assert captured["branch_name"] == expected_branch
    assert Path(captured["workspace"]) == workspace


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_dispatch_does_not_invent_branch_for_directory_workspace(
    kanban_home, tmp_path, monkeypatch, lane
):
    """Only worktree tasks receive branch metadata at spawn time."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(
        cfgmod,
        "load_config",
        lambda *args, **kwargs: {"kanban": {"review_dispatch": True}},
    )
    captured = {}

    def spawn(task, _workspace):
        captured["branch_name"] = task.branch_name

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title=f"{lane} directory",
            assignee="worker",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        if lane == "review":
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
            conn.commit()

        kb.dispatch_once(conn, spawn_fn=spawn)
        persisted = kb.get_task(conn, tid)

    assert persisted is not None
    assert persisted.branch_name is None
    assert captured["branch_name"] is None




