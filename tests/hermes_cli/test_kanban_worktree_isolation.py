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


def _make_repo(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
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


def test_resolve_linked_repo_root_anchor_materializes_task_worktree(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    linked_root = _add_worktree(repo, tmp_path / "linked-root", "anchor/main")
    branch = "wt/linked-anchor-task"

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="linked anchor",
            workspace_kind="worktree",
            workspace_path=str(linked_root),
            branch_name=branch,
        )
        task = kb.get_task(conn, tid)

    assert task is not None
    workspace, resolved_branch = kb._resolve_worktree_workspace(task)
    expected = linked_root / ".worktrees" / tid
    assert workspace == expected
    assert resolved_branch == branch
    head = subprocess.run(
        ["git", "-C", str(workspace), "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head == branch


def test_linked_anchor_inside_source_repo_stays_nested_under_anchor(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    linked_root = _add_worktree(repo, repo / "canonical", "anchor/main")

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="nested linked anchor",
            workspace_kind="worktree",
            workspace_path=str(linked_root),
        )
        task = kb.get_task(conn, tid)

    assert task is not None
    workspace, _branch = kb._resolve_worktree_workspace(task)
    assert workspace == linked_root / ".worktrees" / tid
    assert kb._git_common_dir(workspace) == kb._git_common_dir(repo)


def test_linked_anchor_inside_unrelated_repo_uses_anchor_repository(
    kanban_home, tmp_path
):
    source = _make_repo(tmp_path, "source")
    unrelated = _make_repo(tmp_path, "unrelated")
    linked_root = _add_worktree(source, unrelated / "canonical", "anchor/main")

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="cross-repo linked anchor",
            workspace_kind="worktree",
            workspace_path=str(linked_root),
        )
        task = kb.get_task(conn, tid)

    assert task is not None
    workspace, _branch = kb._resolve_worktree_workspace(task)
    assert workspace == linked_root / ".worktrees" / tid
    assert kb._git_common_dir(workspace) == kb._git_common_dir(source)
    assert kb._git_common_dir(workspace) != kb._git_common_dir(unrelated)


def test_linked_repo_root_matching_branch_is_never_reused(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    linked_root = _add_worktree(repo, tmp_path / "linked-root", "anchor/main")

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="matching linked anchor",
            workspace_kind="worktree",
            workspace_path=str(linked_root),
            branch_name="anchor/main",
        )
        task = kb.get_task(conn, tid)

    assert task is not None
    with pytest.raises(RuntimeError):
        kb._resolve_worktree_workspace(task)
    assert linked_root.is_dir()
    assert not (linked_root / ".worktrees" / tid).exists()


def test_existing_task_target_on_wrong_branch_fails_closed(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "t_existing"
    _add_worktree(repo, target, "wt/other-task")

    with pytest.raises(RuntimeError, match="branch ownership"):
        kb._ensure_git_worktree(repo, target, "wt/t_existing")
    head = subprocess.run(
        ["git", "-C", str(target), "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head == "wt/other-task"


def test_existing_plain_target_on_matching_anchor_branch_fails_closed(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    canonical = _add_worktree(repo, tmp_path / "canonical", "anchor/main")
    target = canonical / ".worktrees" / "t_plain"
    target.mkdir(parents=True)

    # Git commands inherit the surrounding linked checkout, so common-dir,
    # branch, and linked-checkout checks all appear to match. Exact toplevel
    # identity is what proves this plain directory is not the worktree root.
    assert kb._is_linked_worktree_checkout(target)
    assert kb._git_toplevel(target) == canonical
    assert kb._git_common_dir(target) == kb._git_common_dir(canonical)
    assert kb._git_current_branch(target) == "anchor/main"
    with pytest.raises(RuntimeError, match="repository ownership"):
        kb._ensure_git_worktree(canonical, target, "anchor/main")

    assert canonical.is_dir()
    assert (canonical / "README.md").is_file()
    assert target.is_dir()
    assert kb._git_toplevel(target) == canonical


@pytest.mark.parametrize("source_status", ["ready", "review"])
def test_dispatch_workspace_ownership_failure_stays_with_task(
    kanban_home, tmp_path, monkeypatch, all_assignees_spawnable, source_status
):
    repo = _make_repo(tmp_path)
    canonical = _add_worktree(repo, tmp_path / "canonical", "anchor/main")
    spawned = []

    monkeypatch.setattr(kb, "review_dispatch_enabled", lambda: True)
    monkeypatch.setattr(kb, "_memory_pressure_level", lambda: "ok")

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title=f"ambiguous {source_status} workspace",
            assignee="alice",
            workspace_kind="worktree",
            workspace_path=str(canonical),
            branch_name=f"wt/{source_status}-ambiguous",
        )
        target = canonical / ".worktrees" / tid
        target.mkdir(parents=True)
        conn.execute(
            "UPDATE tasks SET workspace_path = ?, status = ? WHERE id = ?",
            (str(target), source_status, tid),
        )
        conn.commit()

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *args, **kwargs: spawned.append((args, kwargs)),
            failure_limit=2,
        )
        task = kb.get_task(conn, tid)
        run = conn.execute(
            "SELECT outcome, status, error FROM task_runs "
            "WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()

    assert not spawned
    assert not result.spawned
    assert not result.auto_blocked
    assert task is not None
    assert task.status == source_status
    assert task.consecutive_failures == 1
    assert task.last_failure_error is not None
    assert task.last_failure_error.startswith("workspace: ")
    assert "repository ownership is ambiguous" in task.last_failure_error
    assert run is not None
    assert run["outcome"] == "spawn_failed"
    assert run["status"] == "spawn_failed"
    assert "repository ownership is ambiguous" in run["error"]


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



