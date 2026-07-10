"""K-amendment coverage: no-code completion waiver, decompose per-child
worktree isolation, occupied-worktree dispatch fallback, and pinned
explicit workspaces.

- K1: `complete_task` waives the CI + merge gates when the task branch
  carries only auto-checkpoint commits touching documentation files —
  but a worker-reported red CI verdict still blocks.
- K2: `decompose_triage_task` never copies a worktree root's literal
  workspace_path onto children (each child resolves its own worktree).
- K2b: `_resolve_worktree_workspace` falls back to a fresh
  `<repo>/.worktrees/<task-id>` when the requested path is occupied by
  another task's branch, and still fails loudly when the occupied path
  IS the task's own canonical worktree.
- K3: an explicitly chosen workspace kind (`workspace_pinned`) is never
  auto-upgraded to a worktree at create or dispatch time.
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


def _commit_in(worktree: Path, relpath: str, message: str) -> None:
    f = worktree / relpath
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("content\n", encoding="utf-8")
    _git(worktree, "add", relpath)
    _git(worktree, "commit", "-m", message)


def _event_kinds(conn, tid: str) -> list:
    rows = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
    ).fetchall()
    return [r["kind"] for r in rows]


# ---------------------------------------------------------------------------
# K1 — no-code completion waiver
# ---------------------------------------------------------------------------

def test_nocode_checkpoint_branch_completes_without_ci(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    worktree = _add_worktree(repo, repo / ".worktrees" / "live", "wt/live")
    _commit_in(
        worktree, "notes/findings.md",
        "wip: kanban worker checkpoint for t_x on wt/live",
    )

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="research card",
            workspace_kind="worktree",
            workspace_path=str(worktree),
            branch_name="wt/live",
        )
        # Unmerged branch, no metadata['ci'] — previously double-blocked.
        assert kb.complete_task(conn, tid, summary="brief written")
        assert kb.get_task(conn, tid).status == "done"
        assert "completion_gate_waived_nocode" in _event_kinds(conn, tid)


def test_nocode_waiver_requires_docs_only_diff(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    worktree = _add_worktree(repo, repo / ".worktrees" / "live", "wt/live")
    # Checkpoint-prefixed commit, but it swallowed real code.
    _commit_in(
        worktree, "src/app.py",
        "wip: kanban worker checkpoint for t_x on wt/live",
    )

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="code card",
            workspace_kind="worktree",
            workspace_path=str(worktree),
            branch_name="wt/live",
        )
        with pytest.raises(kb.CompletionGateError, match="not merged"):
            kb.complete_task(conn, tid, summary="done-ish")
        assert kb.get_task(conn, tid).status != "done"


def test_nocode_waiver_requires_checkpoint_only_commits(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    worktree = _add_worktree(repo, repo / ".worktrees" / "live", "wt/live")
    # Docs-only diff, but a worker-authored commit — no waiver.
    _commit_in(worktree, "notes/findings.md", "add findings")

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="authored docs card",
            workspace_kind="worktree",
            workspace_path=str(worktree),
            branch_name="wt/live",
        )
        with pytest.raises(kb.CompletionGateError, match="not merged"):
            kb.complete_task(conn, tid, summary="done-ish")


def test_nocode_waiver_never_overrides_red_ci(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    worktree = _add_worktree(repo, repo / ".worktrees" / "live", "wt/live")
    # Merged, zero extra commits: waiver-shaped branch...
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="honest red card",
            workspace_kind="worktree",
            workspace_path=str(worktree),
            branch_name="wt/live",
        )
        # ...but the worker explicitly reported failing tests.
        with pytest.raises(kb.CompletionGateError, match="CI failed"):
            kb.complete_task(
                conn,
                tid,
                summary="tests are red",
                metadata={"ci": {"typecheck": True, "lint": True, "tests": False}},
            )
        assert kb.get_task(conn, tid).status != "done"


# ---------------------------------------------------------------------------
# K2 — decompose children never share a worktree
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# K2b — occupied-worktree fallback at resolve time
# ---------------------------------------------------------------------------

def test_resolve_worktree_falls_back_when_path_occupied(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="second sibling",
            workspace_kind="worktree",
            workspace_path=str(occupied),  # inherited stale/shared path
        )
        task = kb.get_task(conn, tid)

    workspace, branch, base_ref, base_commit = kb._resolve_worktree_workspace(task)
    assert workspace == (repo / ".worktrees" / tid).resolve()
    assert branch == f"wt/{tid}"
    assert base_ref == "main"
    assert base_commit
    # The sibling's checkout is untouched.
    assert (occupied / "README.md").exists()


def test_resolve_worktree_still_raises_when_own_path_occupied(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="corrupted worktree",
            workspace_kind="worktree",
        )
        # Occupy this task's OWN canonical path with a foreign branch.
        own_path = repo / ".worktrees" / tid
        _add_worktree(repo, own_path, "wt/foreign")
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?",
            (str(own_path), tid),
        )
        conn.commit()
        task = kb.get_task(conn, tid)

    with pytest.raises(ValueError, match="already on branch"):
        kb._resolve_worktree_workspace(task)


# ---------------------------------------------------------------------------
# K3 — pinned explicit workspaces
# ---------------------------------------------------------------------------

def test_pinned_scratch_not_upgraded_at_create(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    kb.write_board_metadata("default", default_workdir=str(repo))

    with kb.connect() as conn:
        pinned = kb.create_task(
            conn, title="explicit scratch", board="default",
            workspace_kind="scratch", workspace_pinned=True,
        )
        default = kb.create_task(conn, title="default scratch", board="default")

        assert kb.get_task(conn, pinned).workspace_kind == "scratch"
        assert kb.get_task(conn, pinned).workspace_pinned is True
        # Control: the unpinned default still auto-upgrades on a bound board.
        assert kb.get_task(conn, default).workspace_kind == "worktree"


def test_pinned_scratch_not_converted_at_dispatch(
    kanban_home, tmp_path, all_assignees_spawnable
):
    repo = _make_repo(tmp_path)
    kb.write_board_metadata("default", default_workdir=str(repo))
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append((task.id, Path(workspace)))

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="pinned ops card", assignee="alice", board="default",
            workspace_kind="scratch", workspace_pinned=True,
        )
        kb.dispatch_once(conn, spawn_fn=fake_spawn, board="default")

    assert [s[0] for s in spawns] == [tid]
    workspace = spawns[0][1]
    assert ".worktrees" not in workspace.parts
    assert "workspaces" in workspace.parts


def test_legacy_unpinned_scratch_still_converted_at_dispatch(
    kanban_home, tmp_path, all_assignees_spawnable
):
    repo = _make_repo(tmp_path)
    kb.write_board_metadata("default", default_workdir=str(repo))
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append((task.id, Path(workspace)))

    with kb.connect() as conn:
        # Simulate a pre-K3 row: scratch kind, not pinned (e.g. created
        # before the board was bound to a repo).
        tid = kb.create_task(
            conn, title="legacy card", assignee="alice", board="default",
            workspace_kind="scratch", workspace_pinned=True,
        )
        conn.execute(
            "UPDATE tasks SET workspace_pinned = 0 WHERE id = ?", (tid,)
        )
        conn.commit()
        kb.dispatch_once(conn, spawn_fn=fake_spawn, board="default")

    assert [s[0] for s in spawns] == [tid]
    assert ".worktrees" in spawns[0][1].parts
