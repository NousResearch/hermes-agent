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

import json
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


def test_worktree_path_outside_a_repo_is_reported_as_unavailable(kanban_home, tmp_path):
    """No repo above the named path: the resolver flags it for a downgrade
    instead of claiming the path is a broken repo."""
    plain = tmp_path / "not-a-repo" / "wt-target"

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="work outside git",
            workspace_kind="worktree",
            workspace_path=str(plain),
        )
        task = kb.get_task(conn, tid)

    with pytest.raises(kbw.WorktreeUnavailable):
        kbw._resolve_worktree_workspace(task)


def test_worktree_without_a_repo_switches_to_a_plain_dir(kanban_home, tmp_path):
    """The switch: a worktree task whose path has no repo becomes a dir task at
    that path — same workspace the card named, no branch, recorded on the board."""
    plain = tmp_path / "not-a-repo" / "wt-target"

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="work outside git",
            workspace_kind="worktree",
            workspace_path=str(plain),
            branch_name="wt/ghost",
        )
        task = kb.get_task(conn, tid)
        with pytest.raises(kbw.WorktreeUnavailable):
            kbw._resolve_worktree_workspace(task)

        workspace = kbw.switch_task_off_worktree(
            conn, task, reason="no repo at the named path"
        )
        assert workspace == plain
        assert plain.is_dir()

        row = conn.execute(
            "SELECT workspace_kind, workspace_path, branch_name FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert (row["workspace_kind"], row["workspace_path"], row["branch_name"]) == (
            "dir", str(plain), None,
        )

        # Re-resolving now takes the dir branch: stable across dispatcher ticks.
        assert kbw.resolve_workspace(kb.get_task(conn, tid)) == plain

        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
        ).fetchall()
        downgrade = [e for e in events if e["kind"] == "workspace_downgraded"]
        assert len(downgrade) == 1
        assert json.loads(downgrade[0]["payload"])["to"] == "dir"


def test_dispatch_switches_a_repo_less_worktree_card_to_a_plain_dir(
    kanban_home, monkeypatch, all_assignees_spawnable,
):
    """End to end through the dispatcher: a worktree card whose (board-inherited)
    path is not a git repo spawns as a ``dir`` card instead of blocking."""
    from hermes_cli import kanban_db_dispatch as kbd

    non_repo = kanban_home / "not-a-repo"
    non_repo.mkdir()
    monkeypatch.setattr(
        kbw._kb, "read_board_metadata", lambda *a, **k: {"default_workdir": str(non_repo)}
    )

    seen: list[tuple[str, str]] = []

    def _fake_spawn(task, workspace, board=None):
        seen.append((task.workspace_kind, str(workspace)))
        return 0  # falsy pid: dispatch records no worker_pid

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="worktree, no repo", assignee="alice", workspace_kind="worktree",
        )
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn)

        assert seen, f"card was not spawned: {res.auto_blocked}"
        assert seen[0] == ("dir", str(non_repo))
        assert not res.auto_blocked
        row = conn.execute(
            "SELECT workspace_kind, workspace_path, branch_name FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert (row["workspace_kind"], row["workspace_path"], row["branch_name"]) == (
            "dir", str(non_repo), None,
        )


def test_board_default_workdir_that_is_not_a_repo_downgrades_to_scratch(
    kanban_home, monkeypatch, all_assignees_spawnable,
):
    """A card with no path of its own (a decompose child, say) on a board whose
    default_workdir is not a repo has nowhere to branch: it runs in scratch."""
    from hermes_cli import kanban_db_dispatch as kbd

    non_repo = kanban_home / "not-a-repo"
    non_repo.mkdir()
    monkeypatch.setattr(
        kbw._kb, "read_board_metadata", lambda *a, **k: {"default_workdir": str(non_repo)}
    )

    seen: list[tuple[str, str]] = []

    def _fake_spawn(task, workspace, board=None):
        seen.append((task.workspace_kind, str(workspace)))
        return 0

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="worktree, no path", assignee="alice", workspace_kind="worktree",
        )
        conn.execute("UPDATE tasks SET workspace_path = NULL WHERE id = ?", (tid,))
        conn.commit()
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn)

        assert seen, f"card was not spawned: {res.auto_blocked}"
        kind, workspace = seen[0]
        assert kind == "scratch"
        assert workspace == str(kb.workspaces_root() / tid)
        assert not res.auto_blocked

