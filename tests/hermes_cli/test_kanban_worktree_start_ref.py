"""Explicit start commits for per-task Git worktrees."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import projects_db as pdb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo_with_two_commits(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    payload = repo / "payload.txt"
    payload.write_text("A\n", encoding="utf-8")
    _git(repo, "add", "payload.txt")
    _git(repo, "commit", "-m", "A")
    commit_a = _git(repo, "rev-parse", "HEAD")
    payload.write_text("B\n", encoding="utf-8")
    _git(repo, "commit", "-am", "B")
    commit_b = _git(repo, "rev-parse", "HEAD")
    return repo, commit_a, commit_b


def test_worktree_starts_at_explicit_start_ref(kanban_home, tmp_path, capsys):
    repo, commit_a, commit_b = _repo_with_two_commits(tmp_path)
    target = repo / ".worktrees" / "explicit-start"

    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="start from A",
            workspace_kind="worktree",
            workspace_path=str(target),
            start_ref=commit_a,
        )
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.start_ref == commit_a
    workspace, _ = kbw._resolve_worktree_workspace(task)
    worktree_head = _git(workspace, "rev-parse", "HEAD")

    assert _git(repo, "rev-parse", "HEAD") == commit_b
    assert worktree_head == commit_a

    default_target = repo / ".worktrees" / "default-start"
    with kbc.connect() as conn:
        default_id = kb.create_task(
            conn,
            title="keep the default base",
            body=f"start_ref={commit_a}",
            workspace_kind="worktree",
            workspace_path=str(default_target),
        )
        default_task = kb.get_task(conn, default_id)
        child_id = kb.create_task(
            conn,
            title="do not inherit start refs",
            workspace_kind="worktree",
            workspace_path=str(repo / ".worktrees" / "child"),
            parents=(task_id,),
        )
        child_task = kb.get_task(conn, child_id)

    assert default_task is not None
    assert default_task.start_ref is None
    assert child_task is not None
    assert child_task.start_ref is None
    default_workspace, _ = kbw._resolve_worktree_workspace(default_task)
    assert _git(default_workspace, "rev-parse", "HEAD") == commit_b

    from hermes_cli import kanban as kb_cli

    parser = argparse.ArgumentParser()
    kb_cli.build_parser(parser.add_subparsers())
    create_args = parser.parse_args(
        [
            "kanban",
            "create",
            "CLI start ref",
            "--assignee",
            "coder",
            "--workspace",
            f"worktree:{repo}",
            "--start-ref",
            commit_a,
            "--json",
        ]
    )
    assert kb_cli._cmd_create(create_args) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["start_ref"] == commit_a

    show_args = parser.parse_args(["kanban", "show", created["id"], "--json"])
    assert kb_cli._cmd_show(show_args) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["task"]["start_ref"] == commit_a

    invalid_args = parser.parse_args(
        [
            "kanban",
            "create",
            "Invalid start ref workspace",
            "--workspace",
            "scratch",
            "--start-ref",
            commit_a,
        ]
    )
    assert kb_cli._cmd_create(invalid_args) == 2
    assert "--start-ref is only valid with --workspace worktree" in capsys.readouterr().err

    import tools.kanban_tools  # noqa: F401 - registers the real handler
    from tools.kanban_tools_schemas import KANBAN_CREATE_SCHEMA
    from tools.registry import registry

    assert "start_ref" in KANBAN_CREATE_SCHEMA["parameters"]["properties"]
    raw_tool_result = registry.dispatch(
        "kanban_create",
        {
            "title": "Tool start ref",
            "assignee": "coder",
            "workspace_kind": "worktree",
            "workspace_path": str(repo),
            "start_ref": commit_a,
        },
    )
    assert isinstance(raw_tool_result, str)
    tool_result = json.loads(raw_tool_result)
    assert tool_result["ok"] is True
    with kbc.connect() as conn:
        tool_task = kb.get_task(conn, tool_result["task_id"])
    assert tool_task is not None
    assert tool_task.start_ref == commit_a

    legacy_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(legacy_path)
    legacy.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
        "status TEXT NOT NULL, created_at INTEGER NOT NULL)"
    )
    legacy.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy', 'old task', 'ready', 1)"
    )
    legacy.commit()
    legacy.close()

    kbc.init_db(legacy_path)
    with kbc.connect(legacy_path) as conn:
        migrated_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(tasks)")
        }
        legacy_task = kb.get_task(conn, "legacy")
    assert "start_ref" in migrated_columns
    assert legacy_task is not None
    assert legacy_task.start_ref is None


def test_invalid_start_ref_leaves_git_state_untouched(kanban_home, tmp_path):
    repo, commit_a, commit_b = _repo_with_two_commits(tmp_path)
    existing = repo / ".worktrees" / "existing"
    _git(repo, "worktree", "add", "-b", "wt/existing", str(existing), commit_b)

    with kbc.connect() as conn:
        existing_id = kb.create_task(
            conn,
            title="reuse existing worktree",
            workspace_kind="worktree",
            workspace_path=str(existing),
            branch_name="wt/existing",
            start_ref="missing-ref",
        )
        existing_task = kb.get_task(conn, existing_id)

    assert existing_task is not None
    with pytest.raises(ValueError, match="start_ref.*missing-ref.*does not uniquely resolve"):
        kbw._resolve_worktree_workspace(existing_task)
    assert _git(existing, "rev-parse", "HEAD") == commit_b
    assert _git(repo, "rev-parse", "wt/existing") == commit_b

    absent = repo / ".worktrees" / "absent"
    with kbc.connect() as conn:
        absent_id = kb.create_task(
            conn,
            title="reject missing ref",
            workspace_kind="worktree",
            workspace_path=str(absent),
            branch_name="wt/absent",
            start_ref="missing-ref",
        )
        absent_task = kb.get_task(conn, absent_id)

    assert absent_task is not None
    with pytest.raises(ValueError, match="start_ref.*missing-ref.*does not uniquely resolve"):
        kbw._resolve_worktree_workspace(absent_task)
    assert not absent.exists()
    assert subprocess.run(
        ["git", "-C", str(repo), "show-ref", "--verify", "refs/heads/wt/absent"],
        capture_output=True,
        text=True,
        check=False,
    ).returncode != 0

    _git(repo, "branch", "ambiguous", commit_b)
    _git(repo, "tag", "ambiguous", commit_a)
    ambiguous = repo / ".worktrees" / "ambiguous"
    with kbc.connect() as conn:
        ambiguous_id = kb.create_task(
            conn,
            title="reject ambiguous ref",
            workspace_kind="worktree",
            workspace_path=str(ambiguous),
            branch_name="wt/ambiguous",
            start_ref="ambiguous",
        )
        ambiguous_task = kb.get_task(conn, ambiguous_id)

    assert ambiguous_task is not None
    with pytest.raises(ValueError, match="start_ref.*ambiguous.*does not uniquely resolve"):
        kbw._resolve_worktree_workspace(ambiguous_task)
    assert not ambiguous.exists()

    _git(repo, "branch", "wt/collision", commit_b)
    collision = repo / ".worktrees" / "collision"
    with kbc.connect() as conn:
        collision_id = kb.create_task(
            conn,
            title="preserve an existing branch",
            workspace_kind="worktree",
            workspace_path=str(collision),
            branch_name="wt/collision",
            start_ref=commit_a,
        )
        collision_task = kb.get_task(conn, collision_id)

    assert collision_task is not None
    with pytest.raises(RuntimeError, match="existing branch.*wt/collision.*refusing"):
        kbw._resolve_worktree_workspace(collision_task)
    assert not collision.exists()
    assert _git(repo, "rev-parse", "wt/collision") == commit_b


def test_existing_worktree_must_match_resolved_start_ref(kanban_home, tmp_path):
    repo, commit_a, commit_b = _repo_with_two_commits(tmp_path)
    existing = repo / ".worktrees" / "existing-mismatch"
    branch = "wt/existing-mismatch"
    _git(repo, "worktree", "add", "-b", branch, str(existing), commit_b)

    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="reject an existing worktree at another commit",
            workspace_kind="worktree",
            workspace_path=str(existing),
            branch_name=branch,
            start_ref=commit_a,
        )
        task = kb.get_task(conn, task_id)

    assert task is not None
    with pytest.raises(RuntimeError, match="existing worktree.*HEAD.*requested start_ref"):
        kbw._resolve_worktree_workspace(task)
    assert _git(existing, "rev-parse", "HEAD") == commit_b
    assert _git(repo, "rev-parse", branch) == commit_b


def test_existing_branch_race_fails_closed_before_dispatch(
    kanban_home, tmp_path, monkeypatch,
):
    repo, commit_a, commit_b = _repo_with_two_commits(tmp_path)
    target = repo / ".worktrees" / "branch-race"
    branch = "wt/branch-race"
    _git(repo, "branch", branch, commit_a)

    real_git = kbw._git
    moved = False

    def racing_git(repo_root, *args, **kwargs):
        nonlocal moved
        if args == ("worktree", "add", str(target), branch) and not moved:
            moved = True
            real_git(repo_root, "branch", "-f", branch, commit_b, timeout=60)
        return real_git(repo_root, *args, **kwargs)

    monkeypatch.setattr(kbw, "_git", racing_git)
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="detect a branch move before materialization",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name=branch,
            start_ref=commit_a,
        )
        task = kb.get_task(conn, task_id)

    assert task is not None
    with pytest.raises(RuntimeError, match="branch.*changed.*requested start_ref"):
        kbw._resolve_worktree_workspace(task)
    assert moved is True
    assert not target.exists()
    assert _git(repo, "rev-parse", branch) == commit_b


def test_project_start_ref_uses_effective_worktree_across_surfaces(
    kanban_home, tmp_path, capsys,
):
    repo, commit_a, _ = _repo_with_two_commits(tmp_path)
    with pdb.connect_closing() as project_conn:
        project_id = pdb.create_project(
            project_conn,
            name="Start Ref Project",
            primary_path=str(repo),
        )

    with kbc.connect() as conn:
        direct_id = kb.create_task(
            conn,
            title="direct project start ref",
            project_id=project_id,
            start_ref=commit_a,
        )
        direct = kb.get_task(conn, direct_id)
    assert direct is not None
    assert (direct.workspace_kind, direct.project_id, direct.start_ref) == (
        "worktree", project_id, commit_a,
    )
    direct_workspace, _ = kbw._resolve_worktree_workspace(direct)
    assert _git(direct_workspace, "rev-parse", "HEAD") == commit_a

    from hermes_cli import kanban as kb_cli

    parser = argparse.ArgumentParser()
    kb_cli.build_parser(parser.add_subparsers())
    cli_args = parser.parse_args([
        "kanban", "create", "CLI project start ref", "--project", project_id,
        "--start-ref", commit_a, "--json",
    ])
    assert kb_cli._cmd_create(cli_args) == 0
    cli_task = json.loads(capsys.readouterr().out)
    assert (cli_task["workspace_kind"], cli_task["project_id"], cli_task["start_ref"]) == (
        "worktree", project_id, commit_a,
    )

    from tools import kanban_tools as kt

    tool_result = json.loads(kt._handle_create({
        "title": "Tool project start ref",
        "assignee": "coder",
        "project": project_id,
        "start_ref": commit_a,
    }))
    assert tool_result["ok"] is True
    assert (
        tool_result["workspace_kind"], tool_result["project_id"], tool_result["start_ref"],
    ) == ("worktree", project_id, commit_a)

    kb.create_board("scoped-start-ref", project_id=project_id)
    with kbc.connect(board="scoped-start-ref") as conn:
        scoped_id = kb.create_task(
            conn,
            title="board-scoped project start ref",
            board="scoped-start-ref",
            start_ref=commit_a,
        )
        scoped = kb.get_task(conn, scoped_id)
    assert scoped is not None
    assert (scoped.workspace_kind, scoped.project_id, scoped.start_ref) == (
        "worktree", project_id, commit_a,
    )

    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="start_ref is only valid for worktree workspaces"):
            kb.create_task(
                conn,
                title="explicit scratch stays invalid",
                workspace_kind="scratch",
                start_ref=commit_a,
            )
