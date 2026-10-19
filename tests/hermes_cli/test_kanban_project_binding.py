"""Safe, workspace-preserving binding of existing Kanban tasks to Projects."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_project_binding as project_binding
from hermes_cli import projects_db as pdb
from hermes_cli.kanban import kanban_command
from hermes_cli.kanban_parser import build_parser


@pytest.fixture
def kanban_conn(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn, tmp_path


def _make_project(
    tmp_path, *, name="Target project", slug=None, board_slug=None, home=None, git_repo=True,
) -> pdb.Project:
    repo = tmp_path / name.replace(" ", "-")
    repo.mkdir(parents=True, exist_ok=True)
    if git_repo:
        subprocess.run(["git", "init", "--quiet", str(repo)], check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=Hermes Test", "-c",
             "user.email=hermes-test@example.invalid", "commit", "--allow-empty", "--quiet", "-m", "initial"],
            check=True, capture_output=True, text=True,
        )
    if home is None:
        with pdb.connect_closing() as conn:
            project_id = pdb.create_project(
                conn, name=name, slug=slug, board_slug=board_slug, primary_path=str(repo),
            )
            project = pdb.get_project(conn, project_id)
            assert project is not None
            return project
    home.mkdir(parents=True, exist_ok=True)
    with pdb.connect_closing(db_path=home / "projects.db") as conn:
        project_id = pdb.create_project(
            conn, name=name, slug=slug, board_slug=board_slug, primary_path=str(repo),
        )
        project = pdb.get_project(conn, project_id)
        assert project is not None
        return project


def _make_projectless_task(
    conn, project, *, status="blocked", workspace_kind="worktree",
    workspace_path=None, branch_name=None,
):
    task_id = kb.create_task(conn, title="Recover existing task", initial_status=status)
    assert project.primary_path is not None
    path = workspace_path or os.path.join(project.primary_path, ".worktrees", task_id)
    branch = branch_name or f"{project.slug}/{task_id}-recover-existing-task"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subprocess.run(
        ["git", "-C", project.primary_path, "worktree", "add", "--quiet", "-b", branch, path, "HEAD"],
        check=True, capture_output=True, text=True,
    )
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_kind = ?, workspace_path = ?, branch_name = ? WHERE id = ?",
            (workspace_kind, path, branch, task_id),
        )
    return task_id


def _make_projectless_dir_task(conn, project, *, workspace_path=None, status="blocked"):
    assert project.primary_path is not None
    task_id = kb.create_task(conn, title="Recover directory task", initial_status=status)
    path = workspace_path or project.primary_path
    os.makedirs(path, exist_ok=True)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_kind = 'dir', workspace_path = ?, branch_name = NULL WHERE id = ?",
            (path, task_id),
        )
    return task_id


def _make_projectless_task_on_scoped_board(tmp_path, *, board="scoped"):
    project = _make_project(tmp_path, name="Binding destination", board_slug=board)
    board_project = _make_project(tmp_path, name="Board project", board_slug=board)
    kb.create_board(board, name="Scoped board", project_id=board_project.id)
    with kbc.connect_closing(board=board) as conn:
        task_id = _make_projectless_task(conn, project)
        task_before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
        events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))
    return board, project, task_id, task_before, events_before


def _invoke_bind(task_id, project_ref, capsys, *, board=None):
    root = argparse.ArgumentParser()
    build_parser(root.add_subparsers())
    argv = ["kanban"]
    if board is not None:
        argv.extend(["--board", board])
    argv.extend(["bind-project", task_id, "--project", project_ref, "--json"])
    code = kanban_command(root.parse_args(argv))
    captured = capsys.readouterr()
    payload = json.loads(captured.out) if captured.out.strip() else None
    return code, payload, captured


def _rows(conn, sql, params):
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def test_compatible_blocked_projectless_task_can_be_bound_through_cli(kanban_conn, capsys):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    task_id = _make_projectless_task(conn, project)

    code, result, _ = _invoke_bind(task_id, project.id, capsys)

    assert code == 0
    assert result is not None
    assert result == {
        "ok": True,
        "result": "bound",
        "changed": True,
        "task_id": task_id,
        "project_id": project.id,
        "project_slug": project.slug,
        "board": "default",
    }
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.project_id == project.id
    event = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    assert event is not None
    assert event["kind"] == "project_bound"
    event_payload = json.loads(event["payload"])
    assert event_payload["from_project_id"] is None
    assert event_payload["to_project_id"] == project.id
    assert event_payload["project_slug"] == project.slug
    assert event_payload["actor"]


def test_unknown_project_is_rejected_without_mutation(kanban_conn, capsys):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    task_id = _make_projectless_task(conn, project)
    before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))

    code, result, _ = _invoke_bind(task_id, "missing-project", capsys)

    assert code != 0
    assert result is not None
    assert result["ok"] is False
    assert result["result"] == "error"
    assert result["changed"] is False
    assert "not found" in result["error"].lower()
    assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()) == before
    assert _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)) == events_before


def test_ambiguous_project_reference_is_rejected_without_mutation(kanban_conn, capsys):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path, name="Primary")
    _make_project(tmp_path, name="Shadow", slug=project.id)
    task_id = _make_projectless_task(conn, project)
    before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))

    code, result, _ = _invoke_bind(task_id, project.id, capsys)

    assert code != 0
    assert result is not None
    assert result["ok"] is False
    assert "ambiguous" in result["error"].lower()
    assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()) == before
    assert _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)) == events_before


@pytest.mark.parametrize("claim_state", ["running", "claimed"])
def test_running_or_claimed_task_cannot_be_bound(kanban_conn, capsys, claim_state):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    task_id = _make_projectless_task(conn, project)
    with kb.write_txn(conn):
        if claim_state == "running":
            run_id = kb._claim_and_open_run(
                conn, task_id, "blocked", "worker-claim", 9999999999, int(time.time()),
            )
            assert run_id is not None
        else:
            conn.execute(
                "UPDATE tasks SET claim_lock = 'worker-claim', claim_expires = 9999999999 WHERE id = ?",
                (task_id,),
            )
    before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))

    code, result, _ = _invoke_bind(task_id, project.id, capsys)

    assert code != 0
    assert result is not None
    assert result["ok"] is False
    assert "running" in result["error"].lower() or "claim" in result["error"].lower()
    assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()) == before
    assert _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)) == events_before


def test_blocked_task_with_historical_worker_started_at_can_be_bound_without_clearing_it(
    kanban_conn, capsys,
):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    task_id = _make_projectless_task(conn, project)
    historical_started_at = 1_700_000_000
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'blocked', claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL, worker_started_at = ?, current_run_id = NULL WHERE id = ?",
            (historical_started_at, task_id),
        )
    before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))
    assert before["status"] == "blocked"
    assert before["claim_lock"] is None and before["claim_expires"] is None
    assert before["worker_pid"] is None and before["current_run_id"] is None
    assert before["worker_started_at"] == historical_started_at

    code, result, _ = _invoke_bind(task_id, project.id, capsys)

    after = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_after = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))
    assert code == 0
    assert result is not None and result["ok"] is True
    assert after == {**before, "project_id": project.id}
    assert after["worker_started_at"] == historical_started_at
    assert events_after[:-1] == events_before
    assert events_after[-1]["kind"] == "project_bound"


@pytest.mark.parametrize("workspace_issue", ["path", "branch", "kind"])
def test_incompatible_workspace_is_rejected_without_mutation(kanban_conn, capsys, workspace_issue):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    kwargs = {}
    if workspace_issue == "path":
        kwargs["workspace_path"] = os.path.join(project.primary_path, ".worktrees", "another-task")
    elif workspace_issue == "branch":
        kwargs["branch_name"] = "unrelated/feature"
    else:
        kwargs["workspace_kind"] = "dir"
    task_id = _make_projectless_task(conn, project, **kwargs)
    before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))

    code, result, _ = _invoke_bind(task_id, project.id, capsys)

    assert code != 0
    assert result is not None
    assert result["ok"] is False
    assert "workspace" in result["error"].lower()
    assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()) == before
    assert _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)) == events_before


def test_compatible_blocked_projectless_directory_task_can_be_bound_without_git(
    kanban_conn, tmp_path, capsys,
):
    conn, _ = kanban_conn
    project = _make_project(tmp_path, git_repo=False)
    task_id = _make_projectless_dir_task(conn, project)
    before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    assert project.primary_path is not None
    assert not os.path.exists(os.path.join(project.primary_path, ".git"))

    code, result, _ = _invoke_bind(task_id, project.id, capsys)

    task_after = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    assert code == 0
    assert result is not None and result["ok"] is True
    assert task_after == {**before, "project_id": project.id}
    assert task_after["workspace_kind"] == "dir"
    assert task_after["workspace_path"] == project.primary_path
    assert os.path.isdir(task_after["workspace_path"])


def test_directory_workspace_from_another_project_is_rejected_without_mutation(
    kanban_conn, tmp_path, capsys,
):
    conn, _ = kanban_conn
    project = _make_project(tmp_path, name="Destination directory", git_repo=False)
    other_project = _make_project(tmp_path, name="Other directory", git_repo=False)
    task_id = _make_projectless_dir_task(conn, project, workspace_path=other_project.primary_path)
    before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))

    code, result, _ = _invoke_bind(task_id, project.id, capsys)

    assert code != 0
    assert result is not None and result["ok"] is False
    assert "workspace" in result["error"].lower()
    assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()) == before
    assert _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)) == events_before


def test_corrupt_scoped_board_metadata_rejects_binding_without_mutation(
    kanban_conn, tmp_path, capsys,
):
    _, _ = kanban_conn
    board, project, task_id, before, events_before = _make_projectless_task_on_scoped_board(tmp_path)
    kb.board_metadata_path(board).write_text("{corrupt", encoding="utf-8")

    code, result, _ = _invoke_bind(task_id, project.id, capsys, board=board)

    assert code != 0
    assert result is not None and result["ok"] is False
    assert "metadata" in result["error"].lower()
    with kbc.connect_closing(board=board) as conn:
        assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()) == before
        assert _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)) == events_before


def test_unreadable_scoped_board_metadata_rejects_binding_without_mutation(
    kanban_conn, tmp_path, capsys, monkeypatch,
):
    _, _ = kanban_conn
    board, project, task_id, before, events_before = _make_projectless_task_on_scoped_board(tmp_path)
    metadata_path = kb.board_metadata_path(board)
    original_read_text = Path.read_text

    def deny_metadata_read(path, *args, **kwargs):
        if path == metadata_path:
            raise PermissionError("board metadata is unreadable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny_metadata_read)

    code, result, _ = _invoke_bind(task_id, project.id, capsys, board=board)

    assert code != 0
    assert result is not None and result["ok"] is False
    assert "metadata" in result["error"].lower()
    with kbc.connect_closing(board=board) as conn:
        assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()) == before
        assert _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)) == events_before


def test_scoped_board_metadata_without_project_scope_is_rejected_without_mutation(
    kanban_conn, tmp_path, capsys,
):
    _, _ = kanban_conn
    board, project, task_id, before, events_before = _make_projectless_task_on_scoped_board(tmp_path)
    kb.board_metadata_path(board).write_text(json.dumps({"slug": board}), encoding="utf-8")

    code, result, _ = _invoke_bind(task_id, project.id, capsys, board=board)

    assert code != 0
    assert result is not None and result["ok"] is False
    assert "project_id" in result["error"]
    with kbc.connect_closing(board=board) as conn:
        assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()) == before
        assert _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)) == events_before


def test_board_metadata_with_wrong_slug_is_rejected_without_mutation(
    kanban_conn, tmp_path, capsys,
):
    _, _ = kanban_conn
    board, project, task_id, before, events_before = _make_projectless_task_on_scoped_board(tmp_path)
    kb.board_metadata_path(board).write_text(
        json.dumps({"slug": "different-board", "project_id": project.id}), encoding="utf-8",
    )

    code, result, _ = _invoke_bind(task_id, project.id, capsys, board=board)

    assert code != 0
    assert result is not None and result["ok"] is False
    assert "slug integrity" in result["error"].lower()
    with kbc.connect_closing(board=board) as conn:
        assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()) == before
        assert _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)) == events_before


def test_missing_default_board_metadata_remains_unscoped_and_allows_binding(
    kanban_conn, capsys,
):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    task_id = _make_projectless_task(conn, project)
    metadata_path = kb.board_metadata_path(kb.DEFAULT_BOARD)
    metadata_path.unlink(missing_ok=True)

    code, result, _ = _invoke_bind(task_id, project.id, capsys)

    assert code == 0
    assert result is not None and result["ok"] is True
    task = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    assert task["project_id"] == project.id
    assert task["workspace_kind"] == "worktree"

@pytest.mark.parametrize("unsafe_id", [".", ".."])
def test_dot_path_components_cannot_be_used_as_task_ids(
    kanban_conn, tmp_path, monkeypatch, unsafe_id,
):
    _, _ = kanban_conn
    project = _make_project(tmp_path)
    root = project.primary_path
    assert root is not None
    workspace = os.path.join(root, ".worktrees", unsafe_id)
    os.makedirs(workspace, exist_ok=True)
    branch = f"{project.slug}/{unsafe_id}"

    def fake_git(path, *args):
        if args == ("--show-toplevel",):
            return os.path.realpath(path)
        if args == ("--git-common-dir",):
            return os.path.join(root, ".git")
        if args == ("--abbrev-ref", "HEAD"):
            return branch
        raise AssertionError(f"unexpected Git query: {args}")

    monkeypatch.setattr(project_binding, "_git", fake_git)
    task = {
        "id": unsafe_id,
        "workspace_kind": "worktree",
        "workspace_path": workspace,
        "branch_name": branch,
    }

    with pytest.raises(ValueError, match="unsafe task ID path component"):
        project_binding._check_workspace(project, task)


def test_git_metadata_probe_scrubs_profile_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-credential")
    monkeypatch.setenv("GIT_DIR", "/ambient/git/selector")
    observed = {}

    def fake_run(argv, *, env, **kwargs):
        observed.update(env)
        return subprocess.CompletedProcess(argv, 0, stdout="/verified/repo\n", stderr="")

    monkeypatch.setattr(project_binding.subprocess, "run", fake_run)

    assert project_binding._git("/verified/repo", "--show-toplevel") == "/verified/repo"
    assert "OPENAI_API_KEY" not in observed
    assert "GIT_DIR" not in observed


def test_binding_preserves_identity_dependencies_comments_and_run_history(kanban_conn, capsys):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    parent_id = kb.create_task(conn, title="Prerequisite")
    task_id = _make_projectless_task(conn, project)
    kb.link_tasks(conn, parent_id, task_id)
    kb.add_comment(conn, task_id, "operator", "Existing comment")
    with kb.write_txn(conn):
        kb._synthesize_ended_run(conn, task_id, outcome="completed", summary="Prior run")

    task_before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    links_before = _rows(
        conn,
        "SELECT * FROM task_links WHERE parent_id = ? OR child_id = ? ORDER BY parent_id, child_id",
        (task_id, task_id),
    )
    comments_before = _rows(conn, "SELECT * FROM task_comments WHERE task_id = ? ORDER BY id", (task_id,))
    runs_before = _rows(conn, "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,))
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))

    code, result, _ = _invoke_bind(task_id, project.id, capsys)

    task_after = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    expected_task = {**task_before, "project_id": project.id}
    assert code == 0
    assert result is not None
    assert result["task_id"] == task_id
    assert task_after == expected_task
    assert _rows(
        conn,
        "SELECT * FROM task_links WHERE parent_id = ? OR child_id = ? ORDER BY parent_id, child_id",
        (task_id, task_id),
    ) == links_before
    assert _rows(conn, "SELECT * FROM task_comments WHERE task_id = ? ORDER BY id", (task_id,)) == comments_before
    assert _rows(conn, "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,)) == runs_before
    events_after = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))
    assert events_after[:-1] == events_before
    assert events_after[-1]["kind"] == "project_bound"


def test_repeated_same_binding_is_idempotent_and_does_not_duplicate_audit_event(kanban_conn, capsys):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    task_id = _make_projectless_task(conn, project)

    first_code, first_result, _ = _invoke_bind(task_id, project.slug, capsys)
    second_code, second_result, _ = _invoke_bind(task_id, project.id, capsys)

    assert first_code == second_code == 0
    assert first_result is not None and second_result is not None
    assert first_result["result"] == "bound"
    assert first_result["changed"] is True
    assert second_result["result"] == "already_bound"
    assert second_result["changed"] is False
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.project_id == project.id
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'project_bound'", (task_id,)
    ).fetchone()[0] == 1


def test_project_from_another_profile_is_not_resolved(kanban_conn, tmp_path, capsys):
    conn, _ = kanban_conn
    foreign_home = tmp_path / "other-profile"
    foreign_project = _make_project(tmp_path, name="Foreign project", home=foreign_home)
    task_id = _make_projectless_task(conn, foreign_project)

    code, result, _ = _invoke_bind(task_id, foreign_project.id, capsys)

    assert code != 0
    assert result is not None
    assert result["ok"] is False
    assert "not found" in result["error"].lower()
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.project_id is None


@pytest.mark.parametrize("scope_mismatch", ["board_project", "project_board_slug"])
def test_board_and_project_scope_mismatches_are_rejected(kanban_conn, tmp_path, capsys, scope_mismatch):
    conn, _ = kanban_conn
    if scope_mismatch == "project_board_slug":
        project = _make_project(tmp_path, board_slug="other-board")
    else:
        project = _make_project(tmp_path)
        other = _make_project(tmp_path, name="Board project")
        kb.write_board_metadata("default", project_id=other.id)
    task_id = _make_projectless_task(conn, project)
    before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))

    code, result, _ = _invoke_bind(task_id, project.id, capsys)

    assert code != 0
    assert result is not None
    assert result["ok"] is False
    assert "board" in result["error"].lower()
    assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()) == before
    assert _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)) == events_before


def test_bind_project_initialization_sqlite_error_returns_json_error(
    kanban_conn, capsys, monkeypatch,
):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    task_id = _make_projectless_task(conn, project)
    task_before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))

    def fail_initialization(*args, **kwargs):
        raise sqlite3.OperationalError("injected SQLite initialization failure")

    monkeypatch.setattr(kb, "init_db", fail_initialization)

    code, result, captured = _invoke_bind(task_id, project.id, capsys)

    assert code != 0
    assert result is not None, (
        "expected a JSON error envelope; "
        f"stdout={captured.out!r}, stderr={captured.err!r}"
    )
    assert set(result) == {
        "ok", "result", "changed", "task_id", "project_id", "project_slug", "board", "error",
    }
    assert result["ok"] is False
    assert result["result"] == "error"
    assert result["changed"] is False
    assert result["task_id"] == task_id
    assert result["project_id"] is None
    assert result["project_slug"] is None
    assert result["board"] == "default"
    assert "injected SQLite initialization failure" in result["error"]
    assert captured.err == ""
    assert "Bound " not in captured.out
    assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()) == task_before
    assert _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)) == events_before


def test_bind_project_corrupt_database_initialization_returns_json_error(
    kanban_conn, capsys, monkeypatch,
):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    task_id = _make_projectless_task(conn, project)
    task_before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))
    corrupt_error = kbc.KanbanDbCorruptError(
        tmp_path / "corrupt-kanban.db",
        tmp_path / "corrupt-kanban.db.corrupt.backup",
        "integrity_check returned malformed",
    )
    initialization_calls = []
    binding_calls = []

    def fail_initialization(*args, **kwargs):
        initialization_calls.append((args, kwargs))
        raise corrupt_error

    def unexpected_binding(*args, **kwargs):
        binding_calls.append((args, kwargs))
        raise AssertionError("binding handler must not run after initialization failure")

    monkeypatch.setattr(kb, "init_db", fail_initialization)
    monkeypatch.setattr(project_binding, "bind_task_project", unexpected_binding)

    root = argparse.ArgumentParser()
    build_parser(root.add_subparsers())
    args = root.parse_args([
        "kanban", "bind-project", task_id, "--project", project.id, "--json",
    ])
    code = kanban_command(args)
    captured = capsys.readouterr()
    try:
        result = json.loads(captured.out)
    except json.JSONDecodeError:
        result = None

    assert code != 0
    assert len(initialization_calls) == 1
    assert isinstance(corrupt_error, kbc.KanbanDbCorruptError)
    assert result is not None, (
        "expected a JSON error envelope for KanbanDbCorruptError; "
        f"stdout={captured.out!r}, stderr={captured.err!r}"
    )
    assert result["ok"] is False
    assert result["result"] == "error"
    assert result["changed"] is False
    assert result["task_id"] == task_id
    assert result["project_id"] is None
    assert result["project_slug"] is None
    assert "integrity" in result["error"].lower() or "corrupt" in result["error"].lower()
    assert str(tmp_path) not in result["error"]
    assert captured.err == ""
    assert binding_calls == []
    assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()) == task_before
    assert _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)) == events_before


def test_bind_project_audit_sqlite_error_returns_json_and_rolls_back(
    kanban_conn, capsys,
):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    task_id = _make_projectless_task(conn, project)
    task_before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))
    conn.execute(
        """
        CREATE TRIGGER fail_project_bound_event
        BEFORE INSERT ON task_events
        WHEN NEW.kind = 'project_bound'
        BEGIN
            SELECT RAISE(ABORT, 'injected project_bound audit failure');
        END
        """
    )

    escaped_error = None
    try:
        code, result, captured = _invoke_bind(task_id, project.id, capsys)
    except sqlite3.Error as exc:
        escaped_error = exc
        code, result = None, None
        captured = capsys.readouterr()

    task_after = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_after = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))
    assert task_after == task_before
    assert events_after == events_before
    assert not any(event["kind"] == "project_bound" for event in events_after)
    assert escaped_error is None, (
        "SQLite audit failure escaped the CLI instead of producing JSON: "
        f"{escaped_error!r}; stdout={captured.out!r}, stderr={captured.err!r}"
    )
    assert code is not None and code != 0
    assert result is not None and result["ok"] is False
    assert result["result"] == "error"
    assert result["changed"] is False
    assert result["task_id"] == task_id
    assert result["project_id"] is None
    assert result["project_slug"] is None
    assert result["board"] == "default"
    assert "injected project_bound audit failure" in result["error"]
    assert captured.err == ""
    assert "Bound " not in captured.out


def test_post_commit_readback_failure_is_indeterminate_without_retry(
    kanban_conn, capsys, monkeypatch,
):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    task_id = _make_projectless_task(conn, project)
    original_check = kbc._check_file_length_invariant
    committed = []

    def fail_after_commit(candidate_conn):
        if not candidate_conn.in_transaction:
            task = candidate_conn.execute(
                "SELECT project_id FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            event = candidate_conn.execute(
                "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            if task is not None and task["project_id"] == project.id and event is not None:
                committed.append((task["project_id"], event["kind"]))
                raise sqlite3.DatabaseError("injected post-COMMIT invariant failure")
        original_check(candidate_conn)

    original_snapshot = kanban_cli._bind_project_snapshot
    snapshot_calls = []

    def fail_reconciliation_read(candidate_conn, candidate_task_id):
        snapshot_calls.append(candidate_task_id)
        if len(snapshot_calls) == 2:
            raise sqlite3.OperationalError("injected reconciliation readback failure")
        return original_snapshot(candidate_conn, candidate_task_id)

    original_bind = project_binding.bind_task_project
    bind_calls = []

    def count_bind_calls(*args, **kwargs):
        bind_calls.append(1)
        return original_bind(*args, **kwargs)

    monkeypatch.setattr(kbc, "_check_file_length_invariant", fail_after_commit)
    monkeypatch.setattr(kanban_cli, "_bind_project_snapshot", fail_reconciliation_read)
    monkeypatch.setattr(project_binding, "bind_task_project", count_bind_calls)

    code, result, captured = _invoke_bind(task_id, project.id, capsys)

    assert code == 1
    assert result is not None
    assert result["result"] == "reconciliation_required"
    assert result["changed"] is None
    assert result["binding_state"] == "indeterminate"
    assert result["reconciliation_required"] is True
    assert "injected post-COMMIT invariant failure" in result["error"]
    assert "injected reconciliation readback failure" in result["reconciliation_error"]
    assert len(committed) == 1 and committed[0] == (project.id, "project_bound")
    assert len(snapshot_calls) == 2
    assert len(bind_calls) == 1
    assert captured.err == ""
    assert json.loads(captured.out) == result

    with kbc.connect_closing() as read_conn:
        task_after = read_conn.execute(
            "SELECT project_id FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        event_after = read_conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
    assert task_after["project_id"] == project.id
    assert event_after["kind"] == "project_bound"


def test_bind_project_post_commit_sqlite_error_requires_reconciliation(
    kanban_conn, capsys, monkeypatch,
):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    task_id = _make_projectless_task(conn, project)
    task_before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))
    original_check = kbc._check_file_length_invariant
    committed_state = []

    def fail_after_committed_binding(candidate_conn):
        if not candidate_conn.in_transaction:
            task = candidate_conn.execute(
                "SELECT project_id FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            event = candidate_conn.execute(
                "SELECT kind, payload FROM task_events WHERE task_id = ? "
                "ORDER BY id DESC LIMIT 1", (task_id,),
            ).fetchone()
            if task is not None and task["project_id"] == project.id and event is not None:
                payload = json.loads(event["payload"])
                if (
                    event["kind"] == "project_bound"
                    and payload.get("from_project_id") is None
                    and payload.get("to_project_id") == project.id
                ):
                    committed_state.append({
                        "in_transaction": candidate_conn.in_transaction,
                        "project_id": task["project_id"],
                        "event_kind": event["kind"],
                        "event_payload": payload,
                    })
                    raise sqlite3.DatabaseError("injected post-COMMIT invariant failure")
        original_check(candidate_conn)

    monkeypatch.setattr(kbc, "_check_file_length_invariant", fail_after_committed_binding)

    code, result, captured = _invoke_bind(task_id, project.id, capsys)

    assert len(committed_state) == 1, "injection must occur only after task and event COMMIT"
    assert committed_state[0]["in_transaction"] is False
    assert committed_state[0]["project_id"] == project.id
    assert committed_state[0]["event_kind"] == "project_bound"
    assert committed_state[0]["event_payload"]["project_slug"] == project.slug

    # Verify persistence through a fresh connection before asserting the CLI
    # response contract; a RED result must not skip the COMMIT evidence.
    with kbc.connect_closing() as read_conn:
        task_after = dict(read_conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,),
        ).fetchone())
        events_after = _rows(
            read_conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
        )
    assert task_after["project_id"] == project.id
    assert task_after != task_before
    assert events_after[:len(events_before)] == events_before
    assert len(events_after) == len(events_before) + 1
    assert events_after[-1]["kind"] == "project_bound"
    persisted_payload = json.loads(events_after[-1]["payload"])
    assert persisted_payload["to_project_id"] == project.id
    assert persisted_payload["project_slug"] == project.slug

    assert code == 1
    assert result is not None
    assert result["ok"] is False
    assert (result["result"], result["changed"]) == ("reconciliation_required", None)
    assert result["task_id"] == task_id
    assert result["project_id"] is None
    assert result["project_slug"] is None
    assert result["board"] == "default"
    assert result["binding_state"] == "indeterminate"
    assert result["reconciliation_required"] is True
    assert "injected post-COMMIT invariant failure" in result["error"]
    assert captured.err == ""
    assert json.loads(captured.out) == result
    assert captured.out.count("\n") >= 1

    with kbc.connect_closing() as read_conn:
        task_after = dict(read_conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,),
        ).fetchone())
        events_after = _rows(
            read_conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
        )
    assert task_after["project_id"] == project.id
    assert task_after != task_before
    assert events_after[:len(events_before)] == events_before
    assert len(events_after) == len(events_before) + 1
    assert events_after[-1]["kind"] == "project_bound"
    persisted_payload = json.loads(events_after[-1]["payload"])
    assert persisted_payload["to_project_id"] == project.id
    assert persisted_payload["project_slug"] == project.slug


def test_bind_project_missing_result_after_commit_is_indeterminate(
    kanban_conn, capsys, monkeypatch,
):
    conn, tmp_path = kanban_conn
    project = _make_project(tmp_path)
    task_id = _make_projectless_task(conn, project)
    task_before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))
    original_bind = project_binding.bind_task_project
    bind_calls = []

    def commit_without_returning_result(*args, **kwargs):
        bind_calls.append(1)
        committed_result = original_bind(*args, **kwargs)
        assert committed_result["changed"] is True

    monkeypatch.setattr(project_binding, "bind_task_project", commit_without_returning_result)

    code, result, captured = _invoke_bind(task_id, project.id, capsys)

    assert code == 1
    assert len(bind_calls) == 1
    assert result is not None
    assert result["ok"] is False
    assert result["result"] == "reconciliation_required"
    assert result["changed"] is None
    assert result["binding_state"] == "indeterminate"
    assert result["reconciliation_required"] is True
    assert result["task_id"] == task_id
    assert result["project_id"] is None
    assert result["project_slug"] is None
    assert result["board"] == "default"
    assert captured.err == ""
    assert json.loads(captured.out) == result

    with kbc.connect_closing() as read_conn:
        task_after = dict(read_conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,),
        ).fetchone())
        events_after = _rows(
            read_conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
        )
    assert task_before["project_id"] is None
    assert task_after["project_id"] == project.id
    assert task_after != task_before
    assert events_after[:len(events_before)] == events_before
    assert len(events_after) == len(events_before) + 1
    assert events_after[-1]["kind"] == "project_bound"
    payload = json.loads(events_after[-1]["payload"])
    assert payload["to_project_id"] == project.id
    assert payload["project_slug"] == project.slug


def test_concurrent_other_project_binding_is_not_attributed_to_command(
    kanban_conn, capsys, monkeypatch,
):
    conn, tmp_path = kanban_conn
    project_a = _make_project(tmp_path, name="Concurrent Project A")
    project_b = _make_project(tmp_path, name="Requested Project B")
    task_id = _make_projectless_task(conn, project_a)
    task_before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
    events_before = _rows(conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))

    snapshot_calls = []
    writer_commit = []
    original_snapshot = kanban_cli._bind_project_snapshot
    original_bind = project_binding.bind_task_project

    def record_snapshot(candidate_conn, candidate_task_id):
        snapshot_calls.append(candidate_conn)
        return original_snapshot(candidate_conn, candidate_task_id)

    def bind_after_concurrent_writer(command_conn, candidate_task_id, project_ref, **kwargs):
        assert candidate_task_id == task_id
        assert project_ref == project_b.id
        assert len(snapshot_calls) == 1
        assert snapshot_calls[0] is command_conn

        with kbc.connect_closing() as writer_conn:
            assert writer_conn is not command_conn
            writer_result = original_bind(
                writer_conn, task_id, project_a.id,
                board=kwargs["board"], actor="concurrent-writer",
            )
            assert writer_result["changed"] is True
            assert writer_conn.in_transaction is False

            with kbc.connect_closing() as committed_conn:
                committed_task = committed_conn.execute(
                    "SELECT project_id FROM tasks WHERE id = ?", (task_id,),
                ).fetchone()
                committed_events = _rows(
                    committed_conn,
                    "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
                )
            assert committed_task["project_id"] == project_a.id
            assert len(committed_events) == len(events_before) + 1
            committed_event = committed_events[-1]
            committed_payload = json.loads(committed_event["payload"])
            assert committed_event["kind"] == "project_bound"
            assert committed_payload["to_project_id"] == project_a.id
            assert committed_payload["actor"] == "concurrent-writer"
            writer_commit.append({
                "transaction_open_after_bind": writer_conn.in_transaction,
                "task_project_id": committed_task["project_id"],
                "event_kind": committed_event["kind"],
                "event_payload": committed_payload,
            })

        return original_bind(command_conn, candidate_task_id, project_ref, **kwargs)

    monkeypatch.setattr(kanban_cli, "_bind_project_snapshot", record_snapshot)
    monkeypatch.setattr(project_binding, "bind_task_project", bind_after_concurrent_writer)

    code, result, captured = _invoke_bind(task_id, project_b.id, capsys)

    assert len(snapshot_calls) == 2
    assert len(writer_commit) == 1
    assert writer_commit[0]["transaction_open_after_bind"] is False
    assert writer_commit[0]["task_project_id"] == project_a.id
    assert writer_commit[0]["event_kind"] == "project_bound"

    with kbc.connect_closing() as read_conn:
        task_after = dict(read_conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,),
        ).fetchone())
        events_after = _rows(
            read_conn, "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
        )
    assert task_before["project_id"] is None
    assert task_after["project_id"] == project_a.id
    assert events_after[:len(events_before)] == events_before
    assert len(events_after) == len(events_before) + 1
    assert events_after[-1]["kind"] == "project_bound"
    assert json.loads(events_after[-1]["payload"])["to_project_id"] == project_a.id
    assert captured.err == ""
    assert captured.out.strip()
    assert result is not None
    assert json.loads(captured.out) == result

    expected_contract = (
        1, False, "reconciliation_required", None, "indeterminate", True, None, None,
    )
    actual_contract = (
        code, result.get("ok"), result.get("result"), result.get("changed"),
        result.get("binding_state"), result.get("reconciliation_required"),
        result.get("project_id"), result.get("project_slug"),
    )
    assert actual_contract == expected_contract, (
        f"command_json={captured.out.strip()}; "
        f"final_task_project={task_after['project_id']}; "
        f"final_project_bound_events={len(events_after) - len(events_before)}; "
        f"writer_commit={json.dumps(writer_commit[0], sort_keys=True)}"
    )
    assert result["task_id"] == task_id
    assert result["board"] == "default"
    assert "already bound" in result["error"].lower()
    assert project_a.id in result["error"]
