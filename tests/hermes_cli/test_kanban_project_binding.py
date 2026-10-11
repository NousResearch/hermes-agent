"""Safe, workspace-preserving binding of existing Kanban tasks to Projects."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

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
