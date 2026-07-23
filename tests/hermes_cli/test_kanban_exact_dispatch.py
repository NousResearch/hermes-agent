"""Focused regressions for native exact-task Kanban dispatch."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_exact_dispatch_success_is_idempotent_and_structured(
    kanban_home, all_assignees_spawnable
):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="exact", assignee="alice")
        result = kb.dispatch_task(conn, task_id, spawn_fn=lambda *_: os.getpid())

        assert result == kb.ExactDispatchResult(
            task_id=task_id,
            state="spawned",
            spawned=True,
            run_id=kb.get_task(conn, task_id).current_run_id,
            pid=os.getpid(),
            assignee="alice",
            workspace=str(kb.workspaces_root() / task_id),
        )

        repeated = kb.dispatch_task(conn, task_id, spawn_fn=lambda *_: 99999)
        assert repeated.state == "already_running"
        assert repeated.spawned is False
        assert repeated.run_id == result.run_id
        assert repeated.pid == result.pid


def test_exact_dispatch_never_falls_back(kanban_home, all_assignees_spawnable):
    calls = []
    with kb.connect() as conn:
        requested = kb.create_task(conn, title="blocked", assignee="alice")
        other = kb.create_task(conn, title="ready", assignee="bob", priority=100)
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (requested,))

        result = kb.dispatch_task(
            conn, requested, spawn_fn=lambda task, workspace: calls.append(task.id)
        )

        assert result.state == "refused"
        assert result.reason == "state:done"
        assert calls == []
        assert kb.get_task(conn, other).status == "ready"


def test_exact_dispatch_refuses_dependencies_and_bad_workspace(
    kanban_home, all_assignees_spawnable
):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        child = kb.create_task(
            conn, title="child", assignee="bob", parents=[parent]
        )
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))
        dependency = kb.dispatch_task(conn, child, spawn_fn=lambda *_: 123)
        assert dependency.reason == "parents_not_done"
        assert kb.get_task(conn, child).status == "ready"

        bad_workspace = kb.create_task(
            conn,
            title="bad workspace",
            assignee="carol",
            workspace_kind="dir",
            workspace_path="relative/path",
        )
        workspace = kb.dispatch_task(conn, bad_workspace, spawn_fn=lambda *_: 123)
        assert workspace.state == "refused"
        assert workspace.reason.startswith("workspace:")
        assert kb.get_task(conn, bad_workspace).status == "ready"


def test_exact_dispatch_reports_cross_board_profile_capacity(
    kanban_home, all_assignees_spawnable, monkeypatch
):
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: True)
    kb.create_board("other", name="Other")
    with kb.connect(board="other") as other_conn:
        occupied_id = kb.create_task(other_conn, title="occupied", assignee="alice")
        occupied = kb.claim_task(other_conn, occupied_id)
        kb._set_worker_pid(other_conn, occupied_id, 4242)
        occupied_run = occupied.current_run_id

    with kb.connect(board="default") as conn:
        target = kb.create_task(conn, title="target", assignee="alice")
        result = kb.dispatch_task(
            conn, target, board="default", spawn_fn=lambda *_: 9999
        )
        assert result.state == "capacity"
        assert result.reason == "profile_occupied"
        assert result.capacity == [{
            "board": "other",
            "task_id": occupied_id,
            "run_id": occupied_run,
            "pid": 4242,
        }]
        assert kb.get_task(conn, target).status == "ready"


def test_exact_dispatch_reclaims_dead_target_and_retries(
    kanban_home, all_assignees_spawnable, monkeypatch
):
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: pid == 2222)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="retry", assignee="alice")
        kb.claim_task(conn, task_id)
        kb._set_worker_pid(conn, task_id, 1111)
        conn.execute(
            "UPDATE tasks SET started_at = ? WHERE id = ?",
            (int(time.time()) - kb.DEFAULT_CRASH_GRACE_SECONDS - 1, task_id),
        )

        result = kb.dispatch_task(conn, task_id, spawn_fn=lambda *_: 2222)

        assert result.state == "spawned"
        assert result.pid == 2222
        assert kb.get_task(conn, task_id).status == "running"
        assert [run.outcome for run in kb.list_runs(conn, task_id)] == ["crashed", None]


def test_exact_dispatch_cli_parser_accepts_one_task_id():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    kanban_cli.build_parser(subparsers)

    args = parser.parse_args(["kanban", "dispatch", "t_exact", "--json"])
    assert args.task_id == "t_exact"
    assert args.json is True


def test_exact_dispatch_cli_json_shape(monkeypatch, capsys):
    expected = kb.ExactDispatchResult(
        task_id="t_exact",
        state="spawned",
        spawned=True,
        run_id=7,
        pid=1234,
        assignee="alice",
        workspace="/tmp/exact",
    )
    monkeypatch.setattr(kb, "dispatch_task", lambda *args, **kwargs: expected)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})

    args = argparse.Namespace(
        task_id="t_exact", dry_run=False, max=None, failure_limit=2, json=True
    )
    assert kanban_cli._cmd_dispatch(args) == 0
    assert json.loads(capsys.readouterr().out) == {
        "task_id": "t_exact",
        "state": "spawned",
        "spawned": True,
        "run_id": 7,
        "pid": 1234,
        "assignee": "alice",
        "workspace": "/tmp/exact",
        "reason": None,
        "capacity": [],
    }
