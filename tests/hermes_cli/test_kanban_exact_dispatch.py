"""Atomic exact-task Kanban dispatch regressions for issue #95900."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: bool(name))
    monkeypatch.setattr(kb, "_memory_pressure_level", lambda: "normal")
    return db_path


def _spawn(pid=4242, calls=None):
    def inner(task, workspace, board=None):
        if calls is not None:
            calls.append(task.id)
        return pid

    return inner


def test_exact_success_returns_selected_claimed_spawned_run_evidence(board):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="exact", assignee="worker")
        result = kb.dispatch_task_once(
            conn, task_id, spawn_fn=_spawn(), board="default"
        )
        task = kb.get_task(conn, task_id)

    assert result.ok is True
    assert result.reason == "spawned"
    assert result.selected == task_id
    assert result.claimed == task_id
    assert result.spawned == task_id
    assert result.run_id == task.current_run_id
    assert result.worker_pid == 4242


def test_exact_repeat_is_idempotent_for_same_run_and_does_not_respawn(board):
    calls = []
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="exact", assignee="worker")
        first = kb.dispatch_task_once(
            conn, task_id, spawn_fn=_spawn(4242, calls), board="default"
        )
        second = kb.dispatch_task_once(
            conn, task_id, spawn_fn=_spawn(9999, calls), board="default"
        )

    assert calls == [task_id]
    assert second.ok is True
    assert second.idempotent is True
    assert second.run_id == first.run_id
    assert second.worker_pid == 4242


def test_exact_honors_default_assignee_atomically(board):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="fallback", assignee=None)
        result = kb.dispatch_task_once(
            conn,
            task_id,
            spawn_fn=_spawn(),
            board="default",
            default_assignee="worker",
        )
        task = kb.get_task(conn, task_id)

    assert result.ok is True
    assert task is not None
    assert task.assignee == "worker"
    assert task.status == "running"


def test_exact_respects_emergency_stop_without_mutation(board):
    Path(board).parent.joinpath("ESTOP").write_text("{}", encoding="utf-8")
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="paused", assignee="worker")
        before = dict(
            conn.execute(
                "SELECT status, claim_lock, current_run_id FROM tasks WHERE id=?",
                (task_id,),
            ).fetchone()
        )
        result = kb.dispatch_task_once(
            conn, task_id, spawn_fn=_spawn(), board="default"
        )
        after = dict(
            conn.execute(
                "SELECT status, claim_lock, current_run_id FROM tasks WHERE id=?",
                (task_id,),
            ).fetchone()
        )

    assert result.ok is False
    assert result.reason == "dispatch_suspended"
    assert after == before


def test_exact_spawn_failure_reports_closed_run_without_spawn_evidence(board):
    def fail_spawn(task, workspace, board=None):
        raise RuntimeError("synthetic spawn failure")

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="fails", assignee="worker")
        result = kb.dispatch_task_once(
            conn,
            task_id,
            spawn_fn=fail_spawn,
            board="default",
            failure_limit=2,
        )
        run = conn.execute(
            "SELECT status, outcome FROM task_runs WHERE id=?", (result.run_id,)
        ).fetchone()

    assert result.ok is False
    assert result.reason == "spawn_failed"
    assert result.selected == task_id
    assert result.claimed == task_id
    assert result.spawned is None
    assert result.run_status == "spawn_failed"
    assert dict(run) == {"status": "spawn_failed", "outcome": "spawn_failed"}


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        ("missing", "not_found"),
        ("blocked", "status_ineligible"),
        ("dependency", "parents_not_done"),
        ("unassigned", "unassigned"),
        ("profile_cap", "profile_cap"),
        ("cooldown", "rate_limit_cooldown"),
        ("workspace", "workspace_invalid"),
    ],
)
def test_exact_ineligible_fails_closed_without_mutation(
    board, setup, reason, monkeypatch
):
    with kb.connect() as conn:
        if setup == "missing":
            task_id = "t_missing"
        else:
            kwargs = {
                "title": setup,
                "assignee": None if setup == "unassigned" else "worker",
            }
            task_id = kb.create_task(conn, **kwargs)
        if setup == "blocked":
            kb.block_task(conn, task_id, reason="hold")
        elif setup == "dependency":
            parent = kb.create_task(conn, title="parent", assignee="worker")
            kb.link_tasks(conn, parent, task_id)
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        elif setup == "profile_cap":
            other = kb.create_task(conn, title="running", assignee="worker")
            assert kb.claim_task(conn, other) is not None
        elif setup == "cooldown":
            now = __import__("time").time()
            with kb.write_txn(conn):
                cur = conn.execute(
                    "INSERT INTO task_runs(task_id, profile, status, outcome, started_at, ended_at) "
                    "VALUES (?, 'worker', 'rate_limited', 'rate_limited', ?, ?)",
                    (task_id, now - 1, now),
                )
                conn.execute(
                    "UPDATE tasks SET last_failure_error='rate limit exceeded' WHERE id=?",
                    (task_id,),
                )
        elif setup == "workspace":
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET workspace_kind='dir', workspace_path='relative/path' WHERE id=?",
                    (task_id,),
                )

        before = conn.execute(
            "SELECT status, assignee, claim_lock, current_run_id FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        result = kb.dispatch_task_once(
            conn,
            task_id,
            spawn_fn=_spawn(),
            board="default",
            max_in_progress_per_profile=1 if setup == "profile_cap" else None,
        )
        after = conn.execute(
            "SELECT status, assignee, claim_lock, current_run_id FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()

    assert result.ok is False
    assert result.reason == reason
    assert dict(after) == dict(before) if before is not None else after is None


def test_exact_dispatch_loses_lock_without_falling_through_or_mutating(board):
    calls = []
    with kb.connect() as conn:
        named = kb.create_task(conn, title="named", assignee="worker")
        other = kb.create_task(conn, title="other", assignee="worker")
        with kb._dispatch_tick_lock(board) as held:
            assert held
            result = kb.dispatch_task_once(
                conn, named, spawn_fn=_spawn(calls=calls), board="default"
            )
        rows = {
            row["id"]: row["status"]
            for row in conn.execute("SELECT id, status FROM tasks")
        }

    assert result.ok is False
    assert result.reason == "dispatch_locked"
    assert calls == []
    assert rows == {named: "ready", other: "ready"}


def test_cli_exact_json_is_strict_and_feature_detectable(board, monkeypatch, capsys):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="exact", assignee="worker")
    monkeypatch.setattr(kb, "_default_spawn", _spawn())
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {}})

    args = argparse.Namespace(
        task_id=task_id,
        dry_run=False,
        max=None,
        failure_limit=2,
        json=True,
    )
    assert kanban_cli._cmd_dispatch(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload == {
        "capability": "kanban.exact-task-dispatch",
        "version": 1,
        "ok": True,
        "reason": "spawned",
        "idempotent": False,
        "selected": {"task_id": task_id},
        "claimed": {"task_id": task_id, "run_id": payload["claimed"]["run_id"]},
        "spawned": {"task_id": task_id, "worker_pid": 4242},
        "run": {"id": payload["run"]["id"], "status": "running"},
    }
    assert payload["claimed"]["run_id"] == payload["run"]["id"]


def test_cli_exact_requires_json(board, capsys):
    args = argparse.Namespace(
        task_id="t_any",
        dry_run=False,
        max=None,
        failure_limit=2,
        json=False,
    )
    assert kanban_cli._cmd_dispatch(args) == 2
    assert "--task-id requires --json" in capsys.readouterr().err


def test_cli_exact_failure_is_strict_json_with_zero_evidence(
    board, monkeypatch, capsys
):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {}})
    args = argparse.Namespace(
        task_id="t_missing",
        dry_run=False,
        max=None,
        failure_limit=2,
        json=True,
    )
    assert kanban_cli._cmd_dispatch(args) == 1
    assert json.loads(capsys.readouterr().out) == {
        "capability": "kanban.exact-task-dispatch",
        "version": 1,
        "ok": False,
        "reason": "not_found",
        "idempotent": False,
        "selected": None,
        "claimed": None,
        "spawned": None,
        "run": None,
    }
