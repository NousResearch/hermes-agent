"""Human CLI for inspecting and resolving fenced protocol retry holds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban import kanban_command
from hermes_cli.kanban_parser import build_parser


@pytest.fixture
def isolated_board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


def _seed_held_task(conn):
    task_id = kb.create_task(conn, title="protocol hold", body="stable body", assignee="worker")
    failure_event_id = None
    failed_run_id = None
    now = 1_800_000_000
    with kb.write_txn(conn):
        for offset in range(1, 4):
            run = conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, error, metadata) "
                "VALUES (?, 'worker', 'crashed', ?, ?, 'crashed', 'protocol violation', ?)",
                (task_id, now + offset, now + offset, json.dumps({"protocol_violation": True})),
            )
            failed_run_id = int(run.lastrowid)
            event = conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, ?, 'protocol_violation', ?, ?)",
                (task_id, failed_run_id, json.dumps({
                    "protocol_violation": True, "exit_code": 0, "retry_status": "ready",
                }), now + offset),
            )
            failure_event_id = int(event.lastrowid)
        conn.execute(
            "UPDATE tasks SET status='blocked', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL, last_failure_error='protocol violation' WHERE id=?",
            (task_id,),
        )
        kb._append_event(conn, task_id, "gave_up", {
            "sticky": True, "protocol_violations": 3, "protocol_violation_limit": 3,
            "trigger_outcome": "crashed", "retry_status": "ready",
        }, run_id=None)
    body_sha256 = hashlib.sha256(kb.get_task(conn, task_id).body.encode("utf-8")).hexdigest()
    hold = kb.adopt_protocol_failure(
        conn, task_id, expected_failed_run_id=failed_run_id,
        expected_failure_event_id=failure_event_id, expected_body_sha256=body_sha256,
    )
    assert hold["status"] == "scheduled"
    return task_id, hold


def _run_cli(*args: str) -> int:
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_parser(subparsers)
    return kanban_command(parser.parse_args(["kanban", *args]))


def test_holds_list_json_and_resolve_are_audited_and_idempotent(isolated_board, capsys):
    conn = isolated_board
    task_id, hold = _seed_held_task(conn)

    assert _run_cli("--board", "default", "holds", "list", "--json") == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed == {"holds": [hold]}

    resolve_args = (
        "--board", "default", "holds", "resolve", task_id,
        "--action", "resume", "--expected-hold-token", hold["hold_token"],
        "--reason", "operator_reviewed",
    )
    assert _run_cli(*resolve_args) == 0
    first = capsys.readouterr().out
    assert "Resolved" in first and "resume" in first
    assert kb.get_task(conn, task_id).status == "ready"

    assert _run_cli(*resolve_args) == 0
    second = capsys.readouterr().out
    assert "Already resolved" in second
    resolutions = conn.execute(
        "SELECT resolution_event_id FROM task_retry_hold_resolutions WHERE task_id = ?",
        (task_id,),
    ).fetchall()
    events = conn.execute(
        "SELECT id FROM task_events WHERE task_id = ? AND kind = 'retry_hold_resolved'",
        (task_id,),
    ).fetchall()
    assert len(resolutions) == len(events) == 1


def test_holds_list_human_and_retire_audits_non_grade(isolated_board, capsys):
    conn = isolated_board
    task_id, hold = _seed_held_task(conn)

    assert _run_cli("--board", "default", "holds", "list") == 0
    listing = capsys.readouterr().out
    assert task_id in listing and hold["hold_token"] in listing

    assert _run_cli(
        "--board", "default", "holds", "resolve", task_id,
        "--action", "retire", "--expected-hold-token", hold["hold_token"],
        "--reason", "human_retirement",
    ) == 0
    assert "retire -> archived" in capsys.readouterr().out
    assert kb.get_task(conn, task_id).status == "archived"
    assert kb.get_retry_hold(conn, task_id) is None
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'retry_retired'",
        (task_id,),
    ).fetchone()
    assert json.loads(event["payload"])["outcome"] == "retry_retired_non_grade"


def test_holds_resolve_resume_obeys_parent_gate(isolated_board, capsys):
    conn = isolated_board
    task_id, hold = _seed_held_task(conn)
    parent_id = kb.create_task(conn, title="open parent", assignee="parent-worker")
    kb.link_tasks(conn, parent_id, task_id)
    hold = kb.get_retry_hold(conn, task_id)

    assert _run_cli(
        "--board", "default", "holds", "resolve", task_id,
        "--action", "resume", "--expected-hold-token", hold["hold_token"],
        "--reason", "parent_gate_check",
    ) == 0
    capsys.readouterr()
    assert kb.get_task(conn, task_id).status == "todo"


def test_holds_resolve_retire_refuses_live_dependents(isolated_board, capsys):
    conn = isolated_board
    task_id, hold = _seed_held_task(conn)
    child_id = kb.create_task(conn, title="dependent", assignee="child-worker")
    kb.link_tasks(conn, task_id, child_id)

    assert _run_cli(
        "--board", "default", "holds", "resolve", task_id,
        "--action", "retire", "--expected-hold-token", hold["hold_token"],
        "--reason", "retire_reviewed",
    ) == 1
    assert "live_dependents_would_be_released" in capsys.readouterr().err
    assert kb.get_retry_hold(conn, task_id) == hold
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'retry_retired'",
        (task_id,),
    ).fetchone()[0] == 0


def test_holds_resolve_rejects_stale_token_without_releasing_task(isolated_board, capsys):
    conn = isolated_board
    task_id, hold = _seed_held_task(conn)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET body = ? WHERE id = ?", ("edited body", task_id))

    assert _run_cli(
        "--board", "default", "holds", "resolve", task_id,
        "--action", "resume", "--expected-hold-token", hold["hold_token"],
        "--reason", "stale_token_test",
    ) == 1
    assert "stale_hold_token" in capsys.readouterr().err
    assert kb.get_task(conn, task_id).status == "scheduled"
    assert kb.get_retry_hold(conn, task_id)["hold_token"] != hold["hold_token"]


def test_unblock_refusal_points_to_holds_resolve(isolated_board, capsys):
    conn = isolated_board
    task_id, _hold = _seed_held_task(conn)

    assert _run_cli("--board", "default", "unblock", task_id) == 1
    assert "hermes kanban holds resolve" in capsys.readouterr().err
    assert kb.get_task(conn, task_id).status == "scheduled"


def test_holds_resolve_is_operator_only(isolated_board, monkeypatch, capsys):
    conn = isolated_board
    task_id, hold = _seed_held_task(conn)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "worker-task")

    assert _run_cli(
        "--board", "default", "holds", "resolve", task_id,
        "--action", "resume", "--expected-hold-token", hold["hold_token"],
        "--reason", "worker_refusal_check",
    ) == 1
    assert "operator-only" in capsys.readouterr().err
    assert kb.get_retry_hold(conn, task_id) == hold
