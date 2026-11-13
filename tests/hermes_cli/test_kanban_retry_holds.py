"""Regression tests for protocol-violation retry holds."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import config
import hermes_cli.cli_single_query as single_query
from hermes_state import SessionDB
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def isolated_board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


def _exited_status(code: int) -> int:
    return code << 8


_TOOLS = [{"type": "function", "function": {"name": "safe_tool", "parameters": {"type": "object"}}}]


def _schema_sha256(tools) -> str:
    return hashlib.sha256(
        json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _protocol_streak(conn, task_id):
    from hermes_cli import kanban_retry_holds as retry_holds

    return retry_holds._protocol_violation_streak(conn, task_id)


def test_opted_in_protocol_exhaustion_creates_one_scheduled_hold(
    isolated_board, monkeypatch,
):
    """A real dispatch/reap sequence parks opted-in work after exactly three exits."""
    conn = isolated_board
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "unknown")
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda _home: None)
    monkeypatch.setattr(
        config,
        "load_config_readonly",
        lambda: {
            "kanban": {
                "protocol_violation_actions": {
                    "default": {"jobseek-worker": "defer"},
                },
            },
        },
    )
    reaper_statuses = []
    account_crashes = kbd._account_crashes

    def observe_account(conn, crash_details):
        reaper_statuses.extend(
            kb.get_task(conn, task_id).status
            for task_id, _pid, _claimer, dead in crash_details
            if dead.protocol_violation
        )
        return account_crashes(conn, crash_details)

    monkeypatch.setattr(kbd, "_account_crashes", observe_account)

    task_id = kb.create_task(conn, title="protocol retry", assignee="jobseek-worker")
    pids = iter((91001, 91002, 91003, 91004))
    spawned = []

    def spawn(_task, _workspace):
        pid = next(pids)
        spawned.append(pid)
        return pid

    first = kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=4)
    assert first.spawned and spawned == [91001]

    for pid in (91001, 91002, 91003):
        kbd._record_worker_exit(pid, _exited_status(0))
        kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=4)

    task = kb.get_task(conn, task_id)
    assert task.status == "scheduled"
    assert spawned == [91001, 91002, 91003]
    assert reaper_statuses == ["ready", "ready", "scheduled"]
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='protocol_retry_held'",
        (task_id,),
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='gave_up'",
        (task_id,),
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id=? AND outcome='crashed'",
        (task_id,),
    ).fetchone()[0] == 3

    # A subsequent dispatch tick cannot produce an automatic fourth run.
    fourth_tick = kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=4)
    assert fourth_tick.spawned == []
    assert spawned == [91001, 91002, 91003]


def test_default_protocol_exhaustion_remains_blocking(isolated_board, monkeypatch):
    """The retry-hold policy is opt-in; no config preserves existing blocking."""
    conn = isolated_board
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "unknown")
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda _home: None)
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"kanban": {}})

    task_id = kb.create_task(conn, title="default protocol", assignee="worker")
    pids = iter((92001, 92002, 92003))
    spawned = []

    def spawn(_task, _workspace):
        pid = next(pids)
        spawned.append(pid)
        return pid

    kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=4)
    for pid in (92001, 92002, 92003):
        kbd._record_worker_exit(pid, _exited_status(0))
        kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=4)

    task = kb.get_task(conn, task_id)
    assert task.status == "blocked"
    assert spawned == [92001, 92002, 92003]
    gave_up = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='gave_up'",
        (task_id,),
    ).fetchone()
    assert json.loads(gave_up["payload"])["protocol_violations"] == 3


def test_policy_is_scoped_and_invalid_values_fail_closed(isolated_board, monkeypatch):
    conn = isolated_board
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "unknown")
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda _home: None)
    scenarios = [
        ("default", "other-worker", {"default": {"jobseek-worker": "defer"}}),
        ("other-board", "jobseek-worker", {"default": {"jobseek-worker": "defer"}}),
        ("default", "jobseek-worker", {"default": {"jobseek-worker": ["defer"]}}),
        ("default", "jobseek-worker", {"default": {"jobseek-worker": "block"}}),
    ]
    next_pid = 94000
    for index, (board, assignee, actions) in enumerate(scenarios):
        monkeypatch.setattr(
            config, "load_config_readonly",
            lambda actions=actions: {"kanban": {"protocol_violation_actions": actions}},
        )
        task_id = kb.create_task(conn, title=f"fail-closed-{index}", assignee=assignee)
        pids = iter(range(next_pid, next_pid + 3))
        spawned = []

        def spawn(_task, _workspace):
            pid = next(pids)
            spawned.append(pid)
            return pid

        kbd.dispatch_once(conn, spawn_fn=spawn, board=board, max_spawn=4)
        # Keep the same exact three worker exits used by the opted-in path.
        for _ in range(3):
            pid = spawned[-1]
            kbd._record_worker_exit(pid, _exited_status(0))
            kbd.dispatch_once(conn, spawn_fn=spawn, board=board, max_spawn=4)
        assert kb.get_task(conn, task_id).status == "blocked", index
        assert len(spawned) == 3, index
        assert conn.execute(
            "SELECT COUNT(*) FROM task_retry_holds WHERE task_id=?", (task_id,),
        ).fetchone()[0] == 0, index
        next_pid += 3


def test_defer_policy_does_not_hold_other_crashes_auth_or_explicit_blocks(isolated_board, monkeypatch):
    conn = isolated_board
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "unknown")
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda _home: None)
    monkeypatch.setattr(
        config, "load_config_readonly",
        lambda: {"kanban": {"protocol_violation_actions": {"default": {"worker": "defer"}}}},
    )
    pid = 95000
    task_ids = []
    for exit_code in (1, 78):
        task_id = kb.create_task(conn, title=f"other-failure-{exit_code}", assignee="worker")
        task_ids.append(task_id)

        def spawn(_task, _workspace, current_pid=pid):
            return current_pid

        kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=1)
        kbd._record_worker_exit(pid, _exited_status(exit_code))
        kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=0)
        if exit_code == 1:
            assert kb.get_task(conn, task_id).status == "ready"
            assert kb.archive_task(conn, task_id) is True
        else:
            assert kb.get_task(conn, task_id).status == "blocked"
        pid += 1

    explicit = kb.create_task(conn, title="explicit block", assignee="worker")
    task_ids.append(explicit)
    assert kb.block_task(conn, explicit, reason="operator request", kind="needs_input") is True

    for task_id in task_ids:
        assert kb.get_retry_hold(conn, task_id) is None
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='protocol_retry_held'",
            (task_id,),
        ).fetchone()[0] == 0


def test_retry_hold_schema_and_public_api_exist(isolated_board):
    """The new retry-hold tables and trusted recovery API exist on fresh boards."""
    conn = isolated_board
    tables = {
        row["name"] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"task_retry_holds", "task_retry_hold_resolutions"} <= tables
    assert config.DEFAULT_CONFIG["kanban"]["protocol_violation_actions"] == {}
    assert callable(getattr(kb, "get_retry_hold", None))
    assert callable(getattr(kb, "adopt_protocol_failure", None))
    assert callable(getattr(kb, "resolve_retry_hold", None))


def _seed_legacy_protocol_trip(
    conn, *, explicit_block=False, duplicate_source_event=False, workspace_path=None,
):
    """Seed a bounded old-format protocol ``gave_up`` without touching a live board."""
    task_id = kb.create_task(
        conn, title="legacy protocol", body="stable body", assignee="worker",
        workspace_kind="scratch", workspace_path=str(workspace_path) if workspace_path else None,
    )
    now = 1_800_000_000
    failed_run_id = None
    failure_event_id = None
    with kb.write_txn(conn):
        for offset in range(1, 4):
            run = conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, error, metadata) "
                "VALUES (?, 'worker', 'crashed', ?, ?, 'crashed', ?, ?)",
                (task_id, now + offset, now + offset, "protocol violation", json.dumps({"protocol_violation": True})),
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
            if duplicate_source_event and offset == 3:
                conn.execute(
                    "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                    "VALUES (?, ?, 'protocol_violation', ?, ?)",
                    (task_id, failed_run_id, json.dumps({
                        "protocol_violation": True, "exit_code": 0, "retry_status": "ready",
                    }), now + offset + 1),
                )
        conn.execute(
            "UPDATE tasks SET status='blocked', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL, last_failure_error='protocol violation' WHERE id=?",
            (task_id,),
        )
        kb._append_event(conn, task_id, "gave_up", {
            "sticky": True, "protocol_violations": 3, "protocol_violation_limit": 3,
            "trigger_outcome": "crashed", "retry_status": "ready",
        }, run_id=None)
        if explicit_block:
            kb._append_event(conn, task_id, "blocked", {"kind": "needs_input", "reason": "manual"})
    task = kb.get_task(conn, task_id)
    body_sha256 = hashlib.sha256(task.body.encode("utf-8")).hexdigest()
    return task_id, failed_run_id, failure_event_id, body_sha256


def test_adopt_then_resume_preserves_history_and_scopes_the_next_streak(isolated_board):
    conn = isolated_board
    task_id, failed_run_id, event_id, body_sha256 = _seed_legacy_protocol_trip(conn)
    before_runs = [dict(row) for row in conn.execute(
        "SELECT id, outcome, error, metadata FROM task_runs WHERE task_id=? ORDER BY id", (task_id,),
    )]
    before_error = kb.get_task(conn, task_id).last_failure_error

    wrong_body = kb.adopt_protocol_failure(
        conn, task_id, expected_failed_run_id=failed_run_id,
        expected_failure_event_id=event_id, expected_body_sha256="0" * 64,
    )
    assert wrong_body["ok"] is False
    assert wrong_body["reason"] == "body_changed"
    assert kb.get_task(conn, task_id).status == "blocked"

    hold = kb.adopt_protocol_failure(
        conn, task_id, expected_failed_run_id=failed_run_id,
        expected_failure_event_id=event_id, expected_body_sha256=body_sha256,
    )
    assert hold["hold_token"]
    assert set(hold) == {
        "task_id", "failed_run_id", "source_failure_event_id", "hold_token",
        "body_sha256", "assignee", "status", "metadata",
    }
    assert hold["status"] == "scheduled"
    assert hold["failed_run_id"] == failed_run_id
    assert not any(term in json.dumps(hold).lower() for term in (
        "prompt", "arguments", "credential", "grade", "tool_call",
    ))
    assert kb.get_retry_hold(conn, task_id) == hold
    assert kb.adopt_protocol_failure(
        conn, task_id, expected_failed_run_id=failed_run_id,
        expected_failure_event_id=event_id, expected_body_sha256=body_sha256,
    ) == hold
    assert kb.unblock_task(conn, task_id) is False
    assert kb.block_task(conn, task_id, reason="manual", kind="needs_input") is False
    assert kb.complete_task(conn, task_id, result="manual completion") is False
    assert kb.archive_task(conn, task_id) is False
    assert kb.delete_task(conn, task_id) is False
    assert kb.recompute_ready(conn) == 0
    assert kb.claim_task(conn, task_id) is None
    unsafe_receipt = kb.resolve_retry_hold(
        conn, task_id, expected_hold_token=hold["hold_token"], action="resume",
        reason="verified_current_source", metadata={"source": "raw grade 8"},
    )
    assert unsafe_receipt["ok"] is False
    assert unsafe_receipt["reason"] == "invalid_metadata_string"

    receipt = kb.resolve_retry_hold(
        conn, task_id, expected_hold_token=hold["hold_token"], action="resume",
        reason="verified_current_source", metadata={"source": "operator", "schema_sha256": "a" * 64},
    )
    assert receipt["ok"] is True
    assert receipt["status"] == "ready"
    assert receipt["outcome"] == "retry_resumed"
    assert kb.get_retry_hold(conn, task_id) is None
    assert kb.get_task(conn, task_id).last_failure_error == before_error
    after_runs = [dict(row) for row in conn.execute(
        "SELECT id, outcome, error, metadata FROM task_runs WHERE task_id=? ORDER BY id", (task_id,),
    )]
    assert after_runs == before_runs
    assert _protocol_streak(conn, task_id) == 0
    duplicate = kb.resolve_retry_hold(
        conn, task_id, expected_hold_token=hold["hold_token"], action="resume",
        reason="verified_current_source", metadata={"source": "operator", "schema_sha256": "a" * 64},
    )
    assert duplicate["resolution_event_id"] == receipt["resolution_event_id"]
    assert duplicate["idempotent"] is True

    # The next protocol run starts a new group; the three preserved older failures
    # stay in history but do not count against this released group's streak.
    next_run = conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, error, metadata) "
        "VALUES (?, 'worker', 'crashed', 1800000004, 1800000005, 'crashed', 'protocol violation', ?)",
        (task_id, json.dumps({"protocol_violation": True})),
    )
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, ?, 'protocol_violation', ?, 1800000005)",
        (task_id, int(next_run.lastrowid), json.dumps({"protocol_violation": True, "exit_code": 0})),
    )
    conn.commit()
    assert _protocol_streak(conn, task_id) == 1


def test_retry_hold_and_resolution_survive_sqlite_backup_restore(isolated_board, tmp_path):
    conn = isolated_board
    task_id, failed_run_id, event_id, body_sha256 = _seed_legacy_protocol_trip(conn)
    hold = kb.adopt_protocol_failure(
        conn, task_id, expected_failed_run_id=failed_run_id,
        expected_failure_event_id=event_id, expected_body_sha256=body_sha256,
    )
    backup_path = tmp_path / "retry-hold-backup.sqlite3"
    target = sqlite3.connect(backup_path)
    conn.backup(target)
    target.close()

    kbc.init_db(backup_path)
    with kbc.connect_closing(backup_path) as restored:
        assert kb.get_retry_hold(restored, task_id) == hold
        result = kb.resolve_retry_hold(
            restored, task_id, expected_hold_token=hold["hold_token"],
            action="resume", reason="backup_verified",
        )
        assert result["ok"] is True
        assert restored.execute(
            "SELECT COUNT(*) FROM task_retry_hold_resolutions WHERE task_id=?", (task_id,),
        ).fetchone()[0] == 1
        assert restored.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id=?", (task_id,),
        ).fetchone()[0] == 3


def test_adoption_refuses_twenty_newer_explicit_block_authorities(isolated_board):
    conn = isolated_board
    for index in range(20):
        task_id, failed_run_id, event_id, body_sha256 = _seed_legacy_protocol_trip(
            conn, explicit_block=True,
        )
        result = kb.adopt_protocol_failure(
            conn, task_id, expected_failed_run_id=failed_run_id,
            expected_failure_event_id=event_id, expected_body_sha256=body_sha256,
        )
        assert result["ok"] is False, index
        assert result["reason"] == "newer_lifecycle_authority", index
        assert kb.get_task(conn, task_id).status == "blocked", index
        assert kb.get_retry_hold(conn, task_id) is None, index


def test_adoption_refuses_a_newer_assignment_authority(isolated_board):
    conn = isolated_board
    task_id, failed_run_id, event_id, body_sha256 = _seed_legacy_protocol_trip(conn)
    assert kb.assign_task(conn, task_id, "reviewer") is True

    result = kb.adopt_protocol_failure(
        conn, task_id, expected_failed_run_id=failed_run_id,
        expected_failure_event_id=event_id, expected_body_sha256=body_sha256,
    )

    assert result["ok"] is False
    assert result["reason"] == "newer_lifecycle_authority"
    assert kb.get_retry_hold(conn, task_id) is None
    assert kb.get_task(conn, task_id).status == "blocked"


def test_adoption_requires_latest_protocol_event_for_the_failed_run(isolated_board):
    conn = isolated_board
    task_id, failed_run_id, old_event_id, body_sha256 = _seed_legacy_protocol_trip(
        conn, duplicate_source_event=True,
    )

    result = kb.adopt_protocol_failure(
        conn, task_id, expected_failed_run_id=failed_run_id,
        expected_failure_event_id=old_event_id, expected_body_sha256=body_sha256,
    )

    assert result["ok"] is False
    assert result["reason"] == "source_event_not_latest_protocol_failure"
    assert kb.get_retry_hold(conn, task_id) is None


def test_ordinary_unblock_buys_exactly_one_attempt_on_the_default_path(isolated_board, monkeypatch):
    """Default parity: a manual unblock of a sticky protocol block is not a budget reset."""
    conn = isolated_board
    _dispatch_defaults(monkeypatch)
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"kanban": {}})
    task_id = kb.create_task(conn, title="default unblock", assignee="worker")
    spawned = []

    def spawn(_task, _workspace):
        pid = 98000 + len(spawned) + 1
        spawned.append(pid)
        return pid

    kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=1)
    for _ in range(3):
        kbd._record_worker_exit(spawned[-1], _exited_status(0))
        kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=1)
    assert kb.get_task(conn, task_id).status == "blocked"
    assert len(spawned) == 3

    assert kb.unblock_task(conn, task_id) is True
    kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=1)
    assert len(spawned) == 4
    kbd._record_worker_exit(spawned[-1], _exited_status(0))
    kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=1)
    assert kb.get_task(conn, task_id).status == "blocked"
    assert len(spawned) == 4
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='gave_up'", (task_id,),
    ).fetchone()[0] == 2


def test_retire_refuses_to_release_a_live_dependent(isolated_board):
    conn = isolated_board
    task_id, failed_run_id, event_id, body_sha256 = _seed_legacy_protocol_trip(conn)
    hold = kb.adopt_protocol_failure(
        conn, task_id, expected_failed_run_id=failed_run_id,
        expected_failure_event_id=event_id, expected_body_sha256=body_sha256,
    )
    child_id = kb.create_task(conn, title="dependent", assignee="worker")
    kb.link_tasks(conn, task_id, child_id)

    result = kb.resolve_retry_hold(
        conn, task_id, expected_hold_token=hold["hold_token"], action="retire",
        reason="stale_source",
    )
    assert result["ok"] is False
    assert result["reason"] == "live_dependents_would_be_released"
    assert kb.get_task(conn, task_id).status == "scheduled"
    assert kb.get_retry_hold(conn, task_id) is not None


def test_hold_token_fences_body_assignment_claim_and_new_block(isolated_board):
    conn = isolated_board
    task_id, failed_run_id, event_id, body_sha256 = _seed_legacy_protocol_trip(conn)
    hold = kb.adopt_protocol_failure(
        conn, task_id, expected_failed_run_id=failed_run_id,
        expected_failure_event_id=event_id, expected_body_sha256=body_sha256,
    )
    token = hold["hold_token"]
    conn.execute("UPDATE tasks SET body='edited after hold' WHERE id=?", (task_id,))
    changed = kb.resolve_retry_hold(
        conn, task_id, expected_hold_token=token, action="resume", reason="operator_reviewed",
    )
    assert changed["ok"] is False
    assert changed["reason"] == "stale_hold_token"
    token = changed["current_hold"]["hold_token"]

    assert kb.assign_task(conn, task_id, "reviewer") is True
    reassigned = kb.get_retry_hold(conn, task_id)
    assert reassigned["hold_token"] != token
    assert kb.resolve_retry_hold(
        conn, task_id, expected_hold_token=token, action="resume", reason="operator_reviewed",
    )["reason"] == "stale_hold_token"
    assert kb.claim_task(conn, task_id) is None
    assert kb.unblock_task(conn, task_id) is False

    # Even a direct stale status edit cannot bypass the hold's claim fence.
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
    assert kb.claim_task(conn, task_id) is None
    ready_snapshot = kb.get_retry_hold(conn, task_id)
    assert kb.resolve_retry_hold(
        conn, task_id, expected_hold_token=ready_snapshot["hold_token"],
        action="resume", reason="operator_reviewed",
    )["reason"] == "task_not_quiescent"

    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (task_id,))
        kb._append_event(conn, task_id, "blocked", {"kind": "needs_input", "reason": "manual"})
    blocked_snapshot = kb.get_retry_hold(conn, task_id)
    assert blocked_snapshot["hold_token"] != ready_snapshot["hold_token"]
    assert kb.resolve_retry_hold(
        conn, task_id, expected_hold_token=ready_snapshot["hold_token"],
        action="resume", reason="operator_reviewed",
    )["reason"] == "stale_hold_token"
    assert kb.recompute_ready(conn) == 0


def test_retire_archives_with_non_grade_receipt_and_preserves_failures(isolated_board):
    conn = isolated_board
    task_id, failed_run_id, event_id, body_sha256 = _seed_legacy_protocol_trip(conn)
    original_error = kb.get_task(conn, task_id).last_failure_error
    hold = kb.adopt_protocol_failure(
        conn, task_id, expected_failed_run_id=failed_run_id,
        expected_failure_event_id=event_id, expected_body_sha256=body_sha256,
    )

    result = kb.resolve_retry_hold(
        conn, task_id, expected_hold_token=hold["hold_token"], action="retire",
        reason="source_superseded", metadata={"source": "operator"},
    )

    assert result["ok"] is True
    assert result["outcome"] == "retry_retired_non_grade"
    assert result["status"] == "archived"
    assert kb.get_task(conn, task_id).last_failure_error == original_error
    assert conn.execute(
        "SELECT COUNT(*) FROM task_retry_holds WHERE task_id=?", (task_id,),
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id=? AND outcome='crashed'", (task_id,),
    ).fetchone()[0] == 3
    event = conn.execute(
        "SELECT kind, payload FROM task_events WHERE id=?", (result["resolution_event_id"],),
    ).fetchone()
    assert event["kind"] == "retry_retired"
    payload = json.loads(event["payload"])
    assert payload["outcome"] == "retry_retired_non_grade"
    assert payload["reason"] == "source_superseded"


def test_retire_cleanup_stays_inside_the_task_scratch_scope(isolated_board, tmp_path):
    conn = isolated_board
    root = Path.home() / ".hermes" / "kanban" / "boards" / "default" / "workspaces"
    workspace = root / "retry-hold-cleanup"
    sibling = root / "sibling-sentinel"
    workspace.mkdir(parents=True)
    sibling.mkdir(parents=True)
    (workspace / "worker-output.txt").write_text("temporary", encoding="utf-8")
    (sibling / "keep.txt").write_text("preserve", encoding="utf-8")
    task_id, failed_run_id, event_id, body_sha256 = _seed_legacy_protocol_trip(
        conn, workspace_path=workspace,
    )
    hold = kb.adopt_protocol_failure(
        conn, task_id, expected_failed_run_id=failed_run_id,
        expected_failure_event_id=event_id, expected_body_sha256=body_sha256,
    )

    result = kb.resolve_retry_hold(
        conn, task_id, expected_hold_token=hold["hold_token"], action="retire",
        reason="scope_cleanup_verified",
    )

    assert result["ok"] is True
    assert not workspace.exists()
    assert (sibling / "keep.txt").read_text(encoding="utf-8") == "preserve"


def test_run_provenance_writer_is_run_fenced_and_rejects_unbounded_fields(isolated_board):
    conn = isolated_board
    task_id = kb.create_task(conn, title="provenance", assignee="worker")
    claimed = kb.claim_task(conn, task_id)
    run_id = claimed.current_run_id
    provenance = {
        "version": 1,
        "worker_session_id": "worker-session-1",
        "source_generation_id": "gen-" + "a" * 32,
        "schema_sha256": "b" * 64,
        "actual_routes": [{"model": "fallback-model", "provider": "fallback-provider"}],
        "unknown_fields": [],
    }

    assert kb.record_retry_run_provenance(
        conn, task_id, expected_run_id=run_id, provenance=provenance,
    ) is True
    row = conn.execute("SELECT metadata FROM task_runs WHERE id=?", (run_id,)).fetchone()
    assert json.loads(row["metadata"])["retry_provenance"] == provenance
    assert kb.record_retry_run_provenance(
        conn, task_id, expected_run_id=run_id + 1, provenance=provenance,
    ) is False
    assert kb.record_retry_run_provenance(
        conn, task_id, expected_run_id=run_id,
        provenance={**provenance, "prompt": "not allowed"},
    ) is False
    assert kb.record_retry_run_provenance(
        conn, task_id, expected_run_id=run_id,
        provenance={
            **provenance,
            "actual_routes": [{"provider": "provider", "model": "https://credential.invalid"}],
        },
    ) is False


def _dispatch_defaults(monkeypatch):
    """Host probes for a fake-spawn dispatch that still runs the real claim and reap path."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "unknown")
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda _home: None)


def test_worker_finalizer_path_persists_bounded_provenance_for_current_run(
    isolated_board, monkeypatch, tmp_path,
):
    conn = isolated_board
    task_id = kb.create_task(conn, title="actual route", assignee="worker")
    claimed = kb.claim_task(conn, task_id)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    session_db = SessionDB(tmp_path / "state.db")
    try:
        session_db.create_session("worker-session-actual", source="cli", model="configured-primary")
        session_db.update_token_counts(
            "worker-session-actual", input_tokens=10, output_tokens=1,
            model="fallback-model", billing_provider="fallback-provider", api_call_count=1,
        )
        cli = SimpleNamespace(
            session_id="originating-session",
            agent=SimpleNamespace(
                session_id="worker-session-actual", tools=_TOOLS, _session_db=session_db,
            ),
        )
        single_query._record_kanban_retry_provenance(cli)
    finally:
        session_db.close()

    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE id=?", (claimed.current_run_id,),
    ).fetchone()
    stored = json.loads(row["metadata"])["retry_provenance"]
    assert stored["worker_session_id"] == "worker-session-actual"
    assert stored["actual_routes"] == [{"provider": "fallback-provider", "model": "fallback-model"}]
    assert stored["schema_sha256"] == _schema_sha256(_TOOLS)
    assert "configured-primary" not in json.dumps(stored)


def test_reaper_keeps_the_workers_route_projection_on_the_closed_run(isolated_board, monkeypatch):
    """The reaper closes the crashed run with fresh metadata; the worker's projection must survive."""
    from hermes_cli import kanban_retry_holds as retry_holds

    conn = isolated_board
    _dispatch_defaults(monkeypatch)
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"kanban": {}})
    task_id = kb.create_task(conn, title="reaped provenance", assignee="worker")
    spawned = []

    def spawn(_task, _workspace):
        pid = 96000 + len(spawned) + 1
        spawned.append(pid)
        return pid

    kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=1)
    run_id = int(conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (task_id,)).fetchone()[0])
    provenance = {
        "version": 1,
        "worker_session_id": "worker-session-reaped",
        "source_generation_id": None,
        "schema_sha256": "c" * 64,
        "actual_routes": [{"provider": "fallback-provider", "model": "fallback-model"}],
        "unknown_fields": ["source_generation_id"],
    }
    assert retry_holds.record_retry_run_provenance(
        conn, task_id, expected_run_id=run_id, provenance=provenance,
    ) is True
    kbd._record_worker_exit(spawned[0], _exited_status(0))
    kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=1)

    run = conn.execute("SELECT outcome, metadata FROM task_runs WHERE id=?", (run_id,)).fetchone()
    metadata = json.loads(run["metadata"])
    assert run["outcome"] == "crashed"
    assert metadata["protocol_violation"] is True
    assert metadata["retry_provenance"]["worker_session_id"] == "worker-session-reaped"
    assert metadata["retry_provenance"]["actual_routes"] == [
        {"provider": "fallback-provider", "model": "fallback-model"},
    ]


def test_hold_creation_failure_rolls_back_and_keeps_the_sticky_block(isolated_board, monkeypatch):
    """A failure while parking the hold leaves no half-state and falls back to the existing block."""
    conn = isolated_board
    _dispatch_defaults(monkeypatch)
    monkeypatch.setattr(config, "load_config_readonly", lambda: {
        "kanban": {"protocol_violation_actions": {"default": {"jobseek-worker": "defer"}}},
    })
    real_append_event = kb._append_event
    parking_attempts = []

    def crash_while_parking(conn_, task_id, kind, *args, **kwargs):
        if kind == "protocol_retry_held":
            parking_attempts.append(task_id)
            raise sqlite3.OperationalError("simulated crash while parking the hold")
        return real_append_event(conn_, task_id, kind, *args, **kwargs)

    monkeypatch.setattr(kb, "_append_event", crash_while_parking)
    task_id = kb.create_task(conn, title="crash while parking", assignee="jobseek-worker")
    spawned = []

    def spawn(_task, _workspace):
        pid = 99000 + len(spawned) + 1
        spawned.append(pid)
        return pid

    kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=1)
    for _ in range(3):
        kbd._record_worker_exit(spawned[-1], _exited_status(0))
        kbd.dispatch_once(conn, spawn_fn=spawn, board="default", max_spawn=1)

    assert parking_attempts == [task_id]
    assert kb.get_task(conn, task_id).status == "blocked"
    assert len(spawned) == 3
    assert conn.execute(
        "SELECT COUNT(*) FROM task_retry_holds WHERE task_id=?", (task_id,),
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='protocol_retry_held'", (task_id,),
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='gave_up'", (task_id,),
    ).fetchone()[0] == 1
