"""Temporary-home behavior contracts for queue aging and report-only watchdog."""
import json
import multiprocessing
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli.kanban_dispatch_scheduling import (
    SchedulingSettings, queue_rows, report_health, resolve_settings,
)


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    conn = kbc.connect()
    yield conn, home
    conn.close()


def _task(conn, *, priority=0, age=100, status="ready", assignee="builder", now=100000):
    tid = kb.create_task(conn, title="synthetic task", assignee=assignee, priority=priority)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = ?, created_at = ? WHERE id = ?",
                     (status, now - age, tid))
    return tid


@pytest.mark.parametrize("status", ["ready", "review"])
def test_aging_old_waiter_beats_sustained_fresh_priority(board, status):
    conn, _ = board
    old = _task(conn, age=20000, status=status)
    fresh = _task(conn, priority=10, age=0, status=status)
    running = _task(conn, priority=100, status="running")
    settings = SchedulingSettings(enabled=True)
    assert [r["id"] for r in queue_rows(conn, status, settings, now=100000)] == [old, fresh]
    assert [r["id"] for r in queue_rows(conn, status, SchedulingSettings(), now=100000)] == [fresh, old]
    assert kb.get_task(conn, old).priority == 0
    assert kb.get_task(conn, running).status == "running"


def test_aging_is_bounded_and_does_not_mutate_priority(board):
    conn, _ = board
    old = _task(conn, age=1000000)
    urgent = _task(conn, priority=21, age=0)
    assert [r["id"] for r in queue_rows(conn, "ready", SchedulingSettings(enabled=True), now=100000)] == [urgent, old]
    assert kb.get_task(conn, urgent).priority == 21


def test_watchdog_never_closes_run_or_claim_and_dedups_progress(board):
    conn, _ = board
    tid = _task(conn)
    task = kb.claim_task(conn, tid)
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET started_at = 90000 WHERE id = ?", (task.current_run_id,))
        conn.execute("UPDATE tasks SET last_heartbeat_at = 95000 WHERE id = ?", (tid,))
    before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())
    settings = SchedulingSettings(enabled=True)
    assert report_health(conn, settings, now=100000)[0] == [tid]
    assert report_health(conn, settings, now=100001) == ([], [])
    assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()) == before
    assert conn.execute("SELECT ended_at FROM task_runs WHERE id = ?", (task.current_run_id,)).fetchone()[0] is None
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET last_heartbeat_at = 100001 WHERE id = ?", (tid,))
    assert report_health(conn, settings, now=100002)[0] == []
    assert report_health(conn, settings, now=104000)[0] == [tid]


def test_daily_report_oldest_open_includes_blocked_not_done_survives_restart(board):
    conn, _ = board
    _task(conn, status="done", age=500)
    blocked = _task(conn, status="blocked", age=400)
    ready = _task(conn, age=300)
    settings = SchedulingSettings(enabled=True, oldest_limit=2)
    stalled, oldest = report_health(conn, settings, now=100000)
    assert stalled == []
    assert [r["task_id"] for r in oldest] == [blocked, ready]
    path = conn.execute("PRAGMA database_list").fetchone()[2]
    with sqlite3.connect(path) as second:
        second.row_factory = sqlite3.Row
        assert report_health(second, settings, now=100001) == ([], [])
        assert len(report_health(second, settings, now=186400)[1]) == 2
    payloads = [json.loads(r[0]) for r in conn.execute(
        "SELECT payload FROM task_events WHERE kind = 'dispatch_oldest_open'")]
    assert len(payloads) == 2
    assert "title" not in payloads[0]["oldest"][0]


@pytest.mark.parametrize("raw", [{}, {"enabled": False}])
def test_default_off_and_dry_run_do_not_add_reports(board, raw):
    conn, _ = board
    _task(conn)
    assert report_health(conn, resolve_settings(raw), now=100000) == ([], [])
    assert report_health(conn, SchedulingSettings(enabled=True), dry_run=True, now=100000) == ([], [])
    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'dispatch_health_reports'").fetchone() is None


@pytest.mark.parametrize("raw", [{"enabled": "false"}, {"enabled": True, "aging_seconds": 0},
                                 {"enabled": True, "oldest_limit": True},
                                 {"enabled": True, "maximum_bonus": 101},
                                 {"enabled": True, "stall_seconds": 1}])
def test_invalid_enabled_settings_rejected(raw):
    with pytest.raises(ValueError):
        resolve_settings(raw)


def test_real_config_reader_drives_standalone_dispatch(board, monkeypatch, all_assignees_spawnable):
    conn, home = board
    old = _task(conn, age=20000)
    fresh = _task(conn, priority=10, age=0)
    monkeypatch.setattr(dispatch.time, "time", lambda: 100000)
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "ok")
    from hermes_cli.config import atomic_config_write
    atomic_config_write(home / "config.yaml", {"kanban": {
        "dispatch_scheduling": {"enabled": True, "maximum_bonus": 20, "aging_seconds": 900}}})
    settings = resolve_settings()
    assert settings.enabled
    result = dispatch.dispatch_once(conn, dry_run=True, max_spawn=1, reconcile_orphans=False)
    assert [r[0] for r in result.spawned] == [old]
    assert kb.get_task(conn, fresh).status == "ready"
    assert result.oldest_open_report == []


def _report_process(path, barrier, output):
    with sqlite3.connect(path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        barrier.wait(timeout=30)
        output.put(len(report_health(conn, SchedulingSettings(enabled=True), now=100000)[1]))


def test_daily_report_two_process_race(board):
    conn, _ = board
    _task(conn)
    path = conn.execute("PRAGMA database_list").fetchone()[2]
    ctx = multiprocessing.get_context("spawn")
    barrier, output = ctx.Barrier(2), ctx.Queue()
    processes = [ctx.Process(target=_report_process, args=(path, barrier, output)) for _ in range(2)]
    try:
        for process in processes:
            process.start()
        results = [output.get(timeout=40) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert sorted(results) == [0, 1]
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE kind = 'dispatch_oldest_open'").fetchone()[0] == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)
        output.close()


def test_real_dispatch_watchdog_reports_with_full_cap_without_preemption(board, monkeypatch):
    conn, _ = board
    tid = _task(conn)
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET started_at = 90000 WHERE id = ?", (claimed.current_run_id,))
        conn.execute("UPDATE tasks SET last_heartbeat_at = 95000 WHERE id = ?", (tid,))
    before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "ok")
    from hermes_cli import kanban_dispatch_scheduling as scheduling
    monkeypatch.setattr(scheduling.time, "time", lambda: 100000)
    result = dispatch.dispatch_once(
        conn, max_in_progress=1, reconcile_orphans=False,
        stale_timeout_seconds=0, dispatch_scheduling={"enabled": True},
    )
    assert result.stalled_reported == [tid]
    assert [row["task_id"] for row in result.oldest_open_report] == [tid]
    assert result.spawned == []
    assert dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()) == before
    again = dispatch.dispatch_once(
        conn, max_in_progress=1, reconcile_orphans=False,
        stale_timeout_seconds=0, dispatch_scheduling={"enabled": True},
    )
    assert again.stalled_reported == []
    assert again.oldest_open_report == []


def test_embedded_settings_pass_explicit_profile_config(board, monkeypatch):
    from gateway.kanban_watchers_dispatcher import _resolve_dispatcher_settings
    conn, _ = board
    monkeypatch.setattr(dispatch, "resolve_max_in_progress", lambda n: n)
    settings = _resolve_dispatcher_settings({"dispatch_scheduling": {"enabled": True, "aging_seconds": 60}}, kb)
    assert resolve_settings(asdict(settings)["dispatch_scheduling"]).aging_seconds == 60
    assert not resolve_settings(asdict(_resolve_dispatcher_settings({}, kb))["dispatch_scheduling"]).enabled
