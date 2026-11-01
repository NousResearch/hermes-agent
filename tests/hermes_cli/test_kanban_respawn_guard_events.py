"""Durable respawn-guard episodes must not turn dispatch ticks into event spam."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_ops


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _guard_events(conn, task_id):
    return [event for event in kb.list_events(conn, task_id) if event.kind == "respawn_guarded"]


def _no_spawn(*args, **kwargs):
    pytest.fail("a guarded task must not spawn")


@pytest.fixture
def dashboard_api(monkeypatch):
    plugin_file = Path(__file__).resolve().parents[2] / "plugins/kanban/dashboard/plugin_api.py"
    spec = importlib.util.spec_from_file_location("kanban_guard_events_dashboard", plugin_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_guard_ticks_report_every_time_but_persist_only_daily(
    kanban_home, dashboard_api, monkeypatch, capsys,
):
    start = 1_800_000_000
    day = 24 * 60 * 60
    now = start
    monkeypatch.setattr(kbd.time, "time", lambda: now)
    with kbc.connect_closing() as conn:
        task_ids = [
            kb.create_task(conn, title="await workflow approval", assignee="default")
            for _ in range(2)
        ]
        for task_id in task_ids:
            kb.add_comment(conn, task_id, author="worker", body="https://github.com/example/repo/pull/1")
        result = kbd.dispatch_once(conn, dry_run=True)
        assert dict(result.respawn_guarded) == dict.fromkeys(task_ids, "active_pr")
        assert all(not _guard_events(conn, task_id) for task_id in task_ids)

    for elapsed in (0, 60, 120, 3600, day - 1, day, day + 1, 2 * day - 1, 2 * day):
        now = start + elapsed
        # Reopen the on-disk DB each tick, as the daemon does. Renewing PR
        # evidence/comments keeps the hold continuous, not a new episode.
        with kbc.connect_closing() as conn:
            for task_id in task_ids:
                assert kb.assign_task(conn, task_id, "default")
                assert dashboard_api._set_status_direct(conn, task_id, "ready")
                kb.add_comment(
                    conn, task_id, author="worker",
                    body="Still awaiting https://github.com/example/repo/pull/1",
                )
            result = kbd.dispatch_once(conn, spawn_fn=_no_spawn)
            assert dict(result.respawn_guarded) == dict.fromkeys(task_ids, "active_pr")
            assert not result.spawned
            assert kbd.describe_suppression([result]) == f"active_pr={len(task_ids)}"
            for task_id in task_ids:
                events = _guard_events(conn, task_id)
                assert [event.created_at for event in events] == [
                    start + n * day for n in range(elapsed // day + 1)
                ]
                task = kb.get_task(conn, task_id)
                assert task is not None and task.status == "ready"

        # Exercise the CLI renderer with real dispatch, not a canned result.
        for dry_run in (False, True):
            assert kanban_ops._cmd_dispatch(argparse.Namespace(dry_run=dry_run, json=True, max=None)) == 0
            payload = json.loads(capsys.readouterr().out)
            assert {
                entry["task_id"]: entry["reason"] for entry in payload["respawn_guarded"]
            } == dict.fromkeys(task_ids, "active_pr")
        assert kanban_ops._cmd_dispatch(argparse.Namespace(dry_run=True, json=False, max=None)) == 0
        output = capsys.readouterr().out
        assert all(f"Guarded (active_pr): {task_id}" in output for task_id in task_ids)
        with kbc.connect_closing() as conn:
            assert all(len(_guard_events(conn, task_id)) == elapsed // day + 1 for task_id in task_ids)
            assert conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0] == 0


def test_reason_changes_and_lifecycle_reentry_start_new_guard_episodes(
    kanban_home, dashboard_api, monkeypatch,
):
    # All mutations share one second: timestamps alone cannot identify a
    # lifecycle boundary, even when no dispatch tick sees the blocked state.
    monkeypatch.setattr(kbd.time, "time", lambda: 1_800_000_000)
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="guard episodes", assignee="default")
        kb.add_comment(conn, task_id, author="worker", body="https://github.com/example/repo/pull/1")
        expected_reasons = []
        for error, reason in (
            (None, "active_pr"), ("authentication denied", "blocker_auth"), (None, "active_pr"),
        ):
            with kbc.write_txn(conn):
                conn.execute("UPDATE tasks SET last_failure_error = ? WHERE id = ?", (error, task_id))
            expected_reasons.append(reason)
            for _ in range(2):
                result = kbd.dispatch_once(conn, spawn_fn=_no_spawn)
                assert result.respawn_guarded == [(task_id, reason)]
                assert [event.payload for event in _guard_events(conn, task_id)] == [
                    {"reason": value} for value in expected_reasons
                ]

        assert kb.block_task(conn, task_id, kind="needs_input", reason="pause")
        assert kb.unblock_task(conn, task_id)
        # No-ops after a real boundary must not hide that earlier boundary.
        assert kb.assign_task(conn, task_id, "default")
        assert dashboard_api._set_status_direct(conn, task_id, "ready")
        assert kbd.dispatch_once(conn, dry_run=True).respawn_guarded == [(task_id, "active_pr")]
        assert len(_guard_events(conn, task_id)) == len(expected_reasons)

    # The lifecycle boundary and previous reason survive connection replacement.
    with kbc.connect_closing() as conn:
        expected_reasons.append("active_pr")
        for _ in range(2):
            result = kbd.dispatch_once(conn, spawn_fn=_no_spawn)
            assert result.respawn_guarded == [(task_id, "active_pr")]
            assert [event.payload for event in _guard_events(conn, task_id)] == [
                {"reason": value} for value in expected_reasons
            ]

        assert dashboard_api._set_status_direct(conn, task_id, "todo")
        assert dashboard_api._set_status_direct(conn, task_id, "ready")
        assert dashboard_api._set_status_direct(conn, task_id, "ready")
        expected_reasons.append("active_pr")
        for _ in range(2):
            assert kbd.dispatch_once(conn, spawn_fn=_no_spawn).respawn_guarded == [(task_id, "active_pr")]
            assert [event.payload for event in _guard_events(conn, task_id)] == [
                {"reason": value} for value in expected_reasons
            ]
