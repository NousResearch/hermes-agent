"""Estimated-runtime budget for kanban workers.

``enforce_max_runtime`` historically required the card to carry an explicit
``max_runtime_seconds`` (``--max-runtime``), which no card does by default, so
nothing bounded a running worker before the 4 h stale window. These tests pin
the precedence — explicit > worker estimate (with grace) > dispatcher default —
and the ``timed_out`` payload that names which budget fired.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def no_pid(monkeypatch):
    """Make SIGTERM look instantly effective so the grace poll exits fast."""
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)


def _running(conn, *, started_seconds_ago=0, **task_kwargs):
    tid = kb.create_task(conn, title="job", assignee="worker", **task_kwargs)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    started = int(time.time()) - started_seconds_ago
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (started, tid))
        conn.execute(
            "UPDATE task_runs SET started_at = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (started, tid),
        )
    return tid


def _timed_out_payload(conn, tid):
    ev = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'timed_out' "
        "ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    return json.loads(ev["payload"]) if ev and ev["payload"] else None


def test_heartbeat_records_expected_runtime(kanban_home):
    conn = kbc.connect()
    try:
        tid = _running(conn)
        run_id = kb._current_run_id(conn, tid)
        assert kbd.heartbeat_worker(
            conn, tid, expected_run_id=run_id, estimated_runtime_seconds=1800)
        task = conn.execute(
            "SELECT estimated_runtime_seconds FROM tasks WHERE id = ?", (tid,)).fetchone()
        run = conn.execute(
            "SELECT estimated_runtime_seconds FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        assert task["estimated_runtime_seconds"] == 1800
        assert run["estimated_runtime_seconds"] == 1800
    finally:
        conn.close()


def test_estimate_backfills_on_claim(kanban_home, no_pid):
    """An estimate recorded on the task bounds the attempt even with no card cap."""
    conn = kbc.connect()
    try:
        tid = _running(conn, started_seconds_ago=3600)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET estimated_runtime_seconds = 100 WHERE id = ?", (tid,))
        killed = []
        out = kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append(pid))
        assert tid in out
        payload = _timed_out_payload(conn, tid)
        assert payload["limit_source"] == "estimate"
        assert payload["limit_seconds"] == int(100 * kbd.ESTIMATE_GRACE_FACTOR)
        assert payload["estimated_runtime_seconds"] == 100
    finally:
        conn.close()


def test_dispatcher_default_bounds_an_unestimated_card(kanban_home, no_pid, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DEFAULT_MAX_RUNTIME_SECONDS", "300")
    conn = kbc.connect()
    try:
        tid = _running(conn, started_seconds_ago=400)
        out = kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: None)
        assert tid in out, "an unestimated card should still be bounded by the default"
        payload = _timed_out_payload(conn, tid)
        assert payload["limit_source"] == "default"
        assert payload["limit_seconds"] == 300
    finally:
        conn.close()


def test_unbounded_card_is_left_alone(kanban_home, no_pid, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_DEFAULT_MAX_RUNTIME_SECONDS", raising=False)
    conn = kbc.connect()
    try:
        tid = _running(conn, started_seconds_ago=3 * 24 * 3600)
        assert kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: None) == []
    finally:
        conn.close()


def test_explicit_cap_outranks_estimate(kanban_home, no_pid):
    conn = kbc.connect()
    try:
        tid = _running(conn, started_seconds_ago=600, max_runtime_seconds=500)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET estimated_runtime_seconds = 100000 WHERE id = ?", (tid,))
        out = kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: None)
        assert tid in out
        payload = _timed_out_payload(conn, tid)
        assert payload["limit_source"] == "max_runtime"
        assert payload["limit_seconds"] == 500
    finally:
        conn.close()


def test_effective_limit_precedence():
    assert kbd._effective_runtime_limit(500, 100000, 300) == (500, "max_runtime")
    assert kbd._effective_runtime_limit(None, 200, 300) == (300, "estimate")
    assert kbd._effective_runtime_limit(None, None, 300) == (300, "default")
    assert kbd._effective_runtime_limit(None, None, 0) == (None, None)
    assert kbd._effective_runtime_limit(None, 0, 0) == (None, None), "0 estimate is not a budget"
