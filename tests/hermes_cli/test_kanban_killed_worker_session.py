"""RED for #124380: killed kanban worker leaves state.db session open.

Reclaim / max-runtime kill the worker without its in-worker flush, so the
dispatcher must end the worker's live session (compression tip) itself.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from hermes_state import SessionDB


def _make_board(tmp_path, monkeypatch, profile_home):
    from hermes_cli import kanban_db as kb
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    monkeypatch.setattr(
        "hermes_cli.profiles.resolve_profile_env", lambda _p: str(profile_home),
    )
    return kb


def _open_state(profile_home):
    profile_home.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=profile_home / "state.db")
    return db


def _run_task(kb, conn, assignee="wprof"):
    t = kb.create_task(conn, title="long", assignee=assignee)
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, t, claimer=f"{host}:w")
    return t, host


def _record_heartbeat_with_session(kb, kbd, conn, task_id, run_id, session_id):
    kbd.heartbeat_worker(conn, task_id, expected_run_id=run_id)
    conn.execute(
        "UPDATE task_events SET payload = ? WHERE task_id = ? AND kind = 'heartbeat' AND run_id = ?",
        (json.dumps({"worker_session_id": session_id}), task_id, run_id),
    )


def test_reclaim_ends_worker_tip(tmp_path, monkeypatch):
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    prof = tmp_path / "prof"
    kb = _make_board(tmp_path, monkeypatch, prof)
    sdb = _open_state(prof)
    try:
        sdb.create_session(session_id="s-parent", source="kanban")
        sdb.end_session("s-parent", "compression")
        sdb.create_session(session_id="s-tip", source="kanban", parent_session_id="s-parent")
        with kbc.connect() as conn:
            t, _host = _run_task(kb, conn)
            run_id = kb._current_run_id(conn, t)
            _record_heartbeat_with_session(kb, kbd, conn, t, run_id, "s-parent")
            assert kb.reclaim_task(conn, t, reason="op", signal_fn=lambda _p, _s: None) is True
        rows = {r["id"]: r for r in sdb._read_all(
            "SELECT id, ended_at, end_reason FROM sessions WHERE id IN ('s-parent','s-tip')", [])}
        assert rows["s-tip"]["ended_at"] is not None
        assert rows["s-tip"]["end_reason"] == "kanban_reclaimed"
        assert rows["s-parent"]["end_reason"] == "compression"
    finally:
        sdb.close()


def test_max_runtime_ends_worker_session(tmp_path, monkeypatch):
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    prof = tmp_path / "prof2"
    kb = _make_board(tmp_path, monkeypatch, prof)
    sdb = _open_state(prof)
    try:
        sdb.create_session(session_id="s-live", source="kanban")
        with kbc.connect() as conn:
            t, _host = _run_task(kb, conn)
            run_id = kb._current_run_id(conn, t)
            _record_heartbeat_with_session(kb, kbd, conn, t, run_id, "s-live")
            conn.execute(
                "UPDATE tasks SET max_runtime_seconds = ?, started_at = ? WHERE id = ?",
                (1, int(time.time()) - 100, t),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE id = ?",
                (int(time.time()) - 100, run_id),
            )
            # Dead pid keeps the row eligible without a 5s grace sleep.
            conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (1999999, t))
            out = kbd.enforce_max_runtime(conn, signal_fn=lambda _p, _s: None)
            assert out == [t]
        row = sdb._read_all("SELECT ended_at, end_reason FROM sessions WHERE id='s-live'", [])[0]
        assert row["ended_at"] is not None
        assert row["end_reason"] == "kanban_timed_out"
    finally:
        sdb.close()


def test_heartbeat_carries_worker_session_id(tmp_path, monkeypatch):
    """Worker-side half: heartbeat records HERMES_SESSION_ID for the dispatcher."""
    import os
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    prof = tmp_path / "prof3"
    kb = _make_board(tmp_path, monkeypatch, prof)
    monkeypatch.setenv("HERMES_SESSION_ID", "s-hb")
    with kbc.connect() as conn:
        t, _host = _run_task(kb, conn)
        run_id = kb._current_run_id(conn, t)
        assert kbd.heartbeat_worker(
            conn, t, expected_run_id=run_id, worker_session_id=os.environ.get("HERMES_SESSION_ID"),
        ) is True
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'heartbeat' ORDER BY id DESC LIMIT 1",
            (t,),
        ).fetchone()
        assert json.loads(row["payload"])["worker_session_id"] == "s-hb"


def test_stale_running_ends_worker_session(tmp_path, monkeypatch):
    """detect_stale_running kills the worker, so it must end the session too."""
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    prof = tmp_path / "prof4"
    kb = _make_board(tmp_path, monkeypatch, prof)
    sdb = _open_state(prof)
    try:
        sdb.create_session(session_id="s-stale", source="kanban")
        with kbc.connect() as conn:
            t, _host = _run_task(kb, conn)
            run_id = kb._current_run_id(conn, t)
            _record_heartbeat_with_session(kb, kbd, conn, t, run_id, "s-stale")
            # Dead pid + no heartbeat ever: eligible without a 5s grace sleep.
            conn.execute(
                "UPDATE tasks SET started_at = ?, last_heartbeat_at = NULL, worker_pid = ? WHERE id = ?",
                (int(time.time()) - 100, 1999999, t),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE id = ?",
                (int(time.time()) - 100, run_id),
            )
            out = kbd.detect_stale_running(conn, stale_timeout_seconds=1, signal_fn=lambda _p, _s: None)
            assert out == [t]
        row = sdb._read_all("SELECT ended_at, end_reason FROM sessions WHERE id='s-stale'", [])[0]
        assert row["ended_at"] is not None
        assert row["end_reason"] == "kanban_stale_reclaim"
    finally:
        sdb.close()


def test_crashed_worker_ends_session(tmp_path, monkeypatch):
    """The crashed-worker sweep reclaims a dead pid and must end its session."""
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    prof = tmp_path / "prof5"
    kb = _make_board(tmp_path, monkeypatch, prof)
    sdb = _open_state(prof)
    try:
        sdb.create_session(session_id="s-crash", source="kanban")
        with kbc.connect() as conn:
            t, _host = _run_task(kb, conn)
            run_id = kb._current_run_id(conn, t)
            _record_heartbeat_with_session(kb, kbd, conn, t, run_id, "s-crash")
            conn.execute(
                "UPDATE tasks SET worker_pid = ?, started_at = ? WHERE id = ?",
                (1999999, int(time.time()) - 100, t),
            )
            monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 1)
            out = kbd.detect_crashed_workers(conn)
            assert out == [t]
        row = sdb._read_all("SELECT ended_at, end_reason FROM sessions WHERE id='s-crash'", [])[0]
        assert row["ended_at"] is not None
        assert row["end_reason"] == "kanban_crashed"
    finally:
        sdb.close()
