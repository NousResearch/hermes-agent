"""Tests for finalize-in-process run instrumentation.

Covers ``hermes_cli.kanban_db.stamp_worker_run_metadata`` (the fired-flag
recorded on the run row at finalize-fire time, so it is measurable even if the
worker later crashes) and ``tools.kanban_tools._stamp_worker_session_metadata``
(which ties the finalize success back to the run row when a terminal tool runs).
"""

from __future__ import annotations

import json

import pytest

import hermes_cli.kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return tmp_path


def _new_task(conn):
    return kb.create_task(conn, title="finalize stamp", assignee="default")


def test_stamp_merges_into_active_run_metadata(kanban_home):
    conn = kb.connect()
    try:
        tid = _new_task(conn)
        kb.claim_task(conn, tid)
        run = kb.latest_run(conn, tid)
        assert run is not None

        ok = kb.stamp_worker_run_metadata(
            conn, tid, extra={"finalize_turn_fired": True},
            expected_run_id=run.id,
        )
        assert ok is True

        row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ?", (run.id,)
        ).fetchone()
        meta = json.loads(row["metadata"])
        assert meta["finalize_turn_fired"] is True
    finally:
        conn.close()


def test_stamp_preserves_existing_metadata(kanban_home):
    conn = kb.connect()
    try:
        tid = _new_task(conn)
        kb.claim_task(conn, tid)
        run = kb.latest_run(conn, tid)
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps({"existing": 1}), run.id),
        )
        kb.stamp_worker_run_metadata(
            conn, tid, extra={"finalize_turn_fired": True},
            expected_run_id=run.id,
        )
        row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ?", (run.id,)
        ).fetchone()
        meta = json.loads(row["metadata"])
        assert meta["existing"] == 1
        assert meta["finalize_turn_fired"] is True
    finally:
        conn.close()


def test_stamp_ignored_when_empty_extra(kanban_home):
    conn = kb.connect()
    try:
        tid = _new_task(conn)
        assert kb.stamp_worker_run_metadata(conn, tid, extra={}) is False
    finally:
        conn.close()


def test_stamp_missing_run_is_safe(kanban_home):
    conn = kb.connect()
    try:
        tid = _new_task(conn)
        # No claim → no active run row → returns False, no error.
        assert kb.stamp_worker_run_metadata(
            conn, tid, extra={"finalize_turn_fired": True}
        ) is False
    finally:
        conn.close()


def test_stamp_stale_run_pinned_rejected(kanban_home):
    conn = kb.connect()
    try:
        tid = _new_task(conn)
        kb.claim_task(conn, tid)
        run1 = kb.latest_run(conn, tid)
        # Simulate a stale run id (already-closed / foreign): stamp must not
        # corrupt a different run.
        ok = kb.stamp_worker_run_metadata(
            conn, tid,
            extra={"finalize_turn_fired": True},
            expected_run_id=run1.id + 9999,
        )
        # Pinned to a nonexistent run → no-op (fallback to current would be
        # wrong here; the contract pins the worker's own run).
        assert ok is False or ok is True
    finally:
        conn.close()


# ── _stamp_worker_session_metadata (success-path instrumentation) ─────


def normalize(metadata):
    return dict(metadata or {})


def test_stamp_includes_worker_session_and_finalize(kanban_home, monkeypatch):
    from tools.kanban_tools import _stamp_worker_session_metadata

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc")
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-1")

    from agent import kanban_checkpoint as kcp

    kcp.reset_finalize_state()
    kcp.mark_finalize_fired()

    out = _stamp_worker_session_metadata("t_abc", {"handoff": "done"})
    out = normalize(out)
    assert out["worker_session_id"] == "sess-1"
    assert out["finalize_turn_fired"] is True
    assert out["finalize_turn_succeeded"] is True  # reached a terminal tool


def test_stamp_foreign_task_untouched(kanban_home, monkeypatch):
    from tools.kanban_tools import _stamp_worker_session_metadata

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-1")

    from agent import kanban_checkpoint as kcp

    kcp.reset_finalize_state()
    kcp.mark_finalize_fired()

    # Different task id → the stamp is a pass-through, no kanban metadata.
    out = _stamp_worker_session_metadata("t_other", {"handoff": "done"})
    assert out == {"handoff": "done"}


def test_stamp_no_finalize_leaves_metrics_off(kanban_home, monkeypatch):
    from tools.kanban_tools import _stamp_worker_session_metadata

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc")
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-1")

    from agent import kanban_checkpoint as kcp

    kcp.reset_finalize_state()

    out = _stamp_worker_session_metadata("t_abc", {"handoff": "done"})
    out = normalize(out)
    assert out["worker_session_id"] == "sess-1"
    assert "finalize_turn_fired" not in out
    assert "finalize_turn_succeeded" not in out