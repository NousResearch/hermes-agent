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
        assert run1 is not None
        # Sentinel on the real run so we can prove it is never touched.
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps({"sentinel": "kept"}), run1.id),
        )
        # Simulate a stale run id (already-closed / foreign): the contract pins
        # the worker's own run and must NOT fall back onto another run.
        ok = kb.stamp_worker_run_metadata(
            conn, tid,
            extra={"finalize_turn_fired": True},
            expected_run_id=run1.id + 9999,
        )
        # Pinned to a nonexistent run → genuine no-op, not a fallback.
        assert ok is False
        # And the real run's metadata is provably uncorrupted.
        row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ?", (run1.id,)
        ).fetchone()
        meta = json.loads(row["metadata"])
        assert meta == {"sentinel": "kept"}
        assert "finalize_turn_fired" not in meta
    finally:
        conn.close()


def test_stamp_foreign_run_rejected_even_if_id_exists(kanban_home):
    """A pinned run id belonging to a DIFFERENT task must not be merged onto."""
    conn = kb.connect()
    try:
        tid_a = _new_task(conn)
        tid_b = _new_task(conn)
        kb.claim_task(conn, tid_a)
        kb.claim_task(conn, tid_b)
        run_b = kb.latest_run(conn, tid_b)
        if run_b is None:
            pytest.skip("no foreign run row present")
        # Commit a sentinel to run_b so we can prove it is never merged onto.
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps({"foreign_sentinel": True}), run_b.id),
        )
        kb.stamp_worker_run_metadata(
            conn, tid_a,
            extra={"finalize_turn_fired": True},
            expected_run_id=run_b.id,
        )
        meta = json.loads(
            conn.execute(
                "SELECT metadata FROM task_runs WHERE id = ?", (run_b.id,)
            ).fetchone()["metadata"] or "{}"
        )
        # The pinned foreign run must be untouched by task A's stamp.
        assert meta == {"foreign_sentinel": True}
        assert "finalize_turn_fired" not in meta
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

    # kanban_complete is the conclusive close → marks finalize_turn_succeeded.
    out = _stamp_worker_session_metadata(
        "t_abc", {"handoff": "done"}, finalize_conclusive=True
    )
    out = normalize(out)
    assert out["worker_session_id"] == "sess-1"
    assert out["finalize_turn_fired"] is True
    assert out["finalize_turn_succeeded"] is True  # conclusive complete


def test_stamp_review_handoff_does_not_mark_succeeded(kanban_home, monkeypatch):
    """A review handoff after a finalize fired is a valid terminal close but not
    a conclusive complete — it must stamp the fired flag, never succeeded."""
    from tools.kanban_tools import _stamp_worker_session_metadata

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc")
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-1")

    from agent import kanban_checkpoint as kcp

    kcp.reset_finalize_state()
    kcp.mark_finalize_fired()

    # kanban_request_review path (default conclusive=False) → NOT succeeded.
    out = _stamp_worker_session_metadata(
        "t_abc", {"handoff": "review"}
    )
    out = normalize(out)
    assert out["worker_session_id"] == "sess-1"
    assert out["finalize_turn_fired"] is True
    assert out["finalize_turn_succeeded"] is False


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