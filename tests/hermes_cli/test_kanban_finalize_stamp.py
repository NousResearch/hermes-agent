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


# ── 2026-09-07: the finalize-turn stamp tests were removed ───────────
# Four tests here exercised _stamp_worker_session_metadata stamping
# finalize_turn_fired / finalize_turn_succeeded, which came from
# agent.kanban_checkpoint. That module is retired.
#
# The six tests above are KEPT and still matter: they cover
# stamp_worker_run_metadata itself (merge, preserve, empty, missing run,
# stale-run pinning, foreign-run rejection), used by other callers and
# unrelated to the finalize turn.
#
# The replacement pins live in tests/agent/test_kanban_stop.py.
