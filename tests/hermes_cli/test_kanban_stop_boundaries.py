"""Stop-boundary invariants from the 2026-09-13 spend incident.

Two contracts, both violated that night:

1. A repeat human-input block (``needs_input``/``capability``) routes the task
   to triage FOR A HUMAN — the auto-decomposer must never list it
   (``list_triage_ids``), or a timed-out approval becomes six new children.
2. A billing exit (402 / credits exhausted) must count as a task failure and
   park the task — never be classified as a neutral rate-limit requeue, or the
   dispatcher relaunches into the same refusal forever.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import os
import time as _time

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _force_triage_block(conn, task_id: str, kind: str) -> None:
    """Simulate the terminal state of _route_block at the recurrence limit:
    status='triage' with the human-input block kind persisted."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'triage', block_kind = ?, block_recurrences = 2 "
            "WHERE id = ?",
            (kind, task_id),
        )


class TestHumanInputBlockersNotDecomposed:
    def test_needs_input_triage_task_is_excluded(self, kanban_home):
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="needs owner approval", initial_status="running")
            _force_triage_block(conn, tid, "needs_input")
            assert decomp.list_triage_ids() == []

    def test_capability_triage_task_is_excluded(self, kanban_home):
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="no tool for this", initial_status="running")
            _force_triage_block(conn, tid, "capability")
            assert decomp.list_triage_ids() == []

    def test_plain_triage_task_is_still_listed(self, kanban_home):
        """Untyped triage (a human put it there, or a legacy row) still
        decomposes — the exclusion is keyed on the human-input kinds only."""
        with kbc.connect() as conn:
            kb.create_task(conn, title="genuine triage", triage=True)
            assert len(decomp.list_triage_ids()) == 1

    def test_decompose_task_refuses_human_input_triage(self, kanban_home):
        """Direct call with a task id bypasses list_triage_ids; the guard must
        hold there too (defense at the entry point, not just the listing)."""
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="needs owner approval", initial_status="running")
            _force_triage_block(conn, tid, "needs_input")
        outcome = decomp.decompose_task(tid, author="test")
        assert outcome.ok is False
        assert "human" in outcome.reason.lower() or "needs_input" in outcome.reason


class TestBillingExitIsNotRateLimit:
    def test_classify_worker_exit_billing_code(self):
        dispatch._recent_worker_exits[4242] = (76 << 8, _time.time())
        try:
            kind, code = dispatch._classify_worker_exit(4242)
            assert (kind, code) == ("billing", 76)
        finally:
            dispatch._recent_worker_exits.pop(4242, None)

    def test_classify_worker_exit_rate_limit_unchanged(self):
        dispatch._recent_worker_exits[4243] = (75 << 8, _time.time())
        try:
            kind, code = dispatch._classify_worker_exit(4243)
            assert (kind, code) == ("rate_limited", 75)
        finally:
            dispatch._recent_worker_exits.pop(4243, None)

    def test_billing_dead_worker_counts_as_failure(self):
        dispatch._recent_worker_exits[4244] = (76 << 8, _time.time())
        try:
            dead = dispatch._classify_dead_worker(4244, claimer="builder")
        finally:
            dispatch._recent_worker_exits.pop(4244, None)
        assert dead.rate_limited is False
        assert dead.protocol_violation is False
        assert dead.run_outcome == "crashed"
        assert "billing" in dead.error_text

    def test_billing_code_is_distinct_from_rate_limit(self):
        assert kb.KANBAN_BILLING_EXIT_CODE != kb.KANBAN_RATE_LIMIT_EXIT_CODE
