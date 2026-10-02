"""Wake-Guard: budget-exhausted records must not be written against dead cards
(t_bdd69e28, port of PR #91's tests/agent/test_turn_finalizer_expired_scoping.py).

Upstream routes the record through ``kanban_db_dispatch._record_task_failure``;
the spy patches that module attribute, which the late import inside
``_record_kanban_budget_exhausted`` reads at call time. Unknown freshness keeps
the legacy record (fail-open), a live worker keeps it too.
"""

import logging

import pytest

from agent import turn_finalizer as tf
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def _fresh_board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "budget-expiry.db"))
    kb.init_db()
    return kbc.connect()


def _spy_record(monkeypatch):
    calls = []

    def spy(conn, task_id, **kwargs):
        calls.append({"task_id": task_id, **kwargs})

    monkeypatch.setattr(kbd, "_record_task_failure", spy)
    return calls


def test_budget_record_skipped_when_scoping_expired(tmp_path, monkeypatch):
    """A terminal card must not receive a ``timed_out`` failure record."""
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="dead card", assignee="worker")
        kb.complete_task(conn, tid, summary="finished")
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    calls = _spy_record(monkeypatch)

    tf._record_kanban_budget_exhausted(tid, 30, 30, logging.getLogger("test"))

    assert calls == []


def test_budget_record_kept_when_freshness_unknown(tmp_path, monkeypatch):
    """A missing task row is UNKNOWN freshness — the legacy record stands and
    the failure is logged, never guessed."""
    _fresh_board(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_does_not_exist")
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    calls = _spy_record(monkeypatch)

    tf._record_kanban_budget_exhausted("t_does_not_exist", 30, 30, logging.getLogger("test"))

    assert len(calls) == 1
    assert calls[0]["task_id"] == "t_does_not_exist"
    assert calls[0]["outcome"] == "timed_out"


def test_budget_record_kept_for_live_worker(tmp_path, monkeypatch):
    """A live worker (claim open, run not ended) keeps the legacy record."""
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="live card", assignee="worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    calls = _spy_record(monkeypatch)

    tf._record_kanban_budget_exhausted(tid, 30, 30, logging.getLogger("test"))

    assert len(calls) == 1
    assert calls[0]["task_id"] == tid
