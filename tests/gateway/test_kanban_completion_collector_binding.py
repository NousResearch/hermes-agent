"""Collector-side binding: which run summary may be rendered for a completed event.

The formatter trusts ``run_summaries[event.id]``, so the guards that decide WHICH
run is allowed to fill that map are the only thing standing between an unrelated or
failed run's handoff and a Human-facing message. These tests drive the real
collector (``_Collector._claim_for_sub``) against a real board, because a unit test
of the dict lookup cannot see a collector that binds the wrong row.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from gateway import kanban_watchers_notifier as nt
from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


class _Adapter:
    async def send(self, chat_id, text, metadata=None):
        pass


def _runner():
    """A real GatewayRunner shell, the same shape tests/gateway uses."""
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: _Adapter()}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


@pytest.fixture
def board(tmp_path, monkeypatch):
    """A real board with one completed task and a telegram subscription."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "collector.db"))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="collector binding", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        # A real claim so complete_task closes a real run row.
        kb.claim_task(conn, tid)
        yield conn, tid
    finally:
        conn.close()


def _collector(kb_module=None):
    return nt._Collector(
        _runner(), kb_module or kb, notifier_profile=None, gc_due=False, gc_retention_days=7,
    )


def _claim(collector, conn, tid):
    sub = kbn.list_notify_subs(conn, tid)[0]
    return collector._claim_for_sub(conn, "default", sub)


def _completed_event_id(delivery):
    return next(ev.id for ev in delivery["events"] if ev.kind == "completed")


def test_completed_event_binds_its_own_run_summary(board):
    """The happy path: the completed event carries its run's handoff."""
    conn, tid = board
    kb.complete_task(conn, tid, summary="SUMMARY=done\nNEXT_STEP=none\n")

    delivery = _claim(_collector(), conn, tid)

    assert delivery["run_summaries"][_completed_event_id(delivery)] == "SUMMARY=done\nNEXT_STEP=none\n"


def test_run_belonging_to_another_task_is_not_bound(board):
    """A run row for a different task must never be rendered as this task's result."""
    conn, tid = board
    kb.complete_task(conn, tid, summary="SUMMARY=done\nNEXT_STEP=none\n")

    other = kb.create_task(conn, title="someone else", assignee="worker")
    real_get_run = kb.get_run
    foreign = replace(real_get_run(conn, 1), task_id=other)

    class _KB:
        get_task = staticmethod(kb.get_task)
        get_run = staticmethod(lambda _conn, _run_id: foreign)

    delivery = _claim(_collector(_KB), conn, tid)

    assert delivery["run_summaries"] == {}


def _claim_with(kb_module, conn, tid):
    """Claim once; ``None`` means the cursor was already past this batch."""
    return _claim(_collector(kb_module), conn, tid)


def _binding_for(kb_module, conn, tid):
    """The summaries bound for this subscription, rewinding the cursor first."""
    conn.execute("UPDATE kanban_notify_subs SET last_event_id = 0 WHERE task_id = ?", (tid,))
    conn.commit()
    delivery = _claim_with(kb_module, conn, tid)
    return {} if delivery is None else delivery["run_summaries"]


def test_unfinished_run_is_not_bound(board):
    """outcome/ended_at are the proof the attempt actually finished."""
    conn, tid = board
    kb.complete_task(conn, tid, summary="SUMMARY=done\nNEXT_STEP=none\n")
    real = kb.get_run(conn, 1)

    for broken in (replace(real, ended_at=None), replace(real, outcome="crashed"), replace(real, outcome=None)):
        class _KB:
            get_task = staticmethod(kb.get_task)
            get_run = staticmethod(lambda _conn, _run_id, row=broken: row)

        assert _binding_for(_KB, conn, tid) == {}


def test_missing_run_is_not_bound(board):
    """No run row (or a lookup failure) degrades to the note rather than inventing one."""
    conn, tid = board
    kb.complete_task(conn, tid, summary="SUMMARY=done\nNEXT_STEP=none\n")

    for broken in (lambda _c, _r: None, _raise):
        class _KB:
            get_task = staticmethod(kb.get_task)
            get_run = staticmethod(broken)

        assert _binding_for(_KB, conn, tid) == {}


def test_event_without_run_id_is_not_bound(board):
    """A legacy completed event (no run_id stamp) binds no summary and does not raise.

    ``get_run`` coerces with ``int(run_id)``, so passing ``None`` raises rather than
    missing. The collector must skip the row up front; this pins both the empty bind
    and the fact that the tick still delivers.
    """
    conn, tid = board
    kb.complete_task(conn, tid, summary="SUMMARY=done\nNEXT_STEP=none\n")
    conn.execute("UPDATE task_events SET run_id = NULL WHERE task_id = ?", (tid,))
    conn.commit()

    lookups: list = []

    class _KB:
        get_task = staticmethod(kb.get_task)

        @staticmethod
        def get_run(_conn, _run_id):
            lookups.append(_run_id)
            return kb.get_run(_conn, _run_id)

    delivery = _claim(_collector(_KB), conn, tid)

    assert delivery is not None, "a NULL run_id must not abort the delivery"
    assert delivery["run_summaries"] == {}
    # Skipped before the lookup: get_run coerces with int(), so a None would raise.
    assert lookups == [], "a legacy row must not reach get_run at all"


def _raise(_conn, _run_id):
    raise RuntimeError("run lookup exploded")
