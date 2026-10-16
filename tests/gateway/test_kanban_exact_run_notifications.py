"""Disk-backed finalizer -> board -> notifier regressions, no real messages."""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from agent.turn_finalizer import _resolve_budget_fallback
from gateway.config import Platform
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


class Transport:
    def __init__(self):
        self.sent, self.wakes = [], []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append(text)

    async def handle_message(self, event):
        self.wakes.append(event.text)
        event._gateway_accepted = True


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    conn = kbc.connect()
    yield conn
    conn.close()


def claim(conn):
    tid = kb.create_task(conn, title="Synthetic fixture", assignee="builder")
    task = kb.claim_task(conn, tid, claimer="test:builder")
    assert task is not None
    return tid, task.current_run_id


def subscribe(conn, tid):
    kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="42",
                       delivery_mode="notify+wake")


def finalizer(monkeypatch, tid, rid):
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(rid))
    agent = SimpleNamespace(max_iterations=20, iteration_budget=None)
    _resolve_budget_fallback(agent, final_response="retained", api_call_count=20,
                             interrupted=False, failed=False, messages=[],
                             _pending_verification_response=None,
                             _pending_verification_response_previewed=False,
                             _turn_exit_reason="budget_exhausted", logger=logging.getLogger(__name__))


def runner(adapter):
    r = GatewayRunner.__new__(GatewayRunner)
    r.adapters = {Platform.TELEGRAM: adapter}
    r._kanban_dispatcher_lock_handle = object()
    return r


def collect(r):
    return _notifier_collect(r, kb, notifier_profile=None, gc_due=False, gc_retention_days=30)


async def deliver(r, ds):
    for d in ds:
        await _KanbanNotification(r, d, platform_cls=Platform, sub_fail_counts={}).deliver()


def snapshot(conn, tid):
    return (dict(conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()),
            [dict(row) for row in conn.execute("SELECT * FROM task_runs WHERE task_id=?", (tid,))],
            [dict(row) for row in conn.execute("SELECT * FROM task_events WHERE task_id=?", (tid,))])


@pytest.mark.parametrize("state", ["done", "review", "blocked", "triage", "successor", "active", "missing", "malformed", "exhausted"])
def test_finalizer_only_spends_current_active_run(board, monkeypatch, state):
    tid, rid = claim(board)
    if state == "done":
        assert kb.complete_task(board, tid, summary="Retained result", expected_run_id=rid)
    elif state == "review":
        assert kb.request_review(board, tid, summary="Review result", expected_run_id=rid)
    elif state in {"blocked", "triage"}:
        assert kb.block_task(board, tid, reason="Needs decision", kind="needs_input", expected_run_id=rid)
        if state == "triage":
            with kb.write_txn(board):
                board.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
    elif state == "successor":
        finalizer(monkeypatch, tid, rid)
        assert kb.claim_task(board, tid, claimer="test:successor")
    elif state == "exhausted":
        with kb.write_txn(board):
            board.execute("UPDATE tasks SET max_retries=1 WHERE id=?", (tid,))
    subscribe(board, tid)  # existing handoff already consumed by origin
    before = snapshot(board, tid)
    identity = {"missing": "", "malformed": "not-a-run"}.get(state, rid)
    finalizer(monkeypatch, tid, identity)
    adapter = Transport()
    r = runner(adapter)
    asyncio.run(deliver(r, collect(r)))
    if state in {"active", "exhausted"}:
        after = snapshot(board, tid)
        assert after[0]["consecutive_failures"] == before[0]["consecutive_failures"] + 1
        assert after[0]["status"] == ("blocked" if state == "exhausted" else "ready")
        assert len(adapter.sent) == len(adapter.wakes) == 1
        if state == "exhausted":
            assert "retry" not in adapter.sent[0]
        finalizer(monkeypatch, tid, identity)
        assert snapshot(board, tid) == after
        asyncio.run(deliver(r, collect(r)))
        assert len(adapter.sent) == len(adapter.wakes) == 1
        if state == "active":
            event = [e for e in kb.list_events(board, tid) if e.kind == "timed_out"][0]
            with kb.write_txn(board):
                kb._append_event(board, tid, "timed_out", event.payload, run_id=rid)
            before_duplicate = snapshot(board, tid)
            asyncio.run(deliver(r, collect(r)))
            assert snapshot(board, tid) == before_duplicate
            assert len(adapter.sent) == len(adapter.wakes) == 1
    else:
        assert snapshot(board, tid) == before
        assert adapter.sent == adapter.wakes == []


@pytest.mark.parametrize("case", ["legacy-null", "ended-run", "replaced", "claim-race", "resolved", "new-approval", "exhausted-crash"])
def test_obsolete_events_are_silent_but_new_approval_survives(board, monkeypatch, case):
    tid, rid = claim(board)
    adapter = Transport()
    r = runner(adapter)
    if case in {"resolved", "new-approval"}:
        subscribe(board, tid)
        assert kb.block_task(board, tid, reason="Approval A", kind="needs_input", expected_run_id=rid)
        assert kb.unblock_task(board, tid)
        if case == "new-approval":
            assert kb.block_task(board, tid, reason="Approval B", kind="needs_input")
    elif case == "claim-race":
        subscribe(board, tid)
        finalizer(monkeypatch, tid, rid)
    else:
        assert kb.complete_task(board, tid, summary="Preserved", expected_run_id=rid)
        subscribe(board, tid)
        with kb.write_txn(board):
            if case == "replaced":
                board.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
            kb._append_event(board, tid, "timed_out", {"retry_status": "ready"},
                             run_id=None if case == "legacy-null" else rid)
        if case == "replaced":
            successor = kb.claim_task(board, tid, claimer="test:new")
            assert successor is not None
            finalizer(monkeypatch, tid, successor.current_run_id)
            # Old envelope appended after newer run ended must not look current.
            with kb.write_txn(board):
                kb._append_event(board, tid, "timed_out", {"retry_status": "ready"}, run_id=rid)
        if case == "exhausted-crash":
            with kb.write_txn(board):
                board.execute("UPDATE tasks SET status='blocked' WHERE id=?", (tid,))
                kb._append_event(board, tid, "gave_up", {"retry_status": "ready"})
    ds = collect(r)
    if case == "claim-race":
        assert kb.claim_task(board, tid, claimer="test:next")
    before = snapshot(board, tid)
    asyncio.run(deliver(r, ds))
    asyncio.run(deliver(r, collect(r)))
    assert snapshot(board, tid) == before
    if case in {"new-approval", "exhausted-crash"}:
        assert len(adapter.sent) == len(adapter.wakes) == 1
        assert "retry" not in adapter.sent[0]
        if case == "new-approval":
            assert "Approval B" in adapter.sent[0] and "Approval A" not in adapter.sent[0]
    else:
        assert adapter.sent == adapter.wakes == []
