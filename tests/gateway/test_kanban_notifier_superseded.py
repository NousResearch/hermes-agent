"""Resolved attention notices must be dropped at delivery, including post-claim changes."""
import asyncio

import pytest

from gateway.config import Platform
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_notify as kbn
from tests.gateway.test_kanban_notifier import RecordingAdapter, _make_runner


@pytest.mark.parametrize("resolve_at", ["before_claim", "after_claim", "after_ping"])
def test_unblocked_task_drops_old_attention_and_keeps_current_reblock(tmp_path, monkeypatch, resolve_at):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="attention", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1", delivery_mode="notify+wake")
        assert kb.block_task(conn, tid, reason="old reason", kind="needs_input")
        if resolve_at == "before_claim":
            assert kb.unblock_task(conn, tid)
    finally:
        conn.close()
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    if resolve_at == "after_ping":
        send = adapter.send

        async def send_then_resolve(*args, **kwargs):
            result = await send(*args, **kwargs)
            _unblock(tid)
            return result

        monkeypatch.setattr(adapter, "send", send_then_resolve)

    async def deliver_claimed():
        deliveries = _notifier_collect(runner, kb, notifier_profile=None, gc_due=False, gc_retention_days=30)
        if resolve_at == "after_claim":
            _unblock(tid)
        for delivery in deliveries:
            await _KanbanNotification(runner, delivery, platform_cls=Platform, sub_fail_counts={}).deliver()

    asyncio.run(deliver_claimed())
    assert len(adapter.sent) == int(resolve_at == "after_ping") and adapter.handled == []
    adapter.sent.clear()
    if resolve_at == "after_ping":
        monkeypatch.setattr(adapter, "send", send)
    conn = kbc.connect()
    try:
        assert kb.block_task(conn, tid, reason="current reason", kind="needs_input")
    finally:
        conn.close()
    # The stale cursor settled, and a new attention event still reaches both channels.
    asyncio.run(_deliver_current(runner))
    assert len(adapter.sent) == 1 and "current reason" in adapter.sent[0]["text"]
    assert len(adapter.handled) == 1


def _unblock(tid):
    conn = kbc.connect()
    try:
        assert kb.unblock_task(conn, tid)
    finally:
        conn.close()


async def _deliver_current(runner):
    for delivery in _notifier_collect(runner, kb, notifier_profile=None, gc_due=False, gc_retention_days=30):
        await _KanbanNotification(runner, delivery, platform_cls=Platform, sub_fail_counts={}).deliver()
