"""Tests for the kanban notifier zero-subscription early exit.

The notifier used to writable-open EVERY board DB on every tick even when a
board had zero subscriptions — paying schema init/migration on first open,
WAL/-shm sidecar creation, and checkpoint traffic for boards with nothing to
notify. Per-board work is now gated by a read-only subscription probe
(``kanban_db.count_notify_subs``), so boards with zero subscriptions are
only opened writable on hourly retention sweeps.

(The companion machine-global ``.notifier.lock`` singleton gate from PR
#63001 was deliberately NOT salvaged: a lock-winning default-profile gateway
cannot deliver a secondary profile's subscriptions in standalone-profile
deployments — profile routing fails closed in
``gateway/authz_mixin.py::_authorization_adapter`` — so the lock could
suppress delivery entirely. The read-only probe captures the per-tick cost
win without that risk.)
"""

import asyncio

from unittest.mock import patch

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    return runner


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _create_completed_task(*, subscribe: bool) -> str:
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="owner gate", assignee="worker")
        if subscribe:
            kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary="done")
        return tid
    finally:
        conn.close()


def test_zero_sub_board_skips_writable_open_between_gc_sweeps(tmp_path, monkeypatch):
    """Ordinary ticks preserve the read-only fast path."""
    db_path = tmp_path / "zero-subs.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    _create_completed_task(subscribe=False)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    with patch.object(kbc, "connect", wraps=kbc.connect) as spy_connect:
        from gateway.kanban_watchers_notifier import _notifier_collect
        _notifier_collect(runner, kb, notifier_profile="default", gc_due=False, gc_retention_days=30)

    spy_connect.assert_not_called()
    assert adapter.sent == []

def test_zero_sub_board_prunes_terminal_receipts_on_gc_tick(tmp_path, monkeypatch):
    import time
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "gc.db"))
    kb.init_db()
    tid = _create_completed_task(subscribe=False)
    conn = kbc.connect()
    try:
        kbn.record_notify_delivery(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            thread_id="", event_id=1, event_kind="completed", message_id="sent",
            delivered_at=int(time.time()) - 91 * 86400,
        )
    finally:
        conn.close()
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    conn = kbc.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM kanban_notify_deliveries").fetchone()[0] == 0
    finally:
        conn.close()
    assert adapter.sent == []
