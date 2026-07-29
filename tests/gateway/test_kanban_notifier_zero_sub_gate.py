"""Tests for the kanban notifier zero-subscription early exit.

The notifier used to writable-open EVERY board DB on every tick even when a
board had zero subscriptions — paying schema init/migration on first open,
WAL/-shm sidecar creation, and checkpoint traffic for boards with nothing to
notify. Per-board work is now gated by a read-only subscription probe
(``kanban_db.count_notify_subs``), so boards with zero subscriptions are
never opened writable.

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
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="owner gate", assignee="worker")
        if subscribe:
            kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary="done")
        return tid
    finally:
        conn.close()


def test_zero_sub_board_is_never_opened_writable(tmp_path, monkeypatch):
    """A board with zero subscriptions must be skipped BEFORE `_kb.connect`."""
    db_path = tmp_path / "zero-subs.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    _create_completed_task(subscribe=False)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    spy_connect.assert_not_called()
    assert adapter.sent == []


def test_subscribed_board_still_delivers_through_the_gate(tmp_path, monkeypatch):
    """Regression: the zero-sub probe must not change delivery for live subs."""
    db_path = tmp_path / "subscribed.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_task(subscribe=True)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert tid in adapter.sent[0]["text"]


def test_probe_failure_falls_back_to_writable_open(tmp_path, monkeypatch):
    """If the read-only probe raises (locked/corrupt DB), the notifier must
    fall back to the writable open — a broken probe must never silently
    disable notifications for a board with live subscriptions."""
    db_path = tmp_path / "probe-broken.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_task(subscribe=True)

    def _boom(*args, **kwargs):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(kb, "count_notify_subs", _boom)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert tid in adapter.sent[0]["text"]
