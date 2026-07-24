"""Tests for the kanban notifier single-owner gate + zero-subscription skip.

The notifier used to writable-open EVERY board DB from EVERY gateway whose
config left ``kanban.dispatch_in_gateway`` at the default (true) — the exact
N-gateway ``-shm``/``-wal`` contention its own top-of-function comment
claimed to prevent. It now mirrors the dispatcher's machine-global singleton
advisory lock with its own ``<kanban root>/kanban/.notifier.lock``: the
lock-losing gateway polls zero boards and opens zero connections. Per-board
work is further gated by a read-only subscription probe
(``kanban_db.count_notify_subs``), so boards with zero subscriptions are
never opened writable.
"""

import asyncio

from unittest.mock import patch

from gateway.config import Platform
from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock
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
    """Regression: the lock + probe must not change delivery for the owner."""
    db_path = tmp_path / "subscribed.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_task(subscribe=True)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert tid in adapter.sent[0]["text"]


def test_lock_losing_instance_polls_zero_boards(tmp_path, monkeypatch):
    """While another holder owns `.notifier.lock`, the watcher must return at
    the gate: zero board enumerations, zero probes, zero DB opens, zero
    deliveries."""
    db_path = tmp_path / "locked-out.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    _create_completed_task(subscribe=True)

    lock_path = kb.kanban_home() / "kanban" / ".notifier.lock"
    handle, state = _acquire_singleton_lock(lock_path)
    assert state == "held", "test precondition: we hold the notifier lock"
    try:
        adapter = RecordingAdapter()
        runner = _make_runner(adapter)
        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect, \
                patch.object(kb, "list_boards", wraps=kb.list_boards) as spy_boards, \
                patch.object(kb, "count_notify_subs", wraps=kb.count_notify_subs) as spy_probe:
            asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    finally:
        _release_singleton_lock(handle)

    spy_boards.assert_not_called()
    spy_probe.assert_not_called()
    spy_connect.assert_not_called()
    assert adapter.sent == []


def test_two_instances_exactly_one_polls(tmp_path, monkeypatch):
    """Two concurrent notifier instances against one kanban root: the first
    acquires `.notifier.lock` and delivers; the second returns at the lock —
    no second poller, no double delivery."""
    db_path = tmp_path / "two-instances.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    _create_completed_task(subscribe=True)

    adapter1 = RecordingAdapter()
    adapter2 = RecordingAdapter()
    runner1 = _make_runner(adapter1)
    runner2 = _make_runner(adapter2)

    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner1._running = False
        runner2._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def run_both():
        await asyncio.gather(
            runner1._kanban_notifier_watcher(interval=1),
            runner2._kanban_notifier_watcher(interval=1),
        )

    asyncio.run(run_both())

    deliveries = adapter1.sent + adapter2.sent
    assert len(deliveries) == 1, f"exactly one delivery expected, got {deliveries!r}"
    # gather() starts runner1 first, so it wins the lock deterministically;
    # runner2 must be the locked-out instance.
    assert adapter2.sent == []


def test_notifier_lock_released_on_return(tmp_path, monkeypatch):
    """A finished watcher must release `.notifier.lock` so a successor in the
    same process (gateway restart-in-place, next test) can acquire it."""
    db_path = tmp_path / "release.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    _create_completed_task(subscribe=True)

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.sent) == 1

    lock_path = kb.kanban_home() / "kanban" / ".notifier.lock"
    handle, state = _acquire_singleton_lock(lock_path)
    try:
        assert state == "held", "watcher exit must release the notifier lock"
    finally:
        _release_singleton_lock(handle)
