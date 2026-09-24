"""Push admission and durable notifier retries use real adapter/SQLite lifecycles."""
import asyncio

import pytest

from evals.heartbeat_idle_wire import WireAdapter
from gateway.config import Platform, PlatformConfig
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.wake import admit_internal_event, deliver_wake
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


def setup_route(raft=False):
    if raft:
        from plugins.platforms.raft.adapter import RaftAdapter
        adapter = RaftAdapter(PlatformConfig(enabled=True, typing_indicator=False,
            extra={"bridge_token": "owned-test-token", "port": 0}))
    else:
        adapter = WireAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.wire = []
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner.adapters = {adapter.platform: adapter}
    runner._adapter_for_source = lambda source: adapter
    runner._kanban_dispatcher_lock_handle = object()
    source = SessionSource(platform=adapter.platform, chat_id="42", user_id="42", chat_type="dm")
    return runner, adapter, source, build_session_key(source)


async def drain(adapter):
    while adapter._background_tasks:
        await asyncio.gather(*list(adapter._background_tasks))


@pytest.mark.asyncio
@pytest.mark.parametrize("raft", [False, True])
async def test_push_receipt_requires_real_admission_without_displacing_user(raft, monkeypatch):
    runner, adapter, source, key = setup_route(raft)
    if raft:
        monkeypatch.setattr(adapter, "_spawn_bridge", lambda port: None)
    release, started = asyncio.Event(), asyncio.Event()
    received = []

    async def handler(event):
        received.append(event.text)
        started.set()
        await release.wait()
        # Real runner FIFO promotion at the fake model boundary.
        if key not in adapter._pending_messages:
            pending = runner._promote_queued_event(key, adapter, None)
            if pending is not None:
                adapter._pending_messages[key] = pending

    adapter.set_message_handler(handler)
    await adapter.connect()
    try:
        await deliver_wake(adapter, text="idle", source=source)
        await asyncio.wait_for(started.wait(), 2)
        assert await adapter.handle_message(MessageEvent(text="human", source=source)) is None
        with pytest.raises(RuntimeError, match="not accepted"):
            await deliver_wake(adapter, text="no-fifo", source=source)
        assert adapter._pending_messages[key].text == "human"
        adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
        runner._BUSY_QUEUE_MAX_PENDING = 1
        with pytest.raises(RuntimeError, match="not accepted"):
            await deliver_wake(adapter, text="at-cap", source=source)
        assert adapter._pending_messages[key].text == "human"
        runner._BUSY_QUEUE_MAX_PENDING = 3
        await deliver_wake(adapter, text="busy", source=source)
        assert runner._queue_depth(key, adapter=adapter) == 2
        rejected = MessageEvent(text="wrong-key", source=source, internal=True,
                                metadata={"gateway_session_key": "agent:wrong"})
        with pytest.raises(RuntimeError, match="not accepted"):
            await admit_internal_event(adapter, rejected)
        assert rejected._gateway_accepted is False
        release.set()
        await drain(adapter)
        assert received == ["idle", "human", "busy"]
        adapter.set_message_handler(None)
        with pytest.raises(RuntimeError, match="not accepted"):
            await deliver_wake(adapter, text="no-handler", source=source)
    finally:
        release.set()
        await drain(adapter)
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_notifier_retries_unaccepted_wake_without_repeating_pings(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    runner, adapter, source, key = setup_route()
    conn = kbc.connect()
    tids = {}
    try:
        for mode in ("notify+wake", "wake", "notify"):
            tid = kb.create_task(conn, title=mode, assignee="worker", session_id=key)
            kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="42",
                               user_id="42", chat_type="dm", delivery_mode=mode)
            kb.complete_task(conn, tid, summary="handoff")
            tids[mode] = tid
    finally:
        conn.close()

    failures = {}

    async def tick():
        deliveries = await asyncio.to_thread(_notifier_collect, runner, kb,
            notifier_profile=None, gc_due=False, gc_retention_days=30)
        for delivery in deliveries:
            await _KanbanNotification(runner, delivery, platform_cls=Platform,
                                      sub_fail_counts=failures).deliver()

    def unseen(mode):
        conn = kbc.connect()
        try:
            return kbn.unseen_events_for_sub(conn, task_id=tids[mode], platform="telegram",
                                            chat_id="42", kinds=["completed"])[1]
        finally:
            conn.close()

    await adapter.connect()
    await tick()  # send works, but no message handler has been installed yet
    assert len(adapter.wire) == 2
    assert unseen("notify+wake") and unseen("wake")
    assert not unseen("notify")
    # New notifier instances and DB connections replay the durable claim, not the ping.
    for _ in range(13):
        await tick()
    assert len(adapter.wire) == 2
    assert unseen("notify+wake") and unseen("wake")
    assert failures == {}
    received = []

    async def handler(event):
        received.append(event.text)

    adapter.set_message_handler(handler)
    await tick()
    await drain(adapter)
    await tick()
    assert len(received) == 2
    assert all("handoff" in text for text in received)
    assert len(adapter.wire) == 2
    assert not any(unseen(mode) for mode in tids)
    await adapter.disconnect()
