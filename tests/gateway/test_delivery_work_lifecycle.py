"""Final delivery must finish before dormancy or restart drains the transport."""

import asyncio
import threading

import pytest

from gateway import delivery_ledger as ledger
from gateway.delivery_work import delivery_work, messaging_work_count
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def idle_runner(monkeypatch):
    runner, adapter = make_restart_runner()
    runner._last_inbound_at = 0
    monkeypatch.setattr(runner, "_scale_to_zero_idle_timeout_seconds", lambda: 120)
    monkeypatch.setattr(
        runner, "_scale_to_zero_has_live_background_work", lambda: False
    )
    monkeypatch.setattr(
        "gateway.scale_to_zero.dashboard_client_last_seen", lambda: None
    )
    monkeypatch.setattr("cron.scheduler.get_running_job_ids", lambda: set())
    return runner, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["record", "send", "settle"])
async def test_real_final_pipeline_stays_awake_until_ledger_settles(
    tmp_path, monkeypatch, phase
):
    runner, adapter = idle_runner(monkeypatch)
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.db")
    entered, release = threading.Event(), threading.Event()
    send_entered, send_release = asyncio.Event(), asyncio.Event()
    if phase != "send":
        method = "record_obligation" if phase == "record" else "mark_delivered"
        original = getattr(ledger, method)

        def held(*args, **kwargs):
            entered.set()
            assert release.wait(10), "test did not release the ledger operation"
            return original(*args, **kwargs)

        monkeypatch.setattr(ledger, method, held)
    original_send = adapter.send

    async def send(*args, **kwargs):
        send_entered.set()
        if phase == "send":
            await send_release.wait()
        return await original_send(*args, **kwargs)

    adapter.send = send
    source = make_restart_source()
    key = runner._session_key_for_source(source)
    runner._running_agents[key] = object()

    async def handler(event):
        runner._running_agents.pop(key)
        return "completed response"

    adapter._message_handler = handler
    event = MessageEvent(
        text="question",
        message_type=MessageType.TEXT,
        source=source,
        message_id="request",
    )
    task = asyncio.create_task(adapter._process_message_background(event, key))
    adapter._session_tasks[key] = task
    try:
        if phase == "send":
            await asyncio.wait_for(send_entered.wait(), 10)
        else:
            assert await asyncio.to_thread(entered.wait, 10)
        assert runner._running_agent_count() == 0
        assert runner._active_work_count() == 1
        assert runner._awaitable_work_count() == 1
        assert runner._scale_to_zero_is_idle() is False
    finally:
        release.set()
        send_release.set()
        await asyncio.wait_for(task, 10)
    assert runner._scale_to_zero_is_idle() is True
    with ledger._connect() as db:
        assert (
            db.execute("SELECT state FROM delivery_obligations").fetchone()[0]
            == "delivered"
        )


@pytest.mark.asyncio
async def test_task_accounting_preserves_profiles_dedup_and_cancellation(monkeypatch):
    runner, adapter = idle_runner(monkeypatch)
    waiting = asyncio.Event()
    task = asyncio.create_task(waiting.wait())
    runner._running_agents["turn"] = object()
    adapter._session_tasks["turn"] = task
    runner._profile_adapters = {"secondary": {adapter.platform: adapter}}
    try:
        assert messaging_work_count(runner) == 1
        runner._running_agents.clear()
        assert messaging_work_count(runner) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert messaging_work_count(runner) == 0
    finally:
        waiting.set()
        if not task.done():
            await task


@pytest.mark.asyncio
async def test_replay_work_is_reentrant_and_released_on_cancellation(monkeypatch):
    runner, _ = idle_runner(monkeypatch)
    entered = asyncio.Event()
    never = asyncio.Event()

    async def replay():
        with delivery_work(runner), delivery_work(runner):
            entered.set()
            await never.wait()

    task = asyncio.create_task(replay())
    await entered.wait()
    assert runner._scale_to_zero_is_idle() is False
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runner._scale_to_zero_is_idle() is True


def test_unknown_adapter_task_state_fails_awake(monkeypatch):
    runner, adapter = idle_runner(monkeypatch)

    class Unreadable:
        def done(self):
            raise RuntimeError("unreadable task state")

    adapter._session_tasks["turn"] = Unreadable()
    assert runner._scale_to_zero_is_idle() is False
