"""Persisted busy state must settle when the final-delivery owner finishes."""

import asyncio
from unittest.mock import Mock

import pytest

from gateway import delivery_ledger as ledger
from gateway.delivery_work import delivery_work
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["success", "cancel", "followup"])
async def test_adapter_delivery_publishes_idle_after_its_task_finishes(
    tmp_path, monkeypatch, finish
):
    runner, adapter = make_restart_runner()
    adapter.gateway_runner = runner
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.db")
    published = []
    monkeypatch.setattr(
        "gateway.run._write_runtime_status_quiet",
        lambda **kw: published.append(kw["active_agents"]),
    )
    source = make_restart_source()
    key = runner._session_key_for_source(source)
    runner._running_agents[key] = object()
    entered, release = asyncio.Event(), asyncio.Event()
    followup_entered, followup_release = asyncio.Event(), asyncio.Event()
    original_send = adapter.send

    async def handler(event):
        runner._running_agents.pop(key, None)
        runner._persist_active_agents()
        return event.text + " answer"

    async def send(*args, **kwargs):
        if "followup" in kwargs.get("content", ""):
            followup_entered.set()
            await followup_release.wait()
        else:
            entered.set()
            await release.wait()
        return await original_send(*args, **kwargs)

    adapter.set_message_handler(handler)
    monkeypatch.setattr(adapter, "send", send)
    event = MessageEvent(
        text="question",
        message_type=MessageType.TEXT,
        source=source,
        message_id="request",
    )
    assert adapter._start_session_processing(event, key)
    task = adapter._session_tasks[key]
    try:
        await asyncio.wait_for(entered.wait(), 10)
        assert published[-1] == 1
        if finish == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            if finish == "followup":
                adapter._pending_messages[key] = MessageEvent(
                    text="followup",
                    message_type=MessageType.TEXT,
                    source=source,
                    message_id="next",
                )
            release.set()
            await asyncio.wait_for(task, 10)
            if finish == "followup":
                await asyncio.wait_for(followup_entered.wait(), 10)
                assert published[-1] == 1
                successor = adapter._session_tasks[key]
                followup_release.set()
                await asyncio.wait_for(successor, 10)
        assert runner._active_work_count() == 0
        assert published[-1] == 0
    finally:
        release.set()
        followup_release.set()
        if not task.done():
            await task


@pytest.mark.asyncio
async def test_replay_publishes_idle_after_its_delivery_lease_is_released(monkeypatch):
    runner, _ = make_restart_runner()
    published = []
    monkeypatch.setattr(
        "gateway.run._write_runtime_status_quiet",
        lambda **kw: published.append(kw["active_agents"]),
    )
    with delivery_work(runner):
        assert published[-1] == 1
    assert runner._active_work_count() == 0
    assert published[-1] == 0


@pytest.mark.asyncio
async def test_status_storage_failure_does_not_replace_a_delivery_result(monkeypatch):
    runner, _ = make_restart_runner()

    def unavailable():
        raise OSError("status storage unavailable")

    monkeypatch.setattr(runner, "_persist_active_agents", unavailable)
    with delivery_work(runner):
        assert runner._active_work_count() == 1
    assert runner._active_work_count() == 0


@pytest.mark.asyncio
async def test_inactive_runner_does_not_publish_gateway_status(monkeypatch):
    runner, _ = make_restart_runner()
    runner._running = False
    persist = Mock()
    monkeypatch.setattr(runner, "_persist_active_agents", persist)
    with delivery_work(runner):
        assert runner._active_work_count() == 1
    assert runner._active_work_count() == 0
    persist.assert_not_called()
