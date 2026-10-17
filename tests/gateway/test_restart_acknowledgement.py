"""A chat-originated restart must finish its acknowledgement before teardown."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway import slash_commands
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import (
    RestartTestAdapter,
    make_restart_runner,
    make_restart_source,
)


@pytest.fixture
def restart_notice(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr("gateway.restart.is_gateway_supervisor_process", lambda: True)
    runner, primary = make_restart_runner()
    runner.config.multiplex_profiles = True
    runner._primary_profile_name = "default"
    adapter = RestartTestAdapter()
    runner._profile_adapters = {"secondary": {Platform.TELEGRAM: adapter}}
    runner.request_restart = MagicMock(return_value=True)
    source = make_restart_source(thread_id="73")
    source.profile = "secondary"
    event = MessageEvent(
        text="/restart", message_type=MessageType.TEXT, source=source,
        message_id="m1", platform_update_id=100,
    )
    return runner, adapter, primary, event


@pytest.mark.asyncio
@pytest.mark.parametrize("active_agents", [0, 1], ids=["idle", "busy"])
@pytest.mark.parametrize("chat_type,thread_id,reply_to", [
    ("dm", None, "m1"),
    ("dm", "73", "m1"),
    ("group", "73", None),
], ids=["dm", "dm-topic", "forum-topic"])
async def test_restart_waits_for_acknowledgement(
    restart_notice, monkeypatch, active_agents, chat_type, thread_id, reply_to,
):
    runner, adapter, primary, event = restart_notice
    event.source.chat_type, event.source.thread_id = chat_type, thread_id
    runner._running_agent_count = lambda: active_agents
    started, release = asyncio.Event(), asyncio.Event()
    order = []

    async def send(**kwargs):
        started.set()
        await release.wait()
        order.append("delivered")
        return SendResult(success=True, message_id="ack")

    adapter.send = AsyncMock(side_effect=send)
    monkeypatch.setattr(RestartTestAdapter, "delete_message", AsyncMock(return_value=True))
    adapter._get_ephemeral_system_ttl_default = lambda: 60
    adapter._schedule_ephemeral_delete = MagicMock()
    runner.request_restart.side_effect = lambda **kwargs: order.append("restart")
    task = asyncio.create_task(runner._handle_restart_command(event))
    sending = asyncio.create_task(started.wait())
    try:
        await asyncio.wait_for(
            asyncio.wait({task, sending}, return_when=asyncio.FIRST_COMPLETED), 5,
        )
        runner.request_restart.assert_not_called()
        assert started.is_set()
        assert runner._draining
        assert not task.done()
        release.set()
        assert await asyncio.wait_for(task, 5) == ""
    finally:
        release.set()
        for pending in (task, sending):
            if not pending.done():
                pending.cancel()
        await asyncio.gather(task, sending, return_exceptions=True)

    assert order == ["delivered", "restart"]
    runner.request_restart.assert_called_once_with(detached=False, via_service=True)
    adapter.send.assert_awaited_once()
    sent = adapter.send.call_args.kwargs
    assert sent["reply_to"] == reply_to
    assert sent["metadata"]["notify"] is True
    if thread_id:
        assert sent["metadata"]["thread_id"] == thread_id
        assert sent["metadata"]["hermes_profile"] == "secondary"
    if chat_type == "dm" and thread_id:
        assert sent["metadata"]["direct_messages_topic_id"] == thread_id
    assert not primary.sent
    if active_agents:
        adapter._schedule_ephemeral_delete.assert_not_called()
    else:
        adapter._schedule_ephemeral_delete.assert_called_once_with(event.source.chat_id, "ack", 60)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raised", "rejected", "timeout", "missing-adapter"])
async def test_restart_proceeds_when_acknowledgement_cannot_be_delivered(
    restart_notice, monkeypatch, failure,
):
    runner, adapter, _primary, event = restart_notice
    cancelled = asyncio.Event()

    async def stalled_send(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    if failure == "raised":
        adapter.send = AsyncMock(side_effect=OSError("transport unavailable"))
    elif failure == "rejected":
        adapter.send = AsyncMock(return_value=SendResult(success=False, error="ReadTimeout"))
    elif failure == "timeout":
        adapter.send = AsyncMock(side_effect=stalled_send)
        monkeypatch.setattr(slash_commands, "_RESTART_ACK_TIMEOUT_SECONDS", 0)
    else:
        runner._profile_adapters = {}

    assert await asyncio.wait_for(runner._handle_restart_command(event), 5) == ""
    runner.request_restart.assert_called_once_with(detached=False, via_service=True)
    if failure == "timeout":
        assert cancelled.is_set()
