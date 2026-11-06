"""Each request in one Slack thread owns its receipt and processing reactions."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_results import execution_result
from tests.gateway.test_slack_queued_status import RecordingSlack
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["im", "channel"], ids=["dm-thread", "channel-thread"])
@pytest.mark.parametrize("queued", [True, False], ids=["busy-followup", "idle-followup"])
async def test_each_thread_request_has_independent_reactions(monkeypatch, chat_type, queued):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlack()
    adapter._bot_user_id = "U_BOT"
    adapter._team_bot_user_ids["T_FIXTURE"] = "U_BOT"
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="thread-request-reactions")
    runner._wire_adapter_handlers(adapter)
    authority = runner.session_authority
    # Only unavailable Slack enrichment/transport is replaced; intake, channel
    # gating, receipt tracking, native FIFO and settlement remain real.
    adapter._hydrate_thread_context = AsyncMock(return_value=(None, [], []))
    adapter._resolve_user_name = AsyncMock(return_value="Fixture User")
    adapter._resolve_channel_name = AsyncMock(return_value="Fixture Channel")
    adapter._set_assistant_thread_title = AsyncMock()
    channel = "D_FIXTURE" if chat_type == "im" else "C_FIXTURE"
    root, followup = "1234567890.100001", "1234567890.100002"
    started = {name: asyncio.Event() for name in ("one", "two")}
    finish = {name: asyncio.Event() for name in ("one", "two")}
    sending, delivered = asyncio.Event(), asyncio.Event()
    original_send = adapter.send

    async def turn(event, source, quick_key, run_generation):
        name = event.text.strip()
        started[name].set()
        await finish[name].wait()
        execution_result.get()["result"] = {"final_response": f"reply {name}", "messages": []}
        return f"reply {name}"

    async def send(chat_id, content, reply_to=None, metadata=None):
        if "reply two" in content:
            sending.set()
            await delivered.wait()
        return await original_send(chat_id, content, reply_to, metadata)

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    monkeypatch.setattr(adapter, "send", send)

    def event(ts, text, *, thread=None):
        return {"text": text, "user": "fixture-user", "channel": channel,
                "channel_type": chat_type, "team": "T_FIXTURE", "ts": ts,
                "client_msg_id": f"request-{ts}", **({"thread_ts": thread} if thread else {})}

    try:
        await adapter._handle_slack_message(event(root, "one" if chat_type == "im" else "<@U_BOT> one"))
        await asyncio.wait_for(started["one"].wait(), 5)
        if not queued:
            finish["one"].set()
            await async_wait_until(lambda: not adapter._background_tasks, timeout=10)
            assert (root, "add", "white_check_mark") in adapter.reactions
        await adapter._handle_slack_message(event(followup, "two", thread=root))
        assert (followup, "add", "eyes") in adapter.reactions, "new thread request was not acknowledged"
        if queued:
            assert not started["two"].is_set()
            assert (followup, "add", "hourglass_flowing_sand") not in adapter.reactions
            finish["one"].set()
        await asyncio.wait_for(started["two"].wait(), 5)
        assert (followup, "add", "hourglass_flowing_sand") in adapter.reactions
        assert (followup, "add", "white_check_mark") not in adapter.reactions
        finish["two"].set()
        await asyncio.wait_for(sending.wait(), 5)
        assert (followup, "add", "white_check_mark") not in adapter.reactions
        delivered.set()
        await async_wait_until(lambda: not adapter._background_tasks, timeout=10)
        await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task))
        for timestamp in (root, followup):
            assert adapter.reactions.count((timestamp, "add", "white_check_mark")) == 1
            assert (timestamp, "remove", "hourglass_flowing_sand") in adapter.reactions
            assert adapter._workspace_message_marker("T_FIXTURE", timestamp) not in adapter._reacting_message_ids
        assert len(authority.sessions) == 1
    finally:
        for release in finish.values():
            release.set()
        delivered.set()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
        await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task), return_exceptions=True)
