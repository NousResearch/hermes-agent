"""Slack receipt, actual processing and delivered-result reactions."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import ProcessingOutcome
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture
def adapter():
    instance = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    instance._app = MagicMock()
    instance._app.client = AsyncMock()
    instance._app.client.users_info = AsyncMock(return_value={
        "user": {"is_bot": False, "profile": {"display_name": "User"}}
    })
    instance._bot_user_id = "U_BOT"
    instance._running = True
    instance.handle_message = AsyncMock()
    return instance


@pytest.mark.asyncio
async def test_receipt_is_visible_before_dispatch_and_work_has_distinct_marker(adapter):
    operations = []

    async def add(**kwargs):
        operations.append(("add", kwargs["name"]))

    async def remove(**kwargs):
        operations.append(("remove", kwargs["name"]))

    async def dispatch(event):
        assert operations == [("add", "eyes")]

    client = adapter._app.client
    client.reactions_add.side_effect = add
    client.reactions_remove.side_effect = remove
    adapter.handle_message.side_effect = dispatch
    await adapter._handle_slack_message({
        "text": "hello", "user": "U_USER", "channel": "D_CHAT",
        "channel_type": "im", "ts": "1234567890.000001", "client_msg_id": "receipt",
    })
    message = adapter.handle_message.call_args.args[0]
    assert operations == [("add", "eyes")]
    await adapter.on_processing_start(message)
    assert operations == [
        ("add", "eyes"), ("add", "hourglass_flowing_sand"), ("remove", "eyes"),
    ]
    await adapter.on_processing_complete(message, ProcessingOutcome.SUCCESS)
    assert operations == [
        ("add", "eyes"), ("add", "hourglass_flowing_sand"), ("remove", "eyes"),
        ("remove", "hourglass_flowing_sand"), ("remove", "eyes"),
        ("add", "white_check_mark"),
    ]
    assert message.message_id not in adapter._reacting_message_ids


@pytest.mark.asyncio
async def test_failed_or_cancelled_turn_never_receives_success(adapter):
    for outcome in (ProcessingOutcome.FAILURE, ProcessingOutcome.CANCELLED):
        event = {
            "text": "hello", "user": "U_USER", "channel": "D_CHAT",
            "channel_type": "im", "ts": f"1234567890.00000{1 + len(adapter._processed_message_ts)}",
            "client_msg_id": "turn",
        }
        await adapter._handle_slack_message(event)
        message = adapter.handle_message.call_args.args[0]
        await adapter.on_processing_start(message)
        await adapter.on_processing_complete(message, outcome)
    emoji = [call.kwargs["name"] for call in adapter._app.client.reactions_add.call_args_list]
    assert "white_check_mark" not in emoji
    assert emoji.count("eyes") == emoji.count("hourglass_flowing_sand") == 2


@pytest.mark.asyncio
async def test_no_receipt_for_ambient_unaddressed_channel_message(adapter):
    adapter.config.extra.update({"free_response_channels": ["C_CHAT"], "reply_in_thread": True})
    adapter._channel_gate_allows = AsyncMock(return_value=True)
    await adapter._handle_slack_message({
        "text": "hello everyone", "user": "U_USER", "channel": "C_CHAT",
        "ts": "1234567890.000010", "client_msg_id": "ambient",
    })
    assert adapter.handle_message.await_count == 1
    adapter._app.client.reactions_add.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_processing_reaction_keeps_receipt_until_terminal(adapter):
    client = adapter._app.client
    await adapter._handle_slack_message({
        "text": "hello", "user": "U_USER", "channel": "D_CHAT",
        "channel_type": "im", "ts": "1234567890.000011", "client_msg_id": "work",
    })
    message = adapter.handle_message.call_args.args[0]
    client.reactions_add.side_effect = RuntimeError("simulated Slack rejection")
    await adapter.on_processing_start(message)
    client.reactions_remove.assert_not_awaited()
    client.reactions_add.side_effect = None
    await adapter.on_processing_complete(message, ProcessingOutcome.FAILURE)
    removed = [call.kwargs["name"] for call in client.reactions_remove.call_args_list]
    assert removed == ["hourglass_flowing_sand", "eyes"]


@pytest.mark.asyncio
async def test_thread_followup_receipt_precedes_slow_enrichment(adapter):
    adapter._channel_gate_allows = AsyncMock(return_value=True)
    adapter._has_active_session_for_thread = MagicMock(return_value=True)
    original = adapter._hydrate_thread_context

    async def hydrate(**kwargs):
        adapter._app.client.reactions_add.assert_awaited_once_with(
            channel="C_CHAT", timestamp="1234567890.000002", name="eyes")
        return await original(**kwargs)

    adapter._hydrate_thread_context = hydrate
    await adapter._handle_slack_message({
        "text": "and this too", "user": "U_USER", "channel": "C_CHAT",
        "ts": "1234567890.000002", "thread_ts": "1234567890.000001",
        "client_msg_id": "followup",
    })
    assert adapter.handle_message.await_count == 1
