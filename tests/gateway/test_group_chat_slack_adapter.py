"""Slack ingress guarantees used by Group Chat messaging controls."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.hosted_room_messaging import messaging_event_id
from tests.gateway.test_slack import _redirect_cache, adapter


@pytest.mark.asyncio
async def test_users_info_bot_classification_reaches_session_source(adapter):
    adapter.config.extra["allow_bots"] = "all"
    adapter._app.client.users_info = AsyncMock(
        return_value={
            "user": {
                "is_bot": True,
                "profile": {"display_name": "Peer Bot"},
                "real_name": "Peer Bot",
            }
        }
    )
    event = {
        "type": "message",
        "user": "U_PEER_BOT",
        "channel": "D_SHARED",
        "channel_type": "im",
        "text": "hello from a peer bot",
        "ts": "1770000000.000001",
    }

    await adapter._handle_slack_message(event)

    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.source.is_bot is True
    assert delivered.source.is_one_to_one is True


@pytest.mark.asyncio
async def test_channel_slash_command_uses_group_session_semantics(adapter):
    await adapter._handle_slash_command({
        "text": "hello",
        "user_id": "U123",
        "channel_id": "C123",
        "team_id": "T123",
    })

    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "group"
    assert event.source.chat_id == "C123"
    assert event.source.user_id == "U123"
    assert event.source.scope_id == "T123"
    assert event.source.is_one_to_one is False


@pytest.mark.asyncio
async def test_message_edit_stamps_untrusted_command_provenance(adapter):
    await adapter._handle_slack_message({
        "text": "whats the rapchat summary for last 12 hours",
        "user": "U_USER",
        "channel": "C123",
        "channel_type": "mpim",
        "team": "T123",
        "ts": "1234567890.000001",
    })
    adapter.handle_message.assert_not_called()

    await adapter._handle_slack_message({
        "subtype": "message_changed",
        "channel": "C123",
        "channel_type": "mpim",
        "team": "T123",
        "ts": "1234567890.000001",
        "message": {
            "text": "<@U_BOT> whats the rapchat summary for last 12 hours",
            "user": "U_USER",
            "channel": "C123",
            "ts": "1234567890.000001",
            "edited": {"user": "U_USER", "ts": "1234567899.000001"},
        },
    })

    event = adapter.handle_message.call_args[0][0]
    assert event.source.is_one_to_one is False
    assert event.source.message_is_edit is True
    assert event.metadata["message_is_edit"] is True


@pytest.mark.asyncio
async def test_room_control_keeps_slack_trigger_for_idempotency(adapter):
    await adapter._handle_slash_command({
        "command": "/hermes",
        "text": "group 1 send hello",
        "trigger_id": "trigger-room-1",
        "user_id": "U1",
        "channel_id": "C1",
        "team_id": "T1",
    })

    event = adapter.handle_message.call_args[0][0]
    assert event.text == "/group 1 send hello"
    assert messaging_event_id(event) == messaging_event_id(event)
