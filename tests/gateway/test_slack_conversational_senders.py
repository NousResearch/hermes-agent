"""Preserve conversational senders through the subtype filter in PR #110780."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture
def adapter():
    instance = SlackAdapter(PlatformConfig(
        enabled=True,
        token="xoxb-test",
        extra={"free_response_channels": ["C_TEST"], "allow_bots": "all"},
    ))
    instance._bot_user_id = "U_BOT"
    instance._running = True
    instance._app = SimpleNamespace(client=SimpleNamespace(
        users_info=AsyncMock(return_value={
            "ok": True, "user": {"is_bot": False, "real_name": "Test User"},
        }),
        conversations_info=AsyncMock(return_value={
            "ok": True, "channel": {"name": "test"},
        }),
        conversations_replies=AsyncMock(return_value={"ok": True, "messages": []}),
    ))
    instance.handle_message = AsyncMock()
    return instance


def _event(subtype, text, *, edited):
    message = {
        "type": "app_mention" if subtype == "document_mention" else "message",
        "subtype": subtype, "user": "U_TEST", "text": text,
        "channel": "C_TEST", "channel_type": "channel", "team": "T_TEST",
        "ts": "100.000001",
    }
    if subtype == "bot_message":
        message["bot_id"] = "B_OTHER"
    else:
        message["document_mention"] = {
            "file_id": "F_CANVAS", "section_id": "section-1",
            "mentioning_user_ids": ["U_TEST"],
        }
    if not edited:
        return message
    return {
        "type": "message", "subtype": "message_changed",
        "channel": "C_TEST", "channel_type": "channel", "team": "T_TEST",
        "ts": "101.000001", "event_ts": "101.000001", "message": message,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("edited", [False, True], ids=["original", "edited"])
@pytest.mark.parametrize(("policy", "mentioned", "accepted"), [
    ("none", False, False),
    ("none", True, False),
    ("mentions", False, False),
    ("mentions", True, True),
    ("all", False, True),
])
async def test_bot_messages_retain_the_configured_policy(adapter, policy, mentioned, accepted, edited):
    adapter.config.extra["allow_bots"] = policy
    text = "<@U_BOT> A message" if mentioned else "A message"

    await adapter._handle_slack_message(_event("bot_message", text, edited=edited))

    assert adapter.handle_message.await_count == int(accepted)
    if accepted:
        delivered = adapter.handle_message.await_args.args[0]
        assert delivered.text == "A message"
        assert delivered.source.is_bot is True


@pytest.mark.asyncio
@pytest.mark.parametrize("edited", [False, True], ids=["original", "edited"])
async def test_canvas_mentions_reach_the_agent(adapter, edited):
    adapter.config.extra = {"allow_bots": "none"}

    await adapter._handle_slack_message(
        _event("document_mention", "<@U_BOT> Summarize this canvas", edited=edited)
    )

    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.text == "Summarize this canvas"
    assert delivered.source.user_id == "U_TEST"
    assert delivered.raw_message["document_mention"]["file_id"] == "F_CANVAS"
