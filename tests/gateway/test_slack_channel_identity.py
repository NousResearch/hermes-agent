"""Slack sources must carry the channel NAME and topic.

Before: build_source stamped chat_name=channel_id ('Will be resolved later
if needed' — nothing ever did), so the model's Source line read a raw
channel ID (e.g. 'group: C0A094J5RAP') instead of the channel name, and
the bot had no way to tell which channel it was actually posting in."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point document cache to tmp_path so tests don't touch ~/.hermes."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )


def _channel_event(text: str, ts: str, thread_ts: str = None) -> dict:
    """Build a minimal ``message`` event for the Slack Events API
    resembling what ``handle_message_event`` would pass through."""
    event = {
        "channel": "C_CHAN",
        "channel_type": "channel",
        "user": "U_USER",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


def _conv_info(name, topic="", purpose=""):
    return {"ok": True, "channel": {
        "name": name,
        "topic": {"value": topic},
        "purpose": {"value": purpose},
    }}


@pytest.mark.asyncio
async def test_channel_message_carries_name_topic_and_group_type(adapter):
    adapter._app.client.conversations_info = AsyncMock(
        return_value=_conv_info("project-hermes", topic="Project Hermes planning"))
    captured = []
    adapter.handle_message = AsyncMock(side_effect=lambda e: captured.append(e))
    event = _channel_event("<@U_BOT> which project?", ts="1700000000.000001")
    with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="alice")):
        await adapter._handle_slack_message(event)
    source = captured[0].source
    assert source.chat_name == "#project-hermes"
    assert source.chat_topic == "Project Hermes planning"
    # session keys embed chat_type — MUST stay 'group' (session.py:893)
    assert source.chat_type == "group"


@pytest.mark.asyncio
async def test_purpose_falls_back_when_topic_empty(adapter):
    adapter._app.client.conversations_info = AsyncMock(
        return_value=_conv_info("leads", topic="", purpose="Inbound seller leads"))
    captured = []
    adapter.handle_message = AsyncMock(side_effect=lambda e: captured.append(e))
    with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="alice")):
        await adapter._handle_slack_message(
            _channel_event("<@U_BOT> hi", ts="1700000000.000002"))
    assert captured[0].source.chat_topic == "Inbound seller leads"


@pytest.mark.asyncio
async def test_api_failure_falls_back_to_raw_id_and_caches(adapter):
    adapter._app.client.conversations_info = AsyncMock(side_effect=RuntimeError("boom"))
    captured = []
    adapter.handle_message = AsyncMock(side_effect=lambda e: captured.append(e))
    with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="alice")):
        await adapter._handle_slack_message(
            _channel_event("<@U_BOT> a", ts="1700000000.000003"))
        await adapter._handle_slack_message(
            _channel_event("<@U_BOT> b", ts="1700000000.000004"))
    assert captured[0].source.chat_name == "C_CHAN"       # today's behaviour
    assert captured[0].source.chat_topic is None
    # failure is cached: the dead lookup must not retry per message
    assert adapter._app.client.conversations_info.await_count == 1


@pytest.mark.asyncio
async def test_success_is_cached_one_api_call(adapter):
    adapter._app.client.conversations_info = AsyncMock(
        return_value=_conv_info("leads"))
    adapter.handle_message = AsyncMock()
    with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="alice")):
        await adapter._handle_slack_message(_channel_event("<@U_BOT> a", ts="1700000000.000005"))
        await adapter._handle_slack_message(_channel_event("<@U_BOT> b", ts="1700000000.000006"))
    assert adapter._app.client.conversations_info.await_count == 1
