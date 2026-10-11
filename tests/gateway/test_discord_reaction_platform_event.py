"""Discord ``reaction`` fire-site for ``gateway_platform_event`` (Telegram parity)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_discord_platform_events import (  # noqa: F401  (shim + helpers)
    _DiscordThread,
    _adapter,
    _capture,
)


@pytest.fixture(autouse=True)
def _observer_available(monkeypatch):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)


def _payload(*, user_id=777, channel_id=555, message_id=456, emoji_name="👍",
             emoji_id=None, guild_id=999, author_id=1, member_bot=False):
    return SimpleNamespace(
        user_id=user_id, channel_id=channel_id, message_id=message_id,
        emoji=SimpleNamespace(name=emoji_name, id=emoji_id), guild_id=guild_id,
        message_author_id=author_id,
        member=SimpleNamespace(bot=member_bot, display_name="alice"),
    )


def _with_client(a, *, bot_id=1, channel=None):
    a._client = SimpleNamespace(user=SimpleNamespace(id=bot_id),
                                get_channel=lambda _cid: channel)
    return a


def test_reaction_normalized_and_fired():
    a = _with_client(_adapter(), channel=SimpleNamespace(id=555))
    seen = _capture(a)
    asyncio.run(a._on_platform_reaction_add(_payload()))
    assert len(seen) == 1
    event, source = seen[0]
    assert event == {
        "platform": "discord", "event_type": "reaction",
        "payload": {"emojis": ["👍"], "custom_emoji_ids": [], "chat_id": "555",
                    "message_id": "456", "thread_id": None, "message_author_id": "1"},
    }
    json.dumps(event)
    assert source.user_id == "777" and source.chat_id == "555"


def test_custom_emoji_goes_to_ids():
    a = _with_client(_adapter())
    seen = _capture(a)
    asyncio.run(a._on_platform_reaction_add(_payload(emoji_name="party", emoji_id=4242)))
    p = seen[0][0]["payload"]
    assert p["emojis"] == [] and p["custom_emoji_ids"] == ["4242"]


def test_uncached_channel_falls_back_to_payload_channel():
    a = _with_client(_adapter(), channel=None)
    seen = _capture(a)
    asyncio.run(a._on_platform_reaction_add(_payload(channel_id=321)))
    assert seen[0][0]["payload"]["chat_id"] == "321"


def test_reaction_in_thread_carries_thread_id():
    t = _DiscordThread()
    t.id = 888
    a = _with_client(_adapter(), channel=t)
    seen = _capture(a)
    asyncio.run(a._on_platform_reaction_add(_payload(channel_id=888)))
    event, source = seen[0]
    assert event["payload"]["thread_id"] == "888" and source.thread_id == "888"


def test_own_bot_reaction_dropped():
    a = _with_client(_adapter(), bot_id=777)
    seen = _capture(a)
    asyncio.run(a._on_platform_reaction_add(_payload(user_id=777)))
    assert seen == []


def test_other_bot_reaction_dropped():
    a = _with_client(_adapter())
    seen = _capture(a)
    asyncio.run(a._on_platform_reaction_add(_payload(member_bot=True)))
    assert seen == []


def test_missing_ids_drop():
    a = _with_client(_adapter())
    seen = _capture(a)
    asyncio.run(a._on_platform_reaction_add(_payload(message_id=None)))
    assert seen == []


def test_no_subscriber_skips(monkeypatch):
    a = _with_client(_adapter())
    handler = AsyncMock()
    a.set_platform_event_handler(handler)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _n: False)
    asyncio.run(a._on_platform_reaction_add(_payload()))
    handler.assert_not_awaited()


def test_no_gateway_callback_fails_closed():
    a = _with_client(_adapter())
    asyncio.run(a._on_platform_reaction_add(_payload()))  # no raise
