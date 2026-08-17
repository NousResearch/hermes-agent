"""Tests for coalescing rapid bot-authored Discord message chunks."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.discord.adapter import DiscordAdapter


def _message(*, author_id: int, channel_id: int, mentions=()):
    return SimpleNamespace(
        id=author_id * 100 + channel_id,
        author=SimpleNamespace(id=author_id, bot=True),
        channel=SimpleNamespace(id=channel_id),
        mentions=list(mentions),
        content="chunk",
        type=discord.MessageType.default,
    )


def _adapter():
    adapter = object.__new__(DiscordAdapter)
    adapter._bot_tag_debounce_until = {}
    adapter._text_batch_delay_seconds = 0.6
    adapter._text_batch_split_delay_seconds = 2.0
    adapter._self_is_explicitly_mentioned = lambda message: bool(message.mentions)
    return adapter


def test_tag_opens_short_window_for_same_bot_and_channel_only():
    adapter = _adapter()
    tagged = _message(author_id=1, channel_id=10, mentions=[SimpleNamespace(id=99)])
    continuation = _message(author_id=1, channel_id=10)

    adapter._record_bot_tag_debounce(tagged)

    assert adapter._is_bot_tag_debounce_continuation(continuation) is True
    assert adapter._is_bot_tag_debounce_continuation(
        _message(author_id=2, channel_id=10)
    ) is False
    assert adapter._is_bot_tag_debounce_continuation(
        _message(author_id=1, channel_id=11)
    ) is False


def test_disabling_text_batching_disables_bot_tag_debounce():
    adapter = _adapter()
    adapter._text_batch_delay_seconds = 0
    tagged = _message(author_id=1, channel_id=10, mentions=[SimpleNamespace(id=99)])

    adapter._record_bot_tag_debounce(tagged)

    assert adapter._is_bot_tag_debounce_continuation(
        _message(author_id=1, channel_id=10)
    ) is False


def test_uninitialized_debounce_state_is_treated_as_disabled():
    adapter = object.__new__(DiscordAdapter)

    assert adapter._is_bot_tag_debounce_continuation(
        _message(author_id=1, channel_id=10)
    ) is False


def test_mentions_mode_admits_unmentioned_chunk_during_debounce():
    adapter = _adapter()
    own_user = SimpleNamespace(id=99)
    adapter._client = SimpleNamespace(user=own_user)
    adapter._dedup = MessageDeduplicator()
    adapter._get_allow_bots = Mock(return_value="mentions")
    adapter._discord_bots_require_inline_mention = Mock(return_value=True)
    adapter._self_is_raw_mentioned = Mock(return_value=False)
    tagged = _message(author_id=1, channel_id=10, mentions=[own_user])
    continuation = _message(author_id=1, channel_id=10)
    adapter._record_bot_tag_debounce(tagged)

    admitted, _ = adapter._discord_message_admission(continuation, claim=True)

    assert admitted is True


@pytest.mark.asyncio
async def test_rejected_bot_tag_does_not_open_debounce_window():
    adapter = _adapter()
    adapter._ready_event = asyncio.Event()
    adapter._ready_event.set()
    adapter._record_bot_tag_debounce = Mock()
    adapter._discord_message_admission = Mock(return_value=(False, False))
    adapter._handle_message = AsyncMock()
    message = _message(author_id=1, channel_id=10)

    assert await adapter._dispatch_discord_message(message) is False
    adapter._record_bot_tag_debounce.assert_not_called()
    adapter._handle_message.assert_not_awaited()
