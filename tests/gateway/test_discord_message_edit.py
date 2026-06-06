"""Tests for Discord MESSAGE_UPDATE/edit-trigger handling."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import discord

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


class _FakeUser:
    def __init__(self, user_id=42, *, bot=True, name="clawbot"):
        self.id = user_id
        self.bot = bot
        self.name = name
        self.display_name = name

    def __eq__(self, other):
        return getattr(other, "id", None) == self.id

    def __hash__(self):
        return hash(self.id)


def _adapter() -> DiscordAdapter:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._client = SimpleNamespace(user=_FakeUser(42))
    adapter._text_batch_delay_seconds = 0
    adapter._handle_message = AsyncMock()
    return adapter


def _channel(channel_id=100):
    return SimpleNamespace(id=channel_id, name="trading", guild=SimpleNamespace(name="Guild"))


def _author():
    return _FakeUser(7, bot=False, name="Alan")


def _attachment(att_id=1, filename="chart.png", content_type="image/png", size=123):
    return SimpleNamespace(
        id=att_id,
        filename=filename,
        content_type=content_type,
        size=size,
        url=f"https://cdn.discordapp.com/attachments/100/{att_id}/{filename}",
    )


def _message(message_id=555, content="hello", *, mentions=None, attachments=None, channel=None):
    mentions = [] if mentions is None else mentions
    return SimpleNamespace(
        id=message_id,
        content=content,
        mentions=mentions,
        attachments=[] if attachments is None else attachments,
        author=_author(),
        channel=channel or _channel(),
        guild=SimpleNamespace(id=9, name="Guild"),
        type=discord.MessageType.default,
        created_at=None,
        reference=None,
        message_snapshots=[],
    )


@pytest.mark.asyncio
async def test_message_without_mention_then_edit_adds_mention_enqueues_once():
    adapter = _adapter()
    bot = adapter._client.user
    before = _message(content="这是马后炮吗？")
    adapter._record_discord_edit_state(before, processed=False)
    after = _message(content="<@42> 这是马后炮吗？", mentions=[bot])

    await adapter._handle_message_edit(before, after)
    await adapter._handle_message_edit(before, after)

    assert adapter._handle_message.await_count == 1
    assert "edited this Discord message" in adapter._handle_message.await_args.args[0].content


@pytest.mark.asyncio
async def test_edit_mentioned_message_text_only_does_not_enqueue_again():
    adapter = _adapter()
    bot = adapter._client.user
    original = _message(content="<@42> look", mentions=[bot])
    adapter._record_discord_edit_state(original, processed=True, trigger_reason="message_create")
    after = _message(content="<@42> look, typo fixed", mentions=[bot])

    await adapter._handle_message_edit(original, after)

    adapter._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_add_attachment_while_mentioned_enqueues_once_per_attachment_state():
    adapter = _adapter()
    bot = adapter._client.user
    original = _message(content="<@42> look", mentions=[bot], attachments=[])
    adapter._record_discord_edit_state(original, processed=True, trigger_reason="message_create")
    after = _message(content="<@42> look", mentions=[bot], attachments=[_attachment(1)])

    await adapter._handle_message_edit(original, after)
    await adapter._handle_message_edit(original, after)

    assert adapter._handle_message.await_count == 1


@pytest.mark.asyncio
async def test_explicit_redo_reprocesses_mentioned_message():
    adapter = _adapter()
    bot = adapter._client.user
    original = _message(content="<@42> old", mentions=[bot])
    adapter._record_discord_edit_state(original, processed=True, trigger_reason="message_create")
    after = _message(content="<@42> 请重新处理", mentions=[bot])

    await adapter._handle_message_edit(original, after)

    assert adapter._handle_message.await_count == 1


@pytest.mark.asyncio
async def test_raw_update_fetches_full_message_before_processing():
    adapter = _adapter()
    bot = adapter._client.user
    before = _message(content="chart")
    adapter._record_discord_edit_state(before, processed=False)
    fetched = _message(content="<@42> chart", mentions=[bot], attachments=[_attachment(2)])
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=fetched))
    adapter._client = SimpleNamespace(
        user=bot,
        get_channel=lambda channel_id: channel,
        fetch_channel=AsyncMock(),
    )
    payload = SimpleNamespace(channel_id=100, message_id=555, cached_message=before)

    after = await adapter._fetch_message_for_raw_update(payload)
    await adapter._handle_message_edit(payload.cached_message, after, raw=True)

    channel.fetch_message.assert_awaited_once_with(555)
    assert adapter._handle_message.await_count == 1


@pytest.mark.asyncio
async def test_raw_update_missing_fetch_context_skips_gracefully(caplog):
    adapter = _adapter()
    payload = SimpleNamespace(channel_id=None, message_id=None, cached_message=None)

    with caplog.at_level("INFO"):
        result = await adapter._fetch_message_for_raw_update(payload)

    assert result is None
    assert "raw update missing channel_id/message_id" in caplog.text


def test_message_create_with_mention_still_seen_as_bot_mention():
    adapter = _adapter()
    bot = adapter._client.user
    message = _message(content="<@42> hello", mentions=[bot])

    assert adapter._message_has_bot_mention(message) is True
    state = adapter._record_discord_edit_state(message, processed=True, trigger_reason="message_create")
    assert state.last_seen_has_bot_mention is True
