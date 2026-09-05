"""Addressing information must survive routing into the agent's event."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import MessageType
from tests.gateway.test_telegram_group_gating import (
    _dm_message, _group_message, _group_voice_message, _make_adapter,
    _mention_entities,
)


@pytest.mark.parametrize("media", [False, True])
@pytest.mark.parametrize("observe", [False, True])
def test_multi_bot_addressing_survives_real_handlers(media, observe):
    async def run():
        text = "@research_bot , @ops_bot are you both listening?"
        for username in ("research_bot", "ops_bot", "unrelated_bot"):
            adapter = _make_adapter(
                bot_username=username, require_mention=True,
                exclusive_bot_mentions=True, observe_unmentioned_group_messages=observe,
                allowed_chats=["-100"], group_allowed_chats=["-100"],
            )
            adapter.config.extra["channel_prompts"] = {"-100": "Keep answers concise."}
            events = []
            adapter._enqueue_text_event = events.append
            adapter.handle_message = AsyncMock(side_effect=events.append)
            adapter._ensure_forum_commands = AsyncMock()
            adapter._cache_inbound_av = AsyncMock(return_value=False)
            entities = _mention_entities(text, ["@research_bot", "@ops_bot"])
            if media:
                msg = _group_voice_message(caption=text)
                msg.caption_entities = entities
                handler = adapter._handle_media_message
            else:
                msg = _group_message(text, entities=entities)
                handler = adapter._handle_text_message
            update = SimpleNamespace(update_id=1001, message=msg, effective_message=None)

            await handler(update, SimpleNamespace())

            if username == "unrelated_bot":
                assert not events
                continue
            assert len(events) == 1
            event = events[0]
            assert text in event.text  # Preserve both recipients and their positions.
            assert f"Your Telegram bot username: @{username}" in event.channel_prompt
            assert "Current message explicitly mentions you: yes" in event.channel_prompt
            assert "Keep answers concise." in event.channel_prompt
            assert ("observed Telegram group context" in event.channel_prompt) == observe
            assert event.source.user_id == (None if observe else "111")

    asyncio.run(run())


@pytest.mark.parametrize("trigger", ["mention", "text_mention", "reply", "wake_word", "open", "code", "command", "dm"])
def test_addressing_context_reports_original_entities_without_inventing_mentions(trigger):
    async def run():
        adapter = _make_adapter(require_mention=trigger != "open", mention_patterns=["^wake\\b"])
        adapter._ensure_forum_commands = AsyncMock()
        events = []
        adapter._enqueue_text_event = events.append
        adapter.handle_message = AsyncMock(side_effect=events.append)
        msg_type = MessageType.TEXT
        if trigger == "dm":
            msg = _dm_message("hello")
        elif trigger == "command":
            msg_type = MessageType.COMMAND
            msg = _group_message("/new@hermes_bot", entities=[SimpleNamespace(type="bot_command", offset=0, length=15)])
        elif trigger == "text_mention":
            msg = _group_message("Hermes hello", entities=[SimpleNamespace(type="text_mention", offset=0, length=6, user=SimpleNamespace(id=999))])
        elif trigger == "mention":
            text = "😀 @hermes_bot hello"
            msg = _group_message(text, entities=[SimpleNamespace(type="mention", offset=3, length=11)])
        elif trigger == "code":
            # Telegram says this is code, not a mention; a reply admits the turn.
            msg = _group_message("@hermes_bot", reply_to_bot=True, entities=[SimpleNamespace(type="code", offset=0, length=11)])
        else:
            msg = _group_message("wake hello" if trigger == "wake_word" else "hello", reply_to_bot=trigger == "reply")
        if msg.reply_to_message:
            for attr in ("photo", "video", "voice", "audio", "document"):
                setattr(msg.reply_to_message, attr, None)
        update = SimpleNamespace(update_id=1002, message=msg, effective_message=None)
        handler = adapter._handle_command if msg_type == MessageType.COMMAND else adapter._handle_text_message
        await handler(update, SimpleNamespace())

        assert len(events) == 1
        event = events[0]
        if trigger == "dm":
            assert not event.channel_prompt
        else:
            expected = "yes" if trigger in {"mention", "text_mention", "command"} else "no"
            assert f"Current message explicitly mentions you: {expected}" in event.channel_prompt
        assert event.text == ("/new" if trigger == "command" else msg.text)
        assert event.source.user_id == "111"

    asyncio.run(run())
