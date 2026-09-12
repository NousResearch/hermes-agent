"""Forum auto-topic on @mention in General (Discord auto_thread parity)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


def _adapter(*, auto_topic=True, copy_source=True, bot_username="hermes_bot"):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    extra = {
        "auto_topic_on_mention": auto_topic,
        "auto_topic_copy_source": copy_source,
        "allowed_chats": [],
        "allowed_topics": [],
        "group_allowed_chats": [],
    }
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.name = "telegram"
    adapter.config = PlatformConfig(enabled=True, token="***", extra=extra)
    adapter._bot = SimpleNamespace(
        id=999,
        username=bot_username,
        create_forum_topic=AsyncMock(return_value=SimpleNamespace(message_thread_id=77)),
        copy_message=AsyncMock(),
    )
    adapter._mention_patterns = []
    adapter._bot_username_observed = bot_username.lower()
    return adapter


def _forum_general(*, text="@hermes_bot deploy staging", thread_id=None, is_forum=True, chat_type="supergroup"):
    return SimpleNamespace(
        message_id=42,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=thread_id,
        is_topic_message=thread_id is not None,
        chat=SimpleNamespace(id=-1001, type=chat_type, title="Team", is_forum=is_forum),
        from_user=SimpleNamespace(id=111, full_name="Ada", first_name="Ada", username="ada"),
        reply_to_message=None,
    )


def _event_for(message):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=str(message.chat.id),
        chat_type="group",
        user_id="111",
        thread_id="1" if message.message_thread_id is None else str(message.message_thread_id),
        message_id=str(message.message_id),
    )
    return MessageEvent(text=message.text or "", message_type=MessageType.TEXT, source=source, raw_message=message)


def test_topic_name_strips_bot_mention():
    adapter = _adapter()
    assert adapter._derive_auto_topic_name("@hermes_bot deploy staging", "hermes_bot") == "deploy staging"


def test_topic_name_falls_back_when_only_mention():
    adapter = _adapter()
    assert adapter._derive_auto_topic_name("@hermes_bot", "hermes_bot") == "Hermes"


def test_off_by_default():
    adapter = _adapter(auto_topic=False)
    assert adapter._should_open_forum_auto_topic(_forum_general()) is False


def test_skips_existing_topics():
    adapter = _adapter()
    msg = _forum_general(thread_id=12)
    assert adapter._should_open_forum_auto_topic(msg) is False


def test_skips_non_forum_groups():
    adapter = _adapter()
    msg = _forum_general(is_forum=False, chat_type="group")
    assert adapter._should_open_forum_auto_topic(msg) is False


def test_skips_slash_commands():
    adapter = _adapter()
    msg = _forum_general(text="@hermes_bot /status")
    # leading slash after strip still starts with @ so this is not a command;
    # a bare /status in General must not open a topic.
    cmd = _forum_general(text="/status")
    assert adapter._should_open_forum_auto_topic(cmd) is False
    assert adapter._should_open_forum_auto_topic(msg) is True


def test_opens_topic_copies_source_and_reroutes_session():
    adapter = _adapter()
    msg = _forum_general()
    event = _event_for(msg)

    async def _run():
        return await adapter._maybe_open_forum_auto_topic(msg, event)

    out = asyncio.run(_run())
    adapter._bot.create_forum_topic.assert_awaited_once()
    kwargs = adapter._bot.create_forum_topic.await_args.kwargs
    assert kwargs["name"] == "deploy staging"
    adapter._bot.copy_message.assert_awaited_once()
    copy_kwargs = adapter._bot.copy_message.await_args.kwargs
    assert copy_kwargs["message_thread_id"] == 77
    assert copy_kwargs["message_id"] == 42
    assert out.source.thread_id == "77"
    assert out.source.auto_thread_created is True
    assert out.source.auto_thread_initial_name == "deploy staging"
    assert out.source.prospective_thread_id == "77"


def test_create_failure_stays_in_general():
    adapter = _adapter()
    adapter._bot.create_forum_topic = AsyncMock(side_effect=RuntimeError("not enough rights"))
    msg = _forum_general()
    event = _event_for(msg)

    async def _run():
        return await adapter._maybe_open_forum_auto_topic(msg, event)

    out = asyncio.run(_run())
    assert out.source.thread_id == "1"
    assert out.source.auto_thread_created is False
    adapter._bot.copy_message.assert_not_called()
