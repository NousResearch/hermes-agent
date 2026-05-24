"""Tests for Telegram Inbox auto-topic routing."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource
from gateway.platforms.telegram import TelegramAdapter


class FakeBot:
    username = "hermesbot"

    def __init__(self, *, thread_id=777, fail_first=None, copied_message_id=888):
        self.calls = []
        self.copy_calls = []
        self.thread_id = thread_id
        self.fail_first = fail_first
        self.copied_message_id = copied_message_id

    async def create_forum_topic(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_first and len(self.calls) == 1:
            raise self.fail_first
        return SimpleNamespace(message_thread_id=self.thread_id)

    async def copy_message(self, **kwargs):
        self.copy_calls.append(kwargs)
        return SimpleNamespace(message_id=self.copied_message_id)


def _make_adapter(extra):
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="test-token", extra={"inbox_auto_topic": extra})
    )
    bot = FakeBot()
    adapter._bot = bot
    return adapter, bot


def _make_message(*, chat_id=-100123, thread_id=None, text="**Fix prod login?**", is_forum=True):
    return SimpleNamespace(
        chat=SimpleNamespace(
            id=chat_id,
            is_forum=is_forum,
            type="supergroup",
            title="Hermes",
            full_name=None,
        ),
        from_user=SimpleNamespace(id=1234, full_name="Alexander"),
        message_thread_id=thread_id,
        is_topic_message=thread_id is not None,
        message_id=42,
        text=text,
        caption=None,
        reply_to_message=None,
        date=None,
    )


def _make_event(message):
    thread_id = str(message.message_thread_id) if message.message_thread_id is not None else "1"
    return MessageEvent(
        text=message.text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=str(message.chat.id),
            chat_type="group",
            thread_id=thread_id,
        ),
        raw_message=message,
        reply_to_message_id=str(message.message_id),
    )


@pytest.mark.asyncio
async def test_inbox_auto_topic_routes_general_forum_message_to_new_topic():
    adapter, bot = _make_adapter(
        {
            "enabled": True,
            "chat_id": -100123,
            "source_thread_id": "1",
            "topic_prefix": "cw",
        }
    )
    message = _make_message()
    event = _make_event(message)

    routed = await adapter._route_inbox_message_to_new_topic(event)

    assert routed.source.thread_id == "777"
    assert routed.source.chat_topic == "cw Fix prod login"
    assert routed.reply_to_message_id == "888"
    assert bot.calls == [{"chat_id": -100123, "name": "cw Fix prod login"}]
    assert bot.copy_calls == [
        {
            "chat_id": -100123,
            "from_chat_id": -100123,
            "message_id": 42,
            "message_thread_id": 777,
        }
    ]


@pytest.mark.asyncio
async def test_inbox_auto_topic_ignores_non_source_topic():
    adapter, bot = _make_adapter(
        {"enabled": True, "chat_id": -100123, "source_thread_id": "1"}
    )
    message = _make_message(thread_id=99)
    event = _make_event(message)

    routed = await adapter._route_inbox_message_to_new_topic(event)

    assert routed is event
    assert routed.source.thread_id == "99"
    assert bot.calls == []


@pytest.mark.asyncio
async def test_inbox_auto_topic_retries_duplicate_topic_name_with_message_id_suffix():
    adapter, _bot = _make_adapter(
        {"enabled": True, "chat_id": -100123, "source_thread_id": "1"}
    )
    duplicate_retry_bot = FakeBot(fail_first=Exception("Bad Request: TOPIC_NAME_DUPLICATE"))
    adapter._bot = duplicate_retry_bot
    message = _make_message(text="Repeated request")
    event = _make_event(message)

    routed = await adapter._route_inbox_message_to_new_topic(event)

    assert routed.source.thread_id == "777"
    assert duplicate_retry_bot.calls == [
        {"chat_id": -100123, "name": "Repeated request"},
        {"chat_id": -100123, "name": "Repeated request #42"},
    ]


class FakeTelegramFile:
    file_path = "photo.jpg"

    async def download_as_bytearray(self):
        return bytearray(b"\xff\xd8\xff\xe0fake-jpeg-bytes")


class FakePhoto:
    async def get_file(self):
        return FakeTelegramFile()


@pytest.mark.asyncio
async def test_inbox_auto_topic_routes_photo_messages_to_new_topic_and_copies_opener(monkeypatch):
    adapter, bot = _make_adapter(
        {
            "enabled": True,
            "chat_id": -100123,
            "source_thread_id": "1",
            "topic_prefix": "cw",
        }
    )
    message = _make_message(text="")
    message.caption = "Fix this screenshot"
    message.photo = [FakePhoto()]
    message.sticker = None
    message.video = None
    message.audio = None
    message.voice = None
    message.document = None
    update = SimpleNamespace(message=message, update_id=99)
    monkeypatch.setattr(adapter, "_should_process_message", lambda _message: True)

    await adapter._handle_media_message(update, SimpleNamespace())

    assert bot.calls == [{"chat_id": -100123, "name": "cw Fix this screenshot"}]
    assert bot.copy_calls == [
        {
            "chat_id": -100123,
            "from_chat_id": -100123,
            "message_id": 42,
            "message_thread_id": 777,
        }
    ]
    assert list(adapter._pending_photo_batches) == [
        "agent:main:telegram:group:-100123:777:photo-burst"
    ]
    pending = next(iter(adapter._pending_photo_batches.values()))
    assert pending.source.thread_id == "777"
    assert pending.source.chat_topic == "cw Fix this screenshot"
    assert pending.reply_to_message_id == "888"

    for task in adapter._pending_photo_batch_tasks.values():
        task.cancel()
