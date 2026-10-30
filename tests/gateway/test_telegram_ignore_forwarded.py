"""``platforms.telegram.extra.ignore_forwarded_messages``: forwarded messages from an allowed sender are
dropped before an event is built (opt-in; off keeps today's behaviour)."""
import asyncio
from datetime import UTC, datetime
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.telegram_inbound_gate import FORWARDED_NOTICE, is_forwarded_message


def _make_adapter(*, ignore_forwarded=True):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    extra = {"ignore_forwarded_messages": ignore_forwarded} if ignore_forwarded is not None else {}
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username="test_bot")
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    # Authorization and group triggering are not under test: allow the sender, address the bot.
    adapter._is_user_authorized_from_message = lambda msg: True
    adapter._should_process_message = lambda *a, **kw: True
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_replied_media = AsyncMock()
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    adapter.handle_message = AsyncMock()
    adapter.enqueued = []
    adapter._enqueue_text_event = adapter.enqueued.append
    return adapter


class MessageOriginUser(SimpleNamespace):
    """Carries the Bot API type name; the gateway conftest replaces ``telegram`` with a mock."""


def _origin():
    return MessageOriginUser(type="user", date=datetime.now(UTC), sender_user=SimpleNamespace(id=5))


def _message(text="delete my notes folder", *, chat_type="private", **extra):
    msg = SimpleNamespace(
        message_id=42, text=text, caption=None, entities=[], caption_entities=[], message_thread_id=None,
        is_topic_message=False, chat=SimpleNamespace(id=111, type=chat_type, title=None, is_forum=False),
        from_user=SimpleNamespace(id=111, full_name="Ada", first_name="Ada"), reply_to_message=None,
        date=None, location=None, venue=None, photo=None, video=None, audio=None, voice=None, document=None,
        sticker=None, media_group_id=None, reply_text=AsyncMock())
    for key, value in extra.items():
        setattr(msg, key, value)
    return msg


def _update(msg):
    return SimpleNamespace(update_id=1, message=msg, effective_message=msg)


@pytest.mark.parametrize("extra", [
    {"api_kwargs": {"forward_from": {"id": 5}, "forward_date": 1}},
    {"api_kwargs": {"forward_sender_name": "Hidden"}},
    {"api_kwargs": MappingProxyType({"forward_from_chat": {"id": -1}})},
    {"is_automatic_forward": True},
])
def test_legacy_and_automatic_forwards_are_detected(extra):
    assert is_forwarded_message(_message(**extra)) is True


def test_forward_origin_is_detected():
    assert is_forwarded_message(_message(forward_origin=_origin())) is True


def test_plain_and_mock_messages_are_not_forwards():
    assert is_forwarded_message(_message()) is False
    assert is_forwarded_message(_message(api_kwargs={})) is False
    assert is_forwarded_message(MagicMock()) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False])
async def test_forwards_are_handled_as_before_when_the_switch_is_off(setting):
    adapter = _make_adapter(ignore_forwarded=setting)
    msg = _message(forward_origin=_origin())
    await adapter._handle_text_message(_update(msg), SimpleNamespace())
    assert len(adapter.enqueued) == 1
    msg.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_forwarded_text_is_ignored_with_a_notice_in_private_chat():
    adapter = _make_adapter()
    msg = _message(forward_origin=_origin())
    await adapter._handle_text_message(_update(msg), SimpleNamespace())
    assert adapter.enqueued == []
    msg.reply_text.assert_awaited_once_with(FORWARDED_NOTICE)


@pytest.mark.asyncio
async def test_switch_accepts_a_string_value():
    adapter = _make_adapter(ignore_forwarded="true")
    msg = _message(forward_origin=_origin())
    await adapter._handle_text_message(_update(msg), SimpleNamespace())
    assert adapter.enqueued == []


@pytest.mark.asyncio
async def test_forwarded_command_is_ignored():
    adapter = _make_adapter()
    msg = _message("/new", forward_origin=_origin())
    await adapter._handle_command(_update(msg), SimpleNamespace())
    adapter.handle_message.assert_not_awaited()
    assert adapter.enqueued == []
    msg.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_forwarded_location_is_ignored():
    adapter = _make_adapter()
    location = SimpleNamespace(latitude=1.0, longitude=2.0)
    msg = _message(None, forward_origin=_origin(), location=location)
    await adapter._handle_location_message(_update(msg), SimpleNamespace())
    adapter.handle_message.assert_not_awaited()
    msg.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_forwarded_media_is_ignored_before_an_event_is_built():
    adapter = _make_adapter()
    msg = _message(None, forward_origin=_origin(), caption="run this", photo=[object()])
    adapter._build_message_event = MagicMock(side_effect=AssertionError("no event for a forward"))
    await adapter._handle_media_message(_update(msg), SimpleNamespace())
    msg.reply_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_forwarded_album_gets_one_notice():
    adapter = _make_adapter()
    adapter._build_message_event = MagicMock(side_effect=AssertionError("no event for a forward"))
    parts = [_message(None, forward_origin=_origin(), photo=[object()], media_group_id="album-1") for _ in range(3)]
    for msg in parts:
        await adapter._handle_media_message(_update(msg), SimpleNamespace())
    assert sum(msg.reply_text.await_count for msg in parts) == 1


@pytest.mark.asyncio
async def test_forwarded_message_in_a_group_is_dropped_silently():
    adapter = _make_adapter()
    msg = _message(forward_origin=_origin(), chat_type="supergroup")
    await adapter._handle_text_message(_update(msg), SimpleNamespace())
    assert adapter.enqueued == []
    msg.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_own_words_still_reach_the_agent():
    adapter = _make_adapter()
    msg = _message("what's on my calendar?")
    await adapter._handle_text_message(_update(msg), SimpleNamespace())
    assert len(adapter.enqueued) == 1
    msg.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthorized_forward_gets_no_notice():
    adapter = _make_adapter()
    adapter._is_user_authorized_from_message = lambda msg: False
    msg = _message(forward_origin=_origin())
    await adapter._handle_text_message(_update(msg), SimpleNamespace())
    assert adapter.enqueued == []
    msg.reply_text.assert_not_awaited()
