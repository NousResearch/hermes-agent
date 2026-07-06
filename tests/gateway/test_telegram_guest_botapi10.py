import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import _should_suppress_final_send_after_stream
from gateway.session import SessionSource, build_session_key
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter(*, allow_from=None, guest_thinking_text=None):
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    extra = {"allow_from": allow_from or []}
    if guest_thinking_text is not None:
        extra["guest_thinking_text"] = guest_thinking_text
    adapter.config = PlatformConfig(enabled=True, token="token", extra=extra)
    adapter._bot = SimpleNamespace(
        answer_guest_query=AsyncMock(return_value=SimpleNamespace(inline_message_id="inline-1")),
        edit_message_text=AsyncMock(),
    )
    adapter._message_handler = AsyncMock()
    return adapter


def _guest_message(*, text="hello", query_id="q1", caller_id=123, caller_name="Guest User"):
    caller = SimpleNamespace(id=caller_id, full_name=caller_name, first_name="Guest", last_name="User")
    return SimpleNamespace(
        text=text,
        caption=None,
        guest_query_id=query_id,
        guest_bot_caller_user=caller,
        from_user=caller,
        chat=SimpleNamespace(id=-100, title="source chat", type="group"),
        message_id=77,
        date=None,
        reply_to_message=None,
    )


def _event_for(msg):
    return MessageEvent(
        text=msg.text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=str(msg.chat.id),
            chat_type="group",
            user_id=str(msg.from_user.id),
            user_name=msg.from_user.full_name,
            message_id=str(msg.message_id),
        ),
        raw_message=msg,
        message_id=str(msg.message_id),
    )


def test_guest_allowed_updates_includes_guest_message():
    assert "guest_message" in TelegramAdapter._telegram_allowed_updates()


@pytest.mark.asyncio
async def test_authorized_guest_message_creates_event_with_metadata(monkeypatch):
    adapter = _adapter(allow_from=["123"], guest_thinking_text="working")
    msg = _guest_message()
    update = SimpleNamespace(update_id=42, guest_message=msg)
    built = _event_for(msg)
    adapter._build_message_event = lambda *args, **kwargs: built
    adapter._clean_bot_trigger_text = lambda text: text
    adapter.handle_message = AsyncMock()

    await adapter._handle_guest_update(update, SimpleNamespace())

    adapter._bot.answer_guest_query.assert_awaited_once()
    kwargs = adapter._bot.answer_guest_query.await_args.kwargs
    assert kwargs["guest_query_id"] == "q1"
    assert kwargs["result"]["input_message_content"]["message_text"] == "working"
    adapter.handle_message.assert_awaited_once_with(built)
    assert built.source.user_id == "123"
    assert built.source.user_name == "Guest User"
    assert built.source.platform_metadata["telegram_guest_query_id"] == "q1"
    assert built.source.platform_metadata["telegram_guest_inline_message_id"] == "inline-1"
    assert built.source.platform_metadata["session_key_suffix"].startswith("telegram_guest_q1_")


@pytest.mark.asyncio
async def test_guest_messages_with_same_query_id_get_distinct_session_suffixes():
    adapter = _adapter(allow_from=["123"])
    suffixes = []

    for text in ("first", "second"):
        msg = _guest_message(text=text, query_id="q1")
        built = _event_for(msg)
        adapter._build_message_event = lambda *args, built=built, **kwargs: built
        adapter._clean_bot_trigger_text = lambda text: text
        adapter.handle_message = AsyncMock()

        await adapter._handle_guest_update(SimpleNamespace(update_id=42, guest_message=msg), SimpleNamespace())
        suffixes.append(built.source.platform_metadata["session_key_suffix"])

    assert suffixes[0] != suffixes[1]


@pytest.mark.asyncio
async def test_guest_update_without_query_id_is_dropped():
    adapter = _adapter(allow_from=["123"])
    msg = _guest_message(query_id=None)
    adapter._build_message_event = lambda *args, **kwargs: _event_for(msg)
    adapter.handle_message = AsyncMock()

    await adapter._handle_guest_update(SimpleNamespace(update_id=1, guest_message=msg), SimpleNamespace())

    adapter._bot.answer_guest_query.assert_not_called()
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_unauthorized_guest_caller_is_dropped():
    adapter = _adapter(allow_from=["999"])
    msg = _guest_message(caller_id=123)
    adapter._build_message_event = lambda *args, **kwargs: _event_for(msg)
    adapter.handle_message = AsyncMock()

    await adapter._handle_guest_update(SimpleNamespace(update_id=1, guest_message=msg), SimpleNamespace())

    adapter._bot.answer_guest_query.assert_not_called()
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_first_guest_send_answers_query_and_mutates_inline_metadata():
    adapter = _adapter(allow_from=["123"])
    metadata = {"telegram_guest_query_id": "q1"}

    result = await adapter.send("-100", "final **answer**", metadata=metadata)

    assert result.success
    assert result.message_id == "inline-1"
    assert metadata["telegram_guest_inline_message_id"] == "inline-1"
    adapter._bot.answer_guest_query.assert_awaited_once()
    adapter._bot.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_subsequent_guest_send_edits_inline_message_not_normal_send():
    adapter = _adapter(allow_from=["123"])
    metadata = {"telegram_guest_query_id": "q1", "telegram_guest_inline_message_id": "inline-1"}

    result = await adapter.send("-100", "updated", metadata=metadata)

    assert result.success
    adapter._bot.answer_guest_query.assert_not_called()
    adapter._bot.edit_message_text.assert_awaited_once()
    assert adapter._bot.edit_message_text.await_args.kwargs["inline_message_id"] == "inline-1"


@pytest.mark.asyncio
async def test_guest_edit_uses_inline_message_id():
    adapter = _adapter(allow_from=["123"])
    metadata = {"telegram_guest_inline_message_id": "inline-1"}

    result = await adapter.edit_message("-100", "ignored-normal-id", "edited", finalize=True, metadata=metadata)

    assert result.success
    adapter._bot.edit_message_text.assert_awaited_once()
    assert adapter._bot.edit_message_text.await_args.kwargs["inline_message_id"] == "inline-1"


def test_guest_session_key_suffix_is_included():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100",
        chat_type="group",
        user_id="123",
        platform_metadata={"session_key_suffix": "telegram_guest:q1:42"},
    )

    assert build_session_key(source).endswith(":123:telegram_guest_q1_42")


def test_guest_final_send_suppression_follows_stream_delivery_state():
    stream_consumer = SimpleNamespace(final_response_sent=True, final_content_delivered=False)

    assert _should_suppress_final_send_after_stream(
        response_text="done",
        response_previewed=False,
        response_transformed=False,
        stream_consumer=stream_consumer,
        metadata={"telegram_guest_query_id": "q1"},
    )
