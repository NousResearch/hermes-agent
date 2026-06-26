"""Regression coverage for partial Telegram overflow delivery."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.stream_consumer import GatewayStreamConsumer


def _message(message_id: int | str) -> SimpleNamespace:
    return SimpleNamespace(message_id=message_id)


@pytest.fixture
def telegram_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    object.__setattr__(adapter, "MAX_MESSAGE_LENGTH", 160)
    return adapter


@pytest.mark.asyncio
async def test_edit_overflow_split_reports_success_when_all_continuations_land(telegram_adapter):
    """Complete overflow delivery keeps the existing successful contract."""
    content = "word " * 120
    telegram_adapter._bot.edit_message_text = AsyncMock(return_value=True)
    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=[_message(202), _message(203), _message(204), _message(205)]
    )

    result = await telegram_adapter._edit_overflow_split(
        "12345", "201", content, finalize=False, metadata={"thread_id": "77"}
    )

    assert result.success is True
    assert result.message_id == result.continuation_message_ids[-1]
    assert result.raw_response is None
    assert telegram_adapter._bot.edit_message_text.await_count == 1
    assert telegram_adapter._bot.send_message.await_count == len(result.continuation_message_ids)
    for call in telegram_adapter._bot.send_message.await_args_list:
        assert call.kwargs["message_thread_id"] == 77


@pytest.mark.asyncio
async def test_edit_overflow_split_reports_later_partial_failure_after_some_continuations_land(telegram_adapter):
    """Partial metadata tracks the last delivered continuation before failure."""
    content = "word " * 120
    telegram_adapter._bot.edit_message_text = AsyncMock(return_value=True)
    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=[
            _message(202),
            RuntimeError("telegram send failed"),
            RuntimeError("telegram send failed"),
        ]
    )

    result = await telegram_adapter._edit_overflow_split(
        "12345", "201", content, finalize=False, metadata={"thread_id": "77"}
    )

    assert result.success is False
    assert result.message_id == "202"
    assert result.raw_response["partial_overflow"] is True
    assert result.raw_response["delivered_chunks"] == 2
    assert result.raw_response["last_message_id"] == "202"
    assert result.continuation_message_ids == ("202",)


@pytest.mark.asyncio
async def test_edit_overflow_split_reports_partial_failure_when_continuation_fails(telegram_adapter):
    """A failed continuation must not be reported as final delivery."""
    content = "word " * 120
    telegram_adapter._bot.edit_message_text = AsyncMock(return_value=True)
    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=[RuntimeError("telegram send failed"), RuntimeError("telegram send failed")]
    )

    result = await telegram_adapter._edit_overflow_split(
        "12345", "201", content, finalize=False, metadata={"thread_id": "77"}
    )

    assert result.success is False
    assert result.retryable is True
    assert result.error == "overflow_continuation_failed"
    assert result.message_id == "201"
    assert result.raw_response["partial_overflow"] is True
    assert result.raw_response["delivered_chunks"] == 1
    assert result.raw_response["total_chunks"] > 1
    assert result.raw_response["last_message_id"] == "201"
    assert result.raw_response["delivered_prefix"]
    assert result.continuation_message_ids == ()


@pytest.mark.asyncio
async def test_stream_consumer_fallback_sends_tail_after_partial_overflow():
    """A partial overflow edit enters fallback instead of marking final delivered."""
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.edit_message = AsyncMock(
        return_value=SendResult(
            success=False,
            message_id="preview-1",
            error="overflow_continuation_failed",
            retryable=True,
            raw_response={
                "partial_overflow": True,
                "delivered_chunks": 1,
                "total_chunks": 2,
                "last_message_id": "preview-1",
                "delivered_prefix": "hello ",
            },
        )
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="tail-1"))
    adapter.delete_message = AsyncMock(return_value=True)

    consumer = GatewayStreamConsumer(adapter, "chat-1", metadata={"thread_id": "77"})
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "hello "

    ok = await consumer._send_or_edit("hello world", finalize=True)

    assert ok is False
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False
    assert consumer._fallback_final_send is True
    assert consumer._fallback_prefix == "hello "

    await consumer._send_fallback_final("hello world")

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.kwargs["content"] == "world"
    assert adapter.send.await_args.kwargs["metadata"] == {"thread_id": "77", "notify": True}
    adapter.delete_message.assert_not_awaited()
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_send_long_text_delivers_txt_document_instead_of_chunks(telegram_adapter):
    """Long final sends use one document upload, avoiding Telegram flood waits."""
    telegram_adapter.config.extra["long_text_document_threshold"] = 200
    content = "word " * 80
    captured = {}

    async def fake_send_document(chat_id, file_path, caption=None, file_name=None, reply_to=None, metadata=None, **_kwargs):
        with open(file_path, "r", encoding="utf-8") as handle:
            captured["body"] = handle.read()
        captured.update(
            chat_id=chat_id,
            caption=caption,
            file_name=file_name,
            reply_to=reply_to,
            metadata=metadata,
        )
        return SendResult(success=True, message_id="doc-1")

    telegram_adapter.send_document = AsyncMock(side_effect=fake_send_document)
    telegram_adapter._bot.send_message = AsyncMock(return_value=_message(202))

    result = await telegram_adapter.send(
        "12345",
        content,
        reply_to="99",
        metadata={"thread_id": "77", "notify": True},
    )

    assert result.success is True
    assert result.message_id == "doc-1"
    telegram_adapter._bot.send_message.assert_not_awaited()
    telegram_adapter.send_document.assert_awaited_once()
    assert captured["body"] == content
    assert captured["file_name"].startswith("hermes-response-")
    assert captured["file_name"].endswith(".txt")
    assert captured["reply_to"] == "99"
    assert captured["metadata"] == {"thread_id": "77", "notify": True}


@pytest.mark.asyncio
async def test_edit_finalize_long_text_replaces_preview_and_sends_txt_document(telegram_adapter):
    """Finalizing a long streamed reply avoids continuation chunk storms."""
    telegram_adapter.config.extra["long_text_document_threshold"] = 200
    content = "word " * 80
    captured = {}

    async def fake_send_document(chat_id, file_path, caption=None, file_name=None, reply_to=None, metadata=None, **_kwargs):
        with open(file_path, "r", encoding="utf-8") as handle:
            captured["body"] = handle.read()
        captured.update(caption=caption, reply_to=reply_to, metadata=metadata)
        return SendResult(success=True, message_id="doc-2")

    telegram_adapter._bot.edit_message_text = AsyncMock(return_value=True)
    telegram_adapter._bot.send_message = AsyncMock(return_value=_message(202))
    telegram_adapter.send_document = AsyncMock(side_effect=fake_send_document)

    result = await telegram_adapter.edit_message(
        "12345",
        "201",
        content,
        finalize=True,
        metadata={"thread_id": "77", "notify": True},
    )

    assert result.success is True
    assert result.message_id == "doc-2"
    telegram_adapter._bot.edit_message_text.assert_awaited_once()
    telegram_adapter._bot.send_message.assert_not_awaited()
    assert captured["body"] == content
    assert captured["reply_to"] == "201"
    assert captured["metadata"] == {"thread_id": "77", "notify": True}


@pytest.mark.asyncio
async def test_long_text_document_can_be_disabled(telegram_adapter):
    """Operators can opt out and keep the legacy chunked text behavior."""
    telegram_adapter.config.extra["long_text_as_document"] = False
    telegram_adapter.config.extra["long_text_document_threshold"] = 200
    content = "word " * 80
    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=[_message(202), _message(203), _message(204)]
    )
    telegram_adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    result = await telegram_adapter.send("12345", content, metadata={"notify": True})

    assert result.success is True
    telegram_adapter.send_document.assert_not_awaited()
    assert telegram_adapter._bot.send_message.await_count > 1
