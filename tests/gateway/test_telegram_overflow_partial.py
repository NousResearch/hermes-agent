"""Regression coverage for partial Telegram overflow delivery."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


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
async def test_stream_consumer_tracks_saturated_preview_and_recovers_after_timeout(
    telegram_adapter,
):
    """A truncated progressive edit must never count as the complete final answer."""
    content = "word " * 120
    telegram_adapter._bot.edit_message_text = AsyncMock(return_value=True)
    real_edit = telegram_adapter.edit_message

    async def edit_then_timeout(*args, **kwargs):
        if kwargs.get("finalize"):
            return SendResult(
                success=False,
                error="TimedOut: Timed out",
                retryable=True,
            )
        return await real_edit(*args, **kwargs)

    telegram_adapter.edit_message = edit_then_timeout
    telegram_adapter.send = AsyncMock(
        return_value=SendResult(success=True, message_id="fresh-final")
    )
    telegram_adapter.delete_message = AsyncMock(return_value=True)

    consumer = GatewayStreamConsumer(
        telegram_adapter,
        "12345",
        config=StreamConsumerConfig(cursor=""),
    )
    consumer._already_sent = True
    consumer._message_id = "201"

    assert await consumer._send_or_edit(content, finalize=False) is True
    visible_preview = telegram_adapter._bot.edit_message_text.await_args.kwargs["text"]
    assert len(visible_preview) < len(content)
    assert consumer._last_sent_text == visible_preview

    assert await consumer._send_or_edit(
        content,
        finalize=True,
        is_turn_final=True,
    ) is False
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False
    assert consumer._fallback_final_send is True

    await consumer._send_fallback_final(content)

    delivered = "".join(
        call.kwargs["content"]
        for call in telegram_adapter.send.await_args_list
    )
    assert delivered == content
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
