"""Regression coverage for Telegram final delivery after streamed edit failure."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = True
    adapter.FALLBACK_ON_FINAL_EDIT_FLOOD = True
    adapter.RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK = True
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.edit_message = AsyncMock()
    adapter.send = AsyncMock()
    adapter.delete_message = AsyncMock(return_value=True)
    return adapter


@pytest.mark.asyncio
async def test_turn_final_flood_immediately_delivers_missing_tail():
    """A short visible preview must not suppress the completed answer."""
    adapter = _adapter()
    adapter.edit_message.return_value = SendResult(
        success=False,
        error="Flood control exceeded. Retry in 180 seconds",
        retry_after=180.0,
    )
    adapter.send.return_value = SendResult(success=True, message_id="tail-1")

    consumer = GatewayStreamConsumer(
        adapter,
        "chat-1",
        StreamConsumerConfig(cursor=" ▉"),
        metadata={"thread_id": "77"},
    )
    consumer._message_id = "preview-1"
    consumer._last_sent_text = ":("
    consumer._already_sent = True

    ok = await consumer._send_or_edit(
        ":( The completed answer follows here.",
        finalize=True,
        is_turn_final=True,
    )

    assert ok is False
    assert consumer._flood_strikes == 1
    assert consumer._fallback_final_send is True
    assert consumer.final_content_delivered is False
    assert adapter.edit_message.await_count == 1

    await consumer._send_fallback_final(":( The completed answer follows here.")

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.kwargs["content"] == "The completed answer follows here."
    assert adapter.send.await_args.kwargs["metadata"] == {
        "thread_id": "77",
        "notify": True,
    }
    adapter.delete_message.assert_not_awaited()
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_non_opt_in_adapter_keeps_adaptive_final_edit_retry():
    """Immediate final fallback remains scoped to opted-in adapters."""
    adapter = _adapter()
    adapter.FALLBACK_ON_FINAL_EDIT_FLOOD = False
    adapter.edit_message.return_value = SendResult(
        success=False,
        error="Flood control exceeded. Retry in 30 seconds",
        retry_after=30.0,
    )

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "partial"
    consumer._already_sent = True

    ok = await consumer._send_or_edit(
        "partial plus final",
        finalize=True,
        is_turn_final=True,
    )

    assert ok is False
    assert consumer._flood_strikes == 1
    assert consumer._fallback_final_send is False


@pytest.mark.asyncio
async def test_turn_final_flood_commits_empty_tail_as_fresh_message():
    """Telegram gets a durable final even when the internal tail is empty."""
    adapter = _adapter()
    adapter.edit_message.return_value = SendResult(
        success=False,
        error="Flood control exceeded. Retry in 30 seconds",
        retry_after=30.0,
    )
    adapter.send.return_value = SendResult(success=True, message_id="final-1")

    consumer = GatewayStreamConsumer(
        adapter,
        "chat-1",
        StreamConsumerConfig(cursor=" ▉"),
    )
    final_text = "The complete answer"
    consumer._message_id = "preview-1"
    consumer._preview_message_ids = {"preview-1"}
    consumer._last_sent_text = f"{final_text} ▉"
    consumer._already_sent = True

    ok = await consumer._send_or_edit(
        final_text,
        finalize=True,
        is_turn_final=True,
    )

    assert ok is False
    assert consumer._fallback_final_send is True
    assert consumer.final_content_delivered is True
    assert adapter.edit_message.await_count == 1

    await consumer._send_fallback_final(final_text)

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.kwargs["content"] == final_text
    assert adapter.send.await_args.kwargs["metadata"] == {"notify": True}
    adapter.delete_message.assert_awaited_once_with("chat-1", "preview-1")
    assert consumer.message_id == "final-1"
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_empty_tail_commit_honors_retry_after(monkeypatch):
    adapter = _adapter()
    adapter.send.side_effect = [
        SendResult(
            success=False,
            error="Flood control exceeded",
            retry_after=3.0,
        ),
        SendResult(success=True, message_id="final-1"),
    ]
    sleep = AsyncMock()
    monkeypatch.setattr("gateway.stream_consumer.asyncio.sleep", sleep)

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._fallback_final_send = True

    await consumer._send_fallback_final("Final answer")

    assert adapter.send.await_count == 2
    sleep.assert_awaited_once_with(3.0)
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_empty_tail_recovery_keeps_prior_segment_messages():
    """Recovery replaces only its current preview, not earlier preambles."""
    adapter = _adapter()
    adapter.send.return_value = SendResult(success=True, message_id="final-1")
    consumer = GatewayStreamConsumer(adapter, "chat-1")

    consumer._track_preview_id("preamble-1")
    consumer._reset_segment_state()
    consumer._track_preview_id("preview-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._fallback_final_send = True

    await consumer._send_fallback_final("Final answer")

    adapter.delete_message.assert_awaited_once_with("chat-1", "preview-1")
    assert "preamble-1" in consumer._preview_message_ids


@pytest.mark.asyncio
async def test_empty_tail_commit_skips_long_flood_retry(monkeypatch):
    adapter = _adapter()
    adapter.send.return_value = SendResult(
        success=False,
        error="flood_control:30.0",
        retry_after=30.0,
    )
    sleep = AsyncMock()
    monkeypatch.setattr("gateway.stream_consumer.asyncio.sleep", sleep)

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._fallback_final_send = True

    await consumer._send_fallback_final("Final answer")

    adapter.send.assert_awaited_once()
    sleep.assert_not_awaited()
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False


@pytest.mark.asyncio
async def test_telegram_long_flood_result_keeps_retry_after():
    """The real adapter contract preserves the server delay for consumers."""
    class FloodError(Exception):
        retry_after = 30.0

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = MagicMock()
    adapter._bot.edit_message_text = AsyncMock(side_effect=FloodError("Retry after 30"))

    result = await adapter.edit_message("123", "456", "Final answer", finalize=False)

    assert result.success is False
    assert result.error == "flood_control:30.0"
    assert result.retry_after == 30.0


@pytest.mark.asyncio
async def test_ambiguous_empty_tail_timeout_preserves_duplicate_suppression():
    adapter = _adapter()
    adapter.send.return_value = SimpleNamespace(
        success=False,
        error="Timed out",
        retryable=False,
    )

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._fallback_final_send = True

    await consumer._send_fallback_final("Final answer")

    adapter.delete_message.assert_not_awaited()
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_confirmed_empty_tail_send_failure_allows_gateway_retry():
    adapter = _adapter()
    adapter.send.return_value = SendResult(
        success=False,
        error="network unavailable",
        retryable=False,
    )

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._fallback_final_send = True
    consumer._final_content_delivered = True

    await consumer._send_fallback_final("Final answer")

    adapter.delete_message.assert_not_awaited()
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False


def test_timeout_exception_is_treated_as_ambiguous_delivery():
    class TimedOut(Exception):
        pass

    assert GatewayStreamConsumer._send_failure_may_have_delivered(
        TimedOut("request timed out")
    ) is True
