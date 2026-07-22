"""Tests for the stream-consumer flood guards.

When the model API fails (rate-limit / 429 / connection drop) the
``GatewayStreamConsumer`` may hold a partial, possibly huge, accumulated
buffer (e.g. an echoed system prompt / skill context).  Historically that
buffer was flushed to the user as a flood of split Telegram messages.

These tests pin the two guards that prevent that:

* ``api_error_fn`` — when the agent's model call failed, the consumer
  suppresses the accumulated buffer and delivers ONE short clean error.
* ``_MAX_FALLBACK_CHUNKS`` — when a response would split into more chunks
  than the cap, the consumer delivers ONE short safe message instead of a
  flood.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_adapter(*, supports_delete: bool = True) -> MagicMock:
    """Minimal MagicMock adapter wired for send/edit/delete."""
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="preview_1",
    ))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="preview_1",
    ))
    if supports_delete:
        adapter.delete_message = AsyncMock(return_value=True)
    else:
        del adapter.delete_message  # type: ignore[attr-defined]
    return adapter


def _sent_texts(adapter) -> list[str]:
    texts = []
    for call in adapter.send.call_args_list:
        texts.append(call.kwargs.get("content", ""))
    if getattr(adapter, "edit_message", None) is not None:
        for call in adapter.edit_message.call_args_list:
            texts.append(call.kwargs.get("content", ""))
    return texts


class TestApiErrorSuppression:
    @pytest.mark.asyncio
    async def test_api_error_suppresses_accumulated_buffer(self):
        """On API failure the consumer sends the clean error, not the buffer.

        The buffer is held until ``got_done`` (large buffer_threshold) to
        mimic the real fallback scenario: the consumer accumulates the whole
        streamed response and would otherwise dump it as a flood of split
        messages at finalization.
        """
        adapter = _make_adapter()
        # A big buffer that looks like an echoed system prompt + skills.
        big_buffer = "SYSTEM PROMPT ... " + "skill content " * 5000
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=10_000_000),
            api_error_fn=lambda: "HTTP 429: Weekly usage limit reached.",
        )
        consumer.on_delta(big_buffer)
        consumer.finish()
        await consumer.run()

        sent = _sent_texts(adapter)
        # Exactly one message, and it is the short clean error — never the
        # raw buffer.
        assert len(sent) == 1
        assert "HTTP 429" in sent[0]
        assert big_buffer[:20] not in sent[0]

        # Delivery flags set so the gateway skips re-delivering the buffer.
        assert consumer.final_response_sent is True
        assert consumer.final_content_delivered is True
        assert consumer.already_sent is True

    @pytest.mark.asyncio
    async def test_no_api_error_keeps_legacy_behaviour(self):
        """Without an api_error_fn the consumer still delivers the buffer."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=10_000_000),
        )
        consumer.on_delta("Hello from the model")
        consumer.finish()
        await consumer.run()

        sent = _sent_texts(adapter)
        assert any("Hello from the model" in t for t in sent)
        # Delivery flags set (legacy behaviour) — no API-error short-circuit.
        assert consumer.final_response_sent is True


class TestFallbackChunkCap:
    @pytest.mark.asyncio
    async def test_oversized_fallback_sends_single_safe_message(self):
        """A buffer that would split past the cap becomes one safe message.

        We force fallback mode (edits unsupported) and feed a buffer far
        larger than ``_MAX_FALLBACK_CHUNKS * MAX_MESSAGE_LENGTH``.
        """
        from gateway.platforms.base import BasePlatformAdapter, SendResult

        # Use a real BasePlatformAdapter subclass so truncate_message behaves
        # (a plain MagicMock's truncate_message returns a non-iterable stub).
        CapAdapter = type("CapAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 4096})
        CapAdapter.__abstractmethods__ = frozenset()
        adapter = CapAdapter.__new__(CapAdapter)
        adapter._typing_paused = set()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        # Edits fail → promotes the consumer into fallback mode.
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=False, message_id=None))
        adapter.REQUIRES_EDIT_FINALIZE = False

        # Huge buffer that will split into far more than the cap of chunks.
        huge = "x" * (GatewayStreamConsumer._MAX_FALLBACK_CHUNKS * 4096 * 3)
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=10_000_000),
        )
        # Enter fallback mode by exhausting flood strikes up front.
        consumer._fallback_final_send = True
        consumer.on_delta(huge)
        consumer.finish()
        await consumer.run()

        sent = _sent_texts(adapter)
        # The flood guard must NOT have delivered the raw buffer as many
        # messages — it sends exactly one safe message instead.
        assert len(sent) == 1
        assert "too large" in sent[0]
        assert huge[:20] not in sent[0]
        assert consumer.final_response_sent is True
        assert consumer.already_sent is True
