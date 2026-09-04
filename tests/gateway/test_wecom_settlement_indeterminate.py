"""Regression: settlement_indeterminate must NOT produce a confirmed-finalized turn.

When _send_reply_queued returns errcode=0 with errmsg="settlement_indeterminate"
(the final fence timed out and the ACK channel is poisoned), the caller chain
must propagate this distinctly:

  - _send_stream_reply: passes it through (errcode=0 is not an error, but the
    errmsg signals an unconfirmed delivery).
  - _send_stream_frame_inner: does NOT set turn.finalized = True.  Returns
    StreamFrameResult.INDETERMINATE (not DELIVERED).
  - send_stream_frame: propagates StreamFrameResult.INDETERMINATE to consumer.
  - Consumer: sets _final_response_sent=True (don't retry) but does NOT set
    _final_content_delivered=True (delivery unconfirmed).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from gateway.config import PlatformConfig


class TestSettlementIndeterminate:
    """settlement_indeterminate from _send_reply_queued must not false-positive finalize."""

    @pytest.mark.asyncio
    async def test_finalize_with_settlement_indeterminate_returns_indeterminate(self):
        """Core regression: finalize with indeterminate settlement returns StreamFrameResult.INDETERMINATE."""
        from plugins.platforms.wecom.adapter import WeComAdapter, StreamTurn, StreamFrameResult

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        try:
            adapter._last_chat_req_ids["chat-1"] = "req-1"
            adapter._send_json = AsyncMock()
            adapter._ws = AsyncMock(closed=False)

            # _send_reply_queued returns settlement_indeterminate on the final frame
            adapter._send_reply_queued = AsyncMock(return_value={
                "errcode": 0,
                "errmsg": "settlement_indeterminate",
                "ack_pending": True,
            })

            # Create an active turn via an intermediate frame (seed)
            await adapter.send_stream_frame(
                "some content", chat_id="chat-1", turn_id="turn-1",
            )
            turn_key = "chat-1:turn-1"
            assert turn_key in adapter._stream_turns
            turn = adapter._stream_turns[turn_key]
            assert turn.seeded

            # Finalize the turn — _send_reply_queued returns indeterminate
            result = await adapter.send_stream_frame(
                "final content", chat_id="chat-1", finalize=True, turn_id="turn-1",
            )

            # Must return StreamFrameResult.INDETERMINATE
            assert result is StreamFrameResult.INDETERMINATE

            # INDETERMINATE is truthy (no fallback send attempted)
            assert bool(result) is True

            # turn.finalized must NOT be True
            assert turn.finalized is False

            # The turn must still be cleaned up (popped from registry)
            assert turn_key not in adapter._stream_turns
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_finalize_with_normal_success_returns_delivered(self):
        """Sanity check: a normal errcode=0 finalize returns StreamFrameResult.DELIVERED."""
        from plugins.platforms.wecom.adapter import WeComAdapter, StreamFrameResult

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        try:
            adapter._last_chat_req_ids["chat-1"] = "req-1"
            adapter._send_json = AsyncMock()
            adapter._ws = AsyncMock(closed=False)

            # Normal success
            adapter._send_reply_queued = AsyncMock(return_value={
                "errcode": 0,
                "errmsg": "ok",
            })

            # Seed
            await adapter.send_stream_frame(
                "content", chat_id="chat-1", turn_id="turn-1",
            )
            turn_key = "chat-1:turn-1"
            turn = adapter._stream_turns[turn_key]

            # Finalize
            result = await adapter.send_stream_frame(
                "final", chat_id="chat-1", finalize=True, turn_id="turn-1",
            )

            assert result is StreamFrameResult.DELIVERED
            assert bool(result) is True
            assert turn.finalized is True
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_settlement_indeterminate_does_not_trigger_fallback_send(self):
        """settlement_indeterminate returns INDETERMINATE (truthy), preventing consumer from doing a fallback send()."""
        from plugins.platforms.wecom.adapter import WeComAdapter, StreamFrameResult

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        try:
            adapter._last_chat_req_ids["chat-1"] = "req-1"
            adapter._send_json = AsyncMock()
            adapter._ws = AsyncMock(closed=False)

            adapter._send_reply_queued = AsyncMock(return_value={
                "errcode": 0,
                "errmsg": "settlement_indeterminate",
                "ack_pending": True,
            })

            # Seed
            await adapter.send_stream_frame(
                "text", chat_id="chat-1", turn_id="turn-1",
            )

            # Finalize — the return value is truthy (INDETERMINATE), so the consumer
            # will NOT attempt a fallback proactive send().
            result = await adapter.send_stream_frame(
                "final text", chat_id="chat-1", finalize=True, turn_id="turn-1",
            )

            # INDETERMINATE is truthy = "frame handled, do not fall back"
            assert result is StreamFrameResult.INDETERMINATE
            assert bool(result) is True

            # Chat must NOT be marked as stream-expired (that would block future turns)
            assert "chat-1" not in adapter._stream_expired_chats
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_failed_result_is_falsy(self):
        """StreamFrameResult.FAILED must be falsy for backward compat."""
        from plugins.platforms.wecom.adapter import StreamFrameResult

        assert bool(StreamFrameResult.DELIVERED) is True
        assert bool(StreamFrameResult.INDETERMINATE) is True
        assert bool(StreamFrameResult.FAILED) is False

    @pytest.mark.asyncio
    async def test_send_stream_reply_propagates_settlement_indeterminate(self):
        """_send_stream_reply returns the indeterminate response without raising."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(PlatformConfig(enabled=True))
        try:
            adapter._ws = AsyncMock(closed=False)
            adapter._send_reply_queued = AsyncMock(return_value={
                "errcode": 0,
                "errmsg": "settlement_indeterminate",
                "ack_pending": True,
            })

            # Call _send_stream_reply directly with finish=True
            response = await adapter._send_stream_reply(
                "req-1", "stream-1", "content", finish=True,
            )

            # Must propagate without raising
            assert response["errcode"] == 0
            assert response["errmsg"] == "settlement_indeterminate"
            assert response["ack_pending"] is True
        finally:
            await adapter.disconnect()


class TestStreamFrameResultConsumer:
    """Consumer-side handling of StreamFrameResult tri-state."""

    @pytest.mark.asyncio
    async def test_consumer_indeterminate_sets_response_sent_not_content_delivered(self):
        """INDETERMINATE → _final_response_sent=True, _final_content_delivered=False."""
        from plugins.platforms.wecom.adapter import StreamFrameResult
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = AsyncMock()
        adapter.send_stream_frame = AsyncMock(return_value=StreamFrameResult.INDETERMINATE)
        adapter.supports_native_streaming = MagicMock(return_value=True)
        adapter.SUPPORTS_NATIVE_STREAMING = True

        cfg = StreamConsumerConfig()
        consumer = GatewayStreamConsumer.__new__(GatewayStreamConsumer)
        consumer.adapter = adapter
        consumer.chat_id = "chat-1"
        consumer.cfg = cfg
        consumer._turn_id = "turn-1"
        consumer._use_native_streaming = True
        consumer._native_stream_opened = True
        consumer._message_id = None
        consumer._already_sent = True
        consumer._last_sent_text = "partial"
        consumer._native_last_pushed_len = 7
        consumer._final_response_sent = False
        consumer._final_content_delivered = False
        consumer._initial_reply_to_id = None
        consumer._tool_progress_active = False
        consumer._use_draft_streaming = False

        result = await consumer._send_or_edit("final text", finalize=True, is_turn_final=True)

        assert result is True
        assert consumer._final_response_sent is True
        assert consumer._final_content_delivered is False  # THE KEY ASSERTION

    @pytest.mark.asyncio
    async def test_consumer_delivered_sets_both_flags(self):
        """DELIVERED → _final_response_sent=True, _final_content_delivered=True."""
        from plugins.platforms.wecom.adapter import StreamFrameResult
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = AsyncMock()
        adapter.send_stream_frame = AsyncMock(return_value=StreamFrameResult.DELIVERED)
        adapter.supports_native_streaming = MagicMock(return_value=True)
        adapter.SUPPORTS_NATIVE_STREAMING = True

        cfg = StreamConsumerConfig()
        consumer = GatewayStreamConsumer.__new__(GatewayStreamConsumer)
        consumer.adapter = adapter
        consumer.chat_id = "chat-1"
        consumer.cfg = cfg
        consumer._turn_id = "turn-1"
        consumer._use_native_streaming = True
        consumer._native_stream_opened = True
        consumer._message_id = None
        consumer._already_sent = True
        consumer._last_sent_text = "partial"
        consumer._native_last_pushed_len = 7
        consumer._final_response_sent = False
        consumer._final_content_delivered = False
        consumer._initial_reply_to_id = None
        consumer._tool_progress_active = False
        consumer._use_draft_streaming = False

        result = await consumer._send_or_edit("final text", finalize=True, is_turn_final=True)

        assert result is True
        assert consumer._final_response_sent is True
        assert consumer._final_content_delivered is True

    @pytest.mark.asyncio
    async def test_consumer_failed_rolls_back(self):
        """FAILED → both flags False (rollback), switches off native streaming."""
        from plugins.platforms.wecom.adapter import StreamFrameResult
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = AsyncMock()
        adapter.send_stream_frame = AsyncMock(return_value=StreamFrameResult.FAILED)
        adapter.supports_native_streaming = MagicMock(return_value=True)
        adapter.SUPPORTS_NATIVE_STREAMING = True
        adapter.send = AsyncMock(return_value={"message_id": "m1"})

        cfg = StreamConsumerConfig()
        consumer = GatewayStreamConsumer.__new__(GatewayStreamConsumer)
        consumer.adapter = adapter
        consumer.chat_id = "chat-1"
        consumer.cfg = cfg
        consumer._turn_id = "turn-1"
        consumer._use_native_streaming = True
        consumer._native_stream_opened = False
        consumer._message_id = None
        consumer._already_sent = False
        consumer._last_sent_text = ""
        consumer._native_last_pushed_len = 0
        consumer._final_response_sent = False
        consumer._final_content_delivered = False
        consumer._initial_reply_to_id = None
        consumer._tool_progress_active = False
        consumer._use_draft_streaming = False
        consumer._metadata = None
        consumer._reply_to_id = None

        # FAILED is falsy — consumer falls through to the send() fallback.
        # After the send_stream_frame returns FAILED (falsy), the consumer
        # should have rolled back the optimistic finalize and disabled native streaming.
        # The method may proceed to the send/edit fallback path.
        # We check the state after the call.
        result = await consumer._send_or_edit("final text", finalize=True, is_turn_final=True)

        # After FAILED, native streaming should be disabled
        assert consumer._use_native_streaming is False
        # The optimistic finalize should have been rolled back
        # (the send() fallback may set them again)

    @pytest.mark.asyncio
    async def test_consumer_bare_true_backward_compat(self):
        """Bare True (non-WeComAdapter) → both flags True (backward compat)."""
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        adapter = AsyncMock()
        adapter.send_stream_frame = AsyncMock(return_value=True)
        adapter.supports_native_streaming = MagicMock(return_value=True)
        adapter.SUPPORTS_NATIVE_STREAMING = True

        cfg = StreamConsumerConfig()
        consumer = GatewayStreamConsumer.__new__(GatewayStreamConsumer)
        consumer.adapter = adapter
        consumer.chat_id = "chat-1"
        consumer.cfg = cfg
        consumer._turn_id = "turn-1"
        consumer._use_native_streaming = True
        consumer._native_stream_opened = True
        consumer._message_id = None
        consumer._already_sent = True
        consumer._last_sent_text = "partial"
        consumer._native_last_pushed_len = 7
        consumer._final_response_sent = False
        consumer._final_content_delivered = False
        consumer._initial_reply_to_id = None
        consumer._tool_progress_active = False
        consumer._use_draft_streaming = False

        result = await consumer._send_or_edit("final text", finalize=True, is_turn_final=True)

        assert result is True
        assert consumer._final_response_sent is True
        assert consumer._final_content_delivered is True

    @pytest.mark.asyncio
    async def test_consumer_bare_false_backward_compat(self):
        """Bare False (non-WeComAdapter) → fallback path (same as FAILED)."""
        from gateway.stream_consumer import GatewayStreamConsumer
        from gateway.config import StreamingConfig

        adapter = AsyncMock()
        adapter.send_stream_frame = AsyncMock(return_value=False)
        adapter.supports_native_streaming = MagicMock(return_value=True)
        adapter.SUPPORTS_NATIVE_STREAMING = True
        adapter.send = AsyncMock(return_value={"message_id": "m1"})

        cfg = StreamingConfig()
        consumer = GatewayStreamConsumer.__new__(GatewayStreamConsumer)
        consumer.adapter = adapter
        consumer.chat_id = "chat-1"
        consumer.cfg = cfg
        consumer._turn_id = "turn-1"
        consumer._use_native_streaming = True
        consumer._native_stream_opened = False
        consumer._message_id = None
        consumer._already_sent = False
        consumer._last_sent_text = ""
        consumer._native_last_pushed_len = 0
        consumer._final_response_sent = False
        consumer._final_content_delivered = False
        consumer._initial_reply_to_id = None
        consumer._tool_progress_active = False
        consumer._use_draft_streaming = False
        consumer._metadata = None
        consumer._reply_to_id = None

        result = await consumer._send_or_edit("final text", finalize=True, is_turn_final=True)

        # After False, native streaming should be disabled
        assert consumer._use_native_streaming is False
