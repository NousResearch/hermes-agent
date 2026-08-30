"""Regression: settlement_indeterminate must NOT produce a confirmed-finalized turn.

When _send_reply_queued returns errcode=0 with errmsg="settlement_indeterminate"
(the final fence timed out and the ACK channel is poisoned), the caller chain
must propagate this distinctly:

  - _send_stream_reply: passes it through (errcode=0 is not an error, but the
    errmsg signals an unconfirmed delivery).
  - _send_stream_frame_inner: does NOT set turn.finalized = True.  Instead sets
    turn.settlement = "indeterminate".  Still returns True (no fallback resend).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from gateway.config import PlatformConfig


class TestSettlementIndeterminate:
    """settlement_indeterminate from _send_reply_queued must not false-positive finalize."""

    @pytest.mark.asyncio
    async def test_finalize_with_settlement_indeterminate_does_not_mark_finalized(self):
        """Core regression: turn.finalized must remain False when settlement is indeterminate."""
        from plugins.platforms.wecom.adapter import WeComAdapter, StreamTurn

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

            # Must return True (no fallback send attempted)
            assert result is True

            # turn.finalized must NOT be True
            assert turn.finalized is False

            # turn.settlement must reflect the indeterminate state
            assert turn.settlement == "indeterminate"

            # The turn must still be cleaned up (popped from registry)
            assert turn_key not in adapter._stream_turns
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_finalize_with_normal_success_still_marks_finalized(self):
        """Sanity check: a normal errcode=0 finalize DOES set turn.finalized = True."""
        from plugins.platforms.wecom.adapter import WeComAdapter

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

            assert result is True
            assert turn.finalized is True
            assert turn.settlement is None
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_settlement_indeterminate_does_not_trigger_fallback_send(self):
        """settlement_indeterminate returns True, preventing consumer from doing a fallback send()."""
        from plugins.platforms.wecom.adapter import WeComAdapter

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

            # Finalize — the return value is True (not False), so the consumer
            # will NOT attempt a fallback proactive send().
            result = await adapter.send_stream_frame(
                "final text", chat_id="chat-1", finalize=True, turn_id="turn-1",
            )

            # True = "frame handled, do not fall back"
            assert result is True

            # Chat must NOT be marked as stream-expired (that would block future turns)
            assert "chat-1" not in adapter._stream_expired_chats
        finally:
            await adapter.disconnect()

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
