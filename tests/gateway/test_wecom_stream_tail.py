"""Regression tests for the WeCom stream-tail fix.

Contract: a turn-final whose content exceeds the stream frame cap (20,480 BYTES)
must not be silently truncated. The stream bubble seals with a frame-sized head
and the remainder is delivered as regular messages via ``send()``. A failed tail
delivery declines the finalize (return False) so the gateway/consumer fallback
re-delivers the whole answer — duplicate head is the accepted trade-off, silent
loss never is.

Toggle-validated: with the fix disabled (content under the cap) the finalize
takes the exact legacy path — one finish=True frame, no ``send()`` call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.wecom.adapter import WeComAdapter

CHAT_ID = "chat-tail"
REQ_ID = "req-tail"
TURN_ID = "turn-tail"

LIMIT = WeComAdapter.MAX_STREAM_CONTENT_LENGTH  # 20480 bytes


def _make_adapter() -> WeComAdapter:
    adapter = WeComAdapter(PlatformConfig(enabled=True))
    adapter._last_chat_req_ids[CHAT_ID] = REQ_ID
    return adapter


def _big_text(min_bytes: int = LIMIT + 4000) -> str:
    """ASCII body with hard newlines: comfortably over the frame cap, line-splittable."""
    line = "x" * 99  # 100 bytes per line incl. \n
    n = min_bytes // 100 + 1
    return "\n".join(f"L{i:05d} {line}" for i in range(n))


class TestSplitStreamFinal:
    def test_short_content_not_split(self):
        head, tail = WeComAdapter._split_stream_final("hello\nworld")
        assert head == "hello\nworld"
        assert tail == ""

    def test_oversized_splits_at_newline_within_cap(self):
        text = _big_text()
        head, tail = WeComAdapter._split_stream_final(text)
        assert head and tail
        assert head + tail == text
        assert len(head.encode("utf-8")) <= LIMIT - 8
        # head ends at a line boundary (no line bisected); the newline opens the tail
        assert tail.startswith("\n")
        # every tail byte was in the original, none lost
        assert text.endswith(tail)

    def test_multibyte_boundary_never_emits_replacement_chars(self):
        text = ("中" * (LIMIT // 3 + 500))  # 3-byte chars, no newlines at all
        head, tail = WeComAdapter._split_stream_final(text)
        assert "\ufffd" not in head
        assert len(head.encode("utf-8")) <= LIMIT
        assert head + tail == text

    def test_zwsp_room_keeps_finalized_frame_under_cap(self):
        # finalize may append ZWSP (3 bytes) before the split; head must still fit.
        text = "a" * (LIMIT - 4)
        head, _tail = WeComAdapter._split_stream_final(text + "\u200b")
        assert len(head.encode("utf-8")) <= LIMIT


class TestFinalizeDeliversTail:
    @pytest.mark.asyncio
    async def test_oversized_final_seals_head_and_sends_tail(self):
        adapter = _make_adapter()
        try:
            reply = AsyncMock(return_value={"errcode": 0})
            adapter._send_stream_reply = reply
            sent = AsyncMock(return_value=SendResult(success=True, message_id="m-2"))
            adapter.send = sent

            await adapter._send_stream_frame_inner("partial", chat=CHAT_ID, finalize=False, turn_id=TURN_ID)
            big = _big_text()
            ok = await adapter._send_stream_frame_inner(big, chat=CHAT_ID, finalize=True, turn_id=TURN_ID)

            assert ok is True
            finals = [c for c in reply.await_args_list if c.kwargs.get("finish") is True]
            assert len(finals) == 1
            frame = finals[0].args[2]
            assert len(frame.encode("utf-8")) <= LIMIT
            assert big.startswith(frame)
            # tail delivered exactly once, byte-continuous with the head
            sent.assert_awaited_once()
            tail = sent.await_args.kwargs["content"]
            assert frame + tail == big
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_tail_failure_declines_finalize_for_full_resend(self):
        adapter = _make_adapter()
        try:
            adapter._send_stream_reply = AsyncMock(return_value={"errcode": 0})
            adapter.send = AsyncMock(return_value=SendResult(success=False, error="boom"))

            await adapter._send_stream_frame_inner("partial", chat=CHAT_ID, finalize=False, turn_id=TURN_ID)
            ok = await adapter._send_stream_frame_inner(_big_text(), chat=CHAT_ID, finalize=True, turn_id=TURN_ID)

            assert ok is False, "tail lost: finalize must decline so the whole answer is re-delivered"
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_under_cap_finalize_never_calls_send(self):
        """Legacy path intact (the duplicate-bubble regression class): a normal-length
        final seals and returns True with ZERO send() calls."""
        adapter = _make_adapter()
        try:
            adapter._send_stream_reply = AsyncMock(return_value={"errcode": 0})
            sent = AsyncMock()
            adapter.send = sent

            await adapter._send_stream_frame_inner("partial answer", chat=CHAT_ID, finalize=False, turn_id=TURN_ID)
            ok = await adapter._send_stream_frame_inner("the complete final answer",
                                                        chat=CHAT_ID, finalize=True, turn_id=TURN_ID)

            assert ok is True
            sent.assert_not_awaited()
        finally:
            await adapter.disconnect()
