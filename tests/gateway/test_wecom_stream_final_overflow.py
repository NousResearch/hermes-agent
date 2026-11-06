"""A WeCom native-stream final longer than one frame (20,480 UTF-8 bytes) is delivered whole:
the bubble closes with the head and send() delivers the rest, instead of the tail being cut."""

import pytest
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult

CHAT_ID = "chat-1"
TURN_ID = "turn-1"


def _adapter():
    from plugins.platforms.wecom.adapter import WeComAdapter

    adapter = WeComAdapter(PlatformConfig(enabled=True))
    adapter._last_chat_req_ids[CHAT_ID] = "req-1"
    adapter._ws = AsyncMock(closed=False)
    adapter._send_reply_queued = AsyncMock(return_value={"errcode": 0})
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="m-2"))
    return adapter


def _final_frame(adapter) -> dict:
    finals = [c.args[1]["stream"] for c in adapter._send_reply_queued.await_args_list if c.args[1]["stream"]["finish"]]
    assert len(finals) == 1
    return finals[0]


def _long_cjk_reply() -> str:
    # ~36 KB of UTF-8: 12,000 CJK characters in numbered lines, each line ending in a marker.
    return "\n".join(f"第{i}段：企业微信长回复内容测试" + "字" * 10 + f" #{i}" for i in range(500))


class TestWeComStreamFinalOverflow:
    @pytest.mark.asyncio
    async def test_long_final_closes_bubble_with_head_and_sends_the_rest(self):
        adapter = _adapter()
        try:
            text = _long_cjk_reply()
            assert len(text.encode("utf-8")) > adapter.MAX_STREAM_CONTENT_LENGTH
            await adapter.send_stream_frame(text[:200], chat_id=CHAT_ID, turn_id=TURN_ID)

            ok = await adapter.send_stream_frame(text, chat_id=CHAT_ID, finalize=True, turn_id=TURN_ID)

            assert ok is True
            head = _final_frame(adapter)["content"]
            assert len(head.encode("utf-8")) <= adapter.MAX_STREAM_CONTENT_LENGTH
            adapter.send.assert_awaited_once()
            rest = adapter.send.await_args.args[1]
            assert adapter.send.await_args.args[0] == CHAT_ID
            delivered = head + "\n" + rest
            # Every line arrives exactly once, in order, across the bubble and the continuation.
            assert [line for line in delivered.split("\n") if line] == text.split("\n")
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_final_that_fits_is_one_frame_and_no_extra_send(self):
        adapter = _adapter()
        try:
            await adapter.send_stream_frame("hello", chat_id=CHAT_ID, turn_id=TURN_ID)

            ok = await adapter.send_stream_frame("hello world", chat_id=CHAT_ID, finalize=True, turn_id=TURN_ID)

            assert ok is True
            assert _final_frame(adapter)["content"] == "hello world"
            adapter.send.assert_not_awaited()
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_declined_final_sends_nothing_extra(self):
        adapter = _adapter()
        try:
            await adapter.send_stream_frame("partial", chat_id=CHAT_ID, turn_id=TURN_ID)
            adapter._stream_turns[f"{CHAT_ID}:{TURN_ID}"].start_time -= adapter._stream_safe_duration_seconds + 1

            ok = await adapter.send_stream_frame(_long_cjk_reply(), chat_id=CHAT_ID, finalize=True, turn_id=TURN_ID)

            # Layer 2 declined the finalize: the caller's send() fallback owns the whole text.
            assert ok is False
            adapter.send.assert_not_awaited()
        finally:
            await adapter.disconnect()
