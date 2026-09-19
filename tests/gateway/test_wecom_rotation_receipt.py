"""A rotation receipt advances body coordinates exactly once."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.wecom.adapter import WeComAdapter
from plugins.platforms.wecom.streaming import ROTATION_CONTINUATION_SUFFIX


@pytest.mark.asyncio
@pytest.mark.parametrize("report_before_final", [False, True])
async def test_final_after_rotation_preserves_repeated_text(report_before_final):
    adapter = WeComAdapter(PlatformConfig(enabled=True))
    adapter._ws = MagicMock(closed=False, close=AsyncMock())
    adapter._last_chat_req_ids["chat"] = "request"
    frames = []

    async def send(payload):
        frames.append(payload["body"]["stream"].copy())
        adapter._resolve_reply_ack("request", {"errcode": 0})

    adapter._send_json = send
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(cursor=""))
    prefix, tail = "重复段落\n", "重复段落\nFinal detail"
    try:
        await consumer._start_transports()
        consumer._append_accumulated(prefix)
        await consumer._send_or_edit(consumer._accumulated)
        turn = next(iter(adapter._stream_turns.values()))
        assert await adapter._rotate_stream(turn, consumer._turn_id)
        consumer._append_accumulated(tail)
        if report_before_final:
            # The body is deferred, but the receipt already advances the caller.
            await consumer._send_or_edit(consumer._accumulated)
            assert consumer._native_split_offset == len(prefix)
        await consumer._send_or_edit(consumer._accumulated, finalize=True)
        finals = [frame["content"] for frame in frames if frame["finish"]]
        assert finals == [prefix + ROTATION_CONTINUATION_SUFFIX, tail]
        assert consumer.delivered_final_matches(prefix + tail) is True
    finally:
        await adapter.disconnect()
