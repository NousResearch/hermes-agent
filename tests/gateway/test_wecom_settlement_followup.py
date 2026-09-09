"""Settlement must remain honest across ACK ambiguity and the gateway join."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.run_turn import GatewayTurnMixin
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.wecom.adapter import WeComAdapter


def make_adapter():
    adapter = WeComAdapter(PlatformConfig(enabled=True))
    adapter._ws = MagicMock(closed=False, close=AsyncMock())
    adapter._last_chat_req_ids["chat"] = "request"
    adapter.send = AsyncMock()
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["drain", "rotation", "retry"])
async def test_timeout_cannot_certify_successor_from_late_ack(mode):
    adapter = make_adapter()
    adapter._REPLY_ACK_TIMEOUT = 0.05
    frames = []

    async def send(payload):
        frame = payload["body"]["stream"].copy()
        frames.append(frame)
        if mode == "retry":
            if not frame["finish"] or len([f for f in frames if f["finish"]]) >= 2:
                # Either identical final attempt's ACK proves that content, but not a later stream.
                adapter._resolve_reply_ack("request", {"errcode": 0})
        elif frame["finish"]:
            # The only receipt available belongs to the previous seed/body, not this final.
            adapter._resolve_reply_ack("request", {"errcode": 0})

    adapter._send_json = send
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(cursor=""))
    try:
        await consumer._start_transports()
        consumer._append_accumulated("Answer")
        if mode == "rotation":
            turn = next(iter(adapter._stream_turns.values()))
            turn.accumulated_text = "Answer"
            old_id = turn.stream_id
            await adapter._rotate_stream(turn, consumer._turn_id)
            assert turn.stream_id == old_id  # no confirmed seal, no prefix may be retired
            assert turn.pending_rotation_signal is False
        consumer.finish("Answer")
        await consumer.run()
        assert consumer.final_response_sent is True
        assert consumer.final_content_delivered is (mode == "retry")
        adapter.send.assert_not_awaited()
        # Nor may a later stream under the same request borrow that stale ACK authority.
        result = await adapter._send_stream_reply("request", "successor", "Next", finish=True)
        assert result["errmsg"] == "settlement_indeterminate"
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_gateway_join_cancellation_never_confirms_pending_final():
    adapter = make_adapter()
    adapter._REPLY_ACK_TIMEOUT = 0.05
    final_entered, release_write = asyncio.Event(), asyncio.Event()
    frames = []

    async def send(payload):
        frame = payload["body"]["stream"].copy()
        frames.append(frame)
        if frame["finish"]:
            final_entered.set()
            # Keep the real control worker in flight across the real five-second gateway join.
            await release_write.wait()
        else:
            adapter._resolve_reply_ack("request", {"errcode": 0})

    adapter._send_json = send
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(cursor=""))
    consumer.on_delta("Answer")
    consumer.finish("Answer")
    task = asyncio.create_task(consumer.run())
    try:
        await asyncio.wait_for(final_entered.wait(), 5)
        await GatewayTurnMixin._await_stream_task(task)
        assert task.done()
        assert consumer.final_response_sent is True
        assert consumer.final_content_delivered is False
        assert consumer.delivered_final_matches("Answer") is True  # no duplicate retry
        release_write.set()
        await asyncio.wait_for(adapter._control_queues["chat"].join(), 5)
        assert consumer.final_content_delivered is False  # eventual double-timeout is not success
        assert len([f for f in frames if f["finish"]]) == 2
        adapter.send.assert_not_awaited()
    finally:
        release_write.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await adapter.disconnect()
