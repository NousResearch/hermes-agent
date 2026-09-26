"""Regression: a dying turn closes its native stream (PR 85796 review, B8).

Stale-generation exits and cancellations returned from run() with the
native stream still open: the Slack message kept its live streaming
indicator forever, and the adapter's armed interception state survived
into the next turn (which could inherit the key and seal a dead
draft_id). The consumer now calls adapter.abandon_open_draft on both
death paths — sealing in place with the last delivered frame, claiming
nothing (no delivery flags).
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.relay.test_relay_live_cards import _connected_adapter


class RecordingTransport:
    def __init__(self):
        self.ops = []

    async def send_outbound(self, payload, platform=None):
        self.ops.append(dict(payload))
        return {"success": True, "message_id": "ts.1"}


def _consumer(run_still_current=None):
    adapter, _ = _connected_adapter()
    t = RecordingTransport()
    adapter._transport = t
    cfg = StreamConsumerConfig(
        transport="auto", chat_type="dm",
        edit_interval=0.01, buffer_threshold=1, cursor="",
    )
    sc = GatewayStreamConsumer(
        adapter, "C1", cfg,
        metadata={"thread_ts": "1700.8", "message_id": "1700.801"},
        run_still_current=run_still_current,
    )
    return sc, adapter, t


class TestAbandonOnTurnDeath:
    @pytest.mark.asyncio
    async def test_stale_generation_seals_open_stream(self):
        """/new mid-stream: run_still_current flips False → the open
        stream is sealed with the on-screen text, and no delivery is
        claimed."""
        _alive = [True]
        sc, adapter, t = _consumer(run_still_current=lambda: _alive[0])
        task = asyncio.create_task(sc.run())
        sc.on_delta("partial answer on screen")
        await asyncio.sleep(0.08)
        assert adapter._open_draft_by_chat, "stream should be armed"
        _alive[0] = False
        await task
        assert not adapter._open_draft_by_chat, "stream left armed after stale exit"
        seals = [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert len(seals) == 1
        assert seals[0]["content"] == "partial answer on screen"
        assert sc.final_response_sent is False

    @pytest.mark.asyncio
    async def test_cancellation_seals_open_stream(self):
        sc, adapter, t = _consumer()
        task = asyncio.create_task(sc.run())
        sc.on_delta("partial before stop")
        await asyncio.sleep(0.08)
        assert adapter._open_draft_by_chat
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert not adapter._open_draft_by_chat, "stream left armed after cancel"
        seals = [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert len(seals) == 1
        assert sc.final_response_sent is False

    @pytest.mark.asyncio
    async def test_abandon_without_open_stream_is_noop(self):
        sc, adapter, t = _consumer()
        await sc._abandon_native_stream()
        assert t.ops == []

    @pytest.mark.asyncio
    async def test_next_turn_not_intercepted_after_abandon(self):
        """The B8 inheritance hazard: after an abandoned turn, a NEW turn
        on the same key must arm and seal normally with its own content."""
        _alive = [True]
        sc, adapter, t = _consumer(run_still_current=lambda: _alive[0])
        task = asyncio.create_task(sc.run())
        sc.on_delta("old turn partial")
        await asyncio.sleep(0.08)
        _alive[0] = False
        await task
        # Next turn, same chat/keys, new draft_id:
        md = {"thread_ts": "1700.8", "message_id": "1700.801"}
        await adapter.send_draft("C1", 999, "new turn", metadata=md)
        r = await adapter.send("C1", "new turn final", metadata=dict(md))
        assert r.success
        seals = [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert seals[-1]["draft_id"] == 999
        assert seals[-1]["content"] == "new turn final"


class _EditTransportAdapter(BasePlatformAdapter):
    """Edit-capable adapter recording what each message finally shows."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.DISCORD)
        self.screen = {}

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        mid = f"m{len(self.screen) + 1}"
        self.screen[mid] = content
        return SendResult(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.screen[message_id] = content
        return SendResult(success=True, message_id=message_id)

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
@pytest.mark.parametrize(("shown", "sealed"), [
    ("partial answer", "partial answer"),
    ("```py\nx = 1", "```py\nx = 1\n```"),
])
async def test_stale_exit_seals_edit_transport_preview(shown, sealed):
    """/stop or /new mid-stream on the edit transport: the preview keeps what the user
    already saw, loses its live cursor, and nothing later is delivered or claimed."""
    alive = [True]
    adapter = _EditTransportAdapter()
    sc = GatewayStreamConsumer(
        adapter, "C1",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=" ▉"),
        run_still_current=lambda: alive[0],
    )
    task = asyncio.create_task(sc.run())
    sc.on_delta(shown)
    await asyncio.sleep(0.1)
    assert " ▉" in adapter.screen["m1"]
    alive[0] = False
    sc.on_delta(" stale tail")
    await task
    assert adapter.screen == {"m1": sealed}
    assert sc.final_response_sent is False
