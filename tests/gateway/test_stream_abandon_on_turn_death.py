"""Regression: a dying turn closes its native stream (PR 85796 review, B8).

Stale-generation exits and cancellations returned from run() with the
native stream still open: the Slack message kept its live streaming
indicator forever, and the adapter's armed interception state survived
into the next turn (which could inherit the key and seal a dead
draft_id). The consumer now calls adapter.abandon_open_draft on both
death paths — sealing in place with the last delivered frame, claiming
nothing (no delivery flags). Native-stream transports (WeCom) get the
same treatment through a finalize frame.
"""

import asyncio

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.relay.test_relay_live_cards import _connected_adapter
from tests.gateway.test_stream_consumer_wecom_native import _make_native_streaming_adapter


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


class TestAbandonNativeStreamOnTurnDeath:
    """Native transports (WeCom ``send_stream_frame``) keep the stream alive until a
    finalize frame arrives, so a dying turn must send one too."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("death", ["stale", "cancel"])
    async def test_dying_turn_finalizes_open_native_stream(self, death):
        adapter = _make_native_streaming_adapter()
        _alive = [True]
        sc = GatewayStreamConsumer(
            adapter, "C1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
            run_still_current=lambda: _alive[0],
        )
        task = asyncio.create_task(sc.run())
        sc.on_delta("partial answer on screen")
        await asyncio.sleep(0.08)
        assert not any(f["finalize"] for f in adapter.frames)

        if death == "stale":
            _alive[0] = False
            await task
        else:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        finals = [f for f in adapter.frames if f["finalize"]]
        assert [f["text"] for f in finals] == ["partial answer on screen"]
        assert sc.final_response_sent is False
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_during_native_finalize_still_closes_state(self):
        """A cancel that lands inside the finalize send is not an Exception.

        ``_try_frame`` lets it propagate. The stream must still be disarmed,
        and the abandoned text must not suppress the next turn's first frame.
        """
        adapter = _make_native_streaming_adapter()
        started = asyncio.Event()

        async def _send_stream_frame(
            text, *, finalize=False, chat_id=None, reply_to=None, **kwargs
        ):
            if finalize:
                started.set()
                await asyncio.Event().wait()
            adapter.frames.append({
                "text": text, "finalize": finalize,
                "chat_id": chat_id, "reply_to": reply_to,
            })
            return True

        adapter.send_stream_frame = _send_stream_frame
        sc = GatewayStreamConsumer(
            adapter, "C1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        )
        sc._use_native_streaming = True
        sc._native_stream_opened = True
        sc._last_sent_text = "partial answer on screen"

        task = asyncio.create_task(sc._abandon_native_stream())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert sc._native_stream_opened is False
        assert sc._last_sent_text == ""
        assert not any(f["finalize"] for f in adapter.frames)
