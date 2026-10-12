"""Repro tests for the WeCom native-streaming "同一回复发两条" duplicate.

Root cause (see docs/rca-wecom-stream-final-ack-timeout-duplicate.md and
/tmp/claude-wecom-dup-report.md): a **timeout inversion race** between two
independent timers.

  * The consumer's got_done finalize blocks on the finalize frame's ack,
    bounded by ``adapter._REPLY_ACK_TIMEOUT``.
  * The gateway's ``finally`` block joins the ``stream_task`` with a hardcoded
    ``timeout=5.0`` (``gateway/run.py:20749``) and then ``stream_task.cancel()``
    (``run.py:20751``).

When the ack is slower than the gateway's join window, the gateway cancels the
consumer *before* it finishes finalizing. The finalize frame's bytes, however,
were already written to the wire by an **independent control-worker task**
(``_control_send_worker`` / ``_enqueue_chat_send(is_control=True)``) and WeCom
has rendered them. But because the consumer was cancelled mid-await, the
``self._final_content_delivered = True`` line (``stream_consumer.py:2061`` /
``1016``) never runs. The gateway then reads ``final_content_delivered=False``
(``run.py:20801-20803``), does NOT suppress the normal final send
(``run.py:20819``), and emits a second, duplicate bubble.

These tests exercise the **real** ``GatewayStreamConsumer.run()`` lifecycle
against the **real** ``WeComAdapter`` (only the websocket byte-writer and the
ack timing are controlled), so the async interaction between the gateway join /
cancel and the consumer finalize / flag-set actually happens — this is what the
previous PR (#62861) failed to test when it mocked out ``handle_message``.

Assertions target observable causality:
  * whether the finalize frame's bytes reached the wire (WeCom rendered it);
  * the value of ``consumer.final_content_delivered`` (the flag the gateway
    reads);
  * whether the gateway's suppression predicate fires (i.e. whether a second
    normal send would go out).

The suppression predicate below is a faithful mirror of the (non-importable,
nested) logic in ``gateway/run.py:20793-20819`` — it reads the *real* consumer
properties, it does not re-implement the delivery lifecycle.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


CHAT_ID = "chat-1"
REQ_ID = "req-1"


# ---------------------------------------------------------------------------
# Faithful mirrors of the real gateway control flow (cite source lines).
# ---------------------------------------------------------------------------


def _gateway_suppresses_normal_send(consumer, final_text: str) -> bool:
    """Mirror of gateway/run.py:20793-20819 suppression decision.

    Reads the REAL consumer properties (``final_response_sent`` /
    ``final_content_delivered``) — the same attributes ``run.py`` consults —
    and returns True when the gateway would set ``already_sent`` and skip the
    normal final send. When it returns False the gateway emits a second bubble.
    """
    _final = final_text or ""
    _is_empty_sentinel = not _final or _final == "(empty)"
    # run.py:20814 _stream_confirmed_final_delivery(...) with previewed=False
    # collapses to final_response_sent for a non-previewed response.
    _streamed = bool(getattr(consumer, "final_response_sent", False))
    _content_delivered = bool(getattr(consumer, "final_content_delivered", False))
    _transformed = False  # no plugin transform in these scenarios
    return (
        not _is_empty_sentinel
        and not _transformed
        and (_streamed or _content_delivered)
    )


async def _gateway_join_and_cancel(stream_task: asyncio.Task, join_timeout: float) -> None:
    """Faithful copy of gateway/run.py:20748-20755 finally-block join.

    Joins the stream task with a bounded timeout; on timeout it cancels the
    task (exactly what the gateway does when the consumer has not finished
    finalizing within the join window).
    """
    try:
        await asyncio.wait_for(stream_task, timeout=join_timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Real WeComAdapter with a controllable websocket + ack timing.
# ---------------------------------------------------------------------------


def _make_real_wecom_adapter(*, resolve_finalize_ack: bool):
    """Build a real ``WeComAdapter`` whose only fakes are the WS byte-writer
    and the ack timing.

    ``_send_json`` (the actual byte-writer) records every frame so we can prove
    the finalize frame reached the wire. The ack for **non-final** frames (seed
    / intermediate) is always resolved immediately so the finalize's
    pre-drain (``_send_reply_queued`` is_final branch) doesn't block on the
    seed's ack. The finalize frame's own ack is resolved immediately only when
    ``resolve_finalize_ack`` is True.

      * ``resolve_finalize_ack=False`` → finalize ack stays pending → the
        consumer blocks in its got_done finalize (the exact window where the
        gateway's join fires and cancels it). This is the bug scenario.
      * ``resolve_finalize_ack=True``  → finalize ack returns at once → the
        consumer finishes finalize and sets its flags before any join fires.
    """
    from plugins.platforms.wecom.adapter import WeComAdapter
    from gateway.config import PlatformConfig

    adapter = WeComAdapter(PlatformConfig(enabled=True))
    adapter._ws = MagicMock(closed=False)
    adapter._last_chat_req_ids[CHAT_ID] = REQ_ID

    frames: list[dict] = []

    async def _fake_send_json(payload: dict) -> None:
        # This is the real byte-writer boundary: reaching here means the frame
        # was put on the wire and WeCom will render it.
        frames.append(payload)
        stream = payload.get("body", {}).get("stream", {})
        finish = bool(stream.get("finish"))
        req = payload.get("headers", {}).get("req_id")

        should_resolve = (not finish) or resolve_finalize_ack
        if should_resolve:
            queue = adapter._reply_queues.get(req)
            if queue and queue.pending_ack and not queue.pending_ack.future.done():
                # Simulate WeCom's ack coming back on the same WS.
                queue.pending_ack.future.set_result({"body": {"errcode": 0}})

    adapter._send_json = _fake_send_json
    adapter._recorded_frames = frames
    return adapter


def _finalize_frames_on_wire(adapter) -> list[dict]:
    """Return the finish=true stream frames that reached the byte-writer."""
    out = []
    for payload in adapter._recorded_frames:
        stream = payload.get("body", {}).get("stream", {})
        if stream.get("finish") is True:
            out.append(payload)
    return out


async def _cleanup_adapter(adapter) -> None:
    """Cancel any lingering control/normal workers so the event loop is clean."""
    for task in list(adapter._control_workers.values()) + list(adapter._chat_workers.values()):
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# ===========================================================================
# Test group 1: timeout-inversion race (core reproduction)
# ===========================================================================


class TestTimeoutInversionDoubleSend:
    """The finalize ack being slower than the gateway join window must not
    strand ``final_content_delivered=False`` while WeCom already rendered the
    finalize frame — that produces a duplicate bubble."""

    @pytest.mark.asyncio
    async def test_slow_ack_beyond_gateway_join_causes_double_send(self):
        """ack slower than the gateway join → consumer cancelled mid-finalize.

        Timeline (deterministic, no wall-clock sleeps):
          * seed + finalize frame bytes are written to the wire;
          * the finalize ack never returns within the join window
            (``resolve_finalize_ack=False``, ``_REPLY_ACK_TIMEOUT`` > join);
          * the gateway join (0.1s) fires and cancels the consumer;
          * the consumer's got_done finalize await is cancelled BEFORE the
            ``final_content_delivered = True`` line runs.

        The finalize frame is on the wire (rendered), yet the gateway reads
        ``final_content_delivered=False`` and would send a normal duplicate.

        This test asserts the DESIRED post-fix state, so it FAILS today
        (exposing the double send) and PASSES once the flag reflects the
        rendered finalize frame.
        """
        adapter = _make_real_wecom_adapter(resolve_finalize_ack=False)
        # Inversion: ack window (0.3s) is LARGER than the gateway join (0.1s),
        # so the join fires while the finalize ack is still pending.
        adapter._REPLY_ACK_TIMEOUT = 0.3
        join_timeout = 0.1

        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, CHAT_ID, cfg)

        final_text = "这是模型这一轮生成的最终回答，需要作为流式 finalize 帧发出。"
        consumer.on_delta(final_text)
        consumer.finish()

        stream_task = asyncio.create_task(consumer.run())
        try:
            # Gateway finally-block join (run.py:20748-20755) fires while the
            # finalize ack is still pending, cancelling the consumer.
            await _gateway_join_and_cancel(stream_task, join_timeout)

            finalize_frames = _finalize_frames_on_wire(adapter)

            # The finalize frame reached the wire → WeCom rendered it. This is
            # the first (and should be ONLY) user-visible bubble.
            assert len(finalize_frames) >= 1, (
                "finalize frame never reached the wire — repro precondition "
                "not met (WeCom must have rendered the streamed final)"
            )
            assert final_text in finalize_frames[-1]["body"]["stream"]["content"]

            gateway_would_send_normal = not _gateway_suppresses_normal_send(
                consumer, final_text
            )
            total_user_visible = len(finalize_frames) + (
                1 if gateway_would_send_normal else 0
            )

            # DESIRED post-fix invariants (currently violated == bug reproduced):
            assert consumer.final_content_delivered is True, (
                "BUG: consumer was cancelled mid-finalize; the finalize frame "
                "was rendered by WeCom but final_content_delivered stayed False, "
                "so the gateway will not suppress the normal send"
            )
            assert _gateway_suppresses_normal_send(consumer, final_text) is True, (
                "BUG: gateway does not suppress the normal final send → duplicate"
            )
            assert total_user_visible == 1, (
                f"BUG: user sees {total_user_visible} bubbles for one reply "
                f"(finalize frame rendered + normal send fired)"
            )
        finally:
            await _cleanup_adapter(adapter)

    @pytest.mark.asyncio
    async def test_fast_ack_within_join_window_single_send(self):
        """Control: ack returns within the join window → single bubble.

        The finalize ack resolves immediately, so the consumer completes its
        got_done finalize, sets ``final_content_delivered=True``, and the run
        finishes before the gateway join needs to cancel anything. The gateway
        then suppresses the normal send. Passes today and after the fix
        (regression guard).
        """
        adapter = _make_real_wecom_adapter(resolve_finalize_ack=True)
        adapter._REPLY_ACK_TIMEOUT = 5.0
        join_timeout = 0.5

        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, CHAT_ID, cfg)

        final_text = "这是一个能在 join 窗口内正常拿到 ack 的回答。"
        consumer.on_delta(final_text)
        consumer.finish()

        stream_task = asyncio.create_task(consumer.run())
        try:
            await _gateway_join_and_cancel(stream_task, join_timeout)

            finalize_frames = _finalize_frames_on_wire(adapter)
            assert len(finalize_frames) == 1

            assert consumer.final_content_delivered is True
            assert _gateway_suppresses_normal_send(consumer, final_text) is True

            gateway_would_send_normal = not _gateway_suppresses_normal_send(
                consumer, final_text
            )
            total_user_visible = len(finalize_frames) + (
                1 if gateway_would_send_normal else 0
            )
            assert total_user_visible == 1
        finally:
            await _cleanup_adapter(adapter)


# ===========================================================================
# Test group 2: orphan-queue ack routing race (adapter-level)
# ===========================================================================


def _make_manual_ack_adapter():
    """Real WeComAdapter whose ``_send_json`` records frames but NEVER
    auto-resolves any ack, so the test can orchestrate the exact interleaving
    of intermediate-ack arrival vs. finalize registration by hand.
    """
    from plugins.platforms.wecom.adapter import WeComAdapter
    from gateway.config import PlatformConfig

    adapter = WeComAdapter(PlatformConfig(enabled=True))
    adapter._ws = MagicMock(closed=False)
    adapter._last_chat_req_ids[CHAT_ID] = REQ_ID

    frames: list[dict] = []

    async def _fake_send_json(payload: dict) -> None:
        frames.append(payload)

    adapter._send_json = _fake_send_json
    adapter._recorded_frames = frames
    return adapter


class TestOrphanQueueAckRouting:
    """The finalize frame shares the inbound req_id with the intermediate
    frames. While the finalize awaits the pending intermediate ack to drain,
    that intermediate ack can arrive and ``_resolve_reply_ack`` pops the WHOLE
    queue out of ``_reply_queues``. Before the fix, the finalize then
    registered its ``pending_ack`` on the now-orphaned queue object (detached
    from the dict), so its own ack landed in Unrouted and the finalize hit a
    15s timeout. The fix re-attaches the queue to the dict before registering.
    """

    @pytest.mark.asyncio
    async def test_intermediate_ack_mid_drain_does_not_orphan_finalize_queue(self):
        adapter = _make_manual_ack_adapter()
        adapter._REPLY_ACK_TIMEOUT = 0.3  # snappy: bounds the failure mode

        # 1) Fire an intermediate frame (skip_if_pending) — occupies pending_ack.
        await adapter._send_reply_queued(
            REQ_ID,
            {"msgtype": "stream", "stream": {"id": "s1", "finish": False, "content": "hi"}},
            is_final=False, skip_if_pending=True,
        )
        inter_queue = adapter._reply_queues[REQ_ID]
        assert inter_queue.pending_ack is not None

        # 2) Start the finalize: it enters the is_final drain branch and yields
        #    at `await shield(pending_frame.future)`.
        fin_task = asyncio.create_task(adapter._send_reply_queued(
            REQ_ID,
            {"msgtype": "stream", "stream": {"id": "s1", "finish": True, "content": "done"}},
            is_final=True, skip_if_pending=False,
        ))
        await asyncio.sleep(0)  # let finalize reach the drain await

        # 3) Deliver the intermediate ack MID-DRAIN. _resolve_reply_ack resolves
        #    the drain future AND pops the whole queue out of _reply_queues.
        await adapter._dispatch_payload(
            {"headers": {"req_id": REQ_ID}, "body": {"errcode": 0}}
        )
        assert REQ_ID not in adapter._reply_queues, (
            "precondition: intermediate ack should have popped the queue"
        )

        # 4) Let finalize resume: it must clear the drained pending_ack, create
        #    its own frame, re-attach the queue, register pending_ack, and write
        #    bytes. Resuming through the shield/wait_for wrapper takes several
        #    event-loop iterations, so yield until the finalize has re-registered
        #    (bounded, to avoid a hang if the fix regresses).
        for _ in range(20):
            await asyncio.sleep(0)
            q = adapter._reply_queues.get(REQ_ID)
            if q is not None and q.pending_ack is not None:
                break

        # FIX ASSERTION: the finalize must have re-attached its queue to the
        # dict, so its ack is routable. Without the fix the queue would still
        # be absent here (orphaned) and the finalize ack would go Unrouted.
        assert REQ_ID in adapter._reply_queues, (
            "orphan-queue bug: finalize registered pending_ack on a queue "
            "detached from _reply_queues — its ack would be Unrouted"
        )
        assert adapter._reply_queues[REQ_ID].pending_ack is not None

        # 5) Deliver the finalize's own ack — it must route and resolve.
        await adapter._dispatch_payload(
            {"headers": {"req_id": REQ_ID}, "body": {"errcode": 0}}
        )
        resp = await asyncio.wait_for(fin_task, timeout=1.0)

        # Routed successfully → NOT the synthetic timeout-assumed response.
        assert resp.get("errmsg") != "ack_timeout_assumed_delivered", (
            "finalize ack should have been routed, not timed out"
        )

        await _cleanup_adapter(adapter)
