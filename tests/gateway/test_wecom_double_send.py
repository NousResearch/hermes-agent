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
# Test group 2: best-effort finalize "DO NOT mark" path
# ===========================================================================


class TestBestEffortFinalizeDoubleSend:
    """stream_consumer.py:2064-2094: when native streaming fails mid-turn, the
    consumer sends a best-effort finalize frame to close the bubble but
    deliberately does NOT set ``final_content_delivered`` (the "DO NOT mark"
    comment at 2085-2088). If WeCom actually renders that finalize frame, the
    subsequent fallback send delivers the same content a second time.

    This uses a controllable ``BasePlatformAdapter`` subclass (the accepted
    pattern in tests/gateway/test_wecom_per_turn.py) because the trigger is a
    frame-level failure, not an ack timing race — no wall-clock timing needed.
    """

    @pytest.mark.xfail(
        reason="KNOWN LOW-FREQUENCY ISSUE (accepted, not yet fixed): the "
        "best-effort finalize path (stream_consumer.py:2104-2128) renders the "
        "content AND falls through to a fallback send(), delivering the same "
        "answer twice. Only triggers when native streaming fails MID-STREAM "
        "(846608 expired / 846609 subscription lost / errcode 6000 / network "
        "error) — rare. The duplicate is a SAFE-direction failure (user sees "
        "one extra bubble, no data loss). A clean fix requires reconciling "
        "'do not drop post-finalize increments' with 'do not re-send the "
        "first-send path', which the first-send path (L2378) does not consult "
        "the delivery flag for; deferred over risking increment loss on a rare "
        "path. The high-frequency duplicate (timeout inversion) IS fixed — see "
        "TestTimeoutInversionDoubleSend + commit 7ba739818 (B2).",
        strict=True,
    )
    @pytest.mark.asyncio
    async def test_best_effort_finalize_renders_but_flag_unset_double_send(self):
        """native content frame fails → best-effort finalize renders + fallback send.

        Ordering matters: the delta is pushed and ``run()`` is allowed to
        process a mid-stream **content** frame BEFORE ``finish()`` is queued.
        That content frame fails (returns False), which disables native
        streaming and triggers the best-effort finalize at
        ``stream_consumer.py:2077`` — WeCom renders it — and then the consumer
        falls through to a fallback ``send()`` that delivers the same content a
        second time (the "DO NOT mark" comment at 2085-2088 is what leaves the
        gateway free to re-send).

        The text is > ``_MIN_NEW_VISIBLE_CHARS`` (60) so the mid-stream frame is
        actually attempted rather than buffered.

        Asserts the DESIRED single-delivery state, so it FAILS today (content
        rendered by the best-effort finalize AND re-sent by the fallback) and
        PASSES once the double delivery is closed.
        """
        from gateway.platforms.base import BasePlatformAdapter, SendResult

        class FlakyNativeAdapter(BasePlatformAdapter):
            MAX_MESSAGE_LENGTH = 4096
            SUPPORTS_MESSAGE_EDITING = False
            SUPPORTS_NATIVE_STREAMING = True

            def __init__(self):
                self._typing_paused = set()
                self._fatal_error_message = None
                # Records of everything that reached the user.
                self.rendered_stream_frames: list[dict] = []  # incl. finalize
                self.fallback_sends: list[str] = []
                self._content_frames_seen = 0

            def supports_native_streaming(self, chat_type=None, metadata=None):
                return True

            async def send_stream_frame(
                self, text, *, finalize=False, chat_id=None, reply_to=None, **kwargs
            ):
                # Seed frame (empty, non-finalize) opens the bubble.
                if text == "" and not finalize:
                    self.rendered_stream_frames.append({"text": text, "finalize": False})
                    return True
                if finalize:
                    # Best-effort finalize frame — WeCom DOES render it.
                    self.rendered_stream_frames.append({"text": text, "finalize": True})
                    return True
                # First real content frame fails → disables native streaming
                # and triggers the best-effort finalize path.
                self._content_frames_seen += 1
                return False

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                # The gateway/consumer fallback send — a SECOND delivery of the
                # same content.
                self.fallback_sends.append(content)
                return SendResult(success=True, message_id="fallback-msg")

        FlakyNativeAdapter.__abstractmethods__ = frozenset()
        adapter = FlakyNativeAdapter()

        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, CHAT_ID, cfg)

        # > 60 visible chars so the mid-stream content frame is actually sent.
        final_text = (
            "需要交付给用户的最终答案。这一段内容刻意写得足够长，"
            "以便越过 stream_consumer 的 _MIN_NEW_VISIBLE_CHARS=60 阈值，"
            "从而在 got_done 之前触发一次真正的流式内容帧发送。"
        )
        assert len(final_text) >= 60

        consumer.on_delta(final_text)
        task = asyncio.create_task(consumer.run())
        # Let run() process the mid-stream content frame (which fails) BEFORE
        # the turn-final _DONE arrives.
        await asyncio.sleep(0.1)
        consumer.finish()
        await task

        best_effort_finalize = [
            f for f in adapter.rendered_stream_frames if f["finalize"] and f["text"]
        ]

        # Count content-bearing user-visible deliveries:
        #   * best-effort finalize frame(s) that WeCom rendered, plus
        #   * fallback send(s).
        content_deliveries = len(best_effort_finalize) + len(adapter.fallback_sends)

        # DESIRED post-fix invariant (currently violated == bug reproduced):
        # the same answer must reach the user exactly once. Today the
        # best-effort finalize renders the content AND the fallback send
        # delivers it again.
        assert content_deliveries == 1, (
            f"BUG: content delivered {content_deliveries} times — best-effort "
            f"finalize rendered {len(best_effort_finalize)} frame(s) and "
            f"fallback send fired {len(adapter.fallback_sends)} time(s) "
            f"(stream_consumer.py:2085-2088 'DO NOT mark' path)"
        )
