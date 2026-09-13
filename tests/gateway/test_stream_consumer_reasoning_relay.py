"""Live reasoning relay: 💭 bubbles flushed from the stream consumer queue.

Covers the relay added for gateway parity with the CLI's reasoning box:
``agent.reasoning_callback`` → ``GatewayStreamConsumer.on_reasoning`` →
buffered deltas flushed as interim 💭 commentary on an age/length cadence.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _consumer(adapter):
    return GatewayStreamConsumer(
        adapter=adapter,
        chat_id="1",
        config=StreamConsumerConfig(edit_interval=0.05, buffer_threshold=10**6, cursor=""),
        metadata={},
        run_still_current=lambda: True,
    )


class _RecordingAdapter:
    """Minimal async adapter that records every send/edit with timestamps."""

    platform = SimpleNamespace(value="telegram")
    name = "telegram-test"

    def __init__(self):
        self.calls = []  # (kind, monotonic_ts, content)

    async def send(self, chat_id=None, content=None, reply_to=None, metadata=None, **kw):
        self.calls.append(("send", time.monotonic(), content))
        return SimpleNamespace(success=True, message_id=f"msg{len(self.calls)}")

    async def edit_message(self, chat_id=None, message_id=None, content=None, finalize=False, metadata=None, **kw):
        self.calls.append(("edit", time.monotonic(), content))
        return SimpleNamespace(success=True, message_id=message_id)

    async def send_stream_frame(self, *a, **kw):
        return False

    def message_len_fn_for_chat(self, chat_id):
        return len


def _bubbles(adapter):
    return [(ts, c) for kind, ts, c in adapter.calls if kind == "send" and c.startswith("💭")]


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


# ── intake ────────────────────────────────────────────────────────────────


class TestOnReasoningIntake:
    def test_empty_delta_is_dropped(self):
        consumer = _consumer(MagicMock())
        consumer.on_reasoning("")
        assert consumer._queue.qsize() == 0

    def test_delta_is_queued(self):
        consumer = _consumer(MagicMock())
        consumer.on_reasoning("thinking ")
        assert consumer._queue.qsize() == 1

    def test_drain_accumulates_into_buffer(self):
        consumer = _consumer(MagicMock())
        consumer.on_reasoning("alpha ")
        consumer.on_reasoning("beta")
        consumer._drain_queue()
        assert consumer._reasoning_buf == "alpha beta"


# ── flush cadence ─────────────────────────────────────────────────────────


class TestReasoningFlushCadence:
    def test_first_bubble_fast_path(self):
        """First bubble of a phase arrives at the first-bubble deadline (not
        the full cadence) when reasoning streams continuously."""
        GatewayStreamConsumer._REASONING_FLUSH_SECONDS = 6.0
        GatewayStreamConsumer._REASONING_FIRST_BUBBLE_SECONDS = 1.0
        adapter = _RecordingAdapter()
        consumer = _consumer(adapter)
        t0 = time.monotonic()

        async def scenario():
            task = asyncio.create_task(consumer.run())
            for _ in range(30):  # 3s of continuous deltas, no silence gap
                consumer.on_reasoning("chunk ")
                await asyncio.sleep(0.1)
            consumer.finish()
            await task

        _run(scenario())
        bubbles = _bubbles(adapter)
        assert bubbles, "expected at least one 💭 bubble"
        first_ts = bubbles[0][0]
        # Fast path: bubble landed at ~1s, well before the 6s cadence.
        assert first_ts - t0 < 3.0

    def test_char_threshold_flush(self):
        """Buffer at/over the char threshold flushes regardless of age."""
        consumer = _consumer(MagicMock())
        consumer._reasoning_buf = "x" * consumer._REASONING_FLUSH_CHARS
        consumer._reasoning_first_delta_ts = time.monotonic()
        consumer._reasoning_bubble_count = 1  # fast path already spent
        assert consumer._reasoning_flush_due() is True

    def test_young_buffer_under_threshold_not_due(self):
        consumer = _consumer(MagicMock())
        consumer._reasoning_buf = "young"
        consumer._reasoning_first_delta_ts = time.monotonic()
        consumer._reasoning_bubble_count = 1  # fast path spent, cadence rules
        assert consumer._reasoning_flush_due() is False

    def test_empty_buffer_never_due(self):
        consumer = _consumer(MagicMock())
        assert consumer._reasoning_flush_due() is False


# ── turn-level flags ──────────────────────────────────────────────────────


class TestReasoningDeliveredFlag:
    def test_flag_set_after_successful_bubble(self):
        adapter = _RecordingAdapter()
        consumer = _consumer(adapter)
        assert consumer.reasoning_delivered is False

        async def scenario():
            task = asyncio.create_task(consumer.run())
            consumer.on_reasoning("visible reasoning ")
            await asyncio.sleep(0.3)
            consumer.finish()
            await task

        _run(scenario())
        assert consumer.reasoning_delivered is True
        assert _bubbles(adapter)


# ── wiring gates (unit) ──────────────────────────────────────────────────


class TestRelayWiringGates:
    def test_on_reasoning_exists(self):
        consumer = _consumer(MagicMock())
        assert hasattr(consumer, "on_reasoning")
