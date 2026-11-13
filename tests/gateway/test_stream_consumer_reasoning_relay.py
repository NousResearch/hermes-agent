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
        self.deleted = []  # (chat_id, message_id)

    async def send(self, chat_id=None, content=None, reply_to=None, metadata=None, **kw):
        self.calls.append(("send", time.monotonic(), content))
        return SimpleNamespace(success=True, message_id=f"msg{len(self.calls)}")

    async def edit_message(self, chat_id=None, message_id=None, content=None, finalize=False, metadata=None, **kw):
        self.calls.append(("edit", time.monotonic(), content))
        return SimpleNamespace(success=True, message_id=message_id)

    async def delete_message(self, chat_id, message_id):
        self.deleted.append((chat_id, message_id))
        return True

    async def send_stream_frame(self, *a, **kw):
        return False

    def message_len_fn_for_chat(self, chat_id):
        return len


def _bubbles(adapter):
    return [(ts, c) for kind, ts, c in adapter.calls if kind == "send" and ("💭" in c)]


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


# ── safety & entity formatting ───────────────────────────────────────────


class TestReasoningEntitySafety:
    def test_internal_double_pipes_are_sanitized(self):
        """Internal || (bash, OR, discussion) must be escaped to |\\| so they
        never collide with Telegram spoiler entities or leave dangling tags."""
        adapter = _RecordingAdapter()
        consumer = _consumer(adapter)

        async def scenario():
            task = asyncio.create_task(consumer.run())
            consumer.on_reasoning("checking cmd1 || cmd2 in text\n")
            await asyncio.sleep(0.3)
            consumer.finish()
            await task

        _run(scenario())
        bubbles = _bubbles(adapter)
        assert bubbles, "expected at least one 💭 bubble"
        content = bubbles[0][1]
        assert "|\\|" in content, f"expected escaped pipe |\\| in bubble, got: {content}"
        assert content.startswith("> 💭 thinking..."), f"expected plain blue quote >, got: {content}"
        assert not content.startswith("**>"), f"should not be collapsible **>, got: {content}"
        assert not content.endswith("||"), f"should not end with ||, got: {content}"

    def test_large_reasoning_is_windowed_into_single_safe_bubble(self, monkeypatch, tmp_path):
        """Long reasoning (>3200 chars) stays in one live bubble via sliding window and seals cleanly."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(GatewayStreamConsumer, "_REASONING_FLUSH_SECONDS", 6.0)
        monkeypatch.setattr(GatewayStreamConsumer, "_REASONING_FIRST_BUBBLE_SECONDS", 0.6)
        monkeypatch.setattr(GatewayStreamConsumer, "_REASONING_RETAIN_BUBBLES", 3)

        adapter = _RecordingAdapter()
        consumer = _consumer(adapter)

        async def scenario():
            task = asyncio.create_task(consumer.run())
            for i in range(40):
                consumer.on_reasoning(f"line {i:02d} " + ("x" * 120) + "\n")
            await asyncio.sleep(0.3)
            consumer.finish()
            await task

        _run(scenario())
        all_bubbles = [(kind, c) for kind, _, c in adapter.calls if "💭" in (c or "")]
        assert len(all_bubbles) >= 2, f"expected live send + sealed edit, got: {all_bubbles}"
        send_bubbles = [c for kind, c in all_bubbles if kind == "send"]
        assert len(send_bubbles) == 1, f"expected a single live reasoning bubble, got {len(send_bubbles)}"
        final_kind, final_content = all_bubbles[-1]
        assert final_kind == "edit"
        assert final_content.startswith("> 💭 thinking...\n"), f"expected plain sealed header, got: {final_content[:40]}"
        assert not final_content.startswith("**>"), f"should not be collapsible **>, got: {final_content[:40]}"
        assert not final_content.endswith("||"), f"should not end with ||, got: {final_content[-20:]}"
        assert "▉" not in final_content
        assert "earlier thinking omitted" in final_content
        assert len(final_content) <= 3800

    def test_live_reasoning_streams_via_edit_message(self, monkeypatch, tmp_path):
        """Reasoning updates after the first bubble edit the same Telegram message in place."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(GatewayStreamConsumer, "_REASONING_FLUSH_SECONDS", 0.15)
        monkeypatch.setattr(GatewayStreamConsumer, "_REASONING_FIRST_BUBBLE_SECONDS", 0.05)
        monkeypatch.setattr(GatewayStreamConsumer, "_REASONING_MIN_INITIAL_CHARS", 5)
        monkeypatch.setattr(GatewayStreamConsumer, "_REASONING_RETAIN_BUBBLES", 3)

        adapter = _RecordingAdapter()
        consumer = _consumer(adapter)

        async def scenario():
            task = asyncio.create_task(consumer.run())
            consumer.on_reasoning("First thought chunk.")
            await asyncio.sleep(0.25)
            consumer.on_reasoning(" Second thought chunk.")
            await asyncio.sleep(0.30)
            consumer.finish()
            await task

        _run(scenario())
        sends = [c for c in adapter.calls if c[0] == "send" and "💭" in (c[2] or "")]
        edits = [c for c in adapter.calls if c[0] == "edit" and "💭" in (c[2] or "")]
        assert len(sends) == 1, f"expected 1 reasoning send, got {len(sends)}"
        assert len(edits) >= 2, f"expected >=2 edits (live update + seal), got {len(edits)}"
        assert not sends[0][2].endswith("||")
        assert "▉" in sends[0][2]
        assert edits[-1][2].startswith("> 💭 thinking...\n")
        assert "▉" not in edits[-1][2]
        assert not edits[-1][2].endswith("||")

    def test_retains_up_to_two_bubbles_and_vaporizes_oldest(self, monkeypatch, tmp_path):
        """Default retain count (2): when >2 thinking bubbles accumulate for a chat, the oldest ones are deleted."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert GatewayStreamConsumer._REASONING_RETAIN_BUBBLES == 2

        async def scenario():
            for idx in range(1, 5):
                adapter = _RecordingAdapter()
                consumer = _consumer(adapter)
                consumer._reasoning_live_msg_id = f"r_{idx}"
                await consumer._close_reasoning_phase(f"Reasoning {idx}")
                if idx <= 2:
                    assert adapter.deleted == []
                elif idx == 3:
                    assert adapter.deleted == [(consumer.chat_id, "r_1")]
                elif idx == 4:
                    assert adapter.deleted == [(consumer.chat_id, "r_2")]

        _run(scenario())

    def test_immediate_vaporize_when_retain_is_zero(self, monkeypatch, tmp_path):
        """When _REASONING_RETAIN_BUBBLES == 0, closing the reasoning phase deletes the bubble immediately."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(GatewayStreamConsumer, "_REASONING_RETAIN_BUBBLES", 0)

        adapter = _RecordingAdapter()
        consumer = _consumer(adapter)

        async def scenario():
            consumer._reasoning_live_msg_id = "r_live"
            await consumer._close_reasoning_phase("Some thinking")
            assert adapter.deleted == [(consumer.chat_id, "r_live")]
            assert consumer._reasoning_live_msg_id is None

        _run(scenario())

