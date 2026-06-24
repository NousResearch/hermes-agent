"""Tests for StreamDeltaWriter — non-blocking stream delta emission.

Verifies that:
1. Deltas are delivered in order to _emit
2. Drops deltas when queue is full (no blocking)
3. stop() drains remaining deltas
4. Provider thread is never blocked even when _emit is slow
"""

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)


@pytest.fixture
def writer_cls():
    """Import StreamDeltaWriter from the gateway server module."""
    import sys
    import os

    # Add the tui_gateway parent to path
    hermes_root = os.path.expanduser("~/.hermes/hermes-agent")
    if hermes_root not in sys.path:
        sys.path.insert(0, hermes_root)

    from tui_gateway.server import StreamDeltaWriter
    return StreamDeltaWriter


class TestStreamDeltaWriter:
    def test_deltas_delivered_in_order(self, writer_cls):
        """Pushed deltas arrive at _emit in order."""
        received = []

        with patch("tui_gateway.server._emit") as mock_emit:
            mock_emit.side_effect = lambda event, sid, payload: received.append(
                payload["text"]
            )
            writer = writer_cls("test-sid", None, max_queue=64).start()
            for i in range(10):
                writer.push(f"delta-{i}")
            writer.stop(timeout=2.0)

        assert received == [f"delta-{i}" for i in range(10)]

    def test_drops_when_queue_full(self, writer_cls):
        """When queue is full, push doesn't block — it drops."""
        # Use a tiny queue and a slow _emit to force overflow
        emit_barrier = threading.Event()
        call_count = {"n": 0}

        with patch("tui_gateway.server._emit") as mock_emit:
            def slow_emit(event, sid, payload):
                call_count["n"] += 1
                # Block the first drain to fill the queue
                if call_count["n"] == 1:
                    emit_barrier.wait(timeout=2.0)

            mock_emit.side_effect = slow_emit
            writer = writer_cls("test-sid", None, max_queue=4).start()

            # Push enough to fill queue + overflow
            # First one gets picked up by drain loop immediately
            for i in range(10):
                writer.push(f"delta-{i}")
                time.sleep(0.01)  # Give drain loop time to pick up first item

            # Release the barrier so drain can proceed
            emit_barrier.set()
            writer.stop(timeout=2.0)

        # Some deltas were delivered, some were dropped (never more than queue+1)
        assert call_count["n"] < 10  # Not all got through
        assert call_count["n"] >= 1  # At least one did

    def test_push_never_blocks(self, writer_cls):
        """push() returns immediately even with a full queue."""
        with patch("tui_gateway.server._emit") as mock_emit:
            # Make _emit block forever
            block = threading.Event()
            mock_emit.side_effect = lambda *a, **kw: block.wait(timeout=5.0)

            writer = writer_cls("test-sid", None, max_queue=2).start()

            # Measure push time — should be instant even with blocked drain
            start = time.time()
            for i in range(20):
                writer.push(f"delta-{i}")
            elapsed = time.time() - start

            # push should complete in well under 100ms total for 20 items
            assert elapsed < 0.1, f"push blocked for {elapsed:.3f}s"

            block.set()
            writer.stop(timeout=2.0)

    def test_stop_drains_remaining(self, writer_cls):
        """stop() flushes queued deltas before returning."""
        received = []

        with patch("tui_gateway.server._emit") as mock_emit:
            mock_emit.side_effect = lambda event, sid, payload: received.append(
                payload["text"]
            )

            # Pause the drain loop so items accumulate in queue
            writer = writer_cls("test-sid", None, max_queue=64)
            # Don't start yet — push items then start+stop atomically
            writer._stop_event.set()  # Pre-signal stop
            writer.push("a")
            writer.push("b")
            writer.push("c")

            # Now start — the drain loop will see stop immediately, then do final drain
            writer._stop_event.clear()
            writer.start()
            time.sleep(0.1)  # Let drain loop spin
            writer.stop(timeout=2.0)

        # All 3 should have been emitted
        assert "a" in received
        assert "b" in received
        assert "c" in received

    def test_streamer_feed_integration(self, writer_cls):
        """StreamDeltaWriter correctly feeds the streamer and includes rendered output."""
        rendered_output = []

        class FakeStreamer:
            def feed(self, text):
                return f"<rendered>{text}</rendered>"

        with patch("tui_gateway.server._emit") as mock_emit:
            def capture_emit(event, sid, payload):
                if "rendered" in payload:
                    rendered_output.append(payload["rendered"])

            mock_emit.side_effect = capture_emit
            writer = writer_cls("test-sid", FakeStreamer(), max_queue=64).start()
            writer.push("hello")
            writer.push("world")
            writer.stop(timeout=2.0)

        assert "<rendered>hello</rendered>" in rendered_output
        assert "<rendered>world</rendered>" in rendered_output

    def test_interrupt_scenario(self, writer_cls):
        """Simulates interrupt: provider thread unblocked, writer stops cleanly."""
        deltas_received = []

        with patch("tui_gateway.server._emit") as mock_emit:
            mock_emit.side_effect = lambda event, sid, payload: deltas_received.append(
                payload["text"]
            )

            writer = writer_cls("test-sid", None, max_queue=64).start()

            # Simulate provider pushing deltas
            writer.push("token1")
            writer.push("token2")
            time.sleep(0.1)

            # Simulate interrupt — just stop the writer
            writer.stop(timeout=0.5)

            # After stop, push is no-op (queue still accepts but drain is done)
            writer.push("after-stop")  # This won't be emitted

        assert "token1" in deltas_received
        assert "token2" in deltas_received
        # "after-stop" was pushed after stop(), drain loop has exited
        # It goes into the queue but nobody drains it
        assert "after-stop" not in deltas_received
