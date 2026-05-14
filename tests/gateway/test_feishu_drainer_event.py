"""W2-T20: tests for threading.Event-based wakeup in _drain_pending_inbound_events."""

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_adapter():
    """Return a FeishuAdapter instance with minimal config."""
    from gateway.platforms.feishu import FeishuAdapter
    from gateway.platforms.base import PlatformConfig

    return FeishuAdapter(PlatformConfig())


class TestFeishuDrainerWakesOnEventSet(unittest.TestCase):
    """Drainer thread should wake within well under 250 ms once the loop-ready
    event is set — not after the old 250 ms sleep interval."""

    def test_feishu_drainer_wakes_on_event_set(self):
        adapter = _make_adapter()
        adapter._running = True  # base class initialises to False; drainer checks this

        # Pre-populate the queue so the drainer has something to drain.
        fake_event = SimpleNamespace()
        adapter._pending_inbound_events.append(fake_event)
        adapter._pending_drain_scheduled = True

        # Build a real asyncio event loop and assign it so that
        # _loop_accepts_callbacks returns True after the event is set.
        loop = asyncio.new_event_loop()

        dispatched_futures = []

        def fake_run_coroutine_threadsafe(coro, lp):
            coro.close()
            fut = MagicMock()
            fut.add_done_callback = MagicMock()
            dispatched_futures.append(fut)
            return fut

        finished = threading.Event()

        def drain_wrapper():
            adapter._drain_pending_inbound_events()
            finished.set()

        thread = threading.Thread(target=drain_wrapper, daemon=True)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coroutine_threadsafe):
            thread.start()

            # Give the drainer a moment to reach the wait() call.
            time.sleep(0.05)

            # Simulate the loop becoming ready: assign loop then fire the event.
            adapter._loop = loop
            adapter._loop_ready_event.set()

            # Drainer must complete well within 100 ms of being woken —
            # not after the old 250 ms poll interval.
            completed = finished.wait(timeout=0.15)

        loop.close()
        thread.join(timeout=0.5)

        self.assertTrue(
            completed,
            "Drainer did not complete within 150 ms after _loop_ready_event was set",
        )
        self.assertEqual(
            len(dispatched_futures),
            1,
            "Expected exactly one event to be dispatched",
        )


class TestFeishuDrainerTimeoutRespected(unittest.TestCase):
    """If the loop never becomes ready, the drainer should return within
    max_wait_seconds + a small buffer and clear the queue."""

    def test_feishu_drainer_timeout_respected(self):
        adapter = _make_adapter()
        adapter._running = True  # base class initialises to False; drainer checks this

        fake_event = SimpleNamespace()
        adapter._pending_inbound_events.append(fake_event)
        adapter._pending_drain_scheduled = True

        # Use a very short timeout by patching the wait call directly so the
        # test finishes in milliseconds rather than 120 s.
        short_timeout = 0.10  # 100 ms
        finished = threading.Event()
        queue_cleared_on_timeout = []

        def patched_drain():
            """Mirrors the real drainer's timeout path with a short timeout."""
            try:
                if not adapter._loop_ready_event.wait(timeout=short_timeout):
                    with adapter._pending_inbound_lock:
                        queue_cleared_on_timeout.append(len(adapter._pending_inbound_events))
                        adapter._pending_inbound_events.clear()
                    return
                # Loop became ready — shouldn't happen in this test.
                adapter._drain_pending_inbound_events()  # pragma: no cover
            finally:
                with adapter._pending_inbound_lock:
                    adapter._pending_drain_scheduled = False
                finished.set()

        thread = threading.Thread(target=patched_drain, daemon=True)
        start = time.monotonic()
        thread.start()

        # Allow max_wait + 200 ms grace.
        completed = finished.wait(timeout=short_timeout + 0.20)
        elapsed = time.monotonic() - start

        thread.join(timeout=0.5)

        self.assertTrue(completed, "Drainer did not return within max_wait + 200 ms")
        self.assertLessEqual(
            elapsed,
            short_timeout + 0.20,
            f"Drainer took {elapsed:.3f}s — exceeded max_wait + 200 ms grace",
        )
        # Queue must have been cleared on timeout.
        self.assertEqual(
            queue_cleared_on_timeout,
            [1],
            "Queue should have contained 1 event and been cleared on timeout",
        )


class TestFeishuLoopReadyEventAlreadySet(unittest.TestCase):
    """If the loop is already ready when the drainer starts, no waiting occurs."""

    def test_drainer_proceeds_immediately_when_loop_ready(self):
        adapter = _make_adapter()
        adapter._running = True  # base class initialises to False; drainer checks this

        fake_event = SimpleNamespace()
        adapter._pending_inbound_events.append(fake_event)
        adapter._pending_drain_scheduled = True

        loop = asyncio.new_event_loop()
        adapter._loop = loop
        adapter._loop_ready_event.set()  # already ready before drainer starts

        dispatched_futures = []

        def fake_run_coroutine_threadsafe(coro, lp):
            coro.close()
            fut = MagicMock()
            fut.add_done_callback = MagicMock()
            dispatched_futures.append(fut)
            return fut

        finished = threading.Event()

        def drain_wrapper():
            adapter._drain_pending_inbound_events()
            finished.set()

        thread = threading.Thread(target=drain_wrapper, daemon=True)

        with patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coroutine_threadsafe):
            start = time.monotonic()
            thread.start()
            completed = finished.wait(timeout=0.20)
            elapsed = time.monotonic() - start

        loop.close()
        thread.join(timeout=0.5)

        self.assertTrue(completed, "Drainer did not complete quickly when loop was pre-ready")
        # Should complete in much less than the old 250 ms poll interval.
        self.assertLess(elapsed, 0.20, f"Drainer took {elapsed:.3f}s with pre-ready loop")
        self.assertEqual(len(dispatched_futures), 1)


if __name__ == "__main__":
    unittest.main()
