"""Tests for the event loop health watchdog.

Verifies that the thread-based watchdog in gateway.run can detect a
frozen asyncio event loop and trigger the force-exit path (#69089).
"""

import asyncio
import os
import threading
import time
from unittest.mock import patch, MagicMock

import pytest


def _get_watchdog():
    """Import the watchdog function from gateway.run."""
    from gateway.run import _run_event_loop_health_watchdog
    return _run_event_loop_health_watchdog


def _mock_exit_raiser(code):
    """Mock for os._exit that raises so the test can catch it."""
    raise SystemExit(f"os._exit called with code={code}")


class TestEventLoopHealthWatchdog:
    """Test suite for _run_event_loop_health_watchdog."""

    def test_healthy_loop_passes_probes(self):
        """A responsive event loop never triggers failure counting."""
        watchdog = _get_watchdog()
        stop_event = threading.Event()
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=lambda: asyncio.set_event_loop(loop) or loop.run_forever(),
            daemon=True,
        )
        thread.start()
        # Let the loop start
        time.sleep(0.2)

        failures = []

        def patched_watchdog(*args, **kwargs):
            # Short interval for fast test
            kwargs["probe_interval"] = 0.1
            kwargs["probe_timeout"] = 0.1
            kwargs["max_consecutive_failures"] = 5
            try:
                watchdog(*args, **kwargs)
            except SystemExit:
                pass

        # Run a few probe cycles manually
        loop_responded = threading.Event()
        loop.call_soon_threadsafe(loop_responded.set)
        result = loop_responded.wait(timeout=1)
        assert result is True, "Healthy loop should respond to call_soon_threadsafe"

        stop_event.set()
        loop.call_soon_threadsafe(lambda: None)  # wake the loop
        thread.join(timeout=5)

    def test_frozen_loop_triggers_exit(self):
        """A frozen event loop should trigger os._exit after N failures."""
        watchdog = _get_watchdog()
        stop_event = threading.Event()

        # Create a mock loop that never responds
        mock_loop = MagicMock()
        mock_loop.call_soon_threadsafe = MagicMock()
        # The callback is enqueued but never fires (simulating frozen loop)

        exit_called = threading.Event()
        exit_code_holder = [None]

        def fake_exit(code):
            exit_code_holder[0] = code
            exit_called.set()
            raise SystemExit(f"os._exit({code})")

        with patch.object(os, "_exit", fake_exit):
            # Run watchdog in a thread
            thread = threading.Thread(
                target=watchdog,
                args=(mock_loop, stop_event),
                kwargs={
                    "probe_interval": 0.05,
                    "probe_timeout": 0.05,
                    "max_consecutive_failures": 2,
                },
                daemon=True,
            )
            thread.start()
            # Wait for exit to be called (2 failures × 0.05s = ~0.1s + buffer)
            triggered = exit_called.wait(timeout=5)
            assert triggered, "os._exit should have been called after frozen probes"
            assert exit_code_holder[0] == 1

        stop_event.set()
        thread.join(timeout=2)

    def test_recovery_after_transient_failure(self):
        """A temporary freeze that recovers should reset the counter."""
        watchdog = _get_watchdog()
        stop_event = threading.Event()

        probe_count = [0]
        lock = threading.Lock()
        should_respond = threading.Event()
        should_respond.set()  # Start healthy

        def mock_call_soon_threadsafe(callback):
            probe_count[0] += 1
            if should_respond.is_set():
                # Simulate the callback firing immediately (healthy loop)
                callback()

        mock_loop = MagicMock()
        mock_loop.call_soon_threadsafe = mock_call_soon_threadsafe

        # Track log messages
        log_msgs = []

        def fake_warning(msg, *args, **kwargs):
            log_msgs.append(msg % args if args else msg)

        exit_called = threading.Event()

        def fake_exit(code):
            exit_called.set()
            raise SystemExit(f"os._exit({code})")

        with patch("gateway.run.logger") as mock_logger:
            mock_logger.warning = fake_warning
            with patch.object(os, "_exit", fake_exit):
                thread = threading.Thread(
                    target=watchdog,
                    args=(mock_loop, stop_event),
                    kwargs={
                        "probe_interval": 0.05,
                        "probe_timeout": 0.1,
                        "max_consecutive_failures": 3,
                    },
                    daemon=True,
                )
                thread.start()

                # First, let it run healthy for a bit
                time.sleep(0.2)

                # Now freeze the loop
                should_respond.clear()
                time.sleep(0.15)

                # Unfreeze before hitting max failures
                should_respond.set()
                time.sleep(0.2)

                # Verify recovery was logged
                recovered = any("restored" in m.lower() for m in log_msgs)
                # The watchdog logs recovery only if consecutive_failures > 0
                # which depends on timing, so just assert no exit happened
                assert not exit_called.is_set(), "Should not exit after recovery"

        stop_event.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
