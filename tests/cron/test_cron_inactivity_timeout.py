"""Tests for the cron inactivity watchdog loop (runs on its own daemon thread)."""

import threading
import time


class TestInactivityWatchdogLoop:
    """The daemon-thread inactivity helper must not depend on the caller thread."""

    def test_fires_when_idle_crosses_limit(self):
        from cron.scheduler import _inactivity_watchdog_loop

        stop = threading.Event()
        idle = {"s": 0.0}
        results: list = []

        def _watch():
            results.append(
                _inactivity_watchdog_loop(
                    get_idle_seconds=lambda: idle["s"],
                    limit_s=0.2,
                    poll_s=0.05,
                    stop=stop,
                    future_done=lambda: False,
                )
            )

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        time.sleep(0.12)
        idle["s"] = 1.0
        watcher.join(timeout=2.0)
        stop.set()
        assert results == [True]
        assert not watcher.is_alive()

    def test_stops_when_future_completes_before_idle_limit(self):
        from cron.scheduler import _inactivity_watchdog_loop

        stop = threading.Event()
        fired = _inactivity_watchdog_loop(
            get_idle_seconds=lambda: 0.0,
            limit_s=10.0,
            poll_s=0.05,
            stop=stop,
            future_done=lambda: True,
        )
        assert fired is False

    def test_fires_while_caller_thread_is_blocked(self):
        """#94285: a blocked run_job thread must not disable the watchdog."""
        from cron.scheduler import _inactivity_watchdog_loop

        stop = threading.Event()
        idle = {"s": 1.0}
        result = {"fired": None}

        def _watch():
            result["fired"] = _inactivity_watchdog_loop(
                get_idle_seconds=lambda: idle["s"],
                limit_s=0.15,
                poll_s=0.05,
                stop=stop,
                future_done=lambda: False,
            )

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        # Simulate the family-A stall: this thread cannot poll.
        time.sleep(0.4)
        watcher.join(timeout=2.0)
        stop.set()
        assert result["fired"] is True
        assert not watcher.is_alive()



class TestLatchedIdleReport:
    """#127775: the raise path must report the latched idle sample, not a re-sample.

    The watchdog latches only a bool, so post-latch activity makes
    ``_raise_inactivity_timeout`` re-read a near-zero idle (``idle for 3s
    (limit 600s)``). The watchdog must record the latched idle duration
    plus description and the raise path must use it.
    """

    def test_watchdog_records_latched_idle_seconds(self):
        from cron.scheduler import _inactivity_watchdog_loop

        stop = threading.Event()
        latch: dict = {}
        fired = _inactivity_watchdog_loop(
            get_idle_seconds=lambda: 650.0,
            limit_s=600.0,
            poll_s=0.05,
            stop=stop,
            future_done=lambda: False,
            latch=latch,
        )
        assert fired is True
        assert latch.get("idle_seconds") == 650.0

    def test_raise_uses_latched_sample_not_resample(self):
        from cron.scheduler import _raise_inactivity_timeout

        class _FreshAgent:
            def get_activity_summary(self):
                # Post-latch activity: a re-sample would report ~3s idle.
                return {
                    "seconds_since_activity": 3,
                    "last_activity_desc": "fresh activity",
                    "api_call_count": 9,
                    "max_iterations": 50,
                    "current_tool": None,
                }

            def interrupt(self, message=None):
                return True

        try:
            _raise_inactivity_timeout(
                _FreshAgent(),
                "demo-job",
                600.0,
                latched_idle_seconds=601.0,
                latched_desc="stalled tool call",
            )
        except TimeoutError as e:
            message = str(e)
        else:
            raise AssertionError("expected TimeoutError")
        assert "idle for 601s" in message
        assert "stalled tool call" in message
        assert "idle for 3s" not in message
