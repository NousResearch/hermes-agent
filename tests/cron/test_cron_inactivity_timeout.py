"""Tests for the cron inactivity watchdog loop (runs on its own daemon thread)."""

import logging
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
    _raise_inactivity_timeout re-read a near-zero idle (idle for 3s
    (limit 600s)). The watchdog must record the full firing activity summary
    and the raise path must use it, including its iteration and tool.
    """

    def test_raise_uses_latched_sample_not_resample(self, caplog):
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

        with caplog.at_level(logging.ERROR):
            try:
                _raise_inactivity_timeout(
                    _FreshAgent(),
                    "demo-job",
                    600.0,
                    latched_activity={
                        "seconds_since_activity": 601,
                        "last_activity_desc": "stalled tool call",
                        "api_call_count": 7,
                        "max_iterations": 40,
                        "current_tool": "web_search",
                    },
                )
            except TimeoutError as e:
                message = str(e)
            else:
                raise AssertionError("expected TimeoutError")
        assert "idle for 601s" in message
        assert "stalled tool call" in message
        assert "idle for 3s" not in message
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(
            "idle for 601s" in r.getMessage()
            and "iteration=7/40" in r.getMessage()
            and "tool=web_search" in r.getMessage()
            for r in error_records
        ), [r.getMessage() for r in error_records]

    def test_raise_falls_back_to_live_sample_without_latch(self):
        from cron.scheduler import _raise_inactivity_timeout

        class _LiveAgent:
            def get_activity_summary(self):
                return {
                    "seconds_since_activity": 12,
                    "last_activity_desc": "terminal command running",
                    "api_call_count": 8,
                    "max_iterations": 40,
                    "current_tool": "terminal",
                }

            def interrupt(self, message=None):
                return True

        try:
            _raise_inactivity_timeout(_LiveAgent(), "demo-job", 600.0)
        except TimeoutError as e:
            message = str(e)
        else:
            raise AssertionError("expected TimeoutError")
        assert "idle for 12s" in message
        assert "terminal command running" in message

    def test_raise_tolerates_none_snapshot_values(self):
        """Direct callers may hand a snapshot with None values: no TypeError."""
        from cron.scheduler import _raise_inactivity_timeout

        class _NoneAgent:
            def get_activity_summary(self):
                return {
                    "seconds_since_activity": None,
                    "last_activity_desc": "",
                    "api_call_count": 0,
                    "max_iterations": 0,
                    "current_tool": None,
                }

            def interrupt(self, message=None):
                return True

        try:
            _raise_inactivity_timeout(_NoneAgent(), "demo-job", 600.0)
        except TimeoutError as e:
            message = str(e)
        else:
            raise AssertionError("expected TimeoutError")
        assert "idle for 0s" in message
        assert "last activity: unknown" in message

    def test_watchdog_to_raise_reports_firing_state_end_to_end(self, monkeypatch, caplog):
        """The watchdog -> latch -> raise path must pin the firing sample as a unit.

        The fake agent reports a long idle on the first sample (the one that trips
        the limit) and a fresh sample after; both the TimeoutError and the captured
        logger.error record must show the firing state, including iteration/tool.
        """
        from cron.scheduler import _run_agent_with_watchdog

        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "600")

        firing = {
            "seconds_since_activity": 650,
            "last_activity_desc": "executing tool: web_search",
            "api_call_count": 7,
            "max_iterations": 40,
            "current_tool": "web_search",
        }
        fresh = {
            "seconds_since_activity": 3,
            "last_activity_desc": "fresh activity",
            "api_call_count": 8,
            "max_iterations": 40,
            "current_tool": None,
        }
        calls = {"n": 0}
        release = threading.Event()

        class _StallingAgent:
            def get_activity_summary(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    return dict(firing)
                return dict(fresh)

            def run_conversation(self, prompt, task_id=None):
                release.wait(timeout=30)
                return {}

            def interrupt(self, message=None):
                return True

        with caplog.at_level(logging.ERROR):
            try:
                _run_agent_with_watchdog(
                    _StallingAgent(),
                    "prompt",
                    {},
                    "job-e2e-12345678",
                    "probe job",
                    "task-1",
                    None,
                )
            except TimeoutError as e:
                message = str(e)
            else:
                raise AssertionError("expected TimeoutError")
            finally:
                release.set()

        assert "idle for 650s" in message
        assert "executing tool: web_search" in message
        assert "idle for 3s" not in message
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(
            "idle for 650s" in r.getMessage()
            and "iteration=7/40" in r.getMessage()
            and "tool=web_search" in r.getMessage()
            for r in error_records
        ), [r.getMessage() for r in error_records]
