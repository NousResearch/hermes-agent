"""Tests for cron job inactivity-based timeout.

Tests cover:
- Active agent runs indefinitely (no inactivity timeout)
- Idle agent triggers inactivity timeout with diagnostic info
- Unlimited timeout (HERMES_CRON_TIMEOUT=0)
- Backward compat: HERMES_CRON_TIMEOUT env var still works
- Error message includes activity summary
"""

import concurrent.futures
import os
import sys
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class FakeAgent:
    """Mock agent with controllable activity summary for timeout tests."""

    def __init__(self, idle_seconds=0.0, activity_desc="tool_call",
                 current_tool=None, api_call_count=5, max_iterations=90):
        self._idle_seconds = idle_seconds
        self._activity_desc = activity_desc
        self._current_tool = current_tool
        self._api_call_count = api_call_count
        self._max_iterations = max_iterations
        self._interrupted = False
        self._interrupt_msg = None

    def get_activity_summary(self):
        return {
            "last_activity_ts": time.time() - self._idle_seconds,
            "last_activity_desc": self._activity_desc,
            "seconds_since_activity": self._idle_seconds,
            "current_tool": self._current_tool,
            "api_call_count": self._api_call_count,
            "max_iterations": self._max_iterations,
        }

    def interrupt(self, msg):
        self._interrupted = True
        self._interrupt_msg = msg

    def run_conversation(self, prompt):
        """Simulate a quick agent run that finishes immediately."""
        return {"final_response": "Done", "messages": []}


class SlowFakeAgent(FakeAgent):
    """Agent that runs for a while, simulating active work then going idle."""

    def __init__(self, run_duration=0.5, idle_after=None, **kwargs):
        super().__init__(**kwargs)
        self._run_duration = run_duration
        self._idle_after = idle_after  # seconds before becoming idle
        self._start_time = None

    def get_activity_summary(self):
        summary = super().get_activity_summary()
        if self._idle_after is not None and self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed > self._idle_after:
                # Agent has gone idle
                idle_time = elapsed - self._idle_after
                summary["seconds_since_activity"] = idle_time
                summary["last_activity_desc"] = "api_call_streaming"
            else:
                summary["seconds_since_activity"] = 0.0
        return summary

    def run_conversation(self, prompt):
        self._start_time = time.time()
        time.sleep(self._run_duration)
        return {"final_response": "Completed after work", "messages": []}


class TestInactivityTimeout:
    """Test the inactivity-based timeout polling loop in cron scheduler."""

    def test_active_agent_completes_normally(self):
        """An agent that finishes quickly should return its result."""
        agent = FakeAgent(idle_seconds=0.0)
        _cron_inactivity_limit = 10.0
        _POLL_INTERVAL = 0.1

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, "test prompt")
        _inactivity_timeout = False

        result = None
        while True:
            done, _ = concurrent.futures.wait({future}, timeout=_POLL_INTERVAL)
            if done:
                result = future.result()
                break
            _idle_secs = 0.0
            if hasattr(agent, "get_activity_summary"):
                _act = agent.get_activity_summary()
                _idle_secs = _act.get("seconds_since_activity", 0.0)
            if _idle_secs >= _cron_inactivity_limit:
                _inactivity_timeout = True
                break

        pool.shutdown(wait=False)
        assert result is not None
        assert result["final_response"] == "Done"
        assert not _inactivity_timeout
        assert not agent._interrupted

    def test_idle_agent_triggers_timeout(self):
        """An agent that goes idle should be detected and interrupted."""
        # Agent will run for 0.3s, then become idle after 0.1s of that
        agent = SlowFakeAgent(
            run_duration=5.0,  # would run forever without timeout
            idle_after=0.1,    # goes idle almost immediately
            activity_desc="api_call_streaming",
            current_tool="web_search",
            api_call_count=3,
            max_iterations=50,
        )

        _cron_inactivity_limit = 0.5  # 0.5s inactivity triggers timeout
        _POLL_INTERVAL = 0.1

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, "test prompt")
        _inactivity_timeout = False

        result = None
        while True:
            done, _ = concurrent.futures.wait({future}, timeout=_POLL_INTERVAL)
            if done:
                result = future.result()
                break
            _idle_secs = 0.0
            if hasattr(agent, "get_activity_summary"):
                try:
                    _act = agent.get_activity_summary()
                    _idle_secs = _act.get("seconds_since_activity", 0.0)
                except Exception:
                    pass
            if _idle_secs >= _cron_inactivity_limit:
                _inactivity_timeout = True
                break

        pool.shutdown(wait=False, cancel_futures=True)
        assert _inactivity_timeout is True
        assert result is None  # Never got a result — interrupted

    def test_unlimited_timeout(self):
        """HERMES_CRON_TIMEOUT=0 means no timeout at all."""
        agent = FakeAgent(idle_seconds=0.0)
        _cron_inactivity_limit = None  # unlimited

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, "test prompt")

        # With unlimited, we just await the result directly.
        result = future.result()
        pool.shutdown(wait=False)

        assert result["final_response"] == "Done"

    def _parse_cron_timeout(self, raw_value):
        """Mirror the defensive parsing logic from cron/scheduler.py run_job()."""
        if raw_value:
            try:
                return float(raw_value)
            except (ValueError, TypeError):
                return 600.0
        return 600.0

    def test_timeout_env_var_parsing(self, monkeypatch):
        """HERMES_CRON_TIMEOUT env var is respected."""
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "1200")
        raw = os.getenv("HERMES_CRON_TIMEOUT", "").strip()
        _cron_timeout = self._parse_cron_timeout(raw)
        assert _cron_timeout == 1200.0

        _cron_inactivity_limit = _cron_timeout if _cron_timeout > 0 else None
        assert _cron_inactivity_limit == 1200.0

    def test_timeout_zero_means_unlimited(self, monkeypatch):
        """HERMES_CRON_TIMEOUT=0 yields None (unlimited)."""
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
        raw = os.getenv("HERMES_CRON_TIMEOUT", "").strip()
        _cron_timeout = self._parse_cron_timeout(raw)
        _cron_inactivity_limit = _cron_timeout if _cron_timeout > 0 else None
        assert _cron_inactivity_limit is None

    def test_timeout_invalid_value_falls_back_to_default(self, monkeypatch):
        """HERMES_CRON_TIMEOUT=abc should fall back to 600s, not raise ValueError."""
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "abc")
        raw = os.getenv("HERMES_CRON_TIMEOUT", "").strip()
        _cron_timeout = self._parse_cron_timeout(raw)
        assert _cron_timeout == 600.0
        _cron_inactivity_limit = _cron_timeout if _cron_timeout > 0 else None
        assert _cron_inactivity_limit == 600.0

    def test_timeout_empty_string_uses_default(self, monkeypatch):
        """HERMES_CRON_TIMEOUT='' (empty) should use the 600s default."""
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "")
        raw = os.getenv("HERMES_CRON_TIMEOUT", "").strip()
        _cron_timeout = self._parse_cron_timeout(raw)
        assert _cron_timeout == 600.0

    def test_timeout_error_includes_diagnostics(self):
        """The TimeoutError message should include last activity info."""
        agent = SlowFakeAgent(
            run_duration=5.0,
            idle_after=0.05,
            activity_desc="api_call_streaming",
            current_tool="delegate_task",
            api_call_count=7,
            max_iterations=90,
        )

        _cron_inactivity_limit = 0.3
        _POLL_INTERVAL = 0.1

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, "test")
        _inactivity_timeout = False

        while True:
            done, _ = concurrent.futures.wait({future}, timeout=_POLL_INTERVAL)
            if done:
                break
            _idle_secs = 0.0
            if hasattr(agent, "get_activity_summary"):
                try:
                    _act = agent.get_activity_summary()
                    _idle_secs = _act.get("seconds_since_activity", 0.0)
                except Exception:
                    pass
            if _idle_secs >= _cron_inactivity_limit:
                _inactivity_timeout = True
                break

        pool.shutdown(wait=False, cancel_futures=True)
        assert _inactivity_timeout

        # Build the diagnostic message like the scheduler does
        _activity = agent.get_activity_summary()
        _last_desc = _activity.get("last_activity_desc", "unknown")
        _secs_ago = _activity.get("seconds_since_activity", 0)

        err_msg = (
            f"Cron job 'test-job' idle for "
            f"{int(_secs_ago)}s (limit {int(_cron_inactivity_limit)}s) "
            f"— last activity: {_last_desc}"
        )
        assert "idle for" in err_msg
        assert "api_call_streaming" in err_msg

    def test_agent_without_activity_summary_uses_wallclock_fallback(self):
        """If agent lacks get_activity_summary, idle_secs stays 0 (never times out).
        
        This ensures backward compat if somehow an old agent is used.
        The polling loop will eventually complete when the task finishes.
        """
        class BareAgent:
            def run_conversation(self, prompt):
                return {"final_response": "no activity tracker", "messages": []}

        agent = BareAgent()
        _cron_inactivity_limit = 0.1  # tiny limit
        _POLL_INTERVAL = 0.1

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, "test")
        _inactivity_timeout = False

        while True:
            done, _ = concurrent.futures.wait({future}, timeout=_POLL_INTERVAL)
            if done:
                result = future.result()
                break
            _idle_secs = 0.0
            if hasattr(agent, "get_activity_summary"):
                try:
                    _act = agent.get_activity_summary()
                    _idle_secs = _act.get("seconds_since_activity", 0.0)
                except Exception:
                    pass
            if _idle_secs >= _cron_inactivity_limit:
                _inactivity_timeout = True
                break

        pool.shutdown(wait=False)
        # Should NOT have timed out — bare agent has no get_activity_summary
        assert not _inactivity_timeout
        assert result["final_response"] == "no activity tracker"


class TestDeferredCleanupOnStuckWorker:
    """When the worker thread doesn't honor the interrupt within the grace
    window, the scheduler must defer SessionDB/agent close to a done-callback
    so it never closes those resources while the worker is still using them.

    These tests don't drive the full `run_job` machinery — they exercise the
    smaller invariant: a future whose worker stays busy past the grace window
    must still trigger cleanup, but only after the worker actually finishes.
    """

    def test_grace_period_lets_cooperative_worker_finish(self):
        """If the worker exits within the grace window, no deferral is needed."""
        finished = threading.Event()

        def cooperative_worker():
            # Cooperative tool — exits almost immediately after the
            # interrupt is requested.
            time.sleep(0.05)
            finished.set()
            return "ok"

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(cooperative_worker)
        _GRACE_SECONDS = 1.0
        concurrent.futures.wait({future}, timeout=_GRACE_SECONDS)
        pool.shutdown(wait=False)

        assert future.done()
        assert finished.is_set()

    def test_stuck_worker_triggers_done_callback_on_eventual_exit(self):
        """A worker that ignores the interrupt for longer than the grace
        window must NOT have its cleanup run synchronously — it must run via
        add_done_callback once the worker eventually finishes.
        """
        cleanup_ran_before_worker_done = []
        cleanup_ran_after_worker_done = []
        worker_done_at = []

        # Worker that ignores any interrupt for 0.6s — longer than our
        # 0.2s grace window.
        def stuck_worker():
            time.sleep(0.6)
            worker_done_at.append(time.time())
            return "eventually-done"

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(stuck_worker)

        _GRACE_SECONDS = 0.2
        concurrent.futures.wait({future}, timeout=_GRACE_SECONDS)
        assert not future.done(), "worker should still be running after grace"

        callback_ran_at = []

        def _cleanup(_fut):
            callback_ran_at.append(time.time())
            # If the callback fires while the worker is still mid-task,
            # this list would be populated.  It must remain empty.
            if not _fut.done():
                cleanup_ran_before_worker_done.append(True)
            else:
                cleanup_ran_after_worker_done.append(True)

        future.add_done_callback(_cleanup)

        # Now wait for the worker to actually finish.
        future.result(timeout=2.0)
        pool.shutdown(wait=False)

        # Give the callback a brief moment to fire (it runs on the worker
        # thread synchronously after the result is set, so it should
        # already be done — but allow a tiny buffer for slow CI).
        time.sleep(0.1)

        assert not cleanup_ran_before_worker_done, \
            "cleanup callback fired while worker was still running"
        assert cleanup_ran_after_worker_done, \
            "cleanup callback did not fire after worker exit"
        # Callback must fire at or after the worker exit timestamp.
        assert callback_ran_at[0] >= worker_done_at[0]

    def test_callback_close_order_session_then_agent(self):
        """The deferred callback closes SessionDB first, then agent.close.
        Mirrors the synchronous-cleanup ordering in run_job's finally.
        """
        order = []

        class FakeSessionDB:
            def end_session(self, sid, reason):
                order.append(("end_session", sid, reason))
            def close(self):
                order.append("session_close")

        class FakeAgentCloseable:
            def close(self):
                order.append("agent_close")

        sdb = FakeSessionDB()
        ag = FakeAgentCloseable()

        def _finish_deferred_cleanup(_fut, _sdb=sdb, _ag=ag,
                                     _sid="cron_test_123", _jid="jobX"):
            if _sdb is not None:
                try:
                    _sdb.end_session(_sid, "cron_timeout")
                except Exception:
                    pass
                try:
                    _sdb.close()
                except Exception:
                    pass
            if _ag is not None:
                try:
                    _ag.close()
                except Exception:
                    pass

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(lambda: time.sleep(0.05) or "done")
        future.add_done_callback(_finish_deferred_cleanup)
        future.result(timeout=2.0)
        pool.shutdown(wait=False)
        time.sleep(0.05)

        assert order[0][0] == "end_session"
        assert order[0][2] == "cron_timeout"
        assert order[1] == "session_close"
        assert order[2] == "agent_close"


class TestSysPathOrdering:
    """Test that sys.path is set before repo-level imports."""

    def test_hermes_time_importable(self):
        """hermes_time should be importable when cron.scheduler loads."""
        # This import would fail if sys.path.insert comes after the import
        from cron.scheduler import _hermes_now
        assert callable(_hermes_now)

    def test_hermes_constants_importable(self):
        """hermes_constants should be importable from cron context."""
        from hermes_constants import get_hermes_home
        assert callable(get_hermes_home)


class TestRunJobDeferredCleanupIntegration:
    """End-to-end regression test that drives ``run_job()`` itself.

    The unit tests above exercise the grace-window / done-callback mechanics
    in isolation. This class wires a stuck fake agent into the real
    ``scheduler.run_job()`` path and asserts that the inactivity timeout +
    deferred cleanup contract holds when the whole function is executed:

    1. ``run_job`` raises ``TimeoutError`` and returns a failure tuple as
       soon as the inactivity limit trips, **without waiting** for the
       stuck worker to finish.
    2. At the moment ``run_job`` returns, neither ``SessionDB.close`` /
       ``SessionDB.end_session`` nor ``AIAgent.close`` has run yet — the
       worker is still alive and using both resources.
    3. Once the worker eventually exits, the deferred done-callback fires
       and closes SessionDB (with reason ``cron_timeout``) before
       ``agent.close()``.
    """

    @pytest.fixture(autouse=True)
    def _stub_runtime_provider(self):
        """Stub provider resolution so the test runs in a hermetic CI env
        without API keys. ``run_job`` resolves the runtime provider before
        building ``AIAgent`` — without this stub the resolver raises.
        """
        fake_runtime = {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "source": "stub",
            "requested_provider": None,
        }
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ):
            yield

    @pytest.fixture(autouse=True)
    def _short_grace_and_poll(self, monkeypatch):
        """Crunch the inactivity-poll and interrupt-grace windows down so the
        test runs in well under a second. These are now module-level
        constants on ``cron.scheduler`` precisely so tests can override
        them without re-entering ``run_job`` every iteration.
        """
        import cron.scheduler as scheduler
        monkeypatch.setattr(scheduler, "_POLL_INTERVAL", 0.05)
        monkeypatch.setattr(scheduler, "_GRACE_SECONDS", 0.1)

    @pytest.fixture(autouse=True)
    def _short_inactivity_timeout(self, monkeypatch):
        """Trip the inactivity timeout almost immediately."""
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0.1")

    def _make_job(self):
        return {
            "id": "job_deferred_cleanup_integration",
            "name": "deferred-cleanup-integration",
            "prompt": "stall forever please",
            "schedule": "*/5 * * * *",
        }

    def test_run_job_defers_cleanup_when_worker_ignores_interrupt(self):
        """Real ``run_job()`` path with a stuck fake agent: the function
        must return a timeout failure while the worker is still mid-task,
        and the deferred done-callback must close SessionDB + agent only
        after the worker eventually exits.
        """
        import cron.scheduler as scheduler

        # ----- observability counters -----
        close_order: list[str] = []
        worker_exit_time: list[float] = []
        # When did run_job return relative to worker exit?
        run_job_returned_at: list[float] = []

        # ----- fake AIAgent that ignores interrupts -----
        worker_done = threading.Event()
        worker_release = threading.Event()  # set after run_job returns

        class StuckFakeAgent:
            def __init__(self, *args, **kwargs):
                self._idle_since = time.time()
                self._closed = False

            def interrupt(self, message: str = None) -> None:
                # Deliberately ignore — simulate a tool that doesn't honor
                # the cooperative interrupt flag.
                pass

            def get_activity_summary(self) -> dict:
                return {
                    "seconds_since_activity": time.time() - self._idle_since,
                    "last_activity_desc": "stuck_tool_call",
                    "current_tool": "fake_tool",
                    "api_call_count": 1,
                    "max_iterations": 90,
                }

            def run_conversation(self, *args, **kwargs):
                # Block until the outer test releases us — i.e. until AFTER
                # run_job has already returned its TimeoutError tuple. This
                # is what makes the deferred-cleanup path the only viable
                # one.
                worker_release.wait(timeout=5.0)
                worker_exit_time.append(time.time())
                worker_done.set()
                return {"final_response": "late", "messages": [], "completed": True}

            def close(self) -> None:
                self._closed = True
                close_order.append("agent_close")

        # ----- fake SessionDB -----
        class FakeSessionDB:
            def __init__(self):
                self._closed = False

            def end_session(self, sid, reason):
                close_order.append(f"end_session:{reason}")

            def close(self):
                self._closed = True
                close_order.append("session_close")

            # SessionDB is used as a kwarg passed into AIAgent; the real
            # class supports a few methods that run_job may touch. We only
            # need end_session/close for the deferred path. Provide a
            # permissive __getattr__ so any incidental method call is a
            # no-op rather than an AttributeError.
            def __getattr__(self, name):
                return lambda *a, **kw: None

        fake_sdb_instance = FakeSessionDB()

        with patch("run_agent.AIAgent", StuckFakeAgent), \
             patch("hermes_state.SessionDB", return_value=fake_sdb_instance):
            t0 = time.time()
            success, doc, final_response, error = scheduler.run_job(self._make_job())
            run_job_returned_at.append(time.time())

        # --- Assertion 1: run_job returns a timeout failure tuple. ---
        assert success is False
        assert error is not None
        assert "idle" in error.lower() or "timeout" in error.lower(), (
            f"expected an inactivity-timeout error, got: {error!r}"
        )

        # --- Assertion 2: run_job returned BEFORE the worker exited. ---
        # Worker hasn't been released yet, so it must still be alive.
        assert not worker_done.is_set(), (
            "worker exited before run_job returned — deferred path was "
            "not exercised (this regression would re-introduce the "
            "SessionDB use-after-close window)"
        )
        # And cleanup must not have run synchronously inside the finally.
        assert close_order == [], (
            f"sync cleanup ran while worker still in-flight: {close_order}"
        )

        # --- Now release the worker and let the done-callback fire. ---
        worker_release.set()
        # Wait for worker exit + callback. The callback runs synchronously
        # on the worker thread after the future is resolved, so a small
        # buffer is enough.
        assert worker_done.wait(timeout=3.0), "fake worker never exited"
        # Poll until the callback ran (or timeout).
        deadline = time.time() + 2.0
        while time.time() < deadline and not close_order:
            time.sleep(0.02)

        # --- Assertion 3: deferred callback ran AFTER worker exit, with
        #     the correct ordering: end_session(cron_timeout) -> session
        #     close -> agent close. ---
        assert close_order, "deferred cleanup callback never fired"
        assert close_order[0] == "end_session:cron_timeout", (
            f"first cleanup step should be end_session with cron_timeout "
            f"reason; got {close_order!r}"
        )
        assert close_order.index("session_close") < close_order.index("agent_close"), (
            f"SessionDB must close before agent; got {close_order!r}"
        )

        # --- Assertion 4: timing — run_job returned strictly before the
        #     worker's recorded exit timestamp. ---
        assert run_job_returned_at[0] < worker_exit_time[0], (
            "run_job did not return before the stuck worker exited"
        )
