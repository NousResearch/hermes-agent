"""Tests for the cron inactivity watchdog loop (runs on its own daemon thread)."""

import concurrent.futures
import threading
import time

import pytest


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



class _DrainingAgent:
    """Stand-in whose ``run_conversation`` keeps running after the hard interrupt.

    ``drain_s`` is how long the worker needs to wind down once interrupted
    (the in-flight model call aborting); ``None`` never exits on its own.
    """

    def __init__(self, drain_s):
        self.drain_s = drain_s
        self.interrupted = threading.Event()
        self.exited = threading.Event()
        self.release = threading.Event()

    def get_activity_summary(self):
        return {"seconds_since_activity": 9999.0, "last_activity_desc": "streaming"}

    def interrupt(self, message=None):
        self.interrupted.set()

    def run_conversation(self, prompt, task_id=None):
        try:
            self.interrupted.wait(10.0)
            if self.drain_s is None:
                self.release.wait(10.0)
            else:
                time.sleep(self.drain_s)
            return {"final_response": ""}
        finally:
            self.exited.set()


@pytest.fixture
def fast_watchdog(monkeypatch):
    """Fire the inactivity limit immediately and shrink the 5s monitor poll."""
    import cron.scheduler as sched

    real_wait = concurrent.futures.wait

    def _short_wait(fs, timeout=None, return_when=concurrent.futures.ALL_COMPLETED):
        timeout = 0.02 if timeout is None else min(timeout, 0.02)
        return real_wait(fs, timeout=timeout, return_when=return_when)

    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "600")
    monkeypatch.setattr(sched, "load_config", lambda: {})
    monkeypatch.setattr(sched, "_inactivity_watchdog_loop", lambda **_kw: True)
    monkeypatch.setattr(sched.concurrent.futures, "wait", _short_wait)
    return sched


def _run_watchdog(sched, agent, job=None):
    job = job or {"id": "grace-job", "name": "grace"}
    return sched._run_agent_with_watchdog(
        agent, "hello", job, job["id"], job["name"], "task", None, worker_state={})


class TestInactivityCancellationGrace:
    """After the hard interrupt, run_job must not return (and release the job's parallel
    slot) while the interrupted worker is still inside ``run_conversation``."""

    def test_inactivity_interrupt_waits_for_worker_exit(self, fast_watchdog, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_CANCEL_GRACE", "10")
        agent = _DrainingAgent(drain_s=0.3)

        with pytest.raises(TimeoutError, match="idle for"):
            _run_watchdog(fast_watchdog, agent)

        assert agent.interrupted.is_set()
        # The worker had finished draining before the timeout was raised.
        assert agent.exited.is_set()

    def test_never_exiting_worker_still_raises_after_grace(self, fast_watchdog, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_CANCEL_GRACE", "0.3")
        agent = _DrainingAgent(drain_s=None)
        try:
            started = time.monotonic()
            with pytest.raises(TimeoutError, match="idle for"):
                _run_watchdog(fast_watchdog, agent)
            elapsed = time.monotonic() - started

            assert agent.interrupted.is_set()
            assert not agent.exited.is_set()
            assert 0.3 <= elapsed < 5.0
        finally:
            agent.release.set()
            assert agent.exited.wait(5.0)

    def test_zero_grace_keeps_immediate_raise(self, fast_watchdog, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_CANCEL_GRACE", "0")
        agent = _DrainingAgent(drain_s=None)
        try:
            with pytest.raises(TimeoutError, match="idle for"):
                _run_watchdog(fast_watchdog, agent)
            assert agent.interrupted.is_set()
            assert not agent.exited.is_set()
        finally:
            agent.release.set()
            assert agent.exited.wait(5.0)

    def test_oneshot_run_claim_is_heartbeated_during_grace(self, fast_watchdog, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_CANCEL_GRACE", "10")
        monkeypatch.setattr(fast_watchdog, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.0)
        agent = _DrainingAgent(drain_s=0.3)
        beats_after_interrupt = []

        def _heartbeat(job_id, *, expected_owner):
            if agent.interrupted.is_set():
                beats_after_interrupt.append((job_id, expected_owner))
            return True

        monkeypatch.setattr(fast_watchdog, "heartbeat_run_claim", _heartbeat)
        job = {
            "id": "grace-oneshot",
            "name": "grace",
            "schedule": {"kind": "once", "run_at": "2026-07-10T12:00:00Z"},
            "run_claim": {"at": "2026-07-10T12:00:00Z", "by": "owner-token"},
        }

        with pytest.raises(TimeoutError):
            _run_watchdog(fast_watchdog, agent, job)

        assert agent.exited.is_set()
        assert beats_after_interrupt
        assert set(beats_after_interrupt) == {("grace-oneshot", "owner-token")}


class TestCronCancelGraceResolution:
    @pytest.fixture(autouse=True)
    def _no_config(self, monkeypatch):
        monkeypatch.delenv("HERMES_CRON_CANCEL_GRACE", raising=False)
        monkeypatch.setattr("cron.scheduler.load_config", lambda: {})

    def test_default(self):
        from cron.scheduler_script import _DEFAULT_CANCEL_GRACE_SECONDS, _get_cancel_grace_seconds

        assert _get_cancel_grace_seconds() == _DEFAULT_CANCEL_GRACE_SECONDS

    def test_default_matches_config_defaults(self):
        from cron.scheduler_script import _DEFAULT_CANCEL_GRACE_SECONDS
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        cron_defaults = DEFAULT_CONFIG["cron"]
        assert isinstance(cron_defaults, dict)
        assert cron_defaults["cancel_grace_seconds"] == _DEFAULT_CANCEL_GRACE_SECONDS

    def test_env_value_and_zero_disables(self, monkeypatch):
        from cron.scheduler_script import _get_cancel_grace_seconds

        monkeypatch.setenv("HERMES_CRON_CANCEL_GRACE", "12.5")
        assert _get_cancel_grace_seconds() == 12.5
        monkeypatch.setenv("HERMES_CRON_CANCEL_GRACE", "0")
        assert _get_cancel_grace_seconds() == 0.0

    def test_env_wins_over_config(self, monkeypatch):
        from cron.scheduler_script import _get_cancel_grace_seconds

        monkeypatch.setenv("HERMES_CRON_CANCEL_GRACE", "5")
        monkeypatch.setattr(
            "cron.scheduler.load_config", lambda: {"cron": {"cancel_grace_seconds": 90}})
        assert _get_cancel_grace_seconds() == 5.0

    def test_config_value(self, monkeypatch):
        from cron.scheduler_script import _get_cancel_grace_seconds

        monkeypatch.setattr(
            "cron.scheduler.load_config", lambda: {"cron": {"cancel_grace_seconds": 90}})
        assert _get_cancel_grace_seconds() == 90.0

    @pytest.mark.parametrize("raw", ["-5", "abc", "nan", "inf", "-inf"])
    def test_invalid_env_falls_back_to_config(self, monkeypatch, raw):
        from cron.scheduler_script import _get_cancel_grace_seconds

        monkeypatch.setenv("HERMES_CRON_CANCEL_GRACE", raw)
        monkeypatch.setattr(
            "cron.scheduler.load_config", lambda: {"cron": {"cancel_grace_seconds": 7}})
        assert _get_cancel_grace_seconds() == 7.0

    @pytest.mark.parametrize("raw", [-1, True, "soon", float("nan")])
    def test_invalid_config_uses_default(self, monkeypatch, raw):
        from cron.scheduler_script import _DEFAULT_CANCEL_GRACE_SECONDS, _get_cancel_grace_seconds

        monkeypatch.setattr(
            "cron.scheduler.load_config", lambda: {"cron": {"cancel_grace_seconds": raw}})
        assert _get_cancel_grace_seconds() == _DEFAULT_CANCEL_GRACE_SECONDS
