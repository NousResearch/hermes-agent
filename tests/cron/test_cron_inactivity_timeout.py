"""Tests for cron job inactivity-based timeout.

Tests cover:
- Active agent runs indefinitely (no inactivity timeout)
- Idle agent triggers inactivity timeout with diagnostic info
- Unlimited timeout (HERMES_CRON_TIMEOUT=0)
- Backward compat: HERMES_CRON_TIMEOUT env var still works
- Error message includes activity summary
"""

import concurrent.futures
import asyncio
import os
import sys
import threading
import time
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

    def test_timeout_waits_for_worker_exit_while_heartbeating(self):
        """A timed-out worker keeps ownership until it acknowledges termination."""
        from cron.scheduler import _wait_for_cron_worker_termination

        release_worker = threading.Event()
        waiter_finished = threading.Event()
        heartbeat_seen = threading.Event()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(release_worker.wait)

        def wait_for_exit() -> None:
            try:
                _wait_for_cron_worker_termination(
                    future,
                    heartbeat=lambda: heartbeat_seen.set(),
                    poll_interval=0.01,
                )
            finally:
                waiter_finished.set()

        waiter = threading.Thread(target=wait_for_exit)
        waiter.start()
        assert heartbeat_seen.wait(timeout=1)
        assert not waiter_finished.is_set()

        release_worker.set()
        waiter.join(timeout=1)
        pool.shutdown(wait=True)
        assert waiter_finished.is_set()

    def test_timeout_wait_contains_cancelled_error(self):
        """Cancellation is an expected timeout acknowledgment, not an escape."""
        from cron.scheduler import _wait_for_cron_worker_termination

        future = concurrent.futures.Future()
        future.set_exception(asyncio.CancelledError())
        _wait_for_cron_worker_termination(
            future,
            heartbeat=lambda: None,
            poll_interval=0.01,
        )

    def test_timeout_wait_does_not_swallow_process_interrupts(self):
        """KeyboardInterrupt/SystemExit retain their control-flow semantics."""
        import pytest

        from cron.scheduler import _wait_for_cron_worker_termination

        for escape in (KeyboardInterrupt(), SystemExit()):
            future = concurrent.futures.Future()
            future.set_exception(escape)
            with pytest.raises(type(escape)):
                _wait_for_cron_worker_termination(
                    future,
                    heartbeat=lambda: None,
                    poll_interval=0.01,
                )

    def test_fire_claim_loss_interrupts_active_agent_promptly(
        self, tmp_path, monkeypatch
    ):
        """A failed heartbeat CAS terminates stale agent work, not just delivery."""
        import cron.scheduler as scheduler

        class ClaimLossAgent(FakeAgent):
            started_at = None
            interrupted_at = None

            def run_conversation(self, prompt):
                self.started_at = time.monotonic()
                deadline = time.monotonic() + 0.5
                while not self._interrupted and time.monotonic() < deadline:
                    time.sleep(0.005)
                return {"final_response": "stale result", "messages": []}

            def interrupt(self, reason=None):
                self.interrupted_at = time.monotonic()
                super().interrupt(reason)

        agent = ClaimLossAgent()
        runtime = {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
        }
        job = {
            "id": "lost-agent-owner",
            "name": "lost agent owner",
            "prompt": "do work",
            "model": "test/model",
            "_fire_claim_id": "stale-token",
        }
        monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.0)
        monkeypatch.setattr(scheduler, "_CRON_AGENT_POLL_SECONDS", 0.01)
        monkeypatch.setattr(scheduler, "heartbeat_fire_claim", lambda *a, **k: False)

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=MagicMock()), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value=runtime,
             ), \
             patch("run_agent.AIAgent", return_value=agent):
            success, _output, _response, error = scheduler.run_job(job)
        assert success is False
        assert "ownership" in error.lower()
        assert agent._interrupted is True
        assert agent.started_at is not None
        assert agent.interrupted_at is not None
        assert agent.interrupted_at - agent.started_at < 0.1

    def test_windows_worker_termination_uses_taskkill_tree(self, monkeypatch):
        """Windows worker shutdown kills descendants, not only the direct child."""
        import cron.scheduler as scheduler

        process = MagicMock()
        process.pid = 4242
        process.poll.return_value = None
        process.wait.return_value = 0
        taskkill_result = MagicMock(returncode=0)
        monkeypatch.setattr(scheduler.os, "name", "nt")

        with patch(
            "cron.scheduler.subprocess.run",
            return_value=taskkill_result,
        ) as taskkill:
            scheduler._terminate_cron_worker(process)

        taskkill.assert_called_once_with(
            ["taskkill", "/PID", "4242", "/T", "/F"],
            stdout=scheduler.subprocess.DEVNULL,
            stderr=scheduler.subprocess.DEVNULL,
            check=False,
            timeout=scheduler._CRON_WORKER_TERMINATE_GRACE_SECONDS,
        )
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_windows_worker_taskkill_failure_falls_back_to_direct_terminate(
        self,
        monkeypatch,
    ):
        """A failed Windows tree kill still terminates and reaps the worker."""
        import cron.scheduler as scheduler

        process = MagicMock()
        process.pid = 4242
        process.poll.return_value = None
        process.wait.return_value = 0
        monkeypatch.setattr(scheduler.os, "name", "nt")

        with patch(
            "cron.scheduler.subprocess.run",
            return_value=MagicMock(returncode=1),
        ):
            scheduler._terminate_cron_worker(process)

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(
            timeout=scheduler._CRON_WORKER_TERMINATE_GRACE_SECONDS,
        )

    def test_windows_worker_taskkill_timeout_falls_back_to_direct_terminate(
        self,
        monkeypatch,
    ):
        """A wedged taskkill cannot wedge scheduler shutdown."""
        import cron.scheduler as scheduler

        process = MagicMock()
        process.pid = 4242
        process.poll.return_value = None
        process.wait.return_value = 0
        monkeypatch.setattr(scheduler.os, "name", "nt")

        with patch(
            "cron.scheduler.subprocess.run",
            side_effect=scheduler.subprocess.TimeoutExpired("taskkill", 3),
        ):
            scheduler._terminate_cron_worker(process)

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(
            timeout=scheduler._CRON_WORKER_TERMINATE_GRACE_SECONDS,
        )

    def test_agent_activity_summary_refreshes_worker_pulse(self, tmp_path):
        """The parent watchdog observes the real AIAgent activity API."""
        import cron.scheduler as scheduler

        class ActivityAgent:
            def __init__(self):
                self.last_activity = 10.0

            def get_activity_summary(self):
                return {"last_activity_ts": self.last_activity}

        agent = ActivityAgent()
        pulse = tmp_path / "agent.pulse"

        observed = scheduler._refresh_cron_worker_pulse_from_agent(
            agent,
            str(pulse),
            None,
        )
        assert observed == 10.0
        assert pulse.exists()

        pulse.unlink()
        assert scheduler._refresh_cron_worker_pulse_from_agent(
            agent,
            str(pulse),
            observed,
        ) == 10.0
        assert not pulse.exists()

        agent.last_activity = 11.0
        assert scheduler._refresh_cron_worker_pulse_from_agent(
            agent,
            str(pulse),
            observed,
        ) == 11.0
        assert pulse.exists()

    @pytest.mark.live_system_guard_bypass
    def test_killable_worker_kills_detached_descendant(
        self,
        monkeypatch,
        tmp_path,
    ):
        """Hard timeout snapshots and kills descendants that create a new session."""
        import cron.scheduler as scheduler

        marker = tmp_path / "detached-survived"
        pulse_env = "HERMES_CRON_WORKER_PULSE"
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0.05")
        monkeypatch.setattr(scheduler, "_CRON_WORKER_POLL_SECONDS", 0.01)
        monkeypatch.setattr(
            scheduler,
            "_CRON_WORKER_TERMINATE_GRACE_SECONDS",
            0.03,
        )

        child_code = (
            "import pathlib,signal,time;"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "time.sleep(0.25);"
            f"pathlib.Path({str(marker)!r}).write_text('survived')"
        )
        worker_code = (
            "import os,pathlib,subprocess,sys,time;"
            "sys.stdin.read();"
            f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
            "start_new_session=True);"
            f"pathlib.Path(os.environ[{pulse_env!r}]).touch();"
            "time.sleep(5)"
        )

        monkeypatch.setattr(
            scheduler,
            "_cron_worker_command",
            lambda: [sys.executable, "-c", worker_code],
        )

        result = scheduler._run_job_in_killable_process(
            {"id": "tree-timeout-detached"}
        )

        assert result[0] is False
        assert "timed out" in (result[3] or "").lower()
        time.sleep(0.35)
        assert not marker.exists()

    def test_killable_worker_acknowledges_hard_timeout_before_return(
        self,
        monkeypatch,
        tmp_path,
    ):
        """An uncooperative worker is process-killed before ownership can clear."""
        import cron.scheduler as scheduler

        marker = tmp_path / "worker-survived"
        pulse_env = "HERMES_CRON_WORKER_PULSE"
        code = (
            "import os,pathlib,signal,sys,time;"
            "sys.stdin.read();"
            f"pathlib.Path(os.environ[{pulse_env!r}]).touch();"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "time.sleep(0.4);"
            "pathlib.Path(os.environ['CRON_TEST_SURVIVAL_MARKER']).write_text('bad')"
        )
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0.05")
        monkeypatch.setenv("CRON_TEST_SURVIVAL_MARKER", str(marker))
        monkeypatch.setattr(scheduler, "_CRON_WORKER_POLL_SECONDS", 0.01)
        monkeypatch.setattr(scheduler, "_CRON_WORKER_TERMINATE_GRACE_SECONDS", 0.03)
        monkeypatch.setattr(
            scheduler,
            "_cron_worker_command",
            lambda: [sys.executable, "-c", code],
        )

        started = time.monotonic()
        success, output, response, error = scheduler._run_job_in_killable_process(
            {"id": "hard-timeout", "name": "hard-timeout", "prompt": "hang"}
        )
        elapsed = time.monotonic() - started

        assert success is False
        assert output == ""
        assert response == ""
        assert error is not None
        assert "inactivity" in error.lower()
        assert elapsed < 0.3
        time.sleep(0.2)
        assert not marker.exists()

    def test_killable_worker_kills_descendant_after_leader_exits(
        self,
        monkeypatch,
        tmp_path,
    ):
        """A cooperative leader cannot leave a SIGTERM-ignoring child behind."""
        import cron.scheduler as scheduler

        marker = tmp_path / "descendant-survived"
        pulse_env = "HERMES_CRON_WORKER_PULSE"
        descendant = (
            "import pathlib,signal,time;"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "time.sleep(0.25);"
            f"pathlib.Path({str(marker)!r}).write_text('bad')"
        )
        code = (
            "import os,pathlib,subprocess,sys,time;"
            "sys.stdin.read();"
            f"pathlib.Path(os.environ[{pulse_env!r}]).touch();"
            f"subprocess.Popen([sys.executable,'-c',{descendant!r}]);"
            "time.sleep(5)"
        )
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0.05")
        monkeypatch.setattr(scheduler, "_CRON_WORKER_POLL_SECONDS", 0.01)
        monkeypatch.setattr(scheduler, "_CRON_WORKER_TERMINATE_GRACE_SECONDS", 0.03)
        monkeypatch.setattr(
            scheduler,
            "_cron_worker_command",
            lambda: [sys.executable, "-c", code],
        )

        result = scheduler._run_job_in_killable_process(
            {"id": "descendant-timeout", "name": "descendant", "prompt": "hang"}
        )

        assert result[0] is False
        time.sleep(0.3)
        assert not marker.exists()

    def test_killable_worker_returns_nonce_bound_pipe_result(
        self,
        monkeypatch,
    ):
        """Normal workers return structured results without filesystem content."""
        import cron.scheduler as scheduler

        code = (
            "import json,os,pathlib,sys;"
            "request=json.loads(sys.stdin.read());"
            "pathlib.Path(os.environ['HERMES_CRON_WORKER_PULSE']).touch();"
            "result={'result':[True,'doc','response',None]};"
            "print('__HERMES_CRON_RESULT__'+request['nonce']+':'"
            "+json.dumps(result))"
        )
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "1")
        monkeypatch.setattr(scheduler, "_CRON_WORKER_POLL_SECONDS", 0.01)
        monkeypatch.setattr(
            scheduler,
            "_cron_worker_command",
            lambda: [sys.executable, "-c", code],
        )

        assert scheduler._run_job_in_killable_process(
            {"id": "normal-worker", "name": "normal", "prompt": "run"}
        ) == (True, "doc", "response", None)

    def test_default_worker_module_is_profile_scoped(self, monkeypatch, tmp_path):
        """The real child entrypoint runs against the selected profile store."""
        import cron.scheduler as scheduler
        from cron.jobs import use_cron_store

        home = tmp_path / "profile"
        scripts = home / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "worker_smoke.py").write_text("print('profile-worker-ok')\n")
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "2")

        with use_cron_store(home):
            success, _doc, response, error = scheduler._run_job_in_killable_process(
                {
                    "id": "profile-worker",
                    "name": "profile-worker",
                    "script": "worker_smoke.py",
                    "no_agent": True,
                }
            )

        assert success is True
        assert response == "profile-worker-ok"
        assert error is None
        assert not (home / "cron" / ".workers").exists()

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


class TestSysPathOrdering:
    """Test that sys.path is set before repo-level imports."""

    def test_hermes_time_importable(self):
        """hermes_time should be importable when cron.scheduler loads."""
        # This import would fail if sys.path.insert comes after the import
        from cron.scheduler import _hermes_now
        assert callable(_hermes_now)

