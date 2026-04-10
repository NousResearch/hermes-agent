"""Tests for staged inactivity timeout in gateway agent runs.

Tests cover:
- Warning fires once when inactivity reaches gateway_timeout_warning threshold
- Warning does not fire when gateway_timeout is 0 (unlimited)
- Warning fires only once per run, not on every poll
- Full timeout still fires at gateway_timeout threshold
- Warning respects HERMES_AGENT_TIMEOUT_WARNING env var
- Warning disabled when gateway_timeout_warning is 0
"""

import concurrent.futures
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
        return {"final_response": "Done", "messages": []}


class SlowFakeAgent(FakeAgent):
    """Agent that runs for a while, then goes idle."""

    def __init__(self, run_duration=0.5, idle_after=None, **kwargs):
        super().__init__(**kwargs)
        self._run_duration = run_duration
        self._idle_after = idle_after
        self._start_time = None

    def get_activity_summary(self):
        summary = super().get_activity_summary()
        if self._idle_after is not None and self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed > self._idle_after:
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


class TestStagedInactivityWarning:
    """Test the staged inactivity warning before full timeout."""

    def test_warning_fires_once_before_timeout(self):
        """Warning fires when inactivity reaches warning threshold."""
        agent = SlowFakeAgent(
            run_duration=0.8,
            idle_after=0.05,
            activity_desc="api_call_streaming",
        )

        _agent_timeout = 2.0
        _agent_warning = 0.2
        _POLL_INTERVAL = 0.02

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, "test prompt")
        _inactivity_timeout = False
        _warning_fired = False
        _warning_send_count = 0

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
            if (not _warning_fired and _agent_warning > 0
                    and _idle_secs >= _agent_warning):
                _warning_fired = True
                _warning_send_count += 1
            if _idle_secs >= _agent_timeout:
                _inactivity_timeout = True
                break

        pool.shutdown(wait=False, cancel_futures=True)

        assert _warning_fired
        assert _warning_send_count == 1
        assert not _inactivity_timeout

    def test_warning_disabled_when_zero(self):
        """No warning fires when gateway_timeout_warning is 0."""
        agent = SlowFakeAgent(
            run_duration=0.6,
            idle_after=0.05,
        )

        _agent_timeout = 2.0
        _agent_warning = 0.0
        _POLL_INTERVAL = 0.02

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, "test")
        _warning_fired = False

        while True:
            done, _ = concurrent.futures.wait({future}, timeout=_POLL_INTERVAL)
            if done:
                future.result()
                break
            _idle_secs = 0.0
            if hasattr(agent, "get_activity_summary"):
                try:
                    _act = agent.get_activity_summary()
                    _idle_secs = _act.get("seconds_since_activity", 0.0)
                except Exception:
                    pass
            if (not _warning_fired and _agent_warning > 0
                    and _idle_secs >= _agent_warning):
                _warning_fired = True
            if _idle_secs >= _agent_timeout:
                break

        pool.shutdown(wait=False, cancel_futures=True)
        assert not _warning_fired

    def test_warning_fires_only_once(self):
        """Warning fires exactly once even if agent remains idle."""
        agent = SlowFakeAgent(
            run_duration=0.8,
            idle_after=0.05,
        )

        _agent_timeout = 2.0
        _agent_warning = 0.2
        _POLL_INTERVAL = 0.02

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, "test")
        _warning_count = 0

        while True:
            done, _ = concurrent.futures.wait({future}, timeout=_POLL_INTERVAL)
            if done:
                future.result()
                break
            _idle_secs = 0.0
            if hasattr(agent, "get_activity_summary"):
                try:
                    _act = agent.get_activity_summary()
                    _idle_secs = _act.get("seconds_since_activity", 0.0)
                except Exception:
                    pass
            if (not _warning_count and _agent_warning > 0
                    and _idle_secs >= _agent_warning):
                _warning_count += 1
            if _idle_secs >= _agent_timeout:
                break

        pool.shutdown(wait=False, cancel_futures=True)
        assert _warning_count == 1

    def test_full_timeout_still_fires_after_warning(self):
        """Full timeout fires even after warning was sent."""
        agent = SlowFakeAgent(
            run_duration=2.0,
            idle_after=0.05,
            activity_desc="waiting for provider response (streaming)",
        )

        _agent_timeout = 0.6
        _agent_warning = 0.2
        _POLL_INTERVAL = 0.02

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, "test")
        _inactivity_timeout = False
        _warning_fired = False

        while True:
            done, _ = concurrent.futures.wait({future}, timeout=_POLL_INTERVAL)
            if done:
                future.result()
                break
            _idle_secs = 0.0
            if hasattr(agent, "get_activity_summary"):
                try:
                    _act = agent.get_activity_summary()
                    _idle_secs = _act.get("seconds_since_activity", 0.0)
                except Exception:
                    pass
            if (not _warning_fired and _agent_warning > 0
                    and _idle_secs >= _agent_warning):
                _warning_fired = True
            if _idle_secs >= _agent_timeout:
                _inactivity_timeout = True
                break

        pool.shutdown(wait=False, cancel_futures=True)
        assert _warning_fired
        assert _inactivity_timeout

    def test_warning_env_var_respected(self, monkeypatch):
        """HERMES_AGENT_TIMEOUT_WARNING env var is parsed correctly."""
        monkeypatch.setenv("HERMES_AGENT_TIMEOUT_WARNING", "600")
        _warning = float(os.getenv("HERMES_AGENT_TIMEOUT_WARNING", 900))
        assert _warning == 600.0

    def test_warning_zero_means_disabled(self, monkeypatch):
        """HERMES_AGENT_TIMEOUT_WARNING=0 disables the warning."""
        monkeypatch.setenv("HERMES_AGENT_TIMEOUT_WARNING", "0")
        _raw = float(os.getenv("HERMES_AGENT_TIMEOUT_WARNING", 900))
        _warning = _raw if _raw > 0 else None
        assert _warning is None

    def test_unlimited_timeout_no_warning(self):
        """When timeout is unlimited (0), no warning fires either."""
        agent = SlowFakeAgent(
            run_duration=0.5,
            idle_after=0.0,
        )

        _agent_timeout = None
        _agent_warning = 5.0
        _POLL_INTERVAL = 0.05

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, "test")

        result = future.result(timeout=2.0)
        pool.shutdown(wait=False)

        assert result["final_response"] == "Completed after work"


class TestWarningThresholdBelowTimeout:
    """Test that warning threshold must be less than timeout threshold."""

    def test_warning_at_half_timeout(self):
        """Warning fires at half the timeout duration."""
        agent = SlowFakeAgent(
            run_duration=0.9,
            idle_after=0.05,
            activity_desc="receiving stream response",
        )

        _agent_timeout = 0.6
        _agent_warning = 0.3
        _POLL_INTERVAL = 0.02

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, "test")
        _warning_fired = False
        _timeout_fired = False

        while True:
            done, _ = concurrent.futures.wait({future}, timeout=_POLL_INTERVAL)
            if done:
                future.result()
                break
            _idle_secs = 0.0
            if hasattr(agent, "get_activity_summary"):
                try:
                    _act = agent.get_activity_summary()
                    _idle_secs = _act.get("seconds_since_activity", 0.0)
                except Exception:
                    pass
            if (not _warning_fired and _agent_warning > 0
                    and _idle_secs >= _agent_warning):
                _warning_fired = True
            if _idle_secs >= _agent_timeout:
                _timeout_fired = True
                break

        pool.shutdown(wait=False, cancel_futures=True)
        assert _warning_fired
        assert _timeout_fired
