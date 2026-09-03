"""Tests for staged inactivity timeout in gateway agent runs.

Tests cover:
- Warning fires once when inactivity reaches gateway_timeout_warning threshold
- Warning does not fire when gateway_timeout is 0 (unlimited)
- Warning fires only once per run, not on every poll
- Full timeout still fires at gateway_timeout threshold
- Warning respects HERMES_AGENT_TIMEOUT_WARNING env var
- Warning disabled when gateway_timeout_warning is 0
"""

import asyncio
import concurrent.futures
import os
import sys
import threading
import time
import types
from pathlib import Path


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
            run_duration=0.6,
            idle_after=0.05,
            activity_desc="api_call_streaming",
        )

        _agent_timeout = 20.0
        _agent_warning = 0.15
        _POLL_INTERVAL = 0.05

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



    def test_full_timeout_still_fires_after_warning(self):
        """Full timeout fires even after warning was sent."""
        agent = SlowFakeAgent(
            run_duration=5.0,
            idle_after=0.05,
            activity_desc="waiting for provider response (streaming)",
        )

        _agent_timeout = 0.4
        _agent_warning = 0.15
        _POLL_INTERVAL = 0.05

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


class TestExtendCommandHandler:
    """Tests for /extend command parsing and storage (Gap 2)."""

    def test_extend_command_sets_deadline(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._inactivity_extend_deadlines = {}
        runner._inactivity_extend_deadlines_lock = threading.Lock()
        runner._running_agents = {"session:c1": object()}
        runner._session_run_generation = {"session:c1": 7}
        # Minimal stand-ins so the handler runs without a full gateway.
        class _Source:
            platform = "cli"
            chat_id = "c1"
        class _Event:
            def __init__(self, args):
                self.source = _Source()
                self._args = args
            def get_command_args(self):
                return self._args
        monkeypatch.setattr(
            runner, "_session_key_for_source",
            lambda src: f"session:{src.chat_id}",
        )
        # Patch time.monotonic to a fixed clock for deterministic deadline.
        fixed = [1000.0]
        monkeypatch.setattr(time, "monotonic", lambda: fixed[0])
        out = asyncio.run(runner._handle_extend_command(_Event("30")))
        assert "extended by 30" in out
        assert runner._inactivity_extend_deadlines["session:c1"] == (
            7, 1000.0 + 30 * 60
        )

    def test_extend_command_clears_with_zero(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._inactivity_extend_deadlines = {"session:c1": (7, 9999.0)}
        runner._inactivity_extend_deadlines_lock = threading.Lock()
        runner._running_agents = {"session:c1": object()}
        runner._session_run_generation = {"session:c1": 7}
        class _Source:
            platform = "cli"
            chat_id = "c1"
        class _Event:
            def __init__(self):
                self.source = _Source()
            def get_command_args(self):
                return "0"
        monkeypatch.setattr(
            runner, "_session_key_for_source",
            lambda src: f"session:{src.chat_id}",
        )
        out = asyncio.run(runner._handle_extend_command(_Event()))
        assert "cleared" in out
        assert "session:c1" not in runner._inactivity_extend_deadlines

    def test_extend_rejects_when_no_turn_is_running(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._inactivity_extend_deadlines = {}
        runner._inactivity_extend_deadlines_lock = threading.Lock()
        runner._running_agents = {}
        runner._session_run_generation = {}
        monkeypatch.setattr(runner, "_session_key_for_source", lambda _source: "session:c1")

        class _Event:
            source = object()

            def get_command_args(self):
                return "30"

        out = asyncio.run(runner._handle_extend_command(_Event()))
        assert out == "No active agent turn to extend."
        assert runner._inactivity_extend_deadlines == {}

    def test_busy_dispatch_reaches_extend_handler(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        called = []

        async def _extend(event):
            called.append(event)
            return "extended"

        monkeypatch.setattr(runner, "_handle_extend_command", _extend)
        command = types.SimpleNamespace(
            name="extend", busy_handler="extend", busy_policy="reject"
        )
        event = object()
        out = asyncio.run(
            runner._dispatch_busy_slash_command(event, command, "session:c1", object())
        )
        assert out == "extended"
        assert called == [event]

    def test_extend_rejects_non_finite_minutes_without_storing_state(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._inactivity_extend_deadlines = {}
        runner._inactivity_extend_deadlines_lock = threading.Lock()
        runner._running_agents = {"session:c1": object()}
        runner._session_run_generation = {"session:c1": 7}
        monkeypatch.setattr(runner, "_session_key_for_source", lambda _source: "session:c1")

        class _Event:
            source = object()

            def __init__(self, value):
                self.value = value

            def get_command_args(self):
                return self.value

        for value in ("nan", "inf", "-inf"):
            out = asyncio.run(runner._handle_extend_command(_Event(value)))
            assert "finite number" in out
            assert runner._inactivity_extend_deadlines == {}

    def test_fractional_extend_reports_the_applied_duration(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._inactivity_extend_deadlines = {}
        runner._inactivity_extend_deadlines_lock = threading.Lock()
        runner._running_agents = {"session:c1": object()}
        runner._session_run_generation = {"session:c1": 7}
        monkeypatch.setattr(runner, "_session_key_for_source", lambda _source: "session:c1")
        monkeypatch.setattr(time, "monotonic", lambda: 1000.0)

        class _Event:
            source = object()

            def get_command_args(self):
                return "0.5"

        out = asyncio.run(runner._handle_extend_command(_Event()))
        assert "extended by 0.5 min" in out
        assert runner._inactivity_extend_deadlines["session:c1"] == (7, 1030.0)

    def test_extension_state_is_owned_by_one_turn_generation(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._inactivity_extend_deadlines = {}
        runner._inactivity_extend_deadlines_lock = threading.Lock()

        runner._set_inactivity_extension("session:c1", 7, 1200.0)
        assert runner._get_inactivity_extension("session:c1", 7) == 1200.0
        assert runner._get_inactivity_extension("session:c1", 8) is None
        assert not runner._clear_inactivity_extension("session:c1", 8)
        assert runner._get_inactivity_extension("session:c1", 7) == 1200.0

        runner._set_inactivity_extension("session:c1", 8, 1800.0)
        assert not runner._clear_inactivity_extension("session:c1", 7)
        assert runner._get_inactivity_extension("session:c1", 8) == 1800.0
        assert runner._clear_inactivity_extension("session:c1", 8)
        assert runner._inactivity_extend_deadlines == {}


class TestGatewayInactivityPolicy:
    def test_provider_grace_uses_the_shared_production_policy(self):
        from gateway.run import _effective_gateway_inactivity_timeout

        timeout = _effective_gateway_inactivity_timeout(
            30.0,
            {"last_activity_desc": "waiting for provider response (streaming)"},
            provider_grace=300.0,
        )
        assert timeout == 330.0

    def test_non_streaming_provider_description_uses_the_same_grace(self):
        from gateway.run import _effective_gateway_inactivity_timeout

        timeout = _effective_gateway_inactivity_timeout(
            30.0,
            {"last_activity_desc": "waiting for non-streaming API response"},
            provider_grace=300.0,
        )
        assert timeout == 330.0

    def test_extend_is_an_absolute_no_reap_deadline(self):
        from gateway.run import _effective_gateway_inactivity_timeout

        # An already-idle turn must remain protected until the requested
        # deadline; treating deadline-now as a fresh idle budget reaps it early.
        timeout = _effective_gateway_inactivity_timeout(
            30.0, {}, extend_deadline=120.0, now=100.0
        )
        assert timeout == float("inf")
        assert _effective_gateway_inactivity_timeout(
            30.0, {}, extend_deadline=120.0, now=120.0
        ) == 30.0
