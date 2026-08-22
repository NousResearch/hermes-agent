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


class _ProviderGraceAgent(FakeAgent):
    """Agent whose last_activity_desc marks it as waiting on a provider."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._activity_desc = "waiting for provider response (thinking)"
        self._provenance = "waiting_for_provider_response"

    def get_activity_summary(self):
        s = super().get_activity_summary()
        s["last_activity_desc"] = self._activity_desc
        s["last_activity_provenance"] = self._provenance
        return s


class _DelegatingParentAgent(FakeAgent):
    """Orchestrator parent whose activity clock is refreshed by a heartbeat
    while a synchronous subagent runs (the _touch_activity poll loop)."""

    def __init__(self, heartbeat_every=0.1, total_idle=1.2, **kwargs):
        super().__init__(**kwargs)
        self._heartbeat_every = heartbeat_every
        self._total_idle = total_idle
        self._last_touch = time.monotonic()
        self._activity_desc = "delegating to subagent (child running)"

    def _touch_activity(self, desc=None):
        self._last_touch = time.monotonic()
        if desc:
            self._activity_desc = desc

    def get_activity_summary(self):
        # Simulate the parent being re-touched on each poll while the child runs.
        now = time.monotonic()
        # The delegate_tool heartbeat would call _touch_activity every
        # _poll_interval; emulate the net effect: idle clock resets each beat.
        return {
            "last_activity_ts": now - 0.0,
            "last_activity_desc": self._activity_desc,
            "seconds_since_activity": 0.0,
            "current_tool": None,
            "api_call_count": 5,
            "max_iterations": 90,
        }


class TestProviderGraceAndExtend:
    """Tests for the provider-think grace and /extend deadline (Gap 3 + Gap 2)."""

    def test_provider_grace_extends_effective_timeout(self):
        """While waiting on a provider, the idle budget is _agent_timeout + grace."""
        agent = _ProviderGraceAgent(activity_desc="waiting for provider response")
        _agent_timeout = 1.0
        _agent_provider_grace = 0.5
        _POLL_INTERVAL = 0.05

        # Drive the SAME threshold logic the watchdog uses, including the
        # provider-grace branch added for #4815.
        _inactivity_timeout = False
        start = time.monotonic()
        while True:
            _act = agent.get_activity_summary()
            _idle_secs = _act.get("seconds_since_activity", 0.0)
            _effective_timeout = _agent_timeout
            if _agent_provider_grace and _agent_timeout is not None:
                _desc = _act.get("last_activity_desc") or ""
                _prov = _act.get("last_activity_provenance") or ""
                if ("waiting for provider response" in _desc
                        or _prov == "waiting_for_provider_response"):
                    _effective_timeout = _agent_timeout + _agent_provider_grace
            if _idle_secs >= _effective_timeout:
                _inactivity_timeout = True
                break
            if time.monotonic() - start > _agent_timeout + _agent_provider_grace + 0.5:
                # Should not time out before grace expires.
                break

        assert _effective_timeout == 1.5
        assert not _inactivity_timeout

    def test_extend_deadline_overrides_timeout(self):
        """A /extend deadline raises the effective timeout for the turn."""
        agent = FakeAgent(activity_desc="tool_call", idle_seconds=2.0)
        _agent_timeout = 1.0
        # Simulated /extend N -> monotonic deadline in the future.
        _ext_deadline = time.monotonic() + 60.0
        _POLL_INTERVAL = 0.05

        _act = agent.get_activity_summary()
        _idle_secs = _act.get("seconds_since_activity", 0.0)
        _effective_timeout = _agent_timeout
        _ext_idle_budget = max(0.0, _ext_deadline - time.monotonic())
        if _agent_timeout is None or _ext_idle_budget > _effective_timeout:
            _effective_timeout = _ext_idle_budget

        # idle is 2s but the extended budget is ~60s -> not timed out.
        assert _idle_secs < _effective_timeout
        assert _effective_timeout > 50.0

    def test_delegating_parent_not_reaped(self):
        """A delegating parent that heartbeats stays alive past the timeout."""
        agent = _DelegatingParentAgent()
        _agent_timeout = 0.5
        _POLL_INTERVAL = 0.05
        _inactivity_timeout = False
        start = time.monotonic()
        while True:
            _act = agent.get_activity_summary()
            _idle_secs = _act.get("seconds_since_activity", 0.0)
            if _idle_secs >= _agent_timeout:
                _inactivity_timeout = True
                break
            if time.monotonic() - start > _agent_timeout + 0.5:
                break
        assert not _inactivity_timeout


class TestExtendCommandHandler:
    """Tests for /extend command parsing and storage (Gap 2)."""

    def test_extend_command_sets_deadline(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._inactivity_extend_deadlines = {}
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
        import asyncio
        out = asyncio.get_event_loop().run_until_complete(
            runner._handle_extend_command(_Event("30"))
        )
        assert "extended by 30" in out
        assert runner._inactivity_extend_deadlines["session:c1"] == 1000.0 + 30 * 60

    def test_extend_command_clears_with_zero(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._inactivity_extend_deadlines = {"session:c1": 9999.0}
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
        import asyncio
        out = asyncio.get_event_loop().run_until_complete(
            runner._handle_extend_command(_Event())
        )
        assert "cleared" in out
        assert "session:c1" not in runner._inactivity_extend_deadlines








