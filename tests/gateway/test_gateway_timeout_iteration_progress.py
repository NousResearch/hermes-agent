"""#102806: the gateway inactivity-timeout diagnostic message and the
long-running heartbeat -- the two remaining user-facing render sites
alongside the busy-ack (see test_busy_session_ack.py) -- must not print
sys.maxsize as an iteration ceiling either.

``_run_agent_timeout_result`` builds the synthetic ``final_response`` shown
to the user when an agent run is force-timed-out for inactivity. It embeds
``iteration N/M`` twice (the "stuck on tool" branch and the "last activity"
branch). ``_run_agent_notify_long_running`` embeds it once, in the periodic
"still working" heartbeat. All three call sites (this file's two classes,
plus test_busy_session_ack.py's busy-ack test) share the same
``_format_iteration_progress`` helper; see test_format_iteration_progress.py
for the helper's own unit tests.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.run_turn import GatewayTurnMixin
from gateway.turn_context import TurnContext


def _make_worker(agent_timeout=1800.0):
    return SimpleNamespace(agent_timeout=agent_timeout)


def _make_turn_ctx(agent):
    return TurnContext(session_key="sess-1", agent_holder=[agent])


class TestTimeoutDiagnosticIterationProgress:
    def test_unbounded_max_iterations_omits_denominator_when_stuck_on_tool(self, monkeypatch):
        mixin = GatewayTurnMixin()
        monkeypatch.setattr("gateway.run.request_hard_interrupt", MagicMock(), raising=False)
        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "last_activity_desc": "tool_call",
            "seconds_since_activity": 42.0,
            "current_tool": "terminal",
            "api_call_count": 5,
            "max_iterations": sys.maxsize,
        }
        result = mixin._run_agent_timeout_result(_make_worker(), _make_turn_ctx(agent))
        assert "iteration 5" in result["final_response"]
        assert str(sys.maxsize) not in result["final_response"]

    def test_unbounded_max_iterations_omits_denominator_in_last_activity_branch(self, monkeypatch):
        mixin = GatewayTurnMixin()
        monkeypatch.setattr("gateway.run.request_hard_interrupt", MagicMock(), raising=False)
        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "last_activity_desc": "api_call_streaming",
            "seconds_since_activity": 12.0,
            "current_tool": None,
            "api_call_count": 2,
            "max_iterations": sys.maxsize,
        }
        result = mixin._run_agent_timeout_result(_make_worker(), _make_turn_ctx(agent))
        assert "iteration 2" in result["final_response"]
        assert str(sys.maxsize) not in result["final_response"]

    def test_finite_max_iterations_still_shows_both_numbers(self, monkeypatch):
        mixin = GatewayTurnMixin()
        monkeypatch.setattr("gateway.run.request_hard_interrupt", MagicMock(), raising=False)
        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "last_activity_desc": "tool_call",
            "seconds_since_activity": 5.0,
            "current_tool": "code_exec",
            "api_call_count": 7,
            "max_iterations": 250,
        }
        result = mixin._run_agent_timeout_result(_make_worker(), _make_turn_ctx(agent))
        assert "iteration 7/250" in result["final_response"]


class TestLongRunningHeartbeatIterationProgress:
    """The heartbeat's own render call, exercised end to end through its
    async polling loop (one iteration, then the loop is told to stop)."""

    @pytest.mark.asyncio
    async def test_heartbeat_omits_denominator_for_unbounded_max_iterations(self, monkeypatch):
        monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.01")

        mixin = GatewayTurnMixin()
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
        mixin._adapter_for_source = MagicMock(return_value=adapter)
        mixin._should_emit_long_running_notification = MagicMock(side_effect=[True, False])

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 4,
            "max_iterations": sys.maxsize,
            "current_tool": "terminal",
        }

        disp = MagicMock()
        disp._display_surface_mode.return_value = "on"
        disp.resolve_display_setting.return_value = True

        turn_ctx = TurnContext(
            source=SimpleNamespace(chat_id="c1", platform="telegram"),
            session_key="sess-1", agent_holder=[agent],
        )

        await mixin._run_agent_notify_long_running(disp, turn_ctx, [None])

        sent_text = adapter.send.await_args.args[1]
        assert "iteration 4" in sent_text
        assert str(sys.maxsize) not in sent_text
