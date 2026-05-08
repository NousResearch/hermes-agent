"""Regression tests for MCP server availability in cron jobs.

Background
==========
``cron/scheduler.py:run_job()`` constructs ``AIAgent(...)`` directly without
calling ``discover_mcp_tools()`` — the initialization that CLI and gateway
paths do at startup. Cron jobs therefore never saw any MCP tools from
``mcp_servers`` in config.yaml. See #4219.

The fix inserts ``discover_mcp_tools()`` before the ``AIAgent(...)`` call,
wrapped in try/except so a broken MCP server can't kill an otherwise
working cron job. ``discover_mcp_tools`` is idempotent — subsequent ticks
short-circuit on already-connected servers.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


def test_run_job_calls_discover_mcp_tools_before_agent_construction():
    """The LLM-path branch of run_job must call discover_mcp_tools() before
    the AIAgent construction, so MCP tools are in the registry by the time
    the agent asks for its tool schema."""
    from cron import scheduler

    job = {
        "id": "mcp-cron-test",
        "name": "mcp-cron-test",
        "prompt": "test",
    }

    call_order = []

    def fake_discover():
        call_order.append("discover_mcp_tools")
        return ["mcp_server1_tool"]

    # AIAgent is a class; replace with a recording stub
    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            call_order.append("AIAgent.__init__")
            self._kwargs = kwargs
            self._interrupt_requested = False
            self.quiet_mode = True

        def run_conversation(self, *args, **kwargs):
            return {
                "final_response": "ok",
                "messages": [],
            }

    with patch("tools.mcp_tool.discover_mcp_tools", side_effect=fake_discover), \
         patch("run_agent.AIAgent", _FakeAgent), \
         patch("cron.scheduler._resolve_cron_enabled_toolsets", return_value=None):
        scheduler.run_job(job)

    # Discovery must be called, and must be called BEFORE agent construction.
    assert "discover_mcp_tools" in call_order, (
        "run_job did not call discover_mcp_tools — MCP tools unavailable in cron"
    )
    d_idx = call_order.index("discover_mcp_tools")
    a_idx = call_order.index("AIAgent.__init__")
    assert d_idx < a_idx, (
        f"discover_mcp_tools was called AFTER AIAgent construction "
        f"(indices discover={d_idx}, agent={a_idx}); MCP tools missed the "
        f"registry window. Full order: {call_order}"
    )


def test_run_job_tolerates_discover_mcp_tools_failure():
    """A broken MCP server must not kill an otherwise working cron job.
    discover_mcp_tools() raising should be caught and logged, and the agent
    should still run."""
    from cron import scheduler

    job = {
        "id": "mcp-cron-fail",
        "name": "mcp-cron-fail",
        "prompt": "test",
    }

    agent_was_constructed = []

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            agent_was_constructed.append(True)
            self._interrupt_requested = False
            self.quiet_mode = True

        def run_conversation(self, *args, **kwargs):
            return {"final_response": "ok", "messages": []}

    def fake_discover_that_raises():
        raise RuntimeError("MCP server unreachable")

    with patch(
        "tools.mcp_tool.discover_mcp_tools",
        side_effect=fake_discover_that_raises,
    ), patch("run_agent.AIAgent", _FakeAgent), \
         patch("cron.scheduler._resolve_cron_enabled_toolsets", return_value=None):
        # Should NOT raise
        success, doc, final_response, error = scheduler.run_job(job)

    assert agent_was_constructed, (
        "AIAgent was not constructed after discover_mcp_tools raised — "
        "MCP failure incorrectly killed the cron job"
    )


def test_no_agent_cron_job_does_not_initialize_mcp():
    """Cron jobs with no_agent=True are script-only — no AIAgent, no MCP
    tools needed. We must NOT pay the MCP init cost for those."""
    from cron import scheduler

    job = {
        "id": "noagent-job",
        "name": "noagent-job",
        "no_agent": True,
        "script": "/nonexistent/script.sh",
    }

    discover_called = []

    def fake_discover():
        discover_called.append(True)
        return []

    # _run_job_script returns (ok, output); make it fail cleanly so we
    # don't need a real script file.
    with patch("tools.mcp_tool.discover_mcp_tools", side_effect=fake_discover), \
         patch("cron.scheduler._run_job_script", return_value=(False, "no such file")):
        scheduler.run_job(job)

    assert not discover_called, (
        "discover_mcp_tools was called for a no_agent job — wasted MCP init "
        "for a script-only cron tick"
    )
