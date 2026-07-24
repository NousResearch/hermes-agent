import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.context_audit import collect_context_audit


SK = "agent:main:telegram:private:12345"


def _tool(name: str, description: str = "desc") -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {"type": "object"}},
    }


def _runner(session_key=SK, agent=None, cached_agent=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_key_for_source = MagicMock(return_value=session_key)
    if agent is not None:
        runner._running_agents[session_key] = agent
    if cached_agent is not None:
        runner._agent_cache[session_key] = (cached_agent, "sig")
    return runner


def _agent_with_report():
    agent = SimpleNamespace(model="gpt-test", provider="test", tools=[_tool("memory", "SECRET BODY")])
    agent._context_audit_report = collect_context_audit(
        agent,
        prompt_parts={"stable": "identity", "context": "project", "volatile": "SECRET MEMORY BODY"},
    )
    agent._context_audit_report_path = ""
    return agent


@pytest.mark.asyncio
async def test_context_audit_command_uses_cached_agent_and_redacts_content():
    runner = _runner(cached_agent=_agent_with_report())
    event = MagicMock()
    event.source = MagicMock()

    result = await runner._handle_context_audit_command(event)

    assert "Context audit" in result
    assert "tool schemas" in result
    assert "SECRET" not in result
    assert "memory" in result


@pytest.mark.asyncio
async def test_context_audit_command_reports_unavailable_without_report():
    runner = _runner(cached_agent=SimpleNamespace(_context_audit_report=None))
    event = MagicMock()
    event.source = MagicMock()

    result = await runner._handle_context_audit_command(event)

    assert "not available" in result
    assert "agent.startup_context_audit" in result
