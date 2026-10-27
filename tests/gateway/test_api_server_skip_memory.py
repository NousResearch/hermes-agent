"""Per-run ``model_options`` memory opt-out.

``{"memory": false}`` (or ``{"skip_memory": true}``) builds the run's agent with
``skip_memory=True`` and no checked-out session manager, so the external provider does no
prefetch, injects no memory-context and syncs nothing for that run. The session's parked
manager stays in the registry for the next normal turn.
"""

from types import SimpleNamespace

import pytest

import run_agent
from agent import agent_init
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_request_options import _request_skip_memory


@pytest.mark.parametrize("options, expected", [
    (None, False),
    ({}, False),
    ({"fast": True}, False),
    ({"memory": True}, False),
    ({"memory": False}, True),
    ({"memory": "false"}, True),
    ({"memory": None}, False),
    ({"memory": "garbage"}, False),
    ({"skip_memory": True}, True),
    ({"skip_memory": False}, False),
    ({"skip_memory": "true"}, True),
    ("memory=false", False),
])
def test_request_skip_memory_parsing(options, expected):
    assert _request_skip_memory(options) is expected


@pytest.fixture
def adapter_and_capture(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.clear()
            captured.update(kwargs)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {"provider": "anthropic"})
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "claude-opus-5-5")
    monkeypatch.setattr("gateway.run._load_gateway_config", dict)
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config", staticmethod(lambda model="": None))
    monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: {"memory"})
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    return adapter, captured


def _park(adapter, session_id):
    manager = object()
    adapter._memory_sessions.checkin(SimpleNamespace(_memory_manager=manager, session_id=session_id))
    return manager


def test_memory_false_skips_provider_and_leaves_parked_manager(adapter_and_capture):
    adapter, captured = adapter_and_capture
    parked = _park(adapter, "s1")

    adapter._create_agent(session_id="s1", model_options={"memory": False, "reasoning_effort": "low"})

    assert captured["skip_memory"] is True
    assert captured["memory_manager"] is None
    assert captured["reasoning_config"] == {"enabled": True, "effort": "low"}
    # The next normal (text) turn on the same session still gets the session's manager.
    assert adapter._memory_sessions.checkout("s1") is parked


def test_default_run_keeps_memory_and_checks_out_parked_manager(adapter_and_capture):
    adapter, captured = adapter_and_capture
    parked = _park(adapter, "s1")

    adapter._create_agent(session_id="s1", model_options={"fast": True})

    assert captured["skip_memory"] is False
    assert captured["memory_manager"] is parked


def test_no_model_options_keeps_memory(adapter_and_capture):
    adapter, captured = adapter_and_capture
    adapter._create_agent(session_id="s2")
    assert captured["skip_memory"] is False


def test_skipped_agent_prefetches_nothing():
    """The turn_context gate: an agent built with skip_memory has no manager, so no prefetch."""
    from agent.turn_context import _memory_turn_start_and_prefetch

    agent = SimpleNamespace(_memory_manager=None)
    assert _memory_turn_start_and_prefetch(agent, "What is the capital of Australia?") == ""


def test_init_memory_skip_keeps_builtin_store_without_provider(monkeypatch):
    """skip_memory drops only the external provider; the api_server's memory toolset keeps
    MEMORY.md/USER.md, and a handed-in session manager is ignored."""

    agent = SimpleNamespace(enabled_toolsets=["memory"], disabled_toolsets=None, tools=[],
                            valid_tool_names=set())
    monkeypatch.setattr("agent.memory_manager.inject_memory_provider_tools", lambda a: None)
    agent_init._init_memory(agent, {}, True, "api_server", memory_manager=object())
    assert agent._memory_manager is None
