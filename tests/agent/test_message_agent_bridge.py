"""tool_call bridge for injected ``message_agent``.

``message_agent`` is not registered, so a bridge call used to be rejected as
an unknown name and retried forever. An authorized Bot Chat unwraps it onto
the inline executor; any other session keeps the wrapper.
"""

from types import SimpleNamespace

from agent.inline_tool_executors import resolve_invoke_tool_executor
from agent.tool_executor import _unwrap_tool_search_call

_BRIDGE = {
    "calls": [{
        "name": "message_agent",
        "arguments": {"target": "sven", "message": "hi"},
    }],
}


def test_authorized_bridge_unwraps_message_agent(monkeypatch):
    monkeypatch.setattr("tools.bot_mode_dm.message_agent_authorized", lambda _agent: True)
    name, args, block = _unwrap_tool_search_call(SimpleNamespace(), "tool_call", _BRIDGE)
    assert block is None
    assert name == "message_agent"
    assert args == {"target": "sven", "message": "hi"}


def test_unauthorized_bridge_does_not_unwrap_message_agent(monkeypatch):
    monkeypatch.setattr("tools.bot_mode_dm.message_agent_authorized", lambda _agent: False)
    name, _args, _block = _unwrap_tool_search_call(
        SimpleNamespace(enabled_toolsets=None, disabled_toolsets=None),
        "tool_call",
        _BRIDGE,
    )
    assert name == "tool_call"


def test_concurrent_path_dispatches_message_agent_inline():
    executor = resolve_invoke_tool_executor(
        SimpleNamespace(_memory_manager=None), "message_agent",
    )
    assert executor is not None
