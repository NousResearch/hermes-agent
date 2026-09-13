"""Dynamic schemas, prompts and dispatch share the effective provider policy."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.memory_manager import MemoryManager, inject_memory_provider_tools
from tests.agent.test_memory_provider import FakeMemoryProvider
from tests.run_agent.test_run_agent import agent, _mock_tool_call, _mock_assistant_msg  # noqa: F401


@pytest.mark.parametrize("mode", ["invoke", "sequential"])
@pytest.mark.parametrize("ownership", ["absent", "empty", "owned"])
def test_injected_ownership_controls_both_dispatch_paths(agent, monkeypatch, mode, ownership):
    manager = SimpleNamespace(has_tool=lambda name: name == "web_search",
                              handle_tool_call=MagicMock(return_value='{"provider":true}'))
    agent._memory_manager = manager
    if ownership == "absent":
        agent.__dict__.pop("_memory_provider_tool_names", None)
    else:
        agent._memory_provider_tool_names = {"web_search"} if ownership == "owned" else set()
    monkeypatch.setattr("hermes_cli.plugins.resolve_pre_tool_block", lambda *_a, **_kw: None)
    with patch("model_tools.handle_function_call", return_value='{"registry":true}') as generic:
        if mode == "invoke":
            result = agent._invoke_tool("web_search", {"q": "test"}, "task-1")
        else:
            tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="ownership-1")
            messages = []
            agent._execute_tool_calls_sequential(_mock_assistant_msg(content="", tool_calls=[tc]), messages, "task-1")
            result = messages[-1]["content"]
    provider_owned = ownership != "empty"
    assert json.loads(result) == ({"provider": True} if provider_owned else {"registry": True})
    assert manager.handle_tool_call.call_count == int(provider_owned)
    assert generic.call_count == int(not provider_owned)


@pytest.mark.parametrize("tool_name,owned", [("denied_provider_tool", None), ("web_search", set())])
def test_production_prompt_omits_unavailable_or_unowned_provider(agent, tool_name, owned):
    block = f"Use {tool_name} for every memory lookup."
    provider = FakeMemoryProvider("external", tools=[{"name": tool_name}])
    provider._prompt_block = block
    manager = MemoryManager()
    manager._providers.append(provider)
    agent._memory_manager = manager
    agent.valid_tool_names = {"web_search"}
    if owned is not None:
        agent._memory_provider_tool_names = owned
    assert block not in agent._build_system_prompt()


def test_filtered_prompt_does_not_claim_malformed_capability():
    manager = MemoryManager()
    provider = FakeMemoryProvider("external", tools=[{"description": "no name"}])
    provider._prompt_block = "Use the malformed capability"
    manager.add_provider(provider)
    assert manager.build_system_prompt(available_tool_names=set()) == ""
    assert manager.build_system_prompt() == provider._prompt_block


def _tool(name, description=""):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {}}}


def _dynamic_agent(*, disabled=None, context=False):
    provider = FakeMemoryProvider("external", tools=[{"name": "shared_tool", "description": "provider", "parameters": {}}])
    manager = MemoryManager()
    manager.add_provider(provider)
    a = SimpleNamespace(_memory_manager=manager, tools=[], valid_tool_names=set(),
                        enabled_toolsets=None, disabled_toolsets=disabled,
                        _context_engine_tool_names=set(), context_compressor=None)
    if context:
        a._memory_manager = None
        a.context_compressor = SimpleNamespace(get_tool_schemas=provider.get_tool_schemas)
    return a


def test_refresh_applies_exact_dynamic_subtraction(monkeypatch):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    import toolsets
    a = _dynamic_agent(disabled=["custom-deny"])
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [])
    monkeypatch.setattr(toolsets, "validate_toolset", lambda name: name == "custom-deny")
    monkeypatch.setattr(toolsets, "get_toolset", lambda _name: {"tools": ["shared_tool"]})
    monkeypatch.setattr(toolsets, "resolve_toolset", lambda _name: {"shared_tool"})
    refresh_agent_mcp_tools(a)
    assert a.tools == []
    assert a.valid_tool_names == set()
    assert not getattr(a, "_memory_provider_tool_names", set())
    assert not a._context_engine_tool_names


def test_refresh_equal_name_collision_transfers_ownership_with_schema(monkeypatch):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    a = _dynamic_agent()
    inject_memory_provider_tools(a)
    assert a._memory_provider_tool_names == {"shared_tool"}
    registry_schema = _tool("shared_tool", "registry")
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [registry_schema])
    refresh_agent_mcp_tools(a)
    assert a.tools == [registry_schema]
    assert a._memory_provider_tool_names == set()


def test_refresh_failure_preserves_previous_schema_ownership_and_policy(monkeypatch):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    a = _dynamic_agent()
    inject_memory_provider_tools(a)
    old_tools = a.tools
    old_names = a.valid_tool_names
    a._memory_manager.get_all_tool_schemas_strict = MagicMock(side_effect=RuntimeError("provider unavailable"))
    # A snapshot-aware refresher may filter its last published schemas when
    # enumeration fails. This case has no usable policy fallback either.
    monkeypatch.setattr("agent.memory_manager.effective_memory_provider_tool_schemas",
                        MagicMock(side_effect=RuntimeError("provider unavailable")))
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [])
    with pytest.raises(RuntimeError, match="provider unavailable"):
        refresh_agent_mcp_tools(a, disabled_override=["unrelated"])
    assert a.tools is old_tools
    assert a.valid_tool_names is old_names
    assert a._memory_provider_tool_names == {"shared_tool"}
    assert a.disabled_toolsets is None


def test_refresh_establishes_collision_ownership_for_legacy_agent(monkeypatch):
    from agent.memory_manager import memory_provider_owns_tool
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    a = _dynamic_agent()
    a.tools = [_tool("shared_tool", "registry")]
    a.valid_tool_names = {"shared_tool"}
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [_tool("shared_tool", "registry")])
    refresh_agent_mcp_tools(a)
    assert not memory_provider_owns_tool(a, "shared_tool")
    assert a._memory_provider_tool_names == set()


def test_preserved_prefix_cannot_restore_denied_provider_collision(monkeypatch):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    from tools.registry import registry
    import toolsets

    a = _dynamic_agent()
    inject_memory_provider_tools(a)
    registry.register(name="shared_tool", toolset="policy-test", schema={"name": "shared_tool"}, handler=lambda _args, **_kw: "registry")
    try:
        monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [])
        monkeypatch.setattr(toolsets, "validate_toolset", lambda name: name == "custom-deny")
        monkeypatch.setattr(toolsets, "get_toolset", lambda _name: {"tools": ["shared_tool"]})
        monkeypatch.setattr(toolsets, "resolve_toolset", lambda _name: {"shared_tool"})
        refresh_agent_mcp_tools(a, disabled_override=["custom-deny"], preserve_prefix=True)
        assert a.tools == []
        assert a.valid_tool_names == set()
        assert a._memory_provider_tool_names == set()
    finally:
        registry.deregister("shared_tool")


def test_removed_registry_reservation_allows_provider_transfer(monkeypatch):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    a = _dynamic_agent()
    a._tool_registry_routes = {"shared_tool": object()}
    a.tools = [_tool("shared_tool", "old registry")]
    a.valid_tool_names = {"shared_tool"}
    a._memory_provider_tool_names = set()
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [])
    refresh_agent_mcp_tools(a)
    assert a.tools == [_tool("shared_tool", "provider")]
    assert a._memory_provider_tool_names == {"shared_tool"}


def test_preserved_prefix_excludes_legacy_provider_schema(monkeypatch):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    from tools.registry import registry
    a = _dynamic_agent(disabled=["memory"])
    a.tools = [_tool("shared_tool", "old provider")]
    a.valid_tool_names = {"shared_tool"}
    registry.register(name="shared_tool", toolset="policy-test", schema={"name": "shared_tool"}, handler=lambda _args, **_kw: "registry")
    try:
        monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [])
        refresh_agent_mcp_tools(a, preserve_prefix=True)
        assert a.tools == []
        assert a._memory_provider_tool_names == set()
    finally:
        registry.deregister("shared_tool")


def test_disabled_posture_preserves_shared_memory_core(monkeypatch):
    import toolsets
    a = _dynamic_agent(disabled=["custom-posture"])
    monkeypatch.setattr(toolsets, "validate_toolset", lambda name: name == "custom-posture")
    monkeypatch.setattr(toolsets, "get_toolset", lambda _name: {"posture": True})
    monkeypatch.setattr(toolsets, "resolve_toolset", lambda _name: {"memory", "read_file"})
    monkeypatch.setattr(toolsets, "bundle_non_core_tools", lambda _name: set())
    inject_memory_provider_tools(a)
    assert a.valid_tool_names == {"shared_tool"}
    assert a._memory_provider_tool_names == {"shared_tool"}
