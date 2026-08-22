from types import SimpleNamespace

from tools import mcp_tool


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": "", "parameters": {}},
    }


def test_request_tool_names_restores_schema_visible_tools_without_global_write():
    from agent.conversation_loop import request_tool_names

    request_tools = [_tool("terminal"), _tool("read_file"), _tool("write_file")]
    agent = SimpleNamespace(
        tools=request_tools,
        valid_tool_names=set(),
    )

    names = request_tool_names(request_tools)

    assert names == {"terminal", "read_file", "write_file"}
    assert agent.valid_tool_names == set()


def test_request_tool_names_uses_the_request_snapshot_without_rolling_back_live_state():
    from agent.conversation_loop import request_tool_names

    request_tools = [_tool("terminal"), _tool("read_file")]
    agent = SimpleNamespace(
        tools=[_tool("terminal"), _tool("newly_refreshed_tool")],
        valid_tool_names={"terminal", "newly_refreshed_tool"},
    )

    names = request_tool_names(request_tools)

    assert names == {"terminal", "read_file"}
    assert agent.valid_tool_names == {"terminal", "newly_refreshed_tool"}


def test_inflight_response_detects_real_refresh_without_poisoning_live_snapshot(
    monkeypatch,
):
    from agent.conversation_loop import (
        request_tool_bindings_are_stale,
        request_tool_names,
    )
    from tools.registry import registry

    request_tools = [_tool("terminal"), _tool("read_file")]
    request_generation = registry._generation
    agent = SimpleNamespace(
        tools=request_tools,
        valid_tool_names={"terminal", "read_file"},
        _tool_snapshot_generation=request_generation,
        enabled_toolsets=None,
        disabled_toolsets=None,
    )

    live_tools = [_tool("terminal"), _tool("newly_refreshed_tool")]
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **_kwargs: list(live_tools),
    )
    monkeypatch.setattr(registry, "_generation", request_generation + 1)

    assert mcp_tool.refresh_agent_mcp_tools(agent) == {"newly_refreshed_tool"}
    published_generation = agent._tool_snapshot_generation

    assert request_tool_names(request_tools) == {"terminal", "read_file"}
    assert request_tool_bindings_are_stale(agent, request_tools)
    assert {tool["function"]["name"] for tool in agent.tools} == {
        "terminal",
        "newly_refreshed_tool",
    }
    assert agent.valid_tool_names == {"terminal", "newly_refreshed_tool"}
    assert agent._tool_snapshot_generation == published_generation

    assert mcp_tool.refresh_agent_mcp_tools(agent) == set()
    assert agent.valid_tool_names == {
        tool["function"]["name"] for tool in agent.tools
    }
    assert agent._tool_snapshot_generation == published_generation


def test_generation_only_refresh_does_not_stale_an_identical_request_surface():
    from agent.conversation_loop import request_tool_bindings_are_stale

    request_tools = [_tool("terminal")]
    agent = SimpleNamespace(
        tools=request_tools,
        _tool_snapshot_generation=2,
    )

    assert not request_tool_bindings_are_stale(agent, request_tools)


def test_registry_dispatch_rejects_rebound_entry_without_calling_either_handler():
    from tools.registry import registry

    calls = []
    schema = {
        "name": "binding_identity_probe",
        "description": "Probe binding identity.",
        "parameters": {"type": "object", "properties": {}},
    }
    registry.register(
        name="binding_identity_probe",
        toolset="test-binding",
        schema=schema,
        handler=lambda _args, **_kwargs: calls.append("A") or "A",
    )
    expected = registry.capture_bindings({"binding_identity_probe"})[
        "binding_identity_probe"
    ]
    registry.register(
        name="binding_identity_probe",
        toolset="test-binding",
        schema=schema,
        handler=lambda _args, **_kwargs: calls.append("B") or "B",
    )

    result = registry.dispatch(
        "binding_identity_probe",
        {},
        expected_entry=expected,
        enforce_expected_entry=True,
    )

    assert calls == []
    assert "stale_tool_binding" in result


def test_refresh_publishes_deferred_bindings_with_the_live_surface(monkeypatch):
    from tools.mcp_tool import (
        refresh_agent_mcp_tools,
        snapshot_agent_tool_surface,
    )
    from tools.registry import registry

    calls = []
    schema = {
        "name": "deferred_binding_probe",
        "description": "Deferred binding identity probe.",
        "parameters": {"type": "object", "properties": {}},
    }
    registry.register(
        name="deferred_binding_probe",
        toolset="deferred-test",
        schema=schema,
        handler=lambda _args, **_kwargs: calls.append("A") or "A",
    )
    live_tools = [_tool("terminal")]
    agent = SimpleNamespace(
        tools=[],
        valid_tool_names=set(),
        _tool_snapshot_generation=0,
        _tool_registry_bindings={},
        enabled_toolsets=None,
        disabled_toolsets=None,
    )
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **kwargs: (
            [*live_tools, {"type": "function", "function": schema}]
            if kwargs.get("skip_tool_search_assembly")
            else list(live_tools)
        ),
    )

    assert refresh_agent_mcp_tools(agent) == {"terminal"}
    _, published_generation, bindings = snapshot_agent_tool_surface(agent)
    expected = bindings["deferred_binding_probe"]
    assert expected is registry.get_entry("deferred_binding_probe")

    registry.register(
        name="deferred_binding_probe",
        toolset="deferred-test",
        schema=schema,
        handler=lambda _args, **_kwargs: calls.append("B") or "B",
    )
    result = registry.dispatch(
        "deferred_binding_probe",
        {},
        expected_entry=expected,
        enforce_expected_entry=True,
    )

    assert calls == []
    assert "stale_tool_binding" in result
    assert agent._tool_snapshot_generation == published_generation
