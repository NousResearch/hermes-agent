"""Frontend dispatch must never manufacture a successful tool result."""

import pytest

from agui_adapter.continuation import CURRENT_FRONTEND, ContinuationError
from agui_adapter.session import _frontend_tool_handler


def test_frontend_dispatch_without_a_live_relay_fails_closed():
    token = CURRENT_FRONTEND.set(None)
    try:
        with pytest.raises(ContinuationError, match="live frontend relay"):
            _frontend_tool_handler({"color": "blue"})
    finally:
        CURRENT_FRONTEND.reset(token)


def test_stale_worker_release_cannot_evict_a_new_execution():
    from ag_ui.core import RunAgentInput
    from agui_adapter.continuation import ExecutionRegistry
    from tests.agui_adapter.test_modernization_probe import body

    registry = ExecutionRegistry()
    request = RunAgentInput.model_validate(body("same-thread"))
    old = registry.admit(request, object(), lambda event: None, 10)
    registry.release(old)
    newer = registry.admit(request, object(), lambda event: None, 10)
    try:
        registry.release(old)
        assert registry.get(request.thread_id) is newer
        assert registry.conversations.get(request.thread_id) is newer.conversation
    finally:
        registry.release(newer)


@pytest.mark.parametrize("toolset", ["all", "*"])
@pytest.mark.parametrize("kind", ["frontend", "writer"])
def test_eager_catalog_uses_only_current_adapter_declarations(peer, toolset, kind):
    from agui_adapter import session
    from hermes_cli.config import save_config
    from tests.agui_adapter.test_modernization_probe import config_for
    from tools.registry import registry

    name = f"agui_dispatch_catalog_{kind}_{toolset.replace('*', 'star')}"
    parameters = {
        "type": "object",
        "properties": {"color": {"type": "string"}},
        "required": ["color"],
    }
    schema = {"type": "function", "function": {"name": name, "parameters": parameters}}
    specs = {name: session.StateWriterSpec(state_key="color", arg="color")}
    save_config({"auxiliary": {"title_generation": {"enabled": False}}})
    agents = []
    try:
        if kind == "frontend":
            session._ensure_frontend_tools_registered({name})
            declaration = {
                "frontend_tool_names": {name},
                "frontend_tool_schemas": [schema],
            }
        else:
            session._ensure_state_writer_tools_registered(specs)
            declaration = {
                "state_writer_specs": specs,
                "state_writer_schemas": [schema],
            }
        undeclared = session.build_run_agent(config_for(peer, [toolset]))
        agents.append(undeclared)
        server_names = {tool["function"]["name"] for tool in undeclared.tools}
        assert name not in server_names | undeclared.valid_tool_names
        assert "terminal" in server_names
        current = session.build_run_agent(config_for(peer, [toolset]), **declaration)
        agents.append(current)
        admitted = {tool["function"]["name"]: tool for tool in current.tools}
        assert admitted[name] == schema
        assert set(admitted) == current.valid_tool_names == server_names | {name}
        current.tools = []
        current.valid_tool_names = set()
        current._build_system_prompt()
        assert {tool["function"]["name"]: tool for tool in current.tools} == admitted
        assert current.valid_tool_names == set(admitted)
    finally:
        for agent in agents:
            agent.close()
        registry.deregister(name)
        session._registered_frontend_names.discard(name)
        session._registered_state_writer_names.discard(name)


@pytest.mark.parametrize("toolsets", [["memory"], []])
@pytest.mark.parametrize("kind", [None, "frontend", "state_writer"])
def test_eager_catalog_preserves_initialized_memory_provider(
    peer, monkeypatch, toolsets, kind
):
    import json
    from unittest.mock import Mock

    import plugins.memory
    from agui_adapter.session import (
        StateWriterSpec,
        ToolNameCollisionError,
        build_run_agent,
    )
    from hermes_cli.config import save_config
    from tests.agent.test_memory_provider import FakeMemoryProvider
    from tests.agui_adapter.test_modernization_probe import config_for

    schema = {
        "name": f"agui_local_memory_search_{kind or 'ordinary'}",
        "description": "Search local provider memory",
        "parameters": {"type": "object", "properties": {}},
    }
    provider = FakeMemoryProvider("agui-local-memory", tools=[schema])
    initialize = Mock(wraps=provider.initialize)
    monkeypatch.setattr(provider, "initialize", initialize)
    monkeypatch.setattr(plugins.memory, "load_memory_provider", lambda name: provider)
    save_config({"memory": {"provider": provider.name}})
    if kind:
        declaration = (
            {"frontend_tool_names": {schema["name"]}}
            if kind == "frontend"
            else {
                "state_writer_specs": {
                    schema["name"]: StateWriterSpec(state_key="x", arg="x")
                }
            }
        )
        with pytest.raises(ToolNameCollisionError, match=schema["name"]):
            build_run_agent(config_for(peer, toolsets), **declaration)
        assert provider.initialized and provider.shutdown_called
        return
    agent = build_run_agent(config_for(peer, toolsets))
    try:
        assert provider.initialized and initialize.call_count == 1
        assert agent._memory_manager.has_tool(schema["name"])
        admitted = {tool["function"]["name"]: tool["function"] for tool in agent.tools}
        assert (schema["name"] in admitted) == bool(toolsets)
        assert set(admitted) == agent.valid_tool_names
        if toolsets:
            assert admitted[schema["name"]] == schema
            result = json.loads(agent._invoke_tool(schema["name"], {}, "memory-probe"))
            assert result == {"handled": schema["name"], "args": {}}
    finally:
        agent.close()


@pytest.mark.parametrize("kind", ["frontend", "state_writer"])
def test_eager_catalog_rejects_initialized_context_owner(peer, monkeypatch, kind):
    from agent.context_compressor import ContextCompressor
    from agui_adapter.session import (
        StateWriterSpec,
        ToolNameCollisionError,
        build_run_agent,
    )
    from tests.agui_adapter.test_modernization_probe import config_for

    name = f"agui_context_owner_{kind}"
    schema = {"name": name, "parameters": {"type": "object", "properties": {}}}
    monkeypatch.setattr(ContextCompressor, "get_tool_schemas", lambda self: [schema])
    config = config_for(peer, ["context_engine"])
    agent = build_run_agent(config)
    try:
        assert isinstance(agent.context_compressor, ContextCompressor)
        assert name in agent._context_engine_tool_names
        assert {t["function"]["name"]: t["function"] for t in agent.tools}[
            name
        ] == schema
        assert name in agent.valid_tool_names
    finally:
        agent.close()
    declaration = (
        {"frontend_tool_names": {name}}
        if kind == "frontend"
        else {"state_writer_specs": {name: StateWriterSpec(state_key="x", arg="x")}}
    )
    with pytest.raises(ToolNameCollisionError, match=name):
        build_run_agent(config, **declaration)
