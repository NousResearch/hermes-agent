import json
from types import SimpleNamespace

from tools.registry import registry


def test_invoke_tool_forwards_session_search_profile(monkeypatch):
    import agent.agent_runtime_helpers as runtime_helpers
    import hermes_cli.middleware as middleware
    import tools.session_search_tool as session_search_tool

    calls = []

    def fake_session_search(**kwargs):
        calls.append(kwargs)
        return json.dumps({"ok": True})

    monkeypatch.setattr(session_search_tool, "session_search", fake_session_search)
    monkeypatch.setattr(
        middleware,
        "run_tool_execution_middleware",
        lambda _name, args, execute, **_kw: execute(args),
    )

    agent = SimpleNamespace(
        session_id="current-session",
        _get_session_db_for_recall=lambda: object(),
        _current_turn_id="turn",
        _current_api_request_id="api",
    )

    runtime_helpers.invoke_tool(
        agent,
        "session_search",
        {"query": "needle", "profile": "other-profile"},
        "task",
    )

    assert calls[0]["profile"] == "other-profile"


def test_generic_invoke_tool_runs_tool_execution_middleware_once(monkeypatch):
    import agent.agent_runtime_helpers as runtime_helpers
    import hermes_cli.middleware as middleware

    tool_name = "__test_dispatch_once_tool"
    try:
        registry.deregister(tool_name)
    except Exception:
        pass
    registry.register(
        name=tool_name,
        toolset="test",
        schema={
            "name": tool_name,
            "description": "test tool",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda _args, **_kw: json.dumps({"ok": True}),
        check_fn=lambda: True,
    )

    calls = []

    def spy(name, args, execute, **_kwargs):
        calls.append(name)
        return execute(args)

    monkeypatch.setattr(middleware, "run_tool_execution_middleware", spy)
    agent = SimpleNamespace(
        session_id="current-session",
        valid_tool_names={tool_name},
        enabled_toolsets=None,
        disabled_toolsets=None,
        _current_turn_id="turn",
        _current_api_request_id="api",
        _memory_manager=None,
        _context_engine_tool_names=set(),
        context_compressor=None,
    )

    try:
        assert runtime_helpers.invoke_tool(agent, tool_name, {}, "task") == json.dumps({"ok": True})
    finally:
        registry.deregister(tool_name)

    assert calls == [tool_name]
