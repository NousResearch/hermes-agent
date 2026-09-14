"""Gateway execution preserves session-toolset ownership through the real agent boundary."""

from types import SimpleNamespace


def _schema(name: str) -> dict:
    return {
        "name": name,
        "description": f"Return {name}",
        "parameters": {"type": "object", "properties": {}},
    }


def test_gateway_created_agent_executes_its_session_owned_tool() -> None:
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from run_agent import AIAgent
    from tools.registry import registry

    session_key = "agent:main:test:dm:owner"
    manager = PluginManager(scope_key=registry.current_scope_key())
    context = PluginContext(
        PluginManifest(name="gateway-session-tools", key="gateway-session-tools"), manager,
    )
    toolset = context.session_toolset(session_key, name="client-tools", direct=True)
    tool_name = toolset.register_tool(
        "lookup", _schema("lookup"), lambda _args, **_kwargs: "owned-session-result",
    )
    source = SimpleNamespace(
        user_id="owner", user_id_alt=None, user_name="Owner", chat_id="owner",
        chat_name=None, chat_type="dm", thread_id=None,
    )
    gateway_runner = SimpleNamespace(
        _session_db=None, _prefill_messages=None, _service_tier=None,
        _refresh_fallback_model=lambda: None,
    )
    turn_context = TurnContext(
        source=source, session_id="gateway-agent-session", session_key=session_key,
        user_config={}, AIAgent=AIAgent, enabled_toolsets=[toolset.name], disabled_toolsets=[],
    )
    agent = None
    try:
        agent = TurnRunner(gateway_runner, turn_context)._build_fresh_agent(
            {
                "model": "gpt-4o-mini",
                "runtime": {
                    "provider": "openai", "api_key": "test-key",
                    "base_url": "https://api.openai.com/v1",
                },
            },
            "test", None, 1, None, {}, True,
        )
        assert agent._gateway_session_key == session_key
        assert agent._invoke_tool(tool_name, {}, "gateway-tool-task") == "owned-session-result"
    finally:
        if agent is not None:
            agent.close()
        toolset.dispose()
