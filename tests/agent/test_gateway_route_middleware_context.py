from types import SimpleNamespace

from hermes_cli.middleware import RequestMiddlewareResult


def _agent():
    return SimpleNamespace(
        session_id="ephemeral-session",
        _gateway_session_key="agent:main:slack:channel:C1:thread:1700.1",
        _gateway_session_source={
            "platform": "slack",
            "scope_id": "T1",
            "chat_id": "C1",
            "thread_id": "1700.1",
            "user_id": "U2",
        },
        _current_turn_id="turn-1",
        _current_api_request_id="request-1",
        _todo_store=object(),
        valid_tool_names=[],
        enabled_toolsets=None,
        disabled_toolsets=None,
    )


def test_concurrent_tool_path_supplies_gateway_route_context(monkeypatch):
    from agent.agent_runtime_helpers import invoke_tool

    observed = {}

    def apply_request(tool_name, args, **context):
        observed["request"] = context
        return RequestMiddlewareResult(args, args)

    def run_execution(tool_name, args, next_call, **context):
        observed["execution"] = context
        return next_call(args)

    monkeypatch.setattr("hermes_cli.middleware.apply_tool_request_middleware", apply_request)
    monkeypatch.setattr("hermes_cli.middleware.run_tool_execution_middleware", run_execution)
    monkeypatch.setattr("tools.todo_tool.todo_tool", lambda **_kwargs: "ok")

    agent = _agent()
    assert invoke_tool(agent, "todo", {"todos": []}, "task-1") == "ok"
    for context in observed.values():
        assert context["session_key"] == agent._gateway_session_key
        assert context["source"] == agent._gateway_session_source
        assert context["source"] is not agent._gateway_session_source


def test_sequential_tool_path_supplies_gateway_route_context(monkeypatch):
    from agent.tool_executor import _run_agent_tool_execution_middleware

    observed = {}

    def apply_request(tool_name, args, **context):
        observed["request"] = context
        return RequestMiddlewareResult(args, args)

    def run_execution(tool_name, args, next_call, **context):
        observed["execution"] = context
        return next_call(args)

    monkeypatch.setattr("hermes_cli.middleware.apply_tool_request_middleware", apply_request)
    monkeypatch.setattr("hermes_cli.middleware.run_tool_execution_middleware", run_execution)
    agent = _agent()

    outcome = _run_agent_tool_execution_middleware(
        agent,
        function_name="terminal",
        function_args={"command": "pwd"},
        effective_task_id="task-1",
        tool_call_id="call-1",
        execute=lambda _args: "should-not-run",
        scope_block="blocked for test",
    )

    assert outcome.blocked is True
    for context in observed.values():
        assert context["session_key"] == agent._gateway_session_key
        assert context["source"] == agent._gateway_session_source
        assert context["source"] is not agent._gateway_session_source
