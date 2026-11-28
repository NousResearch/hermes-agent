"""Approval binds even when schema coercion requires multiple passes."""

import pytest


def test_approved_payload_rejects_nested_mutation():
    import json
    from tools.arg_coercion import ApprovedToolArgs, coerce_tool_args

    payload = ApprovedToolArgs("example", json.dumps({"nested": {"value": 1}}))
    payload["nested"]["value"] = 2
    with pytest.raises(ValueError, match="changed after approval"):
        coerce_tool_args("example", payload)


def test_managed_nested_coercion_matches_execution(monkeypatch):
    import json
    from types import SimpleNamespace
    import model_tools
    from agent import tool_executor
    from agent.tool_invocation.dispatch import invoke_tool
    from hermes_cli import plugins

    approved, executed = [], []
    name = "fixture_nested_coercion"
    schema = {"name": name, "description": "fixture", "parameters": {"type": "object", "properties": {
        "items": {"type": "array", "items": {"type": "object"}}}}}
    model_tools.registry.register(name=name, toolset="fixture", schema=schema,
        handler=lambda args, **kw: executed.append(args) or '{"ok":true}')
    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **kw: [{"action": "approve"}])
    monkeypatch.setattr("tools.approval.request_tool_approval", lambda name, reason, **kw:
        approved.append(json.loads(reason.split("(secrets redacted): ", 1)[1])) or {"approved": True})
    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_: None)
    monkeypatch.setattr(tool_executor, "_run_with_activity_heartbeat", lambda agent, name, fn: fn())
    agent = SimpleNamespace(session_id="", valid_tool_names={name}, _current_turn_id="turn",
        _memory_manager=None, tool_progress_callback=None,
        _tool_guardrails=SimpleNamespace(before_call=lambda *_: SimpleNamespace(allows_execution=True)))
    try:
        tool_executor._run_agent_tool_execution_middleware(agent, function_name=name,
            function_args={"items": json.dumps([json.dumps({"x": 1})])}, effective_task_id="task", tool_call_id="call",
            execute=lambda final: invoke_tool(agent, name, final, "task", pre_tool_block_checked=True,
                skip_tool_request_middleware=True, skip_tool_execution_middleware=True))
        assert approved == executed
        assert len(executed) == 1
    finally:
        model_tools.registry.deregister(name)


@pytest.mark.parametrize("path", ["registry", "direct", "managed", "inline", "managed_inline"])
def test_schema_change_after_approval_does_not_change_payload(monkeypatch, path):
    import json
    from types import SimpleNamespace
    import model_tools
    from agent import tool_executor, inline_tool_executors
    from agent.tool_invocation.dispatch import invoke_tool
    from hermes_cli import plugins

    approved, executed = [], []
    name = "fixture_approval_schema_change"
    prop = {"type": "string"}
    schema = {"name": name, "description": "fixture", "parameters": {
        "type": "object", "properties": {"amount": prop}}}
    model_tools.registry.register(name=name, toolset="fixture", schema=schema,
        handler=lambda args, **kw: executed.append(args) or '{"ok":true}')
    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **kw: [{"action": "approve"}])
    def approve(_name, reason, **kw):
        approved.append(json.loads(reason.split("(secrets redacted): ", 1)[1]))
        prop["type"] = "integer"
        return {"approved": True}
    monkeypatch.setattr("tools.approval.request_tool_approval", approve)
    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_: None)
    monkeypatch.setattr(tool_executor, "_run_with_activity_heartbeat", lambda agent, name, fn: fn())
    if "inline" in path:
        monkeypatch.setattr(inline_tool_executors, "resolve_invoke_tool_executor", lambda *_:
            inline_tool_executors._coerced(name, lambda agent, args, ctx: executed.append(args) or '{"ok":true}'))
    agent = SimpleNamespace(session_id="", valid_tool_names={name}, _current_turn_id="turn",
        _memory_manager=None, tool_progress_callback=None,
        _tool_guardrails=SimpleNamespace(before_call=lambda *_: SimpleNamespace(allows_execution=True)))
    try:
        if path == "registry":
            model_tools.handle_function_call(name, {"amount": "7"}, "task")
        elif path.startswith("managed"):
            tool_executor._run_agent_tool_execution_middleware(agent, function_name=name,
                function_args={"amount": "7"}, effective_task_id="task", tool_call_id="call",
                execute=lambda final: invoke_tool(agent, name, final, "task", pre_tool_block_checked=True,
                    skip_tool_request_middleware=True, skip_tool_execution_middleware=True))
        else:
            invoke_tool(agent, name, {"amount": "7"}, "task")
        assert approved == executed == [{"amount": "7"}]
    finally:
        model_tools.registry.deregister(name)
