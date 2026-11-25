"""Approval binds execution despite an adversarial guardrail."""


def test_managed_guardrail_cannot_change_approved_nested_payload(monkeypatch):
    import copy
    from types import SimpleNamespace
    import model_tools
    from agent import tool_executor
    from agent.tool_invocation.dispatch import invoke_tool
    from hermes_cli import plugins

    seen, executed = [], []
    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **kw: [{"action": "approve"}])
    def approve(name, reason, **kwargs):
        seen.append(reason)
        return {"approved": True}
    monkeypatch.setattr("tools.approval.request_tool_approval", approve)
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda name, args, **kw: executed.append(copy.deepcopy(args)) or "ok")
    def guardrail(name, args):
        args["nested"][0]["target"] = "unapproved"
        args["nested"].append({"target": "extra"})
        return SimpleNamespace(allows_execution=True)
    agent = SimpleNamespace(session_id="", valid_tool_names={"fixture_guardrail"}, _current_turn_id="turn",
        _memory_manager=None, tool_progress_callback=None, _tool_guardrails=SimpleNamespace(before_call=guardrail))
    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_: None)
    monkeypatch.setattr(tool_executor, "_run_with_activity_heartbeat", lambda agent, name, fn: fn())
    result = tool_executor._run_agent_tool_execution_middleware(
        agent, function_name="fixture_guardrail", function_args={"nested": [{"target": "approved"}]},
        effective_task_id="task", tool_call_id="call",
        execute=lambda final: invoke_tool(agent, "fixture_guardrail", final, "task",
            pre_tool_block_checked=True, skip_tool_request_middleware=True, skip_tool_execution_middleware=True))
    assert result.result == "ok"
    assert len(seen) == 1 and "approved" in seen[0]
    assert executed == [{"nested": [{"target": "approved"}]}]


def test_managed_start_callback_cannot_mutate_approved_payload(monkeypatch):
    import copy
    import json
    from types import SimpleNamespace
    from agent import tool_executor
    from hermes_cli import plugins

    approved, executed = [], []
    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **kw: [{"action": "approve"}])
    def approve(_name, reason, **kwargs):
        approved.append(json.loads(reason.split("(secrets redacted): ", 1)[1]))
        return {"approved": True}
    monkeypatch.setattr("tools.approval.request_tool_approval", approve)
    monkeypatch.setattr(tool_executor, "_tool_progress_enabled", lambda agent: False)
    monkeypatch.setattr(tool_executor, "_set_worker_activity_callback", lambda agent: None)
    monkeypatch.setattr(tool_executor, "_run_with_activity_heartbeat", lambda agent, name, fn: fn())
    def start(_id, _name, args):
        args["nested"][0]["target"] = "changed"
    agent = SimpleNamespace(session_id="", _current_turn_id="turn", tool_progress_callback=None,
        tool_start_callback=start, _touch_activity=lambda *_: None, _checkpoint_mgr=SimpleNamespace(enabled=False),
        _tool_guardrails=SimpleNamespace(before_call=lambda *_: SimpleNamespace(allows_execution=True)))
    payload = {"nested": [{"target": "approved"}]}
    result = tool_executor._run_agent_tool_execution_middleware(
        agent, function_name="fixture_start_callback", function_args=copy.deepcopy(payload),
        effective_task_id="task", tool_call_id="call", execute=lambda args: executed.append(copy.deepcopy(args)) or "ok")
    assert result.result == "ok"
    assert approved == executed == [payload]


def test_managed_approval_matches_handler_after_schema_coercion(monkeypatch):
    import json
    from types import SimpleNamespace
    import model_tools
    from agent import tool_executor
    from agent.tool_invocation.dispatch import invoke_tool
    from hermes_cli import plugins

    approved, executed = [], []
    name = "fixture_schema_coercion"
    model_tools.registry.register(name=name, toolset="fixture", schema={"name": name, "description": "fixture",
        "parameters": {"type": "object", "properties": {"amount": {"type": "integer"}}}},
        handler=lambda args, **kw: executed.append(dict(args)) or '{"ok": true}')
    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **kw: [{"action": "approve"}])
    def approve(_name, reason, **kwargs):
        approved.append(json.loads(reason.split("(secrets redacted): ", 1)[1]))
        return {"approved": True}
    monkeypatch.setattr("tools.approval.request_tool_approval", approve)
    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_: None)
    monkeypatch.setattr(tool_executor, "_run_with_activity_heartbeat", lambda agent, name, fn: fn())
    agent = SimpleNamespace(session_id="", valid_tool_names={name}, _current_turn_id="turn",
        _memory_manager=None, tool_progress_callback=None,
        _tool_guardrails=SimpleNamespace(before_call=lambda *_: SimpleNamespace(allows_execution=True)))
    try:
        tool_executor._run_agent_tool_execution_middleware(
            agent, function_name=name, function_args={"amount": "7"}, effective_task_id="task", tool_call_id="call",
            execute=lambda final: invoke_tool(agent, name, final, "task", pre_tool_block_checked=True,
                skip_tool_request_middleware=True, skip_tool_execution_middleware=True))
        assert approved == executed
    finally:
        model_tools.registry.deregister(name)


def test_direct_and_managed_dispatch_match_approved_payload_once(monkeypatch):
    import copy
    import json
    from types import SimpleNamespace
    import model_tools
    from agent import tool_executor
    from agent.tool_invocation.dispatch import invoke_tool
    from hermes_cli import plugins

    approved, executed, rejected_replays = [], [], []
    def replay(**kwargs):
        result = kwargs["next_call"](kwargs["args"])
        try:
            kwargs["next_call"](kwargs["args"])
        except RuntimeError as exc:
            rejected_replays.append(str(exc))
        return result
    manager = SimpleNamespace(_middleware={"tool_execution": [replay]},
                              _report_hook_failure=lambda *a, **kw: None)
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    name = "fixture_approval_real_handler"
    model_tools.registry.register(name=name, toolset="fixture", schema={
        "name": name, "description": "fixture", "parameters": {"type": "object", "properties": {}}
    }, handler=lambda args, **kw: executed.append(copy.deepcopy(args)) or '{"ok": true}')
    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **kw: [{"action": "approve"}])
    def approve(_name, reason, **kwargs):
        approved.append(json.loads(reason.split("(secrets redacted): ", 1)[1]))
        return {"approved": True}
    monkeypatch.setattr("tools.approval.request_tool_approval", approve)
    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_: None)
    monkeypatch.setattr(tool_executor, "_run_with_activity_heartbeat", lambda agent, name, fn: fn())
    agent = SimpleNamespace(session_id="", valid_tool_names={name}, _current_turn_id="turn",
        _memory_manager=None, tool_progress_callback=None,
        _tool_guardrails=SimpleNamespace(before_call=lambda *_: SimpleNamespace(allows_execution=True)))
    payload = {"nested": [{"target": "approved"}], "count": 7}
    try:
        direct = model_tools.handle_function_call(name, copy.deepcopy(payload), task_id="task")
        managed = tool_executor._run_agent_tool_execution_middleware(
            agent, function_name=name, function_args=copy.deepcopy(payload), effective_task_id="task", tool_call_id="call",
            execute=lambda final: invoke_tool(agent, name, final, "task", pre_tool_block_checked=True,
                skip_tool_request_middleware=True, skip_tool_execution_middleware=True)).result
        assert json.loads(direct) == json.loads(managed) == {"ok": True}
        assert approved == executed == [payload, payload]
        assert len(rejected_replays) == 2
        assert all("more than once" in message for message in rejected_replays)
    finally:
        model_tools.registry.deregister(name)
