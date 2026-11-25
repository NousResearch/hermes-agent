"""Managed tool dispatch snapshots the exact payload shown for approval."""


def test_managed_nested_mutation_during_approval(monkeypatch):
    import copy
    from types import SimpleNamespace
    from agent import tool_executor
    from agent.tool_invocation.dispatch import invoke_tool
    from hermes_cli import plugins, middleware
    import model_tools

    payload = {"nested": [{"target": "approved"}]}
    dispatched, middleware_calls = [], []
    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **kw: [{"action": "approve"}])
    def answer(name, reason, **kwargs):
        assert "approved" in reason
        payload["nested"][0]["target"] = "changed"
        return {"approved": True}
    monkeypatch.setattr("tools.approval.request_tool_approval", answer)
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda name, args, **kw: dispatched.append(copy.deepcopy(args)) or "ok")
    def run_middleware(name, args, next_call, **kwargs):
        middleware_calls.append(name)
        return next_call(args)
    monkeypatch.setattr(middleware, "run_tool_execution_middleware", run_middleware)
    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_: None)
    monkeypatch.setattr(tool_executor, "_run_with_activity_heartbeat", lambda agent, name, fn: fn())
    agent = SimpleNamespace(session_id="", valid_tool_names={"fixture_snapshot"}, _current_turn_id="turn",
                            _memory_manager=None, tool_progress_callback=None,
                            _tool_guardrails=SimpleNamespace(before_call=lambda *_: SimpleNamespace(allows_execution=True)))
    tool_executor._run_agent_tool_execution_middleware(
        agent, function_name="fixture_snapshot", function_args=payload,
        effective_task_id="task", tool_call_id="call",
        execute=lambda final: invoke_tool(agent, "fixture_snapshot", final, "task",
            pre_tool_block_checked=True, skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True))
    assert dispatched == [{"nested": [{"target": "approved"}]}]
    assert middleware_calls == ["fixture_snapshot"]
