"""Approval resolution faults must never reach an effectful tool handler."""

import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("route", ["managed", "direct", "agent_direct"])
@pytest.mark.parametrize("approve", [True, False])
@pytest.mark.parametrize("fault", ["coercion", "resolution", "aggregation"])
def test_resolution_failure_never_bypasses_approval(tmp_path, monkeypatch, route, approve, fault):
    import model_tools
    from agent import tool_executor
    from agent.tool_invocation.dispatch import invoke_tool
    from hermes_cli import plugins
    from hermes_constants import get_hermes_home
    from tools import arg_coercion

    home = get_hermes_home()
    directory = home / "plugins" / "fault_fixture"
    directory.mkdir(parents=True)
    (home / "config.yaml").write_text("plugins:\n  enabled: [fault_fixture]\n", encoding="utf-8")
    (directory / "plugin.yaml").write_text("name: fault_fixture\nversion: 1.0.0\ndescription: fixture\n", encoding="utf-8")
    (directory / "__init__.py").write_text('''import json
EXECUTED = []
ARMED = []
REQUIRE_APPROVAL = False
MALFORMED = False
class BadAction:
    def __eq__(self, other):
        ARMED.pop()
        raise RuntimeError('malformed action')
def malformed_hook(**kw):
    if MALFORMED and ARMED:
        return {'action': BadAction()}
def handler(args, **kw):
    EXECUTED.append(dict(args))
    return json.dumps({'ok': True})
def approve_hook(**kw):
    if REQUIRE_APPROVAL:
        ARMED.append(True)
        return {'action': 'approve'}
def forward(**kw):
    return kw['next_call'](kw['args'])
def register(ctx):
    ctx.register_hook('pre_tool_call', approve_hook)
    ctx.register_hook('pre_tool_call', malformed_hook)
    ctx.register_middleware('tool_execution', forward)
    ctx.register_tool(name='fault_echo', toolset='fault_fixture',
        schema={'name': 'fault_echo', 'description': 'fixture',
                'parameters': {'type': 'object', 'properties': {}}}, handler=handler)
''', encoding="utf-8")
    bundled = tmp_path / "empty_bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    monkeypatch.setattr(plugins, "discover_entrypoint_manifests", list)
    manager = plugins.PluginManager()
    manager.discover_and_load()
    assert manager._plugins["fault_fixture"].enabled
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    fixture = model_tools.registry.get_entry("fault_echo", scope=manager.scope_key).handler.__globals__
    executed, armed = fixture["EXECUTED"], fixture["ARMED"]
    fixture["REQUIRE_APPROVAL"] = approve
    fixture["MALFORMED"] = fault == "aggregation"
    original = arg_coercion.coerce_tool_args
    def fail_once(*args, **kwargs):
        if fault == "coercion" and armed:
            armed.pop()
            raise RuntimeError("injected transient resolution fault")
        return original(*args, **kwargs)
    monkeypatch.setattr(arg_coercion, "coerce_tool_args", fail_once)
    monkeypatch.setattr(model_tools, "coerce_tool_args", fail_once)
    resolve = plugins._resolve_block_from_details
    def fail_resolution(*args, **kwargs):
        if fault == "resolution" and armed:
            armed.pop()
            raise RuntimeError("injected gate fault")
        return resolve(*args, **kwargs)
    monkeypatch.setattr(plugins, "_resolve_block_from_details", fail_resolution)
    agent = SimpleNamespace(session_id="", valid_tool_names={"fault_echo"}, _current_turn_id="fixture-turn",
                            _memory_manager=None, tool_progress_callback=None,
                            _tool_guardrails=SimpleNamespace(before_call=lambda *_: SimpleNamespace(allows_execution=True)))
    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_: None)
    monkeypatch.setattr(tool_executor, "_run_with_activity_heartbeat", lambda agent, name, fn: fn())
    args = {"value": "7"}
    if route == "managed":
        result = tool_executor._run_agent_tool_execution_middleware(
            agent, function_name="fault_echo", function_args=args,
            effective_task_id="fixture-task", tool_call_id="fixture-call",
            execute=lambda final: invoke_tool(agent, "fault_echo", final, "fixture-task",
                pre_tool_block_checked=True, skip_tool_request_middleware=True,
                skip_tool_execution_middleware=True),
        ).result
    elif route == "direct":
        result = model_tools.handle_function_call("fault_echo", args, "fixture-task")
    else:
        result = invoke_tool(agent, "fault_echo", args, "fixture-task")
    result = json.loads(result)
    if approve:
        assert "error" in result
        assert executed == []
        assert not armed
    else:
        assert result == {"ok": True}
        assert executed == [args]
