"""Approval follows the final middleware payload across tool dispatch owners."""

import copy
import json
from collections import Counter
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("path", ["model", "invoke_registry", "invoke_inline", "managed_agent"])
@pytest.mark.parametrize("allowed", [True, False])
def test_approval_handoff_preserves_single_execution(tmp_path, monkeypatch, path, allowed):
    from hermes_cli import plugins
    import model_tools
    from agent import inline_tool_executors, tool_executor
    from agent.agent_runtime_helpers import invoke_tool
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    directory = home / "plugins" / "approval_fixture"
    directory.mkdir(parents=True)
    (home / "config.yaml").write_text("plugins:\n  enabled: [approval_fixture]\n", encoding="utf-8")
    (directory / "plugin.yaml").write_text(
        "name: approval_fixture\nversion: 1.0.0\ndescription: local approval fixture\n",
        encoding="utf-8",
    )
    (directory / "__init__.py").write_text('''import copy
import json
from collections import Counter
CALLS = Counter()
EXECUTED = []
def handler(args, **kw):
    EXECUTED.append(copy.deepcopy(args))
    return json.dumps({'ok': True})
def approve(**kw):
    CALLS['pre'] += 1
    return {'action': 'approve', 'message': 'confirm fixture'}
def request(**kw):
    CALLS['request'] += 1
    return kw['args']
def rewrite(**kw):
    CALLS['execution'] += 1
    return kw['next_call']({**kw['args'], 'revision': kw['args']['revision'] + 1})
def post(**kw):
    CALLS['post'] += 1
def transform(**kw):
    CALLS['transform'] += 1
def register(ctx):
    ctx.register_hook('pre_tool_call', approve)
    ctx.register_hook('post_tool_call', post)
    ctx.register_hook('transform_tool_result', transform)
    ctx.register_middleware('tool_request', request)
    ctx.register_middleware('tool_execution', rewrite)
    ctx.register_tool(name='approval_echo', toolset='approval_fixture',
        schema={'name': 'approval_echo', 'description': 'fixture',
                'parameters': {'type': 'object', 'properties': {}}}, handler=handler)
''', encoding="utf-8")
    bundled = tmp_path / "empty_bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    monkeypatch.setattr(plugins, "discover_entrypoint_manifests", list)
    manager = plugins.PluginManager()
    manager.discover_and_load()
    assert manager._plugins["approval_fixture"].enabled
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    handler = model_tools.registry.get_entry("approval_echo", scope=manager.scope_key).handler
    fixture = handler.__globals__
    approved = []

    def gate(name, reason, **kw):
        approved.append(json.loads(reason.split("(secrets redacted): ", 1)[1]))
        return {"approved": allowed, "message": "fixture declined"}

    monkeypatch.setattr("tools.approval.request_tool_approval", gate)
    agent = SimpleNamespace(
        session_id="", valid_tool_names={"approval_echo"},
        _current_turn_id="fixture-turn", _memory_manager=None,
    )
    args = {"path": "fixture-target", "revision": 0}
    original = copy.deepcopy(args)
    if path == "model":
        result = model_tools.handle_function_call("approval_echo", args)
    elif path == "managed_agent":
        agent._tool_guardrails = SimpleNamespace(
            before_call=lambda *_: SimpleNamespace(allows_execution=True),
        )
        monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_: None)
        monkeypatch.setattr(tool_executor, "_run_with_activity_heartbeat", lambda agent, name, fn: fn())
        result = tool_executor._run_agent_tool_execution_middleware(
            agent, function_name="approval_echo", function_args=args,
            effective_task_id="fixture-task", tool_call_id="fixture-call",
            execute=lambda final: invoke_tool(
                agent, "approval_echo", final, "fixture-task", pre_tool_block_checked=True,
                skip_tool_request_middleware=True, skip_tool_execution_middleware=True,
            ),
        ).result
    else:
        if path == "invoke_inline":
            monkeypatch.setattr(
                inline_tool_executors, "resolve_invoke_tool_executor",
                lambda *_: lambda agent, payload, ctx: handler(payload),
            )
        result = invoke_tool(agent, "approval_echo", args, "fixture-task")

    final = {"path": "fixture-target", "revision": 1}
    assert approved == [final]
    assert fixture["EXECUTED"] == ([final] if allowed else [])
    assert fixture["CALLS"] == Counter(
        request=1, execution=1, pre=1, post=1, transform=int(allowed),
    )
    assert json.loads(result) == ({"ok": True} if allowed else {"error": "fixture declined"})
    assert args == original
