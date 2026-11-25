"""Adversarial check: an old session grant must not cover changed arguments."""


def test_payload_approval_refuses_permanent_choice_without_persisting(monkeypatch):
    from hermes_cli import plugins
    from tools import approval, approval_context

    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_yolo_active", lambda: False)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: "binding-permanent")
    monkeypatch.setattr(approval, "_presence", lambda callback: (callback, True, False, False))
    monkeypatch.setattr(approval, "_present_with_selected_transport", lambda **kw: None)
    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **kw: [{"action": "approve"}])
    prompts = []
    monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **kw: prompts.append(kw) or "always")
    monkeypatch.setattr(approval, "save_permanent_allowlist", lambda *_: __import__("pytest").fail("persisted"))
    try:
        blocked, _ = plugins._dispatch_pre_tool_call_hooks("example", {"pin": "1234"})
        assert blocked is not None
        assert prompts[0]["allow_permanent"] is False
    finally:
        approval.clear_session("binding-permanent")


def test_payload_approval_key_is_not_an_offline_guessable_hash(monkeypatch):
    import hashlib
    import json
    from hermes_cli import plugins

    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **kw: [{"action": "approve", "rule_key": "fixture"}])
    keys = []
    monkeypatch.setattr("tools.approval.request_tool_approval", lambda *a, **kw: keys.append(kw["rule_key"]) or {"approved": False})
    args = {"pin": "1234"}
    plugins._dispatch_pre_tool_call_hooks("example", args)
    binding = json.dumps(["example", "fixture", args], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(binding.encode()).hexdigest() not in keys[0]


def test_payload_gate_ignores_old_permanent_grant(monkeypatch):
    from hermes_cli import plugins
    from tools import approval, approval_context

    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_yolo_active", lambda: False)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: "old-permanent")
    monkeypatch.setattr(approval, "_sync_permanent_from_disk", lambda *args: None)
    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **kw: [{"action": "approve"}])
    captured = []
    real = approval.request_tool_approval
    monkeypatch.setattr(approval, "request_tool_approval", lambda *a, **kw: captured.append(kw["rule_key"]) or {"approved": False})
    plugins._dispatch_pre_tool_call_hooks("example", {"pin": "1234"})
    monkeypatch.setattr(approval, "request_tool_approval", real)
    key = "plugin_rule:" + captured[0]
    approval.approve_permanent(key)
    try:
        blocked, _ = plugins._dispatch_pre_tool_call_hooks("example", {"pin": "1234"})
        assert blocked is not None
    finally:
        with approval._lock:
            approval._permanent_set().discard(key)
        approval.clear_session("old-permanent")


def test_old_session_rule_does_not_authorize_different_payload(monkeypatch):
    from hermes_cli import plugins
    from tools import approval, approval_context

    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_yolo_active", lambda: False)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: "binding-test")
    monkeypatch.setattr(approval, "_sync_permanent_from_disk", lambda *args: None)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *a, **k: [
        {"action": "approve", "message": "confirm write", "rule_key": "fixture"}
    ])
    approval.approve_session("binding-test", "plugin_rule:fixture")
    try:
        assert approval.is_approved("binding-test", "plugin_rule:fixture")
        blocked, _ = plugins._dispatch_pre_tool_call_hooks("example", {"target": "different"})
        assert blocked is not None, "a prior generic rule silently approved different arguments"
    finally:
        approval.clear_session("binding-test")


def test_session_grant_reuses_exact_payload_only_including_secrets(monkeypatch):
    from hermes_cli import plugins
    from tools import approval, approval_context

    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_yolo_active", lambda: False)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: "binding-test")
    monkeypatch.setattr(approval, "_sync_permanent_from_disk", lambda *args: None)
    monkeypatch.setattr(approval, "_presence", lambda callback: (callback, True, False, False))
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *a, **k: [
        {"action": "approve", "message": "confirm write", "rule_key": "fixture"}
    ])
    prompts = []

    def decide(*args, **kwargs):
        prompts.append(kwargs["pattern_key"])
        if len(prompts) == 1:
            approval.approve_session("binding-test", kwargs["pattern_key"])
            return {"approved": True}
        return {"approved": False, "message": "denied"}

    monkeypatch.setattr(approval, "_human_decision", decide)
    try:
        call = plugins._dispatch_pre_tool_call_hooks
        assert call("example", {"target": "a", "secret": "one"})[0] is None
        assert call("example", {"secret": "one", "target": "a"})[0] is None
        assert call("example", {"target": "b", "secret": "one"})[0] == "denied"
        assert call("example", {"target": "a", "secret": "two"})[0] == "denied"
        assert call("other", {"target": "a", "secret": "one"})[0] == "denied"
        assert len(prompts) == 4
    finally:
        approval.clear_session("binding-test")


def test_managed_session_grant_covers_only_executed_payload(tmp_path, monkeypatch):
    """Use the real gate, cache, middleware and plugin tool; fake only the human answer."""
    import json
    from types import SimpleNamespace
    import model_tools
    from agent import tool_executor
    from agent.tool_invocation.dispatch import invoke_tool
    from hermes_cli import plugins
    from hermes_constants import get_hermes_home
    from tools import approval, approval_context

    home = get_hermes_home()
    directory = home / "plugins" / "binding_fixture"
    directory.mkdir(parents=True)
    (home / "config.yaml").write_text("plugins:\n  enabled: [binding_fixture]\n", encoding="utf-8")
    (directory / "plugin.yaml").write_text(
        "name: binding_fixture\nversion: 1.0.0\ndescription: approval test\n", encoding="utf-8")
    (directory / "__init__.py").write_text('''import json
EXECUTED = []
def handler(args, **kw):
    EXECUTED.append(dict(args))
    return json.dumps({'ok': True})
def approve(**kw):
    return {'action': 'approve', 'message': 'confirm', 'rule_key': 'fixture'}
def rewrite(**kw):
    return kw['next_call']({**kw['args'], 'revision': kw['args']['revision'] + 1})
def register(ctx):
    ctx.register_hook('pre_tool_call', approve)
    ctx.register_middleware('tool_execution', rewrite)
    ctx.register_tool(name='binding_echo', toolset='binding_fixture',
        schema={'name': 'binding_echo', 'description': 'fixture',
                'parameters': {'type': 'object', 'properties': {}}}, handler=handler)
''', encoding="utf-8")
    bundled = tmp_path / "empty_bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    monkeypatch.setattr(plugins, "discover_entrypoint_manifests", list)
    manager = plugins.PluginManager()
    manager.discover_and_load()
    assert manager._plugins["binding_fixture"].enabled
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    executed = model_tools.registry.get_entry("binding_echo", scope=manager.scope_key).handler.__globals__["EXECUTED"]
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_yolo_active", lambda: False)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: "binding-e2e")
    monkeypatch.setattr(approval, "_presence", lambda callback: (callback, True, False, False))
    monkeypatch.setattr(approval, "_present_with_selected_transport", lambda **kw: None)
    prompts = []

    def answer(command, description, **kwargs):
        prompts.append(description)
        return "session" if len(prompts) == 1 else "deny"

    monkeypatch.setattr(approval, "prompt_dangerous_approval", answer)
    agent = SimpleNamespace(session_id="", valid_tool_names={"binding_echo"}, _current_turn_id="fixture-turn",
                            _memory_manager=None, tool_progress_callback=None,
                            _tool_guardrails=SimpleNamespace(before_call=lambda *_: SimpleNamespace(allows_execution=True)))
    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_: None)
    monkeypatch.setattr(tool_executor, "_run_with_activity_heartbeat", lambda agent, name, fn: fn())

    def call(args):
        return json.loads(tool_executor._run_agent_tool_execution_middleware(
            agent, function_name="binding_echo", function_args=args,
            effective_task_id="fixture-task", tool_call_id="fixture-call",
            execute=lambda final: invoke_tool(agent, "binding_echo", final, "fixture-task",
                pre_tool_block_checked=True, skip_tool_request_middleware=True,
                skip_tool_execution_middleware=True),
        ).result)

    try:
        assert call({"revision": 0, "secret": "one"}) == {"ok": True}
        assert call({"secret": "one", "revision": 0}) == {"ok": True}
        assert "error" in call({"revision": 1, "secret": "one"})
        assert "error" in call({"revision": 0, "secret": "two"})
        assert executed == [{"revision": 1, "secret": "one"}] * 2
        assert len(prompts) == 3
        assert all("one" not in prompt and "two" not in prompt for prompt in prompts)
    finally:
        approval.clear_session("binding-e2e")
