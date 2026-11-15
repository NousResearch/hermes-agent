from types import SimpleNamespace

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def test_pre_agent_turn_plugin_can_allow_and_dispatch_failure_blocks():
    manager = PluginManager()
    manifest = PluginManifest(name="turn-gate-test", source="user")
    context = PluginContext(manifest, manager)
    context.register_hook("pre_agent_turn", lambda **_payload: {"action": "allow"})

    assert manager.invoke_hook("pre_agent_turn", session_id="s1") == [{"action": "allow"}]

    failing_manager = PluginManager()
    failing_context = PluginContext(manifest, failing_manager)

    def fail(**_payload):
        raise RuntimeError("host project lookup unavailable")

    failing_context.register_hook("pre_agent_turn", fail)
    decision = failing_manager.invoke_hook("pre_agent_turn", session_id="s2")

    assert len(decision) == 1
    assert decision[0]["action"] == "block"
    assert "RuntimeError" in decision[0]["message"]


def test_pre_agent_turn_callback_runs_to_completion_on_caller_thread(monkeypatch):
    import threading

    import hermes_cli.plugins

    caller = threading.current_thread()
    callback_threads = []
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="turn-gate-sync-test", source="user"), manager)

    def gate(**_payload):
        callback_threads.append(threading.current_thread())
        return {"action": "allow"}

    context.register_hook("pre_agent_turn", gate)
    monkeypatch.setattr(hermes_cli.plugins, "_resolve_hook_callback_timeout", lambda: 0.001)

    assert manager.invoke_hook("pre_agent_turn", session_id="s1") == [{"action": "allow"}]
    assert callback_threads == [caller]


def test_pre_agent_turn_implicit_none_is_an_invalid_fail_closed_decision(monkeypatch):
    from agent.turn_lifecycle_hooks import begin_agent_turn_hooks

    manager = PluginManager()
    context = PluginContext(PluginManifest(name="turn-gate-no-decision", source="user"), manager)
    context.register_hook("pre_agent_turn", lambda **_payload: None)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", manager.invoke_hook)

    started, message = begin_agent_turn_hooks(
        SimpleNamespace(platform="cli"), session_id="s1", turn_id="t1", task_id="task1"
    )

    assert started is False
    assert message == "Turn admission hook returned an invalid decision; the turn was not started."
