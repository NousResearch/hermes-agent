import threading
from types import SimpleNamespace


def _context(name: str):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.registry import registry

    manager = PluginManager(scope_key=registry.current_scope_key())
    return PluginContext(PluginManifest(name=name, key=name), manager)


def _schema(name: str) -> dict:
    return {
        "name": name,
        "description": f"Return {name}",
        "parameters": {"type": "object", "properties": {}},
    }


def test_session_toolsets_are_isolated_frozen_and_disposed() -> None:
    from gateway.run_turn import GatewayTurnMixin
    from model_tools import get_tool_definitions
    from tools.registry import registry

    session_a = "agent:main:test:dm:a"
    session_b = "agent:main:test:dm:b"
    toolset_a = _context("session-tools-a").session_toolset(
        session_a, name="client-tools", direct=True,
    )
    toolset_b = _context("session-tools-b").session_toolset(
        session_b, name="client-tools", direct=True,
    )
    tool_a = toolset_a.register_tool(
        "lookup", _schema("lookup"), lambda _args, **_kwargs: "a",
    )
    tool_b = toolset_b.register_tool(
        "lookup", _schema("lookup"), lambda _args, **_kwargs: "b",
    )

    class Runner(GatewayTurnMixin):
        def _adapter_for_source(self, _source):
            return None

        def _session_key_for_source(self, source):
            return source.session_key

    runner = Runner()
    source_a = SimpleNamespace(session_key=session_a)
    source_b = SimpleNamespace(session_key=session_b)
    try:
        enabled_a = runner._resolve_enabled_toolsets_for_source({}, source_a, "telegram")
        enabled_b = runner._resolve_enabled_toolsets_for_source({}, source_b, "telegram")
        assert toolset_a.name in enabled_a and toolset_b.name not in enabled_a
        assert toolset_b.name in enabled_b and toolset_a.name not in enabled_b

        names_a = {
            item["function"]["name"]
            for item in get_tool_definitions(enabled_a, quiet_mode=True)
        }
        names_b = {
            item["function"]["name"]
            for item in get_tool_definitions(enabled_b, quiet_mode=True)
        }
        assert tool_a in names_a and tool_b not in names_a
        assert tool_b in names_b and tool_a not in names_b
        assert registry.dispatch(tool_a, {}, gateway_session_key=session_a) == "a"
        assert "outside its owning session" in registry.dispatch(
            tool_a, {}, gateway_session_key=session_b
        ).lower()

        try:
            toolset_a.register_tool("late", _schema("late"), lambda *_a, **_k: "late")
        except RuntimeError as exc:
            assert "prompt-cache" in str(exc)
        else:
            raise AssertionError("frozen session toolset accepted a late registration")

        toolset_a.dispose()
        assert registry.get_entry(tool_a) is None
        assert registry.get_entry(tool_b) is not None
    finally:
        toolset_a.dispose()
        toolset_b.dispose()


def test_session_toolset_disposal_releases_ledger_and_preserves_replacement(monkeypatch) -> None:
    from tools.registry import registry

    context = _context("session-tools-lifecycle")
    manager = context._manager
    baseline_order = len(manager._registration_order)
    baseline_owned = len(manager._ownership_ledger.get(context.plugin_id, ()))

    for index in range(3):
        handle = context.session_toolset(
            f"agent:main:test:dm:churn-{index}", name="client-tools",
        )
        handle.register_tool("lookup", _schema("lookup"), lambda *_a, **_k: "ok")
        handle.dispose()
        assert len(manager._registration_order) == baseline_order
        assert len(manager._ownership_ledger.get(context.plugin_id, ())) == baseline_owned

    session_key = "agent:main:test:dm:replacement"
    retired = context.session_toolset(
        session_key, name="client-tools", description="retired", direct=True,
    )
    retired.register_tool("lookup", _schema("lookup"), lambda *_a, **_k: "retired")
    original_deregister = registry.deregister_session_toolset
    teardown_entered = threading.Event()
    finish_teardown = threading.Event()

    def delayed_deregister(toolset, metadata, *, scope):
        teardown_entered.set()
        assert finish_teardown.wait(5)
        return original_deregister(toolset, metadata, scope=scope)

    monkeypatch.setattr(registry, "deregister_session_toolset", delayed_deregister)
    thread = threading.Thread(target=retired.dispose)
    thread.start()
    replacement = None
    try:
        assert teardown_entered.wait(5)
        replacement = context.session_toolset(
            session_key, name="client-tools", description="replacement", direct=False,
        )
        replacement.register_tool("lookup", _schema("lookup"), lambda *_a, **_k: "replacement")
        finish_teardown.set()
        thread.join(5)
        assert not thread.is_alive()
        assert registry.get_plugin_toolset_description(replacement.name) == "replacement"
        assert registry.is_direct_toolset(replacement.name) is False
    finally:
        finish_teardown.set()
        thread.join(5)
        retired.dispose()
        if replacement is not None:
            replacement.dispose()

    assert len(manager._registration_order) == baseline_order
    assert len(manager._ownership_ledger.get(context.plugin_id, ())) == baseline_owned
