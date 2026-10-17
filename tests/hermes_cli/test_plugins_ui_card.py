"""PluginContext.register_tool forwards ``ui_card`` to the tool registry."""

from hermes_cli.plugins import PluginManager


def test_register_tool_forwards_ui_card(tmp_path, monkeypatch):
    """`ui_card="delegate"` lands on the registry entry."""
    from tools.registry import registry
    from hermes_cli.plugins import PluginContext, PluginManifest

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))
    mgr = PluginManager()
    ctx = PluginContext(manager=mgr, manifest=PluginManifest(name="ui_card_plugin", source="user"))
    try:
        ctx.register_tool(
            name="ui_card_fanout",
            toolset="ui_card_plugin",
            schema={"name": "ui_card_fanout", "description": "x", "parameters": {"type": "object", "properties": {}}},
            handler=lambda args, **kw: "{}",
            ui_card="delegate",
        )
        assert registry.get_entry("ui_card_fanout").ui_card == "delegate"
        assert registry.get_ui_card("ui_card_fanout") == "delegate"
    finally:
        registry.deregister("ui_card_fanout", scope=mgr.scope_key)
