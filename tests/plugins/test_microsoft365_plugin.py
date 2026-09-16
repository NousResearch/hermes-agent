"""Contract tests for the bundled Microsoft 365 Plugin."""

from __future__ import annotations

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def test_permissions_are_derived_from_operation_flags():
    from plugins.microsoft365.backend import Microsoft365Settings, required_permissions

    settings = Microsoft365Settings.from_mapping({"capabilities": {
        "outlook": {"search": True, "send": True},
        "sharepoint": {"read": True},
        "calendar": {"search": True},
        "teams": False,
        "planner": {"create_tasks": True},
    }})

    assert required_permissions(settings) == {
        "Mail.Read", "Mail.Send", "Sites.Read.All", "Calendars.Read", "Tasks.ReadWrite"
    }


def test_service_true_enables_all_operations_for_compatibility():
    from plugins.microsoft365.backend import Microsoft365Settings

    settings = Microsoft365Settings.from_mapping({"capabilities": {"outlook": True}})
    assert settings.operations("outlook") == {
        "search", "read", "create_draft", "send"
    }


def test_disabled_operation_is_rejected_before_graph_client_creation():
    import asyncio
    from plugins.microsoft365 import tools

    class Context:
        def get_config(self, key, default=None):
            return {"capabilities": {"outlook": {"search": True}}}.get(key, default)

    original = tools.create_graph_client
    tools.create_graph_client = lambda settings: (_ for _ in ()).throw(AssertionError("client created"))
    try:
        result = asyncio.run(tools._run("outlook", {"action": "send"}, Context()))
    finally:
        tools.create_graph_client = original
    assert "disabled" in result.lower()


def test_preflight_is_side_effect_free_and_redacts_configuration():
    from plugins.microsoft365.backend import Microsoft365Settings, preflight

    result = preflight(Microsoft365Settings.from_mapping({
        "tenant_id": "tenant-123", "client_id": "client-123", "client_secret": "do-not-return",
        "capabilities": {"outlook": True},
    }), sdk_available=False)

    assert result["ready"] is False
    assert result["required_permissions"] == ["Mail.Read", "Mail.ReadWrite", "Mail.Send"]
    assert "do-not-return" not in repr(result)
    assert result["configuration"]["client_secret"] == "[redacted]"


def test_bundled_manifest_is_loaded_by_real_plugin_discovery():
    import os
    from pathlib import Path
    import yaml
    hermes_home = Path(os.environ["HERMES_HOME"])
    (hermes_home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"entries": {
        "microsoft365": {"settings": {"capabilities": {"calendar": True}}}
    }}}))
    from hermes_cli import plugins as plugin_module
    manager = plugin_module.PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins["microsoft365"]
    assert loaded.enabled is True
    assert loaded.manifest.name == "microsoft365"
    assert "one bundled connection" in loaded.manifest.description
    assert "microsoft365_calendar" in manager._plugin_tool_names
    assert "microsoft365_outlook" not in manager._plugin_tool_names


def test_registers_only_enabled_capability_tools():
    from plugins.microsoft365 import register

    manager = PluginManager()
    context = PluginContext(PluginManifest(name="microsoft365"), manager)
    context.get_config = lambda key, default=None: {
        "capabilities": {"outlook": True, "sharepoint": False, "calendar": True,
                          "teams": False, "planner": False}
    }.get(key, default)

    register(context)

    assert set(manager._plugin_tool_names) == {
        "microsoft365_outlook", "microsoft365_calendar", "microsoft365_preflight"
    }
