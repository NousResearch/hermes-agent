"""Contract tests for the bundled Microsoft 365 Plugin."""

from __future__ import annotations

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def test_permissions_are_derived_from_capability_flags():
    from plugins.microsoft365.backend import Microsoft365Settings, required_permissions

    settings = Microsoft365Settings.from_mapping({"capabilities": {
        "outlook": True, "sharepoint": False, "calendar": True,
        "teams": False, "planner": True,
    }})

    assert required_permissions(settings) == {
        "Mail.ReadWrite", "Calendars.ReadWrite", "Tasks.ReadWrite"
    }


def test_preflight_is_side_effect_free_and_redacts_configuration():
    from plugins.microsoft365.backend import Microsoft365Settings, preflight

    result = preflight(Microsoft365Settings.from_mapping({
        "tenant_id": "tenant-123", "client_id": "client-123", "client_secret": "do-not-return",
        "capabilities": {"outlook": True},
    }), sdk_available=False)

    assert result["ready"] is False
    assert result["required_permissions"] == ["Mail.ReadWrite"]
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
    assert "broader Microsoft 365 ecosystem" in loaded.manifest.description
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
