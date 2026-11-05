"""The tool, MCP, provider and plugin registries are immutable once the hosted boot freezes them.

Reads keep working after the freeze; every registration path raises ``RegistryFrozen``. A local
process never freezes, so loading at runtime keeps working there.
"""

from __future__ import annotations

import pytest

import registry_freeze
from registry_freeze import RegistryFrozen, freeze_registries, registries_frozen


@pytest.fixture
def frozen(monkeypatch):
    # The freeze is one-way for a process; the test thaws its own module flag on teardown.
    monkeypatch.setattr(registry_freeze, "_frozen", False)
    freeze_registries()
    assert registries_frozen()
    yield


def _schema(name):
    return {"name": name, "description": "d", "parameters": {"type": "object", "properties": {}}}


def test_tool_registry_refuses_changes_after_freeze(monkeypatch):
    from tools.registry import ToolRegistry
    monkeypatch.setattr(registry_freeze, "_frozen", False)
    reg = ToolRegistry()
    reg.register("t_before", "ts", _schema("t_before"), lambda args, **kw: "{}")
    freeze_registries()
    with pytest.raises(RegistryFrozen):
        reg.register("t_after", "ts", _schema("t_after"), lambda args, **kw: "{}")
    with pytest.raises(RegistryFrozen):
        reg.deregister("t_before")
    with pytest.raises(RegistryFrozen):
        reg.register_toolset_alias("alias", "ts")
    with pytest.raises(RegistryFrozen):
        reg.move_to_scope(["t_before"], "/some/profile")
    assert reg.get_entry("t_before") is not None  # reads are unaffected
    assert reg.get_entry("t_after") is None


def test_provider_registry_refuses_changes_after_freeze(monkeypatch):
    from providers import get_provider_profile, list_providers, register_provider
    from providers.base import ProviderProfile
    monkeypatch.setattr(registry_freeze, "_frozen", False)
    assert list_providers()  # boot discovers first, then freezes
    freeze_registries()
    with pytest.raises(RegistryFrozen, match="provider registry"):
        register_provider(ProviderProfile(name="frozen-test-provider"))
    assert get_provider_profile("frozen-test-provider") is None


def test_mcp_registry_refuses_changes_after_freeze(frozen):
    from tools.mcp_tool_discovery import register_mcp_servers
    # An empty map: refused before any server could be spawned, whatever the guard's position.
    with pytest.raises(RegistryFrozen, match="MCP registry"):
        register_mcp_servers({})


def test_plugin_registry_refuses_changes_after_freeze(frozen):
    from hermes_cli.plugins import PluginManager
    from hermes_cli.plugins_manifest import PluginManifest
    manager = PluginManager()
    # The plugin registry refuses before the plugin reaches any registry it would write into.
    with pytest.raises(RegistryFrozen, match="plugin registry"):
        manager._load_plugin(PluginManifest(name="late", source="user"))
    with pytest.raises(RegistryFrozen, match="plugin registry"):
        manager.unload()


def test_unfrozen_registry_accepts_runtime_registration(monkeypatch):
    """Local mode never freezes: register and deregister at runtime keep working."""
    from tools.registry import ToolRegistry
    monkeypatch.setattr(registry_freeze, "_frozen", False)
    reg = ToolRegistry()
    reg.register("t_live", "ts", _schema("t_live"), lambda args, **kw: "{}")
    reg.deregister("t_live")
    assert reg.get_entry("t_live") is None
