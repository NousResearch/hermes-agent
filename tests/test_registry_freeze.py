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


@pytest.fixture
def frozen_tool_registry(monkeypatch):
    """A tool registry with one registration and one plugin policy made before the freeze."""
    from tools.registry import ToolRegistry
    monkeypatch.setattr(registry_freeze, "_frozen", False)
    reg = ToolRegistry()
    schema = _schema("t")
    schema["parameters"]["properties"]["path"] = {"type": "string", "enum": ["a"]}
    reg.register("t", "ts", schema, lambda args, **kw: "{}", requires_env=["T_KEY"])
    policy = reg.register_plugin_override_policy("hermes_plugins.p", True)
    freeze_registries()
    return reg, schema, policy


def test_policy_and_restore_paths_refuse_after_freeze(frozen_tool_registry):
    """Each takes valid pre-freeze state, so only its own guard stops it from changing the registry."""
    reg, _schema_arg, policy = frozen_tool_registry
    entry = reg.get_entry("t")
    # The method's own guard, not the sealed storage underneath it, refuses.
    guard = "the tool registry is frozen"
    with pytest.raises(RegistryFrozen, match=guard):
        reg.register_plugin_override_policy("hermes_plugins.p", False)
    with pytest.raises(RegistryFrozen, match=guard):
        reg.restore_plugin_override_policy("hermes_plugins.p", policy, None)
    with pytest.raises(RegistryFrozen, match=guard):
        reg.restore_registration("t", entry, None)
    assert reg.snapshot_plugin_override_policy("hermes_plugins.p") is policy
    assert reg.snapshot_registration("t") is entry


def _writes_through_reads(reg):
    entry = reg.get_entry("t")
    return {
        "getter schema": lambda: reg.get_schema("t").__setitem__("description", "after"),
        "nested schema": lambda: reg.get_schema("t")["parameters"]["properties"]["path"]["enum"].append("b"),
        "entry attribute": lambda: setattr(entry, "handler", lambda args, **kw: "hijacked"),
        "entry requires_env": lambda: entry.requires_env.append("OTHER"),
        "bulk entries": lambda: reg.get_all_entries()[0].schema.update(description="after"),
        "global map": lambda: reg._tools.pop("t"),
        "overlay map": lambda: reg._scoped_tools.setdefault("/p", {}),
        "toolset checks": lambda: reg._toolset_checks.__setitem__("ts", lambda: False),
        "aliases": lambda: reg._toolset_aliases.__setitem__("a", "ts"),
        "override policy": lambda: reg._plugin_override_policy.clear(),
        "module scopes": lambda: reg._plugin_module_scopes["hermes_plugins.p"].add("/other"),
    }


@pytest.mark.parametrize("path", [
    "getter schema", "nested schema", "entry attribute", "entry requires_env", "bulk entries",
    "global map", "overlay map", "toolset checks", "aliases", "override policy", "module scopes"])
def test_tool_registrations_are_immutable_after_freeze(frozen_tool_registry, path):
    """A reference obtained from a read cannot change a registration other tenants see."""
    reg, _schema_arg, _policy = frozen_tool_registry
    with pytest.raises(RegistryFrozen):
        _writes_through_reads(reg)[path]()
    assert reg.get_schema("t")["description"] == "d"
    assert reg.get_schema("t")["parameters"]["properties"]["path"]["enum"] == ["a"]
    assert reg.get_entry("t").requires_env == ["T_KEY"]
    assert reg.dispatch("t", {}) == "{}"


def test_registration_argument_is_detached_at_freeze(frozen_tool_registry):
    """The dict a module passed to ``register()`` stays the module's; changing it is not a
    registry change."""
    reg, schema_arg, _policy = frozen_tool_registry
    schema_arg["description"] = "after"
    schema_arg["parameters"]["properties"]["path"]["enum"].append("b")
    assert reg.get_schema("t")["description"] == "d"
    assert reg.get_schema("t")["parameters"]["properties"]["path"]["enum"] == ["a"]


def test_frozen_reads_hand_out_writable_copies(frozen_tool_registry):
    """Request shaping rewrites the definitions it gets; that stays possible on a copy, and a
    registered entry keeps its identity (restore and compare-and-swap paths compare with ``is``)."""
    import copy
    import json
    reg, _schema_arg, _policy = frozen_tool_registry
    entry = reg.get_entry("t")
    definition = reg.get_definitions({"t"})[0]
    definition["function"]["parameters"]["properties"]["extra"] = {"type": "string"}
    clone = copy.deepcopy(reg.get_schema("t"))
    clone["parameters"]["properties"]["path"]["enum"].append("b")
    assert reg.get_entry("t") is entry
    assert "extra" not in reg.get_schema("t")["parameters"]["properties"]
    assert json.loads(json.dumps(reg.get_schema("t")))["parameters"]["properties"]["path"]["enum"] == ["a"]


def test_bundled_catalog_reads_unchanged_after_freeze(monkeypatch):
    """``bundled_profiles`` collects into a private list without touching the registry, so the
    freeze must not turn its collection into refused registrations."""
    import providers
    monkeypatch.setattr(registry_freeze, "_frozen", False)
    before = providers.bundled_profiles()
    assert before
    freeze_registries()
    after = providers.bundled_profiles()
    assert [p.name for p in after] == [p.name for p in before]


def test_schema_constructor_cannot_rewrite_frozen_schema(frozen_tool_registry):
    """Re-running ``dict.__init__`` on a schema a getter returned is a write like any other."""
    reg, _schema_arg, _policy = frozen_tool_registry
    with pytest.raises(RegistryFrozen):
        reg.get_schema("t").__init__({"description": "after"})
    with pytest.raises(RegistryFrozen):
        reg.get_entry("t").requires_env.__init__(["OTHER"])
    assert reg.get_schema("t")["description"] == "d"
    assert reg.get_entry("t").requires_env == ["T_KEY"]


@pytest.fixture
def frozen_provider(monkeypatch):
    """A provider profile registered before the freeze, with the containers its plugin passed."""
    import providers
    from providers.base import ProviderProfile
    monkeypatch.setattr(registry_freeze, "_frozen", False)
    providers.list_providers()
    for name in ("_REGISTRY", "_ALIASES", "_SOURCES"):  # the seal replaces the maps; restore them
        monkeypatch.setattr(providers, name, dict(getattr(providers, name)))
    headers = {"X-Probe": "before"}
    profile = ProviderProfile(name="freeze-probe", display_name="before", aliases=("fp",),
                              default_headers=headers, model_aliases={"m": ["a"]})
    providers.register_provider(profile)
    freeze_registries()
    return providers, profile, headers


def test_provider_payload_refuses_after_freeze(frozen_provider):
    """A profile a lookup returned is the registered one: changing it after the freeze would change
    every tenant's provider, so its fields and nested containers refuse."""
    providers, profile, _headers = frozen_provider
    got = providers.get_provider_profile("freeze-probe")
    assert got is profile  # identity kept: identity-keyed memos still match
    with pytest.raises(RegistryFrozen):
        got.display_name = "after"
    with pytest.raises(RegistryFrozen):
        del got.base_url
    with pytest.raises(RegistryFrozen):
        got.default_headers["X-Probe"] = "after"
    with pytest.raises(RegistryFrozen):
        got.model_aliases["m"].append("b")
    with pytest.raises(RegistryFrozen):
        providers._REGISTRY["freeze-probe"] = None
    with pytest.raises(RegistryFrozen):
        providers._ALIASES.pop("fp")
    assert got.display_name == "before" and got.default_headers == {"X-Probe": "before"}
    assert providers.get_provider_profile("fp") is profile
    assert type(got).__name__ == "ProviderProfile"  # overrides checks compare class attributes
    assert dict(got.default_headers) == {"X-Probe": "before"}  # copies stay writable


def test_provider_registration_argument_is_detached_at_freeze(frozen_provider):
    """The containers a plugin passed to its profile stay the plugin's after the freeze."""
    providers, _profile, headers = frozen_provider
    headers["X-Probe"] = "after"
    assert providers.get_provider_profile("freeze-probe").default_headers == {"X-Probe": "before"}
