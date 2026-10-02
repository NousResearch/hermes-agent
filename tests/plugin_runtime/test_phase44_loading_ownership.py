"""Phase 4.4 loading ownership gates."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RETIRED_LOADER = ROOT / "hermes_cli" / "plugins_loader.py"


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_loaded_plugin_is_runtime_owned_and_cli_export_is_canonical():
    import hermes_cli.plugins as plugins
    import plugin_runtime.loading as loading

    assert plugins.LoadedPlugin is loading.LoadedPlugin


def test_loaded_plugin_fields_and_defaults_are_preserved():
    from plugin_runtime.loading import LoadedPlugin
    from plugin_runtime.manifest import PluginManifest

    manifest = PluginManifest(name="fixture", key="fixture", source="user")
    loaded = LoadedPlugin(manifest=manifest)

    assert [field.name for field in fields(LoadedPlugin)] == [
        "manifest",
        "module",
        "tools_registered",
        "hooks_registered",
        "middleware_registered",
        "commands_registered",
        "enabled",
        "error",
        "deferred",
    ]
    assert loaded.manifest is manifest
    assert loaded.module is None
    assert loaded.tools_registered == []
    assert loaded.hooks_registered == []
    assert loaded.middleware_registered == []
    assert loaded.commands_registered == []
    assert loaded.enabled is False
    assert loaded.error is None
    assert loaded.deferred is False

    other = LoadedPlugin(manifest=manifest)
    assert loaded.tools_registered is not other.tools_registered
    assert loaded.hooks_registered is not other.hooks_registered
    assert loaded.middleware_registered is not other.middleware_registered
    assert loaded.commands_registered is not other.commands_registered


def test_plugin_manager_supplies_runtime_owned_context(tmp_path):
    from plugin_runtime.manager import PluginManager
    import hermes_cli.plugins as plugins
    from plugin_runtime.context import PluginContext, PluginToolOverrideError
    from plugin_runtime.manifest import PluginManifest

    manager = PluginManager(scope_key=str(tmp_path))
    manifest = PluginManifest(name="fixture", key="fixture", source="user")

    context = manager.context_for(manifest)

    assert type(context) is PluginContext
    assert plugins.PluginContext is PluginContext
    assert plugins.PluginToolOverrideError is PluginToolOverrideError
    assert context.manifest is manifest
    assert context._manager is manager


def test_cli_module_does_not_define_context_contracts():
    import hermes_cli.plugins as plugins

    tree = ast.parse(Path(plugins.__file__).read_text(encoding="utf-8"))
    class_names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }

    assert "PluginContext" not in class_names
    assert "PluginToolOverrideError" not in class_names


def test_runtime_loading_uses_context_factory_without_cli_context_import():
    import inspect

    import plugin_runtime.loading as loading

    source = inspect.getsource(loading)
    assert "PluginContext(" not in source
    assert "from hermes_cli.plugins import PluginContext" not in source
    assert "context_for(manifest)" in source


def test_runtime_loading_declares_only_narrow_context_protocol():
    import inspect

    import plugin_runtime.loading as loading

    source = inspect.getsource(loading)
    assert "from hermes_cli.plugins import" not in source
    assert "import hermes_cli.plugins" not in source
    assert {
        "_abandon_load",
        "_tool_override_allowed",
        "register_skill",
    } <= set(loading.PluginLoadContext.__dict__)


def test_all_loading_paths_are_runtime_owned():
    import plugin_runtime.loading as loading

    methods = {
        "_warn_python_dependencies",
        "_validate_plugin_config_schema",
        "_load_plugin",
        "_load_plugin_scoped",
        "_track_tool_override_policy",
        "_attribute_registrations",
        "_platform_name_from_manifest",
        "_register_deferred_platform",
        "_lease_deferred_platform",
        "_register_deferred_platform_tools",
        "_load_portable_plugin",
        "_directory_module_name",
        "_policy_module_name",
        "_load_directory_module",
        "_load_entrypoint_module",
    }
    assert methods <= set(loading.PluginLoaderMixin.__dict__)


def test_plugin_manager_inherits_canonical_runtime_loader_directly():
    from plugin_runtime.manager import PluginManager
    import hermes_cli.plugins as plugins
    import plugin_runtime.loading as loading

    assert PluginManager.__bases__[0] is loading.PluginLoaderMixin
    assert plugins.PluginLoaderMixin is loading.PluginLoaderMixin


def test_cli_manager_supplies_compatibility_policy_seam():
    from plugin_runtime.manager import PluginManager
    import hermes_cli.plugins as plugins
    import plugin_runtime.loading as loading

    assert "_plugin_load_disable_reason" in PluginManager.__dict__
    assert "_plugin_load_disable_reason" not in loading.PluginLoaderMixin.__dict__


def test_activation_notifications_remain_manager_owned():
    from plugin_runtime.manager import PluginManager
    import hermes_cli.plugins as plugins
    import plugin_runtime.loading as loading

    assert "on_plugin_loaded" in PluginManager.__dict__
    assert "_notify_plugin_loaded" in PluginManager.__dict__
    assert "on_plugin_loaded" not in loading.PluginLoaderMixin.__dict__
    assert "_notify_plugin_loaded" not in loading.PluginLoaderMixin.__dict__


def test_runtime_loader_has_no_cli_back_edges():
    imports = _imports(ROOT / "plugin_runtime" / "loading.py")

    forbidden = {
        "hermes_cli.plugins",
        "hermes_cli.plugins_loader",
        "hermes_cli.config",
        "hermes_cli.plugin_compat",
        "hermes_cli.agent_plugins",
        "hermes_cli.plugins_state",
        "hermes_cli.plugins_activation",
    }
    assert imports.isdisjoint(forbidden)


def test_retired_cli_loader_cannot_return():
    assert not RETIRED_LOADER.exists()


def test_first_party_production_has_no_plugins_loader_dependency():
    violations = []
    for package in ("hermes_cli", "plugin_runtime"):
        for path in (ROOT / package).rglob("*.py"):
            if "hermes_cli.plugins_loader" in path.read_text(encoding="utf-8"):
                violations.append(str(path.relative_to(ROOT)))

    assert violations == []


def test_portable_runtime_compat_facade_reexports_canonical_objects():
    import hermes_cli.agent_plugins as legacy
    import plugin_runtime.portable as portable

    assert legacy.load_agent_plugin is portable.load_agent_plugin
    assert legacy.read_agent_plugin_manifest is portable.read_agent_plugin_manifest
    assert legacy.liveness_for is portable.liveness_for
    assert legacy._set_liveness is portable._set_liveness
    assert legacy._clear_liveness is portable._clear_liveness
    assert legacy._discover_mcp is portable._discover_mcp
    assert legacy._server_declarations is portable._server_declarations
    assert legacy.AgentPluginError is portable.AgentPluginError
    assert legacy.PLUGIN_SCHEMA_V1 == portable.PLUGIN_SCHEMA_V1
    assert legacy.MCP_SCHEMA_V1 == portable.MCP_SCHEMA_V1
    assert hasattr(legacy, "has_enabled_agent_plugin_mcp")
    assert not hasattr(portable, "has_enabled_agent_plugin_mcp")


def test_runtime_portable_loading_has_no_agent_plugins_cli_back_edge():
    import inspect

    import plugin_runtime.loading as loading
    import plugin_runtime.manifest as manifest

    assert "hermes_cli.agent_plugins" not in inspect.getsource(loading)
    assert "hermes_cli.agent_plugins" not in inspect.getsource(manifest)
