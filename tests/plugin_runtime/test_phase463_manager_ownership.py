"""Ownership contracts for the Phase 4.6.3 PluginManager extraction."""

from __future__ import annotations

import ast
from pathlib import Path


def test_runtime_manager_is_canonical_composition_root() -> None:
    import hermes_cli.plugins as plugin_api
    from plugin_runtime.dispatch import PluginDispatchMixin
    from plugin_runtime.loading import PluginLoaderMixin
    from plugin_runtime.manager import PluginManager
    from plugin_runtime.ownership import PluginOwnershipMixin

    assert plugin_api.PluginManager is PluginManager
    assert PluginManager.__module__ == "plugin_runtime.manager"
    assert PluginManager.__bases__ == (
        PluginLoaderMixin,
        PluginDispatchMixin,
        PluginOwnershipMixin,
    )


def test_cli_plugin_module_does_not_define_manager() -> None:
    import hermes_cli.plugins as plugin_api

    tree = ast.parse(Path(plugin_api.__file__).read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ClassDef) and node.name == "PluginManager"
        for node in tree.body
    )


def test_runtime_manager_does_not_import_cli_plugin_monolith() -> None:
    import plugin_runtime.manager as runtime_manager

    tree = ast.parse(Path(runtime_manager.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "hermes_cli.plugins"
        elif isinstance(node, ast.Import):
            assert all(alias.name != "hermes_cli.plugins" for alias in node.names)


def test_cli_registry_returns_runtime_manager_and_context() -> None:
    import hermes_cli.plugins as plugin_api
    from plugin_runtime.context import PluginContext
    from plugin_runtime.manager import PluginManager
    from plugin_runtime.manifest import PluginManifest

    manager = plugin_api.get_plugin_manager()
    assert type(manager) is PluginManager

    manifest = PluginManifest(name="fixture", key="fixture", source="user")
    assert type(manager.context_for(manifest)) is PluginContext
