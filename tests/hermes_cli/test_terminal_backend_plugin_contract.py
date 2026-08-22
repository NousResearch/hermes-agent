from __future__ import annotations

from types import SimpleNamespace

import pytest


def _factory(request):
    return object()



def test_plugin_context_registers_terminal_backend_definition(monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.environments import BackendDefinition
    from tools.environments import registry as registry_module

    registry = registry_module.TerminalBackendRegistry()
    monkeypatch.setattr(registry_module, "terminal_backend_registry", registry)

    manager = PluginManager()
    manifest = PluginManifest(
        name="coder",
        key="terminal/coder",
        source="entrypoint",
        kind="backend",
    )
    context = PluginContext(manifest, manager)

    context.register_terminal_backend(
        BackendDefinition(name="coder", label="Coder", factory=_factory)
    )

    registered = registry.require("coder")
    assert registered.name == "coder"
    assert registered.source == "entrypoint"
    assert registered.config_schema is None
    assert registered.plugin_name == "terminal/coder"
    assert manager._plugin_terminal_backend_names == {"coder"}


def test_terminal_backend_release_targets_registration_profile(monkeypatch, tmp_path):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.environments import BackendDefinition
    from tools.environments import registry as registry_module

    active_home = tmp_path / "one"
    registry = registry_module.ProfileScopedTerminalBackendRegistry(
        lambda: active_home
    )
    monkeypatch.setattr(registry_module, "terminal_backend_registry", registry)
    context = PluginContext(
        PluginManifest(
            name="coder",
            key="terminal/coder",
            source="entrypoint",
            kind="backend",
        ),
        PluginManager(),
    )

    first_handle = context.register_terminal_backend(
        BackendDefinition(name="coder", factory=_factory)
    )
    first_registry = registry.current()
    active_home = tmp_path / "two"
    second_context = PluginContext(
        PluginManifest(
            name="coder",
            key="terminal/coder",
            source="entrypoint",
            kind="backend",
        ),
        PluginManager(),
    )
    second_handle = second_context.register_terminal_backend(
        BackendDefinition(name="coder", factory=_factory)
    )
    second_registered = registry.require("coder")

    first_handle.dispose()

    assert first_registry.get("coder") is None
    assert registry.require("coder") is second_registered
    second_handle.dispose()
    assert registry.get("coder") is None


def test_terminal_backend_release_uses_original_name_after_mutation(monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.environments import BackendDefinition
    from tools.environments import registry as registry_module

    registry = registry_module.TerminalBackendRegistry()
    monkeypatch.setattr(registry_module, "terminal_backend_registry", registry)
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(
            name="coder",
            key="terminal/coder",
            source="entrypoint",
            kind="backend",
        ),
        manager,
    )
    handle = context.register_terminal_backend(
        BackendDefinition(name="coder", factory=_factory)
    )
    registered = registry.require("coder")

    registered.name = "renamed"
    with pytest.raises(registry_module.BackendDefinitionMutatedError):
        registry.list_definitions()

    handle.dispose()

    assert registry.get("coder") is None
    assert manager._plugin_terminal_backend_names == set()


def test_force_discovery_removes_plugin_terminal_backend_definitions(monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.environments import BackendDefinition
    from tools.environments import registry as registry_module

    registry = registry_module.TerminalBackendRegistry()
    monkeypatch.setattr(registry_module, "terminal_backend_registry", registry)

    manager = PluginManager()
    context = PluginContext(
        PluginManifest(
            name="coder",
            key="terminal/coder",
            source="entrypoint",
            kind="backend",
        ),
        manager,
    )
    context.register_terminal_backend(BackendDefinition(name="coder", factory=_factory))
    manager._discovered = True
    monkeypatch.setattr(manager, "_discover_and_load_inner", lambda: None)

    manager.discover_and_load(force=True)

    assert registry.get("coder") is None
    assert manager._plugin_terminal_backend_names == set()


def test_failed_plugin_registration_rolls_back_terminal_backends(monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.environments import BackendDefinition
    from tools.environments import registry as registry_module

    registry = registry_module.TerminalBackendRegistry()
    monkeypatch.setattr(registry_module, "terminal_backend_registry", registry)

    manager = PluginManager()
    manifest = PluginManifest(
        name="coder",
        key="terminal/coder",
        source="entrypoint",
        kind="backend",
    )

    def register_then_fail(context: PluginContext) -> None:
        context.register_terminal_backend(
            BackendDefinition(name="coder", factory=_factory)
        )
        raise RuntimeError("plugin registration failed")

    monkeypatch.setattr(
        manager,
        "_load_entrypoint_module",
        lambda ignored_manifest: SimpleNamespace(register=register_then_fail),
    )

    manager._load_plugin(manifest)

    assert registry.get("coder") is None
    assert manager._plugin_terminal_backend_names == set()
    assert manager._plugins["terminal/coder"].enabled is False
    assert manager._plugins["terminal/coder"].error == "plugin registration failed"


def test_failed_discovery_sweep_rolls_back_terminal_backends(monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.environments import BackendDefinition
    from tools.environments import registry as registry_module

    registry = registry_module.TerminalBackendRegistry()
    monkeypatch.setattr(registry_module, "terminal_backend_registry", registry)

    manager = PluginManager()
    context = PluginContext(
        PluginManifest(
            name="coder",
            key="terminal/coder",
            source="entrypoint",
            kind="backend",
        ),
        manager,
    )

    def failing_sweep() -> None:
        context.register_terminal_backend(
            BackendDefinition(name="coder", factory=_factory)
        )
        raise RuntimeError("discovery failed")

    monkeypatch.setattr(manager, "_discover_and_load_inner", failing_sweep)

    with pytest.raises(RuntimeError, match="discovery failed"):
        manager.discover_and_load()

    assert registry.get("coder") is None
    assert manager._plugin_terminal_backend_names == set()
    assert manager._discovered is False
