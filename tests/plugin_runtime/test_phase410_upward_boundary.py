"""Focused Phase 4.10 plugin-runtime upward-boundary seam tests."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_plugin_runtime_has_no_direct_cli_imports_outside_config_bridge() -> None:
    violations = []
    for path in (ROOT / "plugin_runtime").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("hermes_cli") and path.name != "config_bridge.py":
                        violations.append((path.name, node.lineno, alias.name))
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("hermes_cli")
                and path.name != "config_bridge.py"
            ):
                violations.append((path.name, node.lineno, node.module))
    assert violations == []


def test_compat_policy_uses_the_config_bridge(monkeypatch) -> None:
    import plugin_runtime.compat as compat

    seen = []
    monkeypatch.setattr(
        compat,
        "load_plugin_config_readonly",
        lambda: seen.append("read") or {"plugins": {compat.ALLOW_KEY: True}},
    )

    assert compat.allow_deprecated_imports() is True
    assert seen == ["read"]


def test_plugin_context_uses_host_bindings_for_host_owned_facades(monkeypatch) -> None:
    import plugin_runtime.context as context_module
    from plugin_runtime.context import PluginContext
    from plugin_runtime.manifest import PluginManifest
    from plugin_runtime.manager import PluginManager

    actions = object()
    callbacks = {
        "platform_actions_factory": lambda plugin_id: (actions, plugin_id),
        "builtin_auxiliary_task_keys": lambda: {"builtin"},
    }
    monkeypatch.setattr(context_module, "get_plugin_host_callback", callbacks.get)
    manager = PluginManager(scope_key="/tmp/phase410")
    context = PluginContext(PluginManifest(name="fixture", key="fixture"), manager)

    assert context.platform_actions == (actions, "fixture")
    context.register_approval_transport("custom", lambda request: request)
    assert manager._approval_transports["custom"].plugin_id == "fixture"


def test_manager_catalog_policy_is_late_bound(monkeypatch) -> None:
    import plugin_runtime.manager as manager_module

    seen = []
    monkeypatch.setattr(
        manager_module,
        "get_plugin_host_callback",
        lambda name: (lambda plugin, path: seen.append((plugin, path)) or "removed")
        if name == "installed_plugin_removal" else None,
    )
    assert manager_module._installed_plugin_removal("fixture", "/tmp/fixture") == "removed"
    assert seen == [("fixture", "/tmp/fixture")]
