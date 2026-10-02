"""Phase 4.2 ownership gates for discovery config and Relay policy."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DISCOVERY = ROOT / "plugin_runtime" / "discovery.py"
RETIRED_DISCOVERY = ROOT / "hermes_cli" / "plugins_discovery.py"
COMPAT_MANIFEST = ROOT / "compat_manifest.json"
RELAY_COMPAT = ROOT / "hermes_cli" / "relay_plugin_cutover.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    tree = _tree(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_discovery_config_and_relay_dependencies_are_runtime_owned():
    imports = _imports(DISCOVERY)

    assert "plugin_runtime.config_bridge" in imports
    assert "plugin_runtime.relay_policy" in imports
    assert not any(
        module == "hermes_cli" or module.startswith("hermes_cli.")
        for module in imports
    )


def test_retired_cli_discovery_module_cannot_return():
    assert not RETIRED_DISCOVERY.exists()


def test_first_party_discovery_consumers_use_runtime_owner():
    violations = []
    for package in ("hermes_cli", "plugin_runtime"):
        for path in (ROOT / package).rglob("*.py"):
            if "hermes_cli.plugins_discovery" in path.read_text(encoding="utf-8"):
                violations.append(str(path.relative_to(ROOT)))

    assert violations == []


def test_compat_manifest_points_discovery_export_at_runtime_owner():
    manifest = json.loads(COMPAT_MANIFEST.read_text(encoding="utf-8"))
    entry = next(
        item for item in manifest["entries"]
        if item.get("facade") == "hermes_cli.plugins"
        and item.get("name") == "ENTRY_POINT_CAPABILITIES_GROUP"
    )

    assert entry["target"] == "plugin_runtime.discovery"


def test_primary_discovery_consumers_bind_canonical_runtime_exports():
    import hermes_cli.plugins as plugins
    import plugin_runtime.discovery as discovery
    import plugin_runtime.loading as loading

    assert plugins.collect_directory_manifests is discovery.collect_directory_manifests
    assert plugins.discover_entrypoint_manifests is discovery.discover_entrypoint_manifests
    assert plugins.gate_manifest is discovery.gate_manifest
    assert plugins.resolve_manifest_winners is discovery.resolve_manifest_winners
    assert plugins.scan_directory is discovery.scan_directory
    assert loading._select_entry_point_group is discovery._select_entry_point_group
    assert loading.ENTRY_POINTS_GROUP == discovery.ENTRY_POINTS_GROUP


def test_cli_relay_cutover_module_is_compatibility_only():
    imports = _imports(RELAY_COMPAT)

    assert imports == {"plugin_runtime.relay_policy"}


def test_first_party_relay_consumers_use_runtime_policy():
    violations = []
    for path in (ROOT / "hermes_cli").rglob("*.py"):
        if path == RELAY_COMPAT:
            continue
        if "hermes_cli.relay_plugin_cutover" in _imports(path):
            violations.append(str(path.relative_to(ROOT)))

    assert violations == []


def test_compatibility_relay_exports_are_canonical():
    import hermes_cli.relay_plugin_cutover as compat
    import plugin_runtime.relay_policy as policy

    assert compat.LEGACY_RELAY_PLUGIN_KEYS is policy.LEGACY_RELAY_PLUGIN_KEYS
    assert compat.LEGACY_RELAY_EXPORT_ENV_VARS is policy.LEGACY_RELAY_EXPORT_ENV_VARS
    assert compat.legacy_relay_plugin_keys is policy.legacy_relay_plugin_keys
    assert compat.configured_legacy_relay_env_vars is policy.configured_legacy_relay_env_vars


def test_catalog_recall_lookup_runs_only_after_runtime_gates():
    from plugin_runtime.discovery import gate_manifest
    from plugin_runtime.manifest import PluginManifest

    calls = []

    class Removed:
        reason = "fixture recall"

    def lookup(name, path):
        calls.append((name, path))
        return Removed()

    bundled = PluginManifest(
        name="bundled-backend", key="bundled-backend", source="bundled", kind="backend",
    )
    disabled = PluginManifest(name="disabled", key="disabled", source="user", path="/disabled")
    exclusive = PluginManifest(
        name="exclusive", key="exclusive", source="user", path="/exclusive", kind="exclusive",
    )
    not_enabled = PluginManifest(
        name="not-enabled", key="not-enabled", source="user", path="/not-enabled",
    )

    assert gate_manifest(
        bundled, set(), set(), installed_plugin_removal=lookup,
    ).action == "load_now"
    assert gate_manifest(
        disabled, {"disabled"}, {"disabled"}, installed_plugin_removal=lookup,
    ).error == "disabled via config"
    assert gate_manifest(
        exclusive, set(), {"exclusive"}, installed_plugin_removal=lookup,
    ).error.startswith("exclusive plugin")
    assert gate_manifest(
        not_enabled, set(), set(), installed_plugin_removal=lookup,
    ).error.startswith("not enabled in config")
    assert calls == []

    enabled = PluginManifest(name="enabled", key="enabled", source="user", path="/enabled")
    verdict = gate_manifest(
        enabled, set(), {"enabled"}, installed_plugin_removal=lookup,
    )

    assert calls == [("enabled", "/enabled")]
    assert verdict.action == "placeholder"
    assert verdict.error == "removed from the Hermes plugin catalog: fixture recall"


def test_plugin_manager_supplies_cli_owned_catalog_lookup(monkeypatch):
    from hermes_cli import plugins_cmd_catalog
    from plugin_runtime.manager import PluginManager
    from plugin_runtime.manifest import PluginManifest

    calls = []

    class Removed:
        reason = "manager fixture"

    def lookup(name, path):
        calls.append((name, path))
        return Removed()

    monkeypatch.setattr(plugins_cmd_catalog, "installed_plugin_removal", lookup)
    manager = PluginManager()
    manifest = PluginManifest(name="fixture", key="fixture", source="user", path="/fixture")

    assert manager._gate_manifest(manifest, set(), {"fixture"}) is False
    assert calls == [("fixture", "/fixture")]
    assert manager._plugins["fixture"].error == (
        "removed from the Hermes plugin catalog: manager fixture"
    )