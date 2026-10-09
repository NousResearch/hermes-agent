"""Phase 4.1 ownership gate for plugin-runtime leaf contracts."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CAPABILITIES = ROOT / "plugin_runtime" / "capabilities.py"
MANIFEST = ROOT / "plugin_runtime" / "manifest.py"
STATE = ROOT / "plugin_runtime" / "state.py"
CONFIG_BRIDGE = ROOT / "plugin_runtime" / "config_bridge.py"
DEBUG_STATE = ROOT / "plugin_runtime" / "debug.py"

RETIRED_CLI_OWNERS = (
    ROOT / "hermes_cli" / "plugin_capabilities.py",
    ROOT / "hermes_cli" / "plugins_manifest.py",
)
SETTINGS_BRIDGE = ROOT / "hermes_cli" / "plugins_state.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(path: Path) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_phase41_runtime_leaf_owners_exist_and_retired_cli_owners_stay_retired():
    owners = (CAPABILITIES, MANIFEST, STATE, CONFIG_BRIDGE, DEBUG_STATE)
    missing = [str(path.relative_to(ROOT)) for path in owners if not path.exists()]
    returned = [
        str(path.relative_to(ROOT)) for path in RETIRED_CLI_OWNERS if path.exists()
    ]

    assert missing == []
    assert returned == []


def test_phase41_capabilities_routes_config_only_through_bridge():
    imports = _imported_modules(CAPABILITIES)

    assert "plugin_runtime.config_bridge" in imports
    assert "hermes_cli.config" not in imports


def test_phase41_manifest_has_no_plugins_facade_back_edge():
    imports = _imported_modules(MANIFEST)

    assert "plugin_runtime.capabilities" in imports
    assert "plugin_runtime.debug" in imports
    assert "hermes_cli.plugins" not in imports


def test_phase41_durable_state_has_no_cli_dependency():
    imports = _imported_modules(STATE)

    assert not any(
        module == "hermes_cli" or module.startswith("hermes_cli.")
        for module in imports
    )


def test_phase41_cli_state_module_is_settings_only():
    tree = _tree(SETTINGS_BRIDGE)
    classes = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    top_level_names = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }

    assert "PluginState" not in classes
    assert "_PLUGIN_STATE_KEY_RE" not in top_level_names
    assert "_PLUGIN_STATE_QUOTA_BYTES" not in top_level_names


def test_phase41_first_party_sources_do_not_import_retired_leaf_modules():
    retired = {
        "hermes_cli.plugin_capabilities",
        "hermes_cli.plugins_manifest",
    }
    violations: list[tuple[str, int, str]] = []
    source_roots = (
        "agent",
        "cron",
        "gateway",
        "hermes_cli",
        "plugins",
        "providers",
        "tools",
        "tui_gateway",
    )

    for root_name in source_roots:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            for node in ast.walk(_tree(path)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in retired:
                            violations.append(
                                (str(path.relative_to(ROOT)), node.lineno, alias.name)
                            )
                elif isinstance(node, ast.ImportFrom) and node.module in retired:
                    violations.append(
                        (str(path.relative_to(ROOT)), node.lineno, node.module)
                    )

    assert violations == []
