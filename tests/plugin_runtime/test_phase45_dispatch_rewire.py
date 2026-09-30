"""Phase 4.5 Step 7 canonical dispatch wiring and retirement gates."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DISPATCH = ROOT / "plugin_runtime" / "dispatch.py"
RETIRED_DISPATCH = ROOT / "hermes_cli" / "plugins_dispatch.py"
FIRST_PARTY_DIRS = ("hermes_cli", "agent", "gateway", "tools", "plugin_runtime")


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_retired_cli_dispatch_module_is_absent():
    assert not RETIRED_DISPATCH.exists()


def test_plugin_manager_inherits_runtime_dispatch_directly():
    from plugin_runtime.manager import PluginManager
    import hermes_cli.plugins as plugins
    import plugin_runtime.dispatch as dispatch
    from plugin_runtime.loading import PluginLoaderMixin
    from plugin_runtime.ownership import PluginOwnershipMixin

    assert plugins.PluginDispatchMixin is dispatch.PluginDispatchMixin
    assert PluginManager.__bases__ == (
        PluginLoaderMixin,
        dispatch.PluginDispatchMixin,
        PluginOwnershipMixin,
    )


def test_runtime_dispatch_has_no_application_back_edges():
    imports = _imports(RUNTIME_DISPATCH)

    assert "plugin_runtime.config_bridge" in imports
    assert not any(
        module == "hermes_cli"
        or module.startswith("hermes_cli.")
        or module == "agent"
        or module.startswith("agent.")
        for module in imports
    )


def test_first_party_python_never_imports_retired_dispatch_module():
    offenders = []
    for dirname in FIRST_PARTY_DIRS:
        for path in (ROOT / dirname).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "plugins_dispatch" not in source:
                continue
            if "hermes_cli.plugins_dispatch" in _imports(path):
                offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_runtime_is_the_only_first_party_dispatch_mixin_owner():
    owners = []
    for dirname in FIRST_PARTY_DIRS:
        for path in (ROOT / dirname).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "class PluginDispatchMixin" not in source:
                continue
            tree = ast.parse(source, filename=str(path))
            if any(
                isinstance(node, ast.ClassDef) and node.name == "PluginDispatchMixin"
                for node in tree.body
            ):
                owners.append(path.relative_to(ROOT).as_posix())

    assert owners == ["plugin_runtime/dispatch.py"]


def test_compat_manifest_targets_runtime_dispatch():
    manifest = json.loads((ROOT / "compat_manifest.json").read_text(encoding="utf-8"))
    targets = {
        entry["name"]: entry["target"]
        for entry in manifest["entries"]
        if entry.get("facade") == "hermes_cli.plugins"
        and entry.get("name") in {
            "MAX_SYSTEM_PROMPT_SECTIONS",
            "OBSERVER_SCHEMA_VERSION",
            "format_system_prompt_section",
        }
    }

    assert targets == {
        "MAX_SYSTEM_PROMPT_SECTIONS": "plugin_runtime.dispatch",
        "OBSERVER_SCHEMA_VERSION": "plugin_runtime.dispatch",
        "format_system_prompt_section": "plugin_runtime.dispatch",
    }
