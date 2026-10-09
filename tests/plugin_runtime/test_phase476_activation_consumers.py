"""Phase 4.7 step 6: activation consumers use the owner matching their boundary."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_dashboard_backend_uses_in_process_runtime_activation() -> None:
    path = ROOT / "hermes_cli" / "web_routers" / "dashboard_ui.py"
    source = path.read_text(encoding="utf-8")
    imports = _imports(path)

    assert "plugin_runtime.activation" in imports
    assert "hermes_cli.plugins_activation" not in imports
    assert "load_and_go_live(name)" in source


def test_cli_plugin_mutations_keep_cross_process_edge_orchestration() -> None:
    path = ROOT / "hermes_cli" / "plugins_cmd.py"
    source = path.read_text(encoding="utf-8")

    assert "from hermes_cli.plugins_activation import activate_plugin_now" in source
    assert "plugin_runtime.activation" not in source


def test_tui_plugin_management_keeps_cross_process_edge_orchestration() -> None:
    path = ROOT / "tui_gateway" / "methods_tools.py"
    source = path.read_text(encoding="utf-8")

    assert '_tools_mod("hermes_cli.plugins_activation").activate_plugin_now' in source
    assert "from plugin_runtime.lifecycle import get_plugin_manager" in source
    assert '_tools_mod("plugin_runtime.activation")' not in source
    assert "from plugin_runtime.activation import" not in source


def test_gateway_does_not_use_activation_edge_orchestrator() -> None:
    violations = []
    for path in (ROOT / "gateway").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "hermes_cli.plugins_activation" in source:
            violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_activation_edge_delegates_in_process_work_to_runtime() -> None:
    path = ROOT / "hermes_cli" / "plugins_activation.py"
    source = path.read_text(encoding="utf-8")
    imports = _imports(path)

    assert "plugin_runtime" in imports
    assert "runtime_activation.load_and_go_live(name)" in source
    assert "runtime_activation.find_activation" in source
