"""Phase 4.7 step 3: manager/activation ownership loop closure."""

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


def test_manager_depends_on_runtime_activation_directly() -> None:
    imports = _imports(ROOT / "plugin_runtime" / "manager.py")
    assert "plugin_runtime.activation" in imports
    assert "hermes_cli.plugins_activation" not in imports
    assert "hermes_cli.plugins_activation_live" not in imports


def test_runtime_activation_modules_do_not_depend_on_cli_activation_edge() -> None:
    for relative in ("activation.py", "activation_live.py"):
        imports = _imports(ROOT / "plugin_runtime" / relative)
        assert "hermes_cli.plugins_activation" not in imports
        assert "hermes_cli.plugins_activation_live" not in imports


def test_plugin_runtime_has_no_old_activation_namespace_reference() -> None:
    violations = []
    for path in (ROOT / "plugin_runtime").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "hermes_cli.plugins_activation" in source:
            violations.append(str(path.relative_to(ROOT)))
    assert violations == []
