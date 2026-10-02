"""Phase 4.7 step 4: Gateway startup/reload paths consume plugin_runtime directly."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GATEWAY_PLUGIN_RUNTIME_PATHS = (
    ROOT / "gateway" / "run_plugin_rewire.py",
    ROOT / "gateway" / "run_adapters.py",
    ROOT / "gateway" / "run_startup.py",
)


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_gateway_startup_paths_do_not_use_cli_plugin_facade() -> None:
    for path in GATEWAY_PLUGIN_RUNTIME_PATHS:
        imports = _imports(path)
        assert "hermes_cli.plugins" not in imports, path


def test_gateway_startup_paths_use_runtime_lifecycle() -> None:
    for path in GATEWAY_PLUGIN_RUNTIME_PATHS:
        imports = _imports(path)
        assert "plugin_runtime.lifecycle" in imports, path


def test_gateway_reload_path_uses_runtime_activation() -> None:
    imports = _imports(ROOT / "gateway" / "run_plugin_rewire.py")
    assert "plugin_runtime.activation" in imports
