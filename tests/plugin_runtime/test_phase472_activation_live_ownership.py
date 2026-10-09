"""Ownership checks for Phase 4.7 step 2 live activation extraction."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RETIRED_LIVE_ACTIVATION = ROOT / "hermes_cli" / "plugins_activation_live.py"
LIVE_FUNCTIONS = {
    "connect_plugin_mcp",
    "_utility_suffixes",
    "_server_error",
    "plugin_skills",
    "live_notice",
    "_tool_listing",
}


def _defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_live_activation_is_runtime_owned() -> None:
    import plugin_runtime.activation_live as activation_live

    assert LIVE_FUNCTIONS <= _defined_functions(Path(activation_live.__file__))


def test_activation_uses_runtime_live_owner() -> None:
    import plugin_runtime.activation as activation

    source = Path(activation.__file__).read_text(encoding="utf-8")
    assert (
        "from plugin_runtime.activation_live import "
        "connect_plugin_mcp, live_notice, plugin_skills"
    ) in source
    assert "hermes_cli.plugins_activation_live" not in source


def test_live_activation_has_no_cli_plugin_monolith_backedge() -> None:
    imports = _imports(ROOT / "plugin_runtime" / "activation_live.py")
    assert "hermes_cli.plugins" not in imports
    assert "hermes_cli.plugins_activation_live" not in imports


def test_retired_cli_live_activation_cannot_return() -> None:
    assert not RETIRED_LIVE_ACTIVATION.exists()


def test_first_party_code_has_no_retired_live_activation_dependency() -> None:
    violations = []
    for package in (
        "agent",
        "gateway",
        "hermes_cli",
        "plugin_runtime",
        "plugins",
        "tools",
        "tui_gateway",
    ):
        for path in (ROOT / package).rglob("*.py"):
            if "hermes_cli.plugins_activation_live" in path.read_text(encoding="utf-8"):
                violations.append(str(path.relative_to(ROOT)))
    assert violations == []

def test_plugin_skills_reads_runtime_manager(monkeypatch) -> None:
    import plugin_runtime.activation_live as activation_live
    import plugin_runtime.lifecycle as lifecycle

    class Manager:
        _plugin_skills = {
            "demo:alpha": {
                "plugin_key": "demo",
                "description": "Alpha skill",
            },
            "demo:beta": {
                "plugin_key": "demo",
                "description": "",
            },
            "other:skill": {
                "plugin_key": "other",
                "description": "Other",
            },
        }

    monkeypatch.setattr(lifecycle, "get_plugin_manager", lambda: Manager())

    assert activation_live.plugin_skills("demo") == [
        {"name": "demo:alpha", "description": "Alpha skill"},
        {"name": "demo:beta", "description": ""},
    ]
