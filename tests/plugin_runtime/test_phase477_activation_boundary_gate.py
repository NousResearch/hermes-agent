"""Phase 4.7 step 7: final activation/live-reload ownership gate."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ACTIVATION = (
    ROOT / "plugin_runtime" / "activation.py",
    ROOT / "plugin_runtime" / "activation_live.py",
)
RETIRED_LIVE_EDGE = ROOT / "hermes_cli" / "plugins_activation_live.py"


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _top_level_defs(path: Path) -> set[str]:
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _top_level_assignments(path: Path) -> set[str]:
    names: set[str] = set()
    for node in _tree(path).body:
        if isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_runtime_activation_does_not_import_application_layers() -> None:
    forbidden = ("gateway", "tui_gateway", "hermes_cli")
    for path in RUNTIME_ACTIVATION:
        imports = _imports(path)
        violations = {
            module
            for module in imports
            if any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden)
        }
        assert violations == set(), (path, violations)


def test_manager_uses_only_canonical_runtime_activation() -> None:
    imports = _imports(ROOT / "plugin_runtime" / "manager.py")
    assert "plugin_runtime.activation" in imports
    assert "hermes_cli.plugins_activation" not in imports
    assert "hermes_cli.plugins_activation_live" not in imports


def test_gateway_never_depends_on_cli_activation_edge() -> None:
    violations = []
    for path in (ROOT / "gateway").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        imports = _imports(path)
        if (
            "hermes_cli.plugins_activation" in imports
            or "hermes_cli.plugins_activation_live" in imports
            or "hermes_cli.plugins_activation" in source
        ):
            violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_runtime_owns_activation_and_live_mechanics() -> None:
    activation_defs = _top_level_defs(ROOT / "plugin_runtime" / "activation.py")
    live_defs = _top_level_defs(ROOT / "plugin_runtime" / "activation_live.py")

    assert {
        "plugin_activation_summary",
        "activation_summaries",
        "find_activation",
        "load_and_go_live",
        "_go_live",
    } <= activation_defs
    assert {
        "connect_plugin_mcp",
        "plugin_skills",
        "live_notice",
    } <= live_defs
    assert RETIRED_LIVE_EDGE.exists() is False


def test_cli_activation_edge_owns_only_cross_process_orchestration_and_presentation() -> None:
    path = ROOT / "hermes_cli" / "plugins_activation.py"
    source = path.read_text(encoding="utf-8")

    assert _top_level_defs(path) == {
        "activate_plugin_now",
        "_serve_backend_record",
        "notify_serve_backend",
        "activation_hint",
    }

    runtime_only_defs = {
        "plugin_activation_summary",
        "activation_summaries",
        "find_activation",
        "load_and_go_live",
        "_go_live",
        "connect_plugin_mcp",
        "plugin_skills",
        "live_notice",
        "_tool_listing",
        "_utility_suffixes",
        "_server_error",
    }
    assert runtime_only_defs.isdisjoint(_top_level_defs(path))

    runtime_state = {
        "_GO_LIVE_LOCK",
        "_GATEWAY_TRANSFORM_HOOKS",
        "_plugin_manager",
        "_plugin_managers_by_home",
        "_published_tui_message_injector",
    }
    assert runtime_state.isdisjoint(_top_level_assignments(path))

    manager_state_markers = (
        "_ownership_ledger",
        "_platform_handler_factories",
        "_portable_mcp_server_plugins",
        "_plugin_skills",
    )
    assert all(marker not in source for marker in manager_state_markers)

    assert "runtime_activation.load_and_go_live(name)" in source
    assert "runtime_activation.find_activation" in source
