"""Architecture guards for Phase 5.6 runtime-route ownership."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUTING_OWNER = ROOT / "providers" / "routing.py"

SCAN_ROOTS = (
    "agent",
    "gateway",
    "hermes_cli",
    "plugins",
    "runtime",
    "tui_gateway",
)

OBSOLETE_ROUTE_DEFINITIONS = {
    "determine_api_mode",
    "host_mandated_api_mode",
    "nous_api_mode",
    "_detect_api_mode_for_url",
    "_fallback_api_mode",
    "_resolve_api_mode",
    "_maybe_apply_codex_app_server_runtime",
}

OLD_ROUTE_IMPORTS = {
    "determine_api_mode",
    "host_mandated_api_mode",
    "nous_api_mode",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _sources():
    for dirname in SCAN_ROOTS:
        base = ROOT / dirname
        if base.exists():
            yield from base.rglob("*.py")


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _top_level_definitions(path: Path) -> set[str]:
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_routing_domain_exists_and_does_not_depend_on_upper_layers():
    assert ROUTING_OWNER.exists()
    forbidden = ("agent", "hermes_cli", "gateway", "tui_gateway")
    violations = [
        module
        for module in sorted(_imports(ROUTING_OWNER))
        if any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden)
    ]
    assert violations == []


def test_obsolete_runtime_route_authorities_are_deleted():
    offenders: list[tuple[str, str]] = []
    for path in _sources():
        for name in sorted(_top_level_definitions(path) & OBSOLETE_ROUTE_DEFINITIONS):
            offenders.append((str(path.relative_to(ROOT)), name))
    assert offenders == []


def test_consumers_do_not_import_old_cli_route_helpers():
    offenders: list[tuple[str, str]] = []
    for path in _sources():
        for node in _tree(path).body:
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module == "hermes_cli.providers"
            ):
                continue
            for alias in node.names:
                if alias.name in OLD_ROUTE_IMPORTS:
                    offenders.append((str(path.relative_to(ROOT)), alias.name))
    assert offenders == []


def test_agent_model_route_heuristics_are_deleted():
    path = ROOT / "run_agent.py"
    defs = {
        node.name
        for node in ast.walk(_tree(path))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_model_requires_responses_api" not in defs
    assert "_provider_model_requires_responses_api" not in defs


def test_config_boundary_does_not_own_api_mode_alias_authority():
    path = ROOT / "hermes_cli" / "config_providers.py"
    defs = _top_level_definitions(path)
    assignments = {
        target.id
        for node in _tree(path).body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    assert "_canonical_api_mode" not in defs
    assert "_API_MODE_ALIASES" not in assignments
