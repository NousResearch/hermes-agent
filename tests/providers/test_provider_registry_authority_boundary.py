"""Architecture guard for Phase 5.2 provider-registry authority."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVIDER_ROOT = ROOT / "providers"
REGISTRY_OWNER = PROVIDER_ROOT / "registry.py"
AUTH_PROJECTION = ROOT / "hermes_cli" / "provider_auth.py"
CATALOG_PROJECTION = ROOT / "hermes_cli" / "provider_catalog.py"
AUTH_SURFACES = (ROOT / "hermes_cli" / "auth.py", ROOT / "hermes_cli" / "auth_commands.py")
SCAN_ROOTS = ("acp_adapter", "agent", "auth", "automation", "cron", "gateway", "hermes_cli",
              "kanban", "plugins", "profiles", "providers", "runtime", "storage", "tools", "tui_gateway")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _sources():
    for dirname in SCAN_ROOTS:
        base = ROOT / dirname
        if base.exists():
            yield from base.rglob("*.py")


def _imports(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_registry_state_is_private_to_provider_package():
    violations = []
    for path in _sources():
        if path.is_relative_to(PROVIDER_ROOT):
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom) and node.module == "providers.registry":
                violations.append(str(path.relative_to(ROOT)))
            elif isinstance(node, ast.Import) and any(a.name == "providers.registry" for a in node.names):
                violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_registry_owner_has_no_cli_dependency():
    assert not any(m == "hermes_cli" or m.startswith("hermes_cli.") for m in _imports(REGISTRY_OWNER))


def test_auth_projection_projects_without_registry_ownership():
    tree = _tree(AUTH_PROJECTION)
    imports = _imports(AUTH_PROJECTION)
    assert "providers" in imports
    assert "providers.registry" not in imports
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "register_provider" for n in ast.walk(tree))


def test_catalog_is_presentation_projection_not_provider_authority():
    tree = _tree(CATALOG_PROJECTION)
    imports = _imports(CATALOG_PROJECTION)
    assert "providers" in imports
    assert "models.catalog_static" not in imports
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "register_provider" for n in ast.walk(tree))


def test_auth_surfaces_do_not_redefine_provider_alias_identity():
    violations = []
    for path in AUTH_SURFACES:
        for node in ast.walk(_tree(path)):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == "_PROVIDER_ALIASES" for t in targets):
                    violations.append(str(path.relative_to(ROOT)))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_plugin_aliases":
                violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_provider_config_construction_stays_at_owned_projection_or_alt_auth_mode():
    allowed = {AUTH_PROJECTION, ROOT / "hermes_cli" / "model_setup_flows_bedrock.py"}
    violations = []
    for path in _sources():
        if path in allowed:
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ProviderConfig":
                violations.append(str(path.relative_to(ROOT)))
    assert violations == []
