"""Architecture guard for canonical inference-provider identity ownership."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IDENTITY_OWNER = ROOT / "providers" / "identity.py"
CLI_RESOLVER = ROOT / "hermes_cli" / "providers.py"
SCAN_ROOTS = (
    "acp_adapter",
    "agent",
    "auth",
    "automation",
    "cron",
    "gateway",
    "hermes_cli",
    "kanban",
    "plugins",
    "profiles",
    "providers",
    "runtime",
    "storage",
    "tools",
    "tui_gateway",
)
IDENTITY_SYMBOLS = {
    "ResolvedProvider",
    "custom_provider_aliases",
    "custom_provider_slug",
    "get_provider_label",
    "is_aggregator",
    "is_routing_aggregator",
    "normalize_provider",
}


def _python_sources():
    for dirname in SCAN_ROOTS:
        base = ROOT / dirname
        if base.exists():
            yield from base.rglob("*.py")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_provider_identity_owner_does_not_depend_on_upper_layers():
    forbidden = ("hermes_cli", "agent", "gateway", "tui_gateway")
    violations = []
    for node in ast.walk(_tree(IDENTITY_OWNER)):
        modules = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        for module in modules:
            if any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden):
                violations.append(module)
    assert violations == []


def test_cli_resolver_does_not_own_or_publicly_reexport_identity_contract():
    violations = []
    for node in _tree(CLI_RESOLVER).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in IDENTITY_SYMBOLS:
                violations.append(("definition", node.name))
        elif isinstance(node, ast.ImportFrom) and node.module == "providers":
            for alias in node.names:
                if alias.name in IDENTITY_SYMBOLS and not (alias.asname or "").startswith("_"):
                    violations.append(("public import", alias.name))
    assert violations == []


def test_consumers_import_identity_from_providers_not_cli_resolver():
    violations = []
    for path in _python_sources():
        if path == CLI_RESOLVER:
            continue
        source = path.read_text(encoding="utf-8")
        if "hermes_cli.providers" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "hermes_cli.providers":
                continue
            for alias in node.names:
                if alias.name in IDENTITY_SYMBOLS:
                    violations.append((str(path.relative_to(ROOT)), alias.name))
    assert violations == []
