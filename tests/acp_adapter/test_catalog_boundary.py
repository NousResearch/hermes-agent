"""5.8.6.4: ACP catalogue reads only public model/provider facts."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
        elif isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
    return found


def test_acp_catalogue_has_no_cli_private_model_imports():
    modules = _imports(ROOT / "acp_adapter" / "model_catalog.py")
    assert not [
        module for module in modules
        if any(module == name or module.startswith(name + ".") for name in (
            "hermes_cli.model_switch", "application_provider_discovery",
            "hermes_cli.models", "hermes_cli.models_local",
        ))
    ]
    assert "hermes_cli.inventory" in modules  # shared app projection; 5.8.7 extraction
    assert "models.catalog_configured" in modules
    assert "models.catalog_endpoint" in modules


def test_public_endpoint_and_configured_queries_have_no_application_imports():
    for name in ("catalog_configured.py", "catalog_endpoint.py"):
        path = ROOT / "models" / name
        assert not [
            module for module in _imports(path)
            if module.startswith(("hermes_cli", "acp_adapter", "gateway", "tui_gateway"))
        ], name
