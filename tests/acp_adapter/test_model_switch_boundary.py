"""Phase 5.8.6.5 ACP server must not depend on CLI semantic model switching."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _imports(file):
    tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
    return imports


def test_acp_switch_owns_selection_in_its_application_and_lower_domains():
    server = _imports("acp_adapter/server.py")
    resolver = _imports("acp_adapter/model_switch_resolution.py")
    assert "acp_adapter.model_switch_resolution" in server
    assert "models.selection" in resolver
    assert "providers.routing" in resolver
    assert "hermes_cli.model_switch" not in server | resolver
    assert "hermes_cli.models" not in server | resolver
    assert "gateway.session_model_resolution" not in resolver
    assert "tui_gateway.model_switch_resolution" not in resolver


def test_runtime_acquisition_stays_phase6_only():
    resolver = _imports("acp_adapter/model_switch_resolution.py")
    assert {module for module in resolver if module.startswith("hermes_cli.")} == {
        "hermes_cli.runtime_provider"
    }


def test_acp_switch_never_writes_process_wide_model_configuration():
    source = (ROOT / "acp_adapter/server.py").read_text(encoding="utf-8")
    section = source.split("    def _switch_model(", 1)[1].split("    @staticmethod", 1)[0]
    for forbidden in ("save_config(", "persist_model_selection(", "os.environ["):
        assert forbidden not in section
