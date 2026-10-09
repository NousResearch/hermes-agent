"""Phase 5.8.6.8: freeze cross-surface semantic ownership and explicit residual debt.

Provider discovery, local-route recovery and credential acquisition still have
narrow application imports; pin each remaining site so none becomes a second
model-selection or routing authority while 5.8.7/Phase 6 migrates them.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SURFACES = ("tui_gateway", "acp_adapter", "hermes_cli/web_routers")
_FORBIDDEN = (
    "hermes_cli.model_switch", "hermes_cli.model_selection",
    "hermes_cli.models", "hermes_cli.providers",
    "hermes_cli.context_switch_guard",
)
# Every permitted old-application import in non-CLI consumers is classified.
# canonical_custom_identity/resolve_requested_provider are provider recovery
# debt for 5.8.7; credential resolution belongs to Phase 6.
_ALLOWED = {
    ("tui_gateway/agent_factory.py", "hermes_cli.runtime_provider",
     "resolve_runtime_provider"),
    ("tui_gateway/methods_complete.py", "hermes_cli.inventory",
     "build_model_options_payload"),
    ("tui_gateway/methods_complete.py", "hermes_cli.inventory",
     "build_models_payload"),
    ("tui_gateway/methods_complete_helpers.py", "hermes_cli.inventory",
     "load_picker_context"),
    ("tui_gateway/methods_config.py", "hermes_cli.runtime_provider",
     "resolve_runtime_provider"),
    ("tui_gateway/model_switch.py", "hermes_cli.runtime_provider",
     "resolve_runtime_provider"),
    ("tui_gateway/model_switch_resolution.py", "hermes_cli.runtime_provider",
     "resolve_runtime_provider"),
    ("acp_adapter/auth.py", "hermes_cli.runtime_provider",
     "resolve_runtime_provider"),
    ("acp_adapter/model_catalog.py", "hermes_cli.inventory",
     "build_models_payload"),
    ("acp_adapter/model_catalog.py", "hermes_cli.inventory",
     "load_picker_context"),
    ("acp_adapter/model_switch_resolution.py", "hermes_cli.runtime_provider",
     "resolve_runtime_provider"),
    ("acp_adapter/session.py", "hermes_cli.runtime_provider",
     "resolve_runtime_provider"),
    ("hermes_cli/web_routers/models.py", "hermes_cli.inventory",
     "build_model_options_payload"),
    ("hermes_cli/web_routers/models.py", "hermes_cli.inventory",
     "build_models_payload"),
    ("hermes_cli/web_routers/models.py", "hermes_cli.inventory",
     "load_picker_context"),
}


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def _imports(path: Path):
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                yield node.module, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, ""


def test_all_tui_acp_and_dashboard_handlers_reject_cli_selection_authority():
    offenders = []
    for folder in _SURFACES:
        for path in (ROOT / folder).rglob("*.py"):
            for module, name in _imports(path):
                forbidden = any(
                    module == entry or module.startswith(entry + ".")
                    for entry in _FORBIDDEN
                )
                if forbidden:
                    offenders.append(f"{path.relative_to(ROOT)}: {module}.{name}")
    assert offenders == []


def test_residual_inventory_and_runtime_imports_are_an_exact_audited_set():
    found = set()
    for folder in _SURFACES:
        for path in (ROOT / folder).rglob("*.py"):
            for module, name in _imports(path):
                if module in ("hermes_cli.inventory", "hermes_cli.runtime_provider"):
                    found.add((path.relative_to(ROOT).as_posix(), module, name))
    assert found == _ALLOWED


def test_dashboard_request_and_selection_have_no_cli_model_policy():
    for relative in (
        "hermes_cli/web_server_config.py",
        "application_dashboard_model_selection.py",
        "application_dashboard_model_detection.py",
        "application_model_selection_defaults.py",
    ):
        modules = set(_imports(ROOT / relative))
        assert ("hermes_cli.model_switch", "switch_model") not in modules
        assert all(module != "hermes_cli.model_selection_defaults" for module, _ in modules)
    dashboard = set(_imports(ROOT / "hermes_cli/web_server_config.py"))
    assert ("application_dashboard_model_selection",
            "select_dashboard_main_model") in dashboard
    assert ("application_model_switch_persistence",
            "apply_model_selection") in dashboard


def test_lower_domains_never_acquire_credentials_or_depend_on_applications():
    for relative in (
        "models/catalog_configured.py", "models/catalog_endpoint.py",
        "models/selection.py", "providers/configured.py",
        "providers/routing.py",
    ):
        for module, name in _imports(ROOT / relative):
            assert not module.startswith((
                "hermes_cli", "agent", "gateway", "tui_gateway",
                "acp_adapter", "application_",
            )), (relative, module, name)
    for relative in (
        "application_model_command_request.py",
        "application_model_switch_enrichment.py",
        "application_model_switch_persistence.py",
        "application_model_selection_guards.py",
        "application_model_selection_defaults.py",
    ):
        assert not any(module.startswith(("gateway", "tui_gateway", "acp_adapter"))
                       for module, _ in _imports(ROOT / relative)), relative


def test_removed_selection_owner_stays_removed_without_a_compatibility_shim():
    assert not (ROOT / "hermes_cli/model_selection_defaults.py").exists()
    source = (ROOT / "application_model_selection_defaults.py").read_text(
        encoding="utf-8"
    )
    assert "select_default_model" in source
    assert "select_nous_default_model" in source
    for folder in ("hermes_cli", "tui_gateway", "acp_adapter", "gateway"):
        for path in (ROOT / folder).rglob("*.py"):
            assert not any(
                module == "hermes_cli.model_selection_defaults"
                for module, _ in _imports(path)
            ), path.relative_to(ROOT)
