"""5.8.6.7 web/desktop consumers: applications own choices, UI transports them."""
from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _imports(path: str) -> set[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
    return modules


def test_dashboard_read_and_write_surfaces_do_not_import_cli_selection_policy():
    for path in ("hermes_cli/web_routers/models.py", "hermes_cli/web_server_config.py"):
        imports = _imports(path)
        assert imports.isdisjoint({
            "hermes_cli.model_switch", "hermes_cli.model_selection_defaults",
            "hermes_cli.model_selection_guards", "hermes_cli.models",
        }), path
    assert "application_model_selection_defaults" in _imports(
        "hermes_cli/web_routers/models.py"
    )
    assert "models.selection" in _imports("application_model_selection_defaults.py")


def test_recommendation_still_uses_account_facts_then_lower_selection(monkeypatch):
    import application_model_selection_defaults as defaults
    from hermes_cli.web_routers import models as dashboard

    selected = SimpleNamespace(selected=SimpleNamespace(ref=SimpleNamespace(model="allowed")))
    visited = []
    monkeypatch.setattr(dashboard, "_config_profile_scope",
                        lambda profile: (visited.append(profile), nullcontext())[1])
    monkeypatch.setattr(defaults, "select_nous_recommended_default",
                        lambda: (selected, True))
    assert dashboard.get_recommended_default_model("nous", profile="team") == {
        "provider": "nous", "model": "allowed", "free_tier": True,
    }
    assert visited == ["team"]


def test_recommended_other_provider_uses_scoped_inventory_and_shared_default(monkeypatch):
    import application_model_selection_defaults as defaults
    import hermes_cli.inventory as inventory
    from hermes_cli.web_routers import models as dashboard

    visited = []
    monkeypatch.setattr(dashboard, "_config_profile_scope",
                        lambda profile: (visited.append(profile), nullcontext())[1])
    monkeypatch.setattr(inventory, "load_picker_context", lambda: object())
    monkeypatch.setattr(inventory, "build_models_payload", lambda _ctx: {
        "providers": [{"slug": "relay", "models": ["id-a", "id-b"]}],
    })
    monkeypatch.setattr(defaults, "select_silent_default",
                        lambda provider, rows: (
                            visited.append((provider, rows)) or
                            SimpleNamespace(selected=SimpleNamespace(ref=SimpleNamespace(model="id-b")))
                        ))
    assert dashboard.get_recommended_default_model("relay", profile="scoped") == {
        "provider": "relay", "model": "id-b", "free_tier": None,
    }
    assert visited == ["scoped", ("relay", ["id-a", "id-b"])]


def test_frontends_delegate_switch_and_never_choose_an_alternate_provider():
    web_picker = (ROOT / "web/src/components/ModelPickerDialog.tsx").read_text(
        encoding="utf-8"
    )
    desktop_api = (ROOT / "apps/desktop/src/api/models.ts").read_text(
        encoding="utf-8"
    )
    desktop_picker = (ROOT / "apps/desktop/src/lib/model-options.ts").read_text(
        encoding="utf-8"
    )
    bots = (ROOT / "apps/desktop/src/plugins/hermes-bots/model-picker.tsx").read_text(
        encoding="utf-8"
    )
    onboarding = (ROOT / "apps/desktop/src/store/onboarding.ts").read_text(
        encoding="utf-8"
    )
    assert '"model.options"' in web_picker and '"config.set"' in web_picker
    assert "/api/model/set" in desktop_api and "/api/model/options" in desktop_api
    assert "if (!request)" in desktop_picker  # owner-routed tiles must not REST fallback
    assert "requestForBot" in bots and "'model.options'" in bots
    assert "setMainModelAssignment" in onboarding  # no separate setup model writer


def test_inventory_discovery_is_explicitly_deferred_not_duplicated():
    # 5.8.7 will move the single inventory's discovery leaf; do not replace
    # it with a second dashboard-specific catalogue or private CLI switch.
    imports = _imports("hermes_cli/web_routers/models.py")
    assert "hermes_cli.inventory" in imports
    source = (ROOT / "hermes_cli/web_routers/models.py").read_text(encoding="utf-8")
    assert "list_authenticated_providers(" not in source
