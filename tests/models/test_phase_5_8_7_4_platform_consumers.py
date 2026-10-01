"""Phase 5.8.7.4: platform consumers use the canonical presentation seam."""
from __future__ import annotations

import ast
from pathlib import Path

import application_provider_groups as groups
from application_model_selection_guards import (
    SelectionContext,
    combined_selection_warning,
)
from providers.identity import get_provider_label

ROOT = Path(__file__).resolve().parents[2]


def _imports(relpath: str) -> set[tuple[str, str]]:
    tree = ast.parse((ROOT / relpath).read_text(encoding="utf-8"))
    return {
        (node.module, name.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for name in node.names
    }


def test_all_platform_provider_labels_have_one_semantic_owner():
    for platform in ("discord", "telegram", "slack", "matrix"):
        source = f"plugins/platforms/{platform}/adapter.py"
        imports = _imports(source)
        assert ("providers.identity", "get_provider_label") in imports
        assert ("hermes_cli.providers", "get_label") not in imports
    assert get_provider_label("provider_not_registered") == "provider_not_registered"


def test_discord_telegram_confirm_from_shared_application_policy():
    for platform in ("discord", "telegram"):
        imports = _imports(f"plugins/platforms/{platform}/adapter.py")
        assert (
            "application_model_selection_guards", "combined_selection_warning"
        ) in imports
        assert (
            "hermes_cli.model_selection_guards", "combined_selection_warning"
        ) not in imports


def test_selection_warning_keeps_large_context_and_same_model_gates():
    warning = combined_selection_warning(
        "next-model",
        provider="provider-x",
        selection_context=SelectionContext(
            context_tokens=200_000, current_model="previous-model"
        ),
        context_threshold=100_000,
        include_kinds={"context_cache"},
    )
    assert warning is not None
    assert warning.kind == "context_cache"
    assert "200,000" in warning.message
    assert combined_selection_warning(
        "next-model", provider="provider-x",
        selection_context=SelectionContext(
            context_tokens=200_000, current_model="next-model"
        ),
        context_threshold=100_000,
        include_kinds={"context_cache"},
    ) is None


def test_combined_platform_warnings_preserve_cost_then_data_order(monkeypatch):
    import application_model_selection_guards as app

    monkeypatch.setattr(
        app, "_cost_warning",
        lambda *_args: app.SelectionWarning("cost", "Cost", "model", "provider", "cost message"),
    )
    monkeypatch.setattr(
        app, "_data_warning",
        lambda *_args: app.SelectionWarning("data_policy", "Privacy", "model", "provider", "data message"),
    )
    warning = app.combined_selection_warning("model", provider="provider", context_threshold=0)
    assert warning is not None
    assert warning.kind == "multiple"
    assert warning.message == "cost message\n\ndata message"


def test_provider_grouping_is_single_application_owner_and_ordered():
    assert not (ROOT / "hermes_cli/provider_groups.py").exists()
    rows = groups.group_providers(["minimax-cn", "unregistered", "minimax"])
    assert rows == [
        {
            "kind": "group", "group_id": "minimax",
            "label": groups.PROVIDER_GROUPS["minimax"][0],
            "description": groups.PROVIDER_GROUPS["minimax"][1],
            "members": ["minimax", "minimax-cn"],
        },
        {"kind": "single", "slug": "unregistered"},
    ]
    source = _imports("plugins/platforms/telegram/adapter.py")
    assert ("application_provider_groups", "group_providers") in source
    assert ("application_provider_groups", "PROVIDER_GROUPS") in source
    assert not any(module == "hermes_cli.provider_groups" for module, _ in source)


def test_cli_picker_consumes_app_grouping_without_duplicating_rules():
    imports = _imports("hermes_cli/main_provider_setup.py")
    assert ("application_provider_groups", "group_providers") in imports
    assert ("application_provider_groups", "provider_group_for_slug") in imports


def test_feishu_already_uses_canonical_gateway_default():
    assert (
        "gateway.model_runtime_facts", "provider_default_model"
    ) in _imports("plugins/platforms/feishu/feishu_comment.py")
