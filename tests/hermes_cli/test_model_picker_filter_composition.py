"""Allowlist filtering composes with upstream picker validation and discovery."""


def test_inventory_keeps_probe_budget_and_both_model_filters(monkeypatch):
    from hermes_cli import inventory, model_switch

    captured = {}

    def discover(**kwargs):
        captured.update(kwargs)
        return [{
            "slug": "openrouter", "models": ["approved", "bad model", "other"],
            "total_models": 3,
        }]

    monkeypatch.setattr(model_switch, "list_authenticated_providers", discover)
    monkeypatch.setattr(inventory, "_local_runtime_row", lambda _ctx: None)
    monkeypatch.setattr(inventory, "_moa_provider_row", lambda _provider: None)
    ctx = inventory.ConfigContext(
        current_provider="openrouter", current_model="approved", current_base_url="",
        user_providers={}, custom_providers=[], allowed_models=[
            {"provider": "openrouter", "model": "approved"},
            {"provider": "openrouter", "model": "bad model"},
        ],
    )

    payload = inventory.build_models_payload(ctx, for_picker=True, fast_custom_probe=False)

    assert captured["fast_custom_probe"] is False
    assert payload["providers"] == [{
        "slug": "openrouter", "models": ["approved"], "total_models": 1,
    }]


def test_picker_filters_refreshed_catalog_for_validity_and_allowlist(monkeypatch):
    from hermes_cli import model_switch, model_switch_providers, models

    monkeypatch.setattr(model_switch, "list_authenticated_providers", lambda **_kwargs: [{
        "slug": "openrouter", "models": ["stale"], "total_models": 1,
    }])
    monkeypatch.setattr(models, "fetch_openrouter_models", lambda **_kwargs: [
        ("approved", "Approved"), ("bad model", "Invalid"), ("other", "Other"),
    ])

    rows = model_switch_providers.list_picker_providers(allowed_models=[
        {"provider": "openrouter", "model": "approved"},
        {"provider": "openrouter", "model": "bad model"},
    ])

    assert rows == [{"slug": "openrouter", "models": ["approved"], "total_models": 1}]
