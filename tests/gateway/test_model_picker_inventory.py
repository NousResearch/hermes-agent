"""Tests for the gateway /model inventory projection."""

from gateway import model_picker_inventory as inventory


def test_picker_projection_passes_gateway_facts_to_application_inventory(monkeypatch):
    seen = {}

    def build(ctx, **kwargs):
        seen["ctx"] = ctx
        seen["kwargs"] = kwargs
        return {
            "providers": [
                {"slug": "openrouter", "models": ["vendor/model"]},
                {"slug": "moa", "models": ["aggregate"]},
            ]
        }

    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", build)
    rows = inventory.model_provider_rows(
        {
            "current_provider": "openrouter",
            "current_model": "vendor/model",
            "current_base_url": "https://openrouter.ai/api/v1",
            "user_providers": {"custom-one": {}},
            "custom_providers": [{"name": "Local"}],
            "excluded_providers": ["hidden"],
        },
        max_models=5,
        interactive=False,
    )

    assert [row["slug"] for row in rows] == ["openrouter"]
    assert seen["ctx"].current_provider == "openrouter"
    assert seen["ctx"].current_model == "vendor/model"
    assert seen["kwargs"]["for_picker"] is False
    assert seen["kwargs"]["non_blocking_catalogs"] is True


def test_picker_refresh_requests_live_catalogue_acquisition(monkeypatch):
    seen = {}

    def build(_ctx, **kwargs):
        seen.update(kwargs)
        return {"providers": [{"slug": "openrouter", "models": ["vendor/model"]}]}

    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", build)
    inventory.model_provider_rows(
        {"current_provider": "openrouter"},
        max_models=50,
        interactive=True,
        refresh=True,
    )

    assert seen["refresh"] is True
    assert seen["for_picker"] is True
    assert seen["probe_custom_providers"] is True
    assert seen["non_blocking_catalogs"] is False
