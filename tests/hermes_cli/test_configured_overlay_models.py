"""Configured models extend overlay picker rows."""

from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


def test_azure_foundry_overlay_merges_configured_models_before_picker_limit():
    from hermes_cli.providers import HERMES_OVERLAYS

    azure_foundry = HERMES_OVERLAYS["azure-foundry"]
    with (
        patch("agent.models_dev.fetch_models_dev", return_value={}),
        patch("agent.models_dev.PROVIDER_TO_MODELS_DEV", {}),
        patch(
            "hermes_cli.models.cached_provider_model_ids",
            return_value=["live-a", "shared"],
        ),
        patch(
            "hermes_cli.providers.HERMES_OVERLAYS",
            {"azure-foundry": azure_foundry},
        ),
        patch.dict("os.environ", {"AZURE_FOUNDRY_API_KEY": "test-key"}),
    ):
        rows = list_authenticated_providers(
            current_provider="azure-foundry",
            user_providers={
                "azure-foundry": {"models": ["configured-x", "shared"]},
            },
            max_models=2,
        )

    row = next(row for row in rows if row["slug"] == "azure-foundry")

    assert row["source"] == "hermes"
    assert row["models"] == ["configured-x", "shared"]
    assert row["total_models"] == 3
