"""Configured models extend built-in picker rows."""

from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.providers import HermesOverlay


def _provider_row(configured_models, *, max_models=None):
    with (
        patch(
            "agent.models_dev.fetch_models_dev",
            return_value={"deepseek": {"env": ["DEEPSEEK_API_KEY"], "name": "DeepSeek"}},
        ),
        patch(
            "agent.models_dev.PROVIDER_TO_MODELS_DEV",
            {"deepseek": "deepseek"},
        ),
        patch(
            "hermes_cli.models.cached_provider_model_ids",
            return_value=["live-a", "shared"],
        ),
        patch("hermes_cli.providers.HERMES_OVERLAYS", {}),
        patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}),
    ):
        rows = list_authenticated_providers(
            current_provider="deepseek",
            user_providers={"deepseek": {"models": configured_models}},
            max_models=max_models,
        )
    return next(row for row in rows if row["slug"] == "deepseek")


def test_configured_models_precede_and_deduplicate_discovered_models():
    row = _provider_row({"configured-x": {}, "shared": {}})

    assert row["models"] == ["configured-x", "shared", "live-a"]
    assert row["total_models"] == 3


def test_configured_models_are_merged_before_picker_limit():
    row = _provider_row(["configured-x", "configured-y"], max_models=2)

    assert row["models"] == ["configured-x", "configured-y"]
    assert row["total_models"] == 4


def _overlay_row(configured_models, *, max_models=None):
    """Same setup as _provider_row, but for a HERMES_OVERLAYS provider
    (section 2 — e.g. nous, openai-codex, copilot, opencode-go) instead of
    a PROVIDER_TO_MODELS_DEV built-in (section 1)."""
    with (
        patch("agent.models_dev.fetch_models_dev", return_value={}),
        patch("agent.models_dev.PROVIDER_TO_MODELS_DEV", {}),
        patch(
            "hermes_cli.providers.HERMES_OVERLAYS",
            {
                "test-overlay": HermesOverlay(
                    auth_type="api_key",
                    extra_env_vars=("TEST_OVERLAY_API_KEY",),
                ),
            },
        ),
        patch(
            "hermes_cli.models.cached_provider_model_ids",
            return_value=["live-a", "shared"],
        ),
        patch.dict("os.environ", {"TEST_OVERLAY_API_KEY": "test-key"}),
    ):
        rows = list_authenticated_providers(
            current_provider="test-overlay",
            user_providers={"test-overlay": {"models": configured_models}},
            max_models=max_models,
        )
    return next(row for row in rows if row["slug"] == "test-overlay")


def test_configured_models_extend_hermes_overlay_provider_row():
    """Regression: providers.<slug>.models must extend a HERMES_OVERLAYS
    row (nous, openai-codex, copilot, opencode-go, ...) the same way it
    already extends a built-in (section 1) row — otherwise a model the user
    can already type via /model <name> never appears in the picker list."""
    row = _overlay_row({"configured-x": {}, "shared": {}})

    assert row["models"] == ["configured-x", "shared", "live-a"]
    assert row["total_models"] == 3


def test_configured_models_are_merged_before_picker_limit_for_overlay():
    row = _overlay_row(["configured-x", "configured-y"], max_models=2)

    assert row["models"] == ["configured-x", "configured-y"]
    assert row["total_models"] == 4
