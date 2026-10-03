"""Configured models extend built-in picker rows."""

from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


def _provider_row(configured_models, *, discover_models=None, max_models=None):
    configured = {"models": configured_models}
    if discover_models is not None:
        configured["discover_models"] = discover_models
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
            user_providers={"deepseek": configured},
            max_models=max_models,
        )
    return next(row for row in rows if row["slug"] == "deepseek")


def test_configured_models_precede_and_deduplicate_discovered_models():
    row = _provider_row({"configured-x": {}, "shared": {}})

    assert row["models"] == ["configured-x", "shared", "live-a"]
    assert row["total_models"] == 3


def test_discover_models_false_narrows_builtin_row_to_configured_models():
    """#107106: discover_models: false must pin the row to the declared list instead of
    merging it with the live catalog."""
    row = _provider_row({"configured-x": {}, "shared": {}}, discover_models=False)

    assert row["models"] == ["configured-x", "shared"]
    assert row["total_models"] == 2


