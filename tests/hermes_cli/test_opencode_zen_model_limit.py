"""Regression tests for OpenCode Zen model picker limits."""

import os
from unittest.mock import patch

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import list_authenticated_providers


def test_opencode_zen_lists_all_models_while_other_providers_remain_capped(monkeypatch):
    """OpenCode Zen is an aggregator product, so the picker must expose its full catalog."""
    zen_models = [f"zen-model-{i}" for i in range(57)]
    deepseek_models = [f"deepseek-model-{i}" for i in range(57)]

    monkeypatch.setattr(
        "agent.models_dev.PROVIDER_TO_MODELS_DEV",
        {
            "opencode-zen": "opencode",
            "deepseek": "deepseek",
        },
    )
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: {"opencode": {}, "deepseek": {}},
    )
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr(
        "hermes_cli.models.cached_provider_model_ids",
        lambda provider: {
            "opencode-zen": zen_models,
            "deepseek": deepseek_models,
        }.get(provider, []),
    )

    with patch.dict(
        os.environ,
        {
            "OPENCODE_ZEN_API_KEY": "test-zen-key",
            "DEEPSEEK_API_KEY": "test-deepseek-key",
        },
        clear=False,
    ):
        providers = list_authenticated_providers(max_models=50)

    opencode_zen = next(p for p in providers if p["slug"] == "opencode-zen")
    deepseek = next(p for p in providers if p["slug"] == "deepseek")

    assert opencode_zen["models"] == zen_models
    assert opencode_zen["total_models"] == len(zen_models)
    assert deepseek["models"] == deepseek_models[:50]
    assert deepseek["total_models"] == len(deepseek_models)
