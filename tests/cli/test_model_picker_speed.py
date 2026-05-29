from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

from cli import HermesCLI
from hermes_cli.inventory import ConfigContext


def _ctx() -> ConfigContext:
    return ConfigContext(
        current_provider="slow-provider",
        current_model="configured-model",
        current_base_url="https://slow.example/v1",
        user_providers={
            "slow-provider": {
                "name": "Slow Provider",
                "base_url": "https://slow.example/v1",
                "api_key": "test-key",
                "models": {"configured-model": {}},
            }
        },
        custom_providers=[],
    )


def _make_cli(opened: dict[str, Any]) -> HermesCLI:
    cli = HermesCLI.__new__(HermesCLI)
    cli.provider = "slow-provider"
    cli.model = "configured-model"
    cli.base_url = "https://slow.example/v1"

    def _open_model_picker(
        providers,
        current_model,
        current_provider,
        user_provs=None,
        custom_provs=None,
    ):
        opened["providers"] = providers
        opened["current_model"] = current_model
        opened["current_provider"] = current_provider
        opened["user_provs"] = user_provs
        opened["custom_provs"] = custom_provs

    cli._open_model_picker = _open_model_picker
    return cli


def _empty_pool():
    class EmptyPool:
        def has_credentials(self) -> bool:
            return False

    return EmptyPool()


def test_no_arg_model_picker_uses_configured_models_without_live_probe():
    """Opening `/model` should not block on live custom-provider /models probes."""
    opened: dict[str, Any] = {}
    cli = _make_cli(opened)

    with patch.dict(os.environ, {}, clear=True), \
         patch("hermes_cli.inventory.load_picker_context", return_value=_ctx()), \
         patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("agent.credential_pool.load_pool") as load_pool, \
         patch("hermes_cli.models.get_curated_nous_model_ids", return_value=[]), \
         patch("hermes_cli.models.fetch_api_models", return_value=["live-only-model"]) as live_fetch:
        cli._handle_model_switch("/model")

    live_fetch.assert_not_called()
    load_pool.assert_not_called()
    providers = opened["providers"]
    assert providers[0]["slug"] == "slow-provider"
    assert providers[0]["models"] == ["configured-model"]


def test_model_refresh_keeps_live_probe_path():
    """`/model --refresh` remains the explicit slow/live catalog path."""
    opened: dict[str, Any] = {}
    cli = _make_cli(opened)

    with patch.dict(os.environ, {}, clear=True), \
         patch("hermes_cli.inventory.load_picker_context", return_value=_ctx()), \
         patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("agent.credential_pool.load_pool", return_value=_empty_pool()), \
         patch("hermes_cli.models.clear_provider_models_cache") as clear_cache, \
         patch("hermes_cli.models.get_curated_nous_model_ids", return_value=[]), \
         patch("hermes_cli.models.fetch_api_models", return_value=["live-only-model"]) as live_fetch:
        cli._handle_model_switch("/model --refresh")

    clear_cache.assert_called_once_with()
    live_fetch.assert_called_once_with("test-key", "https://slow.example/v1")
    providers = opened["providers"]
    assert providers[0]["models"] == ["live-only-model"]
