"""Tests for hermes_cli.auth._update_config_for_provider clearing stale fields.

When the user switches from a custom provider (e.g. MiniMax with
``api_mode: anthropic_messages``, ``api_key: mxp-...``) to a built-in
provider (e.g. OpenRouter), the stale ``api_key`` and ``api_mode`` would
otherwise override the new provider's credentials and transport choice.

Built-in providers that legitimately need a specific ``api_mode`` (copilot,
xai) compute it at request-resolution time in
``_copilot_runtime_api_mode`` / ``_detect_api_mode_for_url``, so removing
the persisted value here is safe.
"""

from __future__ import annotations

import yaml

from hermes_cli.auth import _update_config_for_provider
from hermes_cli.config import get_config_path


def _read_model_cfg() -> dict:
    path = get_config_path()
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    model = data.get("model", {})
    return model if isinstance(model, dict) else {}


def _seed_custom_provider_config(api_mode: str = "anthropic_messages") -> None:
    """Write a config.yaml mimicking a user on a MiniMax-style custom provider."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "provider": "custom",
                    "base_url": "https://api.minimax.io/anthropic",
                    "api_key": "mxp-stale-key",
                    "api_mode": api_mode,
                    "default": "claude-sonnet-4-6",
                }
            },
            sort_keys=False,
        )
    )


class TestUpdateConfigForProviderClearsStaleCustomFields:
    def test_switching_to_openrouter_clears_api_key_and_api_mode(self):
        _seed_custom_provider_config()

        _update_config_for_provider(
            "openrouter",
            "https://openrouter.ai/api/v1",
            default_model="anthropic/claude-sonnet-4.6",
        )

        model_cfg = _read_model_cfg()
        assert model_cfg.get("provider") == "openrouter"
        assert model_cfg.get("base_url") == "https://openrouter.ai/api/v1"
        assert "api_key" not in model_cfg, (
            "Stale custom api_key would leak into OpenRouter requests — must be cleared"
        )
        assert "api_mode" not in model_cfg, (
            "Stale api_mode=anthropic_messages from MiniMax would mis-route "
            "OpenRouter requests to the Anthropic SDK — must be cleared"
        )

    def test_switching_to_nous_clears_stale_api_mode(self):
        _seed_custom_provider_config()
        _update_config_for_provider("nous", "https://inference-api.nousresearch.com/v1")
        model_cfg = _read_model_cfg()
        assert model_cfg.get("provider") == "nous"
        assert "api_mode" not in model_cfg
        assert "api_key" not in model_cfg

    def test_switching_clears_codex_responses_api_mode(self):
        """Also covers codex_responses, not just anthropic_messages."""
        _seed_custom_provider_config(api_mode="codex_responses")
        _update_config_for_provider("openrouter", "https://openrouter.ai/api/v1")
        assert "api_mode" not in _read_model_cfg()
