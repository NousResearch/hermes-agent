"""Regression tests: switching from a custom provider to a built-in one must
clear the stale ``model.api_key`` (and ``model.api_mode``) from config.yaml.

Reported in Discord: after going from a named custom provider (which writes
``model.api_key: ${VAR}`` or a literal key) to Nous Portal / OpenRouter /
any API-key provider, the old ``api_key`` entry was left in config.yaml.
This caused the new provider's credential resolution to be overridden by
the leftover secret.

Affected code paths before the fix:
    - ``_model_flow_nous``: called ``_update_config_for_provider`` (which pops
        ``api_key`` on disk) but then reconstructed ``model_cfg`` from the stale
        in-memory ``config`` dict and called ``save_config(config)``, reinstating
        the key.
    - ``_model_flow_openrouter``: updated provider/base_url/api_mode directly
        via ``load_config()`` but never popped ``api_key``.
    - ``_model_flow_api_key_provider``: same omission as OpenRouter.

All ``patch()`` targets use the SOURCE module where the symbol is defined
(e.g. ``hermes_cli.auth._save_model_choice``), not the model_setup_flows
module.  The flows import these names lazily inside the function body via
``from hermes_cli.auth import ...``, so patching the source module is the
correct interception point.
"""

import copy
from unittest.mock import patch

import pytest
import yaml


@pytest.fixture
def home_with_custom_provider(tmp_path, monkeypatch):
    """Isolated HERMES_HOME pre-configured with a custom provider active."""
    home = tmp_path / "hermes"
    home.mkdir()
    config = {
        "model": {
            "default": "qwen3.6-35b-fast",
            "provider": "custom",
            "base_url": "https://api.custom.example/v1",
            "api_key": "${CUSTOM_API_KEY}",
            "api_mode": "chat_completions",
        },
        "custom_providers": [
            {
                "name": "My Custom Provider",
                "base_url": "https://api.custom.example/v1",
                "api_key": "${CUSTOM_API_KEY}",
                "model": "qwen3.6-35b-fast",
            }
        ],
    }
    config_path = home / "config.yaml"
    config_path.write_text(yaml.dump(config))
    env_file = home / ".env"
    env_file.write_text("CUSTOM_API_KEY=sk-live-custom-secret\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CUSTOM_API_KEY", "sk-live-custom-secret")
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    return home


class TestNousFlowClearsApiKey:
    """_model_flow_nous must remove stale api_key after switching from custom."""

    def test_api_key_and_api_mode_removed_when_switching_to_nous(
        self, home_with_custom_provider
    ):
        """Switching from a custom provider to Nous Portal must not leave
        api_key or api_mode in the model section saved to config.yaml.

        Before the fix: _update_config_for_provider() popped api_key from
        the on-disk file, but save_config(config) was then called with the
        stale in-memory ``config`` dict (which still had api_key), so the
        key was reinstated in config.yaml.
        """
        from hermes_cli.model_setup_flows import _model_flow_nous

        nous_creds = {
            "access_token": "nous-token",
            "api_key": "nous-api-key",
            "base_url": "https://inference.nous.research/v1",
        }

        # Stale in-memory config that the caller passes in — carries the old
        # api_key/api_mode from the previous custom provider session.
        stale_config = {
            "model": {
                "default": "qwen3.6-35b-fast",
                "provider": "custom",
                "base_url": "https://api.custom.example/v1",
                "api_key": "${CUSTOM_API_KEY}",
                "api_mode": "chat_completions",
            }
        }

        saved_configs: list = []

        def capture_save_config(cfg):
            import copy

            saved_configs.append(copy.deepcopy(cfg))

        with (
            patch("hermes_cli.auth.get_provider_auth_state",
                  return_value={"access_token": "nous-token"}),
            patch("hermes_cli.auth.resolve_nous_runtime_credentials",
                  return_value=nous_creds),
            patch("hermes_cli.models.get_curated_nous_model_ids",
                  return_value=["hermes-3-70b", "hermes-3-405b"]),
            patch("hermes_cli.models.get_pricing_for_provider",
                  return_value={}),
            patch("hermes_cli.models.check_nous_free_tier",
                  return_value=False),
            patch("hermes_cli.models.union_with_portal_paid_recommendations",
                  side_effect=lambda models, pricing, url: (models, pricing)),
            patch("hermes_cli.auth._prompt_model_selection",
                  return_value="hermes-3-70b"),
            patch("hermes_cli.auth._save_model_choice"),
            patch("hermes_cli.auth._update_config_for_provider"),
            patch("hermes_cli.config.save_config", side_effect=capture_save_config),
            patch("hermes_cli.config.get_env_value", return_value=None),
            patch("hermes_cli.nous_subscription.prompt_enable_tool_gateway"),
            patch("builtins.print"),
        ):
            _model_flow_nous(stale_config, current_model="qwen3.6-35b-fast")

        assert saved_configs, "save_config should have been called"
        model_section = saved_configs[-1].get("model", {})
        assert "api_key" not in model_section, (
            f"api_key must be cleared when switching to Nous Portal, "
            f"got model section: {model_section}"
        )
        assert "api_mode" not in model_section, (
            f"api_mode must be cleared when switching to Nous Portal, "
            f"got model section: {model_section}"
        )


class TestOpenRouterFlowClearsApiKey:
    """_model_flow_openrouter must remove stale api_key from model config."""

    def test_api_key_removed_after_switch_to_openrouter(
        self, home_with_custom_provider
    ):
        """Switching from a custom provider to OpenRouter must not leave
        api_key in the model section of config.yaml."""
        from hermes_cli.model_setup_flows import _model_flow_openrouter

        saved = yaml.safe_load(
            (home_with_custom_provider / "config.yaml").read_text()
        )
        assert "api_key" in saved["model"], "precondition: custom api_key present"

        with (
            # _prompt_api_key is imported from hermes_cli.main inside the function
            patch("hermes_cli.main._prompt_api_key",
                  return_value=("or-test-key", False)),
            # model_ids / get_pricing_for_provider are imported from hermes_cli.models
            patch("hermes_cli.models.model_ids",
                  return_value=["anthropic/claude-opus-4"]),
            patch("hermes_cli.models.get_pricing_for_provider",
                  return_value={}),
            # _prompt_model_selection / _save_model_choice / deactivate_provider
            # are imported from hermes_cli.auth
            patch("hermes_cli.auth._prompt_model_selection",
                  return_value="anthropic/claude-opus-4"),
            patch("hermes_cli.auth._save_model_choice"),
            patch("hermes_cli.auth.deactivate_provider"),
            patch("builtins.print"),
        ):
            _model_flow_openrouter({})

        saved_after = yaml.safe_load(
            (home_with_custom_provider / "config.yaml").read_text()
        )
        model_section = saved_after.get("model", {})
        assert "api_key" not in model_section, (
            f"Stale api_key must be removed when switching to OpenRouter, "
            f"got: {model_section}"
        )


class TestApiKeyProviderFlowClearsApiKey:
    """_model_flow_api_key_provider must remove stale api_key from model config."""

    def test_api_key_removed_after_switch_to_deepseek(
        self, home_with_custom_provider
    ):
        """Switching from a custom provider to DeepSeek (generic api-key
        provider) must not leave api_key in the model section of config.yaml."""
        from hermes_cli.model_setup_flows import _model_flow_api_key_provider

        saved = yaml.safe_load(
            (home_with_custom_provider / "config.yaml").read_text()
        )
        assert "api_key" in saved["model"], "precondition: custom api_key present"

        with (
            patch("hermes_cli.main._prompt_api_key",
                  return_value=("ds-test-key", False)),
            patch("hermes_cli.auth._prompt_model_selection",
                  return_value="deepseek-chat"),
            patch("hermes_cli.auth._save_model_choice"),
            patch("hermes_cli.auth.deactivate_provider"),
            # Prevent live /models probe from hitting the network
            patch("hermes_cli.models.fetch_api_models", return_value=[]),
            patch("builtins.print"),
        ):
            _model_flow_api_key_provider({}, "deepseek", current_model="")

        saved_after = yaml.safe_load(
            (home_with_custom_provider / "config.yaml").read_text()
        )
        model_section = saved_after.get("model", {})
        assert "api_key" not in model_section, (
            f"Stale api_key must be removed when switching to DeepSeek, "
            f"got: {model_section}"
        )

