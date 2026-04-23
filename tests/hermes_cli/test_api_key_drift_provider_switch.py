"""Tests that switching providers via _model_flow_api_key_provider
clears stale api_key from the model config dict.

Regression test for #14134: when switching from one API-key provider
to another, the old provider's api_key was left in model.api_key,
causing credential drift — the new provider would try to use the
old provider's key and fail with 401.

The sister function in auth.py (set_provider_in_config) correctly
pops both api_key and api_mode on provider switch. This test ensures
_model_flow_api_key_provider does the same.
"""

import os
from unittest.mock import patch

import pytest


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with config that has a stale api_key."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Clear env vars that could interfere
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
    monkeypatch.delenv("STEPFUN_BASE_URL", raising=False)
    return home


def _write_config(home, model_dict):
    """Write a config.yaml with the given model dict."""
    import yaml
    config_yaml = home / "config.yaml"
    config_yaml.write_text(yaml.dump({"model": model_dict}))


def _read_model_config(home):
    """Read the model section from config.yaml."""
    import yaml
    config = yaml.safe_load((home / "config.yaml").read_text()) or {}
    return config.get("model", {})


class TestApiKeyDriftOnProviderSwitch:
    """Switching from one api-key provider to another must clear the
    stale api_key from the model config dict."""

    def test_api_key_cleared_on_provider_switch(self, config_home, monkeypatch):
        """Start with model.api_key = 'sk-old-key' from provider A,
        switch to provider B — api_key must be popped."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("zai")
        if not pconfig:
            pytest.skip("zai not in PROVIDER_REGISTRY")

        # Start with a config from a *previous* provider that had an api_key
        _write_config(config_home, {
            "default": "some-old-model",
            "provider": "ollama-cloud",
            "base_url": "https://api.ola.cloud/v1",
            "api_key": "sk-old-provider-key-12345",
        })

        monkeypatch.setenv("GLM_API_KEY", "test-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        with patch("hermes_cli.auth._prompt_model_selection", return_value="glm-5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "zai", "some-old-model")

        model = _read_model_config(config_home)
        assert isinstance(model, dict)
        assert model.get("provider") == "zai", (
            f"provider should be 'zai', got {model.get('provider')}"
        )
        assert "api_key" not in model, (
            f"api_key should be cleared on provider switch, but found: {model.get('api_key')}"
        )

    def test_api_mode_also_cleared_on_non_opencode_switch(self, config_home, monkeypatch):
        """A stale api_mode from a previous custom provider must also
        be cleared when switching to a non-opencode provider."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("zai")
        if not pconfig:
            pytest.skip("zai not in PROVIDER_REGISTRY")

        # Start with custom-provider config that had api_mode and api_key
        _write_config(config_home, {
            "default": "custom-model",
            "provider": "custom",
            "base_url": "https://custom.old/v1",
            "api_key": "sk-stale-custom-key",
            "api_mode": "anthropic_messages",
        })

        monkeypatch.setenv("GLM_API_KEY", "test-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        with patch("hermes_cli.auth._prompt_model_selection", return_value="glm-5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "zai", "custom-model")

        model = _read_model_config(config_home)
        assert isinstance(model, dict)
        assert "api_key" not in model, (
            f"api_key should be cleared, got: {model.get('api_key')}"
        )
        assert "api_mode" not in model, (
            f"api_mode should be cleared for non-opencode, got: {model.get('api_mode')}"
        )

    def test_switch_preserves_default_model(self, config_home, monkeypatch):
        """The model.default should be updated to the new selection even
        when there was a stale api_key."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("zai")
        if not pconfig:
            pytest.skip("zai not in PROVIDER_REGISTRY")

        _write_config(config_home, {
            "default": "old-model-from-previous-provider",
            "provider": "ollama-cloud",
            "api_key": "sk-orphaned-key",
        })

        monkeypatch.setenv("GLM_API_KEY", "test-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        with patch("hermes_cli.auth._prompt_model_selection", return_value="glm-5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "zai", "old-model-from-previous-provider")

        model = _read_model_config(config_home)
        assert model.get("default") == "glm-5", (
            f"model.default should be 'glm-5', got {model.get('default')}"
        )
        assert "api_key" not in model

    def test_no_api_key_no_error(self, config_home, monkeypatch):
        """If config has no stale api_key, switching should still work fine."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("zai")
        if not pconfig:
            pytest.skip("zai not in PROVIDER_REGISTRY")

        # Clean config, no api_key
        _write_config(config_home, {
            "default": "old-model",
            "provider": "ollama-cloud",
        })

        monkeypatch.setenv("GLM_API_KEY", "test-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        with patch("hermes_cli.auth._prompt_model_selection", return_value="glm-5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "zai", "old-model")

        model = _read_model_config(config_home)
        assert model.get("provider") == "zai"
        assert model.get("default") == "glm-5"
        assert "api_key" not in model