"""Tests that `hermes model` always shows the model selection menu for custom
providers, even when a model is already saved.

Regression test for the bug where _model_flow_named_custom() returned
immediately when provider_info had a saved ``model`` field, making it
impossible to switch models on multi-model endpoints.
"""

import os
from unittest.mock import patch, MagicMock, call

import pytest


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a minimal config."""
    home = tmp_path / "hermes"
    home.mkdir()
    config_yaml = home / "config.yaml"
    config_yaml.write_text("model: old-model\ncustom_providers: []\n")
    env_file = home / ".env"
    env_file.write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return home


class TestCustomProviderModelSwitch:
    """Ensure _model_flow_named_custom always probes and shows menu."""

    def test_saved_model_still_probes_endpoint(self, config_home):
        """When a model is already saved, the function must still call
        fetch_api_models to probe the endpoint — not skip with early return."""
        from hermes_cli.main import _model_flow_named_custom

        provider_info = {
            "name": "My vLLM",
            "base_url": "https://vllm.example.com/v1",
            "api_key": "sk-test",
            "model": "model-A",  # already saved
        }

        with patch("hermes_cli.main.fetch_api_models", return_value=["model-A", "model-B"]) as mock_fetch, \
             patch("builtins.input", return_value="2"), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        # fetch_api_models MUST be called even though model was saved
        mock_fetch.assert_called_once_with("sk-test", "https://vllm.example.com/v1", timeout=8.0)

    def test_can_switch_to_different_model(self, config_home):
        """User selects a different model than the saved one."""
        import yaml
        from hermes_cli.main import _model_flow_named_custom

        provider_info = {
            "name": "My vLLM",
            "base_url": "https://vllm.example.com/v1",
            "api_key": "sk-test",
            "model": "model-A",
        }

        with patch("hermes_cli.main.fetch_api_models", return_value=["model-A", "model-B"]), \
             patch("builtins.input", return_value="2"), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model["default"] == "model-B"

    def test_probe_failure_falls_back_to_saved(self, config_home):
        """When endpoint probe fails and user presses Enter, saved model is used."""
        import yaml
        from hermes_cli.main import _model_flow_named_custom

        provider_info = {
            "name": "My vLLM",
            "base_url": "https://vllm.example.com/v1",
            "api_key": "sk-test",
            "model": "model-A",
        }

        # fetch returns empty list (probe failed), user presses Enter (empty input)
        with patch("hermes_cli.main.fetch_api_models", return_value=[]), \
             patch("builtins.input", return_value=""), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model["default"] == "model-A"

    def test_no_saved_model_still_works(self, config_home):
        """First-time flow (no saved model) still works as before."""
        import yaml
        from hermes_cli.main import _model_flow_named_custom

        provider_info = {
            "name": "My vLLM",
            "base_url": "https://vllm.example.com/v1",
            "api_key": "sk-test",
            # no "model" key
        }

        with patch("hermes_cli.main.fetch_api_models", return_value=["model-X"]), \
             patch("builtins.input", return_value="1"), \
             patch("builtins.print"):
            _model_flow_named_custom({}, provider_info)

        config = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = config.get("model")
        assert isinstance(model, dict)
        assert model["default"] == "model-X"
