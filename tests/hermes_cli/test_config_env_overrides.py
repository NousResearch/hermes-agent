"""Tests for HERMES_* env var overrides in config (Docker Compose support)."""

import os
import pytest
from hermes_cli.config import load_config


class TestEnvOverridesInLoadConfig:
    """Test that HERMES_MODEL, HERMES_INFERENCE_PROVIDER, and HERMES_BASE_URL
    override config.yaml values when set."""

    def test_hermes_model_alone_no_config_file(self, tmp_path, monkeypatch):
        """HERMES_MODEL sets model when no config.yaml exists."""
        config_file = tmp_path / "config.yaml"
        monkeypatch.setenv("HERMES_MODEL", "openrouter/anthropic/claude-opus-4")
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)

        config = load_config()

        assert config["model"]["default"] == "openrouter/anthropic/claude-opus-4"

    def test_hermes_model_overrides_config_file(self, tmp_path, monkeypatch):
        """HERMES_MODEL overrides model.default from config.yaml."""
        config_yaml = "model:\n  default: openai/gpt-4\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("HERMES_MODEL", "anthropic/claude-sonnet-4")
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)

        config = load_config()

        assert config["model"]["default"] == "anthropic/claude-sonnet-4"

    def test_hermes_provider_overrides_config_file(self, tmp_path, monkeypatch):
        """HERMES_INFERENCE_PROVIDER overrides model.provider from config.yaml."""
        config_yaml = "model:\n  provider: openai\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "anthropic")
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)

        config = load_config()

        assert config["model"]["provider"] == "anthropic"

    def test_hermes_base_url_overrides_config_file(self, tmp_path, monkeypatch):
        """HERMES_BASE_URL overrides model.base_url from config.yaml."""
        config_yaml = "model:\n  base_url: https://openrouter.ai/api/v1\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("HERMES_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)

        config = load_config()

        assert config["model"]["base_url"] == "http://localhost:8000/v1"

    def test_all_three_env_overrides_together(self, tmp_path, monkeypatch):
        """HERMES_MODEL, HERMES_INFERENCE_PROVIDER, HERMES_BASE_URL all work together."""
        config_yaml = "model:\n  default: openai/gpt-4\n  provider: openai\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("HERMES_MODEL", "openrouter/meta-llama/llama-3.1-405b")
        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
        monkeypatch.setenv("HERMES_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)

        config = load_config()

        assert config["model"]["default"] == "openrouter/meta-llama/llama-3.1-405b"
        assert config["model"]["provider"] == "openrouter"
        assert config["model"]["base_url"] == "https://openrouter.ai/api/v1"

    def test_empty_env_vars_do_not_override(self, tmp_path, monkeypatch):
        """Empty env vars don't override config.yaml values."""
        config_yaml = (
            "model:\n"
            "  default: openai/gpt-4\n"
            "  provider: openai\n"
            "  base_url: https://api.openai.com/v1\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("HERMES_MODEL", "")
        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "")
        monkeypatch.setenv("HERMES_BASE_URL", "")
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)

        config = load_config()

        assert config["model"]["default"] == "openai/gpt-4"
        assert config["model"]["provider"] == "openai"
        assert config["model"]["base_url"] == "https://api.openai.com/v1"

    def test_partial_env_overrides_preserve_other_fields(self, tmp_path, monkeypatch):
        """Setting one env var doesn't clear other config fields."""
        config_yaml = (
            "model:\n"
            "  default: openai/gpt-4\n"
            "  provider: openai\n"
            "  context_length: 128000\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("HERMES_MODEL", "anthropic/claude-opus-4.6")
        monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
        monkeypatch.delenv("HERMES_BASE_URL", raising=False)
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)

        config = load_config()

        assert config["model"]["default"] == "anthropic/claude-opus-4.6"
        assert config["model"]["provider"] == "openai"  # unchanged
        assert config["model"]["context_length"] == 128000  # unchanged

    def test_hermes_model_with_env_var_templates_in_config(self, tmp_path, monkeypatch):
        """HERMES_MODEL overrides even when config.yaml uses ${VAR} templates."""
        config_yaml = "model:\n  default: ${CONFIG_MODEL}\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("CONFIG_MODEL", "openai/gpt-4")
        monkeypatch.setenv("HERMES_MODEL", "anthropic/claude-opus-4.6")
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)

        config = load_config()

        # HERMES_MODEL should override the expanded ${CONFIG_MODEL}
        assert config["model"]["default"] == "anthropic/claude-opus-4.6"

    def test_plain_string_model_converted_to_dict(self, tmp_path, monkeypatch):
        """If config has a plain-string model value, env overrides work on dict form."""
        config_yaml = "model: openai/gpt-4\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)

        config = load_config()

        assert config["model"]["default"] == "openai/gpt-4"  # preserved from plain string
        assert config["model"]["provider"] == "openrouter"  # added by env override

    def test_docker_compose_style_setup(self, tmp_path, monkeypatch):
        """Simulate Docker Compose: no config.yaml, env vars set for model, provider, base_url."""
        config_file = tmp_path / "config.yaml"
        # No config file — Docker entrypoint will create the default one

        monkeypatch.setenv("HERMES_MODEL", "openrouter/openai/gpt-4o")
        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")
        monkeypatch.setenv("HERMES_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)

        config = load_config()

        # Model should be fully configured from env vars
        assert config["model"]["default"] == "openrouter/openai/gpt-4o"
        assert config["model"]["provider"] == "openrouter"
        assert config["model"]["base_url"] == "https://openrouter.ai/api/v1"
        # API key comes from .env or directly from environ, already handled
