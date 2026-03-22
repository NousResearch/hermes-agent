"""Tests for custom provider context length auto-detection (#2513)."""
import pytest
from unittest.mock import patch


class TestCustomProviderContextDetection:

    def test_autodetect_known_model(self):
        from agent.model_metadata import get_model_context_length
        result = get_model_context_length("anthropic/claude-opus-4-6")
        assert result > 0

    def test_autodetect_unknown_model_returns_default(self):
        from agent.model_metadata import get_model_context_length
        result = get_model_context_length("unknown/totally-fake-model-xyz")
        assert result > 0

    def test_save_custom_provider_with_context_length(self):
        import tempfile, os
        import yaml
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("{}")

            with patch("hermes_cli.config.get_config_path", return_value=config_path):
                from hermes_cli.main import _save_custom_provider
                _save_custom_provider(
                    base_url="http://localhost:11434/v1",
                    model="llama3.1",
                    context_length=131072,
                )
                saved = yaml.safe_load(config_path.read_text()) or {}
                providers = saved.get("custom_providers", [])
                assert len(providers) == 1
                assert providers[0]["models"]["llama3.1"]["context_length"] == 131072

    def test_save_custom_provider_without_context_length(self):
        import tempfile
        import yaml
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("{}")

            with patch("hermes_cli.config.get_config_path", return_value=config_path):
                from hermes_cli.main import _save_custom_provider
                _save_custom_provider(
                    base_url="http://localhost:11434/v1",
                    model="llama3.1",
                    context_length=None,
                )
                saved = yaml.safe_load(config_path.read_text()) or {}
                providers = saved.get("custom_providers", [])
                assert len(providers) == 1
                assert "models" not in providers[0]
