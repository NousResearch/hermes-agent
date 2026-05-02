"""Tests for the Gemini free-tier block in the setup wizard."""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty config."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: some-old-model\n")
    (home / ".env").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Clear any ambient env that could alter provider resolution
    for var in (
        "HERMES_MODEL",
        "LLM_MODEL",
        "HERMES_INFERENCE_PROVIDER",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "GEMINI_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    return home


class TestGeminiSetupFreeTierBlock:
    """_model_flow_api_key_provider should refuse to wire up a free-tier Gemini key."""

    def test_free_tier_key_is_blocked(self, config_home, monkeypatch, capsys):
        """Free-tier probe result -> provider is NOT saved, message is printed."""
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-free-tier-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        # Mock the probe to claim this is a free-tier key
        with patch(
            "agent.gemini_native_adapter.probe_gemini_tier",
            return_value="free",
        ), patch(
            "hermes_cli.auth._prompt_model_selection",
            return_value="gemini-2.5-flash",
        ), patch(
            "hermes_cli.auth.deactivate_provider",
        ), patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "gemini", "old-model")

        output = capsys.readouterr().out
        assert "free tier" in output.lower()
        assert "aistudio.google.com/apikey" in output
        assert "Not saving Gemini as the default provider" in output

        # Config must NOT show gemini as the provider
        import yaml
        cfg = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = cfg.get("model")
        if isinstance(model, dict):
            assert model.get("provider") != "gemini", (
                "Free-tier key should not have saved gemini as provider"
            )
        # If still a string, also fine — nothing was saved

    def test_paid_tier_key_proceeds(self, config_home, monkeypatch, capsys):
        """Paid-tier probe result -> provider IS saved normally."""
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-paid-tier-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        with patch(
            "agent.gemini_native_adapter.probe_gemini_tier",
            return_value="paid",
        ), patch(
            "hermes_cli.auth._prompt_model_selection",
            return_value="gemini-2.5-flash",
        ), patch(
            "hermes_cli.auth.deactivate_provider",
        ), patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "gemini", "old-model")

        output = capsys.readouterr().out
        assert "paid" in output.lower()
        assert "Not saving Gemini" not in output

        import yaml
        cfg = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = cfg.get("model")
        assert isinstance(model, dict), f"model should be dict, got {type(model)}"
        assert model.get("provider") == "gemini"
        assert model.get("default") == "gemini-2.5-flash"

    def test_unknown_tier_proceeds_with_warning(self, config_home, monkeypatch, capsys):
        """Probe returning 'unknown' (network/auth error) -> proceed without blocking."""
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        with patch(
            "agent.gemini_native_adapter.probe_gemini_tier",
            return_value="unknown",
        ), patch(
            "hermes_cli.auth._prompt_model_selection",
            return_value="gemini-2.5-flash",
        ), patch(
            "hermes_cli.auth.deactivate_provider",
        ), patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "gemini", "old-model")

        output = capsys.readouterr().out
        assert "could not verify" in output.lower()
        assert "Not saving Gemini" not in output

        import yaml
        cfg = yaml.safe_load((config_home / "config.yaml").read_text()) or {}
        model = cfg.get("model")
        assert isinstance(model, dict)
        assert model.get("provider") == "gemini"

    def test_non_gemini_provider_skips_probe(self, config_home, monkeypatch):
        """Probe must only run for provider_id == 'gemini', not for other providers."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        with patch(
            "agent.gemini_native_adapter.probe_gemini_tier",
        ) as mock_probe, patch(
            "hermes_cli.auth._prompt_model_selection",
            return_value="deepseek-chat",
        ), patch(
            "hermes_cli.auth.deactivate_provider",
        ), patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "deepseek", "old-model")

        mock_probe.assert_not_called()
