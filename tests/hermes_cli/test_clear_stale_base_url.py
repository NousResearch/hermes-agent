"""Tests for stale-OPENAI_BASE_URL handling after a provider switch.

Originally (#5161) Hermes DELETED OPENAI_BASE_URL from ~/.hermes/.env when the
active provider was not 'custom'.  Per #4165 / cpf-zkw.5 config.yaml is now the
sole base_url source and Hermes no longer mutates the user's .env behind their
back: OPENAI_BASE_URL is left un-consulted but untouched, and a one-time
migration hint is surfaced instead.
"""

from __future__ import annotations

from hermes_cli.config import load_config, save_config, save_env_value, get_env_value


def _write_provider(provider: str, model: str = "test-model"):
    """Helper: write a provider + model to config.yaml."""
    cfg = load_config()
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    model_cfg["provider"] = provider
    model_cfg["default"] = model
    cfg["model"] = model_cfg
    save_config(cfg)


class TestWarnStaleOpenaiBaseUrl:
    """_warn_stale_openai_base_url() warns but PRESERVES OPENAI_BASE_URL."""

    def test_warns_but_preserves_when_provider_is_named(self, capsys):
        """OPENAI_BASE_URL is preserved (not deleted) and a hint is printed."""
        from hermes_cli.main import _warn_stale_openai_base_url

        _write_provider("openrouter")
        save_env_value("OPENAI_BASE_URL", "http://localhost:11434/v1")

        _warn_stale_openai_base_url()

        # No longer deleted — config.yaml is the sole base_url source, and we
        # never silently mutate the user's .env.
        result = get_env_value("OPENAI_BASE_URL")
        assert result == "http://localhost:11434/v1", \
            f"Expected OPENAI_BASE_URL to be PRESERVED, got: {result!r}"

        out = capsys.readouterr().out
        assert "OPENAI_BASE_URL" in out, "Expected a migration hint to be printed"

    def test_silent_when_provider_is_custom(self, capsys):
        """No warning when provider is 'custom' — it legitimately may use it."""
        from hermes_cli.main import _warn_stale_openai_base_url

        _write_provider("custom")
        save_env_value("OPENAI_BASE_URL", "http://localhost:11434/v1")

        _warn_stale_openai_base_url()

        result = get_env_value("OPENAI_BASE_URL")
        assert result == "http://localhost:11434/v1"
        assert "OPENAI_BASE_URL" not in capsys.readouterr().out

    def test_noop_when_no_openai_base_url(self, monkeypatch, capsys):
        """No error and no output when OPENAI_BASE_URL is not set."""
        from hermes_cli.main import _warn_stale_openai_base_url

        _write_provider("openrouter")
        save_env_value("OPENAI_BASE_URL", "")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        _warn_stale_openai_base_url()
        assert "OPENAI_BASE_URL" not in capsys.readouterr().out

    def test_noop_when_provider_empty(self, capsys):
        """No warning / no mutation when provider is not set in config."""
        from hermes_cli.main import _warn_stale_openai_base_url

        cfg = load_config()
        cfg.pop("model", None)
        save_config(cfg)
        save_env_value("OPENAI_BASE_URL", "http://localhost:11434/v1")

        _warn_stale_openai_base_url()

        result = get_env_value("OPENAI_BASE_URL")
        assert result == "http://localhost:11434/v1", \
            "Should not mutate when provider is not configured"
