"""model_aliases.api_key must ride along DirectAlias (#83612)."""

from __future__ import annotations

from hermes_cli.model_switch import DirectAlias, _load_direct_aliases


def test_load_direct_aliases_reads_api_key(monkeypatch):
    cfg = {
        "model_aliases": {
            "theta": {
                "model": "llama-3",
                "provider": "custom",
                "base_url": "https://ondemand.example.com/v1",
                "api_key": "sk-alias-only",
            }
        }
    }
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: cfg,
    )
    aliases = _load_direct_aliases()
    assert "theta" in aliases
    da = aliases["theta"]
    assert da.model == "llama-3"
    assert da.base_url == "https://ondemand.example.com/v1"
    assert da.api_key == "sk-alias-only"


def test_load_direct_aliases_expands_env_api_key(monkeypatch):
    monkeypatch.setenv("THETA_KEY", "from-env")
    cfg = {
        "model_aliases": {
            "theta": {
                "model": "llama-3",
                "provider": "custom",
                "base_url": "https://ondemand.example.com/v1",
                "api_key": "${THETA_KEY}",
            }
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    # Force plain getenv path (no multiplex scope).
    aliases = _load_direct_aliases()
    assert aliases["theta"].api_key == "from-env"


def test_direct_alias_default_api_key_empty():
    da = DirectAlias("m", "custom", "https://x/v1")
    assert da.api_key == ""
