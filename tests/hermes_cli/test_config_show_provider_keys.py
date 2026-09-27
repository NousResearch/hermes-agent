"""Configured model-provider credentials must be visible in `hermes config show`."""

from __future__ import annotations

from pathlib import Path

from hermes_cli import config as cfg
from hermes_cli.provider_catalog import provider_catalog_by_slug


SECRET = "sk-gemini-test-secret-abc123xyz"


def test_configured_provider_rows_use_catalog_label_and_collapse_aliases(monkeypatch):
    gemini = provider_catalog_by_slug()["gemini"]
    values = {
        "GOOGLE_API_KEY": "google-primary-secret",
        "GEMINI_API_KEY": "google-alias-secret",
    }
    monkeypatch.setattr(cfg, "get_env_value", lambda key: values.get(key, ""))

    rows = cfg._configured_provider_key_rows(set())

    google_rows = [(key, label) for key, label in rows if label == gemini.label]
    assert google_rows == [("GOOGLE_API_KEY", gemini.label)]


def test_config_show_appends_configured_provider_key_masked(monkeypatch, capsys):
    gemini = provider_catalog_by_slug()["gemini"]
    values = {"GEMINI_API_KEY": SECRET}
    monkeypatch.setattr(cfg, "get_env_value", lambda key: values.get(key, ""))
    monkeypatch.setattr(cfg, "load_config", lambda: {})
    monkeypatch.setattr(cfg, "_show_managed_banner", lambda: None)
    monkeypatch.setattr(cfg, "get_config_path", lambda: Path("/tmp/config.yaml"))
    monkeypatch.setattr(cfg, "get_env_path", lambda: Path("/tmp/.env"))
    monkeypatch.setattr(cfg, "get_project_root", lambda: Path("/tmp/hermes"))
    monkeypatch.setattr(cfg, "_show_model_section", lambda _config: None)
    monkeypatch.setattr(cfg, "_show_display_section", lambda _config: None)
    monkeypatch.setattr(cfg, "_show_terminal_section", lambda _config: None)
    monkeypatch.setattr(cfg, "_show_compression_section", lambda _config: None)
    monkeypatch.setattr(cfg, "_show_aux_overrides", lambda _config: None)
    monkeypatch.setattr(cfg, "_show_skill_settings", lambda: None)

    import hermes_cli.auth as auth
    monkeypatch.setattr(auth, "get_anthropic_key", lambda: "")

    cfg.show_config()
    out = capsys.readouterr().out

    assert gemini.label in out
    assert SECRET not in out
    assert cfg.redact_key(SECRET) in out


def test_fixed_rows_are_not_duplicated_by_provider_catalog(monkeypatch):
    values = {"OPENROUTER_API_KEY": "or-secret"}
    monkeypatch.setattr(cfg, "get_env_value", lambda key: values.get(key, ""))

    rows = cfg._configured_provider_key_rows(
        {env_key for env_key, _label in cfg._SHOW_CONFIG_API_KEYS}
    )

    assert all(env_key != "OPENROUTER_API_KEY" for env_key, _label in rows)
