"""Regression tests for removing saved custom providers."""

from hermes_cli.config import load_config, save_config


def test_remove_custom_provider_from_providers_dict(tmp_path, monkeypatch, capsys):
    """Removing a saved custom provider must work for migrated `providers:` config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    cfg = load_config()
    cfg["providers"] = {
        "local": {"name": "Local", "api": "http://localhost:8080/v1", "api_key": "x"},
        "other": {"name": "Other", "api": "http://example.com/v1"},
    }
    save_config(cfg)

    # Choose first entry, then keep default behavior for the rest
    monkeypatch.setattr("hermes_cli.main._prompt_provider_choice", lambda choices: 0)

    # Force fallback input path (no /dev/tty in CI)
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")

    from hermes_cli.main import _remove_custom_provider

    _remove_custom_provider(load_config())

    out = capsys.readouterr().out
    assert "Removed" in out

    reloaded = load_config()
    assert "local" not in (reloaded.get("providers") or {})
    assert "other" in (reloaded.get("providers") or {})
