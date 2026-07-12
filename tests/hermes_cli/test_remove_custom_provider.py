"""Regression tests for removing saved custom providers."""


def test_remove_custom_provider_from_providers_dict(tmp_path, monkeypatch, capsys):
    """Removing a saved custom provider must work for migrated `providers:` config."""
    from hermes_cli.config import load_config, save_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    cfg = load_config()
    cfg["providers"] = {
        "local": {"name": "Local", "api": "http://localhost:8080/v1", "api_key": "x"},
        "other": {"name": "Other", "api": "http://example.com/v1"},
    }
    save_config(cfg)

    # Force the fallback numbered-input path by making curses_radiolist import fail
    import builtins

    _real_import = builtins.__import__

    def _block_import(name, *args, **kwargs):
        if name == "hermes_cli.curses_ui":
            raise ImportError("blocked for test")
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_import)
    # Also need to ensure the cached module doesn't exist
    import sys

    sys.modules.pop("hermes_cli.curses_ui", None)

    monkeypatch.setattr("builtins.input", lambda prompt="": "1")

    from hermes_cli.main import _remove_custom_provider

    _remove_custom_provider(load_config())

    out = capsys.readouterr().out
    assert "Removed" in out

    reloaded = load_config()
    assert "local" not in (reloaded.get("providers") or {})
    assert "other" in (reloaded.get("providers") or {})


def test_remove_custom_provider_from_legacy_list(tmp_path, monkeypatch, capsys):
    """Removing a saved custom provider from legacy custom_providers list."""
    from hermes_cli.config import load_config, save_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    cfg = load_config()
    cfg["custom_providers"] = [
        {"name": "Test1", "base_url": "http://localhost:8080/v1", "api_key": "x"},
        {"name": "Test2", "base_url": "http://example.com/v1"},
    ]
    save_config(cfg)

    import builtins

    _real_import = builtins.__import__

    def _block_import(name, *args, **kwargs):
        if name == "hermes_cli.curses_ui":
            raise ImportError("blocked for test")
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_import)
    import sys

    sys.modules.pop("hermes_cli.curses_ui", None)

    monkeypatch.setattr("builtins.input", lambda prompt="": "1")

    from hermes_cli.main import _remove_custom_provider

    _remove_custom_provider(load_config())

    out = capsys.readouterr().out
    assert "Removed" in out

    reloaded = load_config()
    providers = reloaded.get("custom_providers") or []
    assert len(providers) == 1
    assert providers[0]["name"] == "Test2"
