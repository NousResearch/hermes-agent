"""`hermes setup` first-install detection covers every provider family (#126498)."""

from hermes_cli import setup as setup_mod


def _no_env(monkeypatch):
    monkeypatch.setattr(setup_mod, "get_env_value", lambda key: None)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)


def test_anthropic_install_is_existing(monkeypatch):
    _no_env(monkeypatch)
    assert setup_mod._install_looks_configured({"model": {"provider": "anthropic"}}) is True


def test_custom_endpoint_install_is_existing(monkeypatch):
    _no_env(monkeypatch)
    config = {"model": {"provider": "custom", "base_url": "http://127.0.0.1:1234"}}
    assert setup_mod._install_looks_configured(config) is True


def test_base_url_alone_is_existing(monkeypatch):
    _no_env(monkeypatch)
    assert setup_mod._install_looks_configured({"model": {"provider": "auto", "base_url": "http://x"}}) is True


def test_anthropic_env_key_alone_is_existing(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr(
        setup_mod, "get_env_value",
        lambda key: "sk-ant" if key == "ANTHROPIC_API_KEY" else None)
    assert setup_mod._install_looks_configured({"model": {"provider": "auto"}}) is True


def test_fresh_install_still_first_time(monkeypatch):
    _no_env(monkeypatch)
    assert setup_mod._install_looks_configured({"model": {"provider": "auto"}}) is False
    assert setup_mod._install_looks_configured({}) is False
