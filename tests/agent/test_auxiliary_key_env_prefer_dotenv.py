"""Regression tests for #67935.

A long-lived ``hermes serve`` (Desktop local gateway) snapshots ``os.environ``
once at spawn. A custom provider's ``key_env`` was resolved via plain
``os.getenv()``, so a key added or rotated in ``~/.hermes/.env`` mid-session
never reached the backend — the request fell through to the ``no-key-required``
placeholder and 401'd until restart.

``agent.auxiliary_client._resolve_config_key_env()`` now routes through
``get_env_value_prefer_dotenv()``, so a fresh ``.env`` value wins over a stale
inherited ``os.environ`` value. These pin that invariant.
"""
from pathlib import Path

import pytest


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and clear the test key from os.environ."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("LONGCAT_API_KEY", raising=False)
    # load_env caches by (path, mtime, size); reset so a prior test's cache
    # doesn't mask this home's .env.
    from hermes_cli import config as _config

    _config.invalidate_env_cache()
    return home


def _write_env(home: Path, **kwargs) -> None:
    (home / ".env").write_text(
        "\n".join(f"{k}={v}" for k, v in kwargs.items()) + "\n", encoding="utf-8"
    )


def test_resolve_key_env_reads_fresh_dotenv(isolated_hermes_home):
    """Key present only in .env (not os.environ) resolves — the mid-session add."""
    from agent.auxiliary_client import _resolve_config_key_env

    _write_env(isolated_hermes_home, LONGCAT_API_KEY="sk-from-dotenv")
    from hermes_cli import config as _config

    _config.invalidate_env_cache()

    assert _resolve_config_key_env("LONGCAT_API_KEY") == "sk-from-dotenv"


def test_resolve_key_env_prefers_dotenv_over_stale_environ(
    isolated_hermes_home, monkeypatch
):
    """A rotated .env value wins over a stale value inherited in os.environ."""
    from agent.auxiliary_client import _resolve_config_key_env

    # Backend spawned with an old key baked into os.environ ...
    monkeypatch.setenv("LONGCAT_API_KEY", "sk-stale-from-shell")
    # ... user rotates it in ~/.hermes/.env mid-session.
    _write_env(isolated_hermes_home, LONGCAT_API_KEY="sk-rotated-in-dotenv")
    from hermes_cli import config as _config

    _config.invalidate_env_cache()

    assert _resolve_config_key_env("LONGCAT_API_KEY") == "sk-rotated-in-dotenv"


def test_resolve_key_env_empty_when_unset(isolated_hermes_home):
    """No key anywhere → empty string (caller falls back to no-key-required)."""
    from agent.auxiliary_client import _resolve_config_key_env

    assert _resolve_config_key_env("LONGCAT_API_KEY") == ""
    assert _resolve_config_key_env("") == ""


def test_named_custom_provider_entry_resolves_key_from_dotenv(isolated_hermes_home):
    """The named-custom-provider resolution shape (the #67935 repro config):
    an entry declaring ``key_env`` resolves to the live .env key rather than
    falling through to the ``no-key-required`` placeholder."""
    from agent.auxiliary_client import _resolve_config_key_env

    _write_env(isolated_hermes_home, LONGCAT_API_KEY="sk-live-key")
    from hermes_cli import config as _config

    _config.invalidate_env_cache()

    # Mirrors resolve_provider_client's custom_entry handling at the 401 site.
    custom_entry = {
        "name": "longcat",
        "base_url": "https://api.longcat.chat/openai",
        "key_env": "LONGCAT_API_KEY",
    }
    custom_key = (custom_entry.get("api_key") or "").strip()
    custom_key_env = (
        custom_entry.get("key_env") or custom_entry.get("api_key_env") or ""
    ).strip()
    if not custom_key and custom_key_env:
        custom_key = _resolve_config_key_env(custom_key_env)
    custom_key = custom_key or "no-key-required"

    assert custom_key == "sk-live-key"
    assert custom_key != "no-key-required"
