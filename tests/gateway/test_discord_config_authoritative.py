"""End-to-end config-authoritative precedence for Discord bot admission.

Proves the fix at the FINAL bot authorization route -- the gateway's
``_is_user_authorized`` (``gateway/authz_mixin.py``), not the adapter admission
gate alone. That route reads ``DISCORD_ALLOW_BOTS`` from the process env
directly. After ``load_gateway_config`` runs the boot bridge, a stale /
conflicting ``DISCORD_ALLOW_BOTS`` shell value can no longer split the
authorization decision from config.yaml: config wins and the env is
force-synced to match.
"""

import os
from types import SimpleNamespace

import pytest

from gateway.config import load_gateway_config
from gateway.session import Platform, SessionSource


def _make_bare_runner():
    """GatewayRunner skeleton with just enough wiring for the auth route.

    Mirrors ``tests/gateway/test_discord_bot_auth_bypass.py``: skip the heavy
    __init__ and stub the pairing store so the real allowlist / bot-bypass path
    is exercised.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _discord_bot_source(bot_id="999888777"):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="channel",
        user_id=bot_id,
        user_name="SomeBot",
        is_bot=True,
    )


@pytest.fixture
def clean_env(monkeypatch):
    """Isolate the Discord auth env vars the boot bridge writes.

    ``load_gateway_config`` writes directly to ``os.environ``, so pop anything
    it created after the test (monkeypatch's finalizer would otherwise restore
    the loader's mid-test write for vars that were absent at setup).
    """
    managed = (
        "DISCORD_ALLOW_BOTS",
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_ROLES",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    )
    for var in managed:
        monkeypatch.delenv(var, raising=False)
    yield monkeypatch
    for var in managed:
        os.environ.pop(var, None)


def _write_discord_config(tmp_path, monkeypatch, allow_bots):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"discord:\n  allow_bots: {allow_bots}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))


def test_final_authz_route_config_admits_bot_over_stale_env(clean_env, tmp_path):
    """config allow_bots=all admits a bot at the gateway authorization route
    even when a stale DISCORD_ALLOW_BOTS=none shell value disagrees."""
    clean_env.setenv("DISCORD_ALLOW_BOTS", "none")  # stale / conflicting
    _write_discord_config(tmp_path, clean_env, "all")

    load_gateway_config()  # boot bridge force-syncs config -> env

    assert os.environ["DISCORD_ALLOW_BOTS"] == "all"
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_discord_bot_source()) is True


def test_final_authz_route_config_denies_bot_over_stale_env(clean_env, tmp_path):
    """config allow_bots=none denies a bot at the gateway authorization route
    even when a stale DISCORD_ALLOW_BOTS=all shell value would otherwise admit."""
    clean_env.setenv("DISCORD_ALLOW_BOTS", "all")  # stale / conflicting
    _write_discord_config(tmp_path, clean_env, "none")

    load_gateway_config()

    assert os.environ["DISCORD_ALLOW_BOTS"] == "none"
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_discord_bot_source()) is False
