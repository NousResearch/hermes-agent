"""Invariants for the shared adapter boilerplate in ``gateway/platforms/_shared.py``.

Contract under test: every plugin platform's env-driven enablement reads through the ONE
profile-scoped reader (never an ambient ``os.environ``/``get_env_value`` read), and the table-driven
``apply_yaml_bridge`` seeds ``extra`` for a secondary profile even though it must not write env.
"""

from __future__ import annotations

import importlib
import os

import pytest

import agent.secret_scope as ss
import gateway.platforms._shared as shared
import hermes_cli.config as cli_config

_ENV_ENABLEMENT_PLUGINS = ("buzz", "google_chat", "irc", "line", "ntfy", "photon", "simplex", "teams")


@pytest.fixture(autouse=True)
def _multiplex_off():
    prev = ss._MULTIPLEX_ACTIVE
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(prev)


@pytest.mark.parametrize("plugin", _ENV_ENABLEMENT_PLUGINS)
def test_env_enablement_reads_only_through_scoped_getter(plugin, monkeypatch):
    """A secondary profile is seeded only from its own scope: every var the hook consults must go through
    ``_shared.get_scoped_secret`` (a fake scope answers them) and never through a raw env read."""
    mod = importlib.import_module(f"plugins.platforms.{plugin}.adapter")
    # Anything the plugin resolves through raw env would be visible here; the scoped reader hides it.
    for var in list(os.environ):
        if var.startswith(plugin.upper().replace("GOOGLE_CHAT", "GOOGLE")):
            monkeypatch.delenv(var, raising=False)
    fake_scope = {
        "BUZZ_RELAY_URL": "wss://relay", "BUZZ_PRIVATE_KEY": "k" * 8, "BUZZ_CHANNELS": "a,b",
        "GOOGLE_CHAT_PROJECT_ID": "p", "GOOGLE_CHAT_SUBSCRIPTION_NAME": "s", "GOOGLE_CHAT_MAX_MESSAGES": "3",
        "IRC_SERVER": "irc.example", "IRC_CHANNEL": "#h", "IRC_PORT": "6697",
        "LINE_CHANNEL_ACCESS_TOKEN": "t", "LINE_CHANNEL_SECRET": "s", "LINE_PORT": "8081",
        "NTFY_TOPIC": "topic", "NTFY_TOKEN": "tok",
        "PHOTON_PROJECT_ID": "pid", "PHOTON_PROJECT_SECRET": "sec", "PHOTON_HOME_CHANNEL": "line-1",
        "SIMPLEX_WS_URL": "ws://x", "SIMPLEX_GROUP_ALLOWED": "g",
        "TEAMS_CLIENT_ID": "c", "TEAMS_CLIENT_SECRET": "s", "TEAMS_TENANT_ID": "t", "TEAMS_PORT": "3979",
    }
    scoped_hits: list[str] = []
    raw_hits: list[str] = []
    real = shared.get_scoped_secret

    def spy(name, default=None, **kw):
        scoped_hits.append(name)
        return fake_scope.get(name, default)

    monkeypatch.setattr(shared, "get_scoped_secret", spy)
    monkeypatch.setattr(shared, "_scoped_get_secret", lambda name, default=None: fake_scope.get(name, default))
    monkeypatch.setattr(cli_config, "get_env_value", lambda key: raw_hits.append(key))
    monkeypatch.setattr(os, "getenv", lambda key, default=None: raw_hits.append(key) or default)

    seed = mod._env_enablement()

    assert real is not shared.get_scoped_secret  # the spy really replaced the seam
    assert seed, f"{plugin}: fake scope holds the required vars, hook must enable"
    assert scoped_hits, f"{plugin}: enablement never touched the scoped reader"
    assert not raw_hits, f"{plugin}: raw env reads during enablement: {raw_hits}"


def test_home_channel_name_defaults_to_home_everywhere(monkeypatch):
    """One rule for ``home_channel.name`` across adapters: ``<HOME_ENV>_NAME`` else the literal ``Home``
    (irc/ntfy/buzz previously used the chat id as its own name)."""
    monkeypatch.setattr(shared, "get_scoped_secret", lambda n, d=None, **k: {"X_HOME": "room"}.get(n, d))
    seed = shared.seed_extra_from_env((), home_env="X_HOME")
    assert seed["home_channel"] == {"chat_id": "room", "name": "Home"}
    monkeypatch.setattr(shared, "get_scoped_secret", lambda n, d=None, **k: {"X_HOME": "room", "X_HOME_NAME": "Ops"}.get(n, d))
    assert shared.seed_extra_from_env((), home_env="X_HOME")["home_channel"]["name"] == "Ops"
    assert shared.seed_extra_from_env((), home_env="X_HOME", home_default="fallback")["home_channel"]["chat_id"] == "room"
    monkeypatch.setattr(shared, "get_scoped_secret", lambda n, d=None, **k: d)
    assert shared.seed_extra_from_env((), home_env="X_HOME", home_default="fallback")["home_channel"]["chat_id"] == "fallback"
    assert "home_channel" not in shared.seed_extra_from_env((), home_env="X_HOME")


def test_buzz_yaml_bridge_seeds_extra_for_a_secondary_profile(monkeypatch):
    """Under a secondary profile's scope the env write is skipped, so the hook MUST return the values for
    ``PlatformConfig.extra`` — a ``None`` return left a secondary Buzz profile with neither."""
    import plugins.platforms.buzz.adapter as buzz

    for var in ("BUZZ_RELAY_URL", "BUZZ_REPLY_TO_MODE", "BUZZ_CHANNELS", "BUZZ_REQUIRE_MENTION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(shared, "profile_scoped", lambda: True)
    seeded = buzz._apply_yaml_config({}, {"extra": {
        "relay_url": "wss://p.relay", "reply_to_mode": "off", "channels": ["a", "b"], "require_mention": False}})
    assert seeded == {"relay_url": "wss://p.relay", "reply_to_mode": "off", "channels": ["a", "b"], "require_mention": False}
    assert not any(v in os.environ for v in ("BUZZ_RELAY_URL", "BUZZ_REPLY_TO_MODE", "BUZZ_CHANNELS", "BUZZ_REQUIRE_MENTION"))

    monkeypatch.setattr(shared, "profile_scoped", lambda: False)
    buzz._apply_yaml_config({}, {"extra": {"reply_to_mode": "off", "channels": ["a", "b"], "require_mention": False}})
    assert os.environ["BUZZ_REPLY_TO_MODE"] == "off"
    assert os.environ["BUZZ_CHANNELS"] == "a,b"
    assert os.environ["BUZZ_REQUIRE_MENTION"] == "false"
    for var in ("BUZZ_REPLY_TO_MODE", "BUZZ_CHANNELS", "BUZZ_REQUIRE_MENTION"):
        monkeypatch.delenv(var, raising=False)


def test_extra_or_secret_honours_explicit_false_but_not_blank(monkeypatch):
    monkeypatch.setattr(shared, "get_scoped_secret", lambda n, d=None, **k: f"env:{d}")
    assert shared.extra_or_secret({"require_mention": False}, "require_mention", "X", "true") is False
    assert shared.extra_or_secret({"require_mention": ""}, "require_mention", "X", "true") == "env:true"
    assert shared.extra_or_secret(None, "require_mention", "X", "true") == "env:true"


def test_external_fallback_consults_profile_scope_only_when_unscoped(monkeypatch):
    """Buzz's startup gate rung: a Bitwarden-managed key is visible through ``external_fallback`` before any
    scope exists, but never overrides an installed (authoritative) scope."""
    monkeypatch.delenv("K", raising=False)
    monkeypatch.setattr(shared, "_UNSCOPED_PROFILE_SECRETS", {"K": "external"})
    assert shared.get_scoped_secret("K") is None
    assert shared.get_scoped_secret("K", external_fallback=True) == "external"
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"OTHER": "x"})
    try:
        assert shared.get_scoped_secret("K", "dflt", external_fallback=True) == "dflt"
    finally:
        ss.reset_secret_scope(token)
