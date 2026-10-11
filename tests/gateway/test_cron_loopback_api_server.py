"""A cron provider that fires over loopback makes the gateway's api_server required.

Chronos fires reach the dashboard, which forwards them to the gateway api_server on
127.0.0.1. A ``platforms.api_server.enabled: false`` left by the Channels toggle, or a key lost
in a ``.env`` rewrite, used to leave that listener down and every fire 503ing forever while the
gateway looked healthy. These tests pin the contract through the runner's real config loader.
"""
from __future__ import annotations

import pytest

import hermes_yaml as yaml
from gateway.config import Platform

_KEY = "k" * 64


class _LoopbackProvider:
    """Stand-in for an external provider that declares loopback fires (the Chronos shape)."""

    fires_over_loopback = True
    name = "loopback-fake"

    def is_available(self):
        return True


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch):
    import plugins.cron_providers as pc
    from agent import secret_scope as ss

    monkeypatch.setattr(
        pc, "load_cron_scheduler",
        lambda name: _LoopbackProvider() if name == "loopback-fake" else None)
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    monkeypatch.delenv("API_SERVER_ENABLED", raising=False)
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def _home(monkeypatch):
    from gateway import run as run_mod
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    monkeypatch.setattr(run_mod, "get_hermes_home", lambda: home)
    return home


def _write(home, *, provider, env="", multiplex=False):
    cfg = {
        "gateway": {"multiplex_profiles": multiplex},
        "platforms": {"api_server": {"enabled": False}},
    }
    if provider:
        cfg["cron"] = {"provider": provider}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (home / ".env").write_text(env, encoding="utf-8")


def _api(cfg):
    return cfg.platforms.get(Platform.API_SERVER)


def test_loopback_provider_overrides_dashboard_disable(monkeypatch):
    from gateway import run as run_mod

    home = _home(monkeypatch)
    _write(home, provider="loopback-fake", env=f"API_SERVER_KEY={_KEY}\n")
    monkeypatch.setenv("API_SERVER_KEY", _KEY)  # the loaded .env, as in a live gateway

    cfg = run_mod.load_gateway_config_for_runner()

    assert _api(cfg) is not None and _api(cfg).enabled is True
    assert _api(cfg).extra["key"] == _KEY
    # Bind address is left to the adapter default (127.0.0.1), not widened.
    assert "host" not in _api(cfg).extra
    assert Platform.API_SERVER in cfg.get_connected_platforms()


def test_missing_key_is_generated_and_persisted_in_profile_env(monkeypatch):
    from agent.secret_scope import load_env_file
    from gateway import run as run_mod

    home = _home(monkeypatch)
    _write(home, provider="loopback-fake", env="OTHER=1\n")

    cfg = run_mod.load_gateway_config_for_runner()

    persisted = load_env_file(home / ".env")
    key = persisted.get("API_SERVER_KEY", "")
    assert len(key) == 64
    assert persisted.get("OTHER") == "1"  # canonical writer, other lines kept
    assert _api(cfg).enabled is True and _api(cfg).extra["key"] == key

    # Second boot reuses the persisted key instead of minting a new one.
    cfg2 = run_mod.load_gateway_config_for_runner()
    assert _api(cfg2).extra["key"] == key
    assert load_env_file(home / ".env")["API_SERVER_KEY"] == key


def test_operator_container_key_is_never_shadowed(monkeypatch):
    from agent.secret_scope import load_env_file
    from gateway import run as run_mod

    home = _home(monkeypatch)
    _write(home, provider="loopback-fake")
    operator = "o" * 40
    monkeypatch.setenv("API_SERVER_KEY", operator)

    cfg = run_mod.load_gateway_config_for_runner()

    assert _api(cfg).enabled is True and _api(cfg).extra["key"] == operator
    assert "API_SERVER_KEY" not in load_env_file(home / ".env")


def test_weak_operator_key_is_not_replaced(monkeypatch):
    """A generated .env key would shadow the operator's (``.env`` loads with override)."""
    from agent.secret_scope import load_env_file
    from gateway import run as run_mod

    home = _home(monkeypatch)
    _write(home, provider="loopback-fake")
    monkeypatch.setenv("API_SERVER_KEY", "short")

    cfg = run_mod.load_gateway_config_for_runner()

    assert "API_SERVER_KEY" not in load_env_file(home / ".env")
    assert not (_api(cfg) and _api(cfg).enabled)


@pytest.mark.parametrize("provider", [None, "builtin"])
def test_builtin_ticker_keeps_user_disable(monkeypatch, provider):
    from agent.secret_scope import load_env_file
    from gateway import run as run_mod

    home = _home(monkeypatch)
    _write(home, provider=provider, env=f"API_SERVER_KEY={_KEY}\n")
    monkeypatch.setenv("API_SERVER_KEY", _KEY)

    cfg = run_mod.load_gateway_config_for_runner()
    assert _api(cfg) is not None and _api(cfg).enabled is False

    # And no key is minted for a self-hosted gateway that never asked for the listener.
    monkeypatch.delenv("API_SERVER_KEY")
    _write(home, provider=provider)
    run_mod.load_gateway_config_for_runner()
    assert "API_SERVER_KEY" not in load_env_file(home / ".env")


def test_multiplex_with_secondaries_does_not_force(monkeypatch):
    """Several served homes → external providers fall back to the built-in ticker
    (``scheduler_for_profile_mode``), so nothing fires over loopback and no profile is forced."""
    from gateway import run as run_mod

    home = _home(monkeypatch)
    _write(home, provider="loopback-fake", env=f"API_SERVER_KEY={_KEY}\n", multiplex=True)
    secondary = home / "profiles" / "worker"
    secondary.mkdir(parents=True)
    _write(secondary, provider="loopback-fake", env=f"API_SERVER_KEY={_KEY}\n")

    cfg = run_mod.load_gateway_config_for_runner()

    assert cfg.multiplex_profiles is True
    assert _api(cfg) is not None and _api(cfg).enabled is False


def test_multiplex_single_home_forces_on_default_listener(monkeypatch):
    """Explicit multiplex with only the default profile: the scoped default-profile reload owns
    the listener and gets it forced on from the default profile's own .env."""
    from gateway import run as run_mod

    home = _home(monkeypatch)
    _write(home, provider="loopback-fake", env=f"API_SERVER_KEY={_KEY}\n", multiplex=True)

    cfg = run_mod.load_gateway_config_for_runner()

    assert cfg.multiplex_profiles is True
    assert _api(cfg).enabled is True and _api(cfg).extra["key"] == _KEY


def test_requirement_follows_profile_scope_a_b_a(tmp_path):
    """The requirement is read from the BOUND profile's config, not the launch profile's."""
    from cron.loopback_fire import loopback_api_server_required
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    a, b = tmp_path / "a", tmp_path / "b"
    for home, provider in ((a, "loopback-fake"), (b, "builtin")):
        home.mkdir()
        (home / "config.yaml").write_text(
            yaml.safe_dump({"cron": {"provider": provider}}), encoding="utf-8")

    seen = []
    for home in (a, b, a):
        token = set_hermes_home_override(str(home))
        try:
            seen.append(loopback_api_server_required(home_count=1))
        finally:
            reset_hermes_home_override(token)
    assert seen == [True, False, True]


def test_chronos_declares_loopback_fires():
    from cron.loopback_fire import provider_fires_over_loopback
    from cron.scheduler_provider import InProcessCronScheduler
    from plugins.cron_providers.chronos import ChronosCronScheduler

    assert provider_fires_over_loopback(ChronosCronScheduler()) is True
    assert provider_fires_over_loopback(InProcessCronScheduler()) is False
