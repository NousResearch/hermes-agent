"""Regression tests for adapter-creation env freshness (#52293).

When proxy_manager / a secret rotation silently rewrites ``~/.hermes/.env``
while the gateway is already running, the next adapter (re)build must pick up
the new values WITHOUT a full gateway restart. The reload must go through the
safe seam ``_reload_runtime_env_preserving_config_authority`` so that:

* it covers BOTH the startup path and the reconnect path (single seam inside
  ``_create_adapter``);
* under multiplexing it is a NO-OP for credential reload, so a secondary
  profile's adapter / subprocess can never see the default profile's secrets;
* it never runs in the per-turn hot path, so prompt caching is untouched.

The fake adapter is injected through ``platform_registry`` for a real
``Platform`` member, so we exercise the real ``GatewayRunner._create_adapter``
code path (including its env-reload call) with zero network/aiohttp deps.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent import secret_scope
from gateway import run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import platform_registry


class _FreshnessAdapter:
    """Minimal adapter that records the proxy it was built with."""

    def __init__(self, config: PlatformConfig) -> None:
        # Capture the env value at construction time — this proves the reload
        # ran *before* the adapter read its settings.
        self.built_with_proxy = os.environ.get("HTTP_PROXY")
        self.config = config

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None


@pytest.fixture
def fake_adapter_registry(monkeypatch):
    """Make ``_create_adapter`` return our probe for a real Platform member."""
    name = "telegram"  # real enum member, avoids enum hacking / telegram import
    orig_is = platform_registry.is_registered
    orig_create = platform_registry.create_adapter

    def _is_registered(n: str) -> bool:
        return True if n == name else orig_is(n)

    def _create_adapter(n: str, cfg):
        if n == name:
            return _FreshnessAdapter(cfg)
        return orig_create(n, cfg)

    monkeypatch.setattr(platform_registry, "is_registered", _is_registered)
    monkeypatch.setattr(platform_registry, "create_adapter", _create_adapter)
    yield name


def _write_env(home: Path, proxy: str) -> None:
    (home / ".env").write_text(f"HTTP_PROXY={proxy}\n", encoding="utf-8")


def test_create_adapter_picks_up_swapped_dotenv(fake_adapter_registry, tmp_path, monkeypatch):
    """A .env swapped after boot is seen by the next _create_adapter call."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    _write_env(hermes_home, "http://old-proxy:8080")

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.delenv("HTTP_PROXY", raising=False)

    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = gateway_run.GatewayRunner(config)

    # First build reads the current .env.
    a1 = runner._create_adapter(Platform.TELEGRAM, config.platforms[Platform.TELEGRAM])
    assert isinstance(a1, _FreshnessAdapter)
    assert a1.built_with_proxy == "http://old-proxy:8080"

    # proxy_manager silently swaps the .env file (no restart, no env export).
    _write_env(hermes_home, "http://new-proxy:9090")

    # Next (re)build — startup OR reconnect path — must see the new value.
    a2 = runner._create_adapter(Platform.TELEGRAM, config.platforms[Platform.TELEGRAM])
    assert isinstance(a2, _FreshnessAdapter)
    assert a2.built_with_proxy == "http://new-proxy:9090"


def test_create_adapter_reload_is_noop_for_secrets_under_multiplex(
    fake_adapter_registry, tmp_path, monkeypatch
):
    """Under multiplexing the credential reload must be skipped (no leak).

    The safe seam early-returns in multiplex mode so it does NOT mutate global
    os.environ. We prove _create_adapter uses that seam by asserting the global
    env is left untouched by the swapped .env (no cross-profile credential leak).
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    _write_env(hermes_home, "http://should-not-leak:8080")

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    # Simulate a secondary profile turn: multiplex active, profile-scoped env.
    monkeypatch.setenv("HTTP_PROXY", "http://profile-scoped:7777")
    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: True)

    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = gateway_run.GatewayRunner(config)

    a = runner._create_adapter(Platform.TELEGRAM, config.platforms[Platform.TELEGRAM])
    assert isinstance(a, _FreshnessAdapter)
    # The leaked value from .env must NOT have overwritten the profile-scoped one.
    assert a.built_with_proxy == "http://profile-scoped:7777"
