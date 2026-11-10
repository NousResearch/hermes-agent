"""RelayAdapter registration via the platform registry.

The relay platform is registered when a connector relay URL is configured
(``GATEWAY_RELAY_URL`` env or ``gateway.relay_url`` in config.yaml) — the same
config-driven shape as ``gateway.proxy_url``, not a separate feature flag. With
no URL configured, registration is a no-op so direct/single-tenant deployments
are unaffected. ``force=True`` registers a transport-less adapter for tests.
"""

from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from gateway.platform_registry import platform_registry
from gateway.relay import register_relay_adapter, relay_url
from gateway.relay.adapter import RelayAdapter


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    """Each test starts/ends with no 'relay' entry and a clean relay env."""
    monkeypatch.delenv("GATEWAY_RELAY_URL", raising=False)
    monkeypatch.delenv("GATEWAY_RELAY_PLATFORM", raising=False)
    monkeypatch.delenv("GATEWAY_RELAY_BOT_ID", raising=False)
    platform_registry.unregister("relay")
    yield
    platform_registry.unregister("relay")


def test_off_when_no_url_configured(monkeypatch):
    # No GATEWAY_RELAY_URL and (assuming) no gateway.relay_url in config.
    monkeypatch.setattr("gateway.relay.relay_url", lambda: None)
    assert register_relay_adapter() is False
    assert platform_registry.is_registered("relay") is False


def test_registers_when_url_configured(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_URL", "wss://connector.example/relay")
    assert relay_url() == "wss://connector.example/relay"
    assert register_relay_adapter() is True
    assert platform_registry.is_registered("relay") is True




class TestConfigRegistrationAgreementUnderMultiplexScope:
    """The config loader and the relay registration path must resolve the SAME
    GATEWAY_RELAY_URL under an active multiplex profile scope — the routing
    stamp is process-global (agent/secret_scope.py), so neither side may see a
    value the other doesn't. Regression for the split-brain states: adapter
    registered with Platform.RELAY absent from config (process stamp dropped
    by the scoped reload), and enabled RELAY config with no adapter
    (profile-only stamp invisible to relay_url())."""

    def test_process_stamp_agrees_on_both_sides_under_scope(
        self, tmp_path, monkeypatch
    ):
        from agent import secret_scope as ss
        from gateway.config import Platform, load_gateway_config

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "gateway:\n"
            "  platforms:\n"
            "    telegram:\n"
            "      enabled: true\n"
            "      bot_token: '123:abc'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("GATEWAY_RELAY_URL", "wss://deploy.example/relay")

        profile_dir = tmp_path / "profile-a"
        profile_dir.mkdir()
        (profile_dir / ".env").write_text("", encoding="utf-8")

        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope(ss.build_profile_secret_scope(profile_dir))
        try:
            config = load_gateway_config()
            # Registration-side reader agrees with the config side.
            assert relay_url() == "wss://deploy.example/relay"
            assert register_relay_adapter() is True
        finally:
            ss.reset_secret_scope(tok)
            ss.set_multiplex_active(False)

        assert config.platforms[Platform.RELAY].enabled is True
        assert config.platforms[Platform.TELEGRAM].enabled is False
        # The registered factory actually constructs an adapter with a live
        # transport dialing the same URL — the connect loop's contract.
        adapter = platform_registry.create_adapter(
            "relay", config.platforms[Platform.RELAY]
        )
        assert isinstance(adapter, RelayAdapter)
        # A URL-configured registration builds a live transport (force=True's
        # transport-less posture is the test-only path) — private attr, no
        # public accessor on the adapter.
        assert adapter._transport is not None

    def test_profile_only_stamp_is_inert_on_both_sides_under_scope(
        self, tmp_path, monkeypatch
    ):
        from agent import secret_scope as ss
        from gateway.config import Platform, load_gateway_config

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "gateway:\n"
            "  platforms:\n"
            "    telegram:\n"
            "      enabled: true\n"
            "      bot_token: '123:abc'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        profile_dir = tmp_path / "profile-a"
        profile_dir.mkdir()
        (profile_dir / ".env").write_text(
            "GATEWAY_RELAY_URL=wss://profile.example/relay\n", encoding="utf-8"
        )

        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope(ss.build_profile_secret_scope(profile_dir))
        try:
            config = load_gateway_config()
            # Both sides agree the stamp is absent: no half-enabled state.
            assert relay_url() is None
            assert register_relay_adapter() is False
        finally:
            ss.reset_secret_scope(tok)
            ss.set_multiplex_active(False)

        relay_cfg = config.platforms.get(Platform.RELAY)
        assert relay_cfg is None or relay_cfg.enabled is False
        assert config.platforms[Platform.TELEGRAM].enabled is True
