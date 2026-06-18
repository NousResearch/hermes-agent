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


def test_explicit_url_arg_registers():
    assert register_relay_adapter(url="wss://connector.example/relay") is True
    assert platform_registry.is_registered("relay") is True


def test_force_registers_without_url():
    assert register_relay_adapter(force=True) is True
    assert platform_registry.is_registered("relay") is True


def test_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_URL", "wss://connector.example/relay/")
    assert relay_url() == "wss://connector.example/relay"


def test_create_adapter_yields_relay_adapter():
    # force=True builds a transport-less adapter (no live dial in unit tests).
    register_relay_adapter(force=True)
    adapter = platform_registry.create_adapter("relay", PlatformConfig())
    assert isinstance(adapter, RelayAdapter)
    # Placeholder descriptor until handshake negotiates the real one.
    assert adapter.descriptor.platform == "relay"
