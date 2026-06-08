"""RelayAdapter registration via the platform registry (relay Phase 1, Task 1.3).

Verifies the relay platform is registered ONLY behind the flag (dark-launch),
constructed through the same registry path as plugin adapters.
"""

from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from gateway.platform_registry import platform_registry
from gateway.relay import register_relay_adapter, relay_enabled
from gateway.relay.adapter import RelayAdapter


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    """Ensure each test starts/ends with no 'relay' entry and a clean env."""
    monkeypatch.delenv("HERMES_GATEWAY_RELAY", raising=False)
    platform_registry.unregister("relay")
    yield
    platform_registry.unregister("relay")


def test_off_by_default():
    assert relay_enabled() is False
    assert register_relay_adapter() is False
    assert platform_registry.is_registered("relay") is False


def test_enabled_by_env_flag(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_RELAY", "1")
    assert relay_enabled() is True
    assert register_relay_adapter() is True
    assert platform_registry.is_registered("relay") is True


def test_force_registers_without_flag():
    assert register_relay_adapter(force=True) is True
    assert platform_registry.is_registered("relay") is True


def test_create_adapter_yields_relay_adapter():
    register_relay_adapter(force=True)
    adapter = platform_registry.create_adapter("relay", PlatformConfig())
    assert isinstance(adapter, RelayAdapter)
    # Placeholder descriptor until handshake negotiates the real one.
    assert adapter.descriptor.platform == "relay"


@pytest.mark.parametrize("val,expected", [("0", False), ("", False), ("true", True), ("ON", True), ("yes", True)])
def test_flag_parsing(monkeypatch, val, expected):
    monkeypatch.setenv("HERMES_GATEWAY_RELAY", val)
    assert relay_enabled() is expected
