"""A2A adapter must tolerate malformed A2A_PORT / extra.port."""

from __future__ import annotations

from gateway.config import PlatformConfig
from plugins.platforms.a2a.adapter import A2AAdapter, _DEFAULT_PORT


def test_malformed_a2a_port_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("A2A_PORT", "not-a-port")
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    assert adapter.port == _DEFAULT_PORT


def test_malformed_extra_port_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("A2A_PORT", raising=False)
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": "nope"}))
    assert adapter.port == _DEFAULT_PORT


def test_valid_a2a_port_env_is_honored(monkeypatch):
    monkeypatch.setenv("A2A_PORT", "9911")
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    assert adapter.port == 9911
