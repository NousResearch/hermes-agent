"""The ``litco_turn`` platform plugin: discovery, env enablement, and the adapter serving /health."""

from __future__ import annotations

import socket

import aiohttp
import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_plugin_registers_and_env_enables(monkeypatch):
    monkeypatch.setenv("LITCO_HOST_SECRET", "sek")
    monkeypatch.setenv("LITCO_MATTER_ID", "m1")
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
    from gateway.config import Platform, load_gateway_config
    from gateway.platform_registry import platform_registry

    entry = platform_registry.get("litco_turn")
    assert entry is not None and entry.label == "LitCo Turn Server"
    cfg = load_gateway_config()
    pcfg = cfg.platforms.get(Platform("litco_turn"))
    assert pcfg is not None and pcfg.enabled
    assert pcfg.extra.get("matter_id") == "m1"


@pytest.mark.asyncio
async def test_adapter_serves_health(monkeypatch, tmp_path):
    port = _free_port()
    monkeypatch.setenv("LITCO_HOST_SECRET", "sek")
    monkeypatch.setenv("LITCO_MATTER_ID", "m1")
    monkeypatch.setenv("LITCO_MATTER_HOME", str(tmp_path / "matter"))
    monkeypatch.setenv("LITCO_TURN_PORT", str(port))
    from gateway.config import PlatformConfig
    from plugins.platforms.litco_turn.adapter import LitcoTurnAdapter

    adapter = LitcoTurnAdapter(PlatformConfig(enabled=True, extra={}))
    assert await adapter.connect()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                body = await resp.json()
            async with session.post(f"http://127.0.0.1:{port}/turn", json={}) as resp:
                assert resp.status == 401
        assert body["matterId"] == "m1" and body["activeTurns"] == 0
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_adapter_refuses_without_secret(monkeypatch):
    monkeypatch.delenv("LITCO_HOST_SECRET", raising=False)
    monkeypatch.setenv("LITCO_TURN_PORT", str(_free_port()))
    from gateway.config import PlatformConfig
    from plugins.platforms.litco_turn.adapter import LitcoTurnAdapter

    adapter = LitcoTurnAdapter(PlatformConfig(enabled=True, extra={}))
    assert not await adapter.connect()


@pytest.mark.asyncio
async def test_adapter_binds_the_slot_port(monkeypatch, tmp_path):
    port = _free_port()
    monkeypatch.setenv("LITCO_HOST_SECRET", "sek")
    monkeypatch.setenv("LITCO_MATTER_ID", "m1")
    monkeypatch.setenv("LITCO_MATTER_HOME", str(tmp_path / "matter"))
    monkeypatch.setenv("LITCO_TURN_HOST", "127.0.0.1")
    monkeypatch.setenv("LITCO_TURN_PORT", str(_free_port()))
    monkeypatch.setenv("LITCO_SLOT_PORT", str(port))
    from gateway.config import PlatformConfig
    from plugins.platforms.litco_turn.adapter import LitcoTurnAdapter

    adapter = LitcoTurnAdapter(PlatformConfig(enabled=True, extra={}))
    assert await adapter.connect()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                body = await resp.json()
        assert body["matterId"] == "m1" and body["draining"] is False
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_adapter_refuses_a_slot_without_a_matter(monkeypatch):
    monkeypatch.setenv("LITCO_HOST_SECRET", "sek")
    monkeypatch.delenv("LITCO_MATTER_ID", raising=False)
    monkeypatch.setenv("LITCO_SLOT_PORT", str(_free_port()))
    from gateway.config import PlatformConfig
    from plugins.platforms.litco_turn.adapter import LitcoTurnAdapter

    adapter = LitcoTurnAdapter(PlatformConfig(enabled=True, extra={}))
    assert not await adapter.connect()
