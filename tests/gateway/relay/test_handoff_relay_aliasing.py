"""Unit tests for /handoff platform aliasing over relay (Phase 1 parity).

Symptom fixed: on a relay-fronted gateway, ``/handoff discord`` was rejected
in TWO places even though "discord" is deliverable through the relay adapter:

  1. The CLI pre-check (``cli_commands_mixin._handle_handoff_command``) looked
     up ``gw_config.platforms.get(DISCORD)`` — only a RELAY entry exists.
  2. The gateway watcher (``run.py _process_handoff``) did a literal
     ``adapters.get(discord)`` — the adapter is registered under RELAY.

Both now resolve fronted platforms through the same alias-aware machinery the
delivery router uses (``resolve_delivery_transport`` — native adapter wins;
relay eligible only when its authenticated transport advertises the logical
platform). These tests cover the resolver semantics the watcher depends on,
plus the CLI pre-check's env-derived fronted set.
"""

from __future__ import annotations

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.delivery import resolve_delivery_transport


class _RelayStub:
    """Minimal relay adapter double advertising a fronted-platform set."""

    def __init__(self, fronted):
        self._fronted = set(fronted)
        self.sent = []

    def fronts_platform(self, platform):
        value = getattr(platform, "value", platform)
        return str(value) in self._fronted

    async def send_for_platform(self, logical_platform, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((getattr(logical_platform, "value", logical_platform), chat_id, content, metadata))
        return type("R", (), {"success": True, "message_id": "m-1", "error": None})()

    async def send(self, chat_id, content, reply_to=None, metadata=None):  # pragma: no cover
        raise AssertionError("relay transport must use send_for_platform, not send")


def _config_with(platforms):
    cfg = GatewayConfig()
    cfg.platforms = platforms
    return cfg


class TestHandoffRelayAliasing:
    def test_fronted_platform_resolves_to_relay_adapter(self):
        """The watcher's exact miss: adapters holds only RELAY, target is discord."""
        relay = _RelayStub({"discord"})
        cfg = _config_with({Platform.RELAY: PlatformConfig(enabled=True)})
        transport = resolve_delivery_transport(
            Platform.DISCORD, cfg, {Platform.RELAY: relay}
        )
        assert transport is not None
        assert transport.adapter is relay
        assert transport.is_relay is True

    def test_unfronted_platform_still_fails(self):
        """Aliasing must not let relay hijack a platform it does not front."""
        relay = _RelayStub({"telegram"})
        cfg = _config_with({Platform.RELAY: PlatformConfig(enabled=True)})
        transport = resolve_delivery_transport(
            Platform.DISCORD, cfg, {Platform.RELAY: relay}
        )
        assert transport is None

    def test_native_adapter_wins_over_relay(self):
        native = object()
        relay = _RelayStub({"discord"})
        cfg = _config_with(
            {
                Platform.DISCORD: PlatformConfig(enabled=True),
                Platform.RELAY: PlatformConfig(enabled=True),
            }
        )
        transport = resolve_delivery_transport(
            Platform.DISCORD, cfg, {Platform.DISCORD: native, Platform.RELAY: relay}
        )
        assert transport is not None
        assert transport.adapter is native
        assert transport.is_relay is False

    @pytest.mark.asyncio
    async def test_transport_send_stamps_logical_platform(self):
        """The handoff reply leg must go through send_for_platform so the
        outbound frame carries the logical platform tag."""
        relay = _RelayStub({"discord"})
        cfg = _config_with({Platform.RELAY: PlatformConfig(enabled=True)})
        transport = resolve_delivery_transport(
            Platform.DISCORD, cfg, {Platform.RELAY: relay}
        )
        assert transport is not None
        result = await transport.send(
            Platform.DISCORD, "chan-1", "handed-off reply", {"thread_id": "t-9"}
        )
        assert result.success is True
        assert relay.sent == [("discord", "chan-1", "handed-off reply", {"thread_id": "t-9"})]


class TestCliHandoffFrontedSet:
    """The CLI pre-check derives the fronted set from deploy env (no live adapter)."""

    def test_fronted_set_from_env(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord,telegram")
        monkeypatch.delenv("GATEWAY_RELAY_BOT_IDS", raising=False)
        from gateway.relay import relay_platform_identities

        fronted = {p for p, _ in relay_platform_identities()}
        assert "discord" in fronted
        assert "telegram" in fronted
        assert "slack" not in fronted

    def test_unconfigured_env_yields_generic_relay_only(self, monkeypatch):
        monkeypatch.delenv("GATEWAY_RELAY_PLATFORMS", raising=False)
        from gateway.relay import relay_platform_identities

        fronted = {p for p, _ in relay_platform_identities()}
        assert fronted == {"relay"}
