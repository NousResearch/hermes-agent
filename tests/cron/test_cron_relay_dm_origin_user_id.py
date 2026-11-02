"""A cron fire into a relay DM carries the origin's ``user_id``, so a COLD routing cache still egresses.

The relay connector resolves a guild-less DM (Telegram, WhatsApp, Matrix, Signal) to a tenant ONLY from
``metadata.user_id``, the authentic recipient. The RelayAdapter fills it from a cache learned from inbound
events, which is empty after every process start. A backend that stops the guest on sleep restarts the
gateway on every wake, so the first cron fire after a wake sent the DM with no discriminator and the
connector declined it ("target is not an approved destination for this connection"). The persisted job
origin already holds the user, exactly as it holds ``scope_id`` for scoped chats.
"""

import asyncio
import threading

import pytest

import cron.scheduler_delivery as sd
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import SendResult

DM_USER = "861704526"


class _Relay:
    """A live RelayAdapter fronting every logical platform: the connector, not the gateway, owns the credential."""

    def __init__(self):
        self.sent: list[dict] = []

    def fronts_platform(self, _platform):
        return True

    async def send_for_platform(self, platform, chat_id, content, metadata=None):
        self.sent.append(dict(metadata or {}))
        return SendResult(success=True, message_id="m1")


class _Native(_Relay):
    async def send(self, chat_id, content, metadata=None):
        return await self.send_for_platform(None, chat_id, content, metadata)


def _target(origin, *, deliver_to=None, adapters=None, loop=None):
    """The ``_TargetDelivery`` the real per-target prologue builds for ``deliver_to`` (default: the origin)."""
    deliver_to = deliver_to or {"platform": origin["platform"], "chat_id": origin["chat_id"]}
    t = sd._prepare_target_delivery(
        {"id": "job-1", "origin": origin}, deliver_to, adapters=adapters or {Platform.RELAY: _Relay()},
        loop=loop, config=GatewayConfig(), notify_delivery=True, mirror_enabled=False, mirror_text="",
        delivery_errors=[])
    assert t is not None
    return t


def test_origin_discriminators_ride_relay_route_and_media_metadata():
    t = _target({"platform": "slack", "chat_id": "C123", "user_id": DM_USER, "scope_id": "T0AAAA111"})
    _thread, route_metadata, media_metadata = sd._live_route_metadata(t)
    for metadata in (route_metadata, media_metadata):
        assert (metadata["user_id"], metadata["scope_id"]) == (DM_USER, "T0AAAA111")


@pytest.mark.parametrize("native, deliver_to", [
    (True, None),  # a native adapter never reads it
    (False, {"platform": "telegram", "chat_id": "555"}),  # a fan-out recipient is not the origin's author
])
def test_no_user_id_off_the_relay_or_for_fan_out_targets(native, deliver_to):
    adapters = {Platform.TELEGRAM: _Native()} if native else None
    t = _target({"platform": "telegram", "chat_id": DM_USER, "user_id": DM_USER}, deliver_to=deliver_to, adapters=adapters)
    assert t.is_relay is not native
    _thread, route_metadata, media_metadata = sd._live_route_metadata(t)
    assert "user_id" not in route_metadata and "user_id" not in media_metadata


def test_cold_adapter_send_reaches_the_transport_with_user_id(monkeypatch):
    """End to end through the live lane: what the connector sees on the wire for a cold-cache relay DM."""
    monkeypatch.setattr(sd, "_maybe_mirror_cron_delivery", lambda *a, **k: None)
    loop = asyncio.new_event_loop()
    th = threading.Thread(target=loop.run_forever, daemon=True)
    th.start()
    try:
        relay = _Relay()
        t = _target(
            {"platform": "telegram", "chat_id": DM_USER, "user_id": DM_USER}, adapters={Platform.RELAY: relay},
            loop=loop)
        target_errors, delivery_errors = [], []
        sd._deliver_via_live_adapter(
            t, "hi", [], target_errors=target_errors, delivery_errors=delivery_errors, unverified_targets=[])
    finally:
        loop.call_soon_threadsafe(loop.stop)
        th.join(5)
        loop.close()
    assert [m.get("user_id") for m in relay.sent] == [DM_USER]
