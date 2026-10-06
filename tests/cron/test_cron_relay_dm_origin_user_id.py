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


class _Transport:
    adapter = type("Adapter", (), {})()

    def __init__(self, *, is_relay: bool):
        self.is_relay = is_relay
        self.sent: list[dict] = []

    async def send(self, platform, chat_id, content, metadata=None):
        self.sent.append(dict(metadata or {}))
        return SendResult(success=True, message_id="m1")


def _target(transport, *, origin, origin_target=True, loop=None):
    fields = {name: None for name in sd._TargetDelivery.__dataclass_fields__}
    fields.update(
        job={"id": "job-1"}, platform=Platform.TELEGRAM, platform_name="telegram", chat_id=DM_USER,
        thread_id=None, transport=transport, config=GatewayConfig(), loop=loop, target_adapters={},
        notify_delivery=True, mirror_text="", origin=origin, origin_target=origin_target,
        origin_user_id=origin.get("user_id") if origin_target else None,
    )
    return sd._TargetDelivery(**fields)


def test_origin_discriminators_ride_relay_route_and_media_metadata():
    t = _target(
        _Transport(is_relay=True),
        origin={"platform": "slack", "chat_id": "C123", "user_id": DM_USER, "scope_id": "T0AAAA111"},
    )
    _thread, route_metadata, media_metadata = sd._live_route_metadata(t)
    for metadata in (route_metadata, media_metadata):
        assert (metadata["user_id"], metadata["scope_id"]) == (DM_USER, "T0AAAA111")


@pytest.mark.parametrize("transport_is_relay, origin_target", [(False, True), (True, False)])
def test_no_user_id_off_the_relay_or_for_fan_out_targets(transport_is_relay, origin_target):
    """Native adapters never read it, and a fan-out target's recipient is not the origin's author."""
    t = _target(
        _Transport(is_relay=transport_is_relay),
        origin={"platform": "telegram", "chat_id": DM_USER, "user_id": DM_USER},
        origin_target=origin_target,
    )
    _thread, route_metadata, media_metadata = sd._live_route_metadata(t)
    assert "user_id" not in route_metadata and "user_id" not in media_metadata


def test_cold_adapter_send_reaches_the_transport_with_user_id(monkeypatch):
    """End to end through the live lane: what the connector sees on the wire for a cold-cache relay DM."""
    monkeypatch.setattr(sd, "_maybe_mirror_cron_delivery", lambda *a, **k: None)
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    try:
        transport = _Transport(is_relay=True)
        t = _target(
            transport, origin={"platform": "telegram", "chat_id": DM_USER, "user_id": DM_USER}, loop=loop)
        target_errors, delivery_errors = [], []
        sd._deliver_via_live_adapter(
            t, "hi", [], target_errors=target_errors, delivery_errors=delivery_errors, unverified_targets=[])
    finally:
        loop.call_soon_threadsafe(loop.stop)
    assert [m.get("user_id") for m in transport.sent] == [DM_USER]
