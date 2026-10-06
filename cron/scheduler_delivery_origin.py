"""Relay egress discriminators a cron fire carries from its persisted job origin."""

from __future__ import annotations

from typing import Any


def stamp_origin_discriminators(t: Any, route_metadata: dict, media_metadata: dict) -> None:
    """Copy the origin's tenant discriminators onto a live send's text and media metadata.

    Relay egress is fail-closed on a discriminator in metadata: ``scope_id`` for a scoped chat (Slack
    workspace, Discord guild) and ``user_id`` (the recipient author) for a guild-less DM (Telegram,
    WhatsApp, Matrix, Signal), which has no route row. The RelayAdapter learns both from inbound
    events, so its caches are COLD after every process start, and a backend that stops the guest on
    sleep restarts the gateway on every wake. The router stamps only the HOME channel, so the
    persisted origin is the source for everything else.

    Origin targets only: ``origin_user_id`` is None for a fan-out target, whose tenant and recipient
    are not the origin's, and a wrong discriminator is worse than none. ``setdefault`` never overrides
    router or home stamping. ``user_id`` goes to relay transports only; native adapters never read it.
    """
    discriminators = (
        ("scope_id", t.origin.get("scope_id") if t.origin_target else None),
        ("user_id", t.origin_user_id if t.is_relay else None),
    )
    for key, value in discriminators:
        if value:
            route_metadata.setdefault(key, str(value))
            media_metadata.setdefault(key, str(value))
