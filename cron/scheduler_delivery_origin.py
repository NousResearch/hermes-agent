"""What a cron fire reads from its persisted job origin: the DM shape and relay egress discriminators."""

from __future__ import annotations

from typing import Any


def target_is_dm(origin: dict, platform_name: str, chat_id) -> bool:
    """Whether a delivery target is a DM, for the seed's chat_type and the in_channel gate. The
    origin's chat_type decides when there is one; an origin-less job (a manifest-defined or
    explicitly targeted one) reads the id: Slack DM ids start with "D", and a Telegram private
    chat's id is the user's, always positive (groups and channels are negative). A wrong answer
    seeds a ``group`` row a DM-topic reply never resolves to."""
    from gateway.delivery import looks_like_telegram_private_chat_id
    origin_chat_type = str((origin or {}).get("chat_type") or "").lower()
    if origin_chat_type:
        return origin_chat_type == "dm"
    if str(chat_id).startswith("D"):
        return True
    return (str(platform_name).lower() == "telegram"
            and looks_like_telegram_private_chat_id(str(chat_id)))


def stamp_origin_discriminators(t: Any, route_metadata: dict, media_metadata: dict) -> None:
    """Stamp the origin's ``scope_id`` / ``user_id`` onto a live send's metadata.

    Relay egress is fail-closed on a discriminator and the RelayAdapter's caches are cold after every
    boot, so the persisted origin supplies them. Origin targets only (a fan-out target's recipient is not
    the origin's author); ``setdefault`` never overrides router or home stamping; ``user_id`` is read by
    relay transports only.
    """
    discriminators = (
        ("scope_id", t.origin.get("scope_id") if t.origin_target else None),
        ("user_id", t.origin_user_id if t.is_relay else None),
    )
    for key, value in discriminators:
        if value:
            route_metadata.setdefault(key, str(value))
            media_metadata.setdefault(key, str(value))
