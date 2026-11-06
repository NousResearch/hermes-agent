"""Relay egress discriminators a cron fire carries from its persisted job origin."""

from __future__ import annotations

from typing import Any


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


def _origin_thread_is_stale(origin: dict) -> bool:
    """True when a Slack origin's thread is a stale creation-turn artifact. Thread-per-message
    Slack stamps each top-level message id as the session thread (a KEY, not a location); old jobs
    carry it as ``origin.thread_id``. Heuristic: if the origin chat IS the Slack home chat, the
    pinned thread is that artifact and delivery goes top-level (or to the home target's thread)."""
    if str(origin.get("platform") or "").lower() != "slack" or not origin.get("thread_id"):
        return False
    from cron.scheduler_delivery import _get_home_target_chat_id
    home_chat = _get_home_target_chat_id("slack")
    return bool(home_chat) and str(origin.get("chat_id")) == str(home_chat)


def origin_delivery_thread(origin: dict):
    """The thread a deliver=origin job should use, stale stamps dropped.

    The parent-channel provenance lives on the origin itself (``parent_chat_id``, #135667);
    the thread here is only the delivery thread id.
    """
    if _origin_thread_is_stale(origin):
        from cron.scheduler_delivery import _get_home_target_thread_id
        return _get_home_target_thread_id("slack") or None
    return origin.get("thread_id")
