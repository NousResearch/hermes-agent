"""Build one live cron delivery's shared text and media route metadata."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from cron import scheduler_delivery_origin as _origin

if TYPE_CHECKING:
    from cron.scheduler_delivery import _TargetDelivery


def _live_route_metadata(t: _TargetDelivery) -> tuple[Optional[str], dict, dict]:
    """Compute ``(route_thread_id, route_metadata, media_metadata)`` for a live send, ONCE so text
    and media agree. ``telegram:<positive_chat_id>:<numeric_thread_id>`` is ambiguous (private
    forum topic vs channel DM topic need OPPOSITE routing) — see ``_is_channel_dm_topic``.
    ``thread_id`` rides in ``route_metadata`` to bypass the router's private-chat anchor rule."""
    from cron.scheduler_delivery import _is_channel_dm_topic
    from gateway.config import Platform
    from gateway.delivery import _looks_like_int, looks_like_telegram_private_chat_id
    job = t.job
    thread_id = t.thread_id
    is_ambiguous_telegram_topic = (
        t.platform == Platform.TELEGRAM
        and thread_id is not None
        and looks_like_telegram_private_chat_id(str(t.chat_id))
        and _looks_like_int(str(thread_id))
    )
    route_metadata: dict[str, Any]
    media_metadata: dict[str, Any]
    if is_ambiguous_telegram_topic and _is_channel_dm_topic(
        t.runtime_adapter, t.chat_id, t.loop, job["id"]):
        # Channel DM topic: direct_messages_topic_id, no bare thread_id; media mirrors text.
        # See #22773.
        route_thread_id = None
        route_metadata = {
            "direct_messages_topic_id": str(thread_id), "job_id": job["id"],
            "notify": t.notify_delivery,
        }
        media_metadata = {"direct_messages_topic_id": str(thread_id), "notify": t.notify_delivery}
    else:
        # Forum-style topic or non-topic target: message_thread_id.
        # Put thread_id in *route_metadata* (not just the DeliveryTarget) deliberately — the
        # DeliveryRouter's private-chat topic detection (gateway/delivery.py) demands a reply anchor when
        # thread_id is absent from metadata; cron deliveries have no inbound reply anchor, so the metadata
        # key bypasses that check and lets the adapter route via a plain message_thread_id. See #52060.
        route_thread_id = str(thread_id) if thread_id is not None else None
        route_metadata = {"job_id": job["id"], "notify": t.notify_delivery}
        if route_thread_id:
            route_metadata["thread_id"] = route_thread_id
        media_metadata = {"notify": t.notify_delivery}
        if thread_id:
            media_metadata["thread_id"] = thread_id

    # Relay egress discriminators (scope_id / user_id) from the persisted origin: the adapter's caches are cold
    # after a restart. See cron/scheduler_delivery_origin.py.
    _origin.stamp_origin_discriminators(t, route_metadata, media_metadata)
    if t.platform == Platform.FEISHU and thread_id:
        # One cron result includes its text and attachments. A topic decision must survive
        # the router's shallow metadata copy and apply to all of them, without crossing fires.
        state = {}
        route_metadata["_feishu_topic_delivery"] = media_metadata["_feishu_topic_delivery"] = state
        if t.origin_target and t.origin.get("message_id"):
            route_metadata["reply_to_message_id"] = media_metadata["reply_to_message_id"] = str(t.origin["message_id"])
    return route_thread_id, route_metadata, media_metadata
