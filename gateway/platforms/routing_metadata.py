"""Platform-aware thread/reply routing metadata helpers.

Extracted from ``gateway/platforms/base.py`` (god-file decomposition
campaign, wave 1 — shard s1, cluster c1, 12 move votes). Functions moved
verbatim; ``base.py`` re-exports them. ``_platform_name`` stays in
``base.py`` (the audio-delivery helpers that remain there also use it) and
is imported lazily inside the two functions that need it — the same
cycle-avoidance pattern documented in ``gateway/authz_mixin.py`` — so this
module never imports ``gateway.platforms.base`` at import time.
"""

def _thread_metadata_for_source(source, reply_to_message_id: str | None = None) -> dict | None:
    """Build platform-aware thread metadata for adapter sends.

    Most platforms route threaded sends with a generic ``thread_id`` metadata
    value. Telegram private-chat topics created through Hermes' DM-topic helper
    are exposed in updates as ``message_thread_id`` plus a reply anchor. Live
    user-message replies route with ``message_thread_id`` + ``reply_to_message_id``;
    synthetic/resumed sends that have no reply anchor fall back to Telegram's
    ``direct_messages_topic_id`` when the Bot API supports it.
    """
    from gateway.platforms.base import _platform_name
    thread_id = getattr(source, "thread_id", None)
    metadata = {"thread_id": thread_id} if thread_id is not None else {}
    # Slack workspace identity is durable routing state, not ephemeral event
    # metadata. Carry it on every outbound path (including unthreaded sends)
    # so a multi-workspace Socket Mode gateway never falls back to its primary
    # WebClient after an async, stream, or recovery boundary.
    if _platform_name(getattr(source, "platform", None)) == "slack":
        scope_id = getattr(source, "scope_id", None)
        if scope_id:
            metadata["slack_team_id"] = str(scope_id)
    if not metadata:
        return None
    if _platform_name(getattr(source, "platform", None)) == "telegram" and getattr(source, "chat_type", None) == "dm":
        metadata["telegram_dm_topic_reply_fallback"] = True
        tid = str(thread_id)
        if tid and tid not in {"", "1"}:
            metadata["direct_messages_topic_id"] = tid
        anchor = reply_to_message_id or getattr(source, "message_id", None)
        if anchor is not None:
            metadata["telegram_reply_to_message_id"] = str(anchor)
    return metadata


def _mark_notify_metadata(metadata: dict | None) -> dict:
    """Clone metadata and mark a user-visible reply as notify-worthy."""
    notify_metadata = dict(metadata) if metadata else {}
    notify_metadata["notify"] = True
    return notify_metadata


def _reply_anchor_for_event(event) -> str | None:
    """Return reply_to id for platforms that need reply semantics.

    Telegram forum/supergroup topics should be routed by topic metadata, not by
    replying to the triggering message. Hermes-created Telegram private-chat
    topic lanes prefer replying to the triggering user message so the answer
    stays attached to the active lane; synthetic/resumed sends fall back to
    ``direct_messages_topic_id`` metadata when no message id is available.
    """
    from gateway.platforms.base import _platform_name
    source = getattr(event, "source", None)
    platform = _platform_name(getattr(source, "platform", None))
    thread_id = getattr(source, "thread_id", None)
    raw_message = getattr(event, "raw_message", None)
    if (
        platform == "slack"
        and isinstance(raw_message, dict)
        and raw_message.get("_hermes_no_thread_response")
    ):
        # Slack reaction handoffs into a configured target channel are meant
        # to create a new top-level message there. Returning the synthetic
        # event's message_id as reply_to would make
        # SlackAdapter._resolve_thread_ts() treat it as a thread anchor and
        # reply in a (nonexistent) thread anyway.
        return None
    if platform == "telegram" and thread_id and getattr(source, "chat_type", None) == "dm":
        # Reply to the triggering user message. Replying to Telegram's earlier
        # topic seed/anchor can render the bot response outside the active lane.
        return getattr(event, "message_id", None) or getattr(event, "reply_to_message_id", None)
    if platform == "telegram" and thread_id:
        return None
    if platform == "feishu" and thread_id and getattr(event, "reply_to_message_id", None):
        return getattr(event, "reply_to_message_id", None)
    return getattr(event, "message_id", None)
