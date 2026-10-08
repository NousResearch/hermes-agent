"""Thread-aware delivery metadata and reply anchors shared by platform adapters."""

def _platform_name(platform) -> str:
    """Normalize a Platform enum / raw string into a lowercase name."""
    value = getattr(platform, "value", platform)
    return str(value or "").lower()

def _thread_metadata_for_source(source, reply_to_message_id: str | None = None) -> dict | None:
    """Platform-aware thread metadata for adapter sends. Telegram DM topics route with
    ``message_thread_id`` + a reply anchor; anchorless synthetic/resumed sends fall back to
    ``direct_messages_topic_id`` when supported."""
    thread_id = getattr(source, "thread_id", None)
    platform = _platform_name(getattr(source, "platform", None))
    metadata = {"thread_id": thread_id} if thread_id is not None else {}
    # Slack workspace identity is routing state: carry it so a multi-workspace Socket Mode
    # gateway never falls back to its primary WebClient.
    scope_id = getattr(source, "scope_id", None) if platform == "slack" else None
    if scope_id:
        metadata["slack_team_id"] = str(scope_id)
    if not metadata:
        return None
    if platform == "telegram" and getattr(source, "chat_type", None) == "dm":
        metadata["telegram_dm_topic_reply_fallback"] = True
        if str(thread_id) not in {"", "1"}:
            metadata["direct_messages_topic_id"] = str(thread_id)
        anchor = reply_to_message_id or getattr(source, "message_id", None)
        if anchor is not None:
            metadata["telegram_reply_to_message_id"] = str(anchor)
    if platform == "feishu" and thread_id:
        # Feishu topics have no create-message route: metadata-only sends need a real
        # message anchor too (progress, notices, and post-stream attachments).
        anchor = reply_to_message_id or getattr(source, "message_id", None)
        if anchor is not None:
            metadata["reply_to_message_id"] = str(anchor)
    # Routed profile (multiplex / profile_routes): outbound prune paths must not assume the
    # adapter's static profile stamp.
    profile = str(getattr(source, "profile", None) or "").strip()
    if profile:
        metadata["hermes_profile"] = profile
    return metadata

def _thread_metadata_for_event(event) -> dict | None:
    """``_thread_metadata_for_source`` for an event, anchored on its reply id."""
    metadata = _thread_metadata_for_source(event.source, _reply_anchor_for_event(event))
    if (metadata is not None and _platform_name(getattr(event.source, "platform", None)) == "feishu"
            and getattr(event.source, "thread_id", None)):
        # Every send in this turn shares the topic-policy decision, including metadata
        # rebuilt for progress, final text and attachments. Keep it off the serializable
        # event.metadata/source so a later turn in this topic starts with a fresh decision.
        state = getattr(event, "_feishu_topic_delivery", None)
        if not isinstance(state, dict):
            state = event._feishu_topic_delivery = {}
        metadata["_feishu_topic_delivery"] = state
    return metadata

def _mark_notify_metadata(metadata: dict | None) -> dict:
    """Clone metadata and mark a user-visible reply as notify-worthy."""
    notify_metadata = dict(metadata) if metadata else {}
    notify_metadata["notify"] = True
    return notify_metadata

def _reply_anchor_for_event(event) -> str | None:
    """Return reply_to id for platforms that need reply semantics."""
    override = getattr(event, "reply_anchor_override", None)
    if override is not None:
        return override  # the turn was redirected onto another message (#115001)
    source = getattr(event, "source", None)
    platform = _platform_name(getattr(source, "platform", None))
    thread_id = getattr(source, "thread_id", None)
    raw_message = getattr(event, "raw_message", None)
    if (platform == "slack" and isinstance(raw_message, dict)
            and raw_message.get("_hermes_no_thread_response")):
        # Slack reaction handoff = new top-level message; a message_id anchor would make
        # _resolve_thread_ts() reply in a nonexistent thread.
        return None
    if platform == "telegram" and thread_id:
        # Forum topics route by topic metadata (no reply); DM-topic lanes reply to the triggering
        # message — replying to the topic seed/anchor can render outside the active lane.
        if getattr(source, "chat_type", None) != "dm":
            return None
        return getattr(event, "message_id", None) or getattr(event, "reply_to_message_id", None)
    if platform == "feishu" and thread_id:
        return (getattr(event, "reply_to_message_id", None)
                or getattr(event, "message_id", None) or getattr(source, "message_id", None))
    return getattr(event, "message_id", None)
