"""Optional native reply metadata supplied by platform plugins."""


def plugin_reply_metadata(source, anchor):
    from gateway.platform_registry import platform_registry
    platform = getattr(source.platform, "value", source.platform)
    entry = platform_registry.get(platform)
    callback = entry.reply_metadata_for_source_fn if entry else None
    return callback(source, anchor) if callback else {}


def absorb_reply_anchor(existing, incoming):
    """Keep an explicit native anchor paired with the sender that owns it."""
    anchor = incoming.reply_anchor_override
    if anchor is None:
        return
    a, b = existing.source, incoming.source
    if (a.user_id, a.user_id_alt) != (b.user_id, b.user_id_alt):
        return
    existing.reply_anchor_override = anchor
    a.message_id = b.message_id
    existing.raw_message = incoming.raw_message
    existing.message_id = incoming.message_id


def _thread_metadata_for_event(event) -> dict | None:
    """``_thread_metadata_for_source`` for an event, anchored on its reply id."""
    from gateway.platforms.base import _thread_metadata_for_source, _reply_anchor_for_event
    return _thread_metadata_for_source(event.source, _reply_anchor_for_event(event))


def _mark_notify_metadata(metadata: dict | None) -> dict:
    """Clone metadata and mark a user-visible reply as notify-worthy."""
    notify_metadata = dict(metadata) if metadata else {}
    notify_metadata["notify"] = True
    return notify_metadata


def runner_reply_metadata(runner, source, reply_to_message_id=None):
    """Build the metadata dict platforms need for thread-aware replies."""
    metadata = runner._thread_metadata_for_target(
        getattr(source, "platform", None), getattr(source, "chat_id", None),
        getattr(source, "thread_id", None), chat_type=getattr(source, "chat_type", None),
        reply_to_message_id=reply_to_message_id or getattr(source, "message_id", None))
    from gateway.config import Platform
    if getattr(source, "platform", None) == Platform.SLACK:
        # Per-turn egress identity: Slack chat.startStream needs recipient_user_id/team_id; the relay
        # adapter's _with_scope fallback reads per-chat caches a CONCURRENT turn overwrites.
        # Slack's chat.startStream requires recipient_user_id (+ recipient_team_id) when streaming to a
        # channel, and the relay connector fills those from metadata.user_id / metadata.scope_id. The
        # relay adapter's _with_scope fallback resolves BOTH from per-chat caches keyed only by chat_id
        # — mutable state that a CONCURRENT turn overwrites: two users with overlapping turns in one
        # channel would open U1's stream with U2 as the recipient. Stamp the authentic per-turn values
        # from THIS turn's source here, where they are still turn-scoped; _with_scope only fills keys
        # that are absent, so the cache degrades to what it should be — a restart/synthetic-send
        # fallback. See #210.
        team_id = getattr(source, "scope_id", None)
        user_id = getattr(source, "user_id", None)
        if team_id or user_id:
            metadata = dict(metadata or {})
            if team_id:
                metadata["slack_team_id"] = str(team_id)
                metadata.setdefault("scope_id", str(team_id))
            if user_id:
                metadata.setdefault("user_id", str(user_id))
    metadata = {**(metadata or {}), **plugin_reply_metadata(source, reply_to_message_id)} or None
    from gateway.session_context import source_route_metadata
    metadata = source_route_metadata(source, metadata)
    # Routed profile for shared state.db namespaces: under profile_routes the transport adapter's
    # stamp is not the profile that wrote the binding (Telegram prune path needs it).
    # See #76423.
    profile = str(getattr(source, "profile", None) or "").strip()
    if profile and metadata is not None:
        metadata = dict(metadata)
        metadata["hermes_profile"] = profile
    return metadata
