"""Thread-aware target and source metadata for GatewayRunner."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from gateway.config import Platform

logger = logging.getLogger("gateway.run")


class GatewayThreadMetadataMixin:
    def _thread_metadata_for_source(
        self, source, reply_to_message_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Build the metadata dict platforms need for thread-aware replies."""
        metadata = self._thread_metadata_for_target(
            getattr(source, "platform", None), getattr(source, "chat_id", None),
            getattr(source, "thread_id", None), chat_type=getattr(source, "chat_type", None),
            reply_to_message_id=reply_to_message_id or getattr(source, "message_id", None))
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

    def _thread_metadata_for_target(
        self, platform: Optional[Platform], chat_id: Optional[str], thread_id: Optional[str], *,
        chat_type: Optional[str] = None, reply_to_message_id: Optional[str] = None,
        adapter: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Build thread metadata for synthetic sends that only have routing state."""
        if thread_id is None:
            return None
        metadata: Dict[str, Any] = {"thread_id": thread_id}
        if self._is_telegram_dm_topic_target(
            platform, chat_id, thread_id, chat_type=chat_type, adapter=adapter):
            metadata["telegram_dm_topic_reply_fallback"] = True
            # DM topic lanes need direct_messages_topic_id so synthetic sends reach the topic without a reply anchor.
            tid = str(thread_id)
            if tid and tid not in {"", "1"}:
                metadata["direct_messages_topic_id"] = tid
            if reply_to_message_id is not None:
                metadata["telegram_reply_to_message_id"] = str(reply_to_message_id)
        if platform == Platform.SLACK and reply_to_message_id is not None:
            # Slack's reply_in_thread=false path uses message_id to tell real threads from synthetic keys.
            metadata["message_id"] = str(reply_to_message_id)
        if platform == Platform.FEISHU and thread_id and reply_to_message_id is not None:
            metadata["reply_to_message_id"] = str(reply_to_message_id)
        return metadata

    @staticmethod
    def _is_telegram_dm_topic_target(
        platform: Optional[Platform], chat_id: Optional[str], thread_id: Optional[str], *,
        chat_type: Optional[str] = None, adapter: Optional[Any] = None) -> bool:
        """Return True when a target is a Telegram private DM topic lane."""
        if platform != Platform.TELEGRAM or thread_id is None:
            return False
        if chat_type == "dm":
            return True
        # Resolve the lookup on the CLASS, not the instance: getattr() on a MagicMock auto-creates callable
        # children, so an instance lookup would report a DM topic for every test double. Only a dict counts.
        if adapter is not None and chat_id:
            get_dm_topic_info = getattr(type(adapter), "_get_dm_topic_info", None)
            if callable(get_dm_topic_info):
                try:
                    topic_info = get_dm_topic_info(adapter, str(chat_id), str(thread_id))
                except Exception:
                    logger.debug("Failed to inspect Telegram DM topic metadata", exc_info=True)
                else:
                    return isinstance(topic_info, dict)
        return False
