"""Native DM-topic routing metadata for ``TelegramAdapter``.

Telegram bots whose DM carries topics ("Threads") are reached with ``direct_messages_topic_id``;
a forum-style ``message_thread_id`` send is rejected there ("message thread not found"). Hermes
private-chat topic lanes (``telegram_dm_topic_reply_fallback``) add a reply anchor on top of the
lane id, and anchor-less synthetic sends must keep preferring the Hermes topic's
``message_thread_id`` (#87051). These accessors and guards read that routing state off send
metadata for the adapter's send-path resolvers. Split out of ``adapter.py`` to keep the facade
within its code-health line cap.
"""

from __future__ import annotations

from typing import Any, Optional


class TelegramDMTopicsMixin:
    """Send-path metadata accessors/guards for native DM topics and Hermes topic lanes."""

    @classmethod
    def _metadata_thread_id(cls, metadata: Optional[dict[str, Any]]) -> Optional[str]:
        thread_id = (metadata or {}).get("thread_id") or (metadata or {}).get("message_thread_id")
        return str(thread_id) if thread_id is not None else None

    @classmethod
    def _metadata_direct_messages_topic_id(cls, metadata: Optional[dict[str, Any]]) -> Optional[str]:
        topic_id = (metadata or {}).get("direct_messages_topic_id") or (metadata or {}).get("telegram_direct_messages_topic_id")
        return str(topic_id) if topic_id is not None else None

    @classmethod
    def _metadata_reply_to_message_id(cls, metadata: Optional[dict[str, Any]]) -> Optional[int]:
        reply_to = (metadata or {}).get("telegram_reply_to_message_id")
        return int(reply_to) if reply_to is not None else None

    @staticmethod
    def _dm_topic_fallback(metadata: Optional[dict[str, Any]]) -> bool:
        """True for Hermes private-chat topic lanes (``telegram_dm_topic_reply_fallback``)."""
        return bool(metadata and metadata.get("telegram_dm_topic_reply_fallback"))

    @classmethod
    def _is_private_dm_topic_send(cls, chat_id: str, thread_id: Optional[str], metadata: Optional[dict[str, Any]]) -> bool:
        if cls._metadata_direct_messages_topic_id(metadata) is not None:
            return cls._dm_topic_fallback(metadata) and cls._metadata_reply_to_message_id(metadata) is not None
        if metadata and metadata.get("telegram_dm_topic_created_for_send"):
            return False
        return bool(thread_id) and cls._dm_topic_fallback(metadata)

    @staticmethod
    def _dm_topic_missing_anchor_error() -> str:
        return "Telegram DM topic delivery requires a reply anchor; refusing to send outside the requested topic"

    @classmethod
    def _reply_to_message_id_for_send(
        cls, reply_to: Optional[str], metadata: Optional[dict[str, Any]] = None, reply_to_mode: Optional[str] = None) -> Optional[int]:
        if reply_to:
            return int(reply_to)
        if cls._dm_topic_fallback(metadata) and reply_to_mode != "off":
            return cls._metadata_reply_to_message_id(metadata)
        return None

    @classmethod
    def _direct_topic_kwargs(cls, metadata: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Native Bot API DM-topic routing kwargs, or None when no ``direct_messages_topic_id``."""
        direct_topic_id = cls._metadata_direct_messages_topic_id(metadata)
        if direct_topic_id is None:
            return None
        return {"message_thread_id": None, "direct_messages_topic_id": int(direct_topic_id)}
