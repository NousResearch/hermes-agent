"""Passive Telegram group transcript persistence."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from gateway.platforms.event import MessageEvent, MessageType
from telegram import Message

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramObservedImagesMixin:
    def _observe_unmentioned_group_message(
        self, message: Message, msg_type: MessageType, update_id: int | None = None, event: MessageEvent | None = None) -> None:
        """Append skipped group chatter to the target session without dispatching."""
        store = getattr(self, "_session_store", None)
        if not store:
            return
        adapter_name = getattr(self, "name", "telegram")
        try:
            event = event or self._build_message_event(message, msg_type, update_id=update_id)
            session_entry = store.get_or_create_session(self._telegram_group_observe_shared_source(event.source))
            entry = {
                "role": "user", "content": self._telegram_group_observe_attributed_text(event),
                "timestamp": datetime.now(tz=UTC).isoformat(), "observed": True}
            if event.message_id:
                entry["message_id"] = str(event.message_id)
            if event.media_urls:
                from gateway.observed_images import observed_image_content
                entry["content"] = observed_image_content(
                    entry["content"], event.media_urls, event.media_types or [],
                )
            self._accept_update()
            store.append_to_transcript(session_entry.session_id, entry)
            logger.info(
                "[%s] Telegram group message observed (no bot trigger): chat=%s from=%s", adapter_name,
                getattr(getattr(message, "chat", None), "id", "unknown"), event.source.user_id or "unknown")
        except Exception as exc:
            self._fail_update_preparation()
            logger.warning("[%s] Failed to observe Telegram group message: %s", adapter_name, exc, exc_info=True)
