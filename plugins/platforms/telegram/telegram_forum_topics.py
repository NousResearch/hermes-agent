"""Authorized forum-topic service updates for Telegram session titles."""

from typing import Any

from gateway.platforms.event import MessageType


class TelegramForumTopicsMixin:
    def _register_forum_topic_name_handler(self, app, handler_type, status_filters) -> None:
        app.add_handler(handler_type(
            status_filters.FORUM_TOPIC_EDITED,
            self._handle_forum_topic_name_update,
        ))

    async def _handle_forum_topic_created_with_name(self, update: Any, context: Any) -> None:
        try:
            await self._handle_forum_topic_name_update(update, context)
        finally:
            # Building metadata reads the explicit thread and clears a pending opener; record it last.
            await self._handle_forum_topic_created(update, context)

    async def _handle_forum_topic_name_update(self, update: Any, context: Any) -> None:
        """Forward an authorized human's forum topic create/rename metadata."""
        msg = self._effective_update_message(update)
        actor = getattr(msg, "from_user", None) if msg is not None else None
        if (
            msg is None
            or actor is None
            or bool(getattr(actor, "is_bot", False))
            or not self._is_user_authorized_from_message(msg)
        ):
            return

        topic = getattr(msg, "forum_topic_edited", None) or getattr(
            msg, "forum_topic_created", None
        )
        name = getattr(topic, "name", None)
        if not isinstance(name, str) or not name.strip():
            return

        event = self._build_message_event(
            msg, MessageType.TEXT, update_id=getattr(update, "update_id", None)
        )
        event.text = ""
        event.allow_gateway_control = False
        event.source.chat_topic = name
        event.metadata["telegram_forum_topic_name"] = name
        if self._drop_unresolved(event) or self._message_handler is None:
            return
        # Metadata must not join the active turn's pending input or interrupt its owner.
        await self._dispatch_inline_reply(event)
