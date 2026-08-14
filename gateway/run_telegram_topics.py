"""Telegram topic service updates consumed before gateway turn admission."""

import logging

from gateway.config import Platform
from gateway.platforms.event import MessageEvent

logger = logging.getLogger("gateway.run")


class GatewayTelegramTopicsMixin:
    async def _handle_telegram_forum_topic_name(self, event: MessageEvent) -> bool:
        """Persist an authorized Telegram topic create/rename as the canonical session title."""
        metadata = getattr(event, "metadata", None) or {}
        name = metadata.get("telegram_forum_topic_name")
        source = getattr(event, "source", None)
        if (
            not isinstance(name, str)
            or not name.strip()
            or source is None
            or source.platform != Platform.TELEGRAM
            or not source.thread_id
        ):
            return False

        try:
            session_entry = await self.async_session_store.get_or_create_session(
                source, touch_activity=False
            )
            if self._session_db is None:
                logger.warning(
                    "Could not store Telegram forum topic name: session database unavailable"
                )
            else:
                await self._session_db.set_session_title(session_entry.session_id, name)
        except ValueError as exc:
            logger.warning("Could not store Telegram forum topic name: %s", exc)
        except Exception:
            logger.warning("Could not store Telegram forum topic name", exc_info=True)
        return True
