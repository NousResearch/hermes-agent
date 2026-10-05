"""Opt-in delegated Telegram Business inbox policy and event routing."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from gateway.platforms.base import SendResult

from gateway.platforms.event import MessageType

logger = logging.getLogger(__name__)


def normalize_business_config(raw: Any) -> Dict[str, Any]:
    """Normalize opt-in Telegram Business routing configuration."""
    raw = raw if isinstance(raw, dict) else {}

    def _items(value: Any) -> List[str]:
        values = value if isinstance(value, (list, tuple, set)) else [value]
        return [
            normalized
            for item in values
            if item is not None and (normalized := str(item).strip())
        ]

    enabled = raw.get("enabled", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
    else:
        enabled = bool(enabled)
    allow_send_as_account = raw.get("allow_business_send_as_account", False)
    if isinstance(allow_send_as_account, str):
        allow_send_as_account = allow_send_as_account.strip().lower() in {
            "1", "true", "yes", "on",
        }
    else:
        allow_send_as_account = bool(allow_send_as_account)
    return {
        "enabled": enabled,
        "allow_business_send_as_account": allow_send_as_account,
        "allowed_chats": _items(raw.get("allowed_chats", [])),
        "allowed_owner_ids": _items(
            raw.get("allowed_owner_ids", raw.get("allow_from", []))
        ),
        "allowed_connection_ids": _items(raw.get("allowed_connection_ids", [])),
        "trigger_words": _items(raw.get("trigger_words", [])),
    }


class TelegramBusinessMixin:
    @staticmethod
    def _is_business_delivery(metadata) -> bool:
        return bool(metadata and (
            "business_connection_id" in metadata or metadata.get("_delivery_route_blocked")
        ))

    @staticmethod
    def _business_kwargs(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Never erase a requested account identity on a fallback path."""
        metadata = metadata or {}
        connection_id = str(metadata.get("business_connection_id") or "").strip()
        if metadata.get("_delivery_route_blocked") or (
            "business_connection_id" in metadata
            and (metadata.get("allow_business_send_as_account") is not True or not connection_id)
        ):
            raise ValueError("business_authority_unavailable")
        return {"business_connection_id": connection_id} if connection_id else {}

    async def _business_delivery_error(self: Any, chat_id, metadata) -> Optional[SendResult]:
        """Revalidate the immediate event's transport, destination and live policy."""
        if not self._is_business_delivery(metadata):
            return None
        source = (metadata or {}).get("_telegram_business_source")
        connection_id = str((metadata or {}).get("business_connection_id") or "").strip()
        owner_id = getattr(source, "telegram_business_owner_id", None)
        transport_ref = getattr(source, "_transport_adapter_ref", None)
        cfg = self._business_config()
        allowed = (
            not (metadata or {}).get("_delivery_route_blocked")
            and (metadata or {}).get("allow_business_send_as_account") is True
            and getattr(source, "authorized_via_telegram_business", False) is True
            and getattr(source, "scope_id", None) == f"telegram-business:{connection_id}"
            and str(getattr(source, "chat_id", "")) == str(chat_id)
            and callable(transport_ref) and transport_ref() is self
            and cfg["enabled"] and cfg["allow_business_send_as_account"]
            and self._business_scope_allowed(owner_id, connection_id)
            and (not cfg["allowed_chats"] or str(chat_id) in cfg["allowed_chats"])
        )
        if allowed:
            allowed = await self._business_connection_owner_id(connection_id) == owner_id
        if not allowed:
            return SendResult(success=False, error="business_authority_unavailable", retryable=False)
        return None

    async def _require_business_delivery(self: Any, chat_id, metadata) -> None:
        error = await self._business_delivery_error(chat_id, metadata)
        if error is not None:
            raise ValueError(error.error)

    def _business_config(self: Any) -> Dict[str, Any]:
        extra = getattr(self.config, "extra", None)
        raw = extra.get("business", {}) if isinstance(extra, dict) else {}
        return normalize_business_config(raw)

    def _business_scope_allowed(self: Any, owner_id: Any, connection_id: Any) -> bool:
        """Require an explicit owner/connection allowlist for delegated inbox authority."""
        config = self._business_config()
        owner_id = str(owner_id or "").strip()
        connection_id = str(connection_id or "").strip()
        allowed_owners = set(config.get("allowed_owner_ids") or ())
        allowed_connections = set(config.get("allowed_connection_ids") or ())
        if not owner_id or not connection_id or not (allowed_owners or allowed_connections):
            return False
        if allowed_owners and owner_id not in allowed_owners:
            return False
        if allowed_connections and connection_id not in allowed_connections:
            return False
        return True

    def _business_trigger_text(self: Any, message: Any) -> Optional[str]:
        """Return customer text after an explicit configured trigger."""
        text = (
            getattr(message, "text", None)
            or getattr(message, "caption", None)
            or ""
        ).strip()
        if not text:
            return None
        for word in self._business_config()["trigger_words"]:
            match = re.match(
                rf"^{re.escape(word)}(?=$|[\s,.:;!?—–-])"
                rf"[\s,.:;!?—–-]*(.*)$",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                return match.group(1).strip()
        return None

    async def _business_connection_owner_id(
        self: Any, connection_id: str
    ) -> Optional[str]:
        """Resolve the connected account owner from Telegram, failing closed."""
        bot = getattr(self, "_bot", None)
        get_connection = getattr(bot, "get_business_connection", None)
        if not callable(get_connection):
            return None
        try:
            connection = await get_connection(connection_id)
        except Exception as exc:
            logger.warning(
                "[%s] Could not verify Telegram Business connection (%s); dropping delegated inbox update",
                self.name, type(exc).__name__,
            )
            return None
        if (getattr(connection, "is_enabled", False) is not True
                or getattr(getattr(connection, "rights", None), "can_reply", False) is not True):
            return None
        resolved_id = str(getattr(connection, "id", "") or "").strip()
        owner = getattr(getattr(connection, "user", None), "id", None)
        if resolved_id != connection_id or owner is None or isinstance(owner, bool):
            return None
        return str(owner)

    async def _handle_business_message(
        self: Any,
        update: Any,
        context: Any,
    ) -> None:
        """Route explicitly-triggered Telegram Business customer messages.

        Business traffic is disabled by default, isolated from ordinary bot
        DMs by connection scope, denied gateway-control authority, and allowed
        to send as the connected account only on the immediate trusted reply.
        """
        del context
        cfg = self._business_config()
        if not cfg["enabled"] or not cfg["allow_business_send_as_account"] or not cfg["trigger_words"]:
            return
        message = getattr(update, "business_message", None)
        connection_id = str(
            getattr(message, "business_connection_id", "") or ""
        ).strip()
        if message is None or not connection_id:
            return
        actor = getattr(message, "from_user", None)
        sender_business_bot = getattr(message, "sender_business_bot", None)
        if (
            actor is None
            or bool(getattr(actor, "is_bot", False))
            or sender_business_bot is not None
        ):
            # Bot-authored Business updates are delivery echoes, never new
            # customer input. Re-enqueueing them creates a reply loop.
            return
        chat_id = str(getattr(getattr(message, "chat", None), "id", "") or "")
        if (getattr(message.chat, "type", None) != "private" or not chat_id
                or (cfg["allowed_chats"] and chat_id not in cfg["allowed_chats"])):
            return
        text = self._business_trigger_text(message)
        has_media = any(getattr(message, kind, None) for kind in (
            "photo", "video", "audio", "voice", "document", "sticker"))
        # Only an already-triggered pending album may contribute captionless items.
        # No sticky "awake" state: a later standalone upload needs its own caption.
        captionless_album = (has_media and getattr(message, "media_group_id", None)
                             and not (getattr(message, "text", None) or getattr(message, "caption", None)))
        if text is None and not captionless_album:
            return
        actor_id = getattr(actor, "id", None)
        owner_id = await self._business_connection_owner_id(connection_id)
        if (
            owner_id is None
            or actor_id is None
            or isinstance(actor_id, bool)
            or str(actor_id) == owner_id
            or not self._business_scope_allowed(owner_id, connection_id)
        ):
            # Business owners can write through the connected account too.
            # Treat their updates as operator traffic, not delegated-customer
            # prompts; an unverifiable connection likewise fails closed.
            return

        event = self._build_message_event(
            message,
            self._media_message_type(message) if has_media else MessageType.TEXT,
            update_id=getattr(update, "update_id", None),
        )
        event.text = text or ""
        event.source.scope_id = f"telegram-business:{connection_id}"
        event.source.chat_type = "dm"
        event.source.authorized_via_telegram_business = True
        event.source.telegram_business_owner_id = owner_id
        event.metadata = dict(event.metadata or {})
        event.metadata.update(
            {
                "allow_business_send_as_account": cfg[
                    "allow_business_send_as_account"
                ],
                "business_connection_id": connection_id,
            }
        )
        # External customers can converse, but cannot invoke gateway lifecycle
        # or administrative controls through the delegated inbox.
        event.allow_gateway_control = False
        if text is None:
            if self._drop_unresolved(event):
                return
            key = self._media_group_key(str(message.media_group_id), event)
            pending = getattr(self, "_media_group_events", {}).get(key)
            if pending is None or pending.source.user_id != event.source.user_id:
                return
            # Bind to this pending trigger, not just its reusable album key. A slow
            # download must not start another turn after the trigger has flushed.
            event._business_album_parent = pending
        if has_media:
            await self._dispatch_authenticated_media(message, event)
        else:
            self._enqueue_text_event(event)
