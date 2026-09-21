"""Rubika platform adapter: polling-based connection to Rubika's Bot API,
relaying messages between Rubika chats and the Hermes agent."""

import asyncio
import logging
from typing import Any, Dict, Optional, Set

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms._shared import extra_or_secret as _extra_or_secret

from plugins.platforms.rubika.client import RubikaClient, RubikaAPIError
from plugins.platforms.rubika.inbound import parse_update, parse_inline_message

logger = logging.getLogger(__name__)

POLL_LIMIT = 100
POLL_ERROR_BACKOFF_SECONDS = 5
POLL_INTERVAL_SECONDS = 1  # conservative default; no documented Rubika rate limit to tune against yet


def _token(extra: Optional[dict]) -> str:
    return _extra_or_secret(extra, "token", "RUBIKA_BOT_TOKEN", "")


class RubikaAdapter(BasePlatformAdapter):
    """Polling adapter for Rubika's Bot API."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("rubika"))
        extra = config.extra or {}
        self._client = RubikaClient(token=_token(extra))
        self._offset_id: Optional[str] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._allowed_users: Set[str] = {
            item.strip().lower()
            for item in _extra_or_secret(extra, "allowed_users", "RUBIKA_ALLOWED_USERS", "").split(",")
            if item.strip()
        }

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not _token(self.config.extra or {}):
            logger.warning("[%s] RUBIKA_BOT_TOKEN not set", self.name)
            return False
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        self._mark_connected()
        logger.info("[%s] Connected (polling mode)", self.name)
        return True

    async def disconnect(self) -> None:
        self._running = False
        self._mark_disconnected()
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await asyncio.wait_for(self._poll_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._poll_task = None
        logger.info("[%s] Disconnected", self.name)

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                params: Dict[str, Any] = {"limit": POLL_LIMIT}
                if self._offset_id:
                    params["offset_id"] = self._offset_id
                result = await self._client.call("getUpdates", **params)
            except RubikaAPIError as exc:
                logger.warning("[%s] getUpdates failed: %s", self.name, exc)
                await asyncio.sleep(POLL_ERROR_BACKOFF_SECONDS)
                continue
            for update in result.get("updates", []):
                try:
                    await self._dispatch_update(update)
                except Exception as exc:
                    # A bug in parse_update/parse_inline_message, build_source, or handle_message
                    # must not kill the whole poll task: left uncaught it propagates out of
                    # _poll_loop, silently leaving self._running True with no more messages ever
                    # delivered and no automatic recovery. Broad on purpose: RubikaAPIError alone
                    # wouldn't cover a handle_message failure.
                    logger.exception("[%s] Failed to process update, skipping: %s", self.name, exc)
            next_offset = result.get("next_offset_id")
            if next_offset:
                self._offset_id = next_offset
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def _is_user_allowed(self, sender_id: str) -> bool:
        if not self._allowed_users or "*" in self._allowed_users:
            return True
        return sender_id.lower() in self._allowed_users

    async def _dispatch_update(self, update: Dict[str, Any]) -> None:
        update_type = update.get("type")
        if update_type == "NewMessage":
            parsed = parse_update(update)
        elif update_type == "InlineMessage" or "aux_data" in update and "chat_type" not in update:
            parsed = parse_inline_message(update)
        else:
            return logger.debug("[%s] Ignoring unknown update type: %s", self.name, update_type)
        if not self._is_user_allowed(parsed.sender_id):
            return logger.debug("[%s] Dropping message from non-allowlisted sender %s",
                                self.name, parsed.sender_id)
        if not parsed.text:
            return logger.debug("[%s] Empty message, skipping", self.name)
        source = self.build_source(
            chat_id=parsed.chat_id, chat_type="group" if parsed.is_group else "dm",
            user_id=parsed.sender_id, message_id=parsed.message_id)
        await self.handle_message(MessageEvent(
            text=parsed.text, message_type=MessageType.TEXT, source=source,
            message_id=parsed.message_id, raw_message=update,
            reply_to_message_id=parsed.reply_to_message_id))

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        raise NotImplementedError  # Task 7

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        raise NotImplementedError  # Task 10


def register(*args, **kwargs) -> None:
    """No-op until Task 12 wires plugin registration (platform_registry entry, config schema)."""
    pass
