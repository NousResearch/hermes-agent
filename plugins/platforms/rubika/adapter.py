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
        _require_mention_raw = _extra_or_secret(extra, "require_mention", "RUBIKA_REQUIRE_MENTION", "false")
        self._require_mention = (
            _require_mention_raw.lower() in ("true", "1", "yes", "on")
            if isinstance(_require_mention_raw, str) else bool(_require_mention_raw))
        self._bot_username = _extra_or_secret(extra, "bot_username", "RUBIKA_BOT_USERNAME", "")

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

    def _should_process_message(self, parsed: "ParsedMessage") -> bool:
        """DMs always pass. Group messages pass when require_mention is off,
        or when the text opens with an @bot_username mention."""
        if not parsed.is_group or not self._require_mention:
            return True
        if not self._bot_username:
            return True  # nothing configured to match against — fail open, same as DingTalk's own default
        return parsed.text.strip().lower().startswith(f"@{self._bot_username.lower()}")

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
        if not self._should_process_message(parsed):
            return logger.debug("[%s] Dropping group message that failed mention gate: chat_id=%s",
                                self.name, parsed.chat_id)
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
        from plugins.platforms.rubika.keypad import build_chat_keypad, build_inline_keypad
        metadata = metadata or {}
        params: Dict[str, Any] = {"chat_id": chat_id, "text": content}
        if reply_to:
            params["reply_to_message_id"] = reply_to
        if chat_keypad := metadata.get("chat_keypad"):
            params["chat_keypad"] = build_chat_keypad(chat_keypad)
            params["chat_keypad_type"] = "New"
        if inline_keypad := metadata.get("inline_keypad"):
            params["inline_keypad"] = build_inline_keypad(inline_keypad)
        try:
            data = await self._client.call("sendMessage", **params)
            return SendResult(success=True, message_id=str(data.get("message_id") or ""))
        except RubikaAPIError as exc:
            logger.warning("[%s] send() failed: %s", self.name, exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None,
                              reply_to: Optional[str] = None, metadata=None, **kwargs) -> SendResult:
        return await self._send_uploaded_file(
            chat_id, image_path, file_type="Image", caption=caption, reply_to=reply_to)

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            file_name: Optional[str] = None, reply_to=None, metadata=None,
                            **kwargs) -> SendResult:
        return await self._send_uploaded_file(
            chat_id, file_path, file_type="File", caption=caption, reply_to=reply_to)

    async def _send_uploaded_file(self, chat_id: str, file_path: str, *, file_type: str,
                                  caption: Optional[str], reply_to: Optional[str]) -> SendResult:
        try:
            file_id = await self._client.upload_file(file_path, file_type=file_type)
            params: Dict[str, Any] = {"chat_id": chat_id, "file_id": file_id, "text": caption or ""}
            if reply_to:
                params["reply_to_message_id"] = reply_to
            data = await self._client.call("sendFile", **params)
            return SendResult(success=True, message_id=str(data.get("message_id") or ""))
        except RubikaAPIError as exc:
            logger.warning("[%s] media send failed: %s", self.name, exc)
            return SendResult(success=False, error=str(exc), retryable=True)
        except OSError as exc:
            return SendResult(success=False, error=f"Could not read file: {exc}")

    async def edit_message(self, chat_id: str, message_id: str, content: str,
                           *, finalize: bool = False) -> SendResult:
        try:
            await self._client.call("editMessageText", chat_id=chat_id, message_id=message_id, text=content)
            return SendResult(success=True, message_id=message_id)
        except RubikaAPIError as exc:
            logger.warning("[%s] edit_message failed: %s", self.name, exc)
            return SendResult(success=False, error=str(exc))

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        try:
            await self._client.call("deleteMessage", chat_id=chat_id, message_id=message_id)
            return True
        except RubikaAPIError as exc:
            logger.debug("[%s] delete_message failed: %s", self.name, exc)
            return False

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        try:
            data = await self._client.call("getChat", chat_id=chat_id)
        except RubikaAPIError as exc:
            logger.warning("[%s] get_chat_info failed: %s", self.name, exc)
            return {"name": chat_id, "type": "dm"}
        chat = data.get("chat") or {}
        chat_type = str(chat.get("chat_type") or "").lower()
        is_group = chat_type == "group"
        name = (chat.get("title") or chat_id) if is_group else (chat.get("first_name") or chat_id)
        return {"name": name, "type": "group" if is_group else "dm"}


def register(*args, **kwargs) -> None:
    """No-op until Task 12 wires plugin registration (platform_registry entry, config schema)."""
    pass
