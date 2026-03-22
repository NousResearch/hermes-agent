"""
WeChat platform adapter using WeixinClawBot (official OpenClaw integration).

WeChat officially announced OpenClaw bot integration via WeixinClawBot, allowing
bots to receive and send messages through WeChat like a contact.

Requires:
    pip install weixinclawbot httpx
    WECHAT_BOT_TOKEN env var (from WeixinClawBot dashboard)

Configuration in config.yaml:
    platforms:
      wechat:
        enabled: true
        token: "your-weixinclawbot-token"
        home_channel: "openid_of_default_user_or_group"
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

try:
    import weixinclawbot
    from weixinclawbot import ClawBotClient, MessageHandler, IncomingMessage
    WECHAT_AVAILABLE = True
except ImportError:
    WECHAT_AVAILABLE = False
    weixinclawbot = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 2048  # WeChat text message limit
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 1000
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]

# WeixinClawBot API base
WECHAT_API_BASE = "https://api.weixinclawbot.com/v1"


def check_wechat_requirements() -> bool:
    """Check if WeChat dependencies are available and configured."""
    if not HTTPX_AVAILABLE:
        return False
    if not os.getenv("WECHAT_BOT_TOKEN"):
        return False
    return True


class WeChatAdapter(BasePlatformAdapter):
    """WeChat adapter using WeixinClawBot's OpenClaw integration.

    WeixinClawBot provides a long-polling or WebSocket stream to receive
    messages. Outbound messages are sent via the REST API using the bot token.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WECHAT)

        self._token: str = config.token or os.getenv("WECHAT_BOT_TOKEN", "")
        self._api_base: str = os.getenv("WECHAT_API_BASE", WECHAT_API_BASE)

        self._http_client: Optional["httpx.AsyncClient"] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Message deduplication: msg_id -> timestamp
        self._seen_messages: Dict[str, float] = {}

    # -- Connection lifecycle -------------------------------------------------

    async def connect(self) -> bool:
        """Connect to WeChat via WeixinClawBot long-polling."""
        if not HTTPX_AVAILABLE:
            logger.warning("[%s] httpx not installed. Run: pip install httpx", self.name)
            return False
        if not self._token:
            logger.warning("[%s] WECHAT_BOT_TOKEN is required", self.name)
            return False

        try:
            self._http_client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )

            # Verify token by calling /me
            resp = await self._http_client.get(f"{self._api_base}/bot/me")
            if resp.status_code != 200:
                logger.warning(
                    "[%s] Token verification failed: HTTP %d", self.name, resp.status_code
                )
                return False

            bot_info = resp.json()
            logger.info(
                "[%s] Connected as WeChat bot: %s",
                self.name,
                bot_info.get("nickname", "unknown"),
            )

            self._running = True
            self._poll_task = asyncio.create_task(self._poll_loop())
            self._mark_connected()
            return True

        except Exception as e:
            logger.error("[%s] Failed to connect: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        """Disconnect and clean up."""
        self._running = False
        self._mark_disconnected()

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._seen_messages.clear()
        logger.info("[%s] Disconnected", self.name)

    # -- Long-polling loop ----------------------------------------------------

    async def _poll_loop(self) -> None:
        """Poll WeixinClawBot for new messages with reconnection backoff."""
        backoff_idx = 0
        last_msg_id: Optional[str] = None

        while self._running:
            try:
                params = {"limit": 20}
                if last_msg_id:
                    params["after"] = last_msg_id

                resp = await self._http_client.get(
                    f"{self._api_base}/messages",
                    params=params,
                    timeout=30.0,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    messages = data.get("messages", [])
                    for raw in messages:
                        last_msg_id = raw.get("id", last_msg_id)
                        await self._on_message(raw)
                    backoff_idx = 0  # reset on success
                    # Small yield between polls
                    await asyncio.sleep(0.5)

                elif resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.warning("[%s] Rate limited, waiting %ds", self.name, retry_after)
                    await asyncio.sleep(retry_after)

                else:
                    logger.warning("[%s] Poll failed: HTTP %d", self.name, resp.status_code)
                    delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
                    await asyncio.sleep(delay)
                    backoff_idx += 1

            except asyncio.CancelledError:
                return
            except Exception as e:
                if not self._running:
                    return
                delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
                logger.warning("[%s] Poll error: %s — retrying in %ds", self.name, e, delay)
                await asyncio.sleep(delay)
                backoff_idx += 1

    # -- Inbound message processing ------------------------------------------

    async def _on_message(self, raw: Dict[str, Any]) -> None:
        """Process a raw message dict from WeixinClawBot."""
        msg_id = raw.get("id") or uuid.uuid4().hex
        if self._is_duplicate(msg_id):
            return

        # Only handle text messages for now
        msg_type = raw.get("type", "text")
        if msg_type != "text":
            logger.debug("[%s] Skipping non-text message type: %s", self.name, msg_type)
            return

        text = (raw.get("content") or "").strip()
        if not text:
            return

        # Sender / chat info
        sender = raw.get("sender") or {}
        chat = raw.get("chat") or {}

        sender_id = sender.get("openid") or sender.get("id") or ""
        sender_name = sender.get("nickname") or sender.get("name") or sender_id

        chat_id = chat.get("id") or sender_id
        chat_name = chat.get("name") or sender_name
        is_group = chat.get("type") == "group"
        chat_type = "group" if is_group else "dm"

        # Redact sensitive identifiers in logs
        _redacted_id = _redact_openid(sender_id)

        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_name,
        )

        ts_raw = raw.get("timestamp")
        try:
            timestamp = (
                datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
                if ts_raw
                else datetime.now(tz=timezone.utc)
            )
        except (ValueError, OSError, TypeError):
            timestamp = datetime.now(tz=timezone.utc)

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=msg_id,
            raw_message=raw,
            timestamp=timestamp,
        )

        logger.debug(
            "[%s] Message from %s in %s: %s",
            self.name,
            sender_name,
            _redact_openid(chat_id),
            text[:50],
        )
        await self.handle_message(event)

    # -- Deduplication -------------------------------------------------------

    def _is_duplicate(self, msg_id: str) -> bool:
        now = time.time()
        if len(self._seen_messages) > DEDUP_MAX_SIZE:
            cutoff = now - DEDUP_WINDOW_SECONDS
            self._seen_messages = {
                k: v for k, v in self._seen_messages.items() if v > cutoff
            }
        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False

    # -- Outbound messaging --------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message to a WeChat chat via WeixinClawBot API."""
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")

        payload = {
            "chat_id": chat_id,
            "type": "text",
            "content": content[: self.MAX_MESSAGE_LENGTH],
        }

        try:
            resp = await self._http_client.post(
                f"{self._api_base}/messages/send",
                json=payload,
                timeout=15.0,
            )
            if resp.status_code < 300:
                data = resp.json()
                return SendResult(
                    success=True,
                    message_id=data.get("id") or uuid.uuid4().hex[:12],
                )
            body = resp.text
            logger.warning(
                "[%s] Send failed HTTP %d: %s", self.name, resp.status_code, body[:200]
            )
            return SendResult(success=False, error=f"HTTP {resp.status_code}: {body[:200]}")

        except httpx.TimeoutException:
            return SendResult(success=False, error="Timeout sending WeChat message")
        except Exception as e:
            logger.error("[%s] Send error: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """WeChat does not support typing indicators."""
        pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image message via WeixinClawBot."""
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")

        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "type": "image",
            "url": image_url,
        }
        if caption:
            payload["caption"] = caption[:MAX_MESSAGE_LENGTH]

        try:
            resp = await self._http_client.post(
                f"{self._api_base}/messages/send",
                json=payload,
                timeout=30.0,
            )
            if resp.status_code < 300:
                data = resp.json()
                return SendResult(success=True, message_id=data.get("id") or uuid.uuid4().hex[:12])
            return SendResult(success=False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about a WeChat chat."""
        if not self._http_client:
            return {"name": chat_id, "type": "dm", "chat_id": chat_id}
        try:
            resp = await self._http_client.get(
                f"{self._api_base}/chats/{chat_id}",
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "name": data.get("name") or chat_id,
                    "type": data.get("type", "dm"),
                    "chat_id": chat_id,
                }
        except Exception:
            pass
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redact_openid(openid: str) -> str:
    """Redact a WeChat OpenID for safe logging (show first 4 and last 4 chars)."""
    if not openid or len(openid) <= 8:
        return "****"
    return f"{openid[:4]}...{openid[-4:]}"
