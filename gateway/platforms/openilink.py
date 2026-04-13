"""
OpeniLink Hub platform adapter.

Connects Hermes Agent to messaging platforms (WeChat, etc.) via OpeniLink Hub's
WebSocket Bot API. Hub manages the platform connection; this adapter handles
event delivery and message sending over ``wss://host/bot/v1/ws?token=TOKEN``.

Protocol:
- Server sends ``{"type": "init", "data": {...}}`` on connect.
- Server delivers events as ``{"type": "event", "v": 1, "event": {...}}``.
- Client sends ``{"type": "send", "to": "...", "content": "...", "req_id": "..."}``.
- Server acknowledges with ``{"type": "ack", "ok": true, "req_id": "..."}``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_url,
    safe_url_for_log,
)


def check_openilink_requirements() -> bool:
    """Return True when deps and config are available."""
    if not AIOHTTP_AVAILABLE:
        return False
    token = os.getenv("OPENILINK_TOKEN", "")
    hub_url = os.getenv("OPENILINK_HUB_URL", "")
    return bool(token or hub_url)


class OpeniLinkAdapter(BasePlatformAdapter):
    """Connects to OpeniLink Hub via WebSocket to send/receive messages."""

    # Reconnection constants
    _MAX_RECONNECT_ATTEMPTS = 10
    _BASE_DELAY = 2.0
    _MAX_DELAY = 60.0

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.OPENILINK)

        hub_url = (config.extra.get("hub_url") or "").rstrip("/")
        if not hub_url:
            hub_url = "https://localhost:9800"
        # Derive ws_url from hub_url
        ws_scheme = "wss" if hub_url.startswith("https") else "ws"
        http_base = hub_url.replace("https://", "").replace("http://", "")
        self._ws_url = f"{ws_scheme}://{http_base}/bot/v1/ws"

        self._token: str = config.token or ""
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._reconnect_attempts: int = 0
        self._bot_id: str = ""
        self._installation_id: str = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        try:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            url = f"{self._ws_url}?token={self._token}"
            logger.info("[openilink] Connecting to %s", safe_url_for_log(url))
            self._ws = await self._session.ws_connect(url, heartbeat=50)

            self._reconnect_attempts = 0
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._mark_connected()
            logger.info("[openilink] Connected")
            return True
        except Exception as exc:
            logger.error("[openilink] Connection failed: %s", exc)
            return False

    async def disconnect(self) -> None:
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

        self._mark_disconnected()
        logger.info("[openilink] Disconnected")

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        try:
            async for msg in self._ws:  # type: ignore[union-attr]
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.warning("[openilink] Invalid JSON: %s", msg.data[:120])
                        continue
                    await self._dispatch(payload)
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    logger.warning("[openilink] WebSocket %s", msg.type.name)
                    break
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("[openilink] Receive loop error: %s", exc)

        # Connection lost — try to reconnect
        await self._reconnect()

    async def _dispatch(self, payload: dict) -> None:
        msg_type = payload.get("type", "")

        if msg_type == "init":
            data = payload.get("data", {})
            self._bot_id = data.get("bot_id", "")
            self._installation_id = data.get("installation_id", "")
            logger.info(
                "[openilink] Init: bot=%s installation=%s",
                self._bot_id, self._installation_id,
            )

        elif msg_type == "event":
            await self._handle_event(payload)

        elif msg_type == "ack":
            ok = payload.get("ok", False)
            req_id = payload.get("req_id", "")
            if not ok:
                logger.warning("[openilink] ACK failed req_id=%s", req_id)

        elif msg_type == "pong":
            pass  # heartbeat response

        elif msg_type == "error":
            logger.error("[openilink] Server error: %s", payload.get("error", ""))

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, payload: dict) -> None:
        event_data = payload.get("event", {})
        event_type = event_data.get("type", "")  # e.g. "message.text"
        data = event_data.get("data", {})

        sender = data.get("sender", {})
        user_id = sender.get("id", "")
        group = data.get("group")
        items = data.get("items", [])
        message_id = data.get("message_id", "")
        trace_id = payload.get("trace_id", "")

        # Determine chat_id: use group id if available, else user_id
        chat_id = group.get("id") if group else user_id
        if not chat_id:
            return

        chat_type = "group" if group else "dm"

        # Extract text and media from items
        text_parts = []
        media_urls = []
        media_types = []
        for item in items:
            item_type = item.get("type", "")
            if item_type == "text":
                text_parts.append(item.get("text", ""))
            elif item_type in ("image", "video", "file", "voice"):
                media = item.get("media", {})
                media_url = media.get("url", "")
                if media_url:
                    try:
                        ext = ".jpg" if item_type == "image" else f".{item_type}"
                        cached = await cache_image_from_url(media_url, ext=ext)
                        media_urls.append(cached)
                        media_types.append(f"{item_type}/{ext.lstrip('.')}")
                    except Exception as exc:
                        logger.warning("[openilink] Cache media failed: %s", exc)

        text = "\n".join(text_parts)

        # Map event type to MessageType
        if event_type == "message.image" or media_urls:
            msg_type = MessageType.PHOTO
        elif event_type == "message.voice":
            msg_type = MessageType.VOICE
        elif event_type == "message.video":
            msg_type = MessageType.VIDEO
        elif event_type == "message.file":
            msg_type = MessageType.DOCUMENT
        else:
            msg_type = MessageType.TEXT

        source = self.build_source(
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id,
        )

        event = MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=payload,
            message_id=message_id,
            media_urls=media_urls,
            media_types=media_types,
        )
        await self.handle_message(event)

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._ws or self._ws.closed:
            return SendResult(success=False, error="WebSocket not connected", retryable=True)

        try:
            chunks = self.truncate_message(content, max_length=4096)
            last_req_id = ""
            for chunk in chunks:
                req_id = f"r_{uuid.uuid4().hex[:8]}"
                await self._ws.send_json({
                    "type": "send",
                    "req_id": req_id,
                    "to": chat_id,
                    "content": chunk,
                })
                last_req_id = req_id
            return SendResult(success=True, message_id=last_req_id)
        except Exception as exc:
            logger.error("[openilink] Send failed: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if not self._ws or self._ws.closed:
            return
        try:
            await self._ws.send_json({"type": "send_typing", "to": chat_id})
        except Exception:
            pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        # Hub doesn't have a native image-send via WS; fall back to text with URL
        text = f"![image]({image_url})"
        if caption:
            text = f"{caption}\n{text}"
        return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    async def _reconnect(self) -> None:
        if self._reconnect_attempts >= self._MAX_RECONNECT_ATTEMPTS:
            msg = f"Reconnection exhausted after {self._MAX_RECONNECT_ATTEMPTS} attempts"
            logger.error("[openilink] %s", msg)
            self._set_fatal_error("openilink_reconnect", msg, retryable=True)
            return

        self._reconnect_attempts += 1
        delay = min(self._BASE_DELAY * (2 ** (self._reconnect_attempts - 1)), self._MAX_DELAY)
        logger.warning(
            "[openilink] Reconnecting in %.0fs (attempt %d/%d)",
            delay, self._reconnect_attempts, self._MAX_RECONNECT_ATTEMPTS,
        )
        await asyncio.sleep(delay)
        await self.connect()
