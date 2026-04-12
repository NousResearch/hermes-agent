"""
HubAdapter — Slate Agent Hub integration via persistent WebSocket.

Same pattern as Discord: outbound WebSocket for receiving, REST API for sending.
Each Hub correspondent gets a unique Hermes session (chat_id = "hub:{agent_id}").
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource

logger = logging.getLogger("hub")

INITIAL_RECONNECT_DELAY = 5.0
MAX_RECONNECT_DELAY = 300.0
RECONNECT_BACKOFF_MULT = 2.0
PING_INTERVAL = 20.0


def check_hub_requirements() -> bool:
    """Check if Hub platform dependencies are available."""
    try:
        import httpx  # noqa: F401
        import websockets  # noqa: F401
        return True
    except ImportError:
        return False


class HubAdapter(BasePlatformAdapter):
    """Hub platform adapter using outbound WebSocket connection."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.HUB)
        self._agent_id = config.extra.get("agent_id", "")
        self._agent_secret = config.extra.get("agent_secret", "")
        self._ws_url = config.extra.get(
            "ws_url",
            f"wss://admin.slate.ceo/oc/brain/agents/{self._agent_id}/ws"
        )
        self._api_base = config.extra.get(
            "api_base",
            "https://admin.slate.ceo/oc/brain"
        )
        self._ws_lock = asyncio.Lock()
        self._ws_conn: Optional[Any] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._reconnect_delay = INITIAL_RECONNECT_DELAY
        self._should_reconnect = True
        self._connected_event = asyncio.Event()
        self._http_client: Optional[Any] = None

    # ── Required methods ────────────────────────────────────────────

    async def connect(self) -> bool:
        """Spawn WebSocket runner, wait up to 30s for connection."""
        import httpx
        self._http_client = httpx.AsyncClient(timeout=30.0)
        self._should_reconnect = True
        self._reconnect_delay = INITIAL_RECONNECT_DELAY
        self._reader_task = asyncio.create_task(self._run_ws())
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=30.0)
            self._mark_connected()
            logger.info("[Hub] Connected to %s", self._ws_url)
            return True
        except asyncio.TimeoutError:
            logger.error("[Hub] Connection timed out after 30s")
            self._should_reconnect = False
            if self._reader_task:
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except asyncio.CancelledError:
                    pass
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None
            self._mark_disconnected()
            return False

    async def disconnect(self) -> None:
        """Gracefully close WebSocket and stop reconnecting."""
        self._should_reconnect = False
        self._connected_event.clear()
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        async with self._ws_lock:
            if self._ws_conn:
                try:
                    await self._ws_conn.close()
                except Exception:
                    pass
                self._ws_conn = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._mark_disconnected()
        logger.info("[Hub] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message to a Hub agent via REST API.

        chat_id format: "hub:{recipient_id}" or just "{recipient_id}"
        """
        recipient = chat_id[4:] if chat_id.startswith("hub:") else chat_id

        try:
            import httpx
            client = self._http_client or httpx.AsyncClient(timeout=30.0)
            close_after = self._http_client is None
            try:
                resp = await client.post(
                    f"{self._api_base}/agents/{recipient}/message",
                    json={
                        "from": self._agent_id,
                        "secret": self._agent_secret,
                        "message": content,
                        **({"reply_to": reply_to} if reply_to else {}),
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return SendResult(
                        success=True,
                        message_id=str(data.get("message_id", "")),
                    )
                else:
                    return SendResult(
                        success=False,
                        error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                        retryable=resp.status_code >= 500,
                    )
            finally:
                if close_after:
                    await client.aclose()
        except Exception as e:
            if "Timeout" in type(e).__name__:
                return SendResult(success=False, error="Timeout", retryable=True)
            logger.exception("[Hub] send error")
            return SendResult(success=False, error=str(e), retryable=True)

    # ── WebSocket lifecycle ─────────────────────────────────────────

    async def _run_ws(self) -> None:
        """WebSocket runner with automatic reconnect."""
        import websockets

        while self._should_reconnect:
            try:
                logger.info("[Hub] Connecting to %s", self._ws_url)
                async with self._ws_lock:
                    self._ws_conn = await websockets.connect(
                        self._ws_url,
                        ping_interval=None,  # disable library pings — agent
                        ping_timeout=None,   # processing blocks the event loop
                    )

                # Auth handshake — Hub expects secret as first message
                await self._ws_conn.send(json.dumps({
                    "secret": self._agent_secret,
                }))
                auth_raw = await asyncio.wait_for(
                    self._ws_conn.recv(), timeout=10.0
                )
                auth = json.loads(auth_raw)
                if not auth.get("ok"):
                    logger.error("[Hub] Auth failed: %s", auth.get("error"))
                    async with self._ws_lock:
                        if self._ws_conn:
                            try:
                                await self._ws_conn.close()
                            except Exception:
                                pass
                            self._ws_conn = None
                    self._set_fatal_error(
                        "auth_failed",
                        f"Hub auth rejected: {auth.get('error')}",
                        retryable=False,
                    )
                    return

                logger.info("[Hub] Authenticated as %s", self._agent_id)
                self._reconnect_delay = INITIAL_RECONNECT_DELAY
                self._connected_event.set()

                await self._ws_read_loop()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("[Hub] WebSocket error: %s", e)
                if not self._should_reconnect:
                    break

            if self._should_reconnect:
                jitter = random.uniform(0, self._reconnect_delay * 0.3)
                wait = self._reconnect_delay + jitter
                logger.info("[Hub] Reconnecting in %.1fs", wait)
                await asyncio.sleep(wait)
                self._reconnect_delay = min(
                    self._reconnect_delay * RECONNECT_BACKOFF_MULT,
                    MAX_RECONNECT_DELAY,
                )
                self._connected_event.clear()

        self._mark_disconnected()

    async def _ws_read_loop(self) -> None:
        """Read messages from WebSocket until disconnect."""
        async with self._ws_lock:
            conn = self._ws_conn
        assert conn is not None

        while self._should_reconnect:
            try:
                raw = await asyncio.wait_for(
                    conn.recv(), timeout=PING_INTERVAL + 5
                )
                msg = json.loads(raw)
                await self._dispatch_ws_message(msg)
            except asyncio.TimeoutError:
                try:
                    async with self._ws_lock:
                        if self._ws_conn is conn:
                            await conn.send(json.dumps({"type": "ping"}))
                except Exception:
                    break
            except Exception as e:
                logger.warning("[Hub] Read error: %s", e)
                break

    async def _dispatch_ws_message(self, msg: Dict[str, Any]) -> None:
        """Route incoming WebSocket messages by type.

        Hub sends:
          {"ok": true, "type": "auth"}           — auth ack
          {"type": "message", "data": {...}}      — inbound message
          {"type": "pong"}                        — keepalive ack
          {"type": "send_result", ...}            — ack for WS sends (if used)
        """
        msg_type = msg.get("type", "")

        if msg_type == "message":
            await self._handle_inbound_message(msg.get("data", {}))
        elif msg_type in ("auth", "pong", "send_result"):
            pass  # Expected protocol messages — no logging needed
        elif msg.get("ok") is False:
            logger.error("[Hub] Error: %s", msg.get("error"))
        else:
            logger.debug("[Hub] Unknown message type: %s", msg_type)

    async def _handle_inbound_message(self, data: Dict[str, Any]) -> None:
        """Translate Hub message → MessageEvent → handle_message().

        Hub WS push schema:
          {"messageId": "abc", "from": "brain", "text": "Hello!", "timestamp": "..."}
        """
        from_id = data.get("from", "")
        text = data.get("text", "")
        message_id = data.get("messageId", "")
        timestamp_str = data.get("timestamp", "")

        if not from_id or not text:
            logger.debug("[Hub] Skipping message: from=%s len=%d", from_id, len(text))
            return

        # Self-message filter — prevent reply loops
        if from_id == self._agent_id:
            logger.debug("[Hub] Skipping self-message from %s", from_id)
            return

        try:
            ts = datetime.fromisoformat(
                timestamp_str.replace("Z", "+00:00")
            ) if timestamp_str else datetime.now(timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)

        event = MessageEvent(
            text=text,
            message_type=MessageType.COMMAND if text.startswith("/") else MessageType.TEXT,
            source=self.build_source(
                chat_id=f"hub:{from_id}",
                user_id=from_id,
                user_name=from_id,
                chat_type="dm",
            ),
            raw_message=data,
            message_id=message_id,
            reply_to_message_id=data.get("reply_to"),
            timestamp=ts,
        )

        logger.info("[Hub] Message from %s (len=%d)", from_id, len(text))
        await self.handle_message(event)

    # ── Required by BasePlatformAdapter ─────────────────────────────

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get info about a Hub conversation. chat_id format: hub:{agent_id} or {agent_id}."""
        agent_id = chat_id[4:] if chat_id.startswith("hub:") else chat_id
        try:
            import httpx
            client = self._http_client or httpx.AsyncClient(timeout=10.0)
            close_after = self._http_client is None
            try:
                resp = await client.get(f"{self._api_base}/agents/{agent_id}")
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "name": agent_id,
                        "type": "dm",
                        "description": data.get("description", ""),
                        "capabilities": data.get("capabilities", []),
                    }
            finally:
                if close_after:
                    await client.aclose()
        except Exception:
            pass
        return {"name": agent_id, "type": "dm"}
