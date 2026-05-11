"""
WPS Xiezuo (WPS365 协作) built-in platform adapter for Hermes gateway.

Follows the same patterns as FeishuAdapter:
  - WebSocket long-connection (default) + Webhook dual-mode
  - KSO-1 HMAC-SHA256 authentication on WS upgrade + webhook verification
  - AES-256-CBC event decryption (webhook mode; WS events are plaintext)
  - client_credentials token management via AppTokenStore
  - emoji_busy reaction on message receipt, removed after reply
  - Per-chat asyncio.Lock for serial message processing
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    SessionSource,
)
from gateway.session import build_session_key

logger = logging.getLogger("gateway.platforms.wps_xiezuo")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MESSAGE_TOPIC = "kso.app_chat.message"
MAX_MESSAGE_LENGTH = 5000
_WPS_APP_LOCK_SCOPE = "wps_xiezuo_app"
DEFAULT_BASE_URL = "https://openapi.wps.cn"
DEFAULT_CONNECTION_MODE = "websocket"

# Reaction type for acknowledging receipt
_REACTION_BUSY = "emoji_busy"

# ---------------------------------------------------------------------------
# Crypto helpers (port of wps_xiezuo_crypto.py)
# ---------------------------------------------------------------------------


def _derive_key(secret: str) -> bytes:
    """AES-256 key = MD5(secret).hexdigest() as UTF-8 bytes (32 B)."""
    return hashlib.md5(secret.encode()).hexdigest().encode("utf-8")


def _derive_iv(nonce: str) -> bytes:
    """AES-CBC IV = first 16 bytes of nonce string, UTF-8 encoded."""
    return nonce.encode("utf-8")[:16]


def compute_signature(
    app_id: str, app_secret: str, topic: str,
    nonce: str, timestamp: int, encrypted_data: str,
) -> str:
    """HMAC-SHA256 signature for WPS event verification.

    content = "app_id:topic:nonce:timestamp:encrypted_data"
    sig = base64url(HMAC-SHA256(content, app_secret))  stripped of trailing '='
    """
    content = f"{app_id}:{topic}:{nonce}:{timestamp}:{encrypted_data}"
    mac = hmac.new(app_secret.encode(), content.encode(), hashlib.sha256)
    return base64.urlsafe_b64encode(mac.digest()).decode().rstrip("=")


def verify_signature(
    signature: str, app_id: str, app_secret: str, topic: str,
    nonce: str, timestamp: int, encrypted_data: str,
) -> bool:
    expected = compute_signature(app_id, app_secret, topic, nonce, timestamp, encrypted_data)
    return hmac.compare_digest(expected, signature)


def decrypt_event(encrypted_data: str, secret: str, nonce: str) -> str:
    """Decrypt AES-256-CBC encrypted WPS event body."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding as crypto_padding

    key = _derive_key(secret)
    iv = _derive_iv(nonce)
    ciphertext = base64.b64decode(encrypted_data)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = crypto_padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded) + unpadder.finalize()
    return plaintext.decode("utf-8")


def build_handshake(app_id: str, app_secret: str, nonce: str) -> dict:
    """Build KSO-1 handshake frame (opcode=1).

    The server authenticates us by receiving this frame immediately
    after the WebSocket upgrade. The handshake frame carries app_id,
    HMAC-SHA256 signature, nonce, and timestamp.
    """
    timestamp = int(time.time())
    signature = compute_signature(app_id, app_secret, "", nonce, timestamp, "")
    return {
        "opcode": 1,
        "payload": json.dumps({
            "app_id": app_id,
            "signature": signature,
            "nonce": nonce,
            "timestamp": timestamp,
        }),
    }


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------


class AppTokenStore:
    """Manages client_credentials tokens for WPS Open Platform."""

    def __init__(self, base_url: str, app_id: str, app_secret: str):
        self._base_url = base_url
        self._app_id = app_id
        self._app_secret = app_secret
        self._token: Optional[str] = None
        self._expires_at: float = 0
        self._lock = asyncio.Lock()
        self._in_flight: Optional[asyncio.Future] = None

    async def get_token(self) -> str:
        if self._token and time.time() < self._expires_at - 60:
            return self._token

        # Deduplicate in-flight requests
        if self._in_flight and not self._in_flight.done():
            return await self._in_flight

        async with self._lock:
            if self._token and time.time() < self._expires_at - 60:
                return self._token

            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            self._in_flight = fut
            try:
                token, ttl = await self._fetch_token()
                self._token = token
                self._expires_at = time.time() + ttl
                fut.set_result(token)
            except Exception as exc:
                fut.set_exception(exc)
                raise
            finally:
                self._in_flight = None
            return self._token

    async def _fetch_token(self) -> tuple[str, int]:
        import aiohttp

        token_paths = ["/oauth2/token", "/openapi/oauth2/token"]
        last_exc: Optional[Exception] = None
        async with aiohttp.ClientSession() as session:
            for path in token_paths:
                try:
                    async with session.post(
                        f"{self._base_url}{path}",
                        data={
                            "grant_type": "client_credentials",
                            "client_id": self._app_id,
                            "client_secret": self._app_secret,
                        },
                    ) as resp:
                        body = await resp.json()
                        if resp.status == 200 and body.get("access_token"):
                            return body["access_token"], body.get("expires_in", 7200)
                except Exception as exc:
                    last_exc = exc
                    continue
        raise RuntimeError(f"Failed to obtain WPS access_token: {last_exc}")

    def invalidate(self) -> None:
        self._token = None
        self._expires_at = 0


# ---------------------------------------------------------------------------
# WPS API client
# ---------------------------------------------------------------------------


class WpsRequestError(Exception):
    pass


class WpsClient:
    """Lightweight HTTP client for WPS Open Platform APIs."""

    def __init__(self, base_url: str, token_store: AppTokenStore):
        self._base_url = base_url
        self._token_store = token_store

    async def request(self, method: str, path: str, *, body: Optional[dict] = None, json: Optional[dict] = None) -> dict:
        import aiohttp

        token = await self._token_store.get_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"{self._base_url}{path}"

        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, json=json or body) as resp:
                data = await resp.json()

        if data.get("code", -1) != 0:
            err = data.get("msg", data.get("message", str(data)))
            raise WpsRequestError(f"WPS API {path} failed: {err}")

        return data


# ---------------------------------------------------------------------------
# Inbound message normalization
# ---------------------------------------------------------------------------


def _normalize_message_content(message: dict) -> str:
    """Extract displayable text from a WPS message content dict."""
    content = message.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return content

    if not isinstance(content, dict):
        return str(content) if content else ""

    # Text message
    text_obj = content.get("text", {})
    if isinstance(text_obj, dict):
        return text_obj.get("content", "") or ""
    if isinstance(text_obj, str):
        return text_obj

    # Rich text — extract paragraphs
    rich = content.get("rich_text", {})
    if isinstance(rich, dict):
        parts: list[str] = []
        for block in rich.get("content", []):
            if isinstance(block, dict):
                for elem in block.get("body", []):
                    if isinstance(elem, dict):
                        t = elem.get("text", "")
                        if t:
                            parts.append(t)
        return "\n".join(parts) if parts else ""

    return json.dumps(content, ensure_ascii=False) if content else ""


def _strip_at_mention(text: str) -> str:
    """Strip inline <at ...>user_name</at> tags."""
    import re
    return re.sub(r"<at[^>]*>(.*?)</at>", r"\1", text).strip()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class WpsXiezuoAdapter(BasePlatformAdapter):
    """Hermes built-in adapter for WPS Xiezuo (WPS365 协作)."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config):
        super().__init__(config, Platform.WPS_XIEZUO)
        self._settings = self._load_settings(config.extra or {})
        self._apply_settings(self._settings)

        # Connection state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_client = None
        self._ws_session = None
        self._ws_is_websockets_lib = False
        self._recv_task: Optional[asyncio.Task] = None
        self._stop_reconnect = False
        self._connected = False
        self._token_store: Optional[AppTokenStore] = None
        self._wps_client: Optional[WpsClient] = None
        self._bot_user_id: Optional[str] = None
        self._seen_messages: Dict[str, float] = {}
        self._chat_locks: Dict[str, asyncio.Lock] = {}
        self._pending_reactions: Dict[str, str] = {}  # msg_id -> chat_id

    # ── Settings ────────────────────────────────────────────────

    @staticmethod
    def _load_settings(extra: dict) -> dict:
        return {
            "app_id": extra.get("app_id") or os.getenv("WPS_XIEZUO_APP_ID") or os.getenv("WPS_APP_ID", ""),
            "app_secret": extra.get("app_secret") or os.getenv("WPS_XIEZUO_APP_SECRET") or os.getenv("WPS_APP_SECRET", ""),
            "base_url": extra.get("base_url") or os.getenv("WPS_XIEZUO_BASE_URL", DEFAULT_BASE_URL),
            "connection_mode": extra.get("connection_mode") or os.getenv("WPS_XIEZUO_CONNECTION_MODE", DEFAULT_CONNECTION_MODE),
            "enable_encryption": extra.get("enable_encryption", os.getenv("WPS_XIEZUO_ENCRYPTION", "").lower() in ("true", "1", "yes")),
            "encrypt_key": extra.get("encrypt_key") or os.getenv("WPS_XIEZUO_ENCRYPT_KEY", ""),
            "home_channel": extra.get("home_channel") or os.getenv("WPS_XIEZUO_HOME_CHANNEL") or os.getenv("WPS_HOME_CHANNEL", ""),
        }

    def _apply_settings(self, s: dict) -> None:
        self._app_id = s["app_id"]
        self._app_secret = s["app_secret"]
        self._base_url = s["base_url"].rstrip("/")
        self._connection_mode = s["connection_mode"]
        self._enable_encryption = s["enable_encryption"]
        self._encrypt_key = s["encrypt_key"]
        self._home_channel = s["home_channel"]

    # ── Requirements check ───────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Connect ──────────────────────────────────────────────────

    async def connect(self) -> bool:
        if not self._app_id or not self._app_secret:
            logger.error("WPS Xiezuo: WPS_APP_ID and WPS_APP_SECRET must be set")
            return False

        self._loop = asyncio.get_running_loop()
        self._token_store = AppTokenStore(self._base_url, self._app_id, self._app_secret)
        self._wps_client = WpsClient(self._base_url, self._token_store)

        if self._connection_mode == "websocket":
            return await self._connect_ws()
        return await self._connect_webhook()

    async def _connect_ws(self) -> bool:
        """WebSocket long-connection matching open-event-sdk protocol.

        Protocol (matches Node.js open-event-sdk exactly):
        1. KSO-1 signed headers on the WS upgrade request
        2. X-Ack-Mode: required header for ACK mode
        3. No handshake frame — auth is via HTTP headers only
        4. Server sends WebSocket-level PINGs (~30s interval)
        5. Server pushes events as bare JSON with topic/operation/nonce/signature
        6. Client sends ACK: {"type":"ack","nonce":"...","code":200}

        IMPORTANT: The app must have "长连接" event subscription enabled
        on the WPS Open Platform developer dashboard, otherwise the
        server accepts the connection but does not push events.
        """
        ws_url = f"{self._base_url.replace('https://', 'wss://').replace('http://', 'ws://')}/v7/event/ws"

        try:
            import aiohttp

            headers = self._sign_ws_headers("/v7/event/ws")
            headers["X-Ack-Mode"] = "required"

            self._ws_session = aiohttp.ClientSession()
            self._ws_client = await self._ws_session.ws_connect(
                ws_url, headers=headers, heartbeat=None, autoping=False,
            )
            self._ws_is_websockets_lib = False

            # Connected — start recv loop
            self._connected = True
            self._stop_reconnect = False
            self._recv_task = asyncio.create_task(self._recv_loop())
            self._mark_connected()
            logger.info("WPS Xiezuo: WebSocket connected (app_id=%s)", self._app_id)
            return True

        except Exception as exc:
            logger.error("WPS Xiezuo: WS connection failed: %s: %s", type(exc).__name__, exc)
            await self._cleanup_ws()
            return False

    async def _connect_webhook(self) -> bool:
        """Start aiohttp webhook listener."""
        import aiohttp
        from aiohttp import web

        port = int(os.getenv("WPS_XIEZUO_WEBHOOK_PORT", "8765"))
        path = os.getenv("WPS_XIEZUO_WEBHOOK_PATH", "/wps/webhook")

        async def _handler(request: web.Request) -> web.Response:
            try:
                body = await request.json()
            except Exception:
                return web.Response(status=400, text="Invalid JSON")

            if body.get("type") == "url_verification":
                return web.json_response({"challenge": body.get("challenge", "")})

            event = body.get("event") or body
            # Webhook mode: verify signature and decrypt
            if self._enable_encryption:
                event = await self._decrypt_event(event) or event

            await self._handle_inbound_event(event)
            return web.json_response({"code": 0})

        app = web.Application()
        app.router.add_post(path, _handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        try:
            await site.start()
        except OSError as exc:
            logger.error("WPS Xiezuo: webhook failed: %s", exc)
            return False

        self._webhook_runner = runner
        self._connected = True
        self._mark_connected()
        logger.info("WPS Xiezuo: Webhook listening on 127.0.0.1:%d%s", port, path)
        return True

    # ── WS header signing (matches open-event-sdk KSO-1) ────────

    def _sign_ws_headers(self, uri: str) -> dict:
        """KSO-1 signed headers for WS upgrade (matches open-event-sdk)."""
        from email.utils import formatdate

        date_str = formatdate(timeval=None, localtime=False, usegmt=True)
        string_to_sign = f"KSO-1GET{uri}{date_str}"
        signature = hmac.new(
            self._app_secret.encode(), string_to_sign.encode(), hashlib.sha256,
        ).hexdigest()
        return {
            "X-Kso-Date": date_str,
            "X-Kso-Authorization": f"KSO-1 {self._app_id}:{signature}",
        }

    # ── Receive loop ─────────────────────────────────────────────

    async def _recv_loop(self) -> None:
        if not self._ws_client:
            return

        try:
            logger.info("WPS Xiezuo: recv_loop started (websockets_lib=%s)", self._ws_is_websockets_lib)
            while self._ws_client and not self._ws_client.closed:
                try:
                    if self._ws_is_websockets_lib:
                        raw_str = await asyncio.wait_for(self._ws_client.recv(), timeout=30)
                        logger.debug("WPS Xiezuo: WS raw frame len=%d", len(raw_str))
                        try:
                            data = json.loads(raw_str)
                        except json.JSONDecodeError:
                            continue
                        await self._handle_ws_data(data)
                    else:
                        # aiohttp: receive with timeout so we can detect stale connections
                        msg = await asyncio.wait_for(self._ws_client.receive(), timeout=30)
                        mt = msg.type
                        if mt == 1:  # TEXT
                            try:
                                data = json.loads(msg.data)
                            except json.JSONDecodeError:
                                continue
                            await self._handle_ws_data(data)
                        elif mt == 9:  # PING (WebSocket protocol level)
                            logger.info("WPS Xiezuo: WS PING, sending PONG")
                            await self._ws_client.pong(msg.data)
                        elif mt == 8:  # CLOSE
                            logger.info("WPS Xiezuo: WS CLOSE code=%s reason=%s", msg.data, msg.extra)
                            break
                        elif mt == 4:  # ERROR
                            logger.error("WPS Xiezuo: WS error: %s", self._ws_client.exception())
                            break
                        elif mt in (256, 257):  # CLOSING, CLOSED
                            logger.info("WPS Xiezuo: WS CLOSING/CLOSED type=%s", mt)
                            break
                        else:
                            logger.debug("WPS Xiezuo: WS unknown type=%s", mt)
                except asyncio.TimeoutError:
                    # No activity in 30s — check if connection is still alive
                    logger.debug("WPS Xiezuo: no message in 30s (heartbeat check)")
                    # If no PING for 90s+, connection might be dead
                    continue
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("WPS Xiezuo: recv loop error: %s", exc, exc_info=True)
        finally:
            self._connected = False
            if not self._stop_reconnect:
                await self._reconnect()

    async def _handle_ws_data(self, data: dict) -> None:
        """Route a single decoded JSON message from the WPS event stream.

        Matches open-event-sdk protocol exactly:
        - Server sends WebSocket-level PINGs (handled in recv_loop)
        - Server sends goaway messages: {"type":"goaway","reason":"..."}
        - Server pushes events as bare JSON: {"topic":"...","operation":"...",
          "nonce":"...","time":...,"signature":"...","encrypted_data":"..."}
        - Client responds with ACK: {"type":"ack","nonce":"...","code":200}
        """
        msg_type = data.get("type", "")

        # Handle goaway
        if msg_type == "goaway":
            reason = data.get("reason", "")
            logger.warning("WPS Xiezuo: received goaway: reason=%s message=%s",
                            reason, data.get("message", ""))
            if reason == "connection_replaced":
                # Another client connected with same app_id — we should
                # reconnect after a delay rather than giving up.
                logger.info("WPS Xiezuo: connection replaced, will reconnect in 5s")
                self._connected = False
                # Don't set _stop_reconnect — let the normal reconnect path handle it
            else:
                self._stop_reconnect = True
                logger.warning("WPS Xiezuo: goaway reason=%s, stopping reconnect", reason)
            return

        # Event messages have topic + operation
        topic = data.get("topic", "")
        operation = data.get("operation", "")
        if not topic or not operation:
            logger.debug("WPS Xiezuo: ignoring message without topic/operation: %s",
                          json.dumps(data)[:200])
            return

        # Process the event
        await self._handle_inbound_event(data)

        # Send ACK if nonce present
        nonce = data.get("nonce", "")
        if nonce:
            ack = {"type": "ack", "nonce": nonce, "code": 200}
            if self._ws_client and not self._ws_is_websockets_lib:
                await self._ws_client.send_json(ack)
            else:
                await self._send_ws(json.dumps(ack))
            logger.debug("WPS Xiezuo: ACK sent nonce=%s", nonce)

    async def _send_ws(self, text: str) -> None:
        if not self._ws_client:
            return
        try:
            if self._ws_is_websockets_lib:
                await self._ws_client.send(text)
            else:
                await self._ws_client.send_str(text)
        except Exception:
            pass

    # ── Inbound event processing ─────────────────────────────────

    async def _handle_inbound_event(self, event: dict) -> None:
        """Process a single inbound event (shared by WS and Webhook).

        Matches open-event-sdk flow:
        1. Verify event signature
        2. Decrypt encrypted_data
        3. Parse the decrypted JSON
        4. Dispatch to handle_message()
        """
        topic = event.get("topic", "")
        operation = event.get("operation", "")
        if topic != MESSAGE_TOPIC or operation != "create":
            return

        # Verify signature first (matches open-event-sdk behavior)
        signature = event.get("signature", "")
        nonce = event.get("nonce", "")
        event_time = str(event.get("time", "0"))
        encrypted_data = event.get("encrypted_data", "")

        if signature and nonce:
            ts = int(event_time) if event_time.isdigit() else 0
            if not verify_signature(signature, self._app_id, self._app_secret,
                                    topic, nonce, ts, encrypted_data):
                logger.warning("WPS Xiezuo: signature verification failed for nonce=%s", nonce)
                return

        # Decrypt event data
        if encrypted_data:
            try:
                decrypted = decrypt_event(encrypted_data, self._app_secret, nonce)
                data = json.loads(decrypted)
                event = {**event, "data": data}
            except Exception as exc:
                logger.error("WPS Xiezuo: decrypt failed: %s", exc)
                return

        data = event.get("data", {})
        if not data:
            return

        sender = data.get("sender", {})
        message = data.get("message", {})
        chat = data.get("chat", {})

        if not sender or not message:
            return

        msg_id = message.get("id", "")
        sender_id = sender.get("id", "")
        sender_type = sender.get("type", "")

        # Idempotency
        now = time.time()
        if msg_id and msg_id in self._seen_messages:
            return
        if msg_id:
            self._seen_messages[msg_id] = now
        if len(self._seen_messages) > 1000:
            cutoff = now - 300
            self._seen_messages = {k: v for k, v in self._seen_messages.items() if v > cutoff}

        # Anti-loopback
        if sender_type == "robot" or sender_id == self._bot_user_id:
            return

        # Chat type
        chat_id = message.get("chat_id", "") or chat.get("id", "")
        chat_type_raw = str(chat.get("type", "")).lower() if chat else ""
        is_dm = chat_type_raw in ("p2p", "single", "direct") or (not chat_id and sender_id)
        chat_type = "dm" if is_dm else "group"

        # Normalize content
        text = _strip_at_mention(_normalize_message_content(message))
        if not text:
            return

        # Build source and dispatch
        source = self.build_source(
            chat_id=chat_id or sender_id,
            chat_name=chat.get("name", chat_id or sender_id),
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender.get("name", sender_id),
            message_id=msg_id,
        )

        msg_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=msg_id or str(int(now * 1000)),
            timestamp=datetime.now(),
        )

        chat_lock = self._get_chat_lock(chat_id or sender_id)
        async with chat_lock:
            await self.handle_message(msg_event)

    # ── Lifecycle hooks (reaction-based feedback, like Feishu) ────

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add emoji_busy reaction when processing begins."""
        chat_id = getattr(event.source, "chat_id", "") or ""
        msg_id = getattr(event.source, "message_id", "") or event.message_id or ""
        logger.info("WPS Xiezuo: on_processing_start chat=%s msg=%s client=%s", chat_id, msg_id, bool(self._wps_client))
        if not self._wps_client or not chat_id or not msg_id:
            logger.warning("WPS Xiezuo: skipping reaction — client=%s chat=%s msg=%s", bool(self._wps_client), chat_id, msg_id)
            return
        try:
            await self._wps_client.request(
                "POST",
                f"/v7/chats/{chat_id}/messages/{msg_id}/reactions/create",
                json={"reaction_type": _REACTION_BUSY},
            )
            # Track that we added a reaction to this message
            self._pending_reactions[msg_id] = chat_id
            logger.debug("WPS Xiezuo: added thinking reaction msg=%s", msg_id)
        except Exception as exc:
            logger.debug("WPS Xiezuo: add reaction failed: %s", exc)

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Remove emoji_busy reaction after processing completes."""
        chat_id = self._pending_reactions.pop(getattr(event, "message_id", ""), "")
        msg_id = getattr(event.source, "message_id", "") or event.message_id or ""
        if not self._wps_client or not chat_id or not msg_id:
            return
        try:
            await self._wps_client.request(
                "POST",
                f"/v7/chats/{chat_id}/messages/{msg_id}/reactions/delete",
                json={"reaction_type": _REACTION_BUSY},
            )
            logger.debug("WPS Xiezuo: removed thinking reaction msg=%s", msg_id)
        except Exception as exc:
            logger.debug("WPS Xiezuo: remove reaction failed: %s", exc)

    # ── Outbound sending ─────────────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message to a WPS chat or user."""
        if not self._connected or not self._wps_client:
            return SendResult(success=False, error="Not connected")

        receiver = self._parse_receiver(chat_id)
        if not receiver:
            return SendResult(success=False, error=f"Invalid chat_id: {chat_id}")

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)

        last_result = None
        for chunk in chunks:
            try:
                body = {
                    "type": "text",
                    "receiver": receiver,
                    "content": {"text": {"content": chunk, "type": "markdown"}},
                }
                result = await self._wps_client.request("POST", "/v7/messages/create", json=body)
                last_result = SendResult(
                    success=True,
                    message_id=result.get("data", {}).get("message_id", ""),
                )
            except WpsRequestError as exc:
                err_str = str(exc).lower()
                if "401" in err_str or "token" in err_str:
                    self._token_store.invalidate()
                last_result = SendResult(success=False, error=str(exc))
                break

        return last_result or SendResult(success=False, error="No chunks to send")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """WPS doesn't have a typing API — no-op (use reactions instead)."""
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about a WPS chat."""
        if not self._wps_client:
            return {"name": chat_id, "type": "unknown"}
        try:
            data = await self._wps_client.request("GET", f"/v7/chats/{chat_id}")
            chat_data = data.get("data", data)
            return {
                "name": chat_data.get("name", chat_id),
                "type": "group" if chat_data.get("type") == "group" else "dm",
            }
        except Exception:
            return {"name": chat_id, "type": "unknown"}

    def format_message(self, content: str) -> str:
        """WPS supports markdown — pass through."""
        return content.strip()

    @staticmethod
    def _parse_receiver(chat_id: str) -> Optional[dict]:
        """Parse chat_id into WPS receiver dict.

        WPS API requires ``receiver_id`` (not ``id``) inside the receiver object.
        """
        if chat_id.startswith("user:"):
            return {"type": "user", "receiver_id": chat_id[5:]}
        if chat_id.startswith("chat:"):
            return {"type": "chat", "receiver_id": chat_id[5:]}
        # Bare ID → treat as chat
        if chat_id:
            return {"type": "chat", "receiver_id": chat_id}
        return None

    # ── Per-chat serialization ───────────────────────────────────

    def _get_chat_lock(self, chat_id: str) -> asyncio.Lock:
        lock = self._chat_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._chat_locks[chat_id] = lock
        return lock

    # ── Reconnect ────────────────────────────────────────────────

    async def _reconnect(self) -> None:
        attempt = 0
        while not self._stop_reconnect:
            delay = min(2 ** attempt, 60) + random.random()  # exponential + jitter
            logger.info("WPS Xiezuo: reconnecting in %.1fs (attempt %d)", delay, attempt + 1)
            await asyncio.sleep(delay)
            if await self._connect_ws():
                return
            attempt += 1

    # ── Cleanup ──────────────────────────────────────────────────

    async def disconnect(self) -> None:
        self._stop_reconnect = True
        if self._recv_task:
            self._recv_task.cancel()
        await self._cleanup_ws()
        self._connected = False

    async def _cleanup_ws(self) -> None:
        if self._ws_client:
            try:
                if self._ws_is_websockets_lib:
                    await self._ws_client.close()
                elif hasattr(self._ws_client, "closed") and not self._ws_client.closed:
                    await self._ws_client.close()
            except Exception:
                pass
        if self._ws_session:
            await self._ws_session.close()
        self._ws_client = None
        self._ws_session = None
        self._ws_is_websockets_lib = False

    # ── Watchdog ─────────────────────────────────────────────────

    def _on_ping(self) -> None:
        """Called on any inbound activity to keep watchdog alive."""
        # Base class doesn't have a watchdog; this is a placeholder
        # for future reconnect-on-stale logic
        pass


# ---------------------------------------------------------------------------
# Requirements check (called by gateway/run.py)
# ---------------------------------------------------------------------------

def check_wps_xiezuo_requirements() -> bool:
    """Return True if the WPS Xiezuo adapter can be instantiated."""
    app_id = os.getenv("WPS_XIEZUO_APP_ID") or os.getenv("WPS_APP_ID")
    app_secret = os.getenv("WPS_XIEZUO_APP_SECRET") or os.getenv("WPS_APP_SECRET")
    if not app_id or not app_secret:
        return False
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher  # noqa: F401
    except ImportError:
        return False
    return True


# Make Platform available locally (avoid circular import)
from gateway.config import Platform  # noqa: E402
