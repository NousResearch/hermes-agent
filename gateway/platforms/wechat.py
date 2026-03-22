"""
WeChat Official Account platform adapter.

Receives messages via WeChat's webhook push (XML) and sends replies via
the Customer Service Message API (JSON).  Runs an aiohttp server to handle
the verification handshake and incoming message posts.

Important WeChat API constraints:
- Replies must use the Customer Service API (not inline XML) since agent
  responses take longer than WeChat's 5-second timeout.
- WeChat retries message pushes up to 3 times if no "success" response
  within 5 seconds — dedup by MsgId is essential.
- The Customer Service API has a 48-hour reply window: you can only send
  messages within 48h of the user's last message.
- Requires a verified Service Account (not Subscription Account).
- Set the server message mode to "Plain text" (明文模式) in WeChat admin.
  Encrypted mode (安全模式) is not supported yet.

Requires:
    WECHAT_APP_ID and WECHAT_APP_SECRET env vars (from WeChat Official Account dashboard)
    WECHAT_TOKEN env var (the token you configured in WeChat's server settings)

Configuration in config.yaml:
    platforms:
      wechat:
        enabled: true
        token: "your-verification-token"
        extra:
          app_id: "wx1234567890abcdef"
          app_secret: "your-app-secret"
          port: 8680
"""

import asyncio
import hashlib
import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SessionSource,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 2048  # WeChat text message limit
DEFAULT_PORT = 8680
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 1000

# WeChat API endpoints
WECHAT_API_BASE = "https://api.weixin.qq.com"
TOKEN_URL = f"{WECHAT_API_BASE}/cgi-bin/token"
SEND_URL = f"{WECHAT_API_BASE}/cgi-bin/message/custom/send"


def check_wechat_requirements() -> bool:
    """Check if WeChat dependencies are available and configured."""
    if not AIOHTTP_AVAILABLE or not HTTPX_AVAILABLE:
        return False
    app_id = os.getenv("WECHAT_APP_ID", "")
    app_secret = os.getenv("WECHAT_APP_SECRET", "")
    token = os.getenv("WECHAT_TOKEN", "")
    if not (app_id and app_secret and token):
        return False
    return True


def _verify_signature(token: str, signature: str, timestamp: str, nonce: str) -> bool:
    """Verify WeChat's server verification signature."""
    items = sorted([token, timestamp, nonce])
    digest = hashlib.sha1("".join(items).encode("utf-8")).hexdigest()
    return digest == signature


def _parse_xml_message(xml_data: str) -> Optional[Dict[str, str]]:
    """Parse a WeChat XML message into a dict."""
    try:
        root = ET.fromstring(xml_data)
        return {child.tag: (child.text or "") for child in root}
    except ET.ParseError:
        logger.warning("Failed to parse WeChat XML message")
        return None


class WeChatAdapter(BasePlatformAdapter):
    """WeChat Official Account adapter.

    Receives messages via webhook push from WeChat servers.
    Sends replies via the Customer Service Message API.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WECHAT)

        self._verify_token: str = config.token or os.getenv("WECHAT_TOKEN", "")
        self._app_id: str = config.extra.get("app_id", "") or os.getenv("WECHAT_APP_ID", "")
        self._app_secret: str = config.extra.get("app_secret", "") or os.getenv("WECHAT_APP_SECRET", "")
        self._port: int = int(config.extra.get("port", DEFAULT_PORT))

        self._access_token: str = ""
        self._token_expires_at: float = 0

        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._http_client: Optional["httpx.AsyncClient"] = None

        # Message deduplication
        self._seen_messages: Dict[str, float] = {}

    # -- Access token management ----------------------------------------------

    async def _get_access_token(self) -> str:
        """Get a valid access token, refreshing if expired."""
        if time.time() < self._token_expires_at - 60 and self._access_token:
            return self._access_token

        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=30)

        try:
            resp = await self._http_client.get(TOKEN_URL, params={
                "grant_type": "client_credential",
                "appid": self._app_id,
                "secret": self._app_secret,
            })
            data = resp.json()
            if "access_token" in data:
                self._access_token = data["access_token"]
                self._token_expires_at = time.time() + data.get("expires_in", 7200)
                logger.debug("[%s] Access token refreshed, expires in %ds", self.name, data.get("expires_in", 7200))
            else:
                logger.error("[%s] Failed to get access token: %s", self.name, data)
        except Exception as e:
            logger.error("[%s] Access token request failed: %s", self.name, e)

        return self._access_token

    # -- Connection lifecycle -------------------------------------------------

    async def connect(self) -> bool:
        """Start the webhook server for WeChat message push."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False
        if not HTTPX_AVAILABLE:
            logger.warning("[%s] httpx not installed", self.name)
            return False
        if not self._verify_token or not self._app_id or not self._app_secret:
            logger.warning("[%s] WECHAT_TOKEN, WECHAT_APP_ID, and WECHAT_APP_SECRET are required", self.name)
            return False

        self._http_client = httpx.AsyncClient(timeout=30)

        # Verify credentials by fetching an access token
        token = await self._get_access_token()
        if not token:
            logger.error("[%s] Could not obtain access token — check APP_ID and APP_SECRET", self.name)
            return False

        self._app = web.Application()
        self._app.router.add_get("/wechat", self._handle_verification)
        self._app.router.add_post("/wechat", self._handle_message_push)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._port)
        await site.start()

        self._mark_connected()
        logger.info("[%s] Listening on 0.0.0.0:%d/wechat", self.name, self._port)
        return True

    async def disconnect(self) -> None:
        """Stop the webhook server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._app = None
        self._mark_disconnected()
        logger.info("[%s] Disconnected", self.name)

    # -- Webhook handlers -----------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        return web.json_response({"status": "ok", "platform": "wechat"})

    async def _handle_verification(self, request: "web.Request") -> "web.Response":
        """GET /wechat — WeChat server verification handshake."""
        signature = request.query.get("signature", "")
        timestamp = request.query.get("timestamp", "")
        nonce = request.query.get("nonce", "")
        echostr = request.query.get("echostr", "")

        if _verify_signature(self._verify_token, signature, timestamp, nonce):
            logger.info("[%s] Server verification succeeded", self.name)
            return web.Response(text=echostr)
        else:
            logger.warning("[%s] Server verification failed", self.name)
            return web.Response(status=403, text="Invalid signature")

    async def _handle_message_push(self, request: "web.Request") -> "web.Response":
        """POST /wechat — receive message push from WeChat."""
        # Verify signature
        signature = request.query.get("signature", "")
        timestamp = request.query.get("timestamp", "")
        nonce = request.query.get("nonce", "")

        if not _verify_signature(self._verify_token, signature, timestamp, nonce):
            return web.Response(status=403, text="Invalid signature")

        body = await request.text()
        msg = _parse_xml_message(body)
        if not msg:
            return web.Response(text="success")

        msg_type = msg.get("MsgType", "")
        msg_id = msg.get("MsgId", "")

        # Only handle text messages for now
        if msg_type != "text":
            logger.debug("[%s] Skipping non-text message type: %s", self.name, msg_type)
            return web.Response(text="success")

        # Dedup
        if msg_id and msg_id in self._seen_messages:
            return web.Response(text="success")
        if msg_id:
            now = time.time()
            self._seen_messages[msg_id] = now
            # Clean old entries
            if len(self._seen_messages) > DEDUP_MAX_SIZE:
                cutoff = now - DEDUP_WINDOW_SECONDS
                self._seen_messages = {k: v for k, v in self._seen_messages.items() if v > cutoff}

        from_user = msg.get("FromUserName", "")
        to_user = msg.get("ToUserName", "")
        content = msg.get("Content", "").strip()

        if not content or not from_user:
            return web.Response(text="success")

        # Build MessageEvent and dispatch
        source = SessionSource(
            platform=Platform.WECHAT,
            chat_id=from_user,
            user_id=from_user,
            user_name=from_user[:8],  # OpenIDs are opaque; truncate for display
            chat_type="dm",
        )
        event = MessageEvent(
            source=source,
            text=content,
            message_type=MessageType.TEXT,
            raw=msg,
        )

        # Respond with "success" immediately (WeChat requires response within 5s)
        # Process the message asynchronously
        asyncio.create_task(self._dispatch_message(event))
        return web.Response(text="success")

    async def _dispatch_message(self, event: MessageEvent) -> None:
        """Dispatch a message event through the standard handler."""
        try:
            await self.handle_message(event)
        except Exception as e:
            logger.error("[%s] Error handling message: %s", self.name, e, exc_info=True)

    # -- Sending messages -----------------------------------------------------

    async def send(self, chat_id: str, text: str, **kwargs: Any) -> SendResult:
        """Send a text message via the Customer Service Message API."""
        token = await self._get_access_token()
        if not token:
            return SendResult(success=False, error="No access token")

        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=30)

        # Split long messages
        chunks = self._split_message(text)
        last_msg_id = None

        for chunk in chunks:
            try:
                resp = await self._http_client.post(
                    f"{SEND_URL}?access_token={token}",
                    json={
                        "touser": chat_id,
                        "msgtype": "text",
                        "text": {"content": chunk},
                    },
                )
                data = resp.json()
                errcode = data.get("errcode", 0)
                if errcode != 0:
                    # 95026 = 48-hour reply window expired (user hasn't messaged recently)
                    if errcode == 95026:
                        logger.warning("[%s] 48-hour reply window expired for %s", self.name, chat_id)
                        return SendResult(success=False, error="48-hour reply window expired — user must message first")
                    # 45015 = user has blocked the account or hasn't followed
                    if errcode == 45015:
                        logger.warning("[%s] User %s has not followed or has blocked the account", self.name, chat_id[:8])
                        return SendResult(success=False, error="User not following or blocked")
                    # 40001 = invalid access token — force refresh and retry once
                    if errcode == 40001:
                        self._token_expires_at = 0
                        token = await self._get_access_token()
                        if token:
                            retry = await self._http_client.post(
                                f"{SEND_URL}?access_token={token}",
                                json={"touser": chat_id, "msgtype": "text", "text": {"content": chunk}},
                            )
                            retry_data = retry.json()
                            if retry_data.get("errcode", 0) == 0:
                                last_msg_id = str(retry_data.get("msgid", ""))
                                continue
                    logger.error("[%s] Send failed: %s", self.name, data)
                    return SendResult(success=False, error=f"errcode {errcode}: {data.get('errmsg', 'Unknown error')}")
                last_msg_id = str(data.get("msgid", ""))
            except Exception as e:
                logger.error("[%s] Send error: %s", self.name, e)
                return SendResult(success=False, error=str(e))

        return SendResult(success=True, message_id=last_msg_id)

    def _split_message(self, text: str) -> list:
        """Split text into chunks respecting WeChat's message limit."""
        if len(text) <= MAX_MESSAGE_LENGTH:
            return [text]
        chunks = []
        while text:
            if len(text) <= MAX_MESSAGE_LENGTH:
                chunks.append(text)
                break
            # Find a good split point
            split_at = text.rfind("\n", 0, MAX_MESSAGE_LENGTH)
            if split_at < MAX_MESSAGE_LENGTH // 2:
                split_at = text.rfind(" ", 0, MAX_MESSAGE_LENGTH)
            if split_at < MAX_MESSAGE_LENGTH // 2:
                split_at = MAX_MESSAGE_LENGTH
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip()
        return chunks

    async def send_image(self, chat_id: str, image_path: str, caption: str = "", **kwargs) -> SendResult:
        """Send an image via WeChat (upload + send)."""
        # WeChat image sending requires media upload first — skip for now
        if caption:
            return await self.send(chat_id, f"[Image: {caption}]")
        return SendResult(success=False, error="Image sending not yet supported")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get info about a chat/user."""
        return {"chat_id": chat_id, "platform": "wechat", "type": "dm"}
