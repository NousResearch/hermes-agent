"""LINE Messaging API platform adapter.

Uses the LINE Messaging API directly via httpx for:
- Receiving messages via webhook (HMAC-SHA256 signature verification)
- Sending text messages (Reply API + Push API)
- Media handling (images, video, audio, files)
- Typing indicators (loading animation)
- Group and 1:1 chat support

No external LINE SDK dependency — all API calls are made directly via httpx.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import secrets
import struct
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple
from urllib.parse import quote as _urlquote

import httpx
from aiohttp import web

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
)
from gateway.platforms.helpers import strip_markdown

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINE_API_BASE = "https://api.line.me/v2/bot"
LINE_DATA_API_BASE = "https://api-data.line.me/v2/bot"
LINE_LOADING_API = "https://api.line.me/v2/bot/chat/loading/start"
MAX_MESSAGE_LENGTH = 5000
DEFAULT_WEBHOOK_HOST = "0.0.0.0"
DEFAULT_WEBHOOK_PORT = 8443
DEFAULT_WEBHOOK_PATH = "/line/webhook"

# LINE Messaging API file size limits
LINE_IMAGE_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
LINE_AUDIO_MAX_BYTES = 200 * 1024 * 1024  # 200 MB
LINE_VIDEO_MAX_BYTES = 200 * 1024 * 1024  # 200 MB


def _make_preview_png() -> bytes:
    """Return a minimal 1×1 white PNG suitable for LINE video previewImageUrl.

    LINE requires previewImageUrl to be a JPEG or PNG image.  When sending a
    local video we have no thumbnail, so we serve this tiny placeholder instead.
    Generated with pure stdlib (no Pillow required).
    """
    def _chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))  # 1×1 RGB
        + _chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))             # white pixel
        + _chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_line_requirements() -> bool:
    """Check if LINE dependencies are available."""
    try:
        import httpx as _httpx  # noqa: F401
        import aiohttp as _aiohttp  # noqa: F401
    except ImportError:
        return False
    channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
    channel_secret = os.getenv("LINE_CHANNEL_SECRET", "")
    return bool(channel_access_token and channel_secret)


def _verify_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    """Verify LINE webhook signature using HMAC-SHA256."""
    expected = hmac.new(
        channel_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).digest()
    expected_b64 = base64.b64encode(expected).decode("utf-8")
    return hmac.compare_digest(expected_b64, signature)


def _extract_text_from_event(event: dict) -> str:
    """Extract text content from a LINE message event."""
    message = event.get("message", {})
    msg_type = message.get("type", "")
    if msg_type == "text":
        return message.get("text", "")
    if msg_type == "sticker":
        keywords = message.get("keywords", [])
        if keywords:
            return f"[Sticker: {', '.join(keywords)}]"
        return "[Sticker]"
    if msg_type == "location":
        title = message.get("title", "")
        address = message.get("address", "")
        lat = message.get("latitude", "")
        lon = message.get("longitude", "")
        parts = [p for p in [title, address, f"({lat}, {lon})"] if p]
        return f"[Location: {' - '.join(parts)}]"
    return ""


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class LineAdapter(BasePlatformAdapter):
    """LINE Messaging API adapter using webhooks for inbound messages."""

    platform = Platform.LINE
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.LINE)
        extra = config.extra or {}
        self.channel_access_token: str = (
            config.token
            or extra.get("channel_access_token")
            or os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
        )
        self.channel_secret: str = (
            extra.get("channel_secret")
            or os.getenv("LINE_CHANNEL_SECRET", "")
        )
        self.webhook_host: str = (
            extra.get("webhook_host")
            or os.getenv("LINE_WEBHOOK_HOST", DEFAULT_WEBHOOK_HOST)
        )
        try:
            self.webhook_port: int = int(
                extra.get("webhook_port")
                or os.getenv("LINE_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT))
            )
        except (ValueError, TypeError):
            logger.warning("[LINE] Invalid webhook_port value, using default %d", DEFAULT_WEBHOOK_PORT)
            self.webhook_port = DEFAULT_WEBHOOK_PORT
        self.webhook_path: str = (
            extra.get("webhook_path")
            or os.getenv("LINE_WEBHOOK_PATH", DEFAULT_WEBHOOK_PATH)
        )
        self._http_client: Optional[httpx.AsyncClient] = None
        self._webhook_app: Optional[web.Application] = None
        self._webhook_runner: Optional[web.AppRunner] = None
        self._user_profile_cache: Dict[str, Dict[str, str]] = {}
        self._last_loading_at: Dict[str, float] = {}
        self._loading_interval_seconds: float = 18.0
        self._loading_seconds: int = 20
        # token -> (absolute_path, expires_at) for temporary media serving
        self._media_tokens: Dict[str, Tuple[str, float]] = {}
        self._media_ttl: int = 300  # seconds
        # paths of tempfiles created internally (e.g. video preview PNGs) to
        # delete from disk when their token expires
        self._media_temp_paths: Set[str] = set()

    # ------------------------------------------------------------------
    # HTTP client helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.channel_access_token}",
            "Content-Type": "application/json",
        }

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start the webhook server to receive LINE events."""
        if not self.channel_access_token or not self.channel_secret:
            self._set_fatal_error(
                "missing_credentials",
                "LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET are required",
                retryable=False,
            )
            return False

        # Prevent two profiles from running the same bot simultaneously
        if not self._acquire_platform_lock(
            "line-channel-token", self.channel_access_token, "LINE channel access token"
        ):
            return False

        # Verify bot credentials by fetching bot info
        try:
            client = await self._ensure_client()
            resp = await client.get(
                f"{LINE_API_BASE}/info",
                headers=self._auth_headers(),
            )
            if resp.status_code != 200:
                self._release_platform_lock()
                self._set_fatal_error(
                    "auth_failed",
                    f"LINE bot info request failed (HTTP {resp.status_code}): {resp.text[:200]}",
                    retryable=False,
                )
                return False
            bot_info = resp.json()
            bot_name = bot_info.get("displayName", "LINE Bot")
            logger.info("[LINE] Connected as %s", bot_name)
        except Exception as e:
            self._release_platform_lock()
            self._set_fatal_error(
                "connection_error",
                f"Failed to verify LINE bot credentials: {e}",
                retryable=True,
            )
            return False

        # Start webhook server
        try:
            self._webhook_app = web.Application()
            self._webhook_app.router.add_post(self.webhook_path, self._handle_webhook)
            self._webhook_app.router.add_get("/line/media/{token}/{filename}", self._handle_media)

            self._webhook_runner = web.AppRunner(self._webhook_app)
            await self._webhook_runner.setup()
            site = web.TCPSite(
                self._webhook_runner,
                self.webhook_host,
                self.webhook_port,
            )
            await site.start()
            logger.info(
                "[LINE] Webhook server listening on %s:%d%s",
                self.webhook_host,
                self.webhook_port,
                self.webhook_path,
            )
        except Exception as e:
            self._release_platform_lock()
            self._set_fatal_error(
                "webhook_error",
                f"Failed to start LINE webhook server: {e}",
                retryable=True,
            )
            return False

        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Stop the webhook server and close HTTP client."""
        if self._webhook_runner:
            try:
                await self._webhook_runner.cleanup()
            except Exception:
                pass
            self._webhook_runner = None
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
        self._release_platform_lock()
        self._mark_disconnected()

    # ------------------------------------------------------------------
    # Webhook handler
    # ------------------------------------------------------------------

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Process incoming LINE webhook events."""
        body = await request.read()

        # Verify signature
        signature = request.headers.get("X-Line-Signature", "")
        if not signature or not _verify_signature(body, signature, self.channel_secret):
            logger.warning("[LINE] Invalid webhook signature")
            return web.Response(status=403, text="Invalid signature")

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")

        events = payload.get("events", [])
        logger.info("[LINE] webhook received: destination=%s events=%d", payload.get("destination", ""), len(events))
        for event in events:
            logger.info("[LINE] webhook event type=%s source_type=%s", event.get("type", ""), event.get("source", {}).get("type", ""))
            asyncio.create_task(self._safe_process_event(event))

        return web.Response(status=200, text="OK")

    async def _safe_process_event(self, event: dict) -> None:
        """Wrapper for _process_event that logs unhandled exceptions."""
        try:
            await self._process_event(event)
        except Exception:
            logger.exception("[LINE] Unhandled error processing event: %s", event.get("type", "?"))

    async def _process_event(self, event: dict) -> None:
        """Process a single LINE webhook event."""
        event_type = event.get("type", "")
        logger.info("[LINE] processing event type=%s", event_type)

        if event_type == "message":
            await self._handle_message_event(event)
        elif event_type == "follow":
            logger.info("[LINE] New follower: %s", event.get("source", {}).get("userId", "unknown"))
        elif event_type == "unfollow":
            logger.info("[LINE] Unfollowed by: %s", event.get("source", {}).get("userId", "unknown"))
        elif event_type == "join":
            logger.info("[LINE] Joined group: %s", event.get("source", {}).get("groupId", "unknown"))
        elif event_type == "leave":
            logger.info("[LINE] Left group: %s", event.get("source", {}).get("groupId", "unknown"))
        elif event_type == "postback":
            # Treat postback data as text message
            data = event.get("postback", {}).get("data", "")
            if data:
                await self._handle_text_message(event, data)

    async def _handle_message_event(self, event: dict) -> None:
        """Handle a LINE message event."""
        message = event.get("message", {})
        msg_type = message.get("type", "")

        if msg_type == "text":
            text = message.get("text", "")
            await self._handle_text_message(event, text)
        elif msg_type in ("image", "video", "audio", "file"):
            await self._handle_media_message(event, msg_type)
        elif msg_type in ("sticker", "location"):
            text = _extract_text_from_event(event)
            if text:
                await self._handle_text_message(event, text)
        else:
            logger.debug("[LINE] Unsupported message type: %s", msg_type)

    async def _handle_text_message(self, event: dict, text: str) -> None:
        """Process a text message and dispatch to the message handler."""
        if not text.strip():
            return

        source = event.get("source", {})
        user_id = source.get("userId", "")
        chat_id = self._resolve_chat_id(source)
        reply_token = event.get("replyToken", "")

        # Build display name
        display_name = await self._get_user_display_name(user_id, source)
        source_obj = self.build_source(
            chat_id=chat_id,
            chat_name=chat_id,
            chat_type="group" if source.get("type") in ("group", "room") else "dm",
            user_id=user_id,
            user_name=display_name,
        )
        message_id = event.get("message", {}).get("id")
        logger.info("[LINE] inbound text: user=%s chat=%s text=%r", display_name, chat_id, text[:200])
        if str(chat_id).startswith("U"):
            asyncio.create_task(self.send_typing(chat_id))

        msg_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source_obj,
            raw_message=event,
            message_id=str(message_id) if message_id else None,
        )
        if reply_token:
            setattr(msg_event, "metadata", {"reply_token": reply_token})
        await self.handle_message(msg_event)

    async def _handle_media_message(self, event: dict, media_type: str) -> None:
        """Download media content and dispatch as a message with attachment."""
        message = event.get("message", {})
        message_id = message.get("id", "")
        source = event.get("source", {})
        user_id = source.get("userId", "")
        chat_id = self._resolve_chat_id(source)
        reply_token = event.get("replyToken", "")
        display_name = await self._get_user_display_name(user_id, source)

        # Download media content
        image_path = None
        text = f"[{media_type.title()}]"
        try:
            client = await self._ensure_client()
            resp = await client.get(
                f"{LINE_DATA_API_BASE}/message/{message_id}/content",
                headers={"Authorization": f"Bearer {self.channel_access_token}"},
            )
            if resp.status_code == 200:
                data = resp.content
                if media_type == "image":
                    content_type = resp.headers.get("content-type", "")
                    ext = ".png" if "png" in content_type else ".jpg"
                    image_path = cache_image_from_bytes(data, ext)
                elif media_type == "audio":
                    image_path = cache_audio_from_bytes(data, ".m4a")
                elif media_type in ("video", "file"):
                    file_name = message.get("fileName", f"file_{message_id}")
                    image_path = cache_document_from_bytes(data, file_name)
        except Exception as e:
            logger.warning("[LINE] Failed to download %s %s: %s", media_type, message_id, e)

        if media_type == "image":
            msg_type = MessageType.PHOTO
        elif media_type == "audio":
            msg_type = MessageType.VOICE
        elif media_type in ("video", "file"):
            msg_type = MessageType.DOCUMENT
        else:
            msg_type = MessageType.TEXT
        source_obj = self.build_source(
            chat_id=chat_id,
            chat_name=chat_id,
            chat_type="group" if source.get("type") in ("group", "room") else "dm",
            user_id=user_id,
            user_name=display_name,
        )
        logger.info("[LINE] inbound media: type=%s user=%s chat=%s", media_type, display_name, chat_id)
        if str(chat_id).startswith("U"):
            asyncio.create_task(self.send_typing(chat_id))

        msg_event = MessageEvent(
            text=text,
            message_type=msg_type,
            source=source_obj,
            raw_message=event,
            message_id=str(message_id) if message_id else None,
            media_urls=[image_path] if image_path else [],
            media_types=[media_type] if image_path else [],
        )
        if reply_token:
            setattr(msg_event, "metadata", {"reply_token": reply_token})
        await self.handle_message(msg_event)

    # ------------------------------------------------------------------
    # Sending messages
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message via LINE Reply API when possible, else Push API."""
        if not content.strip():
            return SendResult(success=True)

        # Strip markdown — LINE does not render it
        text = strip_markdown(content)

        messages = self._build_text_messages(text)
        try:
            client = await self._ensure_client()
            reply_token = metadata.get("reply_token") if metadata else None
            if reply_token:
                logger.info("[LINE] replying to %s via reply API", chat_id)
                resp = await client.post(
                    f"{LINE_API_BASE}/message/reply",
                    headers=self._auth_headers(),
                    json={
                        "replyToken": reply_token,
                        "messages": messages,
                    },
                )
            else:
                logger.info("[LINE] sending push to %s", chat_id)
                resp = await client.post(
                    f"{LINE_API_BASE}/message/push",
                    headers=self._auth_headers(),
                    json={
                        "to": chat_id,
                        "messages": messages,
                    },
                )
            if resp.status_code == 200:
                logger.info("[LINE] send success to %s", chat_id)
                return SendResult(success=True)
            error_body = resp.text[:300]
            logger.warning("[LINE] Send failed (HTTP %d): %s", resp.status_code, error_body)
            return SendResult(success=False, error=f"HTTP {resp.status_code}: {error_body}")
        except Exception as e:
            logger.error("[LINE] Send error: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def send_reply(
        self,
        reply_token: str,
        text: str,
    ) -> SendResult:
        """Send a reply using the LINE Reply API (uses the one-time reply token)."""
        if not text.strip():
            return SendResult(success=True)

        text = strip_markdown(text)
        messages = self._build_text_messages(text)
        try:
            client = await self._ensure_client()
            resp = await client.post(
                f"{LINE_API_BASE}/message/reply",
                headers=self._auth_headers(),
                json={
                    "replyToken": reply_token,
                    "messages": messages,
                },
            )
            if resp.status_code == 200:
                return SendResult(success=True)
            error_body = resp.text[:300]
            logger.warning("[LINE] Reply failed (HTTP %d): %s", resp.status_code, error_body)
            return SendResult(success=False, error=f"HTTP {resp.status_code}: {error_body}")
        except Exception as e:
            logger.error("[LINE] Reply error: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Send LINE loading animation, throttled like OpenClaw's keepalive."""
        # LINE loading animation is mainly for 1:1 chats; quietly skip groups/rooms.
        if not str(chat_id).startswith("U"):
            return

        now = time.monotonic()
        last = self._last_loading_at.get(chat_id, 0.0)
        if now - last < self._loading_interval_seconds:
            return

        try:
            client = await self._ensure_client()
            resp = await client.post(
                LINE_LOADING_API,
                headers=self._auth_headers(),
                json={"chatId": chat_id, "loadingSeconds": self._loading_seconds},
            )
            if resp.status_code in (200, 202):
                self._last_loading_at[chat_id] = now
                logger.info("[LINE] loading animation shown for %s (HTTP %d)", chat_id, resp.status_code)
            else:
                logger.info("[LINE] loading animation not accepted for %s (HTTP %d): %s", chat_id, resp.status_code, resp.text[:200])
        except Exception as e:
            logger.debug("[LINE] Failed to send typing indicator: %s", e)

    # ------------------------------------------------------------------
    # Temporary media serving (required for LINE image messages)
    # ------------------------------------------------------------------

    def _register_media(self, file_path: str, *, cleanup: bool = False) -> str:
        """Register a local file for temporary HTTPS serving; return an opaque token.

        Args:
            file_path: Absolute or relative path to the file to serve.
            cleanup:   If True, the file will be deleted from disk when its
                       token expires (use for internally generated tempfiles).
        """
        now = time.time()
        # Evict expired tokens and clean up any associated tempfiles
        expired = [t for t, (_, exp) in self._media_tokens.items() if now > exp]
        for t in expired:
            path, _ = self._media_tokens.pop(t)
            if path in self._media_temp_paths:
                self._media_temp_paths.discard(path)
                try:
                    os.unlink(path)
                except OSError:
                    pass

        resolved = str(Path(file_path).resolve())
        token = secrets.token_urlsafe(32)
        self._media_tokens[token] = (resolved, now + self._media_ttl)
        if cleanup:
            self._media_temp_paths.add(resolved)
        return token

    def _media_url(self, token: str, filename: str) -> str:
        """Build the public HTTPS URL for a registered media token."""
        host = self.webhook_host
        port = self.webhook_port
        base = f"https://{host}" if port == 443 else f"https://{host}:{port}"
        safe_name = _urlquote(filename, safe="")
        return f"{base}/line/media/{token}/{safe_name}"

    async def _handle_media(self, request: web.Request) -> web.Response:
        """Serve a registered local file over HTTPS for LINE's image API."""
        token = request.match_info["token"]
        entry = self._media_tokens.get(token)
        if not entry:
            raise web.HTTPNotFound()

        file_path, expires_at = entry
        if time.time() > expires_at:
            del self._media_tokens[token]
            raise web.HTTPGone()

        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise web.HTTPNotFound()

        # Defence-in-depth: verify the resolved path is within an allowed root
        # (system temp dir or HERMES_HOME) to guard against any accidental
        # escalation — all paths reach here only via _register_media which is
        # called from internal code, so this is a belt-and-suspenders check.
        resolved = path.resolve()
        tmp_root = Path(tempfile.gettempdir()).resolve()
        from hermes_constants import get_hermes_home
        hermes_root = Path(get_hermes_home()).resolve()
        if not (resolved.is_relative_to(tmp_root) or resolved.is_relative_to(hermes_root)):
            logger.warning("[LINE] Refusing to serve file outside allowed roots: %s", resolved)
            raise web.HTTPForbidden()

        content_type, _ = mimetypes.guess_type(str(path))
        return web.FileResponse(
            path,
            headers={"Content-Type": content_type or "application/octet-stream"},
        )

    # ------------------------------------------------------------------
    # Image sending
    # ------------------------------------------------------------------

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image via LINE Push API.

        ``image_url`` must be a publicly accessible HTTPS URL — LINE's API
        requires ``originalContentUrl`` and ``previewImageUrl`` to be HTTPS.
        For local files use :meth:`send_image_file` instead.
        """
        if not image_url.lower().startswith("https://"):
            logger.error(
                "[LINE] send_image requires an HTTPS URL; got %.80s — use send_image_file for local paths",
                image_url,
            )
            return SendResult(success=False, error="LINE image URLs must use HTTPS")

        messages = []
        if caption:
            messages.append({"type": "text", "text": caption[:MAX_MESSAGE_LENGTH]})
        messages.append({
            "type": "image",
            "originalContentUrl": image_url,
            "previewImageUrl": image_url,
        })
        try:
            client = await self._ensure_client()
            resp = await client.post(
                f"{LINE_API_BASE}/message/push",
                headers=self._auth_headers(),
                json={"to": chat_id, "messages": messages},
            )
            if resp.status_code == 200:
                return SendResult(success=True)
            return SendResult(success=False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            logger.error("[LINE] send_image error: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,  # noqa: ARG002 — LINE Push API has no reply_to for images
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file to LINE as a native image message.

        LINE's API does not accept binary uploads for image messages — it
        requires publicly accessible HTTPS URLs.  This method registers the
        local file on the webhook server under a short-lived token and passes
        the resulting URL to :meth:`send_image`.
        """
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            logger.error("[LINE] send_image_file: file not found: %s", image_path)
            return SendResult(success=False, error=f"File not found: {image_path}")

        file_size = path.stat().st_size
        if file_size > LINE_IMAGE_MAX_BYTES:
            logger.error(
                "[LINE] send_image_file: file too large (%d MB, limit 10 MB): %s",
                file_size // 1024 // 1024,
                image_path,
            )
            return SendResult(
                success=False,
                error=f"Image exceeds LINE's 10 MB limit ({file_size // 1024 // 1024} MB)",
            )

        token = self._register_media(str(path))
        image_url = self._media_url(token, path.name)
        logger.debug("[LINE] serving local image via %s (token expires in %ds)", image_url, self._media_ttl)
        return await self.send_image(
            chat_id=chat_id,
            image_url=image_url,
            caption=caption,
            metadata=metadata,
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,  # noqa: ARG002
        **kwargs,
    ) -> SendResult:
        """Send a local audio file as a LINE native audio message."""
        path = Path(audio_path)
        if not path.exists() or not path.is_file():
            logger.error("[LINE] send_voice: file not found: %s", audio_path)
            return SendResult(success=False, error=f"File not found: {audio_path}")

        file_size = path.stat().st_size
        if file_size > LINE_AUDIO_MAX_BYTES:
            logger.error(
                "[LINE] send_voice: file too large (%d MB, limit 200 MB): %s",
                file_size // 1024 // 1024,
                audio_path,
            )
            return SendResult(
                success=False,
                error=f"Audio exceeds LINE's 200 MB limit ({file_size // 1024 // 1024} MB)",
            )

        token = self._register_media(str(path))
        audio_url = self._media_url(token, path.name)

        messages = []
        if caption:
            messages.append({"type": "text", "text": caption[:MAX_MESSAGE_LENGTH]})
        messages.append({
            "type": "audio",
            "originalContentUrl": audio_url,
            "duration": 0,  # duration (ms) unknown; LINE shows indeterminate progress bar
        })
        try:
            client = await self._ensure_client()
            resp = await client.post(
                f"{LINE_API_BASE}/message/push",
                headers=self._auth_headers(),
                json={"to": chat_id, "messages": messages},
            )
            if resp.status_code == 200:
                return SendResult(success=True)
            return SendResult(success=False, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.error("[LINE] send_voice error: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,  # noqa: ARG002
        **kwargs,
    ) -> SendResult:
        """Send a local video file as a LINE native video message.

        LINE requires ``previewImageUrl`` to be a JPEG/PNG image.  Since we
        have no thumbnail for a local file, we generate a 1×1 white PNG
        placeholder and serve it alongside the video from the webhook server.
        """
        path = Path(video_path)
        if not path.exists() or not path.is_file():
            logger.error("[LINE] send_video: file not found: %s", video_path)
            return SendResult(success=False, error=f"File not found: {video_path}")

        file_size = path.stat().st_size
        if file_size > LINE_VIDEO_MAX_BYTES:
            logger.error(
                "[LINE] send_video: file too large (%d MB, limit 200 MB): %s",
                file_size // 1024 // 1024,
                video_path,
            )
            return SendResult(
                success=False,
                error=f"Video exceeds LINE's 200 MB limit ({file_size // 1024 // 1024} MB)",
            )

        video_token = self._register_media(str(path))
        video_url = self._media_url(video_token, path.name)

        # Generate and serve a preview placeholder PNG (LINE requires an image URL)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_make_preview_png())
            preview_path = f.name
        preview_token = self._register_media(preview_path, cleanup=True)
        preview_url = self._media_url(preview_token, "preview.png")

        messages = []
        if caption:
            messages.append({"type": "text", "text": caption[:MAX_MESSAGE_LENGTH]})
        messages.append({
            "type": "video",
            "originalContentUrl": video_url,
            "previewImageUrl": preview_url,
        })
        try:
            client = await self._ensure_client()
            resp = await client.post(
                f"{LINE_API_BASE}/message/push",
                headers=self._auth_headers(),
                json={"to": chat_id, "messages": messages},
            )
            if resp.status_code == 200:
                return SendResult(success=True)
            return SendResult(success=False, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.error("[LINE] send_video error: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic chat info for the given chat_id."""
        # For groups, try to fetch group summary
        if chat_id.startswith("C"):  # Group IDs start with C
            try:
                client = await self._ensure_client()
                resp = await client.get(
                    f"{LINE_API_BASE}/group/{chat_id}/summary",
                    headers=self._auth_headers(),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "name": data.get("groupName", chat_id),
                        "type": "group",
                        "chat_id": chat_id,
                    }
            except Exception:
                pass
        # For rooms
        if chat_id.startswith("R"):
            return {"name": chat_id, "type": "room", "chat_id": chat_id}
        # DM — try user profile
        profile = await self._get_user_profile(chat_id)
        if profile:
            return {
                "name": profile.get("displayName", chat_id),
                "type": "dm",
                "chat_id": chat_id,
            }
        return {"name": chat_id, "type": "unknown", "chat_id": chat_id}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_chat_id(self, source: dict) -> str:
        """Determine the chat_id from a LINE event source."""
        source_type = source.get("type", "")
        if source_type == "group":
            return source.get("groupId", "")
        if source_type == "room":
            return source.get("roomId", "")
        return source.get("userId", "")

    async def _get_user_display_name(self, user_id: str, source: dict) -> str:
        """Fetch the user's display name, with caching."""
        if user_id in self._user_profile_cache:
            return self._user_profile_cache[user_id].get("displayName", user_id)

        profile = await self._get_user_profile(user_id)
        if profile:
            name = profile.get("displayName", user_id)
            self._user_profile_cache[user_id] = profile
            return name
        return user_id

    async def _get_user_profile(self, user_id: str) -> Optional[Dict[str, str]]:
        """Fetch a LINE user's profile."""
        try:
            client = await self._ensure_client()
            resp = await client.get(
                f"{LINE_API_BASE}/profile/{user_id}",
                headers=self._auth_headers(),
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug("[LINE] Failed to fetch profile for %s: %s", user_id, e)
        return None

    def _build_text_messages(self, text: str) -> list:
        """Split text into LINE message objects respecting the max length."""
        if len(text) <= MAX_MESSAGE_LENGTH:
            return [{"type": "text", "text": text}]
        # Split into chunks
        messages = []
        while text:
            chunk = text[:MAX_MESSAGE_LENGTH]
            text = text[MAX_MESSAGE_LENGTH:]
            messages.append({"type": "text", "text": chunk})
            if len(messages) >= 5:  # LINE allows max 5 messages per request
                if text:
                    logger.warning(
                        "[LINE] Message truncated: %d chars dropped (25000 char limit per request)",
                        len(text),
                    )
                break
        return messages
