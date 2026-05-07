"""
LINE Messaging API platform adapter for Hermes Agent.

Logic ported from the OpenClaw LINE channel extension
(``~/openclaw/extensions/line``):

* aiohttp webhook server at ``/line/webhook``
* HMAC-SHA256 + base64 signature verification of the raw request body
  (``X-Line-Signature`` header), using a constant-time comparison
* outbound sends via the LINE Messaging API; reply token is preferred
  (one-shot, ~1 minute validity) with automatic fallback to push
* group / room / 1:1 chat type detection from the source ID prefix
* image / sticker handling (image attachments are fetched from the
  ``api-data.line.me`` content endpoint and cached locally)

Configuration in ``config.yaml``::

    gateway:
      platforms:
        line:
          enabled: true
          extra:
            channel_access_token: "..."   # or LINE_CHANNEL_ACCESS_TOKEN
            channel_secret: "..."         # or LINE_CHANNEL_SECRET
            port: 3979                    # or LINE_PORT
            webhook_path: "/line/webhook" # default
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from aiohttp import ClientSession, web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]
    ClientSession = None  # type: ignore[assignment,misc]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_bytes,
)
from gateway.platforms.helpers import MessageDeduplicator

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 3979
_DEFAULT_WEBHOOK_PATH = "/line/webhook"
_MAX_RAW_BODY_BYTES = 64 * 1024
_REPLY_TOKEN_TTL_SECONDS = 50  # LINE reply tokens are valid ~1 minute
_API_BASE = "https://api.line.me/v2/bot"
_DATA_API_BASE = "https://api-data.line.me/v2/bot"


# ── Signature verification ────────────────────────────────────────────────────


def _validate_line_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    """Constant-time HMAC-SHA256 (base64) signature check."""
    if not signature or not channel_secret:
        return False
    digest = hmac.new(
        channel_secret.encode("utf-8"), body, hashlib.sha256
    ).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature)


# ── Plugin-level helpers ──────────────────────────────────────────────────────


def check_requirements() -> bool:
    return AIOHTTP_AVAILABLE


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or extra.get(
        "channel_access_token", ""
    )
    secret = os.getenv("LINE_CHANNEL_SECRET") or extra.get("channel_secret", "")
    return bool(token and secret)


def is_connected(config) -> bool:
    return validate_config(config)


# ── Adapter ───────────────────────────────────────────────────────────────────


class LineAdapter(BasePlatformAdapter):
    """LINE Messaging API adapter (webhook + push)."""

    # Per LINE docs: text messages capped at 5000 characters.
    MAX_MESSAGE_LENGTH = 4500

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("line"))
        extra = config.extra or {}
        self._token = extra.get("channel_access_token") or os.getenv(
            "LINE_CHANNEL_ACCESS_TOKEN", ""
        )
        self._secret = extra.get("channel_secret") or os.getenv(
            "LINE_CHANNEL_SECRET", ""
        )
        self._port = int(extra.get("port") or os.getenv("LINE_PORT", str(_DEFAULT_PORT)))
        self._webhook_path = extra.get("webhook_path") or _DEFAULT_WEBHOOK_PATH
        self._runner: Optional["web.AppRunner"] = None
        self._http: Optional["ClientSession"] = None
        self._dedup = MessageDeduplicator(max_size=1000)
        # chat_id → (reply_token, captured_at_unix)
        self._reply_tokens: Dict[str, Tuple[str, float]] = {}
        self._bot_user_id: Optional[str] = None

    # -- lifecycle ----------------------------------------------------------

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            self._set_fatal_error(
                "MISSING_SDK",
                "aiohttp not installed. Run: pip install aiohttp",
                retryable=False,
            )
            return False
        if not self._token or not self._secret:
            self._set_fatal_error(
                "MISSING_CREDENTIALS",
                "LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET are both required",
                retryable=False,
            )
            return False

        try:
            self._http = ClientSession()

            # Best-effort fetch of bot's own userId for self-message filtering.
            try:
                async with self._http.get(
                    f"{_API_BASE}/info",
                    headers={"Authorization": f"Bearer {self._token}"},
                ) as resp:
                    if resp.status == 200:
                        info = await resp.json()
                        self._bot_user_id = info.get("userId")
            except Exception as e:
                logger.debug("[line] bot info fetch failed: %s", e)

            app = web.Application()
            app.router.add_post(self._webhook_path, self._handle_webhook)
            app.router.add_get("/health", lambda _: web.Response(text="ok"))

            self._runner = web.AppRunner(app)
            await self._runner.setup()
            site = web.TCPSite(self._runner, "0.0.0.0", self._port)
            await site.start()

            self._running = True
            self._mark_connected()
            logger.info(
                "[line] Webhook server listening on 0.0.0.0:%d%s",
                self._port,
                self._webhook_path,
            )
            return True
        except Exception as e:
            self._set_fatal_error(
                "CONNECT_FAILED",
                f"LINE connection failed: {e}",
                retryable=True,
            )
            logger.error("[line] Failed to connect: %s", e)
            return False

    async def disconnect(self) -> None:
        self._running = False
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        if self._http:
            await self._http.close()
            self._http = None
        self._mark_disconnected()
        logger.info("[line] Disconnected")

    # -- webhook ------------------------------------------------------------

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        signature = request.headers.get("X-Line-Signature", "")
        if not signature:
            return web.json_response({"error": "Missing X-Line-Signature"}, status=400)

        raw = await request.read()
        if len(raw) > _MAX_RAW_BODY_BYTES:
            return web.json_response({"error": "Payload too large"}, status=413)

        if not _validate_line_signature(raw, signature, self._secret):
            logger.warning("[line] webhook signature validation failed")
            return web.json_response({"error": "Invalid signature"}, status=401)

        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        events = payload.get("events") or []
        for event in events:
            try:
                await self._dispatch_event(event)
            except Exception as e:
                logger.error("[line] error handling event: %s", e, exc_info=True)

        return web.json_response({"status": "ok"})

    async def _dispatch_event(self, event: Dict[str, Any]) -> None:
        if event.get("type") != "message":
            return

        message = event.get("message") or {}
        msg_id = message.get("id") or event.get("webhookEventId")
        if msg_id and self._dedup.is_duplicate(str(msg_id)):
            return

        source = event.get("source") or {}
        chat_id, chat_type = _resolve_chat(source)
        if not chat_id:
            return

        user_id = source.get("userId") or ""
        if user_id and self._bot_user_id and user_id == self._bot_user_id:
            return  # self-message echo

        # Cache the reply token so the next outbound send can use it.
        reply_token = event.get("replyToken")
        if reply_token:
            self._reply_tokens[chat_id] = (reply_token, time.time())

        text = ""
        media_urls: List[str] = []
        media_types: List[str] = []
        msg_type = MessageType.TEXT
        kind = message.get("type")

        if kind == "text":
            text = message.get("text", "") or ""
        elif kind == "image" and msg_id:
            data = await self._fetch_message_content(str(msg_id))
            if data:
                cached = await cache_image_from_bytes(data, "image/jpeg")
                if cached:
                    media_urls.append(cached)
                    media_types.append("image/jpeg")
                    msg_type = MessageType.PHOTO
        elif kind == "sticker":
            text = "[sticker]"
        else:
            # Other message kinds (video, audio, location, file...) are
            # forwarded as a marker so the agent can acknowledge.
            text = f"[{kind or 'unknown'} message]"

        event_obj = MessageEvent(
            text=text,
            source=self.build_source(
                chat_id=chat_id,
                chat_type=chat_type,
                user_id=user_id or None,
                user_name=None,
                message_id=str(msg_id) if msg_id else None,
            ),
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            message_id=str(msg_id) if msg_id else None,
        )
        await self.handle_message(event_obj)

    async def _fetch_message_content(self, message_id: str) -> Optional[bytes]:
        if not self._http:
            return None
        url = f"{_DATA_API_BASE}/message/{message_id}/content"
        try:
            async with self._http.get(
                url, headers={"Authorization": f"Bearer {self._token}"}
            ) as resp:
                if resp.status != 200:
                    logger.debug(
                        "[line] content fetch %s returned %s", message_id, resp.status
                    )
                    return None
                return await resp.read()
        except Exception as e:
            logger.warning("[line] content fetch failed: %s", e)
            return None

    # -- send ---------------------------------------------------------------

    def _consume_reply_token(self, chat_id: str) -> Optional[str]:
        entry = self._reply_tokens.pop(chat_id, None)
        if not entry:
            return None
        token, captured = entry
        if time.time() - captured > _REPLY_TOKEN_TTL_SECONDS:
            return None
        return token

    async def _post(self, path: str, payload: Dict[str, Any]) -> Tuple[bool, str]:
        if not self._http:
            return False, "LINE adapter not initialized"
        url = f"{_API_BASE}{path}"
        try:
            async with self._http.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
            ) as resp:
                if 200 <= resp.status < 300:
                    return True, ""
                body = await resp.text()
                return False, f"HTTP {resp.status}: {body[:300]}"
        except Exception as e:
            return False, str(e)

    async def _send_messages(
        self, chat_id: str, messages: List[Dict[str, Any]]
    ) -> SendResult:
        # LINE accepts at most 5 messages per push/reply call.
        for i in range(0, len(messages), 5):
            batch = messages[i : i + 5]
            reply_token = self._consume_reply_token(chat_id) if i == 0 else None
            if reply_token:
                ok, err = await self._post(
                    "/message/reply",
                    {"replyToken": reply_token, "messages": batch},
                )
                if not ok:
                    # Reply token may have already been consumed by another
                    # request, or expired between capture and use — fall back
                    # to push.
                    logger.debug("[line] reply failed (%s); falling back to push", err)
                    ok, err = await self._post(
                        "/message/push", {"to": chat_id, "messages": batch}
                    )
            else:
                ok, err = await self._post(
                    "/message/push", {"to": chat_id, "messages": batch}
                )
            if not ok:
                return SendResult(success=False, error=err, retryable=True)
        return SendResult(success=True)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, max_length=self.MAX_MESSAGE_LENGTH)
        messages = [{"type": "text", "text": chunk} for chunk in chunks if chunk]
        if not messages:
            return SendResult(success=True)
        return await self._send_messages(chat_id, messages)

    async def send_typing(
        self, chat_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        # LINE's "loading animation" API is the closest analog to a typing
        # indicator.  It is only valid for 1:1 user chats — group/room IDs
        # are rejected with HTTP 400.  ``loadingSeconds`` accepts 5..60 in
        # multiples of 5; the animation auto-clears as soon as the bot
        # sends a real message (or when the timer runs out).
        if not self._http or not chat_id:
            return None
        if not chat_id.startswith("U"):
            return None  # group/room — API does not support these
        try:
            ok, err = await self._post(
                "/chat/loading/start",
                {"chatId": chat_id, "loadingSeconds": 20},
            )
            if not ok:
                logger.debug("[line] loading animation start failed: %s", err)
        except Exception as e:
            logger.debug("[line] loading animation start raised: %s", e)
        return None

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not (image_url.startswith("http://") or image_url.startswith("https://")):
            return SendResult(
                success=False,
                error="LINE image messages require a public HTTPS URL",
                retryable=False,
            )
        messages: List[Dict[str, Any]] = [
            {
                "type": "image",
                "originalContentUrl": image_url,
                "previewImageUrl": image_url,
            }
        ]
        if caption:
            messages.append({"type": "text", "text": caption})
        return await self._send_messages(chat_id, messages)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        # LINE's image message requires a public URL — local files cannot be
        # uploaded directly.  Callers should host the file first.
        return SendResult(
            success=False,
            error="LINE does not accept inline image uploads — host the file and call send_image() with a public URL",
            retryable=False,
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_type = "dm"
        if chat_id.startswith("C"):
            chat_type = "group"
        elif chat_id.startswith("R"):
            chat_type = "group"  # multi-person room — closest analog
        return {"name": chat_id, "type": chat_type, "chat_id": chat_id}


def _resolve_chat(source: Dict[str, Any]) -> Tuple[str, str]:
    """Return (chat_id, chat_type) for a LINE event source object."""
    src_type = source.get("type")
    if src_type == "group":
        return source.get("groupId", ""), "group"
    if src_type == "room":
        return source.get("roomId", ""), "group"
    if src_type == "user":
        return source.get("userId", ""), "dm"
    return "", "dm"


# ── Interactive setup ─────────────────────────────────────────────────────────


def interactive_setup() -> None:
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli.cli_output import (
        print_info,
        print_success,
        print_warning,
        prompt,
        prompt_yes_no,
    )

    if get_env_value("LINE_CHANNEL_ACCESS_TOKEN"):
        print_info("LINE: already configured")
        if not prompt_yes_no("Reconfigure LINE?", False):
            return

    print_info("Create a Messaging API channel at https://developers.line.biz/console/")
    print_info("From the channel page you need:")
    print_info("  • Channel access token (long-lived) — under the Messaging API tab")
    print_info("  • Channel secret — under the Basic settings tab")
    print()
    print_info("Then expose your webhook port publicly (devtunnel / ngrok / cloudflared)")
    print_info("and set the channel's webhook URL to:  https://<tunnel>/line/webhook")
    print()

    token = prompt(
        "Channel access token",
        default=get_env_value("LINE_CHANNEL_ACCESS_TOKEN") or "",
        password=True,
    )
    if not token:
        print_warning("Channel access token is required — skipping LINE setup")
        return
    save_env_value("LINE_CHANNEL_ACCESS_TOKEN", token.strip())

    secret = prompt(
        "Channel secret",
        default=get_env_value("LINE_CHANNEL_SECRET") or "",
        password=True,
    )
    if not secret:
        print_warning("Channel secret is required — skipping LINE setup")
        return
    save_env_value("LINE_CHANNEL_SECRET", secret.strip())

    if prompt_yes_no("Restrict access to specific LINE user IDs? (recommended)", True):
        allowed = prompt(
            "Allowed LINE user IDs (comma-separated, format Uxxxx...)",
            default=get_env_value("LINE_ALLOWED_USERS") or "",
        )
        if allowed:
            save_env_value("LINE_ALLOWED_USERS", allowed.replace(" ", ""))
            print_success("Allowlist configured")
        else:
            save_env_value("LINE_ALLOWED_USERS", "")
    else:
        save_env_value("LINE_ALLOW_ALL_USERS", "true")
        print_warning("⚠️  Open access — anyone who messages the bot can command it.")

    print()
    print_success("LINE configuration saved to ~/.hermes/.env")
    print_info("Default webhook port: %d  (override with LINE_PORT)" % _DEFAULT_PORT)
    print_info("Restart the gateway:  hermes gateway restart")


# ── Plugin entry point ────────────────────────────────────────────────────────


def register(ctx) -> None:
    ctx.register_platform(
        name="line",
        label="LINE",
        adapter_factory=lambda cfg: LineAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["LINE_CHANNEL_ACCESS_TOKEN", "LINE_CHANNEL_SECRET"],
        install_hint="pip install aiohttp",
        setup_fn=interactive_setup,
        allowed_users_env="LINE_ALLOWED_USERS",
        allow_all_env="LINE_ALLOW_ALL_USERS",
        max_message_length=4500,
        emoji="💬",
        allow_update_command=True,
        platform_hint=(
            "You are chatting via LINE Messaging API. LINE renders plain "
            "text only — markdown, HTML, and code fences are not rendered, "
            "so prefer concise prose. Long replies are split into multiple "
            "bubbles. Image sends require a publicly-reachable HTTPS URL."
        ),
    )
