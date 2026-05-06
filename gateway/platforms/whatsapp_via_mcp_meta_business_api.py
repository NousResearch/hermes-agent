"""WhatsApp via external MCP (Meta Business Cloud API).

Receives webhook POSTs forwarded by an external WhatsApp MCP server that owns
the Meta Cloud API credentials (access token, phone number ID). When a message
arrives, the platform spawns a Hermes session keyed by sender phone, processes
the message with full agent capabilities (memory, skills, MCPs), and sends the
reply back via Meta Graph API directly.

Why this exists: the upstream ``whatsapp.py`` platform uses the Baileys Node
bridge (reverse-engineered WhatsApp Web), which violates Meta's terms of
service and is fragile on Windows. Users who run a phone number on the official
Meta WhatsApp Business Cloud API have no first-class platform in Hermes today.
This module fills that gap by delegating Meta-side concerns (webhook ingress,
outbound delivery, sender tagging, message persistence) to an external MCP
server, while keeping agent reasoning inside Hermes.

Architecture::

    Meta Cloud
        v webhook
    [external WA-MCP server]
        v POST {message_id, phone, type, content, tags, raw}
    [this platform]                    -> handle_message(event)
        ^                                       v
        | reply via Meta Graph /messages   <-  send()

The forward POST is validated against
``WHATSAPP_VIA_MCP_META_BUSINESS_API_WEBHOOK_SECRET`` via the
``X-Webhook-Secret`` header (constant-time comparison). The platform returns
``200 {"ok": true, "queued": true}`` immediately so the MCP's typical
short forward timeout (Meta itself requires <5s on the original webhook) is
not exceeded by LLM latency; the actual reply is sent asynchronously by
``send()`` once Hermes finishes processing.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from typing import Any, Dict, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

META_GRAPH_BASE_URL = "https://graph.facebook.com/v21.0"
MAX_MESSAGE_LENGTH = 4096  # WhatsApp text body limit per Meta docs


def check_whatsapp_via_mcp_meta_business_api_requirements() -> bool:
    """Return True if the platform's runtime dependencies are available."""
    return AIOHTTP_AVAILABLE and HTTPX_AVAILABLE


def _redact_phone(phone: str) -> str:
    """Mask a phone number for log output (keep first 3 and last 2 digits)."""
    if not phone or len(phone) < 6:
        return "***"
    return f"{phone[:3]}***{phone[-2:]}"


class WhatsAppViaMcpMetaBusinessApiAdapter(BasePlatformAdapter):
    """Receive forwarded WhatsApp webhooks from an external MCP, reply via Meta Cloud API.

    Configuration (via :class:`PlatformConfig`):

    * ``token`` (required): Meta Cloud API access token (System User token
      with ``whatsapp_business_messaging`` permission). Loaded from
      ``WHATSAPP_VIA_MCP_META_BUSINESS_API_TOKEN`` env var by
      ``_apply_env_overrides``.
    * ``extra.phone_number_id`` (required): the phone number id used as
      ``{phone-number-id}`` in the Meta Graph API path. Env:
      ``WHATSAPP_VIA_MCP_META_BUSINESS_API_PHONE_NUMBER_ID``.
    * ``extra.webhook_secret`` (recommended): shared secret with the WA-MCP;
      validated as the ``X-Webhook-Secret`` request header on every forward.
      Env: ``WHATSAPP_VIA_MCP_META_BUSINESS_API_WEBHOOK_SECRET``. If unset,
      forwards are accepted unauthenticated and a warning is emitted at startup.
    * ``extra.host`` (default ``0.0.0.0``): aiohttp bind host.
    * ``extra.port`` (default ``8643``): aiohttp bind port.
    * ``extra.path`` (default ``/wa``): URL path for the forward endpoint.
    * ``extra.meta_base_url`` (default ``https://graph.facebook.com/v21.0``):
      override for testing.
    """

    PLATFORM_NAME = "whatsapp_via_mcp_meta_business_api"

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WHATSAPP_VIA_MCP_META_BUSINESS_API)
        self._token: str = config.token or ""
        extra = config.extra or {}
        self._phone_number_id: str = extra.get("phone_number_id") or os.getenv(
            "WHATSAPP_VIA_MCP_META_BUSINESS_API_PHONE_NUMBER_ID", ""
        )
        self._secret: str = extra.get("webhook_secret") or os.getenv(
            "WHATSAPP_VIA_MCP_META_BUSINESS_API_WEBHOOK_SECRET", ""
        )
        self._host: str = extra.get("host") or os.getenv(
            "WHATSAPP_VIA_MCP_META_BUSINESS_API_HOST", "0.0.0.0"
        )
        self._port: int = int(
            extra.get("port") or os.getenv("WHATSAPP_VIA_MCP_META_BUSINESS_API_PORT", "8643")
        )
        self._path: str = extra.get("path") or os.getenv(
            "WHATSAPP_VIA_MCP_META_BUSINESS_API_PATH", "/wa"
        )
        self._meta_base_url: str = (
            extra.get("meta_base_url")
            or os.getenv("WHATSAPP_VIA_MCP_META_BUSINESS_API_META_BASE_URL")
            or META_GRAPH_BASE_URL
        )
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._http_client: Optional["httpx.AsyncClient"] = None

    # ------------------------------------------------------------------ lifecycle

    async def connect(self) -> bool:
        if not self._token:
            logger.error(
                "WhatsApp via MCP: no token configured (WHATSAPP_VIA_MCP_META_BUSINESS_API_TOKEN)"
            )
            return False
        if not self._phone_number_id:
            logger.error(
                "WhatsApp via MCP: no phone_number_id configured "
                "(WHATSAPP_VIA_MCP_META_BUSINESS_API_PHONE_NUMBER_ID)"
            )
            return False
        if not self._secret:
            logger.warning(
                "WhatsApp via MCP: no webhook secret configured — forwards will be accepted "
                "without authentication. Set WHATSAPP_VIA_MCP_META_BUSINESS_API_WEBHOOK_SECRET."
            )

        self._http_client = httpx.AsyncClient(timeout=30)

        self._app = web.Application()
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_post(self._path, self._handle_webhook)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        logger.info(
            "WhatsApp via MCP listening on %s:%s%s (phone_id=%s)",
            self._host,
            self._port,
            self._path,
            self._phone_number_id,
        )
        return True

    async def disconnect(self) -> None:
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("WhatsApp via MCP disconnected")

    # ------------------------------------------------------------------ inbound

    async def _handle_health(self, _request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "ok": True,
                "platform": self.PLATFORM_NAME,
                "phone_number_id": self._phone_number_id,
            }
        )

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        body = await request.read()
        if self._secret:
            provided = request.headers.get("X-Webhook-Secret", "")
            if not hmac.compare_digest(provided, self._secret):
                logger.warning(
                    "WhatsApp via MCP: invalid webhook secret from %s", request.remote
                )
                return web.json_response({"error": "invalid secret"}, status=403)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid json"}, status=400)

        # The wa-mcp router posts:
        #   {message_id, phone, type, content, tags, raw}
        phone = (payload.get("phone") or payload.get("from") or "").strip()
        content = (payload.get("content") or payload.get("text") or "").strip()
        message_id = payload.get("message_id") or ""
        msg_type = payload.get("type") or "text"

        if not phone:
            return web.json_response({"error": "missing phone"}, status=400)
        if msg_type != "text":
            logger.info(
                "WhatsApp via MCP: skipping non-text message type=%s phone=%s msg=%s",
                msg_type,
                _redact_phone(phone),
                message_id,
            )
            return web.json_response({"ok": True, "skipped": "non_text"})
        if not content:
            return web.json_response({"ok": True, "skipped": "empty"})

        # ELIGIA HITL receive-side moved to waba-mcp `payload_routes` →
        # aiops `/webhook/hitl` (eligia-core PR-1..3, 2026-05-05). Hermes
        # no longer sees HITL clicks because button payloads with
        # `interactive.button_reply.id` starting with `hitl:` are short-
        # circuited at waba-mcp before reaching the personal_hermes tag
        # forward. Free-form text replies are no longer interpreted as
        # approvals (the implicit regex was a security smell — any "aprobar"
        # in casual chat could resolve).

        source = self.build_source(
            chat_id=phone,
            user_id=phone,
            chat_type="dm",
        )
        event = MessageEvent(
            source=source,
            message_type=MessageType.TEXT,
            text=content,
            message_id=message_id,
        )
        # Process asynchronously: the WA-MCP applies a short timeout to the
        # forward POST (Meta requires the original webhook handler to reply
        # in <5 s, and the MCP cascades that constraint). We acknowledge
        # immediately and let send() deliver the reply when Hermes finishes.
        asyncio.create_task(self.handle_message(event))
        logger.info(
            "WhatsApp via MCP queued msg=%s phone=%s len=%d",
            message_id,
            _redact_phone(phone),
            len(content),
        )
        return web.json_response({"ok": True, "queued": True})

    # ------------------------------------------------------------------ outbound

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message via Meta Graph API.

        Long messages are split into ``MAX_MESSAGE_LENGTH``-sized chunks and
        sent sequentially.
        """
        if self._http_client is None:
            return SendResult(success=False, error="adapter not connected")

        chunks = (
            [content[i : i + MAX_MESSAGE_LENGTH] for i in range(0, len(content), MAX_MESSAGE_LENGTH)]
            or [""]
        )
        last_id = ""
        for chunk in chunks:
            result = await self._post_meta(
                {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": chat_id,
                    "type": "text",
                    "text": {"body": chunk},
                }
            )
            if not result.success:
                return result
            last_id = result.message_id or last_id
        return SendResult(success=True, message_id=last_id)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        # Meta Cloud API does not expose a typing indicator at the message level.
        # Accept the metadata kwarg to stay compatible with BasePlatformAdapter._keep_typing.
        return None

    async def stop_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        return None

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send an image by public URL.

        Meta accepts either a public URL (``image.link``) or a media id from
        a previous upload to ``/{phone-number-id}/media``. This implementation
        uses ``link`` for simplicity; uploaded media flow can be added later.
        """
        if self._http_client is None:
            return SendResult(success=False, error="adapter not connected")
        payload: Dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": chat_id,
            "type": "image",
            "image": {"link": image_url},
        }
        if caption:
            payload["image"]["caption"] = caption[:1024]
        return await self._post_meta(payload)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        # WhatsApp chats are 1-to-1 keyed by phone; Meta's API does not expose
        # the profile display name without prior message metadata.
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}

    # ------------------------------------------------------------------ internals

    async def _post_meta(self, payload: Dict[str, Any]) -> SendResult:
        assert self._http_client is not None
        url = f"{self._meta_base_url}/{self._phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        try:
            r = await self._http_client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as e:
            logger.error(
                "Meta send failed to %s: %s",
                _redact_phone(payload.get("to", "")),
                e,
            )
            return SendResult(success=False, error=str(e))
        if r.status_code != 200:
            logger.error(
                "Meta send %d to %s: %s",
                r.status_code,
                _redact_phone(payload.get("to", "")),
                r.text[:300],
            )
            return SendResult(
                success=False, error=f"http_{r.status_code}: {r.text[:200]}"
            )
        try:
            mid = r.json().get("messages", [{}])[0].get("id", "") or ""
        except Exception:
            mid = ""
        return SendResult(success=True, message_id=mid)

