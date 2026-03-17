"""SMS (Telnyx) platform adapter.

Connects to the Telnyx REST API for outbound SMS and runs an aiohttp
webhook server to receive inbound messages.

Requires:
  - aiohttp installed: pip install 'hermes-agent[sms]'
  - TELNYX_API_KEY environment variable set
  - TELNYX_FROM_NUMBERS: comma-separated E.164 numbers (e.g. +15551234567)
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

TELNYX_BASE = "https://api.telnyx.com/v2"
MAX_SMS_LENGTH = 1600  # ~10 SMS segments
DEFAULT_WEBHOOK_PORT = 8080

# E.164 phone number pattern for redaction
_PHONE_RE = re.compile(r"\+[1-9]\d{6,14}")


def _redact_phone(phone: str) -> str:
    """Redact a phone number for logging: +15551234567 -> +155****4567."""
    if not phone:
        return "<none>"
    if len(phone) <= 8:
        return phone[:2] + "****" + phone[-2:] if len(phone) > 4 else "****"
    return phone[:4] + "****" + phone[-4:]


def _parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a list, stripping whitespace."""
    return [v.strip() for v in value.split(",") if v.strip()]


def check_sms_requirements() -> bool:
    """Check if SMS adapter dependencies are available."""
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return False
    return bool(os.getenv("TELNYX_API_KEY"))


class SmsAdapter(BasePlatformAdapter):
    """
    Telnyx SMS <-> Hermes gateway adapter.

    Each inbound phone number gets its own Hermes session (multi-tenant).
    Tracks which owned number received each user's message to reply from
    the same number.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SMS)
        self._api_key: str = os.environ["TELNYX_API_KEY"]
        self._webhook_port: int = int(
            os.getenv("SMS_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT))
        )
        # Set of owned numbers
        self._from_numbers: set = set(
            _parse_comma_list(os.getenv("TELNYX_FROM_NUMBERS", ""))
        )
        # Runtime map: user phone -> which owned number to reply from
        self._reply_from: Dict[str, str] = {}
        self._runner = None

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        import aiohttp
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/webhooks/telnyx", self._handle_webhook)
        app.router.add_get("/health", lambda _: web.Response(text="ok"))

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._webhook_port)
        await site.start()
        self._running = True

        from_display = ", ".join(_redact_phone(n) for n in self._from_numbers) or "(none)"
        logger.info(
            "[sms] Webhook server listening on port %d, from numbers: %s",
            self._webhook_port,
            from_display,
        )
        return True

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._running = False
        logger.info("[sms] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        import aiohttp

        from_number = self._get_reply_from(chat_id, metadata)
        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted)
        last_result = SendResult(success=True)

        async with aiohttp.ClientSession() as session:
            for i, chunk in enumerate(chunks):
                payload = {"from": from_number, "to": chat_id, "text": chunk}
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }
                try:
                    async with session.post(
                        f"{TELNYX_BASE}/messages",
                        json=payload,
                        headers=headers,
                    ) as resp:
                        body = await resp.json()
                        if resp.status >= 400:
                            logger.error(
                                "[sms] send failed %s: %s %s",
                                _redact_phone(chat_id),
                                resp.status,
                                body,
                            )
                            return SendResult(
                                success=False,
                                error=f"Telnyx {resp.status}: {body}",
                            )
                        msg_id = body.get("data", {}).get("id", "")
                        last_result = SendResult(success=True, message_id=msg_id)
                except Exception as e:
                    logger.error("[sms] send error %s: %s", _redact_phone(chat_id), e)
                    return SendResult(success=False, error=str(e))

        return last_result

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

    # ------------------------------------------------------------------
    # SMS-specific formatting
    # ------------------------------------------------------------------

    def format_message(self, content: str) -> str:
        """Strip markdown -- SMS renders it as literal characters."""
        content = re.sub(r"\*\*(.+?)\*\*", r"\1", content, flags=re.DOTALL)
        content = re.sub(r"\*(.+?)\*", r"\1", content, flags=re.DOTALL)
        content = re.sub(r"__(.+?)__", r"\1", content, flags=re.DOTALL)
        content = re.sub(r"_(.+?)_", r"\1", content, flags=re.DOTALL)
        content = re.sub(r"```[a-z]*\n?", "", content)
        content = re.sub(r"`(.+?)`", r"\1", content)
        content = re.sub(r"^#{1,6}\s+", "", content, flags=re.MULTILINE)
        content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()

    def truncate_message(
        self, content: str, max_length: int = MAX_SMS_LENGTH
    ) -> List[str]:
        """Split into <=1600-char chunks (10 SMS segments)."""
        if len(content) <= max_length:
            return [content]
        chunks: List[str] = []
        while content:
            if len(content) <= max_length:
                chunks.append(content)
                break
            split_at = content.rfind("\n", 0, max_length)
            if split_at < max_length // 2:
                split_at = content.rfind(" ", 0, max_length)
            if split_at < 1:
                split_at = max_length
            chunks.append(content[:split_at].strip())
            content = content[split_at:].strip()
        return chunks

    # ------------------------------------------------------------------
    # Telnyx webhook handler
    # ------------------------------------------------------------------

    async def _handle_webhook(self, request) -> "aiohttp.web.Response":
        from aiohttp import web

        try:
            raw = await request.read()
            body = json.loads(raw.decode("utf-8"))
        except Exception as e:
            logger.error("[sms] webhook parse error: %s", e)
            return web.json_response({"error": "invalid json"}, status=400)

        # Only handle inbound messages
        if body.get("data", {}).get("event_type") != "message.received":
            return web.json_response({"received": True})

        payload = body["data"]["payload"]
        from_number: str = payload.get("from", {}).get("phone_number", "")
        to_list = payload.get("to", [])
        to_number: str = to_list[0].get("phone_number", "") if to_list else ""
        text: str = payload.get("text", "").strip()

        if not from_number or not text:
            return web.json_response({"received": True})

        # Ignore messages sent FROM one of our own numbers (echo loop prevention)
        if from_number in self._from_numbers:
            logger.debug("[sms] ignoring echo from own number %s", _redact_phone(from_number))
            return web.json_response({"received": True})

        # Remember which owned number received this user's message
        if to_number and to_number in self._from_numbers:
            self._reply_from[from_number] = to_number

        logger.info(
            "[sms] inbound from %s -> %s: %s",
            _redact_phone(from_number),
            _redact_phone(to_number),
            text[:80],
        )

        source = self.build_source(
            chat_id=from_number,
            chat_name=from_number,
            chat_type="dm",
            user_id=from_number,
            user_name=from_number,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=body,
            message_id=payload.get("id"),
        )

        # Non-blocking: Telnyx expects a fast 200
        asyncio.create_task(self.handle_message(event))
        return web.json_response({"received": True})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_reply_from(
        self, user_phone: str, metadata: Optional[Dict] = None
    ) -> str:
        """Determine which owned number to send from."""
        if metadata and "from_number" in metadata:
            return metadata["from_number"]
        if user_phone in self._reply_from:
            return self._reply_from[user_phone]
        if self._from_numbers:
            return next(iter(self._from_numbers))
        raise RuntimeError(
            "No FROM number configured (TELNYX_FROM_NUMBERS) and no prior "
            "reply_from mapping for this user"
        )
