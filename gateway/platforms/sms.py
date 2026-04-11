"""SMS (Twilio) platform adapter.

Connects to the Twilio REST API for outbound SMS and runs an aiohttp
webhook server to receive inbound messages.

Shares credentials with the optional telephony skill — same env vars:
  - TWILIO_ACCOUNT_SID
  - TWILIO_AUTH_TOKEN
  - TWILIO_PHONE_NUMBER  (E.164 from-number, e.g. +15551234567)

Gateway-specific env vars:
  - SMS_WEBHOOK_PORT     (default 8080)
  - SMS_WEBHOOK_HOST     (default 0.0.0.0)
  - SMS_WEBHOOK_URL      (public URL for Twilio signature validation)
  - SMS_ALLOWED_USERS    (comma-separated E.164 phone numbers)
  - SMS_ALLOW_ALL_USERS  (true/false)
  - SMS_HOME_CHANNEL     (phone number for cron delivery)
"""

import asyncio
import base64
import hashlib
import hmac
import logging
import os
import urllib.parse
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    is_network_accessible,
)
from gateway.platforms.helpers import redact_phone, strip_markdown

logger = logging.getLogger(__name__)

TWILIO_API_BASE = "https://api.twilio.com/2010-04-01/Accounts"
MAX_SMS_LENGTH = 1600  # ~10 SMS segments
DEFAULT_WEBHOOK_PORT = 8080
DEFAULT_WEBHOOK_HOST = "0.0.0.0"


def check_sms_requirements() -> bool:
    """Check if SMS adapter dependencies are available."""
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return False
    return bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"))


class SmsAdapter(BasePlatformAdapter):
    """
    Twilio SMS <-> Hermes gateway adapter.

    Each inbound phone number gets its own Hermes session (multi-tenant).
    Replies are always sent from the configured TWILIO_PHONE_NUMBER.
    """

    MAX_MESSAGE_LENGTH = MAX_SMS_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SMS)
        self._account_sid: str = os.environ["TWILIO_ACCOUNT_SID"]
        self._auth_token: str = os.environ["TWILIO_AUTH_TOKEN"]
        self._from_number: str = os.getenv("TWILIO_PHONE_NUMBER", "")
        self._webhook_port: int = int(
            os.getenv("SMS_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT))
        )
        self._webhook_host: str = os.getenv("SMS_WEBHOOK_HOST", DEFAULT_WEBHOOK_HOST)
        self._webhook_url: str = os.getenv("SMS_WEBHOOK_URL", "").strip()
        self._runner = None
        self._http_session: Optional["aiohttp.ClientSession"] = None

    def _basic_auth_header(self) -> str:
        """Build HTTP Basic auth header value for Twilio."""
        creds = f"{self._account_sid}:{self._auth_token}"
        encoded = base64.b64encode(creds.encode("ascii")).decode("ascii")
        return f"Basic {encoded}"

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        import aiohttp
        from aiohttp import web

        if not self._from_number:
            logger.error("[sms] TWILIO_PHONE_NUMBER not set — cannot send replies")
            return False

        if not self._webhook_url:
            logger.warning(
                "[sms] SMS_WEBHOOK_URL not set — Twilio signature validation is "
                "DISABLED. Any client that can reach port %d can inject messages. "
                "Set SMS_WEBHOOK_URL to enable signature validation.",
                self._webhook_port,
            )

        app = web.Application()
        app.router.add_post("/webhooks/twilio", self._handle_webhook)
        app.router.add_get("/health", lambda _: web.Response(text="ok"))

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._webhook_host, self._webhook_port)
        await site.start()
        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        )
        self._running = True

        logger.info(
            "[sms] Twilio webhook server listening on %s:%d, from: %s",
            self._webhook_host,
            self._webhook_port,
            redact_phone(self._from_number),
        )
        return True

    async def disconnect(self) -> None:
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
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

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted)
        last_result = SendResult(success=True)

        url = f"{TWILIO_API_BASE}/{self._account_sid}/Messages.json"
        headers = {
            "Authorization": self._basic_auth_header(),
        }

        session = self._http_session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        )
        try:
            for chunk in chunks:
                form_data = aiohttp.FormData()
                form_data.add_field("From", self._from_number)
                form_data.add_field("To", chat_id)
                form_data.add_field("Body", chunk)

                try:
                    async with session.post(url, data=form_data, headers=headers) as resp:
                        body = await resp.json()
                        if resp.status >= 400:
                            error_msg = body.get("message", str(body))
                            logger.error(
                                "[sms] send failed to %s: %s %s",
                                redact_phone(chat_id),
                                resp.status,
                                error_msg,
                            )
                            return SendResult(
                                success=False,
                                error=f"Twilio {resp.status}: {error_msg}",
                            )
                        msg_sid = body.get("sid", "")
                        last_result = SendResult(success=True, message_id=msg_sid)
                except Exception as e:
                    logger.error("[sms] send error to %s: %s", redact_phone(chat_id), e)
                    return SendResult(success=False, error=str(e))
        finally:
            # Close session only if we created a fallback (no persistent session)
            if not self._http_session and session:
                await session.close()

        return last_result

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

    # ------------------------------------------------------------------
    # SMS-specific formatting
    # ------------------------------------------------------------------

    def format_message(self, content: str) -> str:
        """Strip markdown — SMS renders it as literal characters."""
        return strip_markdown(content)

    # ------------------------------------------------------------------
    # Twilio signature validation
    # ------------------------------------------------------------------

    def _validate_twilio_signature(
        self, url: str, post_params: dict, signature: str,
    ) -> bool:
        """Validate ``X-Twilio-Signature`` header (HMAC-SHA1, base64).

        Algorithm: https://www.twilio.com/docs/usage/security#validating-requests
        """
        data_to_sign = url
        for key in sorted(post_params.keys()):
            data_to_sign += key + post_params[key]
        mac = hmac.new(
            self._auth_token.encode("utf-8"),
            data_to_sign.encode("utf-8"),
            hashlib.sha1,
        )
        computed = base64.b64encode(mac.digest()).decode("utf-8")
        return hmac.compare_digest(computed, signature)

    # ------------------------------------------------------------------
    # Twilio webhook handler
    # ------------------------------------------------------------------

    async def _handle_webhook(self, request) -> "aiohttp.web.Response":
        from aiohttp import web

        try:
            raw = await request.read()
            # Twilio sends form-encoded data, not JSON
            form = urllib.parse.parse_qs(raw.decode("utf-8"), keep_blank_values=True)
        except Exception as e:
            logger.error("[sms] webhook parse error: %s", e)
            return web.Response(
                text='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                content_type="application/xml",
                status=400,
            )

        # Validate Twilio request signature when SMS_WEBHOOK_URL is configured
        if self._webhook_url:
            twilio_sig = request.headers.get("X-Twilio-Signature", "")
            if not twilio_sig:
                logger.warning("[sms] Rejected: missing X-Twilio-Signature header")
                return web.Response(
                    text='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                    content_type="application/xml",
                    status=403,
                )
            flat_params = {k: v[0] for k, v in form.items() if v}
            if not self._validate_twilio_signature(
                self._webhook_url, flat_params, twilio_sig
            ):
                logger.warning("[sms] Rejected: invalid Twilio signature")
                return web.Response(
                    text='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                    content_type="application/xml",
                    status=403,
                )

        # Extract fields (parse_qs returns lists)
        from_number = (form.get("From", [""]))[0].strip()
        to_number = (form.get("To", [""]))[0].strip()
        text = (form.get("Body", [""]))[0].strip()
        message_sid = (form.get("MessageSid", [""]))[0].strip()

        if not from_number or not text:
            return web.Response(
                text='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                content_type="application/xml",
            )

        # Ignore messages from our own number (echo prevention)
        if from_number == self._from_number:
            logger.debug("[sms] ignoring echo from own number %s", redact_phone(from_number))
            return web.Response(
                text='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                content_type="application/xml",
            )

        logger.info(
            "[sms] inbound from %s -> %s: %s",
            redact_phone(from_number),
            redact_phone(to_number),
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
            raw_message=form,
            message_id=message_sid,
        )

        # Non-blocking: Twilio expects a fast response
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        # Return empty TwiML — we send replies via the REST API, not inline TwiML
        return web.Response(
            text='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
            content_type="application/xml",
        )
