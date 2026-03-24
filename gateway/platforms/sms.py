"""SMS (Twilio) platform adapter.

Connects to the Twilio REST API for outbound SMS and runs an aiohttp
webhook server to receive inbound messages.

Shares credentials with the optional telephony skill — same env vars:
  - TWILIO_ACCOUNT_SID
  - TWILIO_AUTH_TOKEN
  - TWILIO_PHONE_NUMBER  (E.164 from-number, e.g. +15551234567)

Gateway-specific env vars:
  - SMS_WEBHOOK_PORT      (default 8080)
  - SMS_WEBHOOK_HOST      (default 127.0.0.1)
  - SMS_PUBLIC_WEBHOOK_URL (canonical public URL for Twilio signature validation)
  - SMS_ALLOWED_USERS     (comma-separated E.164 phone numbers)
  - SMS_ALLOW_ALL_USERS   (true/false)
  - SMS_HOME_CHANNEL      (phone number for cron delivery)
"""

import asyncio
import base64
import hashlib
import hmac
import logging
import os
import re
import time
import urllib.parse
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

TWILIO_API_BASE = "https://api.twilio.com/2010-04-01/Accounts"
MAX_SMS_LENGTH = 1600  # ~10 SMS segments
DEFAULT_WEBHOOK_PORT = 8080
DEFAULT_WEBHOOK_HOST = "127.0.0.1"
DEFAULT_PUBLIC_WEBHOOK_URL = "https://sms.gibbsoft.click/webhooks/twilio"
DEFAULT_WINDOW_SECONDS = 60
DEFAULT_MAX_REQUESTS_PER_WINDOW = 30
DEFAULT_REPLAY_TTL_SECONDS = 600

# E.164 phone number pattern for redaction
_PHONE_RE = re.compile(r"\+[1-9]\d{6,14}")


def _redact_phone(phone: str) -> str:
    """Redact a phone number for logging: +15551234567 -> +1555***4567."""
    if not phone:
        return "<none>"
    if len(phone) <= 8:
        return phone[:2] + "***" + phone[-2:] if len(phone) > 4 else "****"
    return phone[:5] + "***" + phone[-4:]


class SlidingWindowRateLimiter:
    def __init__(self, max_events: int, window_seconds: int):
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._events: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        events = self._events[key]
        cutoff = now - self.window_seconds
        while events and events[0] < cutoff:
            events.popleft()
        if len(events) >= self.max_events:
            return False
        events.append(now)
        return True


class ReplayCache:
    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._seen: dict[str, float] = {}

    def seen(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.ttl_seconds
        expired = [k for k, ts in self._seen.items() if ts < cutoff]
        for old_key in expired:
            self._seen.pop(old_key, None)
        if key in self._seen:
            return True
        self._seen[key] = now
        return False


def _signature_payload(url: str, form: dict[str, list[str]]) -> str:
    payload = url
    for key in sorted(form):
        for value in form[key]:
            payload += key + value
    return payload


def _compute_twilio_signature(url: str, form: dict[str, list[str]], auth_token: str) -> str:
    digest = hmac.new(
        auth_token.encode("utf-8"),
        _signature_payload(url, form).encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def is_valid_twilio_request(url: str, form: dict[str, list[str]], signature: str, auth_token: str) -> bool:
    if not url or not signature or not auth_token:
        return False
    expected = _compute_twilio_signature(url, form, auth_token)
    return hmac.compare_digest(expected, signature)


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
        self._bind_host: str = os.getenv("SMS_WEBHOOK_HOST", DEFAULT_WEBHOOK_HOST)
        self._public_webhook_url: str = os.getenv(
            "SMS_PUBLIC_WEBHOOK_URL", DEFAULT_PUBLIC_WEBHOOK_URL
        )
        self._max_requests_per_window: int = int(
            os.getenv("SMS_WEBHOOK_MAX_REQUESTS_PER_WINDOW", str(DEFAULT_MAX_REQUESTS_PER_WINDOW))
        )
        self._window_seconds: int = int(
            os.getenv("SMS_WEBHOOK_WINDOW_SECONDS", str(DEFAULT_WINDOW_SECONDS))
        )
        self._replay_ttl_seconds: int = int(
            os.getenv("SMS_WEBHOOK_REPLAY_TTL_SECONDS", str(DEFAULT_REPLAY_TTL_SECONDS))
        )
        self._runner = None
        self._http_session: Optional["aiohttp.ClientSession"] = None
        self._rate_limiter = SlidingWindowRateLimiter(
            max_events=self._max_requests_per_window,
            window_seconds=self._window_seconds,
        )
        self._replay_cache = ReplayCache(ttl_seconds=self._replay_ttl_seconds)

    def _basic_auth_header(self) -> str:
        """Build HTTP Basic auth header value for Twilio."""
        creds = f"{self._account_sid}:{self._auth_token}"
        encoded = base64.b64encode(creds.encode("ascii")).decode("ascii")
        return f"Basic {encoded}"

    def _client_ip(self, request) -> str:
        header_ip = request.headers.get("CF-Connecting-IP") or request.headers.get("X-Forwarded-For", "")
        if header_ip:
            return header_ip.split(",", 1)[0].strip()
        return request.remote or "unknown"

    def _is_valid_signature(self, form: dict[str, list[str]], signature: str, original_url: str) -> bool:
        return is_valid_twilio_request(original_url, form, signature, self._auth_token)

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        import aiohttp
        from aiohttp import web

        if not self._from_number:
            logger.error("[sms] TWILIO_PHONE_NUMBER not set — cannot send replies")
            return False

        app = web.Application()
        app.router.add_post("/webhooks/twilio", self._handle_webhook)
        app.router.add_get("/health", lambda _: web.Response(text="ok"))

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._bind_host, self._webhook_port)
        await site.start()
        self._http_session = aiohttp.ClientSession()
        self._running = True

        logger.info(
            "[sms] Twilio webhook server listening on %s:%d, from: %s",
            self._bind_host,
            self._webhook_port,
            _redact_phone(self._from_number),
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

        session = self._http_session or aiohttp.ClientSession()
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
                                _redact_phone(chat_id),
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
                    logger.error("[sms] send error to %s: %s", _redact_phone(chat_id), e)
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

        signature = request.headers.get("X-Twilio-Signature", "")
        original_url = request.headers.get("X-Twilio-Original-Url", self._public_webhook_url)
        if not self._is_valid_signature(form, signature, original_url):
            logger.warning("[sms] invalid or missing Twilio signature")
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
        client_ip = self._client_ip(request)

        if not self._rate_limiter.allow(f"sms:{client_ip}"):
            logger.warning("[sms] rate limit exceeded for source %s", client_ip)
            return web.Response(
                text='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                content_type="application/xml",
                status=429,
            )

        if message_sid and self._replay_cache.seen(f"sms:{message_sid}"):
            logger.info("[sms] ignoring replayed webhook %s", message_sid)
            return web.Response(
                text='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                content_type="application/xml",
            )

        if not from_number or not text:
            return web.Response(
                text='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                content_type="application/xml",
            )

        # Ignore messages from our own number (echo prevention)
        if from_number == self._from_number:
            logger.debug("[sms] ignoring echo from own number %s", _redact_phone(from_number))
            return web.Response(
                text='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                content_type="application/xml",
            )

        logger.info(
            "[sms] inbound from %s -> %s (sid=%s, len=%d)",
            _redact_phone(from_number),
            _redact_phone(to_number),
            message_sid or "-",
            len(text),
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
        asyncio.create_task(self.handle_message(event))

        # Return empty TwiML — we send replies via the REST API, not inline TwiML
        return web.Response(
            text='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
            content_type="application/xml",
        )
