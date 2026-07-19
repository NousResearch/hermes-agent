"""SMS (Twilio) platform adapter.

Connects to the Twilio REST API for outbound SMS and runs an aiohttp
webhook server to receive inbound messages.

Shares credentials with the optional telephony skill — same env vars:
  - TWILIO_ACCOUNT_SID
  - TWILIO_AUTH_TOKEN
  - TWILIO_PHONE_NUMBER  (E.164 from-number, e.g. +15551234567)

Gateway-specific env vars:
  - SMS_WEBHOOK_PORT     (default 8080)
  - SMS_WEBHOOK_HOST     (default 127.0.0.1)
  - SMS_WEBHOOK_URL      (public URL for Twilio signature validation — required)
  - SMS_INSECURE_NO_SIGNATURE  (true to disable signature validation — dev only)
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
)
from gateway.platforms.helpers import redact_phone, strip_markdown

logger = logging.getLogger(__name__)

TWILIO_API_BASE = "https://api.twilio.com/2010-04-01/Accounts"
MAX_SMS_LENGTH = 1600  # ~10 SMS segments
DEFAULT_WEBHOOK_PORT = 8080
DEFAULT_WEBHOOK_HOST = "127.0.0.1"
_TWILIO_WEBHOOK_MAX_BODY_BYTES = 65_536  # 64 KiB — Twilio payloads are small


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

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        import aiohttp
        from aiohttp import web

        if not self._from_number:
            msg = "[sms] TWILIO_PHONE_NUMBER not set — cannot send replies"
            logger.error(msg)
            self._set_fatal_error("sms_missing_phone_number", msg, retryable=False)
            return False

        insecure_no_sig = os.getenv("SMS_INSECURE_NO_SIGNATURE", "").lower() == "true"

        if not self._webhook_url and not insecure_no_sig:
            msg = (
                "[sms] Refusing to start: SMS_WEBHOOK_URL is required for Twilio "
                "signature validation. Set it to the public URL configured in your "
                "Twilio console (e.g. https://example.com/webhooks/twilio). "
                "For local development without validation, set "
                "SMS_INSECURE_NO_SIGNATURE=true (NOT recommended for production)."
            )
            logger.error(msg)
            self._set_fatal_error("sms_missing_webhook_url", msg, retryable=False)
            return False

        if insecure_no_sig and not self._webhook_url:
            logger.warning(
                "[sms] SMS_INSECURE_NO_SIGNATURE=true — Twilio signature validation "
                "is DISABLED. Any client that can reach port %d can inject messages. "
                "Do NOT use this in production.",
                self._webhook_port,
            )

        # client_max_size bounds every read path — including chunked bodies
        # with no Content-Length — before the handler's own 413 checks run
        # (#58536/#58902/#59180 pattern).
        app = web.Application(client_max_size=_TWILIO_WEBHOOK_MAX_BODY_BYTES)
        app.router.add_post("/webhooks/twilio", self._handle_webhook)
        app.router.add_get("/health", lambda _: web.Response(text="ok"))

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._webhook_host, self._webhook_port)
        await site.start()
        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            trust_env=True,
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
            trust_env=True,
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

        Tries both with and without the default port for the URL scheme,
        since Twilio may sign with either variant.

        Algorithm: https://www.twilio.com/docs/usage/security#validating-requests
        """
        if self._check_signature(url, post_params, signature):
            return True

        variant = self._port_variant_url(url)
        if variant and self._check_signature(variant, post_params, signature):
            return True

        return False

    def _check_signature(
        self, url: str, post_params: dict, signature: str,
    ) -> bool:
        """Compute and compare a single Twilio signature."""
        data_to_sign = url
        for key in sorted(post_params.keys()):
            data_to_sign += key + post_params[key]
        mac = hmac.new(
            self._auth_token.encode("utf-8"),
            data_to_sign.encode("utf-8"),
            hashlib.sha1,
        )
        computed = base64.b64encode(mac.digest()).decode("utf-8")
        # Compare as bytes: compare_digest raises TypeError on a str with
        # non-ASCII characters, and the signature is a raw request header.
        return hmac.compare_digest(computed.encode(), signature.encode())

    @staticmethod
    def _port_variant_url(url: str) -> str | None:
        """Return the URL with the default port toggled, or None.

        Only toggles default ports (443 for https, 80 for http).
        Non-standard ports are never modified.
        """
        parsed = urllib.parse.urlparse(url)
        default_ports = {"https": 443, "http": 80}
        default_port = default_ports.get(parsed.scheme)
        if default_port is None:
            return None

        if parsed.port == default_port:
            # Has explicit default port → strip it
            return urllib.parse.urlunparse(
                (parsed.scheme, parsed.hostname, parsed.path,
                 parsed.params, parsed.query, parsed.fragment)
            )
        elif parsed.port is None:
            # No port → add default
            netloc = f"{parsed.hostname}:{default_port}"
            return urllib.parse.urlunparse(
                (parsed.scheme, netloc, parsed.path,
                 parsed.params, parsed.query, parsed.fragment)
            )

        # Non-standard port — no variant
        return None

    # ------------------------------------------------------------------
    # Twilio webhook handler
    # ------------------------------------------------------------------

    async def _handle_webhook(self, request) -> "aiohttp.web.Response":
        from aiohttp import web

        try:
            content_length = request.content_length
            if content_length is not None and content_length > _TWILIO_WEBHOOK_MAX_BODY_BYTES:
                return web.Response(
                    text='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                    content_type="application/xml",
                    status=413,
                )
            raw = await request.read()
            if len(raw) > _TWILIO_WEBHOOK_MAX_BODY_BYTES:
                return web.Response(
                    text='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                    content_type="application/xml",
                    status=413,
                )
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


# ──────────────────────────────────────────────────────────────────────────
# Plugin migration glue (#41112 / #3823)
#
# Added when the SMS (Twilio) adapter moved from gateway/platforms/sms.py into
# this bundled plugin. register() exposes the platform via the registry,
# replacing the Platform.SMS elif in gateway/run.py, the
# _PLATFORM_CONNECTED_CHECKERS entry in gateway/config.py, the _PLATFORMS["sms"]
# static dict in hermes_cli/gateway.py, and the _send_sms dispatch in
# tools/send_message_tool.py. TWILIO_* env→PlatformConfig seeding stays in core.
# ──────────────────────────────────────────────────────────────────────────


def _strip_markdown_for_sms(message: str) -> str:
    """Strip markdown — SMS renders it as literal characters."""
    message = re.sub(r"\*\*(.+?)\*\*", r"\1", message, flags=re.DOTALL)
    message = re.sub(r"\*(.+?)\*", r"\1", message, flags=re.DOTALL)
    message = re.sub(r"__(.+?)__", r"\1", message, flags=re.DOTALL)
    message = re.sub(r"_(.+?)_", r"\1", message, flags=re.DOTALL)
    message = re.sub(r"```[a-z]*\n?", "", message)
    message = re.sub(r"`(.+?)`", r"\1", message)
    message = re.sub(r"^#{1,6}\s+", "", message, flags=re.MULTILINE)
    message = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", message)
    message = re.sub(r"\n{3,}", "\n\n", message)
    return message.strip()


async def _standalone_send(
    pconfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """Out-of-process SMS delivery via the Twilio REST API. Implements the
    standalone_sender_fn contract; replaces the legacy _send_sms helper."""
    auth_token = getattr(pconfig, "api_key", None) or os.getenv("TWILIO_AUTH_TOKEN", "")
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    import base64

    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    from_number = os.getenv("TWILIO_PHONE_NUMBER", "")
    if not account_sid or not auth_token or not from_number:
        return {"error": "SMS not configured (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER required)"}

    message = _strip_markdown_for_sms(message)

    def _redacted_error(text):
        try:
            from tools.send_message_tool import _error as _e
            return _e(text)
        except Exception:
            return {"error": text}

    try:
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _proxy = resolve_proxy_url()
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
        creds = f"{account_sid}:{auth_token}"
        encoded = base64.b64encode(creds.encode("ascii")).decode("ascii")
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        headers = {"Authorization": f"Basic {encoded}"}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), **_sess_kw) as session:
            form_data = aiohttp.FormData()
            form_data.add_field("From", from_number)
            form_data.add_field("To", chat_id)
            form_data.add_field("Body", message)
            async with session.post(url, data=form_data, headers=headers, **_req_kw) as resp:
                body = await resp.json()
                if resp.status >= 400:
                    error_msg = body.get("message", str(body))
                    return _redacted_error(f"Twilio API error ({resp.status}): {error_msg}")
                return {"success": True, "platform": "sms", "chat_id": chat_id, "message_id": body.get("sid", "")}
    except Exception as e:
        return _redacted_error(f"SMS send failed: {e}")


def _is_connected(config) -> bool:
    """SMS is connected when Twilio credentials are present. Mirrors the legacy
    _PLATFORM_CONNECTED_CHECKERS[Platform.SMS] = bool(TWILIO_ACCOUNT_SID)."""
    import hermes_cli.gateway as gateway_mod
    return bool((gateway_mod.get_env_value("TWILIO_ACCOUNT_SID") or "").strip())


def _build_adapter(config):
    """Factory wrapper that constructs SmsAdapter from a PlatformConfig."""
    return SmsAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="sms",
        label="SMS (Twilio)",
        adapter_factory=_build_adapter,
        check_fn=check_sms_requirements,
        is_connected=_is_connected,
        required_env=["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER"],
        install_hint="pip install aiohttp",
        allowed_users_env="SMS_ALLOWED_USERS",
        allow_all_env="SMS_ALLOW_ALL_USERS",
        cron_deliver_env_var="SMS_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=MAX_SMS_LENGTH,
        pii_safe=True,
        emoji="📱",
        allow_update_command=True,
    )
