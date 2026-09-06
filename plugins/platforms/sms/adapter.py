"""SMS (Twilio) platform adapter.

Outbound SMS via the Twilio REST API; inbound via an aiohttp webhook server.

Env vars — shared with the telephony skill: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
TWILIO_PHONE_NUMBER (E.164 from-number). Gateway-specific: SMS_WEBHOOK_PORT (8080),
SMS_WEBHOOK_HOST (127.0.0.1), SMS_WEBHOOK_URL (public URL for Twilio signature
validation — required), SMS_INSECURE_NO_SIGNATURE (true disables validation — dev only),
SMS_ALLOWED_USERS (comma-separated E.164), SMS_ALLOW_ALL_USERS, SMS_HOME_CHANNEL (cron).
"""

import asyncio
import base64
import hashlib
import hmac
import logging
import os
import re
import urllib.parse
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import gateway_trust_env, BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.platforms.helpers import redact_phone, strip_markdown
from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret

logger = logging.getLogger(__name__)

TWILIO_API_BASE = "https://api.twilio.com/2010-04-01/Accounts"
MAX_SMS_LENGTH = 1600  # ~10 SMS segments
DEFAULT_WEBHOOK_PORT = 8080
DEFAULT_WEBHOOK_HOST = "127.0.0.1"
_TWILIO_WEBHOOK_MAX_BODY_BYTES = 65_536  # 64 KiB — Twilio payloads are small
_EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'


def _twiml_response(status: int = 200):
    """Empty TwiML reply — replies go out via the REST API, never inline TwiML."""
    from aiohttp import web

    return web.Response(text=_EMPTY_TWIML, content_type="application/xml", status=status)


def _basic_auth(account_sid: str, auth_token: str) -> str:
    """HTTP Basic auth header value for Twilio."""
    encoded = base64.b64encode(f"{account_sid}:{auth_token}".encode("ascii")).decode("ascii")
    return f"Basic {encoded}"


def _messages_endpoint(account_sid: str, auth_token: str) -> tuple:
    """(Messages.json URL, auth headers) for the account."""
    return f"{TWILIO_API_BASE}/{account_sid}/Messages.json", {"Authorization": _basic_auth(account_sid, auth_token)}


def _twilio_form(from_number: str, to_number: str, body: str):
    """Twilio Messages.json form payload (aiohttp FormData)."""
    import aiohttp

    form_data = aiohttp.FormData()
    form_data.add_field("From", from_number)
    form_data.add_field("To", to_number)
    form_data.add_field("Body", body)
    return form_data


def _new_session(**kwargs):
    import aiohttp

    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), **kwargs)


def _aiohttp_available() -> bool:
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return False
    return True


def check_sms_requirements() -> bool:
    """Check if SMS adapter dependencies are available."""
    return _aiohttp_available() and bool(
        _get_scoped_secret("TWILIO_ACCOUNT_SID") and _get_scoped_secret("TWILIO_AUTH_TOKEN"))


class SmsAdapter(BasePlatformAdapter):
    """Twilio SMS <-> Hermes: one session per inbound number; replies always from TWILIO_PHONE_NUMBER."""

    MAX_MESSAGE_LENGTH = MAX_SMS_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SMS)
        self._account_sid: str = _get_scoped_secret("TWILIO_ACCOUNT_SID", "")
        self._auth_token: str = _get_scoped_secret("TWILIO_AUTH_TOKEN", "")
        self._from_number: str = os.getenv("TWILIO_PHONE_NUMBER", "")
        self._webhook_port: int = int(os.getenv("SMS_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT)))
        self._webhook_host: str = os.getenv("SMS_WEBHOOK_HOST", DEFAULT_WEBHOOK_HOST)
        self._webhook_url: str = os.getenv("SMS_WEBHOOK_URL", "").strip()
        self._runner = None
        self._http_session: Optional["aiohttp.ClientSession"] = None

    # -- Lifecycle -----------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        from aiohttp import web

        insecure_no_sig = os.getenv("SMS_INSECURE_NO_SIGNATURE", "").lower() == "true"
        fatal = None
        if not self._from_number:
            fatal = "sms_missing_phone_number", "[sms] TWILIO_PHONE_NUMBER not set — cannot send replies"
        elif not self._webhook_url and not insecure_no_sig:
            fatal = "sms_missing_webhook_url", (
                "[sms] Refusing to start: SMS_WEBHOOK_URL is required for Twilio "
                "signature validation. Set it to the public URL configured in your "
                "Twilio console (e.g. https://example.com/webhooks/twilio). "
                "For local development without validation, set "
                "SMS_INSECURE_NO_SIGNATURE=true (NOT recommended for production).")
        if fatal:
            logger.error(fatal[1])
            self._set_fatal_error(fatal[0], fatal[1], retryable=False)
            return False
        if insecure_no_sig and not self._webhook_url:
            logger.warning(
                "[sms] SMS_INSECURE_NO_SIGNATURE=true — Twilio signature validation "
                "is DISABLED. Any client that can reach port %d can inject messages. "
                "Do NOT use this in production.",
                self._webhook_port)
        # client_max_size bounds every read path (incl. chunked bodies with no
        # Content-Length) before the handler's own 413 checks run.
        # See #58536, #58902, #59180.
        app = web.Application(client_max_size=_TWILIO_WEBHOOK_MAX_BODY_BYTES)
        app.router.add_post("/webhooks/twilio", self._handle_webhook)
        app.router.add_get("/health", lambda _: web.Response(text="ok"))
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._webhook_host, self._webhook_port)
        await site.start()
        self._http_session = _new_session(trust_env=gateway_trust_env())
        self._running = True
        logger.info(
            "[sms] Twilio webhook server listening on %s:%d, from: %s",
            self._webhook_host, self._webhook_port, redact_phone(self._from_number))
        self._wire_plugin_handlers(None)
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

    # -- Outbound ------------------------------------------------------------

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        last_result = SendResult(success=True)
        url, headers = _messages_endpoint(self._account_sid, self._auth_token)
        session = self._http_session or _new_session(trust_env=gateway_trust_env())
        try:
            for chunk in self.truncate_message(self.format_message(content)):
                form_data = _twilio_form(self._from_number, chat_id, chunk)
                try:
                    async with session.post(url, data=form_data, headers=headers) as resp:
                        body = await resp.json()
                        if resp.status >= 400:
                            error_msg = body.get("message", str(body))
                            logger.error(
                                "[sms] send failed to %s: %s %s", redact_phone(chat_id), resp.status, error_msg,
                            )
                            return SendResult(success=False, error=f"Twilio {resp.status}: {error_msg}")
                        last_result = SendResult(success=True, message_id=body.get("sid", ""))
                except Exception as e:
                    logger.error("[sms] send error to %s: %s", redact_phone(chat_id), e)
                    return SendResult(success=False, error=str(e))
        finally:
            if not self._http_session and session:  # close only a fallback session we created
                await session.close()
        return last_result

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

    def format_message(self, content: str) -> str:
        """Strip markdown — SMS renders it as literal characters."""
        return strip_markdown(content)

    # -- Twilio signature validation -----------------------------------------

    def _validate_twilio_signature(self, url: str, post_params: dict, signature: str) -> bool:
        """Validate ``X-Twilio-Signature`` (HMAC-SHA1, base64).

        Twilio may sign the URL with or without the scheme's default port, so
        both variants are tried. https://www.twilio.com/docs/usage/security#validating-requests
        """
        if self._check_signature(url, post_params, signature):
            return True
        variant = self._port_variant_url(url)
        return bool(variant and self._check_signature(variant, post_params, signature))

    def _check_signature(self, url: str, post_params: dict, signature: str) -> bool:
        data_to_sign = url + "".join(key + post_params[key] for key in sorted(post_params.keys()))
        mac = hmac.new(self._auth_token.encode("utf-8"), data_to_sign.encode("utf-8"), hashlib.sha1)
        computed = base64.b64encode(mac.digest()).decode("utf-8")
        # Compare as bytes: compare_digest raises TypeError on non-ASCII str,
        # and the signature is a raw request header.
        return hmac.compare_digest(computed.encode(), signature.encode())

    @staticmethod
    def _port_variant_url(url: str) -> str | None:
        """URL with the scheme's default port toggled (added/stripped); None for non-default ports."""
        parsed = urllib.parse.urlparse(url)
        default_port = {"https": 443, "http": 80}.get(parsed.scheme)
        if default_port is None:
            return None
        if parsed.port == default_port:
            netloc = parsed.hostname
        elif parsed.port is None:
            netloc = f"{parsed.hostname}:{default_port}"
        else:
            return None
        return urllib.parse.urlunparse(
            (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))

    # -- Inbound webhook -----------------------------------------------------

    async def _handle_webhook(self, request) -> "aiohttp.web.Response":
        try:
            content_length = request.content_length
            if content_length is not None and content_length > _TWILIO_WEBHOOK_MAX_BODY_BYTES:
                return _twiml_response(413)
            raw = await request.read()
            if len(raw) > _TWILIO_WEBHOOK_MAX_BODY_BYTES:
                return _twiml_response(413)
            # Twilio sends form-encoded data, not JSON; parse_qs values are lists.
            form = urllib.parse.parse_qs(raw.decode("utf-8"), keep_blank_values=True)
        except Exception as e:
            logger.error("[sms] webhook parse error: %s", e)
            return _twiml_response(400)
        if self._webhook_url:
            twilio_sig = request.headers.get("X-Twilio-Signature", "")
            flat_params = {k: v[0] for k, v in form.items() if v}
            rejected = ("missing X-Twilio-Signature header" if not twilio_sig
                        else "" if self._validate_twilio_signature(self._webhook_url, flat_params, twilio_sig)
                        else "invalid Twilio signature")
            if rejected:
                logger.warning("[sms] Rejected: %s", rejected)
                return _twiml_response(403)
        from_number, to_number, text, message_sid = (
            form.get(key, [""])[0].strip() for key in ("From", "To", "Body", "MessageSid"))
        if not from_number or not text:
            return _twiml_response()
        if from_number == self._from_number:  # echo prevention
            logger.debug("[sms] ignoring echo from own number %s", redact_phone(from_number))
            return _twiml_response()
        logger.info("[sms] inbound from %s -> %s: %s", redact_phone(from_number), redact_phone(to_number), text[:80])
        source = self.build_source(
            chat_id=from_number, chat_name=from_number, chat_type="dm", user_id=from_number, user_name=from_number)
        event = MessageEvent(
            text=text, message_type=MessageType.TEXT, source=source, raw_message=form, message_id=message_sid)
        # Non-blocking: Twilio expects a fast response
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return _twiml_response()


# -- Plugin registration (TWILIO_* env→PlatformConfig seeding stays in gateway/config.py)

# Standalone-send markdown stripping: looser than helpers.strip_markdown (no
# word-boundary guards on underscores, ``[a-z]*`` fence tags) — kept for parity.
_SMS_MARKDOWN_SUBS = (
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"\1"), (re.compile(r"\*(.+?)\*", re.DOTALL), r"\1"),
    (re.compile(r"__(.+?)__", re.DOTALL), r"\1"), (re.compile(r"_(.+?)_", re.DOTALL), r"\1"),
    (re.compile(r"```[a-z]*\n?"), ""), (re.compile(r"`(.+?)`"), r"\1"),
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""), (re.compile(r"\[([^\]]+)\]\([^\)]+\)"), r"\1"),
    (re.compile(r"\n{3,}"), "\n\n"))


# ────────────────────────────────────────────────────────────────────────── Plugin migration glue (#41112 /
# #3823) Added when the SMS (Twilio) adapter moved from gateway/platforms/sms.py into this bundled plugin.
# register() exposes the platform via the registry, replacing the Platform.SMS elif in gateway/run.py, the
# _PLATFORM_CONNECTED_CHECKERS entry in gateway/config.py, the _PLATFORMS["sms"] static dict in
# hermes_cli/gateway.py, and the _send_sms dispatch in tools/send_message_tool.py. TWILIO_*
# env→PlatformConfig seeding stays in core.
# ──────────────────────────────────────────────────────────────────────────
def _strip_markdown_for_sms(message: str) -> str:
    """Strip markdown — SMS renders it as literal characters."""
    for pattern, repl in _SMS_MARKDOWN_SUBS:
        message = pattern.sub(repl, message)
    return message.strip()


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """Out-of-process SMS delivery via the Twilio REST API (standalone_sender_fn contract)."""
    auth_token = getattr(pconfig, "api_key", None) or _get_scoped_secret("TWILIO_AUTH_TOKEN", "")
    if not _aiohttp_available():
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    account_sid = _get_scoped_secret("TWILIO_ACCOUNT_SID", "")
    from_number = os.getenv("TWILIO_PHONE_NUMBER", "")
    if not account_sid or not auth_token or not from_number:
        return {"error": "SMS not configured (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER required)"}
    message = _strip_markdown_for_sms(message)
    try:
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(resolve_proxy_url())
        url, headers = _messages_endpoint(account_sid, auth_token)
        async with _new_session(**_sess_kw) as session:
            form_data = _twilio_form(from_number, chat_id, message)
            async with session.post(url, data=form_data, headers=headers, **_req_kw) as resp:
                body = await resp.json()
                if resp.status >= 400:
                    error_msg = body.get("message", str(body))
                    return _redacted_error(f"Twilio API error ({resp.status}): {error_msg}")
                return {"success": True, "platform": "sms", "chat_id": chat_id, "message_id": body.get("sid", "")}
    except Exception as e:
        return _redacted_error(f"SMS send failed: {e}")


def _redacted_error(text: str) -> dict:
    """Error dict with phone numbers redacted by send_message_tool when available."""
    try:
        from tools.send_message_tool import _error as _e
        return _e(text)
    except Exception:
        return {"error": text}


def _is_connected(config) -> bool:
    """SMS is connected when Twilio credentials are present (bool(TWILIO_ACCOUNT_SID))."""
    import hermes_cli.gateway as gateway_mod
    return bool((gateway_mod.get_env_value("TWILIO_ACCOUNT_SID") or "").strip())


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="sms", label="SMS (Twilio)", adapter_factory=SmsAdapter,
        check_fn=check_sms_requirements, is_connected=_is_connected,
        required_env=["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER"],
        install_hint="pip install aiohttp", allowed_users_env="SMS_ALLOWED_USERS",
        allow_all_env="SMS_ALLOW_ALL_USERS", cron_deliver_env_var="SMS_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send, max_message_length=MAX_SMS_LENGTH, pii_safe=True,
        emoji="📱", allow_update_command=True)
