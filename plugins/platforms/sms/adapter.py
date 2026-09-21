"""SMS (Twilio) platform adapter.

Outbound SMS via the Twilio REST API; inbound via an aiohttp webhook server.

Env vars — shared with the telephony skill: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
TWILIO_PHONE_NUMBER (E.164 from-number). Gateway-specific: SMS_WEBHOOK_PORT (8080),
SMS_WEBHOOK_HOST (127.0.0.1), SMS_WEBHOOK_URL (public URL for Twilio signature
validation — required), SMS_INSECURE_NO_SIGNATURE (true disables validation — dev only),
SMS_ALLOWED_USERS (comma-separated E.164), SMS_ALLOW_ALL_USERS, SMS_HOME_CHANNEL (cron),
SMS_STATUS_WEBHOOK_URL (public URL of /webhooks/twilio/status for signature validation;
defaults to SMS_WEBHOOK_URL + "/status"), SMS_ALLOW_COMMANDS (set to keep accepting
/slash commands from texters; unset = dropped). Inbound MMS media is downloaded, carrier
audio transcoded with ffmpeg when present, and handed to the STT/vision pipelines.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import re
import os
import shutil
import tempfile
import time
from pathlib import Path
import urllib.parse
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import gateway_trust_env, BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.helpers import redact_phone, strip_markdown
from gateway.platforms._shared import (
    env_is_connected as _env_is_connected, get_scoped_secret as _get_scoped_secret, send_error
)

try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:  # optional ([messaging] extra)
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

TWILIO_API_BASE = "https://api.twilio.com/2010-04-01/Accounts"
MAX_SMS_LENGTH = 900  # ~6 GSM-7 segments per message; longer replies are sent as several texts
DEFAULT_WEBHOOK_PORT = 8080
DEFAULT_WEBHOOK_HOST = "127.0.0.1"
_TWILIO_WEBHOOK_MAX_BODY_BYTES = 65_536  # 64 KiB — Twilio payloads are small
_EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'


def _twiml_response(status: int = 200):
    """Empty TwiML reply — replies go out via the REST API, never inline TwiML."""
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
    form_data = aiohttp.FormData()
    form_data.add_field("From", from_number)
    form_data.add_field("To", to_number)
    form_data.add_field("Body", body)
    return form_data


def _new_session(**kwargs):
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), **kwargs)


def check_sms_requirements() -> bool:
    """Check if SMS adapter dependencies are available."""
    return AIOHTTP_AVAILABLE and bool(
        _get_scoped_secret("TWILIO_ACCOUNT_SID") and _get_scoped_secret("TWILIO_AUTH_TOKEN"))


_GSM_SUBS = {
    "\u2014": "-", "\u2013": "-", "\u2012": "-", "\u2010": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2026": "...", "\u00a0": " ", "\u2022": "-", "\u00b7": "-", "\u2192": "->",
}


def _gsm_normalize(text: str) -> str:
    """Replace typography that forces 16-bit SMS encoding (67 chars/segment) with plain ASCII
    (160 chars/segment). Carriers reject messages over 10 segments (Twilio 30019)."""
    for k, v in _GSM_SUBS.items():
        text = text.replace(k, v)
    return text


_TRANSCODE_MIMES = frozenset({
    "audio/amr", "audio/amr-wb", "audio/3gpp", "audio/3gpp2", "audio/3gp", "video/3gpp",
    "audio/evrc", "audio/qcelp"})
_MAX_MEDIA_FILES = 10           # Twilio delivers up to 10 attachments per MMS
_MAX_MEDIA_BYTES = 20 * 1024 * 1024
_ALERT_MIN_INTERVAL = 60.0      # seconds between operator alerts; extra failures are counted, not sent


class SmsAdapter(BasePlatformAdapter):
    """Twilio SMS <-> Hermes: one session per inbound number; replies always from TWILIO_PHONE_NUMBER."""
    # Answers /p/<profile>/... on the default listener for a served secondary (shared_ingress).
    serves_profile_prefix: bool = True

    MAX_MESSAGE_LENGTH = MAX_SMS_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SMS)
        self._account_sid: str = _get_scoped_secret("TWILIO_ACCOUNT_SID", "")
        self._auth_token: str = _get_scoped_secret("TWILIO_AUTH_TOKEN", "")
        # Scoped like the sibling reads above: a secondary profile must not send from the default
        # profile's TWILIO_PHONE_NUMBER (#98738 class).
        self._from_number: str = _get_scoped_secret("TWILIO_PHONE_NUMBER", "")
        self._webhook_port: int = int(_get_scoped_secret("SMS_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT)))
        self._webhook_host: str = _get_scoped_secret("SMS_WEBHOOK_HOST", DEFAULT_WEBHOOK_HOST)
        self._webhook_url: str = _get_scoped_secret("SMS_WEBHOOK_URL", "").strip()
        self._runner = None
        self._http_session: Optional[aiohttp.ClientSession] = None

    # -- Lifecycle -----------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        insecure_no_sig = _get_scoped_secret("SMS_INSECURE_NO_SIGNATURE", "").lower() == "true"
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
        app.router.add_post("/webhooks/twilio/status", self._handle_status)
        app.router.add_get("/health", lambda _: web.Response(text="ok"))
        # Shared-listener mode (multiplex secondary): no bind; served at /p/<profile>/webhooks/twilio.
        from gateway.platforms.shared_ingress import bind_listener
        self._runner = await bind_listener(self, app, self._webhook_host, self._webhook_port, "/webhooks/twilio")
        self._http_session = _new_session(trust_env=gateway_trust_env())
        self._running = True
        if self._runner is not None:
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
            for chunk in self.truncate_message(_gsm_normalize(self.format_message(content)), self.MAX_MESSAGE_LENGTH):
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

    # ── Inbound media (MMS) ───────────────────────────────────────────────

    async def _ingest_with_media(self, form, from_number, text, message_sid, num_media):
        media_urls, media_types = [], []
        try:
            media_urls, media_types = await self._download_inbound_media(form, min(num_media, _MAX_MEDIA_FILES))
        except Exception as e:  # noqa: BLE001
            logger.warning("[sms] media ingestion error: %s", type(e).__name__)
        if not media_urls:
            text = (text + " " if text else "") + "[The sender attached media that could not be retrieved. Ask them to send it again.]"
        elif num_media > _MAX_MEDIA_FILES:
            text = (text + " " if text else "") + f"[{num_media - _MAX_MEDIA_FILES} further attachment(s) were not retrieved.]"
        source = self.build_source(
            chat_id=from_number, chat_name=from_number, chat_type="dm", user_id=from_number, user_name=from_number,
            message_id=message_sid)
        message_type = MessageType.TEXT
        if media_types:
            first = media_types[0]
            message_type = (MessageType.VOICE if first.startswith("audio/")
                            else MessageType.PHOTO if first.startswith("image/")
                            else MessageType.VIDEO if first.startswith("video/")
                            else MessageType.DOCUMENT)
            logger.info("[sms] media ready from %s: %d file(s), first=%s -> %s",
                        redact_phone(from_number), len(media_urls), first, message_type.value)
        event = MessageEvent(
            text=text, message_type=message_type, source=source, raw_message=form, message_id=message_sid,
            media_urls=media_urls, media_types=media_types)
        await self.handle_message(event)

    async def _transcode_to_m4a(self, data: bytes, mime: str):
        """ffmpeg: carrier audio (AMR, 3GPP, ...) -> 16 kHz mono AAC in an .m4a container.
        File I/O runs off the event loop; a timed-out ffmpeg is killed and reaped."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            logger.warning("[sms] ffmpeg missing; cannot transcode %s", mime)
            return data, mime
        tmp = tempfile.mkdtemp(prefix="sms-mms-")
        src_path, dst = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.m4a")
        try:
            await asyncio.to_thread(Path(src_path).write_bytes, data)
            proc = await asyncio.create_subprocess_exec(
                ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", src_path,
                "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "48k", dst,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            try:
                _, err = await asyncio.wait_for(proc.communicate(), timeout=60)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning("[sms] ffmpeg timeout transcoding %s", mime)
                return data, mime
            if proc.returncode != 0 or not os.path.exists(dst):
                logger.warning("[sms] ffmpeg failed for %s: %s", mime, (err or b"")[:200].decode(errors="replace"))
                return data, mime
            out = await asyncio.to_thread(Path(dst).read_bytes)
        finally:
            await asyncio.to_thread(shutil.rmtree, tmp, True)
        logger.info("[sms] transcoded %s (%d bytes) -> audio/mp4 (%d bytes)", mime, len(data), len(out))
        return out, "audio/mp4"

    async def _download_inbound_media(self, form, count: int):
        """Fetch MediaUrl0..N with Twilio basic auth on the first hop only (Twilio 307s to a signed
        CDN URL that must be fetched without the auth header). Retries briefly: the media URL can
        404 for a few seconds after the webhook fires. Honors the gateway proxy settings."""
        from gateway.platforms.media_cache import cache_media_bytes
        paths, mimes = [], []
        auth = {"Authorization": _basic_auth(self._account_sid, self._auth_token)}
        async with _new_session(trust_env=gateway_trust_env()) as session:
            for i in range(count):
                url = (form.get(f"MediaUrl{i}", [""])[0] or "").strip()
                mime = (form.get(f"MediaContentType{i}", [""])[0] or "").strip().lower()
                if not url.startswith("http"):
                    continue
                data = None
                for attempt, delay in enumerate((0, 2, 4, 6)):
                    if delay:
                        await asyncio.sleep(delay)
                    try:
                        async with session.get(url, headers=auth, allow_redirects=False) as r:
                            if r.status in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                                async with session.get(r.headers["Location"], allow_redirects=True) as r2:
                                    if r2.status == 200:
                                        data = await r2.content.read(_MAX_MEDIA_BYTES + 1)
                                        mime = mime or r2.headers.get("Content-Type", "").split(";")[0].strip().lower()
                                    else:
                                        logger.warning("[sms] media %d redirect fetch HTTP %s (attempt %d)", i, r2.status, attempt + 1)
                            elif r.status == 200:
                                data = await r.content.read(_MAX_MEDIA_BYTES + 1)
                                mime = mime or r.headers.get("Content-Type", "").split(";")[0].strip().lower()
                            else:
                                logger.warning("[sms] media %d fetch HTTP %s (attempt %d)", i, r.status, attempt + 1)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("[sms] media %d fetch error: %s (attempt %d)", i, type(e).__name__, attempt + 1)
                    if data is not None:
                        break
                if data is None:
                    logger.warning("[sms] media %d could not be retrieved after retries", i)
                    continue
                if len(data) > _MAX_MEDIA_BYTES:
                    logger.warning("[sms] media %d exceeds size cap; dropped", i)
                    continue
                mime = mime or "application/octet-stream"
                if mime in _TRANSCODE_MIMES:
                    data, mime = await self._transcode_to_m4a(data, mime)
                try:
                    paths.append(await asyncio.to_thread(cache_media_bytes, data, mime))
                    mimes.append(mime)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[sms] media %d cache error: %s", i, type(e).__name__)
        return paths, mimes

    # ── Delivery-status callbacks ─────────────────────────────────────────

    def _status_url(self) -> str:
        return _get_scoped_secret("SMS_STATUS_WEBHOOK_URL", "").strip() or (
            self._webhook_url.rstrip("/") + "/status" if self._webhook_url else "")

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Twilio message status callbacks. Point the number's (or Messaging Service's) status
        callback at /webhooks/twilio/status. failed/undelivered deliveries log a warning and, when
        TELEGRAM_BOT_TOKEN + TELEGRAM_HOME_CHANNEL are set, notify the operator on Telegram (rate
        limited to one alert per minute; further failures in the window are counted)."""
        content_length = request.content_length
        if content_length is not None and content_length > _TWILIO_WEBHOOK_MAX_BODY_BYTES:
            return _twiml_response(413)
        try:
            raw = await request.read()
        except Exception:  # noqa: BLE001
            return _twiml_response(400)
        if len(raw) > _TWILIO_WEBHOOK_MAX_BODY_BYTES:
            return _twiml_response(413)
        try:
            form = urllib.parse.parse_qs(raw.decode("utf-8"), keep_blank_values=True)
        except Exception:  # noqa: BLE001
            return _twiml_response(400)
        status_url = self._status_url()
        if status_url:
            sig = request.headers.get("X-Twilio-Signature", "")
            flat = {k: v[0] for k, v in form.items() if v}
            if not sig or not self._validate_twilio_signature(status_url, flat, sig):
                logger.warning("[sms] status callback rejected: bad signature")
                return _twiml_response(403)
        g = lambda k: (form.get(k, [""])[0] or "").strip()
        status, err, to, frm, sid = g("MessageStatus"), g("ErrorCode"), g("To"), g("From"), g("MessageSid")
        if status in ("failed", "undelivered"):
            line = (f"SMS delivery {status}: {redact_phone(frm)} -> {redact_phone(to)}"
                    f"{' error ' + err if err else ''} ({sid[-6:]})")
            logger.warning("[sms] %s", line)
            token, chat = _get_scoped_secret("TELEGRAM_BOT_TOKEN", ""), _get_scoped_secret("TELEGRAM_HOME_CHANNEL", "")
            if token and chat:
                now = time.monotonic()
                last = getattr(self, "_last_alert_at", 0.0)
                self._alert_suppressed = getattr(self, "_alert_suppressed", 0)
                if now - last >= _ALERT_MIN_INTERVAL:
                    extra = f" (+{self._alert_suppressed} more in the last minute)" if self._alert_suppressed else ""
                    self._last_alert_at, self._alert_suppressed = now, 0
                    ntask = asyncio.create_task(self._notify_operator(token, chat, "SMS alert: " + line + extra))
                    self._background_tasks.add(ntask)
                    ntask.add_done_callback(self._background_tasks.discard)
                else:
                    self._alert_suppressed += 1
        else:
            logger.debug("[sms] status %s for %s", status, sid)
        return _twiml_response()

    async def _notify_operator(self, token: str, chat_id: str, text: str) -> None:
        try:
            async with _new_session(trust_env=gateway_trust_env()) as s:
                async with s.post(f"https://api.telegram.org/bot{token}/sendMessage",
                                  json={"chat_id": chat_id, "text": text}) as r:
                    if r.status >= 400:
                        logger.warning("[sms] operator notify failed: HTTP %s", r.status)
        except Exception as e:  # noqa: BLE001
            logger.warning("[sms] operator notify error: %s", type(e).__name__)

    async def _handle_webhook(self, request: web.Request) -> web.Response:
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
        try:
            num_media = int((form.get("NumMedia", ["0"])[0] or "0").strip() or 0)
        except ValueError:
            num_media = 0
        if not from_number or (not text and num_media <= 0):
            return _twiml_response()
        if from_number == self._from_number:  # echo prevention
            logger.debug("[sms] ignoring echo from own number %s", redact_phone(from_number))
            return _twiml_response()
        if text.startswith("/") and not _get_scoped_secret("SMS_ALLOW_COMMANDS", "").strip():
            # Slash commands are operator tooling; on a deployment where the texter is an end
            # customer they are dropped. Set SMS_ALLOW_COMMANDS=1 to keep the pre-patch behaviour.
            logger.info("[sms] dropped slash command from %s (SMS_ALLOW_COMMANDS unset)", redact_phone(from_number))
            return _twiml_response()
        if num_media > 0:
            # Twilio expects an answer within 15 s; media download (with retries) runs in the background.
            logger.info("[sms] inbound MMS from %s -> %s (%d attachment(s))", redact_phone(from_number), redact_phone(to_number), num_media)
            mtask = asyncio.create_task(self._ingest_with_media(form, from_number, text, message_sid, num_media))
            self._background_tasks.add(mtask)
            mtask.add_done_callback(self._background_tasks.discard)
            return _twiml_response()
        logger.info("[sms] inbound from %s -> %s: %s", redact_phone(from_number), redact_phone(to_number), text[:80])
        source = self.build_source(
            chat_id=from_number, chat_name=from_number, chat_type="dm", user_id=from_number, user_name=from_number,
            message_id=message_sid)
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
    if not AIOHTTP_AVAILABLE:
        return send_error("aiohttp not installed. Run: pip install aiohttp")
    account_sid = _get_scoped_secret("TWILIO_ACCOUNT_SID", "")
    from_number = _get_scoped_secret("TWILIO_PHONE_NUMBER", "")  # scoped like account_sid: never the default's number
    if not account_sid or not auth_token or not from_number:
        return send_error("SMS not configured (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER required)")
    message = _gsm_normalize(_strip_markdown_for_sms(message))
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
                    return send_error(f"Twilio API error ({resp.status}): {error_msg}")
                return {"success": True, "platform": "sms", "chat_id": chat_id, "message_id": body.get("sid", "")}
    except Exception as e:
        return send_error(f"SMS send failed: {e}")


_is_connected = _env_is_connected("TWILIO_ACCOUNT_SID")



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
