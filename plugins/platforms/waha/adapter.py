"""WAHA (unofficial WhatsApp bridge) platform adapter.

WAHA (https://waha.devlike.pro) is a separately-run Docker service that bridges WhatsApp Web
over HTTP + webhooks. Unlike ``plugins/platforms/whatsapp`` (the Baileys bridge this process
spawns and manages as a subprocess — PID tracking, port scanning, log tailing), WAHA is operated
independently: this adapter only ever talks to it over HTTP. Behavior (DM/group gating,
allowlists, mention detection, WhatsApp markdown, chunk budgeting) is reused unmodified from
``WhatsAppBehaviorMixin`` — only transport (inbound webhook + outbound REST) is new.

LID (privacy) addressing: a WAHA inbound payload may carry an opaque ``"<digits>@lid"`` as the
primary identifier while the real phone-JID form rides along in the same event
(``payload._data.key.remoteJidAlt``, discovered live debugging the cron-based webhook filter
scripts this adapter replaces — see ``webhook_filter_rob_whatsapp.py``). Every message carries
both forms, so identity is resolved PER MESSAGE by preferring any non-``@lid`` candidate across
every JID-bearing field WAHA sends (``_prefer_phone_jid``) — no persisted lid-mapping store is
needed (WAHA has none; that mechanism is Baileys-session-specific).

Multi-sender-one-account: WAHA serves ONE WhatsApp session (one phone number) as a bot shared by
several distinct humans (sender-filtered). This adapter is a SINGLE instance (normally owned by
the ``default`` profile); splitting senders across Hermes profiles is deliberately NOT
reimplemented here — it is ``gateway.profile_routes`` (``gateway/profile_routing.py``), the same
generic mechanism Discord/Telegram secondary-profile routing already uses. ``build_source()``
resolves the owning profile per inbound chat_id for free.

Inbound media (image + voice note): this WAHA instance runs the NOWEB engine with media download
enabled (the default — neither ``WHATSAPP_DOWNLOAD_MEDIA=false`` nor ``WAHA_EVENTS_DOWNLOAD_MEDIA``
is set in its ``.env``), so every ``hasMedia`` event carries a ``payload.media.url`` pointing at
WAHA's own ``/api/files/{fileId}`` endpoint — never inline base64. That endpoint requires the same
``X-Api-Key`` auth as the rest of the REST API (WAHA's documented default), so
``_download_media_bytes`` fetches it authenticated and streamed under the shared
``gateway.max_inbound_media_bytes`` cap (``validate_inbound_media_size``) — the save+validate work
itself is NOT reimplemented, it goes straight into ``cache_image_from_bytes``/
``cache_audio_from_bytes`` (``gateway/platforms/base.py``), same as every other adapter. A voice
note vs. a shared audio file is distinguished via NOWEB's raw engine data
(``payload._data.message.audioMessage.ptt``) — an engine-specific field (the docs say ``_data``
"can be different for each engine"), acceptable since this instance is pinned to NOWEB. Once
cached, the gateway's own STT pipeline (``gateway/run_voice.py``) transcribes
``MessageType.VOICE``/``AUDIO`` attachments automatically — no transcription logic lives here.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms._shared import (
    coerce_port, extra_or_secret as _extra_or_secret, get_scoped_secret as _get_scoped_secret,
    seed_extra_from_env as _seed_extra_from_env, apply_yaml_bridge as _apply_yaml_bridge, send_error,
)
from gateway.platforms.base import (
    BasePlatformAdapter, SendResult, cache_audio_from_bytes_async, cache_image_from_bytes_async,
    get_inbound_media_max_bytes, validate_inbound_media_size,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.helpers import MessageDeduplicator, redact_phone
from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin

try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:  # optional ([messaging] extra)
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:3000"
DEFAULT_SESSION = "default"
DEFAULT_WEBHOOK_HOST = "127.0.0.1"
DEFAULT_WEBHOOK_PORT = 8666
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "ip6-localhost", "ip6-loopback"})
_MAX_LOCAL_MEDIA_BYTES = 16 * 1024 * 1024  # WAHA media goes over base64 JSON, not multipart


def check_waha_requirements() -> bool:
    """Passive "deps importable?" probe — never installs, never makes a network call."""
    return AIOHTTP_AVAILABLE


def _is_loopback_host(host: Optional[str]) -> bool:
    return bool(host) and str(host).strip().lower() in _LOOPBACK_HOSTS


def _digits(value: Any) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _is_group_jid(value: Any) -> bool:
    return str(value or "").strip().lower().endswith("@g.us")


def _prefer_phone_jid(candidates: List[Any]) -> str:
    """First candidate that is NOT an opaque ``@lid`` address; else the first candidate at all.

    WAHA sends both forms on every event (``from``/``remoteJid`` may be ``@lid``, the phone JID
    rides along in ``remoteJidAlt``/``participant``), so this is a per-message resolution — no
    persisted mapping store needed."""
    cleaned = [str(c).strip() for c in candidates if c]
    for candidate in cleaned:
        if "@lid" not in candidate.lower():
            return candidate
    return cleaned[0] if cleaned else ""


def _waha_chat_id(value: str) -> str:
    """A bare phone number becomes a WAHA-style ``<digits>@c.us`` chat id; anything already
    carrying a JID domain (``@c.us``, ``@s.whatsapp.net``, ``@g.us``, ``@lid``) passes through."""
    v = str(value or "").strip()
    if "@" in v:
        return v
    digits = _digits(v)
    return f"{digits}@c.us" if digits else v


def _extract_mentioned_ids(data: Dict[str, Any]) -> List[str]:
    """Best-effort ``mentionedJid`` extraction across the common WhatsApp message types;
    normalized to bare digits so comparisons never depend on a JID domain match."""
    message = data.get("message") or {}
    for key in ("extendedTextMessage", "imageMessage", "videoMessage", "documentMessage"):
        ctx = (message.get(key) or {}).get("contextInfo") or {}
        mentioned = ctx.get("mentionedJid")
        if mentioned:
            return [d for m in mentioned if (d := _digits(m))]
    return []


def _is_ptt_audio(data: Dict[str, Any]) -> bool:
    """True for a WhatsApp voice note (push-to-talk) vs. a regular shared audio file. NOWEB's raw
    engine data mirrors Baileys' own ``audioMessage.ptt`` flag (see module docstring)."""
    audio = ((data.get("message") or {}).get("audioMessage")) or {}
    return bool(audio.get("ptt"))


DEFAULT_IMAGE_EXT = ".jpg"
DEFAULT_AUDIO_EXT = ".ogg"


def _guess_media_ext(mimetype: str, filename: Optional[str], default: str) -> str:
    """Best-effort file extension: the filename's own suffix first, then a MIME guess, else
    ``default``. Cosmetic only — ``cache_image_from_bytes``/``cache_audio_from_bytes`` validate
    the actual bytes (magic-byte sniff / container sniff), not this extension."""
    if filename and (suffix := Path(filename).suffix):
        return suffix
    base_mime = (mimetype or "").split(";")[0].strip()
    return (mimetypes.guess_extension(base_mime) if base_mime else None) or default


class WahaAdapter(WhatsAppBehaviorMixin, BasePlatformAdapter):
    """Transport over a remote WAHA instance's REST API + webhook; behavior lives in
    ``WhatsAppBehaviorMixin``. config.extra: base_url, session, api_key, hmac_secret, host (webhook
    listener), port, dm_policy / group_policy (open|allowlist|disabled|pairing), allow_from /
    group_allow_from."""

    ALLOW_ALL_ENV_PREFIX = "WAHA"
    splits_long_messages = True  # send() chunks via truncate_message()

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("waha"))
        extra = config.extra or {}
        self._base_url = str(
            _extra_or_secret(extra, "base_url", "WAHA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self._session = str(_extra_or_secret(extra, "session", "WAHA_SESSION", DEFAULT_SESSION))
        self._api_key = _get_scoped_secret("WAHA_API_KEY", "") or extra.get("api_key", "")
        self._hmac_secret = _get_scoped_secret("WAHA_HMAC_SECRET", "") or extra.get("hmac_secret", "")
        self._host: Optional[str] = _extra_or_secret(
            extra, "host", "WAHA_WEBHOOK_HOST", DEFAULT_WEBHOOK_HOST) or None
        self._port = coerce_port(
            _extra_or_secret(extra, "port", "WAHA_WEBHOOK_PORT", DEFAULT_WEBHOOK_PORT), DEFAULT_WEBHOOK_PORT)
        self._max_body_bytes = int(extra.get("max_body_bytes", 1_048_576))
        self._auto_register_webhook = extra.get("auto_register_webhook", True)
        self._reply_prefix: Optional[str] = extra.get("reply_prefix")
        self._dm_policy = str(_extra_or_secret(extra, "dm_policy", "WAHA_DM_POLICY", "pairing")).strip().lower()
        self._allow_from = self._coerce_allow_list(
            self._select_dm_allowlist(extra, ("WAHA_ALLOWED_USERS",), _get_scoped_secret))
        self._group_policy = str(
            _extra_or_secret(extra, "group_policy", "WAHA_GROUP_POLICY", "pairing")).strip().lower()
        _, raw_groups = self._select_allowlist(
            extra, ("group_allow_from", "groupAllowFrom"),
            ("WAHA_GROUP_ALLOW_FROM", "WAHA_GROUP_ALLOWED_USERS"), _get_scoped_secret)
        self._group_allow_from = self._coerce_allow_list(raw_groups)
        self._mention_patterns = self._compile_mention_patterns()
        self._bot_own_digits: str = ""
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._runner = None
        self._dedup = MessageDeduplicator()

    @property
    def name(self) -> str:
        return "WAHA"

    def _fail(self, code: str, message: str, *, retryable: bool) -> bool:
        self._set_fatal_error(code, message, retryable=retryable)
        return False

    def _auth_headers(self) -> dict:
        return {"X-Api-Key": self._api_key} if self._api_key else {}

    # ------------------------------------------------------------------ lifecycle

    async def _check_session(self) -> "tuple[bool, Optional[dict], Optional[str]]":
        """GET /api/sessions/{session}; WORKING is the only status that means ready-to-use."""
        try:
            async with self._http_session.get(
                    f"{self._base_url}/api/sessions/{self._session}", headers=self._auth_headers(),
                    timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return False, None, f"WAHA session check failed: HTTP {resp.status} ({await resp.text()[:200]})"
                data = await resp.json()
        except Exception as e:
            return False, None, f"WAHA unreachable at {self._base_url}: {e}"
        status = str((data or {}).get("status") or "")
        if status != "WORKING":
            return False, data, (
                f"WAHA session '{self._session}' is not WORKING (status={status or 'unknown'}) — "
                f"finish pairing it (QR/code) in the WAHA dashboard, then reconnect")
        return True, data, None

    async def _ensure_webhook_registered(self, existing_config: Optional[dict]) -> None:
        """Best-effort: point WAHA's own webhook config at this listener, preserving any other
        configured webhooks. Never blocks connect() — a user may have wired this manually."""
        if not self._auto_register_webhook:
            return
        from gateway.platforms.shared_ingress import listener_base_url
        expected_url = f"{listener_base_url(self._host, self._port)}/waha/webhook"
        webhooks = list(((existing_config or {}).get("webhooks")) or [])
        if any((hook or {}).get("url") == expected_url for hook in webhooks):
            return
        hook: Dict[str, Any] = {"url": expected_url, "events": ["message"]}
        if self._hmac_secret:
            hook["hmac"] = {"key": self._hmac_secret}
        webhooks.append(hook)
        body = {"config": {**(existing_config or {}), "webhooks": webhooks}}
        try:
            async with self._http_session.post(
                    f"{self._base_url}/api/sessions/{self._session}", json=body,
                    headers=self._auth_headers(), timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status >= 400:
                    logger.warning(
                        "[waha] Could not auto-register webhook %s on session '%s' (HTTP %s) — "
                        "configure it manually in WAHA if events don't arrive", expected_url,
                        self._session, resp.status)
                else:
                    logger.info("[waha] Registered webhook %s on session '%s'", expected_url, self._session)
        except Exception as e:
            logger.warning("[waha] Webhook auto-registration failed (configure manually): %s", e)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not AIOHTTP_AVAILABLE:
            return self._fail("missing_dependency", "aiohttp is required for the WAHA adapter", retryable=False)
        if not self._base_url:
            return self._fail("config_missing", "WAHA_BASE_URL must be set", retryable=False)
        if not self._hmac_secret and not _is_loopback_host(self._host):
            return self._fail(
                "config_missing",
                "WAHA_HMAC_SECRET is required when the webhook listener is bound to a "
                "non-loopback host (set it to the session's webhook hmac.key, or bind "
                "WAHA_WEBHOOK_HOST=127.0.0.1 for a local-only setup)", retryable=False)
        if not self._acquire_platform_lock(
                "waha", f"{self._base_url}|{self._session}",
                f"WAHA session '{self._session}' at {self._base_url}"):
            return False
        self._http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        ok, info, err = await self._check_session()
        if not ok:
            await self._http_session.close()
            self._http_session = None
            self._release_platform_lock()
            logger.error("[waha] %s", err)
            return self._fail("session_not_working", err or "WAHA session not reachable", retryable=True)
        me = (info or {}).get("me") or {}
        self._bot_own_digits = _digits(me.get("id"))
        app = web.Application(client_max_size=self._max_body_bytes)
        app.router.add_post("/waha/webhook", self._handle_webhook)
        app.router.add_get("/health", lambda _r: web.json_response({"status": "ok", "platform": "waha"}))
        from gateway.platforms.shared_ingress import bind_listener, listener_base_url
        self._runner = await bind_listener(self, app, self._host, self._port, "/waha/webhook")
        await self._ensure_webhook_registered((info or {}).get("config"))
        self._mark_connected(listener_base=listener_base_url(self._host, self._port))
        logger.info(
            "[waha] Connected to session '%s' at %s (bot=%s); webhook listening on %s:%s",
            self._session, self._base_url, redact_phone(self._bot_own_digits), self._host, self._port)
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        with contextlib.suppress(Exception):
            self._release_platform_lock()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
        self._mark_disconnected()
        logger.info("[waha] Disconnected")

    # ------------------------------------------------------------------ outbound

    async def _api_post(self, path: str, body: dict, *, timeout: float = 30) -> SendResult:
        if not self._http_session:
            return SendResult(success=False, error="WAHA adapter not connected")
        try:
            async with self._http_session.post(
                    f"{self._base_url}{path}", json=body, headers=self._auth_headers(),
                    timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    return SendResult(success=False, error=f"WAHA {path} failed: HTTP {resp.status}: {text[:300]}")
                try:
                    data = json.loads(text) if text else {}
                except ValueError:
                    data = {}
                message_id = data.get("id") if isinstance(data, dict) else None
                if isinstance(message_id, dict):
                    message_id = message_id.get("id") or message_id.get("_serialized")
                return SendResult(success=True, message_id=str(message_id) if message_id else None, raw_response=data)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Format markdown for WhatsApp, chunk preserving code blocks, send sequentially."""
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)
        target = _waha_chat_id(chat_id)
        chunks = self.truncate_message(self.format_message(content), self._outgoing_chunk_limit())
        last_id, sent_ids = None, []
        for idx, chunk in enumerate(chunks):
            payload: Dict[str, Any] = {"session": self._session, "chatId": target, "text": chunk}
            if reply_to and idx == 0:
                payload["reply_to"] = reply_to
            result = await self._api_post("/api/sendText", payload)
            if not result.success:
                return result
            last_id = result.message_id
            if last_id:
                sent_ids.append(last_id)
            if len(chunks) > 1:
                import asyncio
                await asyncio.sleep(0.3)  # avoid rate limiting between chunks
        return SendResult(success=True, message_id=last_id, continuation_message_ids=tuple(sent_ids[:-1]),
                          raw_response={"message_ids": sent_ids})

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        with contextlib.suppress(Exception):
            await self._api_post(
                "/api/startTyping", {"session": self._session, "chatId": _waha_chat_id(chat_id)}, timeout=5)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        cid = _waha_chat_id(chat_id)
        return {"name": cid, "type": "group" if _is_group_jid(cid) else "dm", "chat_id": cid}

    async def _send_local_media(self, endpoint: str, chat_id: str, path: str, caption: Optional[str], *,
                                mime_default: str, filename: Optional[str] = None) -> SendResult:
        if not os.path.exists(path):
            return SendResult(success=False, error=f"File not found: {path}")
        size = os.path.getsize(path)
        if size > _MAX_LOCAL_MEDIA_BYTES:
            return SendResult(
                success=False,
                error=f"File too large for WAHA base64 upload ({size} bytes > {_MAX_LOCAL_MEDIA_BYTES})")
        with open(path, "rb") as fh:
            raw = fh.read()
        mime = mimetypes.guess_type(path)[0] or mime_default
        payload: Dict[str, Any] = {
            "session": self._session, "chatId": _waha_chat_id(chat_id),
            "file": {"mimetype": mime, "filename": filename or os.path.basename(path),
                     "data": base64.b64encode(raw).decode("ascii")}}
        if caption:
            payload["caption"] = caption
        return await self._api_post(endpoint, payload, timeout=120)

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        payload: Dict[str, Any] = {
            "session": self._session, "chatId": _waha_chat_id(chat_id),
            "file": {"mimetype": "image/jpeg", "url": image_url}}
        if caption:
            payload["caption"] = caption
        return await self._api_post("/api/sendImage", payload, timeout=60)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None,
                              reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_local_media("/api/sendImage", chat_id, image_path, caption, mime_default="image/jpeg")

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_local_media("/api/sendVideo", chat_id, video_path, caption, mime_default="video/mp4")

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_local_media(
            "/api/sendVoice", chat_id, audio_path, caption, mime_default="audio/ogg; codecs=opus")

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            file_name: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self._send_local_media(
            "/api/sendFile", chat_id, file_path, caption, mime_default="application/octet-stream",
            filename=file_name or os.path.basename(file_path))

    # ------------------------------------------------------------------ inbound

    def _verify_hmac(self, headers, raw_body: bytes) -> bool:
        """``X-Webhook-Hmac`` = hex HMAC-SHA512 of the raw body (WAHA's documented scheme).
        No secret configured => trusted only because connect() already required a loopback bind."""
        if not self._hmac_secret:
            return True
        provided = headers.get("X-Webhook-Hmac", "")
        algorithm = (headers.get("X-Webhook-Hmac-Algorithm", "") or "").strip().lower()
        if not provided or algorithm != "sha512":
            return False
        expected = hmac.new(self._hmac_secret.encode(), raw_body, hashlib.sha512).hexdigest()
        return hmac.compare_digest(provided, expected)

    async def _download_media_bytes(self, url: str) -> Optional[bytes]:
        """Authenticated, streamed-and-capped GET of a WAHA ``media.url`` (``/api/files/{id}``,
        same ``X-Api-Key`` as every other WAHA endpoint by default). Streams under
        ``gateway.max_inbound_media_bytes`` instead of buffering the whole body first — the cap
        itself is the shared ``validate_inbound_media_size`` every adapter's cache_*_from_bytes
        path enforces, not a second implementation of the limit."""
        max_bytes = get_inbound_media_max_bytes()
        try:
            async with self._http_session.get(
                    url, headers=self._auth_headers(), timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    logger.warning("[waha] Media download failed: HTTP %s for %s", resp.status, url)
                    return None
                content_length = resp.headers.get("Content-Length")
                if content_length:
                    try:
                        validate_inbound_media_size(int(content_length), media_type="media", max_bytes=max_bytes)
                    except (ValueError, TypeError) as e:
                        logger.warning("[waha] Rejected inbound media (declared size): %s", e)
                        return None
                chunks: List[bytes] = []
                total = 0
                async for chunk in resp.content.iter_chunked(65536):
                    total += len(chunk)
                    try:
                        validate_inbound_media_size(total, media_type="media", max_bytes=max_bytes)
                    except ValueError as e:
                        logger.warning("[waha] Rejected inbound media (streamed size): %s", e)
                        return None
                    chunks.append(chunk)
                return b"".join(chunks)
        except Exception as e:
            logger.warning("[waha] Media download error for %s: %s", url, e)
            return None

    async def _collect_inbound_media(
            self, payload: Dict[str, Any], data: Dict[str, Any]) -> tuple[MessageType, list, list]:
        """``hasMedia`` -> ``(message_type, media_urls, media_types)`` for image/voice attachments
        (this adapter's scoped content types); anything else (video, document, sticker) is left
        unhandled for now and reported at debug level. Save+validate is NOT reimplemented here —
        downloaded bytes go straight into ``cache_image_from_bytes``/``cache_audio_from_bytes``."""
        if not payload.get("hasMedia"):
            return MessageType.TEXT, [], []
        media = payload.get("media") or {}
        media_url, mimetype = media.get("url"), str(media.get("mimetype") or "").lower()
        if media.get("error"):
            logger.warning("[waha] WAHA reported a media download error, skipping attachment: %s", media["error"])
            return MessageType.TEXT, [], []
        if not media_url:
            logger.info("[waha] hasMedia=true but media.url is null (not downloaded by WAHA) — skipping attachment")
            return MessageType.TEXT, [], []
        if mimetype.startswith("image/"):
            raw = await self._download_media_bytes(media_url)
            if raw is None:
                return MessageType.TEXT, [], []
            ext = _guess_media_ext(mimetype, media.get("filename"), DEFAULT_IMAGE_EXT)
            try:
                path = await cache_image_from_bytes_async(raw, ext=ext)
            except ValueError as e:
                logger.warning("[waha] Rejected inbound image: %s", e)
                return MessageType.TEXT, [], []
            return MessageType.PHOTO, [path], [mimetype or "image/jpeg"]
        if mimetype.startswith("audio/"):
            raw = await self._download_media_bytes(media_url)
            if raw is None:
                return MessageType.TEXT, [], []
            ext = _guess_media_ext(mimetype, media.get("filename"), DEFAULT_AUDIO_EXT)
            path = await cache_audio_from_bytes_async(raw, ext=ext)
            msg_type = MessageType.VOICE if _is_ptt_audio(data) else MessageType.AUDIO
            return msg_type, [path], [mimetype or "audio/ogg"]
        logger.debug("[waha] Skipping unsupported inbound media type: %s", mimetype or "unknown")
        return MessageType.TEXT, [], []

    async def _build_message_event(self, payload: Dict[str, Any]) -> Optional[MessageEvent]:
        """Normalize a WAHA ``message`` event payload into a MessageEvent, LID-robust (see module
        docstring): identity is resolved per-message by preferring a non-``@lid`` JID candidate."""
        data = payload.get("_data") or {}
        key = data.get("key") or {}
        raw_chat = payload.get("from") or key.get("remoteJid") or ""
        is_group = _is_group_jid(raw_chat)
        if is_group:
            chat_id = str(raw_chat)
            sender_id = _prefer_phone_jid([payload.get("participant"), key.get("participant"),
                                          key.get("participantAlt")])
        else:
            chat_id = _prefer_phone_jid([raw_chat, key.get("remoteJid"), key.get("remoteJidAlt")])
            sender_id = chat_id
        if not chat_id or not sender_id:
            return None
        normalized: Dict[str, Any] = {
            "chatId": chat_id, "isGroup": is_group, "senderId": sender_id, "from": sender_id,
            "body": payload.get("body") or "",
            "botIds": [self._bot_own_digits] if self._bot_own_digits else [],
            "mentionedIds": _extract_mentioned_ids(data), "quotedParticipant": None,
        }
        if not self._should_process_message(normalized):
            return None
        # Media is downloaded only AFTER the authorization/gating check passes above — an
        # unauthorized sender's attachment is never fetched.
        message_type, media_urls, media_types = await self._collect_inbound_media(payload, data)
        chat_name = payload.get("pushName") or data.get("notifyName") or None
        body = normalized["body"]
        if is_group:
            body = self._clean_bot_mention_text(body, normalized)
        message_id = str(payload.get("id") or "") or None
        source = self.build_source(
            chat_id=chat_id, chat_name=chat_name, chat_type="group" if is_group else "dm",
            user_id=sender_id, user_name=chat_name, message_id=message_id)
        return MessageEvent(
            text=body, message_type=message_type, source=source, raw_message=payload,
            message_id=message_id, media_urls=media_urls, media_types=media_types)

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        if (request.content_length or 0) > self._max_body_bytes:
            return web.Response(status=413)
        try:
            raw = await request.read()
        except Exception as e:
            logger.error("[waha] Failed to read webhook body: %s", e)
            return web.Response(status=400)
        if len(raw) > self._max_body_bytes:
            return web.Response(status=413)
        if not self._verify_hmac(request.headers, raw):
            logger.warning("[waha] Rejected webhook: invalid or missing HMAC signature")
            return web.json_response({"error": "invalid signature"}, status=401)
        try:
            body = json.loads(raw)
        except ValueError:
            return web.json_response({"error": "bad json"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"status": "ignored"})
        session_name = str(body.get("session") or "")
        if session_name and self._session and session_name != self._session:
            return web.json_response({"status": "ignored", "reason": "other session"})
        if body.get("event") != "message":
            return web.json_response({"status": "ignored", "event": body.get("event")})
        payload = body.get("payload") or {}
        if payload.get("fromMe"):
            return web.json_response({"status": "ignored", "reason": "own message"})
        msg_id = str(payload.get("id") or "")
        if msg_id and self._dedup.is_duplicate(msg_id):
            return web.json_response({"status": "duplicate"})
        try:
            event = await self._build_message_event(payload)
        except Exception as e:
            logger.error("[waha] Error building message event: %s", e, exc_info=True)
            return web.json_response({"status": "error"}, status=200)  # ack — WAHA would otherwise retry forever
        if event is None:
            return web.json_response({"status": "ignored"})
        import asyncio
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return web.json_response({"status": "accepted"}, status=202)


# ── Plugin glue: register(ctx) + the standalone/env/yaml hooks ADDING_A_PLATFORM.md describes.

_YAML_BRIDGE = (  # (yaml key, env var, kind) for apply_yaml_bridge
    ("base_url", "WAHA_BASE_URL", "str"), ("session", "WAHA_SESSION", "str"),
    ("require_mention", "WHATSAPP_REQUIRE_MENTION", "lower"),  # shared with the Baileys mixin (see adapter docstring)
    ("dm_policy", "WAHA_DM_POLICY", "lower"), ("group_policy", "WAHA_GROUP_POLICY", "lower"),
    ("allow_from", "WAHA_ALLOWED_USERS", "csv"), ("group_allow_from", "WAHA_GROUP_ALLOWED_USERS", "csv"),
)


def _apply_yaml_config(yaml_cfg: dict, waha_cfg: dict) -> Optional[dict]:
    """``apply_yaml_config_fn``: config.yaml waha: keys -> WAHA_* env (env wins) + extra."""
    return _apply_yaml_bridge(waha_cfg, _YAML_BRIDGE)


def _env_enablement() -> Optional[dict]:
    """``env_enablement_fn``: seed PlatformConfig.extra from env BEFORE adapter construction, so an
    env-only setup shows up in ``hermes gateway status`` before the adapter is instantiated."""
    base_url = _get_scoped_secret("WAHA_BASE_URL", "").strip()
    if not base_url:
        return None
    seed = _seed_extra_from_env((
        ("WAHA_SESSION", "session", None), ("WAHA_WEBHOOK_HOST", "host", None),
        ("WAHA_WEBHOOK_PORT", "port", int), ("WAHA_DM_POLICY", "dm_policy", None),
        ("WAHA_GROUP_POLICY", "group_policy", None),
    ), home_env="WAHA_HOME_CHANNEL")
    return {"base_url": base_url, **seed}


def _is_connected(config) -> bool:
    return bool(_get_scoped_secret("WAHA_BASE_URL", "").strip())


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None,
                           force_document=False, caption=None, mentions=None):
    """Out-of-process delivery via the WAHA REST API (standalone_sender_fn: cron apart from the
    gateway) — plain ``/api/sendText``, no webhook/session-health involved."""
    if not AIOHTTP_AVAILABLE:
        return send_error("aiohttp not installed. Run: pip install aiohttp")
    extra = getattr(pconfig, "extra", {}) or {}
    base_url = str(_extra_or_secret(extra, "base_url", "WAHA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
    session = str(_extra_or_secret(extra, "session", "WAHA_SESSION", DEFAULT_SESSION))
    api_key = _get_scoped_secret("WAHA_API_KEY", "") or extra.get("api_key", "")
    if not base_url:
        return send_error("WAHA not configured (WAHA_BASE_URL required)")
    headers = {"X-Api-Key": api_key} if api_key else {}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as http_session:
            async with http_session.post(
                    f"{base_url}/api/sendText", headers=headers,
                    json={"session": session, "chatId": _waha_chat_id(chat_id), "text": message}) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    return send_error(f"WAHA sendText failed: HTTP {resp.status}: {text[:300]}")
                try:
                    data = json.loads(text) if text else {}
                except ValueError:
                    data = {}
                return {"success": True, "platform": "waha", "chat_id": chat_id, "message_id": data.get("id", "")}
    except Exception as e:
        return send_error(f"WAHA send failed: {e}")


def interactive_setup() -> None:
    """``hermes gateway setup`` flow (lazy hermes_cli imports keep the plugin importable outside the CLI)."""
    from hermes_cli.setup import (
        prompt, prompt_yes_no, save_env_value, get_env_value, print_header, print_info, print_success)
    from hermes_cli.setup_platforms import declines_reconfigure

    print_header("WAHA (WhatsApp)")
    if declines_reconfigure("WAHA", "Reconfigure WAHA?", "WAHA_BASE_URL"):
        return
    print_info("WAHA is a separately-run Docker service — this only configures how Hermes talks to it.")
    base_url = prompt("WAHA base URL", default=get_env_value("WAHA_BASE_URL") or DEFAULT_BASE_URL)
    if not base_url:
        return
    save_env_value("WAHA_BASE_URL", base_url.rstrip("/"))
    save_env_value("WAHA_SESSION", prompt("WAHA session name", default=get_env_value("WAHA_SESSION") or DEFAULT_SESSION))
    if api_key := prompt("WAHA API key (leave blank if none)", password=True):
        save_env_value("WAHA_API_KEY", api_key)
    if hmac_secret := prompt("Webhook HMAC secret (leave blank for loopback-only setups)", password=True):
        save_env_value("WAHA_HMAC_SECRET", hmac_secret)
    allowed = prompt("Allowed user phone numbers (comma-separated, leave empty to deny everyone)",
                     default=get_env_value("WAHA_ALLOWED_USERS") or "")
    save_env_value("WAHA_ALLOWED_USERS", allowed.replace(" ", "") if allowed else "")
    if allowed:
        print_success("WAHA allowlist configured")
    home_channel = prompt("Home chat ID for cron delivery (leave empty to skip)").strip()
    if home_channel:
        save_env_value("WAHA_HOME_CHANNEL", home_channel)
    print_success("WAHA configuration saved to ~/.hermes/.env")
    print_info("Restart the gateway for changes to take effect: hermes gateway restart")


def register(ctx) -> None:
    """Plugin entry point: called by the Hermes plugin system."""
    ctx.register_platform(
        name="waha", label="WhatsApp (WAHA)", adapter_factory=WahaAdapter, check_fn=check_waha_requirements,
        is_connected=_is_connected, required_env=["WAHA_BASE_URL"],
        install_hint="pip install aiohttp (a separately-run WAHA instance is also required)",
        setup_fn=interactive_setup, apply_yaml_config_fn=_apply_yaml_config, env_enablement_fn=_env_enablement,
        allowed_users_env="WAHA_ALLOWED_USERS", allow_all_env="WAHA_ALLOW_ALL_USERS",
        cron_deliver_env_var="WAHA_HOME_CHANNEL", standalone_sender_fn=_standalone_send,
        max_message_length=4096, emoji="💬", allow_update_command=True,
        platform_hint=(
            "You are chatting via WhatsApp (bridged through WAHA). Use WhatsApp markdown: "
            "*bold*, _italic_, ~strikethrough~, ``` for code blocks. Keep responses concise."))
