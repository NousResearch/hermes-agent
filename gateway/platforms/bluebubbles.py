"""BlueBubbles iMessage platform adapter: local BlueBubbles macOS server for outbound REST sends and
inbound webhooks (text, media attachments, typing indicators, read receipts)."""

import asyncio
import json
import logging
import os
import re
import uuid
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, quote

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms._shared import extra_or_secret as _extra_or_secret, get_scoped_secret as _get_scoped_secret
from gateway.platforms.base import (
    BasePlatformAdapter, SendResult,
    cache_image_from_bytes_async, cache_audio_from_bytes_async, cache_document_from_bytes_async,
)
from gateway.platforms.event import MessageEvent, MessageType
from .media_cache import ext_for_mime
from gateway.platforms.helpers import compile_mention_patterns, strip_markdown
from utils import TRUTHY_STRINGS

# Historical BlueBubbles mime→ext maps, preserved verbatim as overrides for the shared dispatch in
# gateway.platforms.media_cache. Both maps are CLOSED: unlisted mimes fall back to .jpg / .mp3.
_BLUEBUBBLES_IMAGE_EXT_OVERRIDES = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp",
    "image/heic": ".jpg", "image/heif": ".jpg", "image/tiff": ".jpg",  # historical mapping
}
_BLUEBUBBLES_AUDIO_EXT_OVERRIDES = {
    "audio/mp3": ".mp3", "audio/mpeg": ".mp3", "audio/ogg": ".ogg", "audio/wav": ".wav",
    "audio/x-caf": ".mp3", "audio/mp4": ".m4a",
    "audio/aac": ".m4a",  # historical mapping (shared table says .aac)
}

logger = logging.getLogger(__name__)

DEFAULT_WEBHOOK_HOST = "127.0.0.1"
# Webhook events are small JSON/form payloads (attachments come through the REST API); 1 MiB keeps
# oversized/chunked bodies from buffering unbounded.
_WEBHOOK_MAX_BODY_BYTES = 1_048_576
DEFAULT_WEBHOOK_PORT = 8645
DEFAULT_WEBHOOK_PATH = "/bluebubbles-webhook"
MAX_TEXT_LENGTH = 4000

# Inbound attachments: Messages.app syncs files to disk asynchronously, so a download right after
# the message event can fail with "Attachment does not exist in disk!" and the attachment would be
# lost forever. The webhook path makes exactly one in-band attempt per attachment (BlueBubbles
# holds the HTTP response open meanwhile), and the background recovery task retries with a bounded
# backoff (total added wait ~85s), then keeps polling every _LATE_ATTACHMENT_RECOVERY_INTERVAL_S
# for a longer window so late-syncing files still get delivered.
_ATTACHMENT_DOWNLOAD_RETRY_DELAYS = (5.0, 20.0, 60.0)
_LATE_ATTACHMENT_RECOVERY_INTERVAL_S = 90.0
_LATE_ATTACHMENT_RECOVERY_WINDOW_S = 15 * 60.0
_LATE_ATTACHMENT_RECOVERY_CAP = 200  # LRU cap of attachment guids already delivered

# iMessage has no stable bot mention identity (unlike <@U...>/@botname/MXID), so
# `require_mention: true` without custom aliases uses Hermes wake words.
DEFAULT_MENTION_PATTERNS = [r"(?<![\w@])@?hermes\s+agent\b[,:\-]?", r"(?<![\w@])@?hermes\b[,:\-]?"]

# Tapback associatedMessageType codes: 2000-2005 added, 3000-3005 removed (love, like, dislike, ...).
_TAPBACK_CODES = {*range(2000, 2006), *range(3000, 3006)}
_MESSAGE_EVENTS = {"new-message", "message", "updated-message"}  # webhook event types carrying user messages

_PHONE_RE = re.compile(r"\+?\d{7,15}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PAGINATION_SUFFIX_RE = re.compile(r"\s*\(\d+/\d+\)$")
_ADDRESS_RE = re.compile(r"^\+\d+")

_GUID_CACHE_SIZE = 500  # LRU cap for resolved chat-GUID lookups
_LOCAL_HOSTS = {"0.0.0.0", "127.0.0.1", "localhost", "::"}


def _redact(text: str) -> str:
    """Redact phone numbers and emails from log output."""
    return _EMAIL_RE.sub("[REDACTED]", _PHONE_RE.sub("[REDACTED]", text))


def check_bluebubbles_requirements() -> bool:
    try:
        import aiohttp
    except ImportError:
        return False
    return True


def _normalize_server_url(raw: str) -> str:
    value = (raw or "").strip()
    if value and not re.match(r"^https?://", value, flags=re.IGNORECASE):
        value = f"http://{value}"
    return value.rstrip("/")


def _closed_ext(mime: str, overrides: dict[str, str], fallback: str) -> str:
    """Historical maps were closed: unlisted mimes fall back without consulting mimetypes."""
    return ext_for_mime(mime, overrides=overrides, use_defaults=False, use_mimetypes=False,
                        fallback=fallback) or fallback


def _temp_guid() -> str:
    return f"temp-{datetime.utcnow().timestamp()}"


def _ok():
    """Plain ``ok`` acknowledgement for webhook events we accept but don't process."""
    from aiohttp import web
    return web.Response(text="ok")


class BlueBubblesAdapter(BasePlatformAdapter):
    # Answers /p/<profile>/... on the default listener for a served secondary (shared_ingress).
    serves_profile_prefix: bool = True
    platform = Platform.BLUEBUBBLES
    SUPPORTS_MESSAGE_EDITING = False
    MAX_MESSAGE_LENGTH = MAX_TEXT_LENGTH
    splits_long_messages = True  # send() chunks via truncate_message(MAX_MESSAGE_LENGTH)

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.BLUEBUBBLES)
        extra = config.extra or {}
        self.server_url = _normalize_server_url(_extra_or_secret(extra, "server_url", "BLUEBUBBLES_SERVER_URL"))
        self.password = extra.get("password") or _get_scoped_secret("BLUEBUBBLES_PASSWORD", "")
        self.webhook_host = _extra_or_secret(extra, "webhook_host", "BLUEBUBBLES_WEBHOOK_HOST", DEFAULT_WEBHOOK_HOST)
        self.webhook_port = int(_extra_or_secret(extra, "webhook_port", "BLUEBUBBLES_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT)))
        path = str(_extra_or_secret(extra, "webhook_path", "BLUEBUBBLES_WEBHOOK_PATH", DEFAULT_WEBHOOK_PATH))
        self.webhook_path = path if path.startswith("/") else f"/{path}"
        self.send_read_receipts = bool(extra.get("send_read_receipts", True))
        _require_mention = extra.get("require_mention")
        if _require_mention is None:
            _require_mention = _get_scoped_secret("BLUEBUBBLES_REQUIRE_MENTION")
        self.require_mention = str(_require_mention).strip().lower() in TRUTHY_STRINGS
        self._mention_patterns = self._compile_mention_patterns(
            extra["mention_patterns"] if "mention_patterns" in extra else _get_scoped_secret("BLUEBUBBLES_MENTION_PATTERNS"))
        self.client: Optional[httpx.AsyncClient] = None
        self._runner = None
        self._private_api_enabled: Optional[bool] = None
        self._helper_connected: bool = False
        self._guid_cache: OrderedDict[str, str] = OrderedDict()
        # Attachment delivery bookkeeping: guids with a delivery in flight (dispatch until
        # gateway acceptance), and recently accepted delivery guids (bounded LRU, so the same
        # file is not delivered twice — by late recovery or a replayed webhook event).
        self._late_recovery_pending: Set[str] = set()
        self._late_recovery_done: OrderedDict[str, None] = OrderedDict()
        self._delivery_inflight: Set[str] = set()

    # --- API helpers ---

    def _api_url(self, path: str) -> str:
        return f"{self.server_url}{path}{'&' if '?' in path else '?'}password={quote(self.password, safe='')}"

    @staticmethod
    def _compile_mention_patterns(raw: Any) -> list[re.Pattern]:
        """Compile group-mention wake words; ``raw`` is a list, a raw env string (JSON list or
        comma/newline-separated), or None (Hermes defaults)."""
        return compile_mention_patterns(raw, log_prefix="bluebubbles", defaults=DEFAULT_MENTION_PATTERNS,
                                        logger_=logger)

    def _message_matches_mention_patterns(self, text: str) -> bool:
        return bool(text) and any(pattern.search(text) for pattern in self._mention_patterns)

    def _clean_mention_text(self, text: str) -> str:
        """Strip a leading wake word only — patterns are regexes, so stripping anywhere later in the
        prompt could delete ordinary words."""
        stripped = (text or "").lstrip()
        for pattern in self._mention_patterns:
            if match := pattern.match(stripped):
                return stripped[match.end():].lstrip(" ,:-") or text
        return text

    async def _api_json(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        """Authenticated request to the BlueBubbles REST API; raises on HTTP errors, returns decoded JSON."""
        assert self.client is not None
        res = await getattr(self.client, method)(self._api_url(path), **kwargs)
        res.raise_for_status()
        return res.json()

    async def _api_get(self, path: str) -> dict[str, Any]:
        return await self._api_json("get", path)

    async def _api_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._api_json("post", path, json=payload)

    async def _post_message(self, path: str, payload: dict[str, Any]) -> SendResult:
        """POST a message payload and wrap the outcome as a SendResult."""
        try:
            res = await self._api_post(path, payload)
            data = res.get("data") or {}
            msg_id = str(data.get("guid") or data.get("messageGuid") or "ok")
            return SendResult(success=True, message_id=msg_id, raw_response=res)
        except Exception as exc:
            return SendResult(success=False, error=str(exc) or type(exc).__name__)

    async def _private_api_chat_call(self, chat_id: str, action: str, method: str) -> bool:
        """Fire a private-API chat action (typing/read); True only if the call was made."""
        if not self._private_api_enabled or not self._helper_connected or not self.client:
            return False
        with suppress(Exception):
            if guid := await self._resolve_chat_guid(chat_id):
                url = self._api_url(f"/api/v1/chat/{quote(guid, safe='')}/{action}")
                await getattr(self.client, method)(url, timeout=5)
                return True
        return False

    # --- Lifecycle ---

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.server_url or not self.password:
            logger.error("[bluebubbles] BLUEBUBBLES_SERVER_URL and BLUEBUBBLES_PASSWORD are required")
            return False
        from aiohttp import web
        # Tighter keepalive so idle CLOSE_WAIT drains promptly.
        # See #18451.
        from gateway.platforms._http_client_limits import platform_httpx_limits
        self.client = httpx.AsyncClient(timeout=30.0, limits=platform_httpx_limits())
        try:
            await self._api_get("/api/v1/ping")
            info = await self._api_get("/api/v1/server/info")
            server_data = (info or {}).get("data", {})
            self._private_api_enabled = bool(server_data.get("private_api"))
            self._helper_connected = bool(server_data.get("helper_connected"))
            logger.info("[bluebubbles] connected to %s (private_api=%s, helper=%s)",
                        self.server_url, self._private_api_enabled, self._helper_connected)
        except Exception as exc:
            logger.error("[bluebubbles] cannot reach server at %s: %s", self.server_url, exc)
            await self._close_client()
            return False
        # client_max_size makes aiohttp enforce the cap on every read path, incl. chunked requests
        # with no Content-Length.
        # Explicit body cap: BlueBubbles webhook events are small JSON (or form-encoded) payloads.
        # client_max_size makes aiohttp enforce the cap on every read path — including chunked requests that
        # carry no Content-Length (same pattern as webhook.py / raft, #58536/#58902).
        app = web.Application(client_max_size=_WEBHOOK_MAX_BODY_BYTES)
        app.router.add_get("/health", lambda _: web.Response(text="ok"))
        app.router.add_post(self.webhook_path, self._handle_webhook)
        # The webhook auth value rides in the query string (BlueBubbles cannot send custom headers)
        # — keep it out of aiohttp access logs.
        # Shared-listener mode (multiplex secondary): no bind; served at /p/<profile>/<webhook_path>.
        from gateway.platforms.shared_ingress import bind_listener
        self._runner = await bind_listener(
            self, app, self.webhook_host, self.webhook_port, self.webhook_path, access_log=None)
        self._mark_connected()
        if self._runner is not None:
            logger.info("[bluebubbles] webhook listening on http://%s:%s%s", self.webhook_host, self.webhook_port,
                        self.webhook_path)
        await self._register_webhook()  # the server only sends events to webhooks registered via its API
        # Plugin-registered native handlers (ctx.register_platform_handler).
        self._wire_plugin_handlers(None)
        return True

    async def _close_client(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None

    async def disconnect(self) -> None:
        await self._unregister_webhook()
        await self._close_client()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()

    @property
    def _webhook_url(self) -> str:
        """External webhook URL for BlueBubbles registration (local binds → localhost). In
        shared-listener mode it is the default listener's ``/p/<profile>/`` URL."""
        shared = getattr(self, "_shared_ingress_url", None)
        if shared:
            return shared
        host = "localhost" if self.webhook_host in _LOCAL_HOSTS else self.webhook_host
        return f"http://{host}:{self.webhook_port}{self.webhook_path}"

    def _webhook_register_url_with(self, password_param: str) -> str:
        return f"{self._webhook_url}?password={password_param}" if self.password else self._webhook_url

    @property
    def _webhook_register_url(self) -> str:
        """Registered webhook URL with the password as a query param: BlueBubbles posts to the exact
        registered URL and cannot set custom headers, so this is the only way to authenticate inbound
        webhooks without disabling auth."""
        return self._webhook_register_url_with(quote(self.password, safe=""))

    @property
    def _webhook_register_url_for_log(self) -> str:
        return self._webhook_register_url_with("***")

    async def _find_registered_webhooks(self, url: str) -> list:
        """Return list of BB webhook entries matching *url*."""
        with suppress(Exception):
            data = (await self._api_get("/api/v1/webhook")).get("data")
            if isinstance(data, list):
                return [wh for wh in data if wh.get("url") == url]
        return []

    async def _register_webhook(self) -> bool:
        """Register this webhook URL, reusing an existing registration if present (crash resilience —
        avoids duplicates after an unclean shutdown)."""
        if not self.client:
            return False
        webhook_url, log_url = self._webhook_register_url, self._webhook_register_url_for_log
        if await self._find_registered_webhooks(webhook_url):
            logger.info("[bluebubbles] webhook already registered: %s", log_url)
            return True
        try:
            res = await self._api_post("/api/v1/webhook",
                                       {"url": webhook_url, "events": ["new-message", "updated-message"]})
            status = res.get("status", 0)
            if 200 <= status < 300:
                logger.info("[bluebubbles] webhook registered with server: %s", log_url)
                return True
            logger.warning("[bluebubbles] webhook registration returned status %s: %s", status, res.get("message"))
            return False
        except Exception as exc:
            logger.warning("[bluebubbles] failed to register webhook with server: %s", exc)
            return False

    async def _unregister_webhook(self) -> bool:
        """Remove *all* registrations matching our URL (cleans up crash duplicates)."""
        if not self.client:
            return False
        removed = False
        try:
            for wh in await self._find_registered_webhooks(self._webhook_register_url):
                if wh_id := wh.get("id"):
                    (await self.client.delete(self._api_url(f"/api/v1/webhook/{wh_id}"))).raise_for_status()
                    removed = True
            if removed:
                logger.info("[bluebubbles] webhook unregistered: %s", self._webhook_register_url_for_log)
        except Exception as exc:
            logger.debug("[bluebubbles] failed to unregister webhook (non-critical): %s", exc)
        return removed

    # --- Chat GUID resolution ---

    async def _resolve_chat_guid(self, target: str) -> Optional[str]:
        """Resolve an email/phone to a chat GUID (raw ``a;-;b`` GUIDs pass through). Matches strictly on
        ``chatIdentifier`` / ``identifier``; participant membership is intentionally NOT a fallback —
        the same contact appears in a 1:1 DM and any number of groups, so a participant match could
        leak a DM reply into a group thread. ``None`` lets the caller create a fresh DM.

        See #24157.
        """
        target = (target or "").strip()
        if not target or ";" in target:
            return target or None
        if target in self._guid_cache:
            self._guid_cache.move_to_end(target)
            return self._guid_cache[target]
        with suppress(Exception):
            payload = await self._api_post("/api/v1/chat/query", {"limit": 100, "offset": 0})
            for chat in payload.get("data", []) or []:
                if (chat.get("chatIdentifier") or chat.get("identifier")) != target:
                    continue
                if guid := chat.get("guid") or chat.get("chatGuid"):
                    self._guid_cache[target] = guid
                    while len(self._guid_cache) > _GUID_CACHE_SIZE:
                        self._guid_cache.popitem(last=False)
                return guid
        return None

    async def _create_chat_for_handle(self, address: str, message: str) -> SendResult:
        """Create a new chat by sending the first message to *address*."""
        return await self._post_message(
            "/api/v1/chat/new", {"addresses": [address], "message": message, "tempGuid": _temp_guid()})

    # --- Text sending ---

    @staticmethod
    def truncate_message(content: str, max_length: int = MAX_TEXT_LENGTH) -> list[str]:
        # Base splitter minus "(1/3)" pagination suffixes — iMessage bubbles flow naturally.
        return [_PAGINATION_SUFFIX_RE.sub("", c) for c in BasePlatformAdapter.truncate_message(content, max_length)]

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[dict[str, Any]] = None) -> SendResult:
        text = self.format_message(content)
        if not text:
            return SendResult(success=False, error="BlueBubbles send requires text")
        # Each paragraph becomes its own iMessage bubble; truncate any still too long.
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()] or [text]
        chunks = [c for para in paragraphs for c in (
            [para] if len(para) <= self.MAX_MESSAGE_LENGTH else self.truncate_message(para, self.MAX_MESSAGE_LENGTH))]
        last = SendResult(success=True)
        for chunk in chunks:
            guid = await self._resolve_chat_guid(chat_id)
            if not guid:
                if self._private_api_enabled and ("@" in chat_id or _ADDRESS_RE.match(chat_id)):  # address → new chat
                    return await self._create_chat_for_handle(chat_id, chunk)
                return SendResult(success=False, error=f"BlueBubbles chat not found for target: {chat_id}")
            payload: dict[str, Any] = {"chatGuid": guid, "tempGuid": _temp_guid(), "message": chunk}
            if reply_to and self._private_api_enabled and self._helper_connected:
                payload.update(method="private-api", selectedMessageGuid=reply_to, partIndex=0)
            if not (last := await self._post_message("/api/v1/message/text", payload)).success:
                return last
        return last

    # --- Media sending (outbound) ---

    async def _send_attachment(self, chat_id: str, file_path: str, filename: Optional[str] = None,
                               caption: Optional[str] = None, is_audio_message: bool = False) -> SendResult:
        """Send a file attachment via BlueBubbles multipart upload."""
        if not self.client:
            return SendResult(success=False, error="Not connected")
        if not await asyncio.to_thread(os.path.isfile, file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")
        guid = await self._resolve_chat_guid(chat_id)
        if not guid:
            return SendResult(success=False, error=f"Chat not found: {chat_id}")
        fname = filename or os.path.basename(file_path)
        try:
            # httpx's async multipart iterator reads file objects through a sync chunk generator —
            # read the bytes off the event-loop thread first.
            payload = await asyncio.to_thread(Path(file_path).read_bytes)
            data: dict[str, str] = {"chatGuid": guid, "name": fname, "tempGuid": uuid.uuid4().hex}
            if is_audio_message:
                data["isAudioMessage"] = "true"
            res = await self.client.post(self._api_url("/api/v1/message/attachment"), data=data, timeout=120,
                                         files={"attachment": (fname, payload, "application/octet-stream")})
            res.raise_for_status()
            result = res.json()
            if caption:
                await self.send(chat_id, caption)
            if result.get("status") == 200:
                rdata = result.get("data") or {}
                return SendResult(success=True, message_id=rdata.get("guid") if isinstance(rdata, dict) else None,
                                  raw_response=result)
            return SendResult(success=False, error=result.get("message", "Attachment upload failed"))
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> SendResult:
        try:
            from gateway.platforms.base import cache_image_from_url
            return await self._send_attachment(chat_id, await cache_image_from_url(image_url), caption=caption)
        except Exception:
            return await super().send_image(chat_id, image_url, caption, reply_to)

    async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None, **kw) -> SendResult:
        return await self._send_attachment(chat_id, image_path, caption=caption)

    async def send_voice(self, chat_id, audio_path, caption=None, reply_to=None, **kw) -> SendResult:
        return await self._send_attachment(chat_id, audio_path, caption=caption, is_audio_message=True)

    async def send_video(self, chat_id, video_path, caption=None, reply_to=None, **kw) -> SendResult:
        return await self._send_attachment(chat_id, video_path, caption=caption)

    async def send_document(self, chat_id, file_path, caption=None, file_name=None, reply_to=None, **kw) -> SendResult:
        return await self._send_attachment(chat_id, file_path, filename=file_name, caption=caption)

    async def send_animation(self, chat_id, animation_url, caption=None, reply_to=None, metadata=None) -> SendResult:
        return await self.send_image(chat_id, animation_url, caption, reply_to, metadata)

    # --- Typing indicators / read receipts (private API only) ---

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        await self._private_api_chat_call(chat_id, "typing", "post")

    async def stop_typing(self, chat_id: str) -> None:
        await self._private_api_chat_call(chat_id, "typing", "delete")

    async def mark_read(self, chat_id: str) -> bool:
        return await self._private_api_chat_call(chat_id, "read", "post")

    # --- Chat info ---

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        is_group = ";+;" in (chat_id or "")
        info: dict[str, Any] = {"name": chat_id, "type": "group" if is_group else "dm"}
        with suppress(Exception):
            if guid := await self._resolve_chat_guid(chat_id):
                res = await self._api_get(f"/api/v1/chat/{quote(guid, safe='')}?with=participants")
                data = (res or {}).get("data", {})
                info["name"] = data.get("displayName") or data.get("chatIdentifier") or chat_id
                participants = [addr for p in data.get("participants", []) or []
                                if (addr := (p.get("address") or "").strip())]
                if participants:
                    info["participants"] = participants
        return info

    def format_message(self, content: str) -> str:
        return strip_markdown(content, keep_link_targets=True)  # iMessage auto-links bare URLs only

    # --- Inbound attachment downloading ---

    async def _fetch_attachment_once(self, att_guid: str, att_meta: Dict[str, Any]) -> Optional[str]:
        """Single download attempt: cached local path, or None when not connected. Raises on
        download/caching failure; callers decide about retries and logging."""
        if not self.client:
            return None
        resp = await self.client.get(self._api_url(f"/api/v1/attachment/{quote(att_guid, safe='')}/download"),
                                     timeout=60, follow_redirects=True)
        resp.raise_for_status()
        data = resp.content
        mime = (att_meta.get("mimeType") or "").lower()
        if mime.startswith("image/"):
            return await cache_image_from_bytes_async(data, _closed_ext(mime, _BLUEBUBBLES_IMAGE_EXT_OVERRIDES, ".jpg"))
        if mime.startswith("audio/"):
            return await cache_audio_from_bytes_async(data, _closed_ext(mime, _BLUEBUBBLES_AUDIO_EXT_OVERRIDES, ".mp3"))
        # Videos, documents, and everything else
        return await cache_document_from_bytes_async(data, att_meta.get("transferName", "") or f"file_{uuid.uuid4().hex[:8]}")

    async def _download_attachment(self, att_guid: str, att_meta: Dict[str, Any]) -> Optional[str]:
        """One in-band download attempt; local path or None on failure.

        Messages.app writes inbound attachments to disk asynchronously, so this first attempt can
        fail with "Attachment does not exist in disk!". Failures are NOT waited out here — the
        webhook response is on this path and BlueBubbles holds the HTTP connection open meanwhile.
        Retries live in the background recovery task (_recover_late_attachments), which backs off
        through _ATTACHMENT_DOWNLOAD_RETRY_DELAYS and then keeps polling for a longer window."""
        if not self.client:
            return None
        try:
            return await self._fetch_attachment_once(att_guid, att_meta)
        except Exception as exc:
            logger.debug("[bluebubbles] attachment %s download failed: %s", _redact(att_guid), exc)
            return None

    # --- Webhook handling ---

    def _extract_payload_record(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        data = payload.get("data")
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and (first := next((i for i in data if isinstance(i, dict)), None)):
            return first
        if isinstance(payload.get("message"), dict):
            return payload.get("message")
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _value(*candidates: Any) -> Optional[str]:
        return next((c.strip() for c in candidates if isinstance(c, str) and c.strip()), None)

    @staticmethod
    def _parse_webhook_body(raw: bytes) -> Any:
        """Decode a webhook body: JSON, else form-encoded with a JSON field."""
        body = raw.decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except Exception:
            form = parse_qs(body)
            payload_str = (form.get("payload") or form.get("data") or form.get("message") or [""])[0]
            return json.loads(payload_str) if payload_str else {}

    @staticmethod
    def _message_type_for_attachments(items: List[Tuple[str, Dict[str, Any]]]) -> MessageType:
        """Aggregate one message type from downloaded attachments (``(mime, att)`` pairs).

        Historical rule shared by the webhook and late-recovery paths: the last attachment's
        kind wins, except that any image in a multi-attachment message makes it PHOTO."""
        kinds: List[MessageType] = []
        for mime, att in items:
            is_voice = mime.startswith("audio/") or (att.get("uti") or "").endswith("caf")
            kinds.append(MessageType.PHOTO if mime.startswith("image/") else MessageType.VOICE
                         if is_voice else MessageType.VIDEO if mime.startswith("video/")
                         else MessageType.DOCUMENT)
        if len(kinds) > 1 and MessageType.PHOTO in kinds:
            return MessageType.PHOTO
        return kinds[-1] if kinds else MessageType.TEXT

    async def _collect_attachments(self, record: Dict[str, Any]):
        """Download inbound attachments; returns ``(entries, failed)``.

        ``entries`` pairs every downloaded file with its attachment guid and metadata
        (``(guid, cached_path, mime, att)``). Attachments that could not be downloaded are
        returned in ``failed`` instead of being silently skipped; the caller must surface them
        (see _attachment_failure_notice). Guids already delivered or with a delivery in flight
        are skipped entirely: BlueBubbles replays the same message as both new-message and
        updated-message, and a replay must not download the same file again. Downloading does
        NOT claim the guids — the caller claims the entries before building the event and keeps
        only the media whose claim it won (see _handle_webhook)."""
        entries: List[Tuple[str, str, str, Dict[str, Any]]] = []
        failed: List[Dict[str, Any]] = []
        for att in record.get("attachments") or []:
            att_guid = att.get("guid", "")
            if att_guid and self._attachment_claimed(att_guid):
                logger.debug("[bluebubbles] attachment %s already delivered or in flight; skipping",
                             _redact(att_guid))
                continue
            cached = await self._download_attachment(att_guid, att) if att_guid else None
            if not cached:
                failed.append(att)
                continue
            if att_guid and self._attachment_claimed(att_guid):
                # Claimed by another delivery while this download was in flight: it owns the file.
                logger.debug("[bluebubbles] attachment %s claimed during download; skipping",
                             _redact(att_guid))
                continue
            entries.append((att_guid, cached, (att.get("mimeType") or "").lower(), att))
        return entries, failed

    @staticmethod
    def _attachment_failure_notice(failed: List[Dict[str, Any]]) -> str:
        """Agent-facing notice for attachments BlueBubbles could not serve (never drop silently)."""
        labels: List[str] = []
        for att in failed:
            label = (att.get("transferName") or "").strip() or (att.get("guid") or "").strip() or "attachment"
            labels.append(_redact(label))
        return (f"[attachment download failed: {', '.join(labels) or 'attachment'} — BlueBubbles could not "
                f"retrieve the file yet (it may still be syncing to disk). If it never arrives, ask the "
                f"sender to resend.]")

    def _remember_delivered_attachment(self, att_guid: str) -> None:
        """Track delivered attachment guids (bounded LRU) so late recovery cannot re-deliver them."""
        if not att_guid:
            return
        self._late_recovery_done[att_guid] = None
        self._late_recovery_done.move_to_end(att_guid)
        while len(self._late_recovery_done) > _LATE_ATTACHMENT_RECOVERY_CAP:
            self._late_recovery_done.popitem(last=False)

    def _attachment_claimed(self, att_guid: str) -> bool:
        """True when ``att_guid`` is already delivered or its delivery is currently in flight."""
        return att_guid in self._late_recovery_done or att_guid in self._delivery_inflight

    def _claim_attachment_delivery(self, att_guid: str) -> bool:
        """Reserve ``att_guid`` for a delivery that is about to be dispatched.

        Returns False when the guid was already delivered or is being delivered elsewhere.
        The claim is settled with _settle_attachment_delivery once the gateway accepts or
        rejects the event, so a refused delivery does not keep the attachment marked as done."""
        if not att_guid or self._attachment_claimed(att_guid):
            return False
        self._delivery_inflight.add(att_guid)
        return True

    def _settle_attachment_delivery(self, att_guid: str, accepted: bool) -> None:
        """Release an in-flight delivery claim; accepted deliveries stay marked as done."""
        if not att_guid:
            return
        self._delivery_inflight.discard(att_guid)
        if accepted:
            self._remember_delivered_attachment(att_guid)

    async def _deliver_and_settle(self, event: MessageEvent, claimed_guids: List[str]) -> bool:
        """Dispatch an event and settle its attachment claims against gateway acceptance.

        ``handle_message`` sets ``event._gateway_accepted`` only once the event was actually
        scheduled or queued (refused/dropped events stay False). The receipt is read again in
        ``finally``: a delivery task cancelled after acceptance keeps the accepted work claimed
        (a replay must not deliver it again), while a cancellation before acceptance still
        releases the claims. Returns whether the gateway accepted the event."""
        accepted = False
        try:
            await self.handle_message(event)
            accepted = event._gateway_accepted is True
        except asyncio.CancelledError:
            raise  # finally settles the claims against the event receipt before propagating
        except Exception as exc:
            logger.error("[bluebubbles] attachment delivery failed: %s", exc, exc_info=True)
        finally:
            accepted = accepted or event._gateway_accepted is True
            for guid in claimed_guids:
                self._settle_attachment_delivery(guid, accepted)
            if claimed_guids and not accepted:
                logger.warning("[bluebubbles] gateway did not accept message with attachment(s) %s; "
                               "released delivery claims", ", ".join(_redact(g) for g in claimed_guids))
        return accepted

    async def _recover_late_attachments(self, failed: List[Dict[str, Any]], *, source,
                                        reply_to_message_id: Optional[str] = None) -> None:
        """Poll in the background for attachments that were not downloadable during the webhook.

        Messages.app often finishes syncing the file only minutes later; within a bounded window
        keep retrying and deliver whatever shows up as a follow-up message so the inbound
        attachment is not silently lost. A recovered file stays pending until its handoff is
        accepted (or the window closes): a gateway-refused delivery is retried on the next round
        instead of being dropped after a single attempt."""
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _LATE_ATTACHMENT_RECOVERY_WINDOW_S
            pending = list(failed)
            round_index = 0  # 0-based: the first background retry uses the first retry delay
            while pending and self.client and loop.time() < deadline:
                delay = (_ATTACHMENT_DOWNLOAD_RETRY_DELAYS[round_index]
                         if round_index < len(_ATTACHMENT_DOWNLOAD_RETRY_DELAYS)
                         else _LATE_ATTACHMENT_RECOVERY_INTERVAL_S)
                round_index += 1
                await asyncio.sleep(delay)
                delivered: List[Any] = []  # (cached_path, mime, att)
                still: List[Dict[str, Any]] = []
                for att in pending:
                    guid = att.get("guid") or ""
                    if not guid or guid in self._late_recovery_done:
                        continue  # nothing to retry / already delivered through another event
                    if guid in self._delivery_inflight:
                        still.append(att)  # another delivery owns it right now; re-check next round
                        continue
                    try:
                        path = await self._fetch_attachment_once(guid, att)
                    except ValueError as exc:
                        # Permanent failure (e.g. non-image payload, oversize media): retrying
                        # cannot succeed, so drop it instead of spending the recovery window.
                        logger.debug("[bluebubbles] late attachment %s permanently failed; giving up: %s",
                                     _redact(guid), exc)
                        continue
                    except Exception as exc:
                        logger.debug("[bluebubbles] late attachment %s still unavailable: %s",
                                     _redact(guid), exc)
                        if self.client:
                            still.append(att)
                        continue
                    if path:
                        delivered.append((path, (att.get("mimeType") or "").lower(), att))
                pending = list(still)
                if delivered:
                    # Keep delivered files pending until the handoff is accepted (or the window
                    # closes): a refused admission, or a claim lost to another delivery, comes
                    # back as a retry entry and is re-attempted on the next round.
                    retry = await self._deliver_late_attachments(delivered, source=source,
                                                                 reply_to_message_id=reply_to_message_id)
                    seen = {att.get("guid") or "" for att in pending}
                    for _, _, att in retry:
                        guid = att.get("guid") or ""
                        if guid not in seen:
                            pending.append(att)
                            seen.add(guid)
        finally:
            for att in failed:
                if guid := (att.get("guid") or ""):
                    self._late_recovery_pending.discard(guid)

    async def _deliver_late_attachments(self, delivered: List[Any], *, source,
                                        reply_to_message_id: Optional[str] = None) -> List[Any]:
        """Deliver a follow-up message carrying attachments recovered after the fact.

        Returns the entries that still need a delivery attempt — their claim was won by another
        delivery (deferred) or the gateway refused the handoff (refused) — so the recovery loop
        can keep them pending and retry within its window. The handoff is awaited instead of
        fired off as a background task: the caller learns the admission outcome, and a refused
        handoff is re-attempted rather than dropping a file that was already downloaded."""
        claimed_guids: List[str] = []
        kept: List[Any] = []
        retry: List[Any] = []
        for item in delivered:
            guid = item[2].get("guid") or ""
            if guid and not self._claim_attachment_delivery(guid):
                logger.debug("[bluebubbles] late attachment %s already delivered or in flight; "
                             "not delivering again", _redact(guid))
                retry.append(item)  # deferred: re-judged on the next recovery round
                continue
            if guid:
                claimed_guids.append(guid)
            kept.append(item)
        if not kept:
            return retry
        media_urls = [path for path, _, _ in kept]
        media_types = [mime for _, mime, _ in kept]
        msg_type = self._message_type_for_attachments([(mime, att) for _, mime, att in kept])
        names = ", ".join(_redact((a.get("transferName") or "").strip() or "attachment") for _, _, a in kept)
        text = (f"[late attachment delivered: BlueBubbles could not download {names or 'an attachment'} when "
                f"the message arrived, but the file has now synced and is attached below.]")
        event = MessageEvent(
            text=text, message_type=msg_type, source=source, raw_message={},
            message_id=None, reply_to_message_id=reply_to_message_id,
            media_urls=media_urls, media_types=media_types)
        accepted = await self._deliver_and_settle(event, claimed_guids)
        if not accepted:
            retry.extend(kept)  # refused: keep pending and try again within the window
        return retry

    def _webhook_token(self, request) -> Optional[str]:
        return (request.query.get("password") or request.query.get("guid") or request.headers.get("x-password")
                or request.headers.get("x-guid") or request.headers.get("x-bluebubbles-guid"))

    def _resolve_chat_and_sender(self, payload: dict[str, Any], record: dict[str, Any]):
        """Returns ``(chat_guid, chat_identifier, sender)`` from the many BlueBubbles payload shapes."""
        chat_guid = self._value(record.get("chatGuid"), payload.get("chatGuid"), record.get("chat_guid"),
                                payload.get("chat_guid"), payload.get("guid"))
        # BlueBubbles v1.9+ payloads omit top-level chatGuid; it's nested under data.chats[0].guid.
        _chats = record.get("chats") or []
        if not chat_guid and _chats and isinstance(_chats[0], dict):
            chat_guid = _chats[0].get("guid") or _chats[0].get("chatGuid")
        chat_identifier = self._value(record.get("chatIdentifier"), record.get("identifier"),
                                      payload.get("chatIdentifier"), payload.get("identifier"))
        handle = record.get("handle")
        sender = (self._value(handle.get("address") if isinstance(handle, dict) else None, record.get("sender"),
                              record.get("from"), record.get("address")) or chat_identifier or chat_guid)
        if not (chat_guid or chat_identifier) and sender:
            chat_identifier = sender
        return chat_guid, chat_identifier, sender

    async def _handle_webhook(self, request):
        from aiohttp import web

        if self._webhook_token(request) != self.password:
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            payload = self._parse_webhook_body(await request.read())
        except Exception as exc:
            logger.error("[bluebubbles] webhook parse error: %s", exc)
            return web.json_response({"error": "invalid payload"}, status=400)
        event_type = self._value(payload.get("type"), payload.get("event")) or ""
        if event_type and event_type not in _MESSAGE_EVENTS:  # ack non-message events silently
            return _ok()
        record = self._extract_payload_record(payload) or {}
        if record.get("isFromMe") or record.get("fromMe") or record.get("is_from_me"):
            return _ok()
        assoc_type = record.get("associatedMessageType")
        if isinstance(assoc_type, int) and assoc_type in _TAPBACK_CODES:  # tapback reactions delivered as messages
            return _ok()
        text = self._value(record.get("text"), record.get("message"), record.get("body")) or ""
        chat_guid, chat_identifier, sender = self._resolve_chat_and_sender(payload, record)
        session_chat_id = chat_guid or chat_identifier
        is_group = bool(record.get("isGroup")) or (";+;" in (chat_guid or ""))
        # Mention gate BEFORE the attachment downloads: an unmentioned group message must not
        # pull every attachment through the REST API only to be dropped.
        if is_group and self.require_mention:
            if not self._message_matches_mention_patterns(text):
                logger.debug("[bluebubbles] ignoring group message (require_mention=true, no mention pattern matched)")
                return _ok()
            text = self._clean_mention_text(text)
        entries, failed_atts = await self._collect_attachments(record)
        # Claims are attempted before the event is built and only the media whose claim this
        # dispatch won is kept (the same "retain only claim winners" rule as
        # _deliver_late_attachments): when two concurrent webhooks carry the same attachment,
        # the dispatch that loses the claim must drop that file instead of delivering it a
        # second time. BlueBubbles also replays the same message through both new-message and
        # updated-message, so the claims are synchronous: a replay arriving while this delivery
        # runs must not download and deliver the same file again. _deliver_and_settle releases
        # the claims unless the gateway accepts the event.
        claimed_guids: List[str] = []
        kept: List[Tuple[str, str, str, Dict[str, Any]]] = []
        for guid, cached, mime, att in entries:
            if guid and not self._claim_attachment_delivery(guid):
                logger.debug("[bluebubbles] attachment %s already delivered or in flight; "
                             "dropping it from this event", _redact(guid))
                continue
            if guid:
                claimed_guids.append(guid)
            kept.append((guid, cached, mime, att))
        media_urls = [cached for _, cached, _, _ in kept]
        media_types = [mime for _, _, mime, _ in kept]
        msg_type = self._message_type_for_attachments([(mime, att) for _, _, mime, att in kept])
        if not text and media_urls:
            text = "(attachment)"
        if failed_atts:  # never drop a message because its attachment was not downloadable yet
            notice = self._attachment_failure_notice(failed_atts)
            text = f"{text}\n{notice}" if text else notice
        if not sender or not (chat_guid or chat_identifier):
            # Malformed payload: nothing will be dispatched, so release whatever was claimed.
            for guid in claimed_guids:
                self._settle_attachment_delivery(guid, False)
            return web.json_response({"error": "missing message fields"}, status=400)
        if not text:
            if entries and not kept:
                # Every attachment was already claimed by a concurrent delivery which carries
                # the message content; this dispatch has nothing left to deliver.
                logger.debug("[bluebubbles] every attachment was claimed by a concurrent "
                             "delivery; acknowledging")
                return _ok()
            return web.json_response({"error": "missing message fields"}, status=400)
        source = self.build_source(chat_id=session_chat_id, chat_name=chat_identifier or sender,
                                   chat_type="group" if is_group else "dm", user_id=sender, user_name=sender,
                                   chat_id_alt=chat_identifier)
        event = MessageEvent(
            text=text, message_type=msg_type, source=source, raw_message=payload,
            message_id=self._value(record.get("guid"), record.get("messageGuid"), record.get("id")),
            reply_to_message_id=self._value(record.get("threadOriginatorGuid"), record.get("associatedMessageGuid")),
            media_urls=media_urls, media_types=media_types)
        task = asyncio.create_task(self._deliver_and_settle(event, claimed_guids))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        # The file often lands on disk minutes later: keep retrying in the background and deliver
        # whatever shows up as a follow-up message.
        recoverable: List[Dict[str, Any]] = []
        for att in failed_atts:
            guid = att.get("guid")
            if guid and guid not in self._late_recovery_pending and not self._attachment_claimed(guid):
                self._late_recovery_pending.add(guid)
                recoverable.append(att)
        if recoverable:
            recovery = asyncio.create_task(self._recover_late_attachments(
                recoverable, source=source, reply_to_message_id=event.message_id))
            self._background_tasks.add(recovery)
            recovery.add_done_callback(self._background_tasks.discard)
        if self.send_read_receipts and session_chat_id:  # fire-and-forget read receipt
            asyncio.create_task(self.mark_read(session_chat_id))
        return _ok()
