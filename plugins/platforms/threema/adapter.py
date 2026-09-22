"""Threema Gateway platform adapter (end-to-end encrypted mode).

Threema is the one platform where the traffic runs the other way: Hermes does not
hold a socket open to Threema, Threema POSTs each inbound message to a callback URL
that must terminate a *publicly trusted* TLS certificate. So the adapter is a small
aiohttp receiver plus a REST sender:

    inbound   Threema -> POST <public_url>/threema/callback -> verify MAC -> NaCl
              open -> MessageEvent
    outbound  encrypt container -> POST /send_e2e

Everything that can be tested without a network — the container, the padding, the
MAC, byte-accurate chunking, the file-message JSON — lives in ``crypto.py`` and
``api.py``; this module is the glue and the aiohttp plumbing.
"""

from __future__ import annotations

import asyncio
import binascii
import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - probed by check_requirements()
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    cache_audio_from_bytes_async,
    cache_document_from_bytes_async,
    cache_image_from_bytes_async,
)
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.threema import crypto
from plugins.platforms.threema.api import (
    HTTPX_AVAILABLE,
    ThreemaAPIError,
    ThreemaClient,
    valid_identity,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = None  # dual-stack bind; pin with extra.host
DEFAULT_PORT = 8647
DEFAULT_PATH = "/threema/callback"
# A callback body is a handful of short fields plus one hex box capped at 7812 bytes
# (15,624 hex chars). 64 KiB leaves room for the nickname and form overhead while
# still rejecting junk before any crypto runs.
MAX_BODY = 65_536
# Threema's text container is capped in BYTES, not characters.
MAX_TEXT_BYTES = 3500
# Threema retries a failed callback 3 times at 5-minute intervals, so a dedup window
# has to outlive the whole retry schedule.
DEDUP_TTL_SECONDS = 30 * 60
MAX_SEEN = 4000
MAX_INBOUND_BLOB_BYTES = 25 * 1024 * 1024
_STATUS_TEXT = {401: "invalid mac", 500: "retry"}

_ENV_SEED_KEYS = (
    ("THREEMA_GATEWAY_ID", "gateway_id"),
    ("THREEMA_API_SECRET", "api_secret"),
    ("THREEMA_PRIVATE_KEY", "private_key"),
    ("THREEMA_PRIVATE_KEY_PATH", "private_key_path"),
    ("THREEMA_PUBLIC_URL", "public_url"),
    ("THREEMA_CALLBACK_PATH", "callback_path"),
    ("THREEMA_HOME_CHANNEL", "home_channel"),
)


def check_requirements() -> bool:
    """PASSIVE probe (registry ``check_fn``) — must never install anything."""
    return AIOHTTP_AVAILABLE and HTTPX_AVAILABLE and crypto.NACL_AVAILABLE


def _import_nacl() -> dict:
    from nacl.exceptions import CryptoError as _CryptoError
    from nacl.public import Box as _Box, PrivateKey as _PrivateKey, PublicKey as _PublicKey
    from nacl.secret import SecretBox as _SecretBox
    from nacl.utils import random as _random
    return {"Box": _Box, "PrivateKey": _PrivateKey, "PublicKey": _PublicKey,
            "SecretBox": _SecretBox, "CryptoError": _CryptoError,
            "nacl_random": _random, "NACL_AVAILABLE": True}


def _import_aiohttp() -> dict:
    from aiohttp import web as _web
    return {"web": _web, "AIOHTTP_AVAILABLE": True}


def ensure_requirements() -> bool:
    """ACTIVE installer (``ensure_deps_fn``): PyNaCl, whose wheels carry libsodium.

    Every module-level availability flag the install affects is rebound here. A
    module whose import failed at load keeps its False flag forever otherwise, so
    ``check_fn`` would still refuse the platform right after a successful install.
    """
    if check_requirements():
        return True
    try:
        from tools.lazy_deps import ensure_and_bind
    except Exception:  # pragma: no cover - defensive
        return False
    ensure_and_bind("platform.threema", _import_nacl, vars(crypto), prompt=False)
    if not AIOHTTP_AVAILABLE:
        ensure_and_bind("platform.threema", _import_aiohttp, globals(), prompt=False)
    return check_requirements()


def _text_chunks(text: str, limit: int = MAX_TEXT_BYTES) -> List[str]:
    """Split on UTF-8 BYTE length without splitting a code point.

    The cap Threema enforces is bytes: 3500 emoji are ~14,000 bytes and would be
    rejected, while 3500 ASCII characters fit exactly. Prefer a newline, then a
    space, then a hard cut.
    """
    if not text:
        return []
    chunks: List[str] = []
    remaining = text
    while len(remaining.encode("utf-8")) > limit:
        # Trim the byte window back to a whole code point, then to a word boundary
        # when one is near the end.
        piece = remaining.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
        cut = max(piece.rfind("\n"), piece.rfind(" "))
        if cut < len(piece) // 2:  # no sensible boundary — take the whole window
            cut = len(piece)
        cut = max(cut, 1)  # never make a zero-width step
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
        if not remaining:
            return [c for c in chunks if c]
    chunks.append(remaining)
    return [c for c in chunks if c]


def _rendering_type(media_type: str, *, as_media: bool) -> int:
    """Threema file-message ``j``: 0 file, 1 media, 2 sticker."""
    if not as_media:
        return 0
    return 1


def _guess_media_type(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


class ThreemaAdapter(BasePlatformAdapter):
    """End-to-end encrypted Threema Gateway adapter."""

    # Answers /p/<profile>/... on the default listener for a served secondary (shared_ingress).
    serves_profile_prefix: bool = True
    MAX_MESSAGE_LENGTH = MAX_TEXT_BYTES

    def __init__(self, config: PlatformConfig):
        super().__init__(config=config, platform=Platform("threema"))
        extra = config.extra or {}
        self._gateway_id = str(extra.get("gateway_id") or _get_scoped_secret("THREEMA_GATEWAY_ID") or "").strip().upper()
        self._secret = str(extra.get("api_secret") or _get_scoped_secret("THREEMA_API_SECRET") or "").strip()
        self._key_source = str(
            extra.get("private_key_path") or extra.get("private_key")
            or _get_scoped_secret("THREEMA_PRIVATE_KEY_PATH") or _get_scoped_secret("THREEMA_PRIVATE_KEY") or ""
        ).strip()
        raw_host = extra.get("host") or DEFAULT_HOST
        self._host = str(raw_host) if raw_host else None
        self._port = int(extra.get("callback_port") or extra.get("port") or DEFAULT_PORT)
        self._path = str(extra.get("callback_path") or DEFAULT_PATH)
        self._public_url = str(extra.get("public_url") or _get_scoped_secret("THREEMA_PUBLIC_URL") or "").strip()
        # Delivery receipts are billed like any other message, so they are opt-in.
        self._send_receipts = bool(extra.get("send_delivery_receipts", False))
        self._private_key: Optional[bytes] = None
        self._client: Optional[ThreemaClient] = None
        self._app = self._runner = None
        self._queue: "asyncio.Queue[MessageEvent]" = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None
        self._seen: Dict[str, float] = {}
        self._nicknames: Dict[str, str] = {}

    # -- lifecycle ----------------------------------------------------------

    def _missing_config(self) -> List[str]:
        missing = []
        if not self._gateway_id:
            missing.append("THREEMA_GATEWAY_ID")
        elif not valid_identity(self._gateway_id):
            missing.append(f"THREEMA_GATEWAY_ID ({self._gateway_id!r} is not an 8-character Threema ID)")
        if not self._secret:
            missing.append("THREEMA_API_SECRET")
        if not self._key_source:
            missing.append("THREEMA_PRIVATE_KEY_PATH or THREEMA_PRIVATE_KEY")
        return missing

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        del is_reconnect  # kwarg MUST exist (GatewayRunner passes it) even though unused
        missing = self._missing_config()
        if missing:
            logger.warning("[Threema] Not configured: %s", ", ".join(missing))
            return False
        if not check_requirements():
            logger.warning("[Threema] Missing dependencies (aiohttp, httpx, pynacl)")
            return False
        try:
            self._private_key = crypto.load_private_key(self._key_source)
        except crypto.ThreemaCryptoError as exc:
            logger.error("[Threema] %s", exc)
            return False

        self._client = ThreemaClient(self._gateway_id, self._secret)
        try:
            credits = await self._client.credits()
        except ThreemaAPIError as exc:
            logger.error("[Threema] Credential check failed: %s", exc)
            await self._cleanup()
            return False
        if credits is not None and credits <= 0:
            logger.warning("[Threema] Account has no credits left — sends will fail with HTTP 402")

        self._warn_about_callback_url()
        try:
            from gateway.platforms.shared_ingress import bind_listener
            self._app = web.Application(client_max_size=MAX_BODY)
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_post(self._path, self._handle_callback)
            # Plugin-registered routes must be wired before AppRunner.setup() freezes the router.
            self._wire_plugin_handlers(self._app)
            self._runner = await bind_listener(self, self._app, self._host, self._port, self._path)
            self._worker = asyncio.create_task(self._drain_queue())
            self._mark_connected()
        except Exception:
            await self._cleanup()
            logger.exception("[Threema] Failed to start the callback listener")
            return False
        logger.info(
            "[Threema] Connected as %s (%s credits)%s",
            self._gateway_id, "unknown" if credits is None else credits,
            f", listening on {self._host or '*'}:{self._port}{self._path}" if self._runner else "",
        )
        return True

    def _warn_about_callback_url(self) -> None:
        """Threema refuses self-signed certs and plain HTTP; say so before the user waits."""
        if not self._public_url:
            logger.warning(
                "[Threema] No public_url configured. Threema only delivers to an HTTPS URL with a "
                "publicly trusted certificate — set the callback URL in the Gateway panel to "
                "https://<your-host>%s (reverse proxy, Cloudflare Tunnel, or ngrok for dev).", self._path,
            )
        elif not self._public_url.lower().startswith("https://"):
            logger.warning(
                "[Threema] public_url %r is not HTTPS. Threema will refuse to deliver callbacks to it; "
                "inbound messages will never arrive.", self._public_url,
            )

    async def disconnect(self) -> None:
        self._running = False
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        await self._cleanup()
        self._mark_disconnected()
        logger.info("[Threema] Disconnected")

    async def _cleanup(self) -> None:
        if self._runner:
            await self._runner.cleanup()
        self._runner = self._app = None
        if self._client:
            await self._client.aclose()
        self._client = None

    # -- outbound -----------------------------------------------------------

    async def _encrypt_and_send(self, to: str, message_type: int, inner: bytes) -> str:
        key = await self._client.public_key(to)
        nonce, box = crypto.encrypt_container(message_type, inner, self._private_key, key)
        return await self._client.send_e2e(to, nonce, box)

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        del reply_to, metadata  # Threema has no reply anchors in the gateway API
        to = (chat_id or "").strip().upper()
        if not valid_identity(to):
            return SendResult(success=False, error=f"{chat_id!r} is not a valid Threema ID")
        chunks = _text_chunks(content or "")
        if not chunks:
            return SendResult(success=False, error="Nothing to send")
        if len(chunks) > 1:
            logger.info("[Threema] Message split into %d parts (%d credits)", len(chunks), len(chunks))
        ids: List[str] = []
        try:
            for chunk in chunks:
                ids.append(await self._encrypt_and_send(to, crypto.TYPE_TEXT, chunk.encode("utf-8")))
        except (ThreemaAPIError, crypto.ThreemaCryptoError) as exc:
            if ids:  # part of the message did land — report it so the caller does not resend it all
                logger.warning("[Threema] Sent %d of %d parts before failing: %s", len(ids), len(chunks), exc)
            return SendResult(success=False, error=str(exc),
                              retryable=isinstance(exc, ThreemaAPIError) and exc.rate_limited)
        return SendResult(success=True, message_id=ids[-1], continuation_message_ids=tuple(ids[:-1]))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Threema Gateway has no typing indicator — and a fake one would cost a credit."""
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        identity = (chat_id or "").strip().upper()
        return {"name": self._nicknames.get(identity) or identity, "type": "dm", "chat_id": identity}

    async def _send_blob_message(
        self, chat_id: str, data: bytes, *, file_name: str, media_type: str,
        caption: Optional[str] = None, as_media: bool = True,
    ) -> SendResult:
        """Upload an encrypted blob, then send the file container that points at it (2 credits)."""
        to = (chat_id or "").strip().upper()
        if not valid_identity(to):
            return SendResult(success=False, error=f"{chat_id!r} is not a valid Threema ID")
        try:
            key = crypto.random_blob_key()
            blob_id = await self._client.upload_blob(crypto.encrypt_blob(data, key))
            rendering = _rendering_type(media_type, as_media=as_media)
            payload = {
                "j": rendering,
                "i": 1 if rendering else 0,  # deprecated twin of "j"; older clients still read it
                "k": binascii.hexlify(key).decode("ascii"),
                "b": blob_id,
                "m": media_type,
                "n": file_name,
                "s": len(data),
            }
            if caption:
                payload["d"] = caption
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            message_id = await self._encrypt_and_send(to, crypto.TYPE_FILE, body)
        except (ThreemaAPIError, crypto.ThreemaCryptoError, OSError) as exc:
            return SendResult(success=False, error=str(exc))
        return SendResult(success=True, message_id=message_id)

    async def _send_local_file(self, chat_id: str, path: str, caption: Optional[str],
                               *, as_media: bool, file_name: Optional[str] = None) -> SendResult:
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            return SendResult(success=False, error=f"Cannot read {path}: {exc}")
        return await self._send_blob_message(
            chat_id, data, file_name=file_name or os.path.basename(path),
            media_type=_guess_media_type(path), caption=caption, as_media=as_media,
        )

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None,
                              reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                              **kwargs) -> SendResult:
        del reply_to, metadata, kwargs
        return await self._send_local_file(chat_id, image_path, caption, as_media=True)

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """A URL is not a Threema message — fetch it and send the bytes as a media file."""
        if os.path.isfile(image_url):
            return await self.send_image_file(chat_id, image_url, caption)
        try:
            from tools.url_safety import create_ssrf_safe_async_client
            async with create_ssrf_safe_async_client(timeout=30.0) as client:
                response = await client.get(image_url)
                response.raise_for_status()
                data = response.content
            if len(data) > MAX_INBOUND_BLOB_BYTES:
                raise ValueError(f"{len(data)} bytes exceeds the {MAX_INBOUND_BLOB_BYTES}-byte cap")
        except Exception as exc:
            logger.warning("[Threema] Could not fetch %s (%s); sending the link as text", image_url, exc)
            return await self.send(chat_id=chat_id, content=f"{caption}\n{image_url}" if caption else image_url)
        name = os.path.basename(image_url.split("?", 1)[0]) or "image.jpg"
        media_type = response.headers.get("content-type", "").split(";")[0] or _guess_media_type(name)
        return await self._send_blob_message(chat_id, data, file_name=name, media_type=media_type,
                                             caption=caption, as_media=True)

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            file_name: Optional[str] = None, reply_to: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        del reply_to, metadata, kwargs
        return await self._send_local_file(chat_id, file_path, caption, as_media=False, file_name=file_name)

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                         **kwargs) -> SendResult:
        del reply_to, metadata, kwargs
        return await self._send_local_file(chat_id, audio_path, caption, as_media=True)

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                         **kwargs) -> SendResult:
        del reply_to, metadata, kwargs
        return await self._send_local_file(chat_id, video_path, caption, as_media=True)

    # -- inbound ------------------------------------------------------------

    async def _handle_health(self, request) -> Any:
        del request
        return web.json_response({"status": "ok", "platform": "threema", "identity": self._gateway_id})

    async def process_callback(self, form: Dict[str, Any]) -> Tuple[int, Optional[MessageEvent]]:
        """Decide what a callback deserves, with no aiohttp in sight.

        Returns ``(http_status, event)``. The status is the contract with Threema: a
        non-200 buys three more attempts five minutes apart, so only a transient
        failure may answer 500 — an authentic message this key pair can never open
        is acked instead, or the same box arrives four times.
        """
        if not crypto.verify_callback(form, self._secret):
            logger.warning("[Threema] Callback MAC verification failed (from=%s)", form.get("from", "?"))
            return 401, None
        message_id = str(form.get("messageId") or "")
        if message_id and self._is_duplicate(message_id):
            logger.debug("[Threema] Duplicate callback %s ignored", message_id)
            return 200, None
        try:
            return 200, await self._build_event(form)
        except (crypto.ThreemaCryptoError, binascii.Error, ValueError) as exc:
            logger.warning("[Threema] %s (messageId=%s)", exc, message_id)
            return 200, None
        except ThreemaAPIError as exc:
            logger.warning("[Threema] Could not process callback %s: %s", message_id, exc)
            return 500, None  # transient — let Threema retry

    async def _handle_callback(self, request) -> Any:
        """aiohttp shim over :meth:`process_callback`; queues the event and acks."""
        try:
            form = dict(await request.post())
        except Exception:
            logger.warning("[Threema] Malformed callback body")
            return web.Response(status=400, text="bad request")
        status, event = await self.process_callback(form)
        if event is not None:
            await self._queue.put(event)
        return web.Response(status=status, text="" if status == 200 else _STATUS_TEXT.get(status, "error"))

    def _is_duplicate(self, message_id: str) -> bool:
        now = time.time()
        if now - self._seen.get(message_id, float("-inf")) < DEDUP_TTL_SECONDS:
            return True
        self._seen[message_id] = now
        if len(self._seen) > MAX_SEEN:
            cutoff = now - DEDUP_TTL_SECONDS
            self._seen = {k: v for k, v in self._seen.items() if v > cutoff}
        return False

    async def _build_event(self, form: Dict[str, Any]) -> Optional[MessageEvent]:
        sender = str(form.get("from") or "").strip().upper()
        nickname = str(form.get("nickname") or "").strip()
        if nickname:
            self._nicknames[sender] = nickname
        box = binascii.unhexlify(str(form.get("box") or ""))
        nonce = binascii.unhexlify(str(form.get("nonce") or ""))
        sender_key = await self._client.public_key(sender)
        message_type, inner = crypto.decrypt_container(box, nonce, self._private_key, sender_key)
        message_id = str(form.get("messageId") or "")
        source = self.build_source(chat_id=sender, chat_name=nickname or sender, chat_type="dm",
                                   user_id=sender, user_name=nickname or sender, message_id=message_id)

        if message_type == crypto.TYPE_TEXT:
            return MessageEvent(text=inner.decode("utf-8", errors="replace"), message_type=MessageType.TEXT,
                                source=source, message_id=message_id, user_id=sender,
                                user_name=nickname or sender)
        if message_type == crypto.TYPE_LOCATION:
            return MessageEvent(text=self._format_location(inner), message_type=MessageType.LOCATION,
                                source=source, message_id=message_id, user_id=sender,
                                user_name=nickname or sender)
        if message_type == crypto.TYPE_FILE:
            return await self._file_event(inner, source, message_id, sender, nickname)
        if message_type == crypto.TYPE_DELIVERY_RECEIPT:
            # A receipt is not a user turn. Dispatching one would wake the agent for
            # every checkmark the recipient's app sends back.
            logger.debug("[Threema] Delivery receipt 0x%02x from %s", inner[0] if inner else 0, sender)
            return None
        logger.info("[Threema] Ignoring unsupported container type 0x%02x from %s", message_type, sender)
        return None

    @staticmethod
    def _format_location(inner: bytes) -> str:
        """``<lat>,<lon>[,<accuracy>]`` then an optional name and address on later lines."""
        lines = inner.decode("utf-8", errors="replace").split("\n")
        coords = lines[0].split(",")
        latitude, longitude = (coords + ["", ""])[:2]
        label = " / ".join(part.strip() for part in lines[1:] if part.strip())
        text = f"[location] {latitude.strip()}, {longitude.strip()}"
        return f"{text} — {label}" if label else text

    async def _file_event(self, inner: bytes, source: Any, message_id: str, sender: str,
                          nickname: str) -> Optional[MessageEvent]:
        try:
            meta = json.loads(inner.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("[Threema] File message from %s has an unparsable metadata object", sender)
            return None
        blob_id, key_hex = str(meta.get("b") or ""), str(meta.get("k") or "")
        media_type = str(meta.get("m") or "application/octet-stream")
        file_name = str(meta.get("n") or f"threema-{blob_id or message_id}")
        caption = str(meta.get("d") or "")
        size = int(meta.get("s") or 0)
        if not blob_id or not key_hex:
            logger.warning("[Threema] File message from %s has no blob reference", sender)
            return None
        if size and size > MAX_INBOUND_BLOB_BYTES:
            logger.warning("[Threema] Skipping %s-byte inbound blob from %s (cap %s)", size, sender, MAX_INBOUND_BLOB_BYTES)
            return MessageEvent(text=caption or f"[file too large: {file_name} ({size} bytes)]",
                                message_type=MessageType.TEXT, source=source, message_id=message_id,
                                user_id=sender, user_name=nickname or sender)
        encrypted = await self._client.download_blob(blob_id)
        data = crypto.decrypt_blob(encrypted, binascii.unhexlify(key_hex))
        path, kind = await self._cache_attachment(data, file_name, media_type)
        return MessageEvent(
            text=caption, message_type=kind, source=source, message_id=message_id,
            user_id=sender, user_name=nickname or sender,
            media_urls=[path], media_types=[media_type],
        )

    @staticmethod
    async def _cache_attachment(data: bytes, file_name: str, media_type: str) -> Tuple[str, MessageType]:
        suffix = Path(file_name).suffix
        if media_type.startswith("image/"):
            return await cache_image_from_bytes_async(data, suffix or ".jpg"), MessageType.PHOTO
        if media_type.startswith("audio/"):
            return await cache_audio_from_bytes_async(data, suffix or ".ogg"), MessageType.VOICE
        if media_type.startswith("video/"):
            return await cache_document_from_bytes_async(data, file_name), MessageType.VIDEO
        return await cache_document_from_bytes_async(data, file_name), MessageType.DOCUMENT

    async def _drain_queue(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                task = asyncio.create_task(self.handle_message(event))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception:  # pragma: no cover - defensive
                logger.exception("[Threema] Failed to dispatch an inbound message")


# ---------------------------------------------------------------------------
# Registry hooks
# ---------------------------------------------------------------------------


def _credentials(config: Any) -> Tuple[str, str, str]:
    extra = getattr(config, "extra", {}) or {}
    return (
        str(extra.get("gateway_id") or _get_scoped_secret("THREEMA_GATEWAY_ID") or "").strip(),
        str(extra.get("api_secret") or _get_scoped_secret("THREEMA_API_SECRET") or "").strip(),
        str(extra.get("private_key_path") or extra.get("private_key")
            or _get_scoped_secret("THREEMA_PRIVATE_KEY_PATH") or _get_scoped_secret("THREEMA_PRIVATE_KEY") or "").strip(),
    )


def validate_config(config: Any) -> bool:
    return all(_credentials(config))


def is_connected(config: Any) -> bool:
    """Surface in ``hermes status`` before the adapter is ever instantiated."""
    return validate_config(config)


def _env_enablement() -> Optional[Dict[str, Any]]:
    """Seed ``PlatformConfig.extra`` from an env-only setup so status sees it pre-construction."""
    if not (_get_scoped_secret("THREEMA_GATEWAY_ID") and _get_scoped_secret("THREEMA_API_SECRET")):
        return None
    seeded = {key: str(_get_scoped_secret(env)) for env, key in _ENV_SEED_KEYS if _get_scoped_secret(env)}
    port = _get_scoped_secret("THREEMA_CALLBACK_PORT")
    if port:
        try:
            seeded["callback_port"] = int(port)
        except (TypeError, ValueError):
            logger.warning("[Threema] THREEMA_CALLBACK_PORT=%r is not a number; using %d", port, DEFAULT_PORT)
    return seeded


async def _standalone_send(
    pconfig, chat_id: str, message: str, *,
    thread_id: Optional[str] = None, media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process delivery for cron jobs that run without a live gateway.

    Sending does not need the callback listener — only the key pair and the secret —
    so a detached cron job can deliver to Threema in full.
    """
    del thread_id  # Threema has no threads
    gateway_id, secret, key_source = _credentials(pconfig)
    if not (gateway_id and secret and key_source):
        return {"error": "Threema standalone send: THREEMA_GATEWAY_ID / THREEMA_API_SECRET / private key missing"}
    if not valid_identity(chat_id):
        return {"error": f"Threema standalone send: {chat_id!r} is not a valid Threema ID"}
    client = ThreemaClient(gateway_id, secret)
    try:
        private_key = crypto.load_private_key(key_source)
        recipient_key = await client.public_key(chat_id)
        message_id = None
        for chunk in _text_chunks(message or "") or [""]:
            nonce, box = crypto.encrypt_container(crypto.TYPE_TEXT, chunk.encode("utf-8"), private_key, recipient_key)
            message_id = await client.send_e2e(chat_id, nonce, box)
        for path in media_files or []:
            data = Path(path).read_bytes()
            blob_key = crypto.random_blob_key()
            blob_id = await client.upload_blob(crypto.encrypt_blob(data, blob_key))
            payload = {
                "j": 0 if force_document else 1, "i": 0 if force_document else 1,
                "k": binascii.hexlify(blob_key).decode("ascii"), "b": blob_id,
                "m": _guess_media_type(path), "n": os.path.basename(path), "s": len(data),
            }
            nonce, box = crypto.encrypt_container(
                crypto.TYPE_FILE, json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                private_key, recipient_key)
            message_id = await client.send_e2e(chat_id, nonce, box)
        return {"success": True, "message_id": message_id}
    except (ThreemaAPIError, crypto.ThreemaCryptoError, OSError) as exc:
        return {"error": str(exc)}
    finally:
        await client.aclose()


_SETUP_PROMPTS = (  # (env var, prompt, masked)
    ("THREEMA_GATEWAY_ID", "Gateway ID (8 characters, starts with *)", False),
    ("THREEMA_API_SECRET", "API secret", True),
    ("THREEMA_PRIVATE_KEY_PATH", "Path to the private key file (contains private:<hex>)", False),
    ("THREEMA_PUBLIC_URL", "Public HTTPS base URL for callbacks (e.g. https://hermes.example.com)", False),
    ("THREEMA_ALLOWED_USERS", "Allowed Threema IDs (comma-separated; blank=skip)", False),
)


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes setup threema`` (writes ``~/.hermes/.env``)."""
    print("\nThreema Gateway setup\n---------------------\n"
          "Register an END-TO-END Gateway ID at https://gateway.threema.ch (basic mode cannot\n"
          "receive messages). Save the private key the panel shows you — it is shown once.\n")
    try:
        from hermes_cli.config import get_env_value as _get_env, save_env_value as _set_env
    except ImportError:
        print("hermes_cli.config not available; set THREEMA_* vars manually in ~/.hermes/.env")
        return
    for var, prompt, secret in _SETUP_PROMPTS:
        existing = _get_env(var) if callable(_get_env) else None
        suffix = " [keep current]" if existing else ""
        try:
            if secret:
                from hermes_cli.secret_prompt import masked_secret_prompt
                value = masked_secret_prompt(f"{prompt}{suffix}: ")
            else:
                value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            continue
        if value:
            _set_env(var, value)
    print(
        "\nDone. Two manual steps remain in the Threema Gateway panel — they have no API:\n"
        f"  1. Set the callback URL to <your-public-url>{DEFAULT_PATH}\n"
        "  2. The URL must present a publicly trusted TLS certificate (no self-signed).\n"
        "     A reverse proxy with Let's Encrypt, a Cloudflare Tunnel, or ngrok for dev all work.\n"
    )


def register(ctx) -> None:
    ctx.register_platform(
        name="threema", label="Threema", adapter_factory=lambda cfg: ThreemaAdapter(cfg),
        check_fn=check_requirements, ensure_deps_fn=ensure_requirements,
        validate_config=validate_config, is_connected=is_connected,
        required_env=["THREEMA_GATEWAY_ID", "THREEMA_API_SECRET", "THREEMA_PRIVATE_KEY_PATH"],
        install_hint="pip install pynacl (its wheels bundle libsodium)",
        setup_fn=interactive_setup, env_enablement_fn=_env_enablement,
        cron_deliver_env_var="THREEMA_HOME_CHANNEL", standalone_sender_fn=_standalone_send,
        allowed_users_env="THREEMA_ALLOWED_USERS", allow_all_env="THREEMA_ALLOW_ALL_USERS",
        max_message_length=MAX_TEXT_BYTES, emoji="🔒", pii_safe=True, allow_update_command=True,
        platform_hint=(
            "You are chatting via Threema, an end-to-end encrypted messenger. Threema does NOT "
            "render Markdown: asterisks and backticks appear literally, so write plain prose and "
            "use line breaks instead of headings or tables. Each message is capped at 3500 BYTES "
            "of UTF-8 (emoji cost 4 bytes each), and every message you send — including each part "
            "of a long split answer, and one extra for any file upload — costs the user a Threema "
            "credit, so be concise and avoid chatty filler. There are no threads, no reactions, "
            "no typing indicator, and no message editing."
        ),
    )
