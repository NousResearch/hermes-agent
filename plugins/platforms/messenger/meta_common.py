"""Shared Meta Graph API machinery for the Messenger and Instagram plugins.

Facebook Messenger (Page DMs) and Instagram DMs are two surfaces of the
same Meta app: one webhook subscription, one ``X-Hub-Signature-256``
app-secret, one Page access token, and the same Graph API
``/me/messages`` send endpoint.  Meta delivers events for both surfaces
to the SAME callback URL and distinguishes them by the payload's
top-level ``object`` field (``"page"`` vs ``"instagram"``).

Because of that, the two platform plugins (``plugins/platforms/messenger``
and ``plugins/platforms/instagram``) share this module the same way the
two WhatsApp core adapters share ``gateway/platforms/whatsapp_common.py``:

* :class:`MetaBaseAdapter` — everything protocol-level: the aiohttp
  webhook server (ONE listener shared by both adapters when both are
  enabled), the GET verification handshake, HMAC signature validation,
  ``object``-based event routing, echo/receipt filtering, attachment
  download + caching, and Graph API sends.
* Each plugin's ``adapter.py`` subclasses it with surface-specific
  constants (platform name, chunk limit, env prefix) and registers its
  own ``PlatformEntry``.

The module lives inside the ``messenger`` plugin package (Messenger is
the primary surface — an Instagram professional account must be linked
to a Facebook Page to use the Instagram Messaging API at all) and the
``instagram`` plugin imports it via its canonical absolute path, so both
plugin modules observe the same shared-server state.

Environment variables (shared Meta app credentials):

* ``META_PAGE_ACCESS_TOKEN`` — Page access token used for all sends.
* ``META_APP_SECRET``        — app secret; every webhook POST must carry
  a valid ``X-Hub-Signature-256`` or it is rejected with 403.
* ``META_VERIFY_TOKEN``      — webhook verification handshake token.
* ``META_WEBHOOK_HOST`` / ``META_WEBHOOK_PORT`` / ``META_WEBHOOK_PATH``
  — listener binding (defaults ``127.0.0.1:8647`` at ``/meta/webhook``;
  front it with an HTTPS reverse proxy or tunnel, do not strip the path
  prefix).
* ``META_GRAPH_API_BASE``    — Graph API base URL override (default
  ``https://graph.facebook.com/v23.0``); also used to point tests at a
  mock server.

Per-surface enablement and gating are namespaced by each plugin
(``MESSENGER_*`` / ``INSTAGRAM_*``) — see the plugin ``adapter.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GRAPH_API_BASE = "https://graph.facebook.com/v23.0"
DEFAULT_WEBHOOK_HOST = "127.0.0.1"
DEFAULT_WEBHOOK_PORT = 8647
DEFAULT_WEBHOOK_PATH = "/meta/webhook"

# Webhook payloads are small JSON; attachments arrive as CDN URLs.
WEBHOOK_BODY_MAX_BYTES = 2 * 1024 * 1024
# Cap for downloading a single inbound attachment from Meta's CDN.
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024

# Graph API text limits: Messenger 2000 chars, Instagram 1000 chars.
# Chunk below the hard limit to leave headroom for the ``(1/3)`` chunk
# indicators appended by ``truncate_message``.
MESSENGER_MAX_CHARS = 2000
MESSENGER_SAFE_CHARS = 1900
INSTAGRAM_MAX_CHARS = 1000
INSTAGRAM_SAFE_CHARS = 950

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_AUDIO_EXTS = {".mp3", ".mp4", ".m4a", ".ogg", ".wav", ".aac"}


# ---------------------------------------------------------------------------
# Small helpers (module-level so they are unit-testable)
# ---------------------------------------------------------------------------

def truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def redact_meta_id(value: Any) -> str:
    """Partially mask a PSID/IGSID for log output."""
    s = str(value or "")
    if len(s) <= 6:
        return "***"
    return f"{s[:4]}…{s[-2:]}"


def verify_meta_signature(body: bytes, signature_header: str, app_secret: str) -> bool:
    """Validate a webhook POST's ``X-Hub-Signature-256`` header.

    The header value is ``sha256=<hex HMAC-SHA256 of the raw body keyed by
    the app secret>``.  Empty header or empty secret fails closed.
    """
    if not signature_header or not app_secret:
        return False
    expected = "sha256=" + hmac.new(
        app_secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature_header, expected)


def platform_for_webhook_object(obj: str) -> Optional[str]:
    """Map a webhook payload's top-level ``object`` to a platform name.

    ``page`` events are Messenger, ``instagram`` events are Instagram DM.
    Anything else (e.g. ``whatsapp_business_account`` from an over-broad
    app subscription) is not ours — return ``None`` so the caller drops it.
    """
    if obj == "page":
        return "messenger"
    if obj == "instagram":
        return "instagram"
    return None


def classify_attachment(att_type: str, url: str) -> Tuple[str, str]:
    """Return ``(kind, ext)`` for an inbound attachment.

    ``kind`` is one of ``image`` / ``audio`` / ``document`` and decides
    which media cache the payload lands in; ``ext`` is a normalized file
    extension for the cache filename.
    """
    ext = os.path.splitext(url.split("?", 1)[0])[1].lower()
    if att_type == "image":
        return "image", (ext if ext in _IMAGE_EXTS else ".jpg")
    if att_type == "audio":
        return "audio", (ext if ext in _AUDIO_EXTS else ".mp4")
    return "document", (ext or ".bin")


# ---------------------------------------------------------------------------
# Shared webhook server
# ---------------------------------------------------------------------------

class _SharedWebhookServer:
    """One aiohttp listener shared by every connected Meta adapter.

    Both surfaces receive events on the same callback URL, so when both
    plugins are enabled only one TCP listener must exist.  Adapters
    attach on ``connect()`` and detach on ``disconnect()``; the listener
    is started by the first attach and stopped by the last detach.
    """

    def __init__(self) -> None:
        self.adapters: Dict[str, "MetaBaseAdapter"] = {}
        self._runner = None
        self._site = None
        self._lock = asyncio.Lock()
        self._host = ""
        self._port = 0
        self._path = ""

    @property
    def running(self) -> bool:
        return self._runner is not None

    async def attach(self, adapter: "MetaBaseAdapter") -> bool:
        from aiohttp import web

        async with self._lock:
            self.adapters[adapter.PLATFORM_NAME] = adapter
            if self.running:
                if (adapter.webhook_host, adapter.webhook_port) != (self._host, self._port):
                    logger.warning(
                        "Meta: %s configured %s:%s but the shared webhook "
                        "server is already bound to %s:%s — attaching to the "
                        "existing listener (both surfaces share one callback "
                        "URL; configure META_WEBHOOK_HOST/PORT once).",
                        adapter.PLATFORM_NAME,
                        adapter.webhook_host,
                        adapter.webhook_port,
                        self._host,
                        self._port,
                    )
                else:
                    logger.info(
                        "Meta: %s attached to shared webhook server on %s:%s%s",
                        adapter.PLATFORM_NAME, self._host, self._port, self._path,
                    )
                return True

            app = web.Application(client_max_size=WEBHOOK_BODY_MAX_BYTES)
            path = adapter.webhook_path
            app.router.add_get(path, self._handle_verify)
            app.router.add_post(path, self._handle_event)
            app.router.add_get(f"{path}/health", self._handle_health)

            runner = web.AppRunner(app)
            try:
                await runner.setup()
                site = web.TCPSite(runner, adapter.webhook_host, adapter.webhook_port)
                await site.start()
            except OSError as exc:
                self.adapters.pop(adapter.PLATFORM_NAME, None)
                try:
                    await runner.cleanup()
                except Exception:
                    pass
                logger.error(
                    "Meta: could not bind webhook server on %s:%s: %s",
                    adapter.webhook_host, adapter.webhook_port, exc,
                )
                return False

            self._runner, self._site = runner, site
            self._host, self._port, self._path = (
                adapter.webhook_host, adapter.webhook_port, path,
            )
            logger.info(
                "Meta: webhook server listening on %s:%s%s",
                self._host, self._port, self._path,
            )
            return True

    async def detach(self, adapter: "MetaBaseAdapter") -> None:
        async with self._lock:
            self.adapters.pop(adapter.PLATFORM_NAME, None)
            runner = self._runner
            if self.adapters or runner is None:
                return
            try:
                await runner.cleanup()
            except Exception as exc:
                logger.debug("Meta: webhook server cleanup failed: %s", exc)
            self._runner = None
            self._site = None
            logger.info("Meta: webhook server stopped")

    def _any_adapter(self) -> Optional["MetaBaseAdapter"]:
        for adapter in self.adapters.values():
            return adapter
        return None

    # -- HTTP handlers ------------------------------------------------------

    async def _handle_health(self, request) -> Any:
        from aiohttp import web
        return web.json_response(
            {"status": "ok", "platforms": sorted(self.adapters.keys())}
        )

    async def _handle_verify(self, request) -> Any:
        """GET — Meta webhook verification handshake."""
        from aiohttp import web

        adapter = self._any_adapter()
        verify_token = adapter.verify_token if adapter else ""
        mode = request.query.get("hub.mode", "")
        token = request.query.get("hub.verify_token", "")
        challenge = request.query.get("hub.challenge", "")
        if (
            mode == "subscribe"
            and verify_token
            and hmac.compare_digest(token, verify_token)
        ):
            logger.info("Meta: webhook verification handshake OK")
            return web.Response(text=challenge, content_type="text/plain")
        logger.warning("Meta: webhook verification failed (mode=%r)", mode)
        return web.Response(status=403, text="verification failed")

    async def _handle_event(self, request) -> Any:
        """POST — inbound webhook events for both surfaces."""
        from aiohttp import web

        try:
            body = await request.read()
        except Exception:
            return web.Response(status=400, text="bad request")
        if len(body) > WEBHOOK_BODY_MAX_BYTES:
            return web.Response(status=413, text="payload too large")

        adapter = self._any_adapter()
        app_secret = adapter.app_secret if adapter else ""
        signature = request.headers.get("X-Hub-Signature-256", "")
        if not verify_meta_signature(body, signature, app_secret):
            logger.warning("Meta: invalid X-Hub-Signature-256 — rejecting event")
            return web.Response(status=403, text="invalid signature")

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("Meta: non-JSON webhook body ignored")
            return web.Response(status=200, text="EVENT_RECEIVED")

        target_name = platform_for_webhook_object(payload.get("object", ""))
        target = self.adapters.get(target_name) if target_name else None
        if target is None:
            logger.warning(
                "Meta: %r event received but no adapter for it is connected — dropped",
                payload.get("object", ""),
            )
            return web.Response(status=200, text="EVENT_RECEIVED")

        for entry in payload.get("entry", []) or []:
            for messaging in entry.get("messaging", []) or []:
                try:
                    await target._process_messaging(messaging)
                except Exception:
                    logger.exception(
                        "Meta: failed to process %s event", target.PLATFORM_NAME
                    )

        # Always ACK fast — Meta retries and eventually disables webhooks
        # that respond slowly or with errors.
        return web.Response(status=200, text="EVENT_RECEIVED")


# Module-level singleton: both plugin adapters import THIS module (the
# instagram plugin uses the canonical ``plugins.platforms.messenger.
# meta_common`` path), so they share one server instance.
_shared_server = _SharedWebhookServer()


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class MetaBaseAdapter(BasePlatformAdapter):
    """Common behavior for the Messenger and Instagram DM adapters.

    Subclasses set the class constants below and inherit the entire
    webhook + Graph API pipeline.
    """

    # -- subclass contract ---------------------------------------------------
    PLATFORM_NAME: str = ""       # "messenger" | "instagram"
    ENV_PREFIX: str = ""          # "MESSENGER" | "INSTAGRAM"
    CHAT_LABEL: str = ""          # e.g. "Messenger DM"
    MAX_MESSAGE_LENGTH: int = MESSENGER_MAX_CHARS
    SAFE_CHUNK_CHARS: int = MESSENGER_SAFE_CHARS

    def __init__(self, config, **kwargs):
        super().__init__(config=config, platform=Platform(self.PLATFORM_NAME))
        extra = getattr(config, "extra", {}) or {}

        # Shared Meta app credentials (env wins, config extra as fallback).
        self.page_access_token: str = (
            os.getenv("META_PAGE_ACCESS_TOKEN")
            or extra.get("page_access_token", "")
        )
        self.app_secret: str = (
            os.getenv("META_APP_SECRET") or extra.get("app_secret", "")
        )
        self.verify_token: str = (
            os.getenv("META_VERIFY_TOKEN") or extra.get("verify_token", "")
        )

        # Webhook listener binding (shared by both surfaces).
        self.webhook_host: str = (
            os.getenv("META_WEBHOOK_HOST") or extra.get("host", DEFAULT_WEBHOOK_HOST)
        )
        try:
            self.webhook_port: int = int(
                os.getenv("META_WEBHOOK_PORT")
                or extra.get("port", DEFAULT_WEBHOOK_PORT)
            )
        except (TypeError, ValueError):
            self.webhook_port = DEFAULT_WEBHOOK_PORT
        self.webhook_path: str = (
            os.getenv("META_WEBHOOK_PATH")
            or extra.get("webhook_path", DEFAULT_WEBHOOK_PATH)
        )

        self.graph_api_base: str = (
            os.getenv("META_GRAPH_API_BASE")
            or extra.get("graph_api_base", DEFAULT_GRAPH_API_BASE)
        ).rstrip("/")

        # One-shot warning latch for missing send credentials.
        self._warned_no_token = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        missing = [
            name
            for name, value in (
                ("META_PAGE_ACCESS_TOKEN", self.page_access_token),
                ("META_APP_SECRET", self.app_secret),
                ("META_VERIFY_TOKEN", self.verify_token),
            )
            if not value
        ]
        if missing:
            self._set_fatal_error(
                "config_missing",
                f"{' and '.join(missing)} must be set for the "
                f"{self.PLATFORM_NAME} adapter",
                retryable=False,
            )
            return False

        try:
            import aiohttp  # noqa: F401
        except ImportError:
            self._set_fatal_error(
                "missing_dep",
                "aiohttp is required for the Meta adapters — install with "
                "`pip install aiohttp`",
                retryable=False,
            )
            return False

        if not await _shared_server.attach(self):
            self._set_fatal_error(
                "bind_failed",
                f"Could not bind Meta webhook server on "
                f"{self.webhook_host}:{self.webhook_port}",
                retryable=True,
            )
            return False

        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        await _shared_server.detach(self)
        logger.info("Meta: %s disconnected", self.PLATFORM_NAME)

    # ------------------------------------------------------------------
    # Inbound
    # ------------------------------------------------------------------

    @staticmethod
    def _skip_reason(messaging: Dict[str, Any]) -> Optional[str]:
        """Classify a webhook ``messaging`` item we must NOT feed the agent."""
        if "delivery" in messaging or "read" in messaging:
            return "receipt"
        message = messaging.get("message")
        if not isinstance(message, dict):
            return "non-message"
        if message.get("is_echo"):
            return "echo"
        if not str((messaging.get("sender") or {}).get("id", "")):
            return "no-sender"
        return None

    async def _process_messaging(self, messaging: Dict[str, Any]) -> None:
        """Turn one ``entry[].messaging[]`` item into a MessageEvent."""
        skip = self._skip_reason(messaging)
        if skip is not None:
            logger.debug(
                "Meta(%s): skipping %s event", self.PLATFORM_NAME, skip
            )
            return

        sender_id = str((messaging.get("sender") or {}).get("id", ""))
        recipient_id = str((messaging.get("recipient") or {}).get("id", ""))
        message = messaging["message"]

        text = message.get("text") or ""
        media_urls: List[str] = []
        media_types: List[str] = []
        message_type = MessageType.TEXT

        for att in message.get("attachments", []) or []:
            att_type = att.get("type", "")
            url = (att.get("payload") or {}).get("url", "")
            if not url:
                continue
            data = await self._download_attachment(url)
            if data is None:
                continue
            kind, ext = classify_attachment(att_type, url)
            if kind == "image":
                path = cache_image_from_bytes(data, ext=ext)
                mtype = f"image/{ext.lstrip('.')}"
                fallback = MessageType.PHOTO
            elif kind == "audio":
                path = cache_audio_from_bytes(data, ext=ext)
                mtype = f"audio/{ext.lstrip('.')}"
                fallback = MessageType.VOICE
            else:
                filename = (
                    os.path.basename(url.split("?", 1)[0]) or f"attachment{ext}"
                )
                path = cache_document_from_bytes(data, filename)
                mtype = att_type or "file"
                fallback = (
                    MessageType.VIDEO if att_type == "video"
                    else MessageType.DOCUMENT
                )
            if path:
                media_urls.append(path)
                media_types.append(mtype)
                if message_type == MessageType.TEXT:
                    message_type = fallback

        if not text and not media_urls:
            logger.debug(
                "Meta(%s): empty message from %s ignored "
                "(unsupported attachment?)",
                self.PLATFORM_NAME, redact_meta_id(sender_id),
            )
            return

        source = self.build_source(
            chat_id=sender_id,
            chat_name=self.CHAT_LABEL,
            chat_type="dm",
            user_id=sender_id,
            # The receiving Page / IG professional-account id — kept so
            # multi-page routing has an anchor if it's ever needed.
            chat_id_alt=recipient_id or None,
        )
        event = MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            raw_message=messaging,
            message_id=message.get("mid"),
            media_urls=media_urls,
            media_types=media_types,
        )
        logger.info(
            "Meta(%s): message from %s text_len=%d media=%d",
            self.PLATFORM_NAME,
            redact_meta_id(sender_id),
            len(text),
            len(media_urls),
        )
        # Dispatch in the background so the webhook 200s immediately —
        # Meta disables webhooks that respond slowly.
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _download_attachment(self, url: str) -> Optional[bytes]:
        """Download an inbound attachment from Meta's CDN (None on failure)."""
        import aiohttp

        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "Meta(%s): attachment download HTTP %d",
                            self.PLATFORM_NAME, resp.status,
                        )
                        return None
                    if (resp.content_length or 0) > MAX_ATTACHMENT_BYTES:
                        logger.warning(
                            "Meta(%s): attachment exceeds %d bytes, skipped",
                            self.PLATFORM_NAME, MAX_ATTACHMENT_BYTES,
                        )
                        return None
                    data = await resp.content.read(MAX_ATTACHMENT_BYTES + 1)
                    if len(data) > MAX_ATTACHMENT_BYTES:
                        logger.warning(
                            "Meta(%s): attachment exceeds %d bytes, skipped",
                            self.PLATFORM_NAME, MAX_ATTACHMENT_BYTES,
                        )
                        return None
                    return data
        except Exception as exc:
            logger.warning(
                "Meta(%s): attachment download failed: %s",
                self.PLATFORM_NAME, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Outbound (Graph API /me/messages, shared by both surfaces)
    # ------------------------------------------------------------------

    async def _graph_post(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """POST to the Graph API. Returns ``(ok, response_json)``.

        The access token travels as a query parameter, so the URL is
        never logged — only the endpoint name and the error body.
        """
        import aiohttp

        if not self.page_access_token:
            if not self._warned_no_token:
                logger.error(
                    "Meta(%s): META_PAGE_ACCESS_TOKEN not configured — "
                    "cannot send", self.PLATFORM_NAME,
                )
                self._warned_no_token = True
            return False, {
                "error": {"message": "META_PAGE_ACCESS_TOKEN not configured"}
            }

        url = f"{self.graph_api_base}/{endpoint}"
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url,
                    params={"access_token": self.page_access_token},
                    json=payload,
                ) as resp:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {"error": {"message": await resp.text()}}
                    if resp.status == 200:
                        return True, data
                    logger.error(
                        "Meta(%s): Graph API %s HTTP %d: %s",
                        self.PLATFORM_NAME,
                        endpoint,
                        resp.status,
                        json.dumps(data.get("error", data))[:500],
                    )
                    return False, data
        except Exception as exc:
            logger.error(
                "Meta(%s): Graph API request failed: %s",
                self.PLATFORM_NAME, exc,
            )
            return False, {"error": {"message": str(exc), "transient": True}}

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send text via ``/me/messages`` (chunked to the surface limit)."""
        if not content or not content.strip():
            return SendResult(success=False, error="Empty message")

        chunks = self.truncate_message(content, self.SAFE_CHUNK_CHARS)
        last_mid: Optional[str] = None
        for chunk in chunks:
            ok, data = await self._graph_post(
                "me/messages",
                {
                    "recipient": {"id": str(chat_id)},
                    "messaging_type": "RESPONSE",
                    "message": {"text": chunk},
                },
            )
            if not ok:
                err = (data.get("error") or {}).get("message", "unknown error")
                return SendResult(
                    success=False,
                    error=err,
                    raw_response=data,
                    retryable=bool((data.get("error") or {}).get("transient")),
                )
            last_mid = data.get("message_id")
        return SendResult(success=True, message_id=last_mid)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Best-effort ``typing_on`` sender action."""
        try:
            await self._graph_post(
                "me/messages",
                {
                    "recipient": {"id": str(chat_id)},
                    "sender_action": "typing_on",
                },
            )
        except Exception as exc:
            logger.debug(
                "Meta(%s): send_typing failed: %s", self.PLATFORM_NAME, exc
            )

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image by public URL as a native attachment."""
        ok, data = await self._graph_post(
            "me/messages",
            {
                "recipient": {"id": str(chat_id)},
                "messaging_type": "RESPONSE",
                "message": {
                    "attachment": {
                        "type": "image",
                        "payload": {"url": image_url, "is_reusable": False},
                    }
                },
            },
        )
        if not ok:
            err = (data.get("error") or {}).get("message", "unknown error")
            return SendResult(success=False, error=err, raw_response=data)
        if caption and caption.strip():
            return await self.send(chat_id, caption)
        return SendResult(success=True, message_id=data.get("message_id"))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": self.CHAT_LABEL, "type": "dm", "chat_id": str(chat_id)}


# ---------------------------------------------------------------------------
# Shared PlatformEntry helper functions
# ---------------------------------------------------------------------------

def _have_meta_credentials(extra: Optional[Dict[str, Any]] = None) -> bool:
    extra = extra or {}
    return bool(
        (os.getenv("META_PAGE_ACCESS_TOKEN") or extra.get("page_access_token"))
        and (os.getenv("META_APP_SECRET") or extra.get("app_secret"))
        and (os.getenv("META_VERIFY_TOKEN") or extra.get("verify_token"))
    )


def make_check_requirements(env_prefix: str):
    """Build a ``check_fn`` for one surface.

    Requires the surface's explicit ``<PREFIX>_ENABLED`` opt-in (the Meta
    credentials are shared, so their presence alone cannot tell WHICH
    surface the user wants), the shared credentials, and aiohttp.
    """

    def check_requirements() -> bool:
        if not truthy_env(f"{env_prefix}_ENABLED"):
            return False
        if not _have_meta_credentials():
            return False
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            return False
        return True

    return check_requirements


def make_is_connected(env_prefix: str):
    """Build ``is_connected``/``validate_config`` for one surface."""

    def is_connected(config) -> bool:
        extra = getattr(config, "extra", {}) or {}
        return truthy_env(f"{env_prefix}_ENABLED") and _have_meta_credentials(extra)

    return is_connected


def make_env_enablement(env_prefix: str):
    """Build an ``env_enablement_fn`` seeding PlatformConfig.extra from env.

    Lets ``hermes status`` / ``get_connected_platforms()`` reflect an
    env-only setup before the adapter is instantiated.
    """

    def _env_enablement() -> Optional[Dict[str, Any]]:
        if not truthy_env(f"{env_prefix}_ENABLED"):
            return None
        if not _have_meta_credentials():
            return None
        seeded: Dict[str, Any] = {}
        if os.getenv("META_WEBHOOK_PORT"):
            try:
                seeded["port"] = int(os.environ["META_WEBHOOK_PORT"])
            except ValueError:
                pass
        if os.getenv("META_WEBHOOK_HOST"):
            seeded["host"] = os.environ["META_WEBHOOK_HOST"]
        if os.getenv("META_WEBHOOK_PATH"):
            seeded["webhook_path"] = os.environ["META_WEBHOOK_PATH"]
        home = os.getenv(f"{env_prefix}_HOME_CHANNEL", "").strip()
        if home:
            seeded["home_channel"] = {
                "chat_id": home,
                "name": os.getenv(f"{env_prefix}_HOME_CHANNEL_NAME", "Home"),
            }
        return seeded

    return _env_enablement


def make_standalone_send(adapter_cls):
    """Build a ``standalone_sender_fn`` for one surface.

    Sends via the Graph API without a live gateway adapter (the send
    pipeline is stateless HTTP), so ``deliver=messenger:...`` /
    ``deliver=instagram:...`` cron jobs work when cron runs in a separate
    process from the gateway.
    """

    async def _standalone_send(
        pconfig,
        chat_id: str,
        message: str,
        *,
        thread_id: Optional[str] = None,
        media_files: Optional[List[str]] = None,
        force_document: bool = False,
    ) -> Dict[str, Any]:
        # thread_id accepted for signature parity; Meta DMs have no threads.
        try:
            adapter = adapter_cls(pconfig)
        except Exception as exc:
            return {"error": f"{adapter_cls.PLATFORM_NAME} init failed: {exc}"}
        if not adapter.page_access_token:
            return {
                "error": (
                    f"{adapter_cls.PLATFORM_NAME} standalone send: "
                    "META_PAGE_ACCESS_TOKEN not configured"
                )
            }
        text = message or ""
        if media_files:
            # Graph API attachments need a public URL; local cron artifacts
            # aren't reachable, so surface them as a note instead of
            # silently dropping.
            text = (
                f"{text}\n[{len(media_files)} attachment(s) generated; "
                "not deliverable from cron]"
            ).strip()
        result = await adapter.send(chat_id, text)
        if not result.success:
            return {"error": result.error or "send failed"}
        return {"success": True, "message_id": result.message_id}

    return _standalone_send
