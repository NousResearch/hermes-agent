"""ntfy platform adapter: HTTP-streaming subscription (``/json``, ``poll=false``) in, POST out.

config.yaml ``platforms.ntfy.extra``: ``server`` (default https://ntfy.sh), ``topic`` (required),
``publish_topic`` (defaults to topic), ``token`` (Bearer or ``user:pass`` Basic), ``markdown``
(default false). Env (read at construct time; ``extra`` wins over env): NTFY_TOPIC, NTFY_SERVER_URL,
NTFY_TOKEN, NTFY_PUBLISH_TOPIC, NTFY_MARKDOWN ("true"/"1"/"yes"), NTFY_ALLOWED_USERS (topic names),
NTFY_ALLOW_ALL_USERS (dev only), NTFY_HOME_CHANNEL, NTFY_HOME_CHANNEL_NAME.
Outgoing attachments (issue #46447): a local file is published as the POST body with a
``filename`` query param (ntfy's ``X-Filename``; attachment fields ride as query params because
httpx rejects non-ASCII header values), message text as ``message=`` (``X-Message``); a URL
attachment publishes with ``attach=`` (``X-Attach``) and an empty body. Attachment size: the
public ntfy.sh server allows 2 MB per attachment, a self-hosted server's shipped default is
15 MB — the client cap follows the configured server unless ``extra.attachment_max_mb``
overrides it. ntfy.sh expires attachments after 3 h.
Identity: ntfy has no authenticated user; ``title`` is publisher-controlled and NOT used for
authorization. Each topic is one trusted channel (``user_id`` == topic). Protect it with a read token.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret

logger = logging.getLogger(__name__)


class _FatalStreamError(Exception):
    """Unrecoverable stream error (401, 404)."""


DEFAULT_SERVER = "https://ntfy.sh"
MAX_MESSAGE_LENGTH = 4096  # ntfy message body limit
# Attachment caps. The ntfy server default is 15 MB, but the public ntfy.sh allows 2 MB per
# attachment (20 MB per visitor) — probe: 2 MB accepted, 2 MB + 1 byte rejected with HTTP 413.
# The server re-checks either way, so the client cap only decides which files fail fast: it must
# never be more permissive than the server the adapter publishes to.
DEFAULT_ATTACHMENT_MAX_BYTES = 15 * 1024 * 1024
PUBLIC_SERVER_ATTACHMENT_MAX_BYTES = 2 * 1024 * 1024
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 1000
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]
STREAM_TIMEOUT_SECONDS = 90  # ntfy keepalive default is 55s; give margin
_ECHO_TAG = "hermes-agent"  # tag added to outgoing messages for echo-loop prevention
_MARKDOWN_TRUTHY = ("1", "true", "yes")


def _build_auth_header(token: str) -> Dict[str, str]:
    """``Authorization`` header from an ntfy token; ``{}`` when unset.

    Tokens are whitespace-stripped (pasted tokens often carry newlines that
    would malform the header). ``user:pass`` → Basic, anything else → Bearer.
    """
    token = (token or "").strip()
    if not token:
        return {}
    if ":" in token:
        import base64
        return {"Authorization": f"Basic {base64.b64encode(token.encode()).decode()}"}
    return {"Authorization": f"Bearer {token}"}


def _publish_headers(token: str, markdown: bool, *, auth_first: bool = True) -> Dict[str, str]:
    """Headers for a publish POST: auth (if any), plain-text body, echo tag, optional X-Markdown.

    ``auth_first`` pins the header order each call site has always sent on the wire.
    """
    auth = _build_auth_header(token)
    base = {"Content-Type": "text/plain; charset=utf-8", "X-Tags": _ECHO_TAG}
    headers = {**auth, **base} if auth_first else {**base, **auth}
    if markdown:
        headers["X-Markdown"] = "true"
    return headers


def _attachment_fields(
    *, message: Optional[str] = None, file_name: Optional[str] = None, attach_url: Optional[str] = None,
) -> Dict[str, str]:
    """Attachment publish fields as query params (ntfy accepts the X-Filename/X-Message/X-Attach
    headers or these ``filename``/``message``/``attach`` aliases). Query params, not headers:
    httpx rejects non-ASCII header values (``UnicodeEncodeError``) and captions/filenames/URLs
    can carry emoji or CJK; httpx URL-encodes params as UTF-8."""
    fields: Dict[str, str] = {}
    if message is not None:
        fields["message"] = message
    if file_name is not None:
        fields["filename"] = file_name
    if attach_url is not None:
        fields["attach"] = attach_url
    return fields


def _attachment_headers(token: str, markdown: bool) -> Dict[str, str]:
    """Headers for an attachment publish POST: auth (if any), echo tag, optional X-Markdown.

    No ``Content-Type``: the server sniffs the attachment bytes.
    """
    auth = _build_auth_header(token)
    headers = {"X-Tags": _ECHO_TAG}
    if markdown:
        headers["X-Markdown"] = "true"
    return {**auth, **headers}


async def _publish_attachment(
    client, server: str, publish_topic: str, *, headers: Dict[str, str],
    params: Dict[str, str], body: Optional[bytes],
) -> SendResult:
    """One attachment publish POST; never raises for expected failures. The 120 s timeout
    accommodates multi-MB uploads (text sends use 15 s)."""
    url = f"{server}/{publish_topic}"
    try:
        resp = await client.post(url, content=body, headers=headers, params=params, timeout=120.0)
        if resp.status_code < 300:
            return SendResult(success=True, message_id=_response_message_id(resp))
        logger.warning(
            "Attachment publish failed HTTP %d: %s", resp.status_code, resp.text[:200])
        return SendResult(success=False, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
    except httpx.TimeoutException:
        return SendResult(success=False, error="Timeout publishing attachment to ntfy")
    except Exception as e:
        logger.error("Attachment publish error: %s", e)
        return SendResult(success=False, error=str(e))


def _attachment_max_bytes(extra: Dict[str, Any], server: str) -> int:
    """Client-side attachment cap for ``server``: ``extra.attachment_max_mb`` when set (integer
    MB), else the server's known limit — 2 MB for the public ntfy.sh, 15 MB for a self-hosted
    server (ntfy's shipped default). A non-numeric override is ignored with a warning rather
    than failing mid-send; an unusable value must not take the whole send path down.
    """
    public = (urlsplit(server).hostname or "").lower() == "ntfy.sh"
    default = PUBLIC_SERVER_ATTACHMENT_MAX_BYTES if public else DEFAULT_ATTACHMENT_MAX_BYTES
    configured = extra.get("attachment_max_mb")
    if configured in (None, ""):
        return default
    try:
        megabytes = int(configured)
    except (TypeError, ValueError):
        logger.warning("Ignoring non-numeric attachment_max_mb=%r", configured)
        return default
    return megabytes * 1024 * 1024 if megabytes > 0 else default


def _read_attachment(path_value: Any, *, max_bytes: int) -> "Tuple[Path, None] | Tuple[None, str]":
    """``(path, None)`` or ``(None, error)`` for a local attachment input (str or Path
    accepted; the base-class contract is str). Error names the concrete problem:
    wrong type vs missing/not-a-file vs unreadable vs too large (over ``max_bytes``)."""
    if not isinstance(path_value, (str, Path)):
        return None, f"invalid attachment path type: {type(path_value).__name__}"
    path = Path(path_value)
    try:
        if not path.is_file():
            return None, "attachment file not found or is not a regular file"
        size = path.stat().st_size
    except FileNotFoundError:
        return None, "attachment file not found"
    except OSError as e:
        return None, f"attachment file unreadable: {e}"
    if size > max_bytes:
        return None, (f"attachment exceeds the {max_bytes // (1024 * 1024)} MB "
                      f"attachment limit ({size} bytes)")
    return path, None


def _cap_to_message_limit(text: Optional[str], *, context: str) -> Optional[str]:
    """Byte-aware cap to ntfy's message limit: it counts BYTES (a 4096-char CJK/emoji
    caption is up to 4x that), so truncate the UTF-8 encoding at a character boundary."""
    if text is None:
        return None
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_MESSAGE_LENGTH:
        return text
    logger.warning(
        "%s: truncating message from %d to %d bytes (ntfy limit)",
        context, len(encoded), MAX_MESSAGE_LENGTH)
    return encoded[:MAX_MESSAGE_LENGTH].decode("utf-8", errors="ignore")


def _truncate_body(message: str, *, context: str) -> bytes:
    """Encode ``message`` for a publish body, capped at ntfy's 4096-BYTE limit.

    The limit counts bytes, not characters, and an over-limit body is not rejected: the
    server silently converts it into a file attachment, so a long CJK/emoji message would
    arrive as a file instead of a notification. Truncation is logged (tagged ``context``).
    """
    capped = _cap_to_message_limit(message, context=context)
    return (capped or "").encode("utf-8")


def _response_message_id(resp) -> str:
    """ntfy's returned message id, or a random 12-hex fallback."""
    try:
        return resp.json().get("id") or uuid.uuid4().hex[:12]
    except Exception:
        return uuid.uuid4().hex[:12]


def _setting(extra: Dict[str, Any], key: str, env: str, default: str = "") -> str:
    """config.yaml ``extra[key]`` wins over the env var ``env``."""
    return extra.get(key) or _get_scoped_secret(env, default)


def _server_url(extra: Dict[str, Any]) -> str:
    return _setting(extra, "server", "NTFY_SERVER_URL", DEFAULT_SERVER).rstrip("/")


def check_requirements() -> bool:
    """Installable and minimally configured (reads NTFY_TOPIC directly — no full config load)."""
    return HTTPX_AVAILABLE and bool(_get_scoped_secret("NTFY_TOPIC", "").strip())


def validate_config(config) -> bool:
    """True when a topic is configured (config.yaml ``extra`` or env)."""
    return bool(_setting(getattr(config, "extra", {}) or {}, "topic", "NTFY_TOPIC"))


def is_connected(config) -> bool:
    """Check whether ntfy is configured (env or config.yaml)."""
    return bool(_get_scoped_secret("NTFY_TOPIC") or (getattr(config, "extra", {}) or {}).get("topic", ""))


class NtfyAdapter(BasePlatformAdapter):
    """ntfy adapter: HTTP-streaming subscription in, HTTP POST publish out."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config=config, platform=Platform("ntfy"))
        extra = config.extra or {}
        self._server: str = _server_url(extra)
        self._topic: str = _setting(extra, "topic", "NTFY_TOPIC")
        self._publish_topic: str = _setting(extra, "publish_topic", "NTFY_PUBLISH_TOPIC") or self._topic
        self._token: str = _setting(extra, "token", "NTFY_TOKEN")
        self._stream_task: Optional[asyncio.Task] = None
        self._http_client: Optional["httpx.AsyncClient"] = None
        self._seen_messages: Dict[str, float] = {}  # msg_id -> timestamp (dedup)

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to ntfy by starting the streaming subscription task."""
        if not HTTPX_AVAILABLE:
            logger.warning("[%s] httpx not installed. Run: pip install httpx", self.name)
            return False
        if not self._topic:
            logger.warning("[%s] NTFY_TOPIC not configured", self.name)
            return False
        try:
            self._http_client = httpx.AsyncClient(timeout=None)
            self._stream_task = asyncio.create_task(self._run_stream())
            self._mark_connected()
            logger.info("[%s] Connected — subscribing to %s/%s", self.name, self._server, self._topic)
            self._wire_plugin_handlers(None)
            return True
        except Exception as e:
            logger.error("[%s] Failed to connect: %s", self.name, e)
            return False

    async def _run_stream(self) -> None:
        """Subscribe to the ntfy topic with automatic reconnection."""
        backoff_idx = 0
        stream_start: float = 0.0
        url = f"{self._server}/{self._topic}/json"
        headers = self._auth_headers()
        while self._running:
            try:
                logger.debug("[%s] Opening stream to %s", self.name, url)
                stream_start = time.monotonic()
                await self._consume_stream(url, headers)
            except asyncio.CancelledError:
                return
            except _FatalStreamError:
                self._running = False
                return
            except Exception as e:
                if not self._running:
                    return
                logger.warning("[%s] Stream error: %s", self.name, e)
            if not self._running:
                return
            # Reset backoff if stream stayed alive for at least 60s
            if time.monotonic() - stream_start >= 60.0:
                backoff_idx = 0
            delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
            logger.info("[%s] Reconnecting in %ds...", self.name, delay)
            await asyncio.sleep(delay)
            backoff_idx += 1

    def _fatal_status(self, status_code: int) -> None:
        """401/404 are unrecoverable: log, set the fatal state and raise ``_FatalStreamError``."""
        if status_code == 401:
            logger.error(
                "[%s] Authentication failed (401) — stopping reconnect loop. Check NTFY_TOKEN.", self.name)
            code, detail = "ntfy_unauthorized", "ntfy server rejected auth (401). Check NTFY_TOKEN."
            reason = "401 Unauthorized"
        elif status_code == 404:
            logger.error("[%s] Topic not found (404): %s — stopping reconnect loop.", self.name, self._topic)
            code, detail = "ntfy_topic_not_found", f"ntfy topic '{self._topic}' returned 404. Check NTFY_TOPIC."
            reason = "404 Not Found"
        else:
            return
        self._set_fatal_error(code, detail, retryable=False)
        raise _FatalStreamError(reason)

    async def _consume_stream(self, url: str, headers: Dict[str, str]) -> None:
        """Open an HTTP streaming connection and dispatch events."""
        # poll=false keeps a persistent streaming connection alive with keepalive events
        async with self._http_client.stream(
            "GET", url, headers=headers, params={"poll": "false"},
            timeout=httpx.Timeout(connect=15.0, read=STREAM_TIMEOUT_SECONDS, write=15.0, pool=15.0),
        ) as response:
            self._fatal_status(response.status_code)
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not self._running:
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "message":
                    await self._on_message(event)

    async def disconnect(self) -> None:
        """Disconnect from ntfy."""
        self._running = False
        self._mark_disconnected()
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._seen_messages.clear()
        logger.info("[%s] Disconnected", self.name)

    # -- Inbound message processing -----------------------------------------

    async def _on_message(self, event: Dict[str, Any]) -> None:
        """Process an incoming ntfy message event."""
        msg_id = event.get("id") or uuid.uuid4().hex
        if self._is_duplicate(msg_id):
            logger.debug("[%s] Duplicate message %s, skipping", self.name, msg_id)
            return
        if _ECHO_TAG in (event.get("tags") or []):
            logger.debug("[%s] Skipping own message (echo tag)", self.name)
            return
        text = (event.get("message") or "").strip()
        if not text:
            logger.debug("[%s] Empty message body, skipping", self.name)
            return
        # No native user identity on ntfy: the publisher-controlled title must
        # NOT drive authorization, so user_id is fixed to the topic name.
        topic = event.get("topic") or self._topic
        source = self.build_source(
            chat_id=topic, chat_name=topic, chat_type="dm", user_id=topic, user_name=topic)
        unix_ts, timestamp = event.get("time"), datetime.now(tz=timezone.utc)
        try:
            timestamp = datetime.fromtimestamp(int(unix_ts), tz=timezone.utc) if unix_ts else timestamp
        except (ValueError, OSError, TypeError):
            pass
        message_event = MessageEvent(
            text=text, message_type=MessageType.TEXT, source=source, message_id=msg_id,
            raw_message=event, timestamp=timestamp)
        logger.debug("[%s] Message on topic %s: %s", self.name, topic, text[:80])
        await self.handle_message(message_event)

    def _is_duplicate(self, msg_id: str) -> bool:
        """True if this message ID was already seen within the dedup window."""
        now = time.time()
        if len(self._seen_messages) > DEDUP_MAX_SIZE:
            cutoff = now - DEDUP_WINDOW_SECONDS
            self._seen_messages = {k: v for k, v in self._seen_messages.items() if v > cutoff}
        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False

    # -- Outbound messaging -------------------------------------------------

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Publish a message to the configured publish topic."""
        publish_topic = self._resolve_publish_topic(chat_id, metadata)
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")
        headers = _publish_headers(self._token, bool((self.config.extra or {}).get("markdown", False)))
        body = _truncate_body(content, context=f"[{self.name}]")
        try:
            resp = await self._http_client.post(
                f"{self._server}/{publish_topic}", content=body, headers=headers, timeout=15.0)
            if resp.status_code < 300:
                return SendResult(success=True, message_id=_response_message_id(resp))
            body_text = resp.text
            logger.warning("[%s] Send failed HTTP %d: %s", self.name, resp.status_code, body_text[:200])
            return SendResult(success=False, error=f"HTTP {resp.status_code}: {body_text[:200]}")
        except httpx.TimeoutException:
            return SendResult(success=False, error="Timeout publishing to ntfy")
        except Exception as e:
            logger.error("[%s] Send error: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

    def _resolve_publish_topic(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> str:
        return (metadata or {}).get("publish_topic") or self._publish_topic or chat_id

    def _markdown_enabled(self) -> bool:
        return bool((self.config.extra or {}).get("markdown", False))

    def _attachment_cap(self, caption: Optional[str]) -> Optional[str]:
        """Caption for ``message=`` — byte-aware cap to ntfy's message limit."""
        return _cap_to_message_limit(caption, context="ntfy attachment caption")

    def _attachment_limit_bytes(self) -> int:
        """Fail-fast attachment cap for the configured server (see ``_attachment_max_bytes``)."""
        return _attachment_max_bytes(self.config.extra or {}, self._server)

    def _require_client(self) -> Optional[SendResult]:
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")
        return None

    async def _send_file_attachment(
        self, chat_id: str, file_path: str, *, caption: Optional[str] = None,
        file_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Publish a local file as an attachment (body upload + ``filename`` param)."""
        not_ready = self._require_client()
        if not_ready:
            return not_ready
        path, error = _read_attachment(file_path, max_bytes=self._attachment_limit_bytes())
        if error:
            logger.warning("[%s] Attachment send skipped: %s", self.name, error)
            return SendResult(success=False, error=error)
        publish_topic = self._resolve_publish_topic(chat_id, metadata)
        params = _attachment_fields(
            message=self._attachment_cap(caption), file_name=file_name or path.name)
        headers = _attachment_headers(self._token, self._markdown_enabled())
        try:
            body = path.read_bytes()
        except OSError as e:
            return SendResult(success=False, error=f"attachment file unreadable: {e}")
        return await _publish_attachment(
            self._http_client, self._server, publish_topic, headers=headers, params=params, body=body)

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a local file as an ntfy attachment; ``caption`` rides as the message text."""
        return await self._send_file_attachment(
            chat_id, file_path, caption=caption, file_name=file_name, metadata=metadata)

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a local image file as an ntfy attachment."""
        return await self._send_file_attachment(
            chat_id, image_path, caption=caption, metadata=metadata)

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a local video file as an ntfy attachment."""
        return await self._send_file_attachment(
            chat_id, video_path, caption=caption, metadata=metadata)

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """ntfy has no voice-note primitive: the audio file goes out as a normal attachment."""
        return await self._send_file_attachment(
            chat_id, audio_path, caption=caption, metadata=metadata)

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an image URL attachment: the ntfy SERVER fetches it (``attach=``), the
        publish body stays empty. Non-str URLs and non-http(s) URLs are refused without a POST."""
        if not isinstance(image_url, str):
            return SendResult(success=False, error=f"invalid image URL type: {type(image_url).__name__}")
        if not image_url.startswith(("http://", "https://")):
            return SendResult(success=False, error=f"not an http(s) URL: {image_url[:80]}")
        not_ready = self._require_client()
        if not_ready:
            return not_ready
        publish_topic = self._resolve_publish_topic(chat_id, metadata)
        params = _attachment_fields(message=self._attachment_cap(caption), attach_url=image_url)
        headers = _attachment_headers(self._token, self._markdown_enabled())
        return await _publish_attachment(
            self._http_client, self._server, publish_topic, headers=headers, params=params, body=b"")

    def _auth_headers(self) -> Dict[str, str]:
        return _build_auth_header(self._token)


# -- Plugin registration -----------------------------------------------------


def _env_enablement() -> dict | None:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Runs BEFORE adapter construction so ``gateway status`` reflects env-only
    setups without instantiating the HTTP client. ``None`` = not configured.
    The ``home_channel`` key is lifted by the core hook into a ``HomeChannel``
    on the ``PlatformConfig`` instead of being merged into ``extra``.
    """
    topic = _get_scoped_secret("NTFY_TOPIC", "").strip()
    if not topic:
        return None
    seed: dict = {
        "topic": topic, "server": _get_scoped_secret("NTFY_SERVER_URL", DEFAULT_SERVER).rstrip("/")}
    for key, env in (("publish_topic", "NTFY_PUBLISH_TOPIC"), ("token", "NTFY_TOKEN")):
        value = _get_scoped_secret(env, "").strip()
        if value:
            seed[key] = value
    markdown = _get_scoped_secret("NTFY_MARKDOWN", "").strip().lower()
    if markdown:
        seed["markdown"] = markdown in _MARKDOWN_TRUTHY
    home = _get_scoped_secret("NTFY_HOME_CHANNEL", "").strip() or topic
    if home:
        seed["home_channel"] = {"chat_id": home, "name": _get_scoped_secret("NTFY_HOME_CHANNEL_NAME", home)}
    return seed


async def _standalone_send(
    pconfig, chat_id: str, message: str, *,
    thread_id: Optional[str] = None, media_files: Optional[List[str]] = None, force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process publish for cron / send_message_tool when no gateway adapter is live.

    ``thread_id`` is signature parity only (ntfy has no threads). ``media_files`` —
    ``(path, is_voice)`` tuples — publish as attachments: the message text rides as the
    ``message=`` caption on the FIRST attachment only (association beyond one file is
    ambiguous); remaining attachments publish without text. ``force_document`` is a no-op
    (every ntfy attachment is transferred as a file). Markdown is honored if
    ``NTFY_MARKDOWN`` is set OR ``pconfig.extra["markdown"]`` is True.
    """
    if not HTTPX_AVAILABLE:
        return {"error": "ntfy standalone send: httpx not installed"}
    extra = getattr(pconfig, "extra", {}) or {}
    server = _server_url(extra)
    publish_topic = (
        chat_id or extra.get("publish_topic") or _get_scoped_secret("NTFY_PUBLISH_TOPIC", "").strip()
        or extra.get("topic") or _get_scoped_secret("NTFY_TOPIC", "").strip())
    if not publish_topic:
        return {"error": "ntfy standalone send: NTFY_TOPIC not configured"}
    token = _setting(extra, "token", "NTFY_TOKEN")
    markdown_env = _get_scoped_secret("NTFY_MARKDOWN", "").strip().lower()
    markdown = bool(extra.get("markdown")) or markdown_env in _MARKDOWN_TRUTHY
    files = [(path, bool(is_voice)) for path, is_voice in (media_files or [])]
    if files:
        caption = _cap_to_message_limit(
            (message or "").strip() or None, context="ntfy standalone")
        # Validate every attachment BEFORE posting any: a bad path (missing, not a file,
        # wrong type, oversized) fails the whole batch instead of delivering a prefix of it.
        # A file that becomes unreadable after this check, or a mid-batch publish failure,
        # can still leave earlier attachments delivered.
        read_paths = []
        max_bytes = _attachment_max_bytes(extra, server)
        for path_value, _is_voice in files:
            path, error = _read_attachment(path_value, max_bytes=max_bytes)
            if error:
                return {"error": f"ntfy standalone send: {error}"}
            read_paths.append(path)
        last_message_id: Optional[str] = None
        async with httpx.AsyncClient(timeout=120.0) as client:
            for index, path in enumerate(read_paths):
                params = _attachment_fields(
                    message=caption if index == 0 else None, file_name=path.name)
                headers = _attachment_headers(token, markdown)
                try:
                    body = path.read_bytes()
                except OSError as e:
                    return {"error": f"ntfy standalone send: attachment file unreadable: {e}"}
                result = await _publish_attachment(
                    client, server, publish_topic, headers=headers, params=params, body=body)
                if not result.success:
                    return {"error": f"ntfy standalone send: {result.error}"}
                last_message_id = result.message_id
        return {
            "success": True, "platform": "ntfy", "chat_id": publish_topic,
            "message_id": last_message_id}
    headers = _publish_headers(token, markdown, auth_first=False)
    body = _truncate_body(message, context="ntfy standalone")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{server}/{publish_topic}", content=body, headers=headers)
        if resp.status_code >= 300:
            return {"error": f"ntfy HTTP {resp.status_code}: {resp.text[:200]}"}
        return {"success": True, "platform": "ntfy", "chat_id": publish_topic, "message_id": _response_message_id(resp)}
    except Exception as e:
        return {"error": f"ntfy standalone send failed: {e}"}


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="ntfy", label="ntfy", adapter_factory=lambda cfg: NtfyAdapter(cfg),
        check_fn=check_requirements, validate_config=validate_config, is_connected=is_connected,
        required_env=["NTFY_TOPIC"], install_hint="pip install httpx   # already a Hermes dependency",
        env_enablement_fn=_env_enablement,  # env-only setups show in `gateway status`
        cron_deliver_env_var="NTFY_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,  # out-of-process cron delivery
        allowed_users_env="NTFY_ALLOWED_USERS", allow_all_env="NTFY_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH, emoji="🔔",
        pii_safe=True,  # topic names only — no phone numbers / emails to redact
        allow_update_command=True,
        platform_hint=(
            "You are communicating via ntfy push notifications. "
            "Use plain text by default — ntfy supports optional markdown "
            "(set markdown: true in config or NTFY_MARKDOWN=true). "
            "Keep responses concise; ntfy is a push notification service "
            "with a 4096-byte per-message limit."
        ))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
