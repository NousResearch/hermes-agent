"""ntfy platform adapter: HTTP-streaming subscription (``/json``, ``poll=false``) in, POST out.

config.yaml ``platforms.ntfy.extra``: ``server`` (default https://ntfy.sh), ``topic`` (required),
``publish_topic`` (defaults to topic), ``token`` (Bearer or ``user:pass`` Basic), ``markdown``
(default false). Env (read at construct time; ``extra`` wins over env): NTFY_TOPIC, NTFY_SERVER_URL,
NTFY_TOKEN, NTFY_PUBLISH_TOPIC, NTFY_MARKDOWN ("true"/"1"/"yes"), NTFY_ALLOWED_USERS (topic names),
NTFY_ALLOW_ALL_USERS (dev only), NTFY_HOME_CHANNEL, NTFY_HOME_CHANNEL_NAME.
Identity: ntfy has no authenticated user; ``title`` is publisher-controlled and NOT used for
authorization. Each topic is one trusted channel (``user_id`` == topic). Protect it with a read token.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret

logger = logging.getLogger(__name__)


class _FatalStreamError(Exception):
    """Unrecoverable stream error (401, 404)."""


DEFAULT_SERVER = "https://ntfy.sh"
MAX_MESSAGE_LENGTH = 4096  # ntfy message body limit
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


def _truncate_body(message: str, *, context: str) -> bytes:
    """Apply the ntfy 4096-char limit, logging a warning (tagged ``context``) on truncation."""
    if len(message) > MAX_MESSAGE_LENGTH:
        logger.warning(
            "%s: truncating message from %d to %d chars (ntfy limit)",
            context, len(message), MAX_MESSAGE_LENGTH)
    return message[:MAX_MESSAGE_LENGTH].encode("utf-8")


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
        publish_topic = (metadata or {}).get("publish_topic") or self._publish_topic or chat_id
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")
        headers = _publish_headers(self._token, bool((self.config.extra or {}).get("markdown", False)))
        if len(content) > self.MAX_MESSAGE_LENGTH:
            logger.warning(
                "[%s] Message truncated from %d to %d chars (ntfy limit)",
                self.name, len(content), self.MAX_MESSAGE_LENGTH)
        body = content[:self.MAX_MESSAGE_LENGTH].encode("utf-8")
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

    ``thread_id``/``media_files`` are signature parity only (ntfy has no thread
    or attachment primitive). Markdown is honored if ``NTFY_MARKDOWN`` is set
    OR ``pconfig.extra["markdown"]`` is True.
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
            "with a 4096-character per-message limit."
        ))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
