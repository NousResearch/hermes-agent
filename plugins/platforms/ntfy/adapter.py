"""ntfy platform adapter (Hermes plugin).

Subscribes to a topic on ntfy.sh or any self-hosted ntfy server via
HTTP streaming (``/json`` endpoint with ``poll=false``) and publishes
replies via HTTP POST. No external SDK — only httpx, which is already
a Hermes dependency.

This adapter ships as a Hermes platform plugin under
``plugins/platforms/ntfy/``. The Hermes plugin loader scans the
directory at startup, calls :func:`register`, and the platform becomes
available to ``gateway/run.py`` and ``tools/send_message_tool`` through
the registry — no edits to core files required.

Configuration in config.yaml::

    platforms:
      ntfy:
        enabled: true
        extra:
          server: "https://ntfy.sh"       # or self-hosted URL
          topic: "hermes-in"              # subscribe topic (incoming)
          publish_topic: "hermes-out"     # optional — defaults to topic
          token: "..."                    # optional Bearer / Basic auth token
          markdown: true                  # optional — enable markdown (default: false)

Environment variables (all read at adapter construct time, env wins over
config.yaml ``extra``):

    NTFY_TOPIC                 Topic to subscribe to (required)
    NTFY_SERVER_URL            Server URL (default: https://ntfy.sh)
    NTFY_TOKEN                 Bearer token or 'user:pass' for Basic auth
    NTFY_PUBLISH_TOPIC         Reply topic (defaults to NTFY_TOPIC)
    NTFY_MARKDOWN              "true"/"1"/"yes" enables X-Markdown header
    NTFY_ALLOWED_USERS         Allowlist (treated by gateway as user IDs;
                               on ntfy these are topic names)
    NTFY_ALLOW_ALL_USERS       Allow any topic — dev only
    NTFY_HOME_CHANNEL          Default topic for cron / notification delivery
    NTFY_HOME_CHANNEL_NAME     Human label for the home channel

Identity model: ntfy has no native authenticated user identity. The
``title`` field is publisher-controlled and is NOT used for
authorization. Each topic is treated as a single trusted channel —
``user_id`` is fixed to the topic name. Use a private topic protected
by a read token for any real trust boundary.
"""

import asyncio
import json
import logging
import os
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
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)


class _FatalStreamError(Exception):
    """Raised when a stream error is unrecoverable (e.g. 401, 404)."""


DEFAULT_SERVER = "https://ntfy.sh"
MAX_MESSAGE_LENGTH = 4096  # ntfy message body limit
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 1000
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]
STREAM_TIMEOUT_SECONDS = 90  # ntfy keepalive default is 55s; give margin
_ECHO_TAG = "hermes-agent"  # tag added to outgoing messages for echo-loop prevention


def _build_auth_header(token: str) -> Dict[str, str]:
    """Build an ``Authorization`` header from an ntfy token.

    Shared by :class:`NtfyAdapter._auth_headers` and :func:`_standalone_send`
    so both paths follow the same auth shape and whitespace-stripping rules.

    Tokens are stripped of surrounding whitespace — pasted tokens often
    carry trailing newlines that would otherwise render the header
    malformed (``Authorization: Bearer foo\\n``).  ``user:pass`` tokens
    become Basic auth; anything else is treated as a Bearer token.
    Returns ``{}`` when no token is configured.
    """
    if not token:
        return {}
    token = token.strip()
    if not token:
        return {}
    if ":" in token:
        import base64
        encoded = base64.b64encode(token.encode("utf-8")).decode("utf-8")
        return {"Authorization": f"Basic {encoded}"}
    return {"Authorization": f"Bearer {token}"}


def _truncate_body(message: str, *, context: str) -> bytes:
    """Apply the ntfy 4096-char limit, logging a warning on truncation.

    ``context`` is included in the log message so adapter and standalone
    truncations can be told apart in logs.
    """
    if len(message) > MAX_MESSAGE_LENGTH:
        logger.warning(
            "%s: truncating message from %d to %d chars (ntfy limit)",
            context, len(message), MAX_MESSAGE_LENGTH,
        )
    return message[:MAX_MESSAGE_LENGTH].encode("utf-8")


def check_requirements() -> bool:
    """Check whether the ntfy adapter is installable and minimally configured.

    Reads ``NTFY_TOPIC`` directly to avoid the cost of a full
    ``load_gateway_config()`` (which also writes to ``os.environ``) on
    every pre-flight check.
    """
    if not HTTPX_AVAILABLE:
        return False
    topic = os.getenv("NTFY_TOPIC", "").strip()
    return bool(topic)


def validate_config(config) -> bool:
    """Validate that the configured ntfy platform has a topic set."""
    extra = getattr(config, "extra", {}) or {}
    topic = extra.get("topic") or os.getenv("NTFY_TOPIC", "")
    return bool(topic)


def is_connected(config) -> bool:
    """Check whether ntfy is configured (env or config.yaml)."""
    extra = getattr(config, "extra", {}) or {}
    topic = os.getenv("NTFY_TOPIC") or extra.get("topic", "")
    return bool(topic)


class NtfyAdapter(BasePlatformAdapter):
    """ntfy adapter.

    Subscribes to a topic via HTTP streaming (``/json`` endpoint) and
    publishes replies via HTTP POST. No external SDK — only httpx.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        platform = Platform("ntfy")
        super().__init__(config=config, platform=platform)

        extra = config.extra or {}
        self._server: str = (
            extra.get("server")
            or os.getenv("NTFY_SERVER_URL", DEFAULT_SERVER)
        ).rstrip("/")
        self._topic: str = extra.get("topic") or os.getenv("NTFY_TOPIC", "")
        self._publish_topic: str = (
            extra.get("publish_topic")
            or os.getenv("NTFY_PUBLISH_TOPIC", "")
            or self._topic
        )
        self._token: str = extra.get("token") or os.getenv("NTFY_TOKEN", "")

        self._stream_task: Optional[asyncio.Task] = None
        self._http_client: Optional["httpx.AsyncClient"] = None

        # Message deduplication: msg_id -> timestamp
        self._seen_messages: Dict[str, float] = {}

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self) -> bool:
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

    async def _consume_stream(self, url: str, headers: Dict[str, str]) -> None:
        """Open an HTTP streaming connection and dispatch events."""
        # poll=false keeps a persistent streaming connection alive with keepalive events
        params = {"poll": "false"}
        async with self._http_client.stream(
            "GET",
            url,
            headers=headers,
            params=params,
            timeout=httpx.Timeout(connect=15.0, read=STREAM_TIMEOUT_SECONDS, write=15.0, pool=15.0),
        ) as response:
            if response.status_code == 401:
                logger.error(
                    "[%s] Authentication failed (401) — stopping reconnect loop. Check NTFY_TOKEN.",
                    self.name,
                )
                self._set_fatal_error(
                    "ntfy_unauthorized",
                    "ntfy server rejected auth (401). Check NTFY_TOKEN.",
                    retryable=False,
                )
                raise _FatalStreamError("401 Unauthorized")
            if response.status_code == 404:
                logger.error(
                    "[%s] Topic not found (404): %s — stopping reconnect loop.",
                    self.name, self._topic,
                )
                self._set_fatal_error(
                    "ntfy_topic_not_found",
                    f"ntfy topic '{self._topic}' returned 404. Check NTFY_TOPIC.",
                    retryable=False,
                )
                raise _FatalStreamError("404 Not Found")
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

        # Echo-loop prevention: skip messages tagged by this adapter.
        tags = event.get("tags") or []
        if _ECHO_TAG in tags:
            logger.debug("[%s] Skipping own message (echo tag)", self.name)
            return

        text = (event.get("message") or "").strip()
        if not text:
            logger.debug("[%s] Empty message body, skipping", self.name)
            return

        topic = event.get("topic") or self._topic
        # ntfy has no native authenticated user identity. The title field is
        # publisher-controlled and must NOT be used for authorization — any
        # publisher who knows the topic can set title to an allowed username.
        # Treat ntfy as a single trusted channel; user_id is fixed to the
        # topic name. NTFY_ALLOWED_USERS is only a real trust boundary when
        # the topic itself is protected by a read token.
        user_id = topic
        user_name = topic

        source = self.build_source(
            chat_id=topic,
            chat_name=topic,
            chat_type="dm",
            user_id=user_id,
            user_name=user_name,
        )

        unix_ts = event.get("time")
        try:
            timestamp = (
                datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)
                if unix_ts else datetime.now(tz=timezone.utc)
            )
        except (ValueError, OSError, TypeError):
            timestamp = datetime.now(tz=timezone.utc)

        message_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=msg_id,
            raw_message=event,
            timestamp=timestamp,
        )

        logger.debug("[%s] Message on topic %s: %s", self.name, topic, text[:80])
        await self.handle_message(message_event)

    # -- Deduplication ------------------------------------------------------

    def _is_duplicate(self, msg_id: str) -> bool:
        """Return True if this message ID was already seen within the dedup window."""
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
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Publish a message to the configured publish topic."""
        metadata = metadata or {}
        publish_topic = metadata.get("publish_topic") or self._publish_topic or chat_id

        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")

        url = f"{self._server}/{publish_topic}"
        markdown_enabled = (self.config.extra or {}).get("markdown", False)
        headers = {
            **self._auth_headers(),
            "Content-Type": "text/plain; charset=utf-8",
            "X-Tags": _ECHO_TAG,
        }
        if markdown_enabled:
            headers["X-Markdown"] = "true"

        if len(content) > self.MAX_MESSAGE_LENGTH:
            logger.warning(
                "[%s] Message truncated from %d to %d chars (ntfy limit)",
                self.name, len(content), self.MAX_MESSAGE_LENGTH,
            )
        body = content[:self.MAX_MESSAGE_LENGTH]

        try:
            resp = await self._http_client.post(
                url, content=body.encode("utf-8"), headers=headers, timeout=15.0,
            )
            if resp.status_code < 300:
                try:
                    data = resp.json()
                    returned_id = data.get("id") or uuid.uuid4().hex[:12]
                except Exception:
                    returned_id = uuid.uuid4().hex[:12]
                return SendResult(success=True, message_id=returned_id)
            body_text = resp.text
            logger.warning("[%s] Send failed HTTP %d: %s", self.name, resp.status_code, body_text[:200])
            return SendResult(success=False, error=f"HTTP {resp.status_code}: {body_text[:200]}")
        except httpx.TimeoutException:
            return SendResult(success=False, error="Timeout publishing to ntfy")
        except Exception as e:
            logger.error("[%s] Send error: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """ntfy does not support typing indicators."""
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about an ntfy topic."""
        return {"name": chat_id, "type": "dm"}

    # -- Helpers ------------------------------------------------------------

    def _auth_headers(self) -> Dict[str, str]:
        """Build Authorization header if a token is configured."""
        return _build_auth_header(self._token)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def _env_enablement() -> dict | None:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Called by the platform registry's env-enablement hook BEFORE adapter
    construction, so ``gateway status`` and ``get_connected_platforms()``
    reflect env-only configuration without instantiating the HTTP client.
    Returns ``None`` when ntfy isn't minimally configured; the caller skips
    auto-enabling.

    The special ``home_channel`` key in the returned dict is handled by the
    core hook — it becomes a proper ``HomeChannel`` dataclass on the
    ``PlatformConfig`` rather than being merged into ``extra``.
    """
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        return None
    seed: dict = {
        "topic": topic,
        "server": os.getenv("NTFY_SERVER_URL", DEFAULT_SERVER).rstrip("/"),
    }
    publish_topic = os.getenv("NTFY_PUBLISH_TOPIC", "").strip()
    if publish_topic:
        seed["publish_topic"] = publish_topic
    token = os.getenv("NTFY_TOKEN", "").strip()
    if token:
        seed["token"] = token
    markdown = os.getenv("NTFY_MARKDOWN", "").strip().lower()
    if markdown:
        seed["markdown"] = markdown in ("1", "true", "yes")
    home = os.getenv("NTFY_HOME_CHANNEL", "").strip() or topic
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("NTFY_HOME_CHANNEL_NAME", home),
        }
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process publish for cron / send_message_tool fallbacks.

    Used by ``tools/send_message_tool._send_via_adapter`` and the cron
    scheduler when the gateway runner is not in this process (e.g.
    ``hermes cron`` running standalone). Without this hook,
    ``deliver=ntfy`` cron jobs fail with ``No live adapter for platform``.

    ``thread_id`` and ``media_files`` are accepted for signature parity
    only — ntfy has no thread or attachment primitive. Markdown is
    honored if ``NTFY_MARKDOWN`` is set OR ``pconfig.extra["markdown"]``
    is True.
    """
    if not HTTPX_AVAILABLE:
        return {"error": "ntfy standalone send: httpx not installed"}

    extra = getattr(pconfig, "extra", {}) or {}
    server = (
        extra.get("server")
        or os.getenv("NTFY_SERVER_URL", DEFAULT_SERVER)
    ).rstrip("/")
    publish_topic = (
        chat_id
        or extra.get("publish_topic")
        or os.getenv("NTFY_PUBLISH_TOPIC", "").strip()
        or extra.get("topic")
        or os.getenv("NTFY_TOPIC", "").strip()
    )
    if not publish_topic:
        return {"error": "ntfy standalone send: NTFY_TOPIC not configured"}

    token = extra.get("token") or os.getenv("NTFY_TOKEN", "")
    markdown_env = os.getenv("NTFY_MARKDOWN", "").strip().lower()
    markdown_enabled = bool(extra.get("markdown")) or markdown_env in ("1", "true", "yes")

    headers = {"Content-Type": "text/plain; charset=utf-8", "X-Tags": _ECHO_TAG, **_build_auth_header(token)}
    if markdown_enabled:
        headers["X-Markdown"] = "true"

    body = _truncate_body(message, context="ntfy standalone")

    url = f"{server}/{publish_topic}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, content=body, headers=headers)
        if resp.status_code >= 300:
            return {"error": f"ntfy HTTP {resp.status_code}: {resp.text[:200]}"}
        try:
            data = resp.json()
            msg_id = data.get("id") or uuid.uuid4().hex[:12]
        except Exception:
            msg_id = uuid.uuid4().hex[:12]
        return {"success": True, "platform": "ntfy", "chat_id": publish_topic, "message_id": msg_id}
    except Exception as e:
        return {"error": f"ntfy standalone send failed: {e}"}


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="ntfy",
        label="ntfy",
        adapter_factory=lambda cfg: NtfyAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["NTFY_TOPIC"],
        install_hint="pip install httpx   # already a Hermes dependency",
        # Env-driven auto-configuration: seeds PlatformConfig.extra so
        # env-only setups show up in `hermes gateway status` without
        # instantiating the HTTP client.
        env_enablement_fn=_env_enablement,
        # Cron home-channel delivery support — `deliver=ntfy` cron jobs
        # route to NTFY_HOME_CHANNEL when set.
        cron_deliver_env_var="NTFY_HOME_CHANNEL",
        # Out-of-process cron delivery. Without this hook, deliver=ntfy
        # cron jobs fail with "No live adapter" when cron runs separately
        # from the gateway.
        standalone_sender_fn=_standalone_send,
        # Auth env vars for _is_user_authorized() integration.
        allowed_users_env="NTFY_ALLOWED_USERS",
        allow_all_env="NTFY_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🔔",
        # ntfy publishers have no persistent identity — topic names are
        # the only identifier, no phone numbers / emails to redact.
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are communicating via ntfy push notifications. "
            "Use plain text by default — ntfy supports optional markdown "
            "(set markdown: true in config or NTFY_MARKDOWN=true). "
            "Keep responses concise; ntfy is a push notification service "
            "with a 4096-character per-message limit."
        ),
    )
