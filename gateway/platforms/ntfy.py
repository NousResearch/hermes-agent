"""
ntfy platform adapter.

Uses httpx streaming to receive messages published to a subscribed topic,
and HTTP POST to publish replies. Works with ntfy.sh or any self-hosted
ntfy server.

Requires:
    pip install httpx  (already a dependency)
    NTFY_TOPIC env var (and optionally NTFY_SERVER_URL, NTFY_TOKEN,
    NTFY_PUBLISH_TOPIC)

Configuration in config.yaml:
    platforms:
      ntfy:
        enabled: true
        extra:
          server: "https://ntfy.sh"       # or self-hosted URL
          topic: "hermes-in"              # subscribe topic (incoming)
          publish_topic: "hermes-out"     # optional — defaults to topic
          token: "..."                    # optional Bearer / Basic auth token
          markdown: true                  # optional — enable markdown formatting (default: false)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

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


def check_ntfy_requirements() -> bool:
    """Check if ntfy adapter dependencies are available and configured."""
    if not HTTPX_AVAILABLE:
        return False
    # Check env var directly — avoids the full config load (which also
    # writes to os.environ) on every adapter pre-check call.
    topic = os.getenv("NTFY_TOPIC", "").strip()
    return bool(topic)


class NtfyAdapter(BasePlatformAdapter):
    """ntfy adapter.

    Subscribes to a topic via HTTP streaming (/json endpoint) and publishes
    replies via HTTP POST. No external SDK — only httpx is required.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.NTFY)

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
                logger.error("[%s] Authentication failed (401) — stopping reconnect loop. Check NTFY_TOKEN.", self.name)
                raise _FatalStreamError("401 Unauthorized")
            if response.status_code == 404:
                logger.error("[%s] Topic not found (404): %s — stopping reconnect loop.", self.name, self._topic)
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

        text = (event.get("message") or "").strip()
        if not text:
            logger.debug("[%s] Empty message body, skipping", self.name)
            return

        topic = event.get("topic") or self._topic
        # ntfy has no native authenticated user identity. The title field is
        # publisher-controlled and must NOT be used for authorization — any
        # publisher who knows the topic can set title to an allowed username.
        # Treat ntfy as a single trusted channel; user_id is fixed to the
        # topic name. Document that NTFY_ALLOWED_USERS is only a real trust
        # boundary when the topic has a read token protecting it.
        user_id = topic
        user_name = topic

        source = self.build_source(
            chat_id=topic,
            chat_name=topic,
            chat_type="dm",
            user_id=user_id,
            user_name=user_name,
        )

        # Parse timestamp
        unix_ts = event.get("time")
        try:
            timestamp = datetime.fromtimestamp(int(unix_ts), tz=timezone.utc) if unix_ts else datetime.now(tz=timezone.utc)
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
        headers = {**self._auth_headers(), "Content-Type": "text/plain; charset=utf-8"}
        if markdown_enabled:
            headers["X-Markdown"] = "true"

        if len(content) > self.MAX_MESSAGE_LENGTH:
            logger.warning(
                "[%s] Message truncated from %d to %d chars (ntfy limit)",
                self.name, len(content), self.MAX_MESSAGE_LENGTH,
            )
        body = content[:self.MAX_MESSAGE_LENGTH]

        try:
            resp = await self._http_client.post(url, content=body.encode("utf-8"), headers=headers, timeout=15.0)
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
        if not self._token:
            return {}
        # ntfy supports both Bearer tokens and Base64-encoded Basic auth;
        # prefer Bearer for API tokens, Basic for username:password pairs.
        if ":" in self._token:
            import base64
            encoded = base64.b64encode(self._token.encode()).decode()
            return {"Authorization": f"Basic {encoded}"}
        return {"Authorization": f"Bearer {self._token}"}
