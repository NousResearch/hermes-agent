"""
QQ Bot platform adapter using the Official QQ Bot API (v2).

Connects to the QQ Bot WebSocket Gateway for inbound events and uses the
REST API (``api.sgroup.qq.com``) for outbound messages and media uploads.

Configuration in config.yaml:
    platforms:
      qq:
        enabled: true
        extra:
          app_id: "your-app-id"            # or QQ_APP_ID env var
          client_secret: "your-secret"     # or QQ_CLIENT_SECRET env var
          markdown_support: true           # enable QQ markdown (msg_type 2)
          dm_policy: "open"                # open | allowlist | disabled
          allow_from: ["openid_1"]
          group_policy: "open"             # open | allowlist | disabled
          group_allow_from: ["group_openid_1"]
          stt:                             # Voice-to-text config (optional)
            provider: "zai"                # zai (GLM-ASR), openai (Whisper), etc.
            baseUrl: "https://open.bigmodel.cn/api/coding/paas/v4"
            apiKey: "your-stt-api-key"     # or set QQ_STT_API_KEY env var
            model: "glm-asr"               # glm-asr, whisper-1, etc.

    Voice transcription priority:
      1. QQ's built-in ``asr_refer_text`` (Tencent ASR — free, always tried first)
      2. Configured STT provider via ``stt`` config or ``QQ_STT_*`` env vars

Reference: https://bot.q.qq.com/wiki/develop/api-v2/
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]

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
    _ssrf_redirect_guard,
    cache_document_from_bytes,
    cache_image_from_bytes,
)
from gateway.platforms.helpers import strip_markdown

logger = logging.getLogger(__name__)


class QQCloseError(Exception):
    """Raised when QQ WebSocket closes with a specific code.

    Carries the close code and reason for proper handling in the reconnect loop.
    """

    def __init__(self, code, reason=""):
        self.code = int(code) if code else None
        self.reason = str(reason) if reason else ""
        super().__init__(f"WebSocket closed (code={self.code}, reason={self.reason})")


# ---------------------------------------------------------------------------
# Constants — imported from the shared constants module.
# ---------------------------------------------------------------------------

from gateway.platforms.qqbot.constants import (
    API_BASE,
    TOKEN_URL,
    GATEWAY_URL_PATH,
    DEFAULT_API_TIMEOUT,
    FILE_UPLOAD_TIMEOUT,
    CONNECT_TIMEOUT_SECONDS,
    RECONNECT_BACKOFF,
    MAX_RECONNECT_ATTEMPTS,
    RATE_LIMIT_DELAY,
    QUICK_DISCONNECT_THRESHOLD,
    MAX_QUICK_DISCONNECT_COUNT,
    MAX_MESSAGE_LENGTH,
    DEDUP_WINDOW_SECONDS,
    DEDUP_MAX_SIZE,
    MSG_TYPE_TEXT,
    MSG_TYPE_MARKDOWN,
    MSG_TYPE_MEDIA,
    MSG_TYPE_INPUT_NOTIFY,
    MEDIA_TYPE_IMAGE,
    MEDIA_TYPE_VIDEO,
    MEDIA_TYPE_VOICE,
    MEDIA_TYPE_FILE,
)
from gateway.platforms.qqbot.utils import (
    coerce_list as _coerce_list_impl,
    build_user_agent,
)


def check_qq_requirements() -> bool:
    """Check if QQ runtime dependencies are available."""
    return AIOHTTP_AVAILABLE and HTTPX_AVAILABLE


def _coerce_list(value: Any) -> List[str]:
    """Coerce config values into a trimmed string list."""
    return _coerce_list_impl(value)


# ---------------------------------------------------------------------------
# QQAdapter
# ---------------------------------------------------------------------------


class QQAdapter(BasePlatformAdapter):
    """QQ Bot adapter backed by the official QQ Bot WebSocket Gateway + REST API."""

    # QQ Bot API does not support editing sent messages.
    SUPPORTS_MESSAGE_EDITING = False
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    _TYPING_INPUT_SECONDS = 60  # input_notify duration reported to QQ
    _TYPING_DEBOUNCE_SECONDS = 50  # refresh before it expires

    @property
    def _log_tag(self) -> str:
        """Log prefix including app_id for multi-instance disambiguation."""
        app_id = getattr(self, "_app_id", None)
        if app_id:
            return f"QQBot:{app_id}"
        return "QQBot"

    def _fail_pending(self, reason: str) -> None:
        """Fail all pending response futures."""
        for fut in self._pending_responses.values():
            if not fut.done():
                fut.set_exception(RuntimeError(reason))
        self._pending_responses.clear()

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.QQBOT)

        extra = config.extra or {}
        self._app_id = str(extra.get("app_id") or os.getenv("QQ_APP_ID", "")).strip()
        self._client_secret = str(
            extra.get("client_secret") or os.getenv("QQ_CLIENT_SECRET", "")
        ).strip()
        self._markdown_support = bool(extra.get("markdown_support", True))

        # Auth/ACL policies
        self._dm_policy = str(extra.get("dm_policy", "open")).strip().lower()
        self._allow_from = _coerce_list(
            extra.get("allow_from") or extra.get("allowFrom")
        )
        self._group_policy = str(extra.get("group_policy", "open")).strip().lower()
        self._group_allow_from = _coerce_list(
            extra.get("group_allow_from") or extra.get("groupAllowFrom")
        )

        # Connection state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval: float = 30.0  # seconds, updated by Hello
        self._session_id: Optional[str] = None
        self._last_seq: Optional[int] = None
        self._chat_type_map: Dict[str, str] = {}  # chat_id → "c2c"|"group"|"guild"|"dm"

        # Request/response correlation
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._seen_messages: Dict[str, float] = {}

        # Last inbound message ID per chat — used by send_typing
        self._last_msg_id: Dict[str, str] = {}
        # Typing debounce: chat_id → last send_typing timestamp
        self._typing_sent_at: Dict[str, float] = {}

        # Token cache
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()

        # Upload cache: content_hash -> {file_info, file_uuid, expires_at}
        self._upload_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "QQBot"

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Authenticate, obtain gateway URL, and open the WebSocket."""
        if not AIOHTTP_AVAILABLE:
            message = "QQ startup failed: aiohttp not installed"
            self._set_fatal_error("qq_missing_dependency", message, retryable=True)
            logger.warning("[%s] %s. Run: pip install aiohttp", self._log_tag, message)
            return False
        if not HTTPX_AVAILABLE:
            message = "QQ startup failed: httpx not installed"
            self._set_fatal_error("qq_missing_dependency", message, retryable=True)
            logger.warning("[%s] %s. Run: pip install httpx", self._log_tag, message)
            return False
        if not self._app_id or not self._client_secret:
            message = "QQ startup failed: QQ_APP_ID and QQ_CLIENT_SECRET are required"
            self._set_fatal_error("qq_missing_credentials", message, retryable=True)
            logger.warning("[%s] %s", self._log_tag, message)
            return False

        # Prevent duplicate connections with the same credentials
        if not self._acquire_platform_lock("qqbot-appid", self._app_id, "QQBot app ID"):
            return False

        try:
            # Tighter keepalive pool so idle CLOSE_WAIT sockets drain
            # faster behind proxies like Cloudflare Warp (#18451).
            from gateway.platforms._http_client_limits import platform_httpx_limits
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                event_hooks={"response": [_ssrf_redirect_guard]},
                limits=platform_httpx_limits(),
            )

            # 1. Get access token
            await self._ensure_token()

            # 2. Get WebSocket gateway URL
            gateway_url = await self._get_gateway_url()
            logger.info("[%s] Gateway URL: %s", self._log_tag, gateway_url)

            # 3. Open WebSocket
            await self._open_ws(gateway_url)

            # 4. Start listeners
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._mark_connected()
            logger.info("[%s] Connected", self._log_tag)
            return True
        except Exception as exc:
            message = f"QQ startup failed: {exc}"
            self._set_fatal_error("qq_connect_error", message, retryable=True)
            logger.error("[%s] %s", self._log_tag, message, exc_info=True)
            await self._cleanup()
            self._release_platform_lock()
            return False

    async def disconnect(self) -> None:
        """Close all connections and stop listeners."""
        self._running = False
        self._mark_disconnected()

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        await self._cleanup()
        self._release_platform_lock()
        logger.info("[%s] Disconnected", self._log_tag)

    async def _cleanup(self) -> None:
        """Close WebSocket, HTTP session, and client."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        # Fail pending
        for fut in self._pending_responses.values():
            if not fut.done():
                fut.set_exception(RuntimeError("Disconnected"))
        self._pending_responses.clear()

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    async def _ensure_token(self) -> str:
        """Return a valid access token, refreshing if needed (with singleflight)."""
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token

        async with self._token_lock:
            # Double-check after acquiring lock
            if self._access_token and time.time() < self._token_expires_at - 60:
                return self._access_token

            try:
                resp = await self._http_client.post(
                    TOKEN_URL,
                    json={"appId": self._app_id, "clientSecret": self._client_secret},
                    timeout=DEFAULT_API_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                raise RuntimeError(f"Failed to get QQ Bot access token: {exc}") from exc

            token = data.get("access_token")
            if not token:
                raise RuntimeError(
                    f"QQ Bot token response missing access_token: {data}"
                )

            expires_in = int(data.get("expires_in", 7200))
            self._access_token = token
            self._token_expires_at = time.time() + expires_in
            logger.info(
                "[%s] Access token refreshed, expires in %ds", self._log_tag, expires_in
            )
            return self._access_token

    async def _get_gateway_url(self) -> str:
        """Fetch the WebSocket gateway URL from the REST API."""
        token = await self._ensure_token()
        try:
            resp = await self._http_client.get(
                f"{API_BASE}{GATEWAY_URL_PATH}",
                headers={
                    "Authorization": f"QQBot {token}",
                    "User-Agent": build_user_agent(),
                },
                timeout=DEFAULT_API_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"Failed to get QQ Bot gateway URL: {exc}") from exc

        url = data.get("url")
        if not url:
            raise RuntimeError(f"QQ Bot gateway response missing url: {data}")
        return url

    # ------------------------------------------------------------------
    # WebSocket lifecycle
    # ------------------------------------------------------------------

    async def _open_ws(self, gateway_url: str) -> None:
        """Open a WebSocket connection to the QQ Bot gateway."""
        # Only clean up WebSocket resources — keep _http_client alive for REST API calls.
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

        # Honor WSL proxy env for QQ WebSocket. Hermes upgrades overwrite this
        # local patch, so QQ can regress to direct-connect timeouts after update.
        self._session = aiohttp.ClientSession(trust_env=True)
        ws_proxy = (
            os.getenv("WSS_PROXY")
            or os.getenv("wss_proxy")
            or os.getenv("HTTPS_PROXY")
            or os.getenv("https_proxy")
            or os.getenv("ALL_PROXY")
            or os.getenv("all_proxy")
        )
        self._ws = await self._session.ws_connect(
            gateway_url,
            headers={
                "User-Agent": build_user_agent(),
            },
            timeout=CONNECT_TIMEOUT_SECONDS,
            proxy=ws_proxy,
        )
        logger.info("[%s] WebSocket connected to %s", self._log_tag, gateway_url)

    async def _listen_loop(self) -> None:
        """Read WebSocket events and reconnect on errors.

        Close code handling follows the OpenClaw qqbot reference implementation:
          4004 → invalid token, refresh and reconnect
          4006/4007/4009 → session invalid, clear session and re-identify
          4008 → rate limited, back off 60s
          4914 → bot offline/sandbox, stop reconnecting
          4915 → bot banned, stop reconnecting
        """
        backoff_idx = 0
        connect_time = 0.0
        quick_disconnect_count = 0

        while self._running:
            try:
                connect_time = time.monotonic()
                await self._read_events()
                backoff_idx = 0
                quick_disconnect_count = 0
            except asyncio.CancelledError:
                return
            except QQCloseError as exc:
                if not self._running:
                    return

                code = exc.code
                logger.warning(
                    "[%s] WebSocket closed: code=%s reason=%s",
                    self._log_tag,
                    code,
                    exc.reason,
                )

                # Quick disconnect detection (permission issues, misconfiguration)
                duration = time.monotonic() - connect_time
                if duration < QUICK_DISCONNECT_THRESHOLD and connect_time > 0:
                    quick_disconnect_count += 1
                    logger.info(
                        "[%s] Quick disconnect (%.1fs), count: %d",
                        self._log_tag,
                        duration,
                        quick_disconnect_count,
                    )
                    if quick_disconnect_count >= MAX_QUICK_DISCONNECT_COUNT:
                        logger.error(
                            "[%s] Too many quick disconnects. "
                            "Check: 1) AppID/Secret correct 2) Bot permissions on QQ Open Platform",
                            self._log_tag,
                        )
                        self._set_fatal_error(
                            "qq_quick_disconnect",
                            "Too many quick disconnects — check bot permissions",
                            retryable=True,
                        )
                        return
                else:
                    quick_disconnect_count = 0

                self._mark_disconnected()
                self._fail_pending("Connection closed")

                # Stop reconnecting for fatal codes
                if code in (4914, 4915):
                    desc = "offline/sandbox-only" if code == 4914 else "banned"
                    logger.error(
                        "[%s] Bot is %s. Check QQ Open Platform.", self._log_tag, desc
                    )
                    self._set_fatal_error(
                        f"qq_{desc}", f"Bot is {desc}", retryable=False
                    )
                    return

                # Rate limited
                if code == 4008:
                    logger.info(
                        "[%s] Rate limited (4008), waiting %ds",
                        self._log_tag,
                        RATE_LIMIT_DELAY,
                    )
                    if backoff_idx >= MAX_RECONNECT_ATTEMPTS:
                        return
                    await asyncio.sleep(RATE_LIMIT_DELAY)
                    if await self._reconnect(backoff_idx):
                        backoff_idx = 0
                        quick_disconnect_count = 0
                    else:
                        backoff_idx += 1
                    continue

                # Token invalid → clear cached token so _ensure_token() refreshes
                if code == 4004:
                    logger.info(
                        "[%s] Invalid token (4004), will refresh and reconnect",
                        self._log_tag,
                    )
                    self._access_token = None
                    self._token_expires_at = 0.0

                # Session invalid → clear session, will re-identify on next Hello
                if code in (
                        4006,
                        4007,
                        4009,
                        4900,
                        4901,
                        4902,
                        4903,
                        4904,
                        4905,
                        4906,
                        4907,
                        4908,
                        4909,
                        4910,
                        4911,
                        4912,
                        4913,
                ):
                    logger.info(
                        "[%s] Session error (%d), clearing session for re-identify",
                        self._log_tag,
                        code,
                    )
                    self._session_id = None
                    self._last_seq = None

                if await self._reconnect(backoff_idx):
                    backoff_idx = 0
                    quick_disconnect_count = 0
                else:
                    backoff_idx += 1
                    if backoff_idx >= MAX_RECONNECT_ATTEMPTS:
                        logger.error("[%s] Max reconnect attempts reached (QQCloseError)", self._log_tag)
                        return

            except Exception as exc:
                if not self._running:
                    return
                logger.warning("[%s] WebSocket error: %s", self._log_tag, exc)
                self._mark_disconnected()
                self._fail_pending("Connection interrupted")

                if backoff_idx >= MAX_RECONNECT_ATTEMPTS:
                    logger.error("[%s] Max reconnect attempts reached", self._log_tag)
                    return

                if await self._reconnect(backoff_idx):
                    backoff_idx = 0
                    quick_disconnect_count = 0
                else:
                    backoff_idx += 1

    async def _reconnect(self, backoff_idx: int) -> bool:
        """Attempt to reconnect the WebSocket. Returns True on success."""
        delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
        logger.info(
            "[%s] Reconnecting in %ds (attempt %d)...",
            self._log_tag,
            delay,
            backoff_idx + 1,
        )
        await asyncio.sleep(delay)

        self._heartbeat_interval = 30.0  # reset until Hello
        try:
            await self._ensure_token()
            gateway_url = await self._get_gateway_url()
            await self._open_ws(gateway_url)
            self._mark_connected()
            logger.info("[%s] Reconnected", self._log_tag)
            return True
        except Exception as exc:
            logger.warning("[%s] Reconnect failed: %s", self._log_tag, exc)
            return False

    async def _read_events(self) -> None:
        """Read WebSocket frames until connection closes."""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        while self._running and self._ws and not self._ws.closed:
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                payload = self._parse_json(msg.data)
                if payload:
                    self._dispatch_payload(payload)
            elif msg.type in (aiohttp.WSMsgType.PING,):
                # aiohttp auto-replies with PONG
                pass
            elif msg.type == aiohttp.WSMsgType.CLOSE:
                raise QQCloseError(msg.data, msg.extra)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                raise RuntimeError("WebSocket closed")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats (QQ Gateway expects op 1 heartbeat with latest seq).

        The interval is set from the Hello (op 10) event's heartbeat_interval.
        QQ's default is ~41s; we send at 80% of the interval to stay safe.
        """
        try:
            while self._running:
                await asyncio.sleep(self._heartbeat_interval)
                if not self._ws or self._ws.closed:
                    continue
                try:
                    # d should be the latest sequence number received, or null
                    await self._ws.send_json({"op": 1, "d": self._last_seq})
                except Exception as exc:
                    logger.debug("[%s] Heartbeat failed: %s", self._log_tag, exc)
        except asyncio.CancelledError:
            pass

    async def _send_identify(self) -> None:
        """Send op 2 Identify to authenticate the WebSocket connection.

        After receiving op 10 Hello, the client must send op 2 Identify with
        the bot token and intents. On success the server replies with a
        READY dispatch event.

        Reference: https://bot.q.qq.com/wiki/develop/api-v2/dev-prepare/interface-framework/reference.html
        """
        token = await self._ensure_token()
        identify_payload = {
            "op": 2,
            "d": {
                "token": f"QQBot {token}",
                "intents": (1 << 25)
                           | (1 << 30)
                           | (
                                   1 << 12
                           ),  # C2C_GROUP_AT_MESSAGES + PUBLIC_GUILD_MESSAGES + DIRECT_MESSAGE
                "shard": [0, 1],
                "properties": {
                    "$os": "macOS",
                    "$browser": "hermes-agent",
                    "$device": "hermes-agent",
                },
            },
        }
        try:
            if self._ws and not self._ws.closed:
                await self._ws.send_json(identify_payload)
                logger.info("[%s] Identify sent", self._log_tag)
            else:
                logger.warning(
                    "[%s] Cannot send Identify: WebSocket not connected", self._log_tag
                )
        except Exception as exc:
            logger.error("[%s] Failed to send Identify: %s", self._log_tag, exc)

    async def _send_resume(self) -> None:
        """Send op 6 Resume to re-authenticate after a reconnection.

        Reference: https://bot.q.qq.com/wiki/develop/api-v2/dev-prepare/interface-framework/reference.html
        """
        token = await self._ensure_token()
        resume_payload = {
            "op": 6,
            "d": {
                "token": f"QQBot {token}",
                "session_id": self._session_id,
                "seq": self._last_seq,
            },
        }
        try:
            if self._ws and not self._ws.closed:
                await self._ws.send_json(resume_payload)
                logger.info(
                    "[%s] Resume sent (session_id=%s, seq=%s)",
                    self._log_tag,
                    self._session_id,
                    self._last_seq,
                )
            else:
                logger.warning(
                    "[%s] Cannot send Resume: WebSocket not connected", self._log_tag
                )
        except Exception as exc:
            logger.error("[%s] Failed to send Resume: %s", self._log_tag, exc)
            # If resume fails, clear session and fall back to identify on next Hello
            self._session_id = None
            self._last_seq = None

    @staticmethod
    def _create_task(coro):
        """Schedule a coroutine, silently skipping if no event loop is running.

        This avoids ``RuntimeError: no running event loop`` when tests call
        ``_dispatch_payload`` synchronously outside of ``asyncio.run()``.
        """
        try:
            loop = asyncio.get_running_loop()
            return loop.create_task(coro)
        except RuntimeError:
            return None

    def _dispatch_payload(self, payload: Dict[str, Any]) -> None:
        """Route inbound WebSocket payloads (dispatch synchronously, spawn async handlers)."""
        op = payload.get("op")
        t = payload.get("t")
        s = payload.get("s")
        d = payload.get("d")
        if isinstance(s, int) and (self._last_seq is None or s > self._last_seq):
            self._last_seq = s

        # op 10 = Hello (heartbeat interval) — must reply with Identify/Resume
        if op == 10:
            d_data = d if isinstance(d, dict) else {}
            interval_ms = d_data.get("heartbeat_interval", 30000)
            # Send heartbeats at 80% of the server interval to stay safe
            self._heartbeat_interval = interval_ms / 1000.0 * 0.8
            logger.debug(
                "[%s] Hello received, heartbeat_interval=%dms (sending every %.1fs)",
                self._log_tag,
                interval_ms,
                self._heartbeat_interval,
            )
            # Authenticate: send Resume if we have a session, else Identify.
            # Use _create_task which is safe when no event loop is running (tests).
            if self._session_id and self._last_seq is not None:
                self._create_task(self._send_resume())
            else:
                self._create_task(self._send_identify())
            return

        # op 0 = Dispatch
        if op == 0 and t:
            if t == "READY":
                self._handle_ready(d)
            elif t == "RESUMED":
                logger.info("[%s] Session resumed", self._log_tag)
            elif t in (
                    "C2C_MESSAGE_CREATE",
                    "GROUP_AT_MESSAGE_CREATE",
                    "DIRECT_MESSAGE_CREATE",
                    "GUILD_MESSAGE_CREATE",
                    "GUILD_AT_MESSAGE_CREATE",
            ):
                asyncio.create_task(self._on_message(t, d))
            else:
                logger.debug("[%s] Unhandled dispatch: %s", self._log_tag, t)
            return

        # op 11 = Heartbeat ACK
        if op == 11:
            return

        logger.debug("[%s] Unknown op: %s", self._log_tag, op)

    def _handle_ready(self, d: Any) -> None:
        """Handle the READY event — store session_id for resume."""
        if isinstance(d, dict):
            self._session_id = d.get("session_id")
            logger.info("[%s] Ready, session_id=%s", self._log_tag, self._session_id)

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: Any) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(raw)
        except Exception:
            logger.warning("[QQBot] Failed to parse JSON: %r", raw)
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _next_msg_seq(msg_id: str) -> int:
        """Generate a message sequence number in 0..65535 range."""
        time_part = int(time.time()) % 100000000
        rand = int(uuid.uuid4().hex[:4], 16)
        return (time_part ^ rand) % 65536

    # ------------------------------------------------------------------
    # Inbound message handling
    # ------------------------------------------------------------------

    async def handle_message(self, event: MessageEvent) -> None:
        """Cache the last message ID per chat, then delegate to base."""
        if event.message_id and event.source.chat_id:
            self._last_msg_id[event.source.chat_id] = event.message_id
        await super().handle_message(event)

    async def _on_message(self, event_type: str, d: Any) -> None:
        """Process an inbound QQ Bot message event."""
        if not isinstance(d, dict):
            return

        # Extract common fields
        msg_id = str(d.get("id", ""))
        if not msg_id or self._is_duplicate(msg_id):
            logger.debug(
                "[%s] Duplicate or missing message id: %s", self._log_tag, msg_id
            )
            return

        timestamp = str(d.get("timestamp", ""))
        content = str(d.get("content", "")).strip()
        author = d.get("author") if isinstance(d.get("author"), dict) else {}

        # Route by event type
        if event_type == "C2C_MESSAGE_CREATE":
            await self._handle_c2c_message(d, msg_id, content, author, timestamp)
        elif event_type in ("GROUP_AT_MESSAGE_CREATE",):
            await self._handle_group_message(d, msg_id, content, author, timestamp)
        elif event_type in ("GUILD_MESSAGE_CREATE", "GUILD_AT_MESSAGE_CREATE"):
            await self._handle_guild_message(d, msg_id, content, author, timestamp)
        elif event_type == "DIRECT_MESSAGE_CREATE":
            await self._handle_dm_message(d, msg_id, content, author, timestamp)

    async def _handle_c2c_message(
            self,
            d: Dict[str, Any],
            msg_id: str,
            content: str,
            author: Dict[str, Any],
            timestamp: str,
    ) -> None:
        """Handle a C2C (private) message event."""
        user_openid = str(author.get("user_openid", ""))
        if not user_openid:
            return
        if not self._is_dm_allowed(user_openid):
            return

        text = content
        attachments_raw = d.get("attachments")
        logger.info(
            "[%s] C2C message: id=%s content=%r attachments=%s",
            self._log_tag,
            msg_id,
            content[:50] if content else "",
            (
                f"{len(attachments_raw) if isinstance(attachments_raw, list) else 0} items"
                if attachments_raw
                else "None"
            ),
        )
        if attachments_raw and isinstance(attachments_raw, list):
            for _i, _att in enumerate(attachments_raw):
                if isinstance(_att, dict):
                    logger.info(
                        "[%s] attachment[%d]: content_type=%s url=%s filename=%s",
                        self._log_tag,
                        _i,
                        _att.get("content_type", ""),
                        str(_att.get("url", ""))[:80],
                        _att.get("filename", ""),
                    )

        # Process all attachments uniformly (images, voice, files)
        att_result = await self._process_attachments(attachments_raw)
        image_urls = att_result["image_urls"]
        image_media_types = att_result["image_media_types"]
        voice_transcripts = att_result["voice_transcripts"]
        attachment_info = att_result["attachment_info"]

        # Append voice transcripts to the text body
        if voice_transcripts:
            voice_block = "\n".join(voice_transcripts)
            text = (
                (text + "\n\n" + voice_block).strip() if text.strip() else voice_block
            )
        # Append non-media attachment info
        if attachment_info:
            text = (
                (text + "\n\n" + attachment_info).strip()
                if text.strip()
                else attachment_info
            )

        logger.info(
            "[%s] After processing: images=%d, voice=%d",
            self._log_tag,
            len(image_urls),
            len(voice_transcripts),
        )

        if not text.strip() and not image_urls:
            return

        self._chat_type_map[user_openid] = "c2c"
        event = MessageEvent(
            source=self.build_source(
                chat_id=user_openid,
                user_id=user_openid,
                chat_type="dm",
            ),
            text=text,
            message_type=self._detect_message_type(image_urls, image_media_types),
            raw_message=d,
            message_id=msg_id,
            media_urls=image_urls,
            media_types=image_media_types,
            timestamp=self._parse_qq_timestamp(timestamp),
        )
        await self.handle_message(event)

    async def _handle_group_message(
            self,
            d: Dict[str, Any],
            msg_id: str,
            content: str,
            author: Dict[str, Any],
            timestamp: str,
    ) -> None:
        """Handle a group @-message event."""
        group_openid = str(d.get("group_openid", ""))
        if not group_openid:
            return
        if not self._is_group_allowed(
                group_openid, str(author.get("member_openid", ""))
        ):
            return

        # Strip the @bot mention prefix from content
        text = self._strip_at_mention(content)
        att_result = await self._process_attachments(d.get("attachments"))
        image_urls = att_result["image_urls"]
        image_media_types = att_result["image_media_types"]
        voice_transcripts = att_result["voice_transcripts"]
        attachment_info = att_result["attachment_info"]

        # Append voice transcripts
        if voice_transcripts:
            voice_block = "\n".join(voice_transcripts)
            text = (
                (text + "\n\n" + voice_block).strip() if text.strip() else voice_block
            )
        if attachment_info:
            text = (
                (text + "\n\n" + attachment_info).strip()
                if text.strip()
                else attachment_info
            )

        if not text.strip() and not image_urls:
            return

        self._chat_type_map[group_openid] = "group"
        event = MessageEvent(
            source=self.build_source(
                chat_id=group_openid,
                user_id=str(author.get("member_openid", "")),
                chat_type="group",
            ),
            text=text,
            message_type=self._detect_message_type(image_urls, image_media_types),
            raw_message=d,
            message_id=msg_id,
            media_urls=image_urls,
            media_types=image_media_types,
            timestamp=self._parse_qq_timestamp(timestamp),
        )
        await self.handle_message(event)

    async def _handle_guild_message(
            self,
            d: Dict[str, Any],
            msg_id: str,
            content: str,
            author: Dict[str, Any],
            timestamp: str,
    ) -> None:
        """Handle a guild/channel message event."""
        channel_id = str(d.get("channel_id", ""))
        if not channel_id:
            return

        # Apply group_policy ACL — guild channels are group-like contexts.
        # Without this check any member of any guild the bot is in could
        # bypass the configured allowlist.
        guild_id = str(d.get("guild_id", ""))
        author_id = str(author.get("id", ""))
        if not self._is_group_allowed(guild_id or channel_id, author_id):
            logger.debug(
                "[%s] Guild message blocked by ACL: channel=%s user=%s",
                self._log_tag, channel_id, author_id,
            )
            return

        member = d.get("member") if isinstance(d.get("member"), dict) else {}
        nick = str(member.get("nick", "")) or str(author.get("username", ""))

        text = content
        att_result = await self._process_attachments(d.get("attachments"))
        image_urls = att_result["image_urls"]
        image_media_types = att_result["image_media_types"]
        voice_transcripts = att_result["voice_transcripts"]
        attachment_info = att_result["attachment_info"]

        if voice_transcripts:
            voice_block = "\n".join(voice_transcripts)
            text = (
                (text + "\n\n" + voice_block).strip() if text.strip() else voice_block
            )
        if attachment_info:
            text = (
                (text + "\n\n" + attachment_info).strip()
                if text.strip()
                else attachment_info
            )

        if not text.strip() and not image_urls:
            return

        self._chat_type_map[channel_id] = "guild"
        event = MessageEvent(
            source=self.build_source(
                chat_id=channel_id,
                user_id=str(author.get("id", "")),
                user_name=nick or None,
                chat_type="group",
            ),
            text=text,
            message_type=self._detect_message_type(image_urls, image_media_types),
            raw_message=d,
            message_id=msg_id,
            media_urls=image_urls,
            media_types=image_media_types,
            timestamp=self._parse_qq_timestamp(timestamp),
        )
        await self.handle_message(event)

    async def _handle_dm_message(
            self,
            d: Dict[str, Any],
            msg_id: str,
            content: str,
            author: Dict[str, Any],
            timestamp: str,
    ) -> None:
        """Handle a guild DM message event."""
        guild_id = str(d.get("guild_id", ""))
        if not guild_id:
            return

        # Apply dm_policy ACL — guild DMs were previously unauthenticated.
        # Without this check any member of any guild the bot is in could
        # bypass the configured allowlist via direct messages.
        author_id = str(author.get("id", ""))
        if not self._is_dm_allowed(author_id):
            logger.debug(
                "[%s] Guild DM blocked by ACL: guild=%s user=%s",
                self._log_tag, guild_id, author_id,
            )
            return

        text = content
        att_result = await self._process_attachments(d.get("attachments"))
        image_urls = att_result["image_urls"]
        image_media_types = att_result["image_media_types"]
        voice_transcripts = att_result["voice_transcripts"]
        attachment_info = att_result["attachment_info"]

        if voice_transcripts:
            voice_block = "\n".join(voice_transcripts)
            text = (
                (text + "\n\n" + voice_block).strip() if text.strip() else voice_block
            )
        if attachment_info:
            text = (
                (text + "\n\n" + attachment_info).strip()
                if text.strip()
                else attachment_info
            )

        if not text.strip() and not image_urls:
            return

        self._chat_type_map[guild_id] = "dm"
        event = MessageEvent(
            source=self.build_source(
                chat_id=guild_id,
                user_id=str(author.get("id", "")),
                chat_type="dm",
            ),
            text=text,
            message_type=self._detect_message_type(image_urls, image_media_types),
            raw_message=d,
            message_id=msg_id,
            media_urls=image_urls,
            media_types=image_media_types,
            timestamp=self._parse_qq_timestamp(timestamp),
        )
        await self.handle_message(event)

    # ------------------------------------------------------------------
    # Attachment processing
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_message_type(media_urls: list, media_types: list):
        """Determine MessageType from attachment content types."""
        if not media_urls:
            return MessageType.TEXT
        if not media_types:
            return MessageType.PHOTO
        first_type = media_types[0].lower() if media_types else ""
        if "audio" in first_type or "voice" in first_type or "silk" in first_type:
            return MessageType.VOICE
        if "video" in first_type:
            return MessageType.VIDEO
        if "image" in first_type or "photo" in first_type:
            return MessageType.PHOTO
        logger.debug(
            "Unknown media content_type '%s', defaulting to TEXT",
            first_type,
        )
        return MessageType.TEXT

    async def _process_attachments(
            self,
            attachments: Any,
    ) -> Dict[str, Any]:
        """Process inbound attachments (all message types).

        Mirrors OpenClaw's ``processAttachments`` — handles images, voice, and
        other files uniformly.

        Returns a dict with:
        - image_urls: list[str]  — cached local image paths
        - image_media_types: list[str] — MIME types of cached images
        - voice_transcripts: list[str] — STT transcripts for voice messages
        - attachment_info: str — text description of non-image, non-voice attachments
        """
        if not isinstance(attachments, list):
            return {
                "image_urls": [],
                "image_media_types": [],
                "voice_transcripts": [],
                "attachment_info": "",
            }

        image_urls: List[str] = []
        image_media_types: List[str] = []
        voice_transcripts: List[str] = []
        other_attachments: List[str] = []

        for att in attachments:
            if not isinstance(att, dict):
                continue

            ct = str(att.get("content_type", "")).strip().lower()
            url_raw = str(att.get("url", "")).strip()
            filename = str(att.get("filename", ""))
            if url_raw.startswith("//"):
                url = f"https:{url_raw}"
            elif url_raw:
                url = url_raw
            else:
                url = ""
                continue

            logger.debug(
                "[%s] Processing attachment: content_type=%s, url=%s, filename=%s",
                self._log_tag,
                ct,
                url[:80],
                filename,
            )

            if self._is_voice_content_type(ct, filename):
                # Voice: use QQ's asr_refer_text first, then voice_wav_url, then STT.
                asr_refer = (
                    str(att.get("asr_refer_text", "")).strip()
                    if isinstance(att.get("asr_refer_text"), str)
                    else ""
                )
                voice_wav_url = (
                    str(att.get("voice_wav_url", "")).strip()
                    if isinstance(att.get("voice_wav_url"), str)
                    else ""
                )

                transcript = await self._stt_voice_attachment(
                    url,
                    ct,
                    filename,
                    asr_refer_text=asr_refer or None,
                    voice_wav_url=voice_wav_url or None,
                )
                if transcript:
                    voice_transcripts.append(f"[Voice] {transcript}")
                    logger.debug("[%s] Voice transcript: %s", self._log_tag, transcript)
                else:
                    logger.warning("[%s] Voice STT failed for %s", self._log_tag, url[:60])
                    voice_transcripts.append("[Voice] [语音识别失败]")
            elif ct.startswith("image/"):
                # Image: download and cache locally.
                try:
                    cached_path = await self._download_and_cache(url, ct)
                    if cached_path and os.path.isfile(cached_path):
                        image_urls.append(cached_path)
                        image_media_types.append(ct or "image/jpeg")
                    elif cached_path:
                        logger.warning(
                            "[%s] Cached image path does not exist: %s",
                            self._log_tag,
                            cached_path,
                        )
                except Exception as exc:
                    logger.debug("[%s] Failed to cache image: %s", self._log_tag, exc)
            else:
                # Other attachments (video, file, etc.): record as text.
                try:
                    cached_path = await self._download_and_cache(url, ct)
                    if cached_path:
                        other_attachments.append(f"[Attachment: {filename or ct}]")
                except Exception as exc:
                    logger.debug("[%s] Failed to cache attachment: %s", self._log_tag, exc)

        attachment_info = "\n".join(other_attachments) if other_attachments else ""
        return {
            "image_urls": image_urls,
            "image_media_types": image_media_types,
            "voice_transcripts": voice_transcripts,
            "attachment_info": attachment_info,
        }

    async def _download_and_cache(self, url: str, content_type: str) -> Optional[str]:
        """Download a URL and cache it locally."""
        from tools.url_safety import is_safe_url

        if not is_safe_url(url):
            raise ValueError(f"Blocked unsafe URL: {url[:80]}")

        if not self._http_client:
            return None

        try:
            resp = await self._http_client.get(
                url,
                timeout=30.0,
                headers=self._qq_media_headers(),
            )
            resp.raise_for_status()
            data = resp.content
        except Exception as exc:
            logger.debug(
                "[%s] Download failed for %s: %s", self._log_tag, url[:80], exc
            )
            return None

        if content_type.startswith("image/"):
            ext = mimetypes.guess_extension(content_type) or ".jpg"
            return cache_image_from_bytes(data, ext)
        elif content_type == "voice" or content_type.startswith("audio/"):
            # QQ voice messages are typically .amr or .silk format.
            # Convert to .wav using ffmpeg so STT engines can process it.
            return await self._convert_audio_to_wav(data, url)
        else:
            filename = Path(urlparse(url).path).name or "qq_attachment"
            return cache_document_from_bytes(data, filename)

    @staticmethod
    def _is_voice_content_type(content_type: str, filename: str) -> bool:
        """Check if an attachment is a voice/audio message."""
        ct = content_type.strip().lower()
        fn = filename.strip().lower()
        if ct == "voice" or ct.startswith("audio/"):
            return True
        _VOICE_EXTENSIONS = (
            ".silk",
            ".amr",
            ".mp3",
            ".wav",
            ".ogg",
            ".m4a",
            ".aac",
            ".speex",
            ".flac",
        )
        if any(fn.endswith(ext) for ext in _VOICE_EXTENSIONS):
            return True
        return False

    def _qq_media_headers(self) -> Dict[str, str]:
        """Return Authorization headers for QQ multimedia CDN downloads.

        QQ's multimedia URLs (multimedia.nt.qq.com.cn) require the bot's
        access token in an Authorization header, otherwise the download
        returns a non-200 status.
        """
        if self._access_token:
            return {"Authorization": f"QQBot {self._access_token}"}
        return {}

    async def _stt_voice_attachment(
            self,
            url: str,
            content_type: str,
            filename: str,
            *,
            asr_refer_text: Optional[str] = None,
            voice_wav_url: Optional[str] = None,
    ) -> Optional[str]:
        """Download a voice attachment, convert to wav, and transcribe.

        Priority:
        1. QQ's built-in ``asr_refer_text`` (Tencent's own ASR — free, no API call).
        2. Self-hosted STT on ``voice_wav_url`` (pre-converted WAV from QQ, avoids SILK decoding).
        3. Self-hosted STT on the original attachment URL (requires SILK→WAV conversion).

        Returns the transcript text, or None on failure.
        """
        # 1. Use QQ's built-in ASR text if available
        if asr_refer_text:
            logger.debug(
                "[%s] STT: using QQ asr_refer_text: %r", self._log_tag, asr_refer_text[:100]
            )
            return asr_refer_text

        # Determine which URL to download (prefer voice_wav_url — already WAV)
        download_url = url
        is_pre_wav = False
        if voice_wav_url:
            if voice_wav_url.startswith("//"):
                voice_wav_url = f"https:{voice_wav_url}"
            download_url = voice_wav_url
            is_pre_wav = True
            logger.debug("[%s] STT: using voice_wav_url (pre-converted WAV)", self._log_tag)

        from tools.url_safety import is_safe_url
        if not is_safe_url(download_url):
            logger.warning("[QQ] STT blocked unsafe URL: %s", download_url[:80])
            return None

        try:
            # 2. Download audio (QQ CDN requires Authorization header)
            if not self._http_client:
                logger.warning("[%s] STT: no HTTP client", self._log_tag)
                return None

            download_headers = self._qq_media_headers()
            logger.debug(
                "[%s] STT: downloading voice from %s (pre_wav=%s, headers=%s)",
                self._log_tag,
                download_url[:80],
                is_pre_wav,
                bool(download_headers),
            )
            resp = await self._http_client.get(
                download_url,
                timeout=30.0,
                headers=download_headers,
                follow_redirects=True,
            )
            resp.raise_for_status()
            audio_data = resp.content
            logger.debug(
                "[%s] STT: downloaded %d bytes, content_type=%s",
                self._log_tag,
                len(audio_data),
                resp.headers.get("content-type", "unknown"),
            )

            if len(audio_data) < 10:
                logger.warning(
                    "[%s] STT: downloaded data too small (%d bytes), skipping",
                    self._log_tag,
                    len(audio_data),
                )
                return None

            # 3. Convert to wav (skip if we already have a pre-converted WAV)
            if is_pre_wav:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_data)
                    wav_path = tmp.name
                logger.debug(
                    "[%s] STT: using pre-converted WAV directly (%d bytes)",
                    self._log_tag,
                    len(audio_data),
                )
            else:
                logger.debug(
                    "[%s] STT: converting to wav, filename=%r", self._log_tag, filename
                )
                wav_path = await self._convert_audio_to_wav_file(audio_data, filename)
                if not wav_path or not Path(wav_path).exists():
                    logger.warning(
                        "[%s] STT: ffmpeg conversion produced no output", self._log_tag
                    )
                    return None

            # 4. Call STT API
            logger.debug("[%s] STT: calling ASR on %s", self._log_tag, wav_path)
            transcript = await self._call_stt(wav_path)

            # 5. Cleanup temp file
            try:
                os.unlink(wav_path)
            except OSError:
                pass

            if transcript:
                logger.debug("[%s] STT success: %r", self._log_tag, transcript[:100])
            else:
                logger.warning("[%s] STT: ASR returned empty transcript", self._log_tag)
            return transcript
        except (httpx.HTTPStatusError, httpx.TransportError, IOError) as exc:
            logger.warning(
                "[%s] STT failed for voice attachment: %s: %s",
                self._log_tag,
                type(exc).__name__,
                exc,
            )
            return None

    async def _convert_audio_to_wav_file(
            self, audio_data: bytes, filename: str
    ) -> Optional[str]:
        """Convert audio bytes to a temp .wav file using pilk (SILK) or ffmpeg.

        QQ voice messages are typically SILK format which ffmpeg cannot decode.
        Strategy: always try pilk first, fall back to ffmpeg if pilk fails.

        Returns the wav file path, or None on failure.
        """
        import tempfile

        ext = (
            Path(filename).suffix.lower()
            if Path(filename).suffix
            else self._guess_ext_from_data(audio_data)
        )
        logger.info(
            "[%s] STT: audio_data size=%d, ext=%r, first_20_bytes=%r",
            self._log_tag,
            len(audio_data),
            ext,
            audio_data[:20],
        )

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_src:
            tmp_src.write(audio_data)
            src_path = tmp_src.name

        wav_path = src_path.rsplit(".", 1)[0] + ".wav"

        # Try pilk first (handles SILK and many other formats)
        result = await self._convert_silk_to_wav(src_path, wav_path)

        # If pilk failed, try ffmpeg
        if not result:
            result = await self._convert_ffmpeg_to_wav(src_path, wav_path)

        # If ffmpeg also failed, try writing raw PCM as WAV (last resort)
        if not result:
            result = await self._convert_raw_to_wav(audio_data, wav_path)

        # Cleanup source file
        try:
            os.unlink(src_path)
        except OSError:
            pass

        return result

    @staticmethod
    def _guess_ext_from_data(data: bytes) -> str:
        """Guess file extension from magic bytes."""
        if data[:9] == b"#!SILK_V3" or data[:5] == b"#!SILK":
            return ".silk"
        if data[:2] == b"\x02!":
            return ".silk"
        if data[:4] == b"RIFF":
            return ".wav"
        if data[:4] == b"fLaC":
            return ".flac"
        if data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            return ".mp3"
        if data[:4] == b"\x30\x26\xb2\x75" or data[:4] == b"\x4f\x67\x67\x53":
            return ".ogg"
        if data[:4] == b"\x00\x00\x00\x20" or data[:4] == b"\x00\x00\x00\x1c":
            return ".amr"
        # Default to .amr for unknown (QQ's most common voice format)
        return ".amr"

    @staticmethod
    def _looks_like_silk(data: bytes) -> bool:
        """Check if bytes look like a SILK audio file."""
        return data[:4] == b"#!SILK" or data[:2] == b"\x02!" or data[:9] == b"#!SILK_V3"

    async def _convert_silk_to_wav(self, src_path: str, wav_path: str) -> Optional[str]:
        """Convert audio file to WAV using the pilk library.

        Tries the file as-is first, then as .silk if the extension differs.
        pilk can handle SILK files with various headers (or no header).
        """
        try:
            import pilk
        except ImportError:
            logger.warning(
                "[%s] pilk not installed — cannot decode SILK audio. Run: pip install pilk",
                self._log_tag,
            )
            return None

        # Try converting the file as-is
        try:
            pilk.silk_to_wav(src_path, wav_path, rate=16000)
            if Path(wav_path).exists() and Path(wav_path).stat().st_size > 44:
                logger.debug(
                    "[%s] pilk converted %s to wav (%d bytes)",
                    self._log_tag,
                    Path(src_path).name,
                    Path(wav_path).stat().st_size,
                )
                return wav_path
        except Exception as exc:
            logger.debug("[%s] pilk direct conversion failed: %s", self._log_tag, exc)

        # Try renaming to .silk and converting (pilk checks the extension)
        silk_path = src_path.rsplit(".", 1)[0] + ".silk"
        try:
            import shutil

            shutil.copy2(src_path, silk_path)
            pilk.silk_to_wav(silk_path, wav_path, rate=16000)
            if Path(wav_path).exists() and Path(wav_path).stat().st_size > 44:
                logger.debug(
                    "[%s] pilk converted %s (as .silk) to wav (%d bytes)",
                    self._log_tag,
                    Path(src_path).name,
                    Path(wav_path).stat().st_size,
                )
                return wav_path
        except Exception as exc:
            logger.debug("[%s] pilk .silk conversion failed: %s", self._log_tag, exc)
        finally:
            try:
                os.unlink(silk_path)
            except OSError:
                pass

        return None

    async def _convert_raw_to_wav(self, audio_data: bytes, wav_path: str) -> Optional[str]:
        """Last resort: try writing audio data as raw PCM 16-bit mono 16kHz WAV.

        This will produce garbage if the data isn't raw PCM, but at least
        the ASR engine won't crash — it'll just return empty.
        """
        try:
            import wave

            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data)
            return wav_path
        except Exception as exc:
            logger.debug("[%s] raw PCM fallback failed: %s", self._log_tag, exc)
            return None

    async def _convert_ffmpeg_to_wav(self, src_path: str, wav_path: str) -> Optional[str]:
        """Convert audio file to WAV using ffmpeg."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",
                "-i",
                src_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                wav_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.wait(), timeout=30)
            if proc.returncode != 0:
                stderr = await proc.stderr.read() if proc.stderr else b""
                logger.warning(
                    "[%s] ffmpeg failed for %s: %s",
                    self._log_tag,
                    Path(src_path).name,
                    stderr[:200].decode(errors="replace"),
                )
                return None
        except (asyncio.TimeoutError, FileNotFoundError) as exc:
            logger.warning("[%s] ffmpeg conversion error: %s", self._log_tag, exc)
            return None

        if not Path(wav_path).exists() or Path(wav_path).stat().st_size <= 44:
            logger.warning(
                "[%s] ffmpeg produced no/small output for %s",
                self._log_tag,
                Path(src_path).name,
            )
            return None
        logger.debug(
            "[%s] ffmpeg converted %s to wav (%d bytes)",
            self._log_tag,
            Path(src_path).name,
            Path(wav_path).stat().st_size,
        )
        return wav_path

    def _resolve_stt_config(self) -> Optional[Dict[str, str]]:
        """Resolve STT backend configuration from config/environment.

        Priority:
        1. Plugin-specific: ``channels.qqbot.stt`` in config.yaml → ``self.config.extra["stt"]``
        2. QQ-specific env vars: ``QQ_STT_API_KEY`` / ``QQ_STT_BASE_URL`` / ``QQ_STT_MODEL``
        3. Return None if nothing is configured (STT will be skipped, QQ built-in ASR still works).
        """
        extra = self.config.extra or {}

        # 1. Plugin-specific STT config (matches OpenClaw's channels.qqbot.stt)
        stt_cfg = extra.get("stt")
        if isinstance(stt_cfg, dict) and stt_cfg.get("enabled") is not False:
            base_url = stt_cfg.get("baseUrl") or stt_cfg.get("base_url", "")
            api_key = stt_cfg.get("apiKey") or stt_cfg.get("api_key", "")
            model = stt_cfg.get("model", "")
            if base_url and api_key:
                return {
                    "base_url": base_url.rstrip("/"),
                    "api_key": api_key,
                    "model": model or "whisper-1",
                }
            # Provider-only config: just model name, use default provider
            if api_key:
                provider = stt_cfg.get("provider", "zai")
                # Map provider to base URL
                _PROVIDER_BASE_URLS = {
                    "zai": "https://open.bigmodel.cn/api/coding/paas/v4",
                    "openai": "https://api.openai.com/v1",
                    "glm": "https://open.bigmodel.cn/api/coding/paas/v4",
                }
                base_url = _PROVIDER_BASE_URLS.get(provider, "")
                if base_url:
                    return {
                        "base_url": base_url,
                        "api_key": api_key,
                        "model": model
                                 or ("glm-asr" if provider in ("zai", "glm") else "whisper-1"),
                    }

        # 2. QQ-specific env vars (set by `hermes setup gateway` / `hermes gateway`)
        qq_stt_key = os.getenv("QQ_STT_API_KEY", "")
        if qq_stt_key:
            base_url = os.getenv(
                "QQ_STT_BASE_URL",
                "https://open.bigmodel.cn/api/coding/paas/v4",
            )
            model = os.getenv("QQ_STT_MODEL", "glm-asr")
            return {
                "base_url": base_url.rstrip("/"),
                "api_key": qq_stt_key,
                "model": model,
            }

        return None

    async def _call_stt(self, wav_path: str) -> Optional[str]:
        """Call an OpenAI-compatible STT API to transcribe a wav file.

        Uses the provider configured in ``channels.qqbot.stt`` config,
        falling back to QQ's built-in ``asr_refer_text`` if not configured.
        Returns None if STT is not configured or the call fails.
        """
        stt_cfg = self._resolve_stt_config()
        if not stt_cfg:
            logger.warning(
                "[%s] STT not configured (no stt config or QQ_STT_API_KEY)",
                self._log_tag,
            )
            return None

        base_url = stt_cfg["base_url"]
        api_key = stt_cfg["api_key"]
        model = stt_cfg["model"]

        try:
            with open(wav_path, "rb") as f:
                resp = await self._http_client.post(
                    f"{base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (Path(wav_path).name, f, "audio/wav")},
                    data={"model": model},
                    timeout=30.0,
                )
            resp.raise_for_status()
            result = resp.json()
            # Zhipu/GLM format: {"choices": [{"message": {"content": "transcript text"}}]}
            choices = result.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                if content.strip():
                    return content.strip()
            # OpenAI/Whisper format: {"text": "transcript text"}
            text = result.get("text", "")
            if text.strip():
                return text.strip()
            return None
        except (httpx.HTTPStatusError, IOError) as exc:
            logger.warning(
                "[%s] STT API call failed (model=%s, base=%s): %s",
                self._log_tag,
                model,
                base_url[:50],
                exc,
            )
            return None

    async def _convert_audio_to_wav(
            self, audio_data: bytes, source_url: str
    ) -> Optional[str]:
        """Convert audio bytes to .wav using pilk (SILK) or ffmpeg, caching the result."""
        import tempfile

        # Determine source format from magic bytes or URL
        ext = (
            Path(urlparse(source_url).path).suffix.lower()
            if urlparse(source_url).path
            else ""
        )
        if not ext or ext not in (
                ".silk",
                ".amr",
                ".mp3",
                ".wav",
                ".ogg",
                ".m4a",
                ".aac",
                ".flac",
        ):
            ext = self._guess_ext_from_data(audio_data)

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_src:
            tmp_src.write(audio_data)
            src_path = tmp_src.name

        wav_path = src_path.rsplit(".", 1)[0] + ".wav"
        try:
            is_silk = ext == ".silk" or self._looks_like_silk(audio_data)
            if is_silk:
                result = await self._convert_silk_to_wav(src_path, wav_path)
            else:
                result = await self._convert_ffmpeg_to_wav(src_path, wav_path)

            if not result:
                logger.warning(
                    "[%s] audio conversion failed for %s (format=%s)",
                    self._log_tag,
                    source_url[:60],
                    ext,
                )
                return cache_document_from_bytes(audio_data, f"qq_voice{ext}")
        except Exception:
            return cache_document_from_bytes(audio_data, f"qq_voice{ext}")
        finally:
            try:
                os.unlink(src_path)
            except OSError:
                pass

        # Verify output and cache
        try:
            wav_data = Path(wav_path).read_bytes()
            os.unlink(wav_path)
            return cache_document_from_bytes(wav_data, "qq_voice.wav")
        except Exception as exc:
            logger.debug("[%s] Failed to read converted wav: %s", self._log_tag, exc)
            return None

    # ------------------------------------------------------------------
    # Outbound messaging — REST API
    # ------------------------------------------------------------------

    async def _api_request(
            self,
            method: str,
            path: str,
            body: Optional[Dict[str, Any]] = None,
            timeout: float = DEFAULT_API_TIMEOUT,
    ) -> Dict[str, Any]:
        """Make an authenticated REST API request to QQ Bot API."""
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized — not connected?")

        token = await self._ensure_token()
        headers = {
            "Authorization": f"QQBot {token}",
            "Content-Type": "application/json",
            "User-Agent": build_user_agent(),
        }

        try:
            resp = await self._http_client.request(
                method,
                f"{API_BASE}{path}",
                headers=headers,
                json=body,
                timeout=timeout,
            )
            data = resp.json()
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"QQ Bot API error [{resp.status_code}] {path}: "
                    f"{data.get('message', data)}"
                )
            return data
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"QQ Bot API timeout [{path}]: {exc}") from exc

    async def _upload_media(
            self,
            target_type: str,
            target_id: str,
            file_type: int,
            url: Optional[str] = None,
            file_data: Optional[str] = None,
            srv_send_msg: bool = False,
            file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload media and return file_info."""
        path = (
            f"/v2/users/{target_id}/files"
            if target_type == "c2c"
            else f"/v2/groups/{target_id}/files"
        )

        body: Dict[str, Any] = {
            "file_type": file_type,
            "srv_send_msg": srv_send_msg,
        }
        if url:
            body["url"] = url
        elif file_data:
            body["file_data"] = file_data
        if file_type == MEDIA_TYPE_FILE and file_name:
            body["file_name"] = file_name

        # Retry transient upload failures
        for attempt in range(3):
            try:
                return await self._api_request(
                    "POST", path, body, timeout=FILE_UPLOAD_TIMEOUT
                )
            except RuntimeError as exc:
                err_msg = str(exc)
                if any(
                        kw in err_msg
                        for kw in ("400", "401", "Invalid", "timeout", "Timeout")
                ):
                    raise
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                else:
                    raise

    # Maximum time (seconds) to wait for reconnection before giving up on send.
    _RECONNECT_WAIT_SECONDS = 15.0
    # How often (seconds) to poll is_connected while waiting.
    _RECONNECT_POLL_INTERVAL = 0.5

    async def _wait_for_reconnection(self) -> bool:
        """Wait for the WebSocket listener to reconnect.

        The listener loop (_listen_loop) auto-reconnects on disconnect, but
        there is a race window where send() is called right after a disconnect
        and before the reconnect completes.  This method polls is_connected
        for up to _RECONNECT_WAIT_SECONDS.

        Returns True if reconnected, False if still disconnected.
        """
        logger.info("[%s] Not connected — waiting for reconnection (up to %.0fs)",
                    self._log_tag, self._RECONNECT_WAIT_SECONDS)
        waited = 0.0
        while waited < self._RECONNECT_WAIT_SECONDS:
            await asyncio.sleep(self._RECONNECT_POLL_INTERVAL)
            waited += self._RECONNECT_POLL_INTERVAL
            if self.is_connected:
                logger.info("[%s] Reconnected after %.1fs", self._log_tag, waited)
                return True
        logger.warning("[%s] Still not connected after %.0fs", self._log_tag, self._RECONNECT_WAIT_SECONDS)
        return False

    async def send(
            self,
            chat_id: str,
            content: str,
            reply_to: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text or markdown message to a QQ user or group.

        Applies format_message(), splits long messages via truncate_message(),
        and retries transient failures with exponential backoff.
        """
        del metadata

        if not self.is_connected:
            if not await self._wait_for_reconnection():
                return SendResult(success=False, error="Not connected", retryable=True)

        if not content or not content.strip():
            return SendResult(success=True)

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)

        last_result = SendResult(success=False, error="No chunks")
        for chunk in chunks:
            last_result = await self._send_chunk(chat_id, chunk, reply_to)
            if not last_result.success:
                return last_result
            # Only reply_to the first chunk
            reply_to = None
        return last_result

    async def _send_chunk(
            self,
            chat_id: str,
            content: str,
            reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send a single chunk with retry + exponential backoff."""
        last_exc: Optional[Exception] = None
        chat_type = self._guess_chat_type(chat_id)

        for attempt in range(3):
            try:
                if chat_type == "c2c":
                    return await self._send_c2c_text(chat_id, content, reply_to)
                elif chat_type == "group":
                    return await self._send_group_text(chat_id, content, reply_to)
                elif chat_type == "guild":
                    return await self._send_guild_text(chat_id, content, reply_to)
                else:
                    return SendResult(
                        success=False, error=f"Unknown chat type for {chat_id}"
                    )
            except Exception as exc:
                last_exc = exc
                err = str(exc).lower()
                # Permanent errors — don't retry
                if any(
                        k in err
                        for k in ("invalid", "forbidden", "not found", "bad request")
                ):
                    break
                # Transient — back off and retry
                if attempt < 2:
                    delay = 1.0 * (2 ** attempt)
                    logger.warning(
                        "[%s] send retry %d/3 after %.1fs: %s",
                        self._log_tag,
                        attempt + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        error_msg = str(last_exc) if last_exc else "Unknown error"
        logger.error("[%s] Send failed: %s", self._log_tag, error_msg)
        retryable = not any(
            k in error_msg.lower() for k in ("invalid", "forbidden", "not found")
        )
        return SendResult(success=False, error=error_msg, retryable=retryable)

    async def _send_c2c_text(
            self, openid: str, content: str, reply_to: Optional[str] = None
    ) -> SendResult:
        """Send text to a C2C user via REST API."""
        self._next_msg_seq(reply_to or openid)
        body = self._build_text_body(content, reply_to)
        if reply_to:
            body["msg_id"] = reply_to

        data = await self._api_request("POST", f"/v2/users/{openid}/messages", body)
        msg_id = str(data.get("id", uuid.uuid4().hex[:12]))
        return SendResult(success=True, message_id=msg_id, raw_response=data)

    async def _send_group_text(
            self, group_openid: str, content: str, reply_to: Optional[str] = None
    ) -> SendResult:
        """Send text to a group via REST API."""
        self._next_msg_seq(reply_to or group_openid)
        body = self._build_text_body(content, reply_to)
        if reply_to:
            body["msg_id"] = reply_to

        data = await self._api_request(
            "POST", f"/v2/groups/{group_openid}/messages", body
        )
        msg_id = str(data.get("id", uuid.uuid4().hex[:12]))
        return SendResult(success=True, message_id=msg_id, raw_response=data)

    async def _send_guild_text(
            self, channel_id: str, content: str, reply_to: Optional[str] = None
    ) -> SendResult:
        """Send text to a guild channel via REST API."""
        body: Dict[str, Any] = {"content": content[: self.MAX_MESSAGE_LENGTH]}
        if reply_to:
            body["msg_id"] = reply_to

        data = await self._api_request("POST", f"/channels/{channel_id}/messages", body)
        msg_id = str(data.get("id", uuid.uuid4().hex[:12]))
        return SendResult(success=True, message_id=msg_id, raw_response=data)

    def _build_text_body(
            self, content: str, reply_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build the message body for C2C/group text sending."""
        msg_seq = self._next_msg_seq(reply_to or "default")

        if self._markdown_support:
            body: Dict[str, Any] = {
                "markdown": {"content": content[: self.MAX_MESSAGE_LENGTH]},
                "msg_type": MSG_TYPE_MARKDOWN,
                "msg_seq": msg_seq,
            }
        else:
            body = {
                "content": content[: self.MAX_MESSAGE_LENGTH],
                "msg_type": MSG_TYPE_TEXT,
                "msg_seq": msg_seq,
            }

        if reply_to:
            # For non-markdown mode, add message_reference
            if not self._markdown_support:
                body["message_reference"] = {"message_id": reply_to}

        return body

    # ------------------------------------------------------------------
    # Native media sending
    # ------------------------------------------------------------------

    async def send_image(
            self,
            chat_id: str,
            image_url: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image natively via QQ Bot API upload."""
        del metadata

        result = await self._send_media(
            chat_id, image_url, MEDIA_TYPE_IMAGE, "image", caption, reply_to
        )
        if result.success or not self._is_url(image_url):
            return result

        # Fallback to text URL
        logger.warning(
            "[%s] Image send failed, falling back to text: %s",
            self._log_tag,
            result.error,
        )
        fallback = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id=chat_id, content=fallback, reply_to=reply_to)

    async def send_image_file(
            self,
            chat_id: str,
            image_path: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            **kwargs,
    ) -> SendResult:
        """Send a local image file natively."""
        del kwargs
        return await self._send_media(
            chat_id, image_path, MEDIA_TYPE_IMAGE, "image", caption, reply_to
        )

    async def send_voice(
            self,
            chat_id: str,
            audio_path: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            **kwargs,
    ) -> SendResult:
        """Send a voice message natively."""
        del kwargs
        return await self._send_media(
            chat_id, audio_path, MEDIA_TYPE_VOICE, "voice", caption, reply_to
        )

    async def send_video(
            self,
            chat_id: str,
            video_path: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            **kwargs,
    ) -> SendResult:
        """Send a video natively."""
        del kwargs
        return await self._send_media(
            chat_id, video_path, MEDIA_TYPE_VIDEO, "video", caption, reply_to
        )

    async def send_document(
            self,
            chat_id: str,
            file_path: str,
            caption: Optional[str] = None,
            file_name: Optional[str] = None,
            reply_to: Optional[str] = None,
            **kwargs,
    ) -> SendResult:
        """Send a file/document natively."""
        del kwargs
        return await self._send_media(
            chat_id,
            file_path,
            MEDIA_TYPE_FILE,
            "file",
            caption,
            reply_to,
            file_name=file_name,
        )

    async def _send_media(
            self,
            chat_id: str,
            media_source: str,
            file_type: int,
            kind: str,
            caption: Optional[str] = None,
            reply_to: Optional[str] = None,
            file_name: Optional[str] = None,
    ) -> SendResult:
        """Upload media and send as a native message."""
        if not self.is_connected:
            if not await self._wait_for_reconnection():
                return SendResult(success=False, error="Not connected", retryable=True)

        try:
            # Resolve media source
            data, content_type, resolved_name = await self._load_media(
                media_source, file_name
            )

            # Route
            chat_type = self._guess_chat_type(chat_id)

            if chat_type == "guild":
                # Guild channels don't support native media upload in the same way
                # Send as URL fallback
                return SendResult(
                    success=False, error="Guild media send not supported via this path"
                )

            # Upload
            upload = await self._upload_media(
                chat_type,
                chat_id,
                file_type,
                file_data=data if not self._is_url(media_source) else None,
                url=media_source if self._is_url(media_source) else None,
                srv_send_msg=False,
                file_name=resolved_name if file_type == MEDIA_TYPE_FILE else None,
            )

            file_info = upload.get("file_info")
            if not file_info:
                return SendResult(
                    success=False, error=f"Upload returned no file_info: {upload}"
                )

            # Send media message
            msg_seq = self._next_msg_seq(chat_id)
            body: Dict[str, Any] = {
                "msg_type": MSG_TYPE_MEDIA,
                "media": {"file_info": file_info},
                "msg_seq": msg_seq,
            }
            if caption:
                body["content"] = caption[: self.MAX_MESSAGE_LENGTH]
            if reply_to:
                body["msg_id"] = reply_to

            send_data = await self._api_request(
                "POST",
                (
                    f"/v2/users/{chat_id}/messages"
                    if chat_type == "c2c"
                    else f"/v2/groups/{chat_id}/messages"
                ),
                body,
            )
            return SendResult(
                success=True,
                message_id=str(send_data.get("id", uuid.uuid4().hex[:12])),
                raw_response=send_data,
            )
        except Exception as exc:
            logger.error("[%s] Media send failed: %s", self._log_tag, exc)
            return SendResult(success=False, error=str(exc))

    async def _load_media(
            self, source: str, file_name: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """Load media from URL or local path. Returns (base64_or_url, content_type, filename)."""
        source = str(source).strip()
        if not source:
            raise ValueError("Media source is required")

        parsed = urlparse(source)
        if parsed.scheme in ("http", "https"):
            # For URLs, pass through directly to the upload API
            content_type = mimetypes.guess_type(source)[0] or "application/octet-stream"
            resolved_name = file_name or Path(parsed.path).name or "media"
            return source, content_type, resolved_name

        # Local file — encode as raw base64 for QQ Bot API file_data field.
        # The QQ API expects plain base64, NOT a data URI.
        local_path = Path(source).expanduser()
        if not local_path.is_absolute():
            local_path = (Path.cwd() / local_path).resolve()

        if not local_path.exists() or not local_path.is_file():
            # Guard against placeholder paths like "<path>" that the LLM
            # sometimes emits instead of real file paths.
            if source.startswith("<") or len(source) < 3:
                raise ValueError(
                    f"Invalid media source (looks like a placeholder): {source!r}"
                )
            raise FileNotFoundError(f"Media file not found: {local_path}")

        raw = local_path.read_bytes()
        resolved_name = file_name or local_path.name
        content_type = (
                mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
        )
        b64 = base64.b64encode(raw).decode("ascii")
        return b64, content_type, resolved_name

    # ------------------------------------------------------------------
    # Typing indicator
    # ------------------------------------------------------------------

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send an input notify to a C2C user (only supported for C2C).

        Debounced to one request per ~50s (the API sets a 60s indicator).
        The QQ API requires the originating message ID — retrieved from
        ``_last_msg_id`` which is populated by ``_on_message``.
        """
        if not self.is_connected:
            return

        chat_type = self._guess_chat_type(chat_id)
        if chat_type != "c2c":
            return

        msg_id = self._last_msg_id.get(chat_id)
        if not msg_id:
            return

        # Debounce — skip if we sent recently
        now = time.time()
        last_sent = self._typing_sent_at.get(chat_id, 0.0)
        if now - last_sent < self._TYPING_DEBOUNCE_SECONDS:
            return

        try:
            msg_seq = self._next_msg_seq(chat_id)
            body = {
                "msg_type": MSG_TYPE_INPUT_NOTIFY,
                "msg_id": msg_id,
                "input_notify": {
                    "input_type": 1,
                    "input_second": self._TYPING_INPUT_SECONDS,
                },
                "msg_seq": msg_seq,
            }
            await self._api_request("POST", f"/v2/users/{chat_id}/messages", body)
            self._typing_sent_at[chat_id] = now
        except Exception as exc:
            logger.debug("[%s] send_typing failed: %s", self._log_tag, exc)

    # ------------------------------------------------------------------
    # Format
    # ------------------------------------------------------------------

    def format_message(self, content: str) -> str:
        """Format message for QQ.

        When markdown_support is enabled, content is sent as-is (QQ renders it).
        When disabled, strip markdown via shared helper (same as BlueBubbles/SMS).
        """
        if self._markdown_support:
            return content
        return strip_markdown(content)

    # ------------------------------------------------------------------
    # Chat info
    # ------------------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return chat info based on chat type heuristics."""
        chat_type = self._guess_chat_type(chat_id)
        return {
            "name": chat_id,
            "type": "group" if chat_type in ("group", "guild") else "dm",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_url(source: str) -> bool:
        return urlparse(str(source)).scheme in ("http", "https")

    def _guess_chat_type(self, chat_id: str) -> str:
        """Determine chat type from stored inbound metadata, fallback to 'c2c'."""
        if chat_id in self._chat_type_map:
            return self._chat_type_map[chat_id]
        return "c2c"

    @staticmethod
    def _strip_at_mention(content: str) -> str:
        """Strip the @bot mention prefix from group message content."""
        # QQ group @-messages may have the bot's QQ/ID as prefix
        import re

        stripped = re.sub(r"^@\S+\s*", "", content.strip())
        return stripped

    def _is_dm_allowed(self, user_id: str) -> bool:
        if self._dm_policy == "disabled":
            return False
        if self._dm_policy == "allowlist":
            return self._entry_matches(self._allow_from, user_id)
        return True

    def _is_group_allowed(self, group_id: str, user_id: str) -> bool:
        if self._group_policy == "disabled":
            return False
        if self._group_policy == "allowlist":
            return self._entry_matches(self._group_allow_from, group_id)
        return True

    @staticmethod
    def _entry_matches(entries: List[str], target: str) -> bool:
        normalized_target = str(target).strip().lower()
        for entry in entries:
            normalized = str(entry).strip().lower()
            if normalized == "*" or normalized == normalized_target:
                return True
        return False

    def _parse_qq_timestamp(self, raw: str) -> datetime:
        """Parse QQ API timestamp (ISO 8601 string or integer ms).

        The QQ API changed from integer milliseconds to ISO 8601 strings.
        This handles both formats gracefully.
        """
        if not raw:
            return datetime.now(tz=timezone.utc)
        try:
            return datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            pass
        try:
            return datetime.fromtimestamp(int(raw) / 1000, tz=timezone.utc)
        except (ValueError, TypeError):
            pass
        return datetime.now(tz=timezone.utc)

    def _is_duplicate(self, msg_id: str) -> bool:
        now = time.time()
        if len(self._seen_messages) > DEDUP_MAX_SIZE:
            cutoff = now - DEDUP_WINDOW_SECONDS
            self._seen_messages = {
                key: ts for key, ts in self._seen_messages.items() if ts > cutoff
            }
        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False
