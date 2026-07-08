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
          dm_policy: "pairing"             # open | allowlist | disabled | pairing
          allow_from: ["openid_1"]
          group_policy: "disabled"         # open | allowlist | disabled
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
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
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

from gateway.config import (
    Platform,
    PlatformConfig,
)
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
from gateway.platforms.qqbot.chunked_upload import (
    ChunkedUploader,
    UploadDailyLimitExceededError,
    UploadFileTooLargeError,
)
from gateway.platforms.qqbot.keyboards import (
    ApprovalRequest,
    InlineKeyboard,
    InteractionEvent,
    build_approval_keyboard,
    build_update_prompt_keyboard,
    parse_approval_button_data,
    parse_interaction_event,
    parse_update_prompt_button_data,
)
from gateway.platforms.qqbot.group_activation import (
    detect_mentioned,
    resolve_require_mention,
)
from gateway.platforms.qqbot.group_context import (
    GroupContextBuffer,
    summarize_attachments,
)
from gateway.platforms.qqbot.streaming import (
    DEFAULT_SESSION_TTL_SECONDS,
    MAX_STREAM_CONTENT_LEN,
    StreamManager,
    StreamSession,
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

    # QQ Bot API does not support editing regular sent messages, but the
    # C2C stream_messages endpoint (used for streaming replies to private
    # chats) DOES accept in-place updates for the same ``stream_msg_id``.
    #
    # We advertise ``SUPPORTS_MESSAGE_EDITING = True`` so the gateway
    # stream consumer is willing to attach for C2C sessions.  The actual
    # gating happens inside :meth:`edit_message`: if no C2C stream
    # session is registered for the given ``message_id`` (which is the
    # case for group / guild replies), we return
    # ``SendResult(success=False, error="stream session not found …")``
    # and the stream consumer falls back to a fresh ``send()`` — so
    # non-C2C paths still get plain "send new message" semantics.
    SUPPORTS_MESSAGE_EDITING = True

    # Streaming replies to QQ AI-assistant surfaces stay in a "generating"
    # visual state until the caller sends ``input_state=10 (DONE)``.  The
    # stream consumer honours this flag by routing a final edit through
    # even when content is unchanged, ensuring the UI transitions out of
    # the streaming indicator.  Group / guild replies fall back to plain
    # sends so this only affects C2C, where it's required.
    REQUIRES_EDIT_FINALIZE = True

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

    def _mark_transport_disconnected(self) -> None:
        """Mark QQ WS down without stopping the reconnect loop.

        BasePlatformAdapter uses _running for both process lifecycle and
        connection status. QQBot needs to keep the listener task alive across
        transient transport drops so it can continue reconnect attempts after a
        short-lived gateway or network failure.
        """
        if self.has_fatal_error:
            return
        self._write_runtime_status_safe(
            "disconnected",
            platform_state="disconnected",
            error_code=None,
            error_message=None,
        )

    @property
    def is_connected(self) -> bool:
        """Return True only when the QQ WebSocket transport is usable."""
        return bool(self._running and self._ws and not self._ws.closed)

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.QQBOT)

        extra = config.extra or {}
        self._app_id = str(extra.get("app_id") or os.getenv("QQ_APP_ID", "")).strip()
        self._client_secret = str(
            extra.get("client_secret") or os.getenv("QQ_CLIENT_SECRET", "")
        ).strip()
        self._markdown_support = bool(extra.get("markdown_support", True))

        # Auth/ACL policies
        self._dm_policy = str(extra.get("dm_policy", "pairing")).strip().lower()
        self._allow_from = _coerce_list(
            extra.get("allow_from") or extra.get("allowFrom")
        )
        # Group ACL policy — mirrors Feishu (open | allowlist | disabled).
        # Default is "disabled" (safe-by-default: no group receives replies
        # until explicitly enabled). QQ does not implement a group "pairing"
        # handshake, so pairing is intentionally not offered for groups
        # (unlike dm_policy, where QQ C2C supports pairing).
        raw_group_policy = str(extra.get("group_policy", "disabled")).strip().lower()
        if raw_group_policy not in {"open", "allowlist", "disabled"}:
            logger.warning(
                "[%s] Unknown group_policy=%r; falling back to 'disabled'. "
                "Supported: open | allowlist | disabled.",
                self._log_tag, raw_group_policy,
            )
            raw_group_policy = "disabled"
        self._group_policy = raw_group_policy
        self._group_allow_from = _coerce_list(
            extra.get("group_allow_from") or extra.get("groupAllowFrom")
        )

        # Group activation mode (always vs mention). Default: mention — the bot
        # only replies when @-ed (aligns with openclaw requireMention default).
        # always mode (require_mention=False) lets any group message trigger a
        # reply; it also needs the "receive all group messages" permission on
        # the QQ platform to actually receive GROUP_MESSAGE_CREATE events.
        self._group_require_mention = bool(extra.get("group_require_mention", True))
        # Per-group override: groups.{group_openid}.require_mention (group > global).
        self._group_mode_overrides: Dict[str, bool] = {}
        _groups_cfg = extra.get("groups")
        if isinstance(_groups_cfg, dict):
            for _gid, _gcfg in _groups_cfg.items():
                if isinstance(_gcfg, dict) and "require_mention" in _gcfg:
                    self._group_mode_overrides[str(_gid)] = bool(
                        _gcfg["require_mention"]
                    )
        # Reserved for a future runtime mode-switch command (solution 2.2.3):
        # a /-command would write here and persist via cli.save_config_value.
        # Empty in this release.
        self._group_mode_runtime_overrides: Dict[str, bool] = {}

        # Group context buffer (mention mode): non-@ messages are remembered per
        # group and injected as CONTEXT ONLY on the next @-reply. Default 20;
        # group_history_limit <= 0 disables buffering.
        self._group_history_limit = int(extra.get("group_history_limit", 20))
        self._group_context = GroupContextBuffer(limit=self._group_history_limit)

        # ── C2C streaming reply ─────────────────────────────────────────
        # QQ's official ``stream_messages`` endpoint is C2C-only (see
        # openclaw ``shouldUseOfficialC2cStream`` which hard-rejects
        # non-C2C targets).  We enable it by default because failures
        # gracefully fall back to fresh sends via the stream consumer's
        # own error-handling; a config knob is exposed for operators who
        # want to force the legacy path (e.g. during incident triage).
        self._streaming_enabled = bool(extra.get("streaming_enabled", True))
        self._stream_session_ttl = float(
            extra.get("streaming_session_ttl_seconds", DEFAULT_SESSION_TTL_SECONDS)
        )
        self._stream_manager = StreamManager(ttl_seconds=self._stream_session_ttl)

        # Deferred streaming sessions for group / guild chats.
        #
        # QQ has no editable-message API for group or guild targets, and
        # its ``stream_messages`` endpoint is C2C-only.  To avoid the
        # "two messages per turn" symptom described in
        # ``send()``/``edit_message()``, we intercept the streaming
        # first-send for group/guild chats: we don't hit the QQ API on
        # ``send()``, we hand out a sentinel ``message_id``, and we
        # buffer subsequent ``edit_message()`` updates locally until the
        # stream consumer calls with ``finalize=True`` — at that point
        # we deliver ONE complete reply via the normal legacy path.
        #
        # After the first successful ``finalize=True`` edit, the session
        # entry is **kept but flagged as finalized** and the delivered
        # ``SendResult`` is memoised on it.  This is intentional: the
        # ``stream_consumer`` occasionally issues a second finalize edit
        # after the mid-stream tick already finalized (adapters with
        # ``REQUIRES_EDIT_FINALIZE=True`` skip the "final edit already
        # delivered content" fast path — see the ``got_done`` branch in
        # ``gateway/stream_consumer.py``).  Without idempotence that
        # second call would re-enter the "session missing → recover via
        # fresh send" branch below and post the reply a second time
        # (the "two identical messages per group turn" bug this comment
        # is guarding against).  We evict finalized entries only when
        # they exceed the LRU cap to bound memory.
        #
        # ``sentinel_id -> {
        #     "chat_id",
        #     "chat_type",
        #     "reply_to",
        #     "finalized": bool,          # True after first finalize
        #     "finalized_result": SendResult | None,  # memoised delivery
        # }``
        self._group_defer_sessions: Dict[str, Dict[str, Any]] = {}
        # LRU cap on finalized-session bookkeeping — a single QQ bot
        # rarely holds more than a handful of concurrent group replies,
        # so 1024 is comfortably above any realistic burst but low
        # enough to bound memory even if some pathological consumer
        # never issues finalize edits.
        self._group_defer_sessions_cap: int = 1024

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

        # Inline-keyboard interaction routing. The callback (if set) is invoked
        # for every INTERACTION_CREATE event after the adapter has already
        # ACKed it. Callers (gateway wiring for approvals / update prompts)
        # register via set_interaction_callback().
        self._interaction_callback: Optional[
            Callable[[InteractionEvent], Awaitable[None]]
        ] = None

        # Default interaction dispatcher: routes approval-button clicks to
        # tools.approval.resolve_gateway_approval() and update-prompt clicks
        # to ~/.hermes/.update_response. Set here so the cross-adapter gateway
        # contract (send_exec_approval / send_update_prompt) works out of the
        # box; callers can override with set_interaction_callback(None) or
        # register a custom handler.
        self._interaction_callback = self._default_interaction_dispatch

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "QQBot"

    @property
    def enforces_own_access_policy(self) -> bool:
        """QQBot gates DM/group access at intake via dm_policy/group_policy."""
        return True

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """
        Authenticate, obtain gateway URL, and open the WebSocket.

        Args:
            is_reconnect: False on a cold first boot; True when the
                reconnect watcher is re-establishing this platform after
                an outage. QQBot has no server-side update queue so this
                flag is accepted for interface conformance only.
        """
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

                self._mark_transport_disconnected()
                self._fail_pending("Connection closed")

                # Stop reconnecting for fatal codes (unrecoverable errors)
                if code in {
                        4001,  # Invalid opcode
                        4002,  # Invalid payload
                        4010,  # Invalid shard
                        4011,  # Sharding required
                        4012,  # Invalid API version
                        4013,  # Invalid intent
                        4014,  # Intent not authorized
                        4914,  # Offline/sandbox-only
                        4915,  # Banned
                }:
                    fatal_descriptions = {
                        4001: "invalid opcode",
                        4002: "invalid payload",
                        4010: "invalid shard",
                        4011: "sharding required",
                        4012: "invalid API version",
                        4013: "invalid intent",
                        4014: "intent not authorized",
                        4914: "offline/sandbox-only",
                        4915: "banned",
                    }
                    desc = fatal_descriptions.get(code, f"fatal error (code={code})")
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
                        self._mark_disconnected()
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
                # Note: 4009 (connection timeout) is NOT included here — it is
                # resumable per the QQ protocol and should preserve session state.
                if code in {
                        4006,
                        4007,
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
                }:
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
                        self._mark_disconnected()
                        return

            except Exception as exc:
                if not self._running:
                    return
                logger.warning("[%s] WebSocket error: %s", self._log_tag, exc)
                self._mark_transport_disconnected()
                self._fail_pending("Connection interrupted")

                if backoff_idx >= MAX_RECONNECT_ATTEMPTS:
                    logger.error("[%s] Max reconnect attempts reached", self._log_tag)
                    self._mark_disconnected()
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
        if self._ws.closed:
            # A closed-but-non-None ws makes the while-condition false on entry,
            # so this would return normally — which _listen_loop treats as a
            # clean read and immediately retries with backoff reset to 0,
            # producing a 100% CPU spin. Raise so the reconnect/backoff path runs.
            raise RuntimeError("WebSocket closed")

        while self._running and self._ws and not self._ws.closed:
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                payload = self._parse_json(msg.data)
                if payload:
                    self._dispatch_payload(payload)
            elif msg.type in {aiohttp.WSMsgType.PING,}:
                # aiohttp auto-replies with PONG
                pass
            elif msg.type == aiohttp.WSMsgType.CLOSE:
                raise QQCloseError(msg.data, msg.extra)
            elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
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
                           | (1 << 12)
                           | (1 << 26),  # C2C_GROUP_AT_MESSAGES + PUBLIC_GUILD_MESSAGES + DIRECT_MESSAGE + INTERACTION
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
            elif t in {
                    "C2C_MESSAGE_CREATE",
                    "GROUP_AT_MESSAGE_CREATE",
                    "GROUP_MESSAGE_CREATE",
                    "DIRECT_MESSAGE_CREATE",
                    "GUILD_MESSAGE_CREATE",
                    "GUILD_AT_MESSAGE_CREATE",
            }:
                asyncio.create_task(self._on_message(t, d))
            elif t == "INTERACTION_CREATE":
                self._create_task(self._on_interaction(d))
            else:
                logger.debug("[%s] Unhandled dispatch: %s", self._log_tag, t)
            return

        # op 11 = Heartbeat ACK
        if op == 11:
            return

        # op 7 = Server Reconnect — server asks client to reconnect (e.g.
        # load-balancing, maintenance).  Close the WS so _read_events raises
        # and the outer loop triggers a reconnect with Resume.
        if op == 7:
            logger.info("[%s] Server requested reconnect (op 7)", self._log_tag)
            if self._ws and not self._ws.closed:
                self._create_task(self._ws.close())
            return

        # op 9 = Invalid Session — d=True means session is resumable,
        # d=False means we must re-identify from scratch.
        if op == 9:
            resumable = bool(d) if d is not None else False
            if not resumable:
                logger.info(
                    "[%s] Invalid session (op 9, not resumable), clearing session",
                    self._log_tag,
                )
                self._session_id = None
                self._last_seq = None
            else:
                logger.info("[%s] Invalid session (op 9, resumable)", self._log_tag)
            if self._ws and not self._ws.closed:
                self._create_task(self._ws.close())
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
        elif event_type in {"GROUP_AT_MESSAGE_CREATE", "GROUP_MESSAGE_CREATE"}:
            await self._handle_group_message(
                d, msg_id, content, author, timestamp, event_type
            )
        elif event_type in {"GUILD_MESSAGE_CREATE", "GUILD_AT_MESSAGE_CREATE"}:
            await self._handle_guild_message(d, msg_id, content, author, timestamp)
        elif event_type == "DIRECT_MESSAGE_CREATE":
            await self._handle_dm_message(d, msg_id, content, author, timestamp)

    # ------------------------------------------------------------------
    # Inline-keyboard interactions (INTERACTION_CREATE)
    # ------------------------------------------------------------------

    def set_interaction_callback(
        self,
        callback: Optional[Callable[[InteractionEvent], Awaitable[None]]],
    ) -> None:
        """Register (or clear) the interaction callback.

        Invoked once per ``INTERACTION_CREATE`` event *after* the adapter has
        ACKed the interaction. The callback is responsible for routing the
        button click to the right subsystem (approval resolver, update-prompt
        resolver, etc.) based on the ``button_data`` payload.
        """
        self._interaction_callback = callback

    async def _on_interaction(self, d: Any) -> None:
        """Handle an ``INTERACTION_CREATE`` event.

        Responsibilities:

        1. Parse the raw payload into an :class:`InteractionEvent`.
        2. ACK the interaction (``PUT /interactions/{id}``) so the client
           stops showing a loading indicator on the button.
        3. Dispatch to the registered interaction callback, if any.
        """
        if not isinstance(d, dict):
            return
        try:
            event = parse_interaction_event(d)
        except Exception as exc:
            logger.warning(
                "[%s] Failed to parse INTERACTION_CREATE: %s", self._log_tag, exc
            )
            return

        if not event.id:
            logger.warning(
                "[%s] INTERACTION_CREATE missing id, skipping ACK", self._log_tag
            )
            return

        # ACK the interaction promptly — per the QQ docs the client will show
        # an error icon on the button if we don't respond quickly.
        try:
            await self._acknowledge_interaction(event.id)
        except Exception as exc:
            logger.warning(
                "[%s] Failed to ACK interaction %s: %s",
                self._log_tag, event.id, exc,
            )

        logger.info(
            "[%s] Interaction: scene=%s button_data=%r operator=%s",
            self._log_tag, event.scene, event.button_data, event.operator_openid,
        )

        callback = self._interaction_callback
        if callback is None:
            logger.debug(
                "[%s] No interaction callback registered; dropping button "
                "click %r",
                self._log_tag, event.button_data,
            )
            return
        try:
            await callback(event)
        except Exception as exc:
            logger.error(
                "[%s] Interaction callback raised: %s",
                self._log_tag, exc, exc_info=True,
            )

    async def _acknowledge_interaction(
            self,
            interaction_id: str,
            code: int = 0,
    ) -> None:
        """ACK a button interaction via ``PUT /interactions/{id}``.

        :param interaction_id: The ``id`` field from the
            ``INTERACTION_CREATE`` event.
        :param code: Response code (``0`` = success).
        """
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized — not connected?")
        token = await self._ensure_token()
        headers = {
            "Authorization": f"QQBot {token}",
            "Content-Type": "application/json",
            "User-Agent": build_user_agent(),
        }
        resp = await self._http_client.put(
            f"{API_BASE}/interactions/{interaction_id}",
            headers=headers,
            json={"code": code},
            timeout=DEFAULT_API_TIMEOUT,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Interaction ACK failed [{resp.status_code}]: "
                f"{resp.text[:200]}"
            )

    # Mapping from QQ keyboard button decisions → the ``choice`` vocabulary
    # accepted by ``tools.approval.resolve_gateway_approval``. QQ's 3-button
    # layout (mobile-space constraint) collapses "session" and "always" into
    # a single "always" button; users wanting session-only approval can fall
    # back to the ``/approve session`` text command.
    _APPROVAL_BUTTON_TO_CHOICE = {
        "allow-once": "once",
        "allow-always": "always",
        "deny": "deny",
    }

    @staticmethod
    def _parse_gateway_session_key(session_key: str) -> Optional[Dict[str, str]]:
        """Parse ``agent:main:<platform>:<chat_type>:<chat_id>[:<user_id>]``."""
        parts = str(session_key or "").split(":")
        if len(parts) < 5 or parts[0] != "agent" or parts[1] != "main":
            return None
        parsed = {
            "platform": parts[2],
            "chat_type": parts[3],
            "chat_id": parts[4],
        }
        if len(parts) > 5:
            parsed["user_id"] = parts[5]
        return parsed

    def _is_authorized_interaction_for_session(
            self,
            event: InteractionEvent,
            session_key: str,
    ) -> bool:
        """Authorize approval/update interactions against session + operator."""
        parsed = self._parse_gateway_session_key(session_key)
        operator = str(event.operator_openid or "").strip()
        if not parsed or parsed.get("platform") != "qqbot" or not operator:
            return False

        chat_type = parsed.get("chat_type", "")
        chat_id = parsed.get("chat_id", "")
        if chat_type == "c2c":
            return bool(chat_id) and operator == chat_id

        if chat_type in {"group", "guild"}:
            event_chat = str(event.group_openid or event.guild_id or "").strip()
            if not event_chat or event_chat != chat_id:
                return False
            session_user = str(parsed.get("user_id", "")).strip()
            return bool(session_user) and operator == session_user

        return False

    async def _default_interaction_dispatch(
            self,
            event: InteractionEvent,
    ) -> None:
        """Route ``INTERACTION_CREATE`` button clicks to the right subsystem.

        - ``approve:<session_key>:<decision>`` →
          :func:`tools.approval.resolve_gateway_approval`
          (unblocks the agent thread waiting on a dangerous-command approval).
        - ``update_prompt:<answer>`` →
          writes the answer to ``~/.hermes/.update_response`` for the
          detached ``hermes update --gateway`` process to consume.
        - Anything else is logged at DEBUG and ignored.

        Installed as the adapter's default interaction callback in
        ``__init__``. Callers can replace via
        :meth:`set_interaction_callback` to route clicks elsewhere (or pass
        ``None`` to drop them entirely).
        """
        button_data = event.button_data
        if not button_data:
            return

        approval = parse_approval_button_data(button_data)
        if approval is not None:
            session_key, decision = approval
            choice = self._APPROVAL_BUTTON_TO_CHOICE.get(decision)
            if choice is None:
                logger.warning(
                    "[%s] Unknown approval decision %r (session=%s)",
                    self._log_tag, decision, session_key,
                )
                return
            if not self._is_authorized_interaction_for_session(event, session_key):
                logger.warning(
                    "[%s] Rejected unauthorized approval click for session %s "
                    "(operator=%s)",
                    self._log_tag, session_key, event.operator_openid,
                )
                return
            try:
                # Import lazily to keep the adapter importable in tests that
                # don't exercise the approval subsystem.
                from tools.approval import resolve_gateway_approval
                count = resolve_gateway_approval(session_key, choice)
                logger.info(
                    "[%s] Button resolved %d approval(s) for session %s "
                    "(choice=%s, operator=%s)",
                    self._log_tag, count, session_key, choice,
                    event.operator_openid,
                )
            except Exception as exc:
                logger.error(
                    "[%s] resolve_gateway_approval failed for session %s: %s",
                    self._log_tag, session_key, exc,
                )
            return

        update_answer = parse_update_prompt_button_data(button_data)
        if update_answer is not None:
            update_session_key = f"agent:main:qqbot:{event.scene}:{event.group_openid or event.guild_id or event.user_openid}"
            if not self._is_authorized_interaction_for_session(event, update_session_key):
                logger.warning(
                    "[%s] Rejected unauthorized update prompt click (operator=%s)",
                    self._log_tag, event.operator_openid,
                )
                return
            self._write_update_response(update_answer, event.operator_openid)
            return

        logger.debug(
            "[%s] Unrecognised button_data %r from interaction %s",
            self._log_tag, button_data, event.id,
        )

    @staticmethod
    def _write_update_response(answer: str, operator: str = "") -> None:
        """Atomically write the update-prompt answer to ``.update_response``.

        Mirrors the Discord / Telegram / Feishu adapters: the detached
        ``hermes update --gateway`` watcher polls this file for a ``y``/``n``
        response to its interactive prompts (stash-restore, config migration).
        Writes via ``tmp + rename`` so a partial write can't fool the reader.
        """
        try:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
            response_path = home / ".update_response"
            tmp = response_path.with_suffix(".tmp")
            tmp.write_text(answer)
            tmp.replace(response_path)
            logger.info(
                "QQ update prompt answered %r by %s",
                answer, operator or "(unknown)",
            )
        except Exception as exc:
            logger.error("Failed to write update response: %s", exc)

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
        if not self._is_dm_intake_allowed(user_openid):
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

        # Merge any quoted-message context (message_type=103 → msg_elements[0]).
        quoted = await self._process_quoted_context(d)
        text = self._merge_quote_into(text, quoted["quote_block"])
        if quoted["image_urls"]:
            image_urls = image_urls + quoted["image_urls"]
            image_media_types = image_media_types + quoted["image_media_types"]

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
            event_type: str = "GROUP_AT_MESSAGE_CREATE",
    ) -> None:
        """Handle a group message event.

        Handles both ``GROUP_AT_MESSAGE_CREATE`` (bot @-ed) and, when the QQ
        platform pushes full group traffic and/or the bot runs in always mode,
        ``GROUP_MESSAGE_CREATE`` (any group message). The activation decision
        is:

        1. group-level ACL (``_is_group_allowed``; user_id is intentionally not
           used — per-user filtering is out of scope);
        2. mention detection (``detect_mentioned``);
        3. mode resolution (``resolve_require_mention``: global +
           per-group override);
        4. gate: mention mode + not-mentioned → skip (no reply); otherwise
           process and reply.
        """
        group_openid = str(d.get("group_openid", ""))
        if not group_openid:
            return
        member_openid = str(author.get("member_openid", ""))
        # (1) ACL — group-level (per-user filtering intentionally out of scope).
        if not self._is_group_allowed(group_openid, member_openid):
            logger.debug(
                "[%s] Group message blocked by ACL: group=%r "
                "policy=%s (set platforms.qqbot.extra.group_policy=open, or "
                "'allowlist' with this group in group_allow_from, to enable).",
                self._log_tag, group_openid, self._group_policy,
            )
            return

        # (2) mention detection + (3) mode resolution + (4) activation gate.
        mentioned = detect_mentioned(event_type, d, content, self._app_id)
        require_mention = resolve_require_mention(
            group_openid,
            global_default=self._group_require_mention,
            per_group=self._group_mode_overrides,
            runtime_overrides=self._group_mode_runtime_overrides,
        )
        if require_mention and not mentioned:
            # mention mode + message not addressed to the bot → do not reply,
            # but remember it as pending context for the next @-reply (2.2.1).
            self._group_context.record(
                group_openid,
                sender=member_openid,
                text=content,
                msg_id=msg_id,
                attachment_tag=summarize_attachments(d.get("attachments")),
            )
            logger.debug(
                "[%s] Group %s: buffered non-mention message (mention mode, event=%s)",
                self._log_tag, group_openid, event_type,
            )
            return

        # (5) pass: assemble the message (reuse the existing media / quote /
        # markdown path). Only strip the @bot prefix when actually mentioned.
        text = self._strip_at_mention(content) if mentioned else content
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

        # Merge any quoted-message context (message_type=103 → msg_elements[0]).
        quoted = await self._process_quoted_context(d)
        text = self._merge_quote_into(text, quoted["quote_block"])
        if quoted["image_urls"]:
            image_urls = image_urls + quoted["image_urls"]
            image_media_types = image_media_types + quoted["image_media_types"]

        # (6) mention-mode @-activation: drain any buffered non-@ messages and
        # carry them as CONTEXT ONLY via ``channel_context`` — NOT merged into
        # ``text``. Keeping the buffered history out of ``text`` leaves the
        # trigger message at the start of ``text`` so slash-command detection
        # (event.get_command) and sender-prefix logic in run.py operate on the
        # current message alone; run.py prepends channel_context afterwards.
        # This fixes ``/stop``-style commands failing to match in mention/context
        # modes where the buffered block previously sat ahead of the command.
        # Done BEFORE the empty check so a bare @ (no body) still flushes and
        # injects the pending context. In always mode the buffer is empty
        # (nothing was recorded), so drain is a harmless no-op.
        pending = self._group_context.drain(group_openid)
        channel_context = (
            GroupContextBuffer.format_context_block(pending) if pending else None
        )

        # An empty trigger with no media is normally dropped, but a bare @ that
        # flushes pending context must still produce a turn — otherwise the
        # buffered context would be stranded.
        if not text.strip() and not image_urls and not channel_context:
            return

        self._chat_type_map[group_openid] = "group"
        event = MessageEvent(
            source=self.build_source(
                chat_id=group_openid,
                user_id=str(author.get("member_openid", "")),
                chat_type="group",
            ),
            text=text,
            channel_context=channel_context,
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

        # Merge any quoted-message context (message_type=103 → msg_elements[0]).
        quoted = await self._process_quoted_context(d)
        text = self._merge_quote_into(text, quoted["quote_block"])
        if quoted["image_urls"]:
            image_urls = image_urls + quoted["image_urls"]
            image_media_types = image_media_types + quoted["image_media_types"]

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
        if not self._is_dm_intake_allowed(author_id):
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

        # Merge any quoted-message context (message_type=103 → msg_elements[0]).
        quoted = await self._process_quoted_context(d)
        text = self._merge_quote_into(text, quoted["quote_block"])
        if quoted["image_urls"]:
            image_urls = image_urls + quoted["image_urls"]
            image_media_types = image_media_types + quoted["image_media_types"]

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
    # Quoted-message handling
    # ------------------------------------------------------------------

    async def _process_quoted_context(
            self,
            d: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process the quoted message a user is replying to.

        When a user replies while quoting another message, the platform sets
        ``message_type = 103`` and pushes the referenced message's content and
        attachments inside ``msg_elements[0]``. The old adapter ignored
        ``msg_elements`` entirely, so:

        - Quoted text was surfaced only when the user typed something of
          their own — bare quote-replies showed nothing.
        - Quoted attachments (images, voice, files) were never downloaded
          or described.
        - Quoted voice messages specifically produced no transcript, so the
          LLM had no way to see what the user was referring to.

        This method parses ``msg_elements`` and runs the quoted attachments
        through the same :meth:`_process_attachments` pipeline as the main
        message body, so quoted voice messages get STT transcripts and
        quoted images are cached identically.

        :param d: Raw inbound message dict (from the WS dispatch payload).
        :returns: Dict with keys:

            - ``quote_block``: string to prepend to the user's text body
              (empty when there's nothing quoted).
            - ``image_urls``: list of cached quoted-image paths.
            - ``image_media_types``: parallel list of image MIME types.
        """
        empty = {
            "quote_block": "",
            "image_urls": [],
            "image_media_types": [],
        }
        # Short-circuit: only message_type 103 indicates a quote.
        try:
            if int(d.get("message_type", 0) or 0) != 103:
                return empty
        except (TypeError, ValueError):
            return empty

        elements = d.get("msg_elements")
        if not isinstance(elements, list) or not elements:
            return empty

        # msg_elements[0] carries the referenced message. Additional elements
        # (if any) are very rare in practice; we concatenate their text and
        # union their attachments for completeness.
        quoted_text_parts: List[str] = []
        all_attachments: List[Dict[str, Any]] = []
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            etext = str(elem.get("content", "")).strip()
            if etext:
                quoted_text_parts.append(etext)
            eatts = elem.get("attachments")
            if isinstance(eatts, list):
                for a in eatts:
                    if isinstance(a, dict):
                        all_attachments.append(a)

        att_result = await self._process_attachments(all_attachments)
        quoted_voice = att_result.get("voice_transcripts") or []
        quoted_info = att_result.get("attachment_info") or ""
        quoted_images = att_result.get("image_urls") or []
        quoted_image_types = att_result.get("image_media_types") or []

        lines: List[str] = []
        if quoted_text_parts:
            lines.append(" ".join(quoted_text_parts))
        for t in quoted_voice:
            lines.append(t)
        if quoted_info:
            lines.append(quoted_info)

        if not lines and not quoted_images:
            return empty

        if lines:
            quote_block = "[Quoted message]:\n" + "\n".join(lines)
        else:
            # Images-only quote: give the LLM at least a marker so it knows
            # context was referenced.
            quote_block = "[Quoted message]: (image)"

        return {
            "quote_block": quote_block,
            "image_urls": quoted_images,
            "image_media_types": quoted_image_types,
        }

    @staticmethod
    def _merge_quote_into(text: str, quote_block: str) -> str:
        """Prepend ``quote_block`` to *text*, separated by a blank line."""
        if not quote_block:
            return text
        if text.strip():
            return f"{quote_block}\n\n{text}".strip()
        return quote_block

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
                    cached_path = await self._download_and_cache(url, ct, filename)
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
                # Other attachments (video, file, etc.): download and record with path.
                try:
                    cached_path = await self._download_and_cache(url, ct, filename)
                    if cached_path:
                        name = filename or ct
                        if ct.startswith("video/"):
                            other_attachments.append(f"[video: {name} ({cached_path})]")
                        else:
                            other_attachments.append(f"[file: {name} ({cached_path})]")
                except Exception as exc:
                    logger.debug("[%s] Failed to cache attachment: %s", self._log_tag, exc)

        attachment_info = "\n".join(other_attachments) if other_attachments else ""
        return {
            "image_urls": image_urls,
            "image_media_types": image_media_types,
            "voice_transcripts": voice_transcripts,
            "attachment_info": attachment_info,
        }

    async def _download_and_cache(
            self, url: str, content_type: str, original_name: str = "",
    ) -> Optional[str]:
        """Download a URL and cache it locally.

        :param original_name: Preferred filename from attachment metadata.
            Falls back to the URL path basename if empty.
        """
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
            filename = (
                original_name
                or Path(urlparse(url).path).name
                or "qq_attachment"
            )
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
        if data[:9] == b"#!SILK_V3" or data[:6] == b"#!SILK":
            return ".silk"
        if data[:2] == b"\x02!":
            return ".silk"
        if data[:4] == b"RIFF":
            return ".wav"
        if data[:4] == b"fLaC":
            return ".flac"
        if data[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
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
        return data[:6] == b"#!SILK" or data[:2] == b"\x02!" or data[:9] == b"#!SILK_V3"

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
                                 or ("glm-asr" if provider in {"zai", "glm"} else "whisper-1"),
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
        if not ext or ext not in {
                ".silk",
                ".amr",
                ".mp3",
                ".wav",
                ".ogg",
                ".m4a",
                ".aac",
                ".flac",
        }:
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

    def _evict_group_defer_sessions_if_needed(self) -> None:
        """Bound ``_group_defer_sessions`` size by dropping the oldest
        finalized entries (LRU on insertion order) when the map exceeds
        its cap.  Non-finalized entries are preserved — dropping one
        would silently discard an in-flight reply and force the fallback
        "session missing → fresh send" path.
        """
        cap = self._group_defer_sessions_cap
        if len(self._group_defer_sessions) <= cap:
            return
        # Dicts preserve insertion order (Py 3.7+); walk oldest → newest
        # and evict finalized entries first.  If everything is still
        # pending we bail — capacity is best-effort, correctness wins.
        to_drop = len(self._group_defer_sessions) - cap
        stale_keys = [
            k for k, v in self._group_defer_sessions.items()
            if v.get("finalized")
        ][:to_drop]
        for k in stale_keys:
            self._group_defer_sessions.pop(k, None)

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

        When the caller signals a streaming reply (``metadata.expect_edits``)
        and the target is a C2C chat, the first chunk is delivered through
        QQ's native ``stream_messages`` endpoint instead of the regular
        ``/messages`` API.  The returned ``SendResult.message_id`` is an
        adapter-generated opaque handle (``StreamSession.logical_id``); the
        stream consumer uses it in follow-up ``edit_message`` calls.  On
        streaming-start failure we log a warning and fall through to the
        legacy path so the user still receives an answer.
        """
        if not self.is_connected:
            if not await self._wait_for_reconnection():
                return SendResult(success=False, error="Not connected", retryable=True)

        if not content or not content.strip():
            return SendResult(success=True)

        expect_edits = bool((metadata or {}).get("expect_edits"))
        chat_type = self._guess_chat_type(chat_id)

        # Streaming fast-path: C2C + expect_edits + globally enabled.
        # QQ's ``stream_messages`` endpoint only accepts C2C targets, so
        # group / guild replies take a different route (see below).
        if (
                self._streaming_enabled
                and expect_edits
                and chat_type == "c2c"
        ):
            stream_result = await self._start_stream_reply(chat_id, content, reply_to)
            if stream_result is not None:
                return stream_result
            # Falls through to the legacy path on start failure — the
            # helper has already emitted a warning log.

        # Group / guild streaming: no editable message id on QQ side.
        #
        # For group and guild targets QQ does not support in-place
        # message edits, and the ``stream_messages`` endpoint is C2C
        # only.  If we let the streaming first-send deliver a real
        # message here, the stream consumer would try ``edit_message``
        # on it for every subsequent tick — each edit fails, and the
        # consumer's fallback re-sends the full accumulated text as a
        # brand-new message.  Users then see the first chunk followed
        # by the complete answer, i.e. two messages per turn.
        #
        # To collapse that to a single complete reply we:
        #
        #   1. buffer the streaming session inside the adapter,
        #   2. hand out a sentinel ``message_id`` so the consumer
        #      stays on its edit path (avoiding the multi-send
        #      fallback machinery — including its lossy prefix-based
        #      continuation logic which would drop the first chunk),
        #   3. no-op every ``edit_message`` (see ``edit_message``
        #      below), and
        #   4. on the final ``edit_message(finalize=True)``, send ONE
        #      complete message via the regular legacy path (which
        #      handles chunking, keyboards, retries etc. correctly).
        #      Subsequent ``finalize=True`` edits on the same sentinel
        #      are idempotent — the stream consumer may issue a second
        #      finalize tick on ``REQUIRES_EDIT_FINALIZE`` adapters
        #      (see the ``got_done`` branch in
        #      ``gateway/stream_consumer.py``), and re-delivering the
        #      reply would surface as duplicate messages on screen.
        if expect_edits and chat_type in ("group", "guild"):
            sentinel_id = f"__qqbot_group_defer_{uuid.uuid4().hex}__"
            self._group_defer_sessions[sentinel_id] = {
                "chat_id": chat_id,
                "chat_type": chat_type,
                "reply_to": reply_to,
                "finalized": False,
                "finalized_result": None,
            }
            self._evict_group_defer_sessions_if_needed()
            logger.debug(
                "[%s] streaming first-send buffered for %s chat %s "
                "(sentinel=%s): the complete reply will be delivered "
                "on finalize",
                self._log_tag, chat_type, chat_id, sentinel_id,
            )
            return SendResult(success=True, message_id=sentinel_id)

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
            self,
            openid: str,
            content: str,
            reply_to: Optional[str] = None,
            keyboard: Optional[InlineKeyboard] = None,
    ) -> SendResult:
        """Send text to a C2C user via REST API.

        :param keyboard: Optional inline keyboard attached to the message.
        """
        self._next_msg_seq(reply_to or openid)
        body = self._build_text_body(content, reply_to)
        if reply_to:
            body["msg_id"] = reply_to
        if keyboard is not None:
            body["keyboard"] = keyboard.to_dict()

        data = await self._api_request("POST", f"/v2/users/{openid}/messages", body)
        msg_id = str(data.get("id", uuid.uuid4().hex[:12]))
        return SendResult(success=True, message_id=msg_id, raw_response=data)

    async def _send_group_text(
            self,
            group_openid: str,
            content: str,
            reply_to: Optional[str] = None,
            keyboard: Optional[InlineKeyboard] = None,
    ) -> SendResult:
        """Send text to a group via REST API.

        :param keyboard: Optional inline keyboard attached to the message.
        """
        self._next_msg_seq(reply_to or group_openid)
        body = self._build_text_body(content, reply_to)
        if reply_to:
            body["msg_id"] = reply_to
        if keyboard is not None:
            body["keyboard"] = keyboard.to_dict()

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

    # ------------------------------------------------------------------
    # C2C streaming reply (POST /v2/users/{openid}/stream_messages)
    # ------------------------------------------------------------------

    async def _start_stream_reply(
            self,
            openid: str,
            content: str,
            reply_to: Optional[str],
    ) -> Optional[SendResult]:
        """Open a C2C streaming session and send the first chunk.

        Returns a successful :class:`SendResult` whose ``message_id`` is
        the adapter's opaque ``logical_id`` on success.  On failure
        returns ``None`` (a warning is logged) so the caller can fall
        back to the legacy send path — per user requirement (d): first
        chunk failure degrades to a normal message within the same
        ``send()`` call to minimise user-visible latency.
        """
        # QQ's stream_messages endpoint requires a valid passive-reply
        # msg_id (event_id is the same value in openclaw's mapping).  If
        # the consumer hasn't provided reply_to, fall back to the most
        # recent inbound id we've seen for this chat.  With no passive
        # id we cannot start streaming — return None to trigger legacy
        # fallback rather than making a request we know will fail.
        passive_msg_id = reply_to or self._last_msg_id.get(openid)
        if not passive_msg_id:
            logger.warning(
                "[%s] Streaming reply skipped for %s: no passive msg_id available; "
                "falling back to regular send",
                self._log_tag, openid,
            )
            return None

        session = self._stream_manager.create(
            openid=openid,
            passive_msg_id=passive_msg_id,
            msg_seq=self._next_msg_seq(passive_msg_id),
        )
        # First chunk has no prior text to reconcile against and the
        # gateway suppresses the typewriter cursor for QQBOT (see
        # ``gateway/run.py`` ``_effective_cursor`` handling), so we
        # forward ``content`` verbatim.
        try:
            resp = await self._call_stream_api(
                session, content, input_state=1,
            )
        except Exception as exc:
            # Purge the abandoned session so its logical_id can't leak
            # into an ``edit_message`` lookup later.
            self._stream_manager.drop(session.logical_id)
            logger.warning(
                "[%s] Failed to start C2C streaming reply for %s "
                "(passive_msg_id=%s): %s — falling back to regular send",
                self._log_tag, openid, passive_msg_id, exc,
            )
            return None

        stream_msg_id = resp.get("id") if isinstance(resp, dict) else None
        if not stream_msg_id:
            self._stream_manager.drop(session.logical_id)
            logger.warning(
                "[%s] Streaming reply for %s: response missing 'id' field "
                "(resp=%r); falling back to regular send",
                self._log_tag, openid, resp,
            )
            return None

        session.stream_msg_id = str(stream_msg_id)
        session.last_sent_index = session.next_index
        session.last_sent_content = content[:MAX_STREAM_CONTENT_LEN]
        session.next_index += 1
        logger.debug(
            "[%s] C2C streaming reply started: logical_id=%s stream_msg_id=%s",
            self._log_tag, session.logical_id, session.stream_msg_id,
        )
        return SendResult(
            success=True,
            message_id=session.logical_id,
            raw_response=resp,
        )

    async def edit_message(
            self,
            chat_id: str,
            message_id: str,
            content: str,
            *,
            finalize: bool = False,
    ) -> SendResult:
        """Edit a streamed reply.

        Two flavours are supported:

        * **C2C streaming sessions** (``message_id`` matches an entry
          in ``_stream_manager``): forwarded to QQ's ``stream_messages``
          endpoint as an append.
        * **Group / guild deferred sessions** (``message_id`` starts
          with ``__qqbot_group_defer_``): buffered locally.  Non-final
          edits are silent no-ops; on ``finalize=True`` the accumulated
          text is delivered as ONE complete message via the regular
          legacy send path (chunking, keyboards, retries all handled by
          the standard ``send()``).  This collapses what would otherwise
          become two-plus messages per turn — see ``send()`` for the
          full rationale.  A **second** ``finalize=True`` edit on the
          same sentinel is idempotent: we memoise the delivery result
          and replay it instead of re-sending (the stream consumer's
          ``REQUIRES_EDIT_FINALIZE`` code path can issue a redundant
          second finalize tick).

        For any other unrecognised ``message_id`` we return
        ``success=False`` so the stream consumer falls back to a fresh
        ``send()``.  Content is treated as the full accumulated text
        (``input_mode=replace`` semantics on the QQ side), matching the
        hermes base ``edit_message`` contract of "replace the message
        with this content".

        For the simplified out-of-order policy agreed for this release,
        the adapter tracks the last successfully-sent ``index`` and
        silently drops any edit that arrives out of sequence (reports
        ``success=True`` without invoking QQ, so the consumer keeps its
        internal cursor advancing).  In practice the consumer awaits
        each edit serially, so this branch is defensive only.
        """
        # Group / guild deferred session: accumulate silently, flush
        # on finalize.  ``content`` here is the FULL accumulated text
        # for this tick (per the base edit_message contract).
        if message_id and message_id.startswith("__qqbot_group_defer_"):
            defer = self._group_defer_sessions.get(message_id)
            if defer is None:
                # Session expired or already flushed — fresh-send the
                # content so the user still gets a reply.  Falls
                # through to the regular ``send()`` path (no
                # expect_edits → legacy chunked delivery).
                logger.warning(
                    "[%s] deferred group session %s missing on edit "
                    "(finalize=%s); recovering via fresh send to %s",
                    self._log_tag, message_id, finalize, chat_id,
                )
                if not finalize:
                    # Nothing to do until finalize — the consumer will
                    # call us again with the full text.
                    return SendResult(success=True, message_id=message_id)
                return await self.send(chat_id, content, reply_to=None)

            if not finalize:
                # Silent no-op: the next edit (or the finalize edit)
                # will carry the newer accumulated text.  We do NOT
                # store ``content`` here — the consumer always sends
                # the full text on every edit call, so the finalize
                # payload is authoritative.
                #
                # If a previous finalize already delivered the reply,
                # any lingering non-final edits (e.g. cancellation
                # cleanup) are silent no-ops as well — the message is
                # already out.
                return SendResult(success=True, message_id=message_id)

            # ``finalize=True`` re-entry guard.
            #
            # The stream consumer's ``got_done`` branch may issue TWO
            # ``_send_or_edit(..., finalize=True)`` calls per turn on
            # adapters that set ``REQUIRES_EDIT_FINALIZE=True`` — the
            # mid-stream flush (line 742 in ``stream_consumer.py``)
            # plus the explicit final tick right below (line 792).
            # The first call already delivered the complete reply via
            # ``self.send(...)``, so re-delivering here would post a
            # second identical message ("two full events per group
            # turn").  Idempotently return the memoised result.
            if defer.get("finalized"):
                logger.debug(
                    "[%s] deferred group finalize replayed for %s "
                    "(sentinel=%s); returning memoised result",
                    self._log_tag, defer.get("chat_id"), message_id,
                )
                cached = defer.get("finalized_result")
                if isinstance(cached, SendResult):
                    return cached
                return SendResult(success=True, message_id=message_id)

            # Finalize: deliver ONE complete reply via the legacy
            # (non-streaming) send path, using the reply_to captured
            # at ``send()`` time so threading is preserved.
            defer_chat_id = defer.get("chat_id") or chat_id
            reply_to = defer.get("reply_to")
            # Flip the flag BEFORE the send so a re-entrant edit
            # (shouldn't happen, but defensive) hits the memoised
            # branch above and can't double-send even if the send
            # itself awaits at an ``await`` boundary.
            defer["finalized"] = True
            result = await self.send(
                defer_chat_id, content, reply_to=reply_to,
            )
            defer["finalized_result"] = result
            if not result.success:
                logger.warning(
                    "[%s] deferred group finalize failed for %s "
                    "(sentinel=%s): %s",
                    self._log_tag, defer_chat_id, message_id, result.error,
                )
            return result

        del chat_id  # Session lookup is by logical_id, chat is implied.

        session = self._stream_manager.get(message_id)
        if session is None:
            # Unknown / expired session: the consumer will re-send from
            # scratch on the next tick.  This also covers group / guild
            # ``edit_message`` calls — those never have a matching
            # session because ``send()`` never opened one for them.
            return SendResult(
                success=False,
                error="stream session not found or expired",
            )

        if session.finalized:
            # Idempotent no-op — protects against double-DONE which
            # would confuse QQ (index would regress from its POV).
            logger.debug(
                "[%s] edit_message on already-finalized session %s ignored",
                self._log_tag, message_id,
            )
            return SendResult(success=True, message_id=message_id)

        # Defensive out-of-order guard.  next_index tracks what we're
        # about to send; last_sent_index tracks what QQ has already
        # accepted.  next_index <= last_sent_index means we've already
        # sent something at or past this position, which shouldn't
        # happen with a serial consumer — treat as a no-op success.
        my_index = session.next_index
        if my_index <= session.last_sent_index:
            logger.debug(
                "[%s] out-of-order stream chunk dropped: my_index=%d last_sent=%d",
                self._log_tag, my_index, session.last_sent_index,
            )
            return SendResult(success=True, message_id=message_id)

        input_state = 10 if finalize else 1
        outgoing = self._enforce_stream_prefix(session, content)
        try:
            resp = await self._call_stream_api(
                session, outgoing, input_state=input_state,
            )
        except Exception as exc:
            error_str = str(exc)
            retryable = not any(
                kw in error_str.lower()
                for kw in ("invalid", "forbidden", "not found", "bad request")
            )
            logger.warning(
                "[%s] Stream edit failed (logical_id=%s stream_msg_id=%s "
                "index=%d finalize=%s): %s",
                self._log_tag, message_id, session.stream_msg_id,
                my_index, finalize, exc,
            )
            # On terminal failures purge the session so subsequent edits
            # get a fresh ``success=False`` fast instead of retrying
            # against a broken stream.
            if not retryable:
                self._stream_manager.drop(message_id)
            return SendResult(success=False, error=error_str, retryable=retryable)

        session.last_sent_index = my_index
        session.last_sent_content = outgoing[:MAX_STREAM_CONTENT_LEN]
        session.next_index += 1
        if finalize:
            session.finalized = True
            self._stream_manager.drop(message_id)
        return SendResult(
            success=True,
            message_id=message_id,
            raw_response=resp,
        )

    # ------------------------------------------------------------------
    # Stream content prefix guard
    # ------------------------------------------------------------------
    #
    # QQ's ``stream_messages`` endpoint uses ``input_mode=replace`` which
    # means each chunk carries the FULL accumulated text and QQ verifies
    # that the new text starts with the previously accepted text.  Two
    # upstream behaviours can break that invariant:
    #
    # 1. **Typewriter cursor.**  ``GatewayStreamConsumer`` appends a
    #    configurable "typing" cursor (``streaming.cursor``) to every
    #    intermediate frame.  On QQBOT we suppress that injection at
    #    the source — ``gateway/run.py`` sets ``_effective_cursor = ""``
    #    for ``Platform.QQBOT`` (same treatment as ``Platform.MATRIX``),
    #    so no cursor glyph ever reaches this adapter and there is
    #    nothing to reverse-strip here.
    #
    # 2. **Occasional shorter frames / trims.**  A rare edge is a
    #    frame that is shorter than the previous one (retries, dedupe
    #    after a re-emit, model backtrack).  Sending that verbatim
    #    triggers "系统繁忙" (HTTP 500) and permanently breaks the
    #    stream — QQ won't accept any further chunk on the same
    #    ``stream_msg_id``.  We defend against that by forcing each new
    #    chunk to start with ``session.last_sent_content``; if it does
    #    not, we replay the previously-accepted text so QQ sees a
    #    no-op-shaped chunk instead of a rejection.

    def _enforce_stream_prefix(
            self, session: StreamSession, content: str,
    ) -> str:
        """Return a version of ``content`` safe to send on this session.

        The returned text is guaranteed to start with
        ``session.last_sent_content`` (the prefix invariant QQ enforces
        server-side).  If ``content`` does not satisfy that invariant
        (upstream trimmed the frame or the model backtracked), we log a
        warning and fall back to the previously-accepted text — QQ will
        treat the resulting chunk as a stylistic no-op and the stream
        stays alive for the next real update.
        """
        last = session.last_sent_content
        if not last:
            # First edit after the initial send failed to record content
            # (shouldn't happen — ``_start_stream_reply`` always writes
            # ``last_sent_content`` on success), just forward as-is.
            return content
        if content.startswith(last):
            return content
        # Divergence: either the new frame is shorter (upstream trimmed
        # or replayed a smaller partial) or its prefix drifted.  Replay
        # ``last`` so we don't get rejected — the next tick will bring a
        # longer frame that properly extends it.
        logger.warning(
            "[%s] stream prefix divergence on session %s "
            "(last_len=%d new_len=%d); replaying last chunk to keep "
            "QQ session alive",
            self._log_tag, session.logical_id,
            len(last), len(content),
        )
        return last

    async def _call_stream_api(
            self,
            session: StreamSession,
            content: str,
            *,
            input_state: int,
    ) -> Dict[str, Any]:
        """POST one chunk to ``/v2/users/{openid}/stream_messages``.

        Body fields mirror the shape openclaw uses (see
        ``streaming-c2c.ts`` ``sendStreamChunk``):

        - ``input_mode="replace"`` — every call carries the full
          accumulated text.  QQ's Go backend unmarshals this into a
          ``string`` field, so numeric enum values (``1``) are rejected
          with ``json: cannot unmarshal number into Go value of type
          string``.  Use the string form.
        - ``content_type="markdown"`` — fixed per operator requirement;
          same string-vs-number caveat as ``input_mode``.  QQ markdown
          gracefully renders plain-text bodies too.
        - ``event_id`` reuses ``passive_msg_id`` (matches openclaw's
          ``eventId: event.messageId``)
        - ``msg_seq`` is session-wide constant; ``index`` post-increments

        First-chunk requests omit ``stream_msg_id`` — the response body
        carries the platform-assigned id.  Subsequent chunks must include
        it or QQ will treat them as a new stream.
        """
        body: Dict[str, Any] = {
            "input_mode": "replace",  # REPLACE — string, not numeric enum.
            "input_state": input_state,
            "content_type": "markdown",  # MARKDOWN — string, not numeric enum.
            "content_raw": content[:MAX_STREAM_CONTENT_LEN],
            "event_id": session.passive_msg_id,
            "msg_id": session.passive_msg_id,
            "msg_seq": session.msg_seq,
            "index": session.next_index,
        }
        if session.stream_msg_id:
            body["stream_msg_id"] = session.stream_msg_id
        return await self._api_request(
            "POST",
            f"/v2/users/{session.openid}/stream_messages",
            body,
        )

    # ------------------------------------------------------------------
    # Inline-keyboard outbound helpers (approval / update-prompt flows)
    # ------------------------------------------------------------------

    async def send_with_keyboard(
            self,
            chat_id: str,
            content: str,
            keyboard: InlineKeyboard,
            reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send a single text message with an inline keyboard attached.

        Unlike :meth:`send`, this does NOT split long content into chunks —
        a keyboard message has exactly one interactive surface, and splitting
        would orphan the buttons from the first chunk. Callers should keep
        approval/update-prompt bodies short.

        Guild (channel) chats don't support inline keyboards; returns a
        non-retryable failure for those.
        """
        if not self.is_connected:
            if not await self._wait_for_reconnection():
                return SendResult(
                    success=False, error="Not connected", retryable=True
                )

        chat_type = self._guess_chat_type(chat_id)
        formatted = self.format_message(content)
        truncated = formatted[: self.MAX_MESSAGE_LENGTH]
        try:
            if chat_type == "c2c":
                return await self._send_c2c_text(
                    chat_id, truncated, reply_to, keyboard=keyboard,
                )
            if chat_type == "group":
                return await self._send_group_text(
                    chat_id, truncated, reply_to, keyboard=keyboard,
                )
            return SendResult(
                success=False,
                error=(
                    f"Inline keyboards not supported for chat_type "
                    f"{chat_type!r}"
                ),
                retryable=False,
            )
        except Exception as exc:
            logger.error(
                "[%s] send_with_keyboard failed: %s", self._log_tag, exc
            )
            return SendResult(success=False, error=str(exc))

    async def send_approval_request(
            self,
            chat_id: str,
            req: ApprovalRequest,
            reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send a 3-button approval request (``allow-once / allow-always / deny``).

        The rendered text comes from :func:`build_approval_text`; callers can
        override by passing a custom :class:`ApprovalRequest`.

        Users click the button → ``INTERACTION_CREATE`` fires → the adapter's
        registered :meth:`set_interaction_callback` handler decodes
        ``button_data`` via :func:`parse_approval_button_data`.
        """
        from gateway.platforms.qqbot.keyboards import build_approval_text
        return await self.send_with_keyboard(
            chat_id,
            build_approval_text(req),
            build_approval_keyboard(
                req.session_key,
                allow_permanent=getattr(req, "allow_permanent", True),
            ),
            reply_to=reply_to,
        )

    # ------------------------------------------------------------------
    # Cross-adapter gateway contract — send_exec_approval + send_update_prompt
    # ------------------------------------------------------------------
    #
    # These mirror the signatures that gateway/run.py detects on the adapter
    # class (e.g. type(adapter).send_exec_approval, type(adapter).send_update_prompt)
    # for button-based approval / update-confirm UX. Discord, Telegram, Slack,
    # Matrix, and Feishu already implement the same contract.

    async def send_exec_approval(
            self,
            chat_id: str,
            command: str,
            session_key: str,
            description: str = "dangerous command",
            metadata: Optional[Dict[str, Any]] = None,
        allow_permanent: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        """Send a button-based exec-approval prompt for a dangerous command.

        Called by ``gateway/run.py``'s ``_approval_notify_sync`` when the
        agent is blocked waiting for approval. Button clicks resolve via
        :func:`tools.approval.resolve_gateway_approval` — dispatched by the
        adapter's interaction callback (:meth:`_default_interaction_dispatch`).
        """
        del metadata  # QQ doesn't have thread_id / DM targeting overrides.
        if smart_denied:
            description += " Owner override applies to this one operation only."

        # Use the reply-to message for passive-message context when we have one.
        # QQ requires a msg_id on outbound messages to a user we've never
        # seen; the last inbound msg_id is the natural choice.
        msg_id = self._last_msg_id.get(chat_id)

        req = ApprovalRequest(
            session_key=session_key,
            title="Execute this command?",
            description=description,
            command_preview=command,
            timeout_sec=self._APPROVAL_TIMEOUT_SECONDS,
            allow_permanent=allow_permanent and not smart_denied,
        )
        return await self.send_approval_request(
            chat_id, req, reply_to=msg_id,
        )

    _APPROVAL_TIMEOUT_SECONDS = 300  # matches gateway's default gateway_timeout

    async def send_update_prompt(
            self,
            chat_id: str,
            prompt: str,
            default: str = "",
            session_key: str = "",
            metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a Yes/No update-confirmation prompt with inline buttons.

        Matches the cross-adapter contract used by
        ``gateway/run.py``'s ``hermes update --gateway`` watcher. Button
        clicks surface as ``INTERACTION_CREATE`` with
        ``button_data = 'update_prompt:y'`` or ``'update_prompt:n'``;
        the adapter's interaction callback writes the answer to
        ``~/.hermes/.update_response`` so the detached update process
        can read it.
        """
        del session_key, metadata  # present for contract parity only.

        default_hint = f" (default: {default})" if default else ""
        content = f"⚕ **Update Needs Your Input**\n\n{prompt}{default_hint}"
        msg_id = self._last_msg_id.get(chat_id)
        return await self.send_with_keyboard(
            chat_id,
            content,
            build_update_prompt_keyboard(),
            reply_to=msg_id,
        )

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
        """Upload media and send as a native message.

        Upload strategy:

        - **HTTP(S) URLs** → single ``POST /v2/{users|groups}/{id}/files``
          with ``url=...``. The QQ platform fetches the URL directly; fastest
          path when the source is already hosted.
        - **Local files** → three-step chunked upload (prepare / PUT parts /
          complete). Handles files up to the platform's ~100 MB per-file
          limit without the ~10 MB inline-base64 cap of the old adapter.
        """
        if not self.is_connected:
            if not await self._wait_for_reconnection():
                return SendResult(success=False, error="Not connected", retryable=True)

        chat_type = self._guess_chat_type(chat_id)
        if chat_type == "guild":
            # Guild channels don't support native media upload in the same way.
            return SendResult(
                success=False,
                error="Guild media send not supported via this path",
            )

        try:
            if self._is_url(media_source):
                # URL upload — let the platform fetch it directly.
                resolved_name = (
                    file_name
                    or Path(urlparse(media_source).path).name
                    or "media"
                )
                upload = await self._upload_media(
                    chat_type,
                    chat_id,
                    file_type,
                    url=media_source,
                    srv_send_msg=False,
                    file_name=resolved_name if file_type == MEDIA_TYPE_FILE else None,
                )
            else:
                # Local file — chunked upload (prepare / PUT parts / complete).
                resolved_name, upload = await self._upload_local_file(
                    chat_type,
                    chat_id,
                    media_source,
                    file_type,
                    file_name,
                )

            file_info = upload.get("file_info") or (
                upload.get("data", {}) or {}
            ).get("file_info")
            if not file_info:
                return SendResult(
                    success=False,
                    error=f"Upload returned no file_info: {upload}",
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
        except UploadDailyLimitExceededError as exc:
            # Non-retryable: daily quota hit. Give the caller actionable text
            # so the model can compose a helpful reply.
            logger.warning(
                "[%s] Daily upload limit exceeded for %s (%s)",
                self._log_tag, exc.file_name, exc.file_size_human,
            )
            return SendResult(
                success=False,
                error=(
                    f"QQ daily upload limit exceeded for {exc.file_name!r} "
                    f"({exc.file_size_human}). Retry tomorrow."
                ),
                retryable=False,
            )
        except UploadFileTooLargeError as exc:
            logger.warning(
                "[%s] File too large: %s (%s, platform limit %s)",
                self._log_tag, exc.file_name, exc.file_size_human, exc.limit_human,
            )
            return SendResult(
                success=False,
                error=(
                    f"{exc.file_name!r} ({exc.file_size_human}) exceeds the "
                    f"QQ per-file upload limit ({exc.limit_human})."
                ),
                retryable=False,
            )
        except Exception as exc:
            logger.error("[%s] Media send failed: %s", self._log_tag, exc)
            return SendResult(success=False, error=str(exc))

    async def _upload_local_file(
            self,
            chat_type: str,
            chat_id: str,
            media_source: str,
            file_type: int,
            file_name: Optional[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """Chunked-upload a local file and return ``(resolved_name, complete_response)``.

        The returned ``complete_response`` contains the ``file_info`` token
        that goes into the subsequent RichMedia message body.

        :raises UploadDailyLimitExceededError: On biz_code 40093002.
        :raises UploadFileTooLargeError: When the file exceeds the platform limit.
        :raises FileNotFoundError: If the path does not exist.
        :raises ValueError: If the path looks like a placeholder (``<path>``).
        :raises RuntimeError: If the HTTP client is not initialized.
        """
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized — not connected?")

        local_path = Path(media_source).expanduser()
        if not local_path.is_absolute():
            local_path = (Path.cwd() / local_path).resolve()

        if not local_path.exists() or not local_path.is_file():
            if media_source.startswith("<") or len(media_source) < 3:
                raise ValueError(
                    f"Invalid media source (looks like a placeholder): {media_source!r}"
                )
            raise FileNotFoundError(f"Media file not found: {local_path}")

        resolved_name = file_name or local_path.name
        uploader = ChunkedUploader(
            api_request=self._api_request,
            http_put=self._http_client.put,
            log_tag=self._log_tag,
        )
        complete = await uploader.upload(
            chat_type=chat_type,
            target_id=chat_id,
            file_path=str(local_path),
            file_type=file_type,
            file_name=resolved_name,
        )
        return resolved_name, complete

    async def _load_media(
            self, source: str, file_name: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """Load media from URL or local path. Returns (base64_or_url, content_type, filename)."""
        source = str(source).strip()
        if not source:
            raise ValueError("Media source is required")

        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
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
        logger.debug(
            "[%s] send_typing called: chat_id=%s connected=%s",
            self._log_tag,
            chat_id,
            self.is_connected,
        )

        if not self.is_connected:
            logger.debug(
                "[%s] send_typing skipped: adapter not connected, chat_id=%s",
                self._log_tag,
                chat_id,
            )
            return

        chat_type = self._guess_chat_type(chat_id)
        if chat_type != "c2c":
            logger.debug(
                "[%s] send_typing skipped: chat_type=%s (only c2c supported), chat_id=%s",
                self._log_tag,
                chat_type,
                chat_id,
            )
            return

        msg_id = self._last_msg_id.get(chat_id)
        if not msg_id:
            logger.debug(
                "[%s] send_typing skipped: no passive msg_id available, chat_id=%s "
                "(user has not sent a message yet or _last_msg_id was cleared)",
                self._log_tag,
                chat_id,
            )
            return

        # Debounce — skip if we sent recently
        now = time.time()
        last_sent = self._typing_sent_at.get(chat_id, 0.0)
        elapsed = now - last_sent
        if elapsed < self._TYPING_DEBOUNCE_SECONDS:
            logger.debug(
                "[%s] send_typing skipped: debounced, chat_id=%s elapsed=%.1fs "
                "debounce=%ds (last sent %.1fs ago, will refresh after debounce window)",
                self._log_tag,
                chat_id,
                elapsed,
                self._TYPING_DEBOUNCE_SECONDS,
                elapsed,
            )
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
            logger.debug(
                "[%s] send_typing sending input_notify: chat_id=%s msg_id=%s "
                "msg_seq=%s input_second=%d",
                self._log_tag,
                chat_id,
                msg_id,
                msg_seq,
                self._TYPING_INPUT_SECONDS,
            )
            await self._api_request("POST", f"/v2/users/{chat_id}/messages", body)
            self._typing_sent_at[chat_id] = now
            logger.debug(
                "[%s] send_typing succeeded: chat_id=%s msg_id=%s msg_seq=%s "
                "(indicator will auto-expire in %ds)",
                self._log_tag,
                chat_id,
                msg_id,
                msg_seq,
                self._TYPING_INPUT_SECONDS,
            )
        except Exception as exc:
            logger.debug(
                "[%s] send_typing failed: chat_id=%s msg_id=%s err=%s",
                self._log_tag,
                chat_id,
                msg_id,
                exc,
            )

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
            "type": "group" if chat_type in {"group", "guild"} else "dm",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_url(source: str) -> bool:
        return urlparse(str(source)).scheme in {"http", "https"}

    def _guess_chat_type(self, chat_id: str) -> str:
        """Determine chat type from stored inbound metadata, fallback to 'c2c'."""
        if chat_id in self._chat_type_map:
            return self._chat_type_map[chat_id]
        return "c2c"

    @staticmethod
    def _strip_at_mention(content: str) -> str:
        """Strip @bot mention markers from group message content.

        Handles both forms QQ may deliver:

        - the explicit ``<@!123>`` mention tag (full-push GROUP_MESSAGE_CREATE),
          aligning with openclaw ``MENTION_TAG_RE``;
        - the legacy leading ``@name `` prefix (GROUP_AT_MESSAGE_CREATE where the
          platform already rendered the tag as plain text).
        """
        import re

        # Remove explicit mention tags anywhere in the content.
        stripped = re.sub(r"<@!?\d+>", "", content).strip()
        # Remove a leading "@name " prefix if present.
        stripped = re.sub(r"^@\S+\s*", "", stripped)
        return stripped.strip()

    def _set_group_mode_override(
        self, group_openid: str, require_mention: bool
    ) -> None:
        """Set an in-memory activation-mode override for a single group.

        Reserved integration point for a future runtime mode-switch command
        (solution 2.2.3). It updates the highest-priority override consulted by
        :func:`resolve_require_mention`. A command handler would call this and
        then persist the change via ``cli.save_config_value`` — persistence and
        the command wiring are intentionally out of scope this round because a
        ``/``-command requires changes to the shared command framework.
        """
        if group_openid:
            self._group_mode_runtime_overrides[str(group_openid)] = bool(
                require_mention
            )

    def _open_dm_opted_in(self) -> bool:
        if os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}:
            return True
        return os.getenv("QQ_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}

    def _is_dm_allowed(self, user_id: str) -> bool:
        if self._dm_policy == "disabled":
            return False
        if self._dm_policy == "allowlist":
            return self._entry_matches(self._allow_from, user_id)
        if self._dm_policy == "open":
            return self._open_dm_opted_in()
        return False

    def _is_dm_intake_allowed(self, user_id: str) -> bool:
        principal = str(user_id or "").strip()
        if not principal:
            return False
        if self._dm_policy == "disabled":
            return False
        if self._dm_policy == "allowlist":
            return self._entry_matches(self._allow_from, principal)
        if self._dm_policy == "pairing":
            return True
        if self._dm_policy == "open":
            return self._open_dm_opted_in()
        return False

    def _is_group_allowed(self, group_id: str, user_id: str) -> bool:
        if self._group_policy == "disabled":
            return False
        if self._group_policy == "allowlist":
            return self._entry_matches(self._group_allow_from, group_id)
        if self._group_policy == "open":
            return True
        return False

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
