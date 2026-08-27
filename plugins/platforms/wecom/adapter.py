"""WeCom (Enterprise WeChat) AI Bot adapter over the ``openws`` WebSocket gateway.
Streaming lives in ``streaming.py``, media in ``media.py``, per-chat send queue in ``send_queue.py``.
Config (``platforms.wecom.extra``): ``bot_id``/``secret`` (or WECOM_BOT_ID / WECOM_SECRET), ``websocket_url``,
``dm_policy``/``group_policy`` (open|allowlist|disabled|pairing), ``allow_from``, ``group_allow_from``,
``groups: {<group_id>: {allow_from: [...]}}``."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]
try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]
AIOHTTP_AVAILABLE = aiohttp is not None
HTTPX_AVAILABLE = httpx is not None

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator, bounded_put
from gateway.platforms.access_policy_mixin import OwnAccessPolicyMixin
from gateway.platforms.base import gateway_trust_env, BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from utils import env_float

from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret, send_error
from plugins.platforms.wecom.send_queue import ChatSendQueueMixin
from plugins.platforms.wecom.media import WeComMediaMixin, APP_CMD_SEND
from plugins.platforms.wecom.streaming import (
    WeComStreamMixin, ReplyQueue, StreamTurn, APP_CMD_RESPONSE,
    STREAM_NOT_SUBSCRIBED_ERRCODE, MAX_STREAM_CONTENT_LENGTH,
    STREAM_SAFE_DURATION_SECONDS, STREAM_KEEPALIVE_INTERVAL_SECONDS, STREAM_KEEPALIVE_ENABLED_DEFAULT,
)


logger = logging.getLogger(__name__)

DEFAULT_WS_URL = "wss://openws.work.weixin.qq.com"

APP_CMD_SUBSCRIBE = "aibot_subscribe"
APP_CMD_CALLBACK = "aibot_msg_callback"
APP_CMD_LEGACY_CALLBACK = "aibot_callback"
APP_CMD_EVENT_CALLBACK = "aibot_event_callback"
APP_CMD_PING = "ping"

CALLBACK_COMMANDS = {APP_CMD_CALLBACK, APP_CMD_LEGACY_CALLBACK}
NON_RESPONSE_COMMANDS = CALLBACK_COMMANDS | {APP_CMD_EVENT_CALLBACK}

MAX_MESSAGE_LENGTH = 4000
CONNECT_TIMEOUT_SECONDS = 20.0
REQUEST_TIMEOUT_SECONDS = 15.0
HEARTBEAT_INTERVAL_SECONDS = 30.0
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]

DEDUP_MAX_SIZE = 1000

# Native streaming (msgtype: stream) constants — modeled on WeCom's official
# OpenClaw plugin behavior. WeCom's AI Bot supports cumulative stream frames
# via aibot_respond_msg; the first frame sends a <think></think> placeholder
# (matching the plugin's THINKING_MESSAGE) to signal a reasoning turn,
# subsequent frames push cumulative content, and a final frame with
# finish=true closes the stream.
STREAM_EXPIRED_ERRCODE = 846608  # >6 min without update — stream is dead
STREAM_REQUEST_EXPIRED_ERRCODE = 846604  # passive-reply request (req_id) itself
# expired — "websocket request expired, response is invalid". Sibling of 846608:
# 846608 is the stream update window, 846604 is the req_id reply channel window.
# Both mean the reply flow is dead and further finish=true / finish=false frames
# on it will be rejected — so keep-alive treats either as "stream expired".
STREAM_NOT_SUBSCRIBED_ERRCODE = 846609  # ws connection lost the subscription
STREAM_VERSION_CONFLICT_ERRCODE = 6000  # finalize raced a newer frame on the
# same stream_id — the bubble was ALREADY replaced by that newer version, so
# for a finalize frame this is benign (idempotent re-finalize hitting an
# already-updated bubble), NOT a delivery failure. See _send_stream_reply.
MAX_STREAM_CONTENT_LENGTH = 20480  # WeCom server-enforced byte limit per frame
# Per-turn cap on intermediate frames.  WeCom SDK has an internal 100-frame
# per-reqId queue limit; we cap at 85 (matching openclaw plugin) to guarantee
# room for the finalize frame.  Once hit, all further intermediate frames are
# silently dropped — finalize still sends unconditionally.
MAX_INTERMEDIATE_FRAMES = 85

# ── Stream-level keep-alive (aligned with wecom-openclaw-plugin PR #90) ──────
# WeCom binds a ~6-minute lifetime timer to each *reply stream* (stream_id +
# req_id).  The connection-level ping (_heartbeat_loop) does NOT refresh it —
# it only keeps the WS socket open.  Long turns (e.g. the daily-report cron,
# which spends minutes fetching data / rendering charts with no LLM tokens to
# push) let that window elapse, so the finalize frame lands on a dead stream
# and comes back 846604 / 846608 — a real delivery risk in group chats, which
# cannot fall back to a proactive send.  See docs/wecom-stream-keepalive-*.md.
#
# Two independent defences, both defaulting to the SAFE (non-aggressive) side:
#
#   Layer 2 — clock fallback (always on, zero new uplink frames):
#     finalize computes stream age from StreamTurn.start_time; past
#     STREAM_SAFE_DURATION_SECONDS it declines the finish=true frame (which
#     would almost certainly hit 846604/846608) and returns False so the
#     gateway consumer's existing fallback send() path delivers the content.
#
#   Layer 1 — keep-alive heartbeat (OFF by default; opt-in via config):
#     every STREAM_KEEPALIVE_INTERVAL_SECONDS re-send the already-accumulated
#     text as a finish=false frame to refresh the server window.  Never sends a
#     placeholder (that would pollute last_sent_content and could strand the
#     user on "still working…"); when there is no accumulated text yet the tick
#     is simply skipped.  Guarded off by default because a heartbeat frame is an
#     extra intermediate frame sharing the finalize's req_id, which widens the
#     ack race the double-send coordination depends on (see ANALYSIS §4.2).
STREAM_SAFE_DURATION_SECONDS = 330.0  # 5.5 min — Layer 2 clock fallback
STREAM_KEEPALIVE_INTERVAL_SECONDS = 120.0  # 2 min — Layer 1 heartbeat cadence
STREAM_KEEPALIVE_ENABLED_DEFAULT = False  # Layer 1 off unless config opts in

IMAGE_MAX_BYTES = 10 * 1024 * 1024
VIDEO_MAX_BYTES = 10 * 1024 * 1024
VOICE_MAX_BYTES = 2 * 1024 * 1024
FILE_MAX_BYTES = 20 * 1024 * 1024
ABSOLUTE_MAX_BYTES = FILE_MAX_BYTES
UPLOAD_CHUNK_SIZE = 512 * 1024
MAX_UPLOAD_CHUNKS = 100
VOICE_SUPPORTED_MIMES = {"audio/amr"}


def check_wecom_requirements() -> bool:
    return AIOHTTP_AVAILABLE and HTTPX_AVAILABLE


def _coerce_list(value: Any) -> List[str]:
    """Coerce config values (None | "a, b" | iterable | scalar) into a trimmed, non-empty string list."""
    if isinstance(value, str):
        value = value.split(",")
    elif not isinstance(value, (list, tuple, set)):
        value = [] if value is None else [value]
    return [item for item in (str(item).strip() for item in value) if item]


def _normalize_entry(raw: str) -> str:
    """Normalize allowlist entries such as ``wecom:user:foo``."""
    value = re.sub(r"^wecom:", "", str(raw).strip(), flags=re.IGNORECASE)
    return re.sub(r"^(user|group):", "", value, flags=re.IGNORECASE).strip()


def _entry_matches(entries: List[str], target: str) -> bool:
    """Case-insensitive allowlist match with ``*`` support."""
    normalized_target = str(target).strip().lower()
    return any(_normalize_entry(e).lower() in ("*", normalized_target) for e in entries)


class WeComStreamExpiredError(RuntimeError):
    """Raised when WeCom returns errcode 846608 or 846604 (stream/req expired).

    WeCom's stream protocol caps a stream session at ~6 minutes from the
    first frame. After that window the server refuses further updates with
    846608 (stream update window) or 846604 (req_id reply-request window) and
    the reply flow is dead — callers must fall back to a proactive
    ``aibot_send_msg`` to deliver the remaining content.
    """

    def __init__(self, errcode: int = STREAM_EXPIRED_ERRCODE, errmsg: str = ""):
        super().__init__(
            f"WeCom stream expired (errcode={errcode}): {errmsg or 'no detail'}"
        )
        self.errcode = errcode
        self.errmsg = errmsg


@dataclass
class ReplyFrame:
    """A queued reply frame waiting to be sent via aibot_respond_msg.

    Used for ack tracking and FIFO ordering per req_id, aligning with
    the official WeCom SDK's replyStreamNonBlocking semantics.
    """
    body: Dict[str, Any]
    future: asyncio.Future
    is_final: bool = False
    sent_at: Optional[float] = None


class ReplyQueue:
    """Per-req_id pending ack tracker.

    Ensures:
    - Intermediate frames skip if a previous frame's ack is pending
    - Final frames wait for pending ack before sending

    Aligned with official SDK's replyStreamNonBlocking + 5s ack timeout.
    """
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.pending_ack: Optional[ReplyFrame] = None



class StreamTurn:
    """Per-turn stream state to avoid global state conflicts.

    Each inbound message creates its own StreamTurn, ensuring concurrent
    messages don't interfere with each other's stream state.
    """
    def __init__(self, chat_id: str, req_id: str):
        self.chat_id = chat_id
        self.req_id = req_id
        self.stream_id = f"stream_{uuid.uuid4().hex[:12]}"
        self.accumulated_text = ""
        self.finalized = False
        self.seeded = False  # True after seed frame sent (prevents double seed)
        self.start_time = time.monotonic()
        self.expired = False
        # Track the last content that was ACTUALLY sent to WeCom (not skipped).
        # Used by finalize to detect duplicate content and avoid silent ack drops.
        self.last_sent_content: str = ""
        # Per-turn intermediate-frame counter (count-based cap at
        # MAX_INTERMEDIATE_FRAMES to leave room for the finalize frame).
        self._last_frame_sent_at: float = 0.0
        self._intermediate_frames_sent: int = 0
        # Idle flush handle — retained for _cancel_idle_flush() compatibility
        # (called in finalize/boundary paths; always None in fire-and-forget).
        self.idle_flush_handle: Optional[asyncio.TimerHandle] = None
        # Keep-alive handle (Layer 1) — set when the stream-level keep-alive
        # timer is armed.  Structurally identical to idle_flush_handle: a
        # per-turn asyncio TimerHandle that MUST be cancelled on every turn
        # exit path (finalize / expired / error / cleanup) to avoid a leaked
        # timer firing on a dead turn.  None when keep-alive is disabled or
        # the turn has no armed timer.
        self.keepalive_handle: Optional[asyncio.TimerHandle] = None


class WeComAdapter(BasePlatformAdapter):
    """WeCom AI Bot adapter backed by a persistent WebSocket connection."""

    ALLOW_ALL_ENV_PREFIX = "WECOM"
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    SUPPORTS_MESSAGE_EDITING = False
    # WeCom AI Bot supports msgtype: "stream" via aibot_respond_msg, which
    # the gateway streaming consumer treats as a transport that bypasses the
    # edit-based path. See ``send_stream_frame`` and ``supports_native_streaming``.
    SUPPORTS_NATIVE_STREAMING = True
    MAX_STREAM_CONTENT_LENGTH = MAX_STREAM_CONTENT_LENGTH
    # Threshold for detecting WeCom client-side message splits.
    # When a chunk is near the 4000-char limit, a continuation is almost certain.
    _SPLIT_THRESHOLD = 3900

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WECOM)
        extra = config.extra or {}

        self._dm_policy = str(extra.get("dm_policy") or _get_scoped_secret("WECOM_DM_POLICY", "pairing")).strip().lower()
        # dm_policy already honors WECOM_DM_POLICY, so the allowlist must honor
        # WECOM_ALLOWED_USERS too. Without the env fallback an env-only setup
        # (dm_policy=allowlist via env, no config extra) runs with an empty
        # allowlist and drops every authorized DM at intake.
        self._allow_from = _coerce_list(
            extra.get("allow_from")
            or extra.get("allowFrom")
            or _get_scoped_secret("WECOM_ALLOWED_USERS", "")
        )

        self._group_policy = str(extra.get("group_policy") or _get_scoped_secret("WECOM_GROUP_POLICY", "pairing")).strip().lower()
        self._group_allow_from = _coerce_list(extra.get("group_allow_from") or extra.get("groupAllowFrom"))
        self._groups = extra.get("groups") if isinstance(extra.get("groups"), dict) else {}
        self._session = self._ws = self._http_client = self._listen_task = self._heartbeat_task = None
        self._pending_responses: Dict[str, asyncio.Future] = {}
        # Per-req_id reply queue with ack tracking — aligns with official
        # SDK's replyStreamNonBlocking (skip if pending, wait before final).
        self._reply_queues: Dict[str, ReplyQueue] = {}
        self._dedup = MessageDeduplicator(max_size=DEDUP_MAX_SIZE)
        self._reply_req_ids: Dict[str, str] = {}

        # Text batching: merge rapid successive messages (Telegram-style).
        # WeCom clients split long messages around 4000 chars.
        self._text_batch_delay_seconds = env_float("HERMES_WECOM_TEXT_BATCH_DELAY_SECONDS", 0.6)
        self._text_batch_split_delay_seconds = env_float("HERMES_WECOM_TEXT_BATCH_SPLIT_DELAY_SECONDS", 2.0)
        # Attachment/text merge window: WeCom clients send "image + text" as two
        # separate inbound callbacks (one attachment-only, one text) a few
        # hundred ms apart. Holding an attachment-only message for this window
        # lets the following text merge into the SAME event, so it dispatches
        # as one turn instead of the attachment spawning a run that the text
        # then "interrupts" (mirrors the official plugin's
        # ATTACHMENT_TEXT_MERGE_WINDOW_MS = 800). Behavioral config lives in
        # config.extra, not an env var.
        try:
            self._attachment_text_merge_delay_seconds = float(
                extra.get("attachment_text_merge_delay_seconds", 0.8)
            )
        except (TypeError, ValueError):
            self._attachment_text_merge_delay_seconds = 0.8
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}

        # ── Stream-level keep-alive config (config.extra, not env) ──────────
        # Behavioral config lives in config.extra, matching
        # attachment_text_merge_delay_seconds above.  See the module-level
        # STREAM_* constants and docs/wecom-stream-keepalive-ANALYSIS.md.
        def _extra_float(key: str, default: float) -> float:
            try:
                return float(extra.get(key, default))
            except (TypeError, ValueError):
                return default

        # Layer 2 clock fallback: decline the finalize frame once the stream is
        # older than this, so we don't hand the server a doomed finish=true.
        self._stream_safe_duration_seconds = _extra_float(
            "stream_safe_duration_seconds", STREAM_SAFE_DURATION_SECONDS
        )
        # Layer 1 heartbeat: off unless config opts in (see ANALYSIS §4.2/§5).
        self._stream_keepalive_enabled = bool(
            extra.get("stream_keepalive_enabled", STREAM_KEEPALIVE_ENABLED_DEFAULT)
        )
        self._stream_keepalive_interval_seconds = _extra_float(
            "stream_keepalive_interval_seconds", STREAM_KEEPALIVE_INTERVAL_SECONDS
        )

        self._device_id = uuid.uuid4().hex
        self._last_chat_req_ids: Dict[str, str] = {}
        # Turns keyed f"{chat_id}:{req_id|turn_id}"; expired chats clear on the next inbound req_id.
        self._stream_turns: Dict[str, StreamTurn] = {}
        self._stream_expired_chats, self._group_chat_ids = set(), set()  # groups can't receive proactive APP_CMD_SEND
        # Per-chat FIFO send queues (normal + control lanes) + token buckets — see send_queue.py.
        self._chat_queues, self._chat_workers, self._control_queues, self._control_workers, self._chat_token_usage = {}, {}, {}, {}, {}

        # Per-turn stream state: keyed by (chat_id, req_id) to support concurrent messages.
        # Replaces global _active_stream_id to avoid conflicts when multiple messages
        # are processed simultaneously (e.g., approval during streaming).
        self._stream_turns: Dict[str, StreamTurn] = {}  # key = f"{chat_id}:{req_id}"

        # Chats whose stream session has been retired (846608 / 846609 / no
        # req_id). Cleared whenever a fresh inbound callback for the chat
        # arrives — a new inbound message gives us a new req_id and the
        # stream channel becomes usable again.
        self._stream_expired_chats: set[str] = set()

        # Track which chat_ids are group chats. Populated in _on_message
        # when chattype=="group". Used by _send_inner to avoid APP_CMD_SEND
        # for groups (WeCom AI Bots cannot initiate proactive sends in groups).
        self._group_chat_ids: set[str] = set()

        # Per-chat FIFO send queue with token-bucket rate limiting.
        # Mirrors OpenClaw's chat-queue.ts (serial per chat) plus a
        # token bucket to stay within WeCom's 30 msgs/min/chat limit.
        self._chat_queues: Dict[str, asyncio.Queue] = {}
        self._chat_workers: Dict[str, asyncio.Task] = {}

        # Control lane: high-priority queue for approval prompts, finalize frames,
        # and error notifications. These bypass normal queue to prevent blocking.
        self._control_queues: Dict[str, asyncio.Queue] = {}
        self._control_workers: Dict[str, asyncio.Task] = {}

        # Token bucket with reserved tokens for control messages.
        # Per-chat usage tracking: {chat_id: {"normal": used, "reserved": used, "last_reset": ts}}
        self._chat_token_usage: Dict[str, Dict[str, float]] = {}

    # Token bucket parameters: 30 tokens max per minute, split between normal and reserved.
    _BUCKET_MAX_TOKENS = 30
    _BUCKET_NORMAL_TOKENS = 24      # For normal messages
    _BUCKET_RESERVED_TOKENS = 6     # Reserved for control lane (approval, finalize, errors)

    def _get_token_usage(self, chat_id: str) -> Dict[str, float]:
        """Get or create token usage tracking for a chat."""
        key = str(chat_id or "").strip()
        if key not in self._chat_token_usage:
            self._chat_token_usage[key] = {
                "normal": 0.0,
                "reserved": 0.0,
                "last_reset": time.monotonic(),
            }
        return self._chat_token_usage[key]

    def _bucket_try_consume(self, chat_id: str) -> float:
        """Try to consume a normal token. Returns 0 if available, or seconds to wait."""
        usage = self._get_token_usage(chat_id)
        now = time.monotonic()

        # Reset counters every minute
        if now - usage["last_reset"] > 60.0:
            usage["normal"] = 0.0
            usage["reserved"] = 0.0
            usage["last_reset"] = now

        # Normal messages can only use normal quota
        if usage["normal"] < self._BUCKET_NORMAL_TOKENS:
            usage["normal"] += 1.0
            return 0.0  # token available, no wait
        else:
            # Wait until next minute
            return 60.0 - (now - usage["last_reset"])

    def _bucket_try_consume_control(self, chat_id: str) -> float:
        """Try to consume a control token. Can use normal remaining + reserved pool."""
        usage = self._get_token_usage(chat_id)
        now = time.monotonic()

        # Reset counters every minute
        if now - usage["last_reset"] > 60.0:
            usage["normal"] = 0.0
            usage["reserved"] = 0.0
            usage["last_reset"] = now

        # Control messages prefer normal quota first (don't waste reserved)
        normal_available = self._BUCKET_NORMAL_TOKENS - usage["normal"]
        if normal_available > 0:
            usage["normal"] += 1.0
            return 0.0

        # Normal exhausted, use reserved pool
        reserved_available = self._BUCKET_RESERVED_TOKENS - usage["reserved"]
        if reserved_available > 0:
            usage["reserved"] += 1.0
            return 0.0

        # Both exhausted, wait until next minute
        return 60.0 - (now - usage["last_reset"])

    async def _enqueue_chat_send(self, chat_id: str, coro_factory, is_control: bool = False):
        """Enqueue a send task for a chat and await its result.

        FIFO per chat, parallel across chats. Two lanes:
        - Control lane: approval prompts, finalize frames, errors (uses reserved tokens)
        - Normal lane: regular messages (uses normal tokens only)

        Control messages bypass normal queue to prevent approval prompt blocking.
        """
        key = str(chat_id or "").strip()

        if is_control:
            # Control lane: high priority, reserved token pool
            if key not in self._control_queues:
                logger.debug(
                    "[%s] Creating control queue + worker for chat %s",
                    self.name, key,
                )
                self._control_queues[key] = asyncio.Queue()
                self._control_workers[key] = asyncio.create_task(
                    self._control_send_worker(key)
                )
            queue = self._control_queues[key]
        else:
            # Normal lane
            if key not in self._chat_queues:
                logger.debug(
                    "[%s] Creating normal queue + worker for chat %s",
                    self.name, key,
                )
                self._chat_queues[key] = asyncio.Queue()
                self._chat_workers[key] = asyncio.create_task(
                    self._chat_send_worker(key)
                )
            queue = self._chat_queues[key]

        logger.debug(
            "[%s] Enqueuing send for chat %s (lane=%s, qsize=%d)",
            self.name, key, "control" if is_control else "normal", queue.qsize(),
        )
        future = asyncio.get_running_loop().create_future()
        await queue.put((coro_factory, future))
        return await future

    async def _chat_send_worker(self, chat_key: str) -> None:
        """Per-chat worker: drain normal queue with token-bucket rate limiting."""
        queue = self._chat_queues[chat_key]
        logger.debug("[%s] Normal send worker started for chat %s", self.name, chat_key)
        try:
            while True:
                coro_factory, future = await queue.get()
                try:
                    # Token bucket: wait only if bucket is empty
                    wait = self._bucket_try_consume(chat_key)
                    if wait > 0:
                        logger.debug(
                            "[%s] Normal worker rate-limited for chat %s, waiting %.1fs",
                            self.name, chat_key, wait,
                        )
                        await asyncio.sleep(wait)
                        # Re-consume after wait
                        self._bucket_try_consume(chat_key)

                    result = await coro_factory()
                    if not future.done():
                        future.set_result(result)
                except Exception as exc:
                    if not future.done():
                        future.set_exception(exc)
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            while not queue.empty():
                try:
                    _, future = queue.get_nowait()
                    if not future.done():
                        future.set_exception(
                            RuntimeError("WeCom adapter shutting down")
                        )
                except asyncio.QueueEmpty:
                    break

    async def _control_send_worker(self, chat_key: str) -> None:
        """Control lane worker: drain control queue with reserved token pool."""
        queue = self._control_queues[chat_key]
        try:
            while True:
                coro_factory, future = await queue.get()
                try:
                    # Control messages use reserved + normal remaining tokens
                    wait = self._bucket_try_consume_control(chat_key)
                    if wait > 0:
                        await asyncio.sleep(wait)
                        self._bucket_try_consume_control(chat_key)

                    result = await coro_factory()
                    if not future.done():
                        future.set_result(result)
                except Exception as exc:
                    if not future.done():
                        future.set_exception(exc)
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            while not queue.empty():
                try:
                    _, future = queue.get_nowait()
                    if not future.done():
                        future.set_exception(
                            RuntimeError("WeCom adapter shutting down")
                        )
                except asyncio.QueueEmpty:
                    break

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        for available, dep in ((AIOHTTP_AVAILABLE, "aiohttp"), (HTTPX_AVAILABLE, "httpx")):
            if not available:
                return self._startup_failure("wecom_missing_dependency", f"WeCom startup failed: {dep} not installed", "[%s] %s. Run: pip install %s", dep)
        if not self._bot_id or not self._secret:
            return self._startup_failure("wecom_missing_credentials", "WeCom startup failed: WECOM_BOT_ID and WECOM_SECRET are required", "[%s] %s")
        try:
            # Tighter keepalive so idle CLOSE_WAIT drains promptly.
            # See #18451.
            from gateway.platforms._http_client_limits import platform_httpx_limits
            from gateway.platforms.base import _ssrf_redirect_guard
            from tools.url_safety import create_ssrf_safe_async_client
            self._http_client = create_ssrf_safe_async_client(timeout=30.0, follow_redirects=True, event_hooks={"response": [_ssrf_redirect_guard]}, limits=platform_httpx_limits())
            await self._open_connection()
            self._mark_connected()
            self._listen_task, self._heartbeat_task = asyncio.create_task(self._listen_loop()), asyncio.create_task(self._heartbeat_loop())
            logger.info("[%s] Connected to %s", self.name, self._ws_url)
            # Plugin-registered native handlers (ctx.register_platform_handler).
            self._wire_plugin_handlers(None)
            return True
        except Exception as exc:
            self._set_fatal_error("wecom_connect_error", f"WeCom startup failed: {exc}", retryable=True)
            logger.error("[%s] Failed to connect: %s", self.name, exc, exc_info=True)
            await self._teardown()
            return False

    async def disconnect(self) -> None:
        self._running = False
        self._mark_disconnected()
        # Force-close any lingering stream so the WeCom client doesn't show
        # a permanent typing bubble after the gateway goes down.
        self._reset_native_stream_state()

        # Cancel per-chat send workers (normal + control lanes) so queued tasks get cleaned up.
        for task in list(self._chat_workers.values()) + list(self._control_workers.values()):
            task.cancel()
        self._chat_workers.clear()
        self._control_workers.clear()
        self._chat_queues.clear()
        self._control_queues.clear()

    def _fail_all(self, exc: Exception) -> None:
        self._fail_pending_responses(exc)
        self._fail_reply_queues(exc)

    async def _cleanup_ws(self) -> None:
        """Close the live websocket, then its session, if any."""
        for attr in ("_ws", "_session"):
            live = getattr(self, attr)
            if live and not live.closed:
                await live.close()
            setattr(self, attr, None)

        self._fail_pending_responses(RuntimeError("WeCom adapter disconnected"))
        self._fail_reply_queues(RuntimeError("WeCom adapter disconnected"))
        await self._cleanup_ws()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _open_connection(self) -> None:
        await self._cleanup_ws()
        # Use certifi's CA bundle so aiohttp trusts the same roots as
        # urllib/requests — avoids SSL_CERTIFICATE_VERIFY_FAILED on macOS
        # where the OpenSSL default path may be empty or stale.
        import ssl as _ssl
        try:
            import certifi
            _ssl_ctx = _ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            _ssl_ctx = _ssl.create_default_context()
        _connector = aiohttp.TCPConnector(ssl=_ssl_ctx)
        self._session = aiohttp.ClientSession(trust_env=True, connector=_connector)
        self._ws = await self._session.ws_connect(
            self._ws_url,
            heartbeat=HEARTBEAT_INTERVAL_SECONDS * 2,
            timeout=CONNECT_TIMEOUT_SECONDS,
        )

        req_id = self._new_req_id("subscribe")
        await self._send_json({"cmd": APP_CMD_SUBSCRIBE, "headers": {"req_id": req_id}, "body": {"bot_id": self._bot_id, "secret": self._secret, "device_id": self._device_id}})
        auth_payload = await self._wait_for_handshake(req_id)
        errcode = auth_payload.get("errcode", 0)
        if errcode not in {0, None}:
            raise RuntimeError(f"{auth_payload.get('errmsg', 'authentication failed')} (errcode={errcode})")

    async def _wait_for_handshake(self, req_id: str) -> Dict[str, Any]:
        if not self._ws:
            raise RuntimeError("WebSocket not initialized")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + CONNECT_TIMEOUT_SECONDS
        while (remaining := deadline - loop.time()) > 0:
            msg = await asyncio.wait_for(self._ws.receive(), timeout=remaining)
            if msg.type == aiohttp.WSMsgType.TEXT:
                payload = self._parse_json(msg.data)
                if not payload or payload.get("cmd") == APP_CMD_PING:
                    continue
                if self._payload_req_id(payload) == req_id:
                    return payload
                logger.debug("[%s] Ignoring pre-auth payload: %s", self.name, payload.get("cmd"))
            elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR}:
                raise RuntimeError("WeCom websocket closed during authentication")
        raise TimeoutError("Timed out waiting for WeCom subscribe acknowledgement")

    async def _listen_loop(self) -> None:
        backoff_idx = 0
        while self._running:
            try:
                await self._read_events()
                backoff_idx = 0
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if not self._running:
                    return
                logger.warning("[%s] WebSocket error: %s", self.name, exc)
                self._fail_pending_responses(RuntimeError("WeCom connection interrupted"))
                self._fail_reply_queues(RuntimeError("WeCom connection interrupted"))

                delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
                backoff_idx += 1
                try:
                    await self._open_connection()
                    backoff_idx = 0
                    self._mark_connected()
                    logger.info("[%s] Reconnected", self.name)
                except Exception as reconnect_exc:
                    logger.warning("[%s] Reconnect failed: %s", self.name, reconnect_exc)

    async def _read_events(self) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        while self._running and self._ws and not self._ws.closed:
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                payload = self._parse_json(msg.data)
                if payload:
                    await self._dispatch_payload(payload)
                else:
                    # Parse returned None (JSON decode failed or non-dict
                    # payload). _parse_json already logs the failure detail;
                    # this makes the DROP itself visible at INFO so we can
                    # correlate a missing inbound message with a bad frame.
                    logger.info(
                        "[%s] Inbound TEXT frame dropped (unparseable/non-dict) len=%d",
                        self.name,
                        len(msg.data) if isinstance(msg.data, (str, bytes)) else -1,
                    )
            elif msg.type == aiohttp.WSMsgType.BINARY:
                # WeCom is expected to send TEXT frames; a BINARY frame is
                # unexpected. Log at INFO with a decoded preview so we can
                # tell whether group messages are arriving in an unhandled
                # transport instead of being silently discarded.
                try:
                    decoded = msg.data.decode("utf-8", errors="replace")
                except Exception:
                    decoded = "<undecodable>"
                logger.info(
                    "[%s] Inbound BINARY frame received (len=%d) head=%r — attempting JSON parse",
                    self.name,
                    len(msg.data) if isinstance(msg.data, (bytes, bytearray)) else -1,
                    decoded[:200],
                )
                payload = self._parse_json(msg.data)
                if payload:
                    await self._dispatch_payload(payload)
                else:
                    logger.info("[%s] BINARY frame not parseable as JSON — dropped", self.name)
            elif msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSING}:
                raise RuntimeError("WeCom websocket closed")
            else:
                logger.info("[%s] Inbound frame ignored: WSMsgType=%s", self.name, msg.type)

    async def _heartbeat_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                try:
                    if self._ws and not self._ws.closed:
                        await self._send_json({"cmd": APP_CMD_PING, "headers": {"req_id": self._new_req_id("ping")}, "body": {}})
                except Exception as exc:
                    logger.debug("[%s] Heartbeat send failed: %s", self.name, exc)
        except asyncio.CancelledError:
            pass

    async def _dispatch_payload(self, payload: Dict[str, Any]) -> None:
        req_id = self._payload_req_id(payload)
        cmd = str(payload.get("cmd") or "")

        # --- Diagnostic: log ALL non-ping inbound payloads when any reply queue
        # is active, to detect whether WeCom acks arrive at all.
        if self._reply_queues and cmd != APP_CMD_PING:
            logger.debug(
                "[%s] _dispatch_payload[ALL]: req_id=%s cmd=%r active_queues=%s",
                self.name, req_id or "(none)", cmd or "(empty)",
                list(self._reply_queues.keys()),
            )

        # --- Diagnostic: log all payloads that carry a req_id matching an
        # active reply queue, regardless of whether they get routed there.
        # This helps diagnose ack timeout issues (e.g., ack arriving with
        # unexpected cmd that gets filtered out).
        if req_id and self._reply_queues.get(req_id):
            queue = self._reply_queues[req_id]
            has_pending = queue.pending_ack is not None
            logger.debug(
                "[%s] _dispatch_payload: req_id=%s cmd=%r has_pending_ack=%s "
                "errcode=%s in_NON_RESPONSE=%s payload_keys=%s",
                self.name, req_id, cmd, has_pending,
                payload.get("body", {}).get("errcode", "N/A") if isinstance(payload.get("body"), dict) else "N/A",
                cmd in NON_RESPONSE_COMMANDS,
                list(payload.keys()),
            )

        # Check reply queue ack first — aibot_respond_msg acks arrive with
        # the original inbound req_id and no cmd (or non-callback cmd).
        # This must be checked before _pending_responses to avoid the old
        # _send_reply_request path stealing acks meant for the queue.
        if req_id and cmd not in NON_RESPONSE_COMMANDS:
            if self._resolve_reply_ack(req_id, payload):
                return

        if req_id and req_id in self._pending_responses and cmd not in NON_RESPONSE_COMMANDS:
            future = self._pending_responses.get(req_id)
            if future and not future.done():
                future.set_result(payload)
            return

        if cmd in CALLBACK_COMMANDS:
            await self._on_message(payload)
            return
        if cmd == APP_CMD_PING:
            return
        if cmd == APP_CMD_EVENT_CALLBACK:
            # Check for "kicked by server" event — WeCom sends this when a new
            # connection is established elsewhere (another instance). Mirror the
            # official OpenClaw SDK: suppress reconnect to avoid mutual kicking.
            body = payload.get("body") or {}
            event_type = str(body.get("event_type") or "")
            if event_type == "disconnected_event":
                logger.warning(
                    "[%s] Kicked by server (another WS connection established). "
                    "Suppressing reconnect to avoid mutual kicking. "
                    "Check for duplicate gateway instances.",
                    self.name,
                )
                self._running = False  # stop _listen_loop from reconnecting
            return

        # Unrouted payload — did not match reply-queue, pending-response,
        # callback, ping, or event. If WeCom delivers group messages under a
        # cmd not in CALLBACK_COMMANDS, they land here and are dropped. Log at
        # INFO with cmd + body keys so we can spot an unhandled callback cmd.
        body_keys = list(payload.get("body", {}).keys()) if isinstance(payload.get("body"), dict) else None
        logger.info(
            "[%s] Unrouted websocket payload dropped: cmd=%r req_id=%s body_keys=%s",
            self.name, cmd or "(empty)", req_id or "(none)", body_keys,
        )

    def _fail_pending_responses(self, exc: Exception) -> None:
        for req_id, future in list(self._pending_responses.items()):
            if not future.done():
                future.set_exception(exc)
            self._pending_responses.pop(req_id, None)

    def _require_ws(self) -> None:
        if not self._ws or self._ws.closed:
            raise RuntimeError("WeCom websocket is not connected")

    async def _send_json(self, payload: Dict[str, Any]) -> None:
        self._require_ws()
        await self._ws.send_json(payload)

    async def _request(self, cmd: str, req_id: str, body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        future = self._pending_responses[req_id] = asyncio.get_running_loop().create_future()
        try:
            await self._send_json({"cmd": cmd, "headers": {"req_id": req_id}, "body": body})
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_responses.pop(req_id, None)

    async def _send_request(self, cmd: str, body: Dict[str, Any], timeout: float = REQUEST_TIMEOUT_SECONDS) -> Dict[str, Any]:
        self._require_ws()
        return await self._request(cmd, self._new_req_id(cmd), body, timeout)

    async def _send_reply_request(self, reply_req_id: str, body: Dict[str, Any], cmd: str = APP_CMD_RESPONSE, timeout: float = REQUEST_TIMEOUT_SECONDS) -> Dict[str, Any]:
        """Send a reply frame correlated to an inbound callback req_id."""
        self._require_ws()
        return await self._request(cmd, self._require_reply_req_id(reply_req_id), body, timeout)

    @staticmethod
    def _require_reply_req_id(reply_req_id: str) -> str:
        normalized = str(reply_req_id or "").strip()
        if not normalized:
            raise ValueError("reply_req_id is required")
        return normalized

    # ── Per-req_id Reply Queue (ack tracking) ────────────────────────────
    # Aligns with official SDK replyStreamNonBlocking:
    #   - intermediate frame: skip if pending ack on this req_id
    #   - final frame: wait for pending ack to drain before sending
    #   - ack timeout: 15 seconds
    #
    # Matches the official @wecom/wecom-openclaw-plugin's REPLY_SEND_TIMEOUT_MS
    # = 15_000. The prior 5s value was too aggressive for the bilibili WeCom
    # environment where ack > 5s is not rare on long replies (server-side queue
    # lag, WS jitter, concurrent replies on the same WS). A short window widens
    # the race where the final-frame ack is still in flight while the gateway's
    # normal final-send fires, producing duplicate messages
    # (see docs/rca-wecom-stream-final-ack-timeout-duplicate.md).

    _REPLY_ACK_TIMEOUT = 15.0

    async def _send_reply_queued(
        self,
        reply_req_id: str,
        body: Dict[str, Any],
        *,
        is_final: bool = False,
        skip_if_pending: bool = False,
    ) -> Dict[str, Any]:
        """Send a reply via aibot_respond_msg with per-req_id ack tracking.

        Args:
            reply_req_id: The inbound callback req_id to reply to.
            body: Reply body (msgtype: stream/markdown/...).
            is_final: If True, wait for any pending ack before sending.
            skip_if_pending: If True and a previous frame's ack is pending,
                return immediately with {"skipped": True}.

        Returns:
            Response dict from WeCom, or {"skipped": True} if skipped.
        """
        if not self._ws or self._ws.closed:
            raise RuntimeError("WeCom websocket is not connected")

        normalized = str(reply_req_id or "").strip()
        if not normalized:
            raise ValueError("reply_req_id is required")

        queue = self._reply_queues.get(normalized)
        if queue is None:
            queue = ReplyQueue(normalized)
            self._reply_queues[normalized] = queue

        # NonBlocking semantics: skip if a prior frame ack is pending
        if skip_if_pending and queue.pending_ack is not None:
            return {"skipped": True, "errcode": 0, "errmsg": "pending_ack"}

        # Final frame: wait for pending ack to drain first
        if is_final and queue.pending_ack is not None:
            pending_frame = queue.pending_ack
            _pending_stream = pending_frame.body.get("stream", {}) if isinstance(pending_frame.body.get("stream"), dict) else {}
            logger.debug(
                "[%s] _send_reply_queued: final waiting for pending ack drain — "
                "req_id=%s pending_stream_id=%s pending_finish=%s pending_sent_at=%.1fs_ago",
                self.name, normalized,
                _pending_stream.get("id", "N/A"),
                _pending_stream.get("finish", "N/A"),
                time.monotonic() - (pending_frame.sent_at or time.monotonic()),
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(pending_frame.future),
                    timeout=self._REPLY_ACK_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] Reply ack timeout waiting for pending (req_id=%s) — "
                    "pending_stream_id=%s pending_finish=%s elapsed=%.1fs. "
                    "Possible causes: ack cmd filtered, ack req_id mismatch, or WeCom did not ack.",
                    self.name, normalized,
                    _pending_stream.get("id", "N/A"),
                    _pending_stream.get("finish", "N/A"),
                    time.monotonic() - (pending_frame.sent_at or time.monotonic()),
                )
            except Exception:
                pass
            # Clear pending regardless — either resolved or timed out
            queue.pending_ack = None

        # Create future for THIS frame's ack
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        frame = ReplyFrame(body=body, future=future, is_final=is_final)
        frame.sent_at = time.monotonic()

        # Register as pending BEFORE sending to avoid race:
        # If WeCom acks during _send_json await, _dispatch_payload needs
        # to find the pending frame to resolve it. Registering after would
        # miss the ack and timeout.
        #
        # Fix (orphan-queue race): re-attach `queue` to the dict before
        # registering pending_ack. A final frame shares the inbound req_id
        # with the intermediate frames; while it awaits the pending
        # intermediate ack to drain (the `is_final` branch above yields at
        # `await`), that intermediate ack can arrive and _resolve_reply_ack
        # pops the WHOLE queue out of self._reply_queues (the "cleanup empty
        # queue" pop). The local `queue` reference captured at the top is then
        # an ORPHAN — detached from the dict — so registering pending_ack on it
        # is invisible to _dispatch_payload, and the final frame's own ack
        # lands in Unrouted → 15s timeout. Writing the reference back here
        # closes that window: the final frame's ack can always be routed.
        self._reply_queues[normalized] = queue
        queue.pending_ack = frame

        # Diagnostic: log every frame send for ack tracking analysis
        _stream_info = body.get("stream", {}) if isinstance(body.get("stream"), dict) else {}
        logger.debug(
            "[%s] _send_reply_queued: req_id=%s is_final=%s skip_if_pending=%s "
            "stream_id=%s finish=%s content_len=%d",
            self.name, normalized, is_final, skip_if_pending,
            _stream_info.get("id", "N/A"),
            _stream_info.get("finish", "N/A"),
            len(_stream_info.get("content", "") or ""),
        )

        # Send the frame
        try:
            await self._send_json(
                {"cmd": APP_CMD_RESPONSE, "headers": {"req_id": normalized}, "body": body}
            )
        except Exception as e:
            # Send failed — clear pending and reject future. The future has
            # no awaiter on the send-failure branch (we re-raise immediately),
            # so cancel it instead of setting an exception that would otherwise
            # be logged as "Future exception was never retrieved".
            if queue.pending_ack is frame:
                queue.pending_ack = None
                if not self._reply_queues.get(normalized) or queue.pending_ack is None:
                    self._reply_queues.pop(normalized, None)
            if not future.done():
                future.cancel()
            raise

        # For final frames: await the ack (blocking)
        if is_final:
            try:
                response = await asyncio.wait_for(future, timeout=self._REPLY_ACK_TIMEOUT)
                return response
            except asyncio.TimeoutError:
                # Final-frame ack timeout: WeCom received the frame (we wrote
                # the bytes successfully — _send_json above did not raise) but
                # the ack didn't return within the window.  In practice the
                # server has already rendered the message to the client; the
                # ack is delayed for unrelated reasons (server-side queue lag,
                # WS jitter, concurrent reply on the same WS).
                #
                # The official wecom-openclaw-plugin treats this case as an
                # error and surfaces it to its caller, which then *does not*
                # resend.  Hermes' prior behaviour — raising RuntimeError so
                # the upper layer falls back to a normal markdown send —
                # produced duplicate messages whenever WeCom *had* rendered
                # the streamed frame (see docs/rca-wecom-stream-final-ack-
                # timeout-duplicate.md).
                #
                # Aligning with the official plugin: log a warning and
                # synthesise a success-shaped response so the caller treats
                # the message as delivered.  The thinking-bubble is already
                # closed on the client side by the finish=true frame; if WeCom
                # never queued it (rare), the user sees no answer — same
                # outcome as the official plugin.
                logger.warning(
                    "[%s] Final frame ack timeout (req_id=%s) — treating as "
                    "delivered (matches official wecom-openclaw-plugin "
                    "behaviour). No fallback send.",
                    self.name, normalized,
                )
                return {
                    "errcode": 0,
                    "errmsg": "ack_timeout_assumed_delivered",
                    "ack_pending": True,
                }
            finally:
                if queue.pending_ack is frame:
                    queue.pending_ack = None
                # Cleanup empty queue
                if queue.pending_ack is None:
                    self._reply_queues.pop(normalized, None)
        else:
            # Intermediate frame: fire-and-forget (don't await ack)
            # But the pending_ack stays registered so subsequent frames can
            # check and skip. The ack will be resolved by _dispatch_payload.
            return {"errcode": 0, "errmsg": "sent_nonblocking"}

    def _resolve_reply_ack(self, req_id: str, payload: Dict[str, Any]) -> bool:
        """Resolve a pending reply ack. Returns True if handled."""
        queue = self._reply_queues.get(req_id)
        if queue is None or queue.pending_ack is None:
            return False
        frame = queue.pending_ack
        if not frame.future.done():
            _body = payload.get("body", {}) if isinstance(payload.get("body"), dict) else {}
            logger.debug(
                "[%s] _resolve_reply_ack: resolved req_id=%s is_final=%s "
                "elapsed=%.2fs errcode=%s",
                self.name, req_id, frame.is_final,
                time.monotonic() - (frame.sent_at or time.monotonic()),
                _body.get("errcode", "N/A"),
            )
            frame.future.set_result(payload)
        queue.pending_ack = None
        # Cleanup empty queue
        if queue.pending_ack is None:
            self._reply_queues.pop(req_id, None)
        return True

    def _fail_reply_queues(self, error: Exception) -> None:
        """Fail all pending reply acks (called on disconnect/error)."""
        for queue in list(self._reply_queues.values()):
            if queue.pending_ack and not queue.pending_ack.future.done():
                queue.pending_ack.future.set_exception(error)
        self._reply_queues.clear()

    @staticmethod
    def _new_req_id(prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"

    @staticmethod
    def _payload_req_id(payload: Dict[str, Any]) -> str:
        headers = payload.get("headers")
        return str(headers.get("req_id") or "") if isinstance(headers, dict) else ""

    @staticmethod
    def _parse_json(raw: Any) -> Optional[Dict[str, Any]]:
        raw_len = len(raw) if isinstance(raw, (str, bytes)) else -1
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # WeCom sometimes sends unescaped control characters (e.g. raw
            # newlines) inside JSON string values. Retry with strict=False
            # which accepts control chars in strings per the JSON decoder.
            try:
                decoder = json.JSONDecoder(strict=False)
                payload = decoder.decode(raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace"))
                logger.info(
                    "WeCom payload required strict=False fallback (len=%d)",
                    len(raw) if isinstance(raw, (str, bytes)) else -1,
                )
            except Exception as exc2:
                logger.warning(
                    "Failed to parse WeCom payload (strict=False also failed): "
                    "error=%s len=%d tail=%r",
                    exc2,
                    len(raw) if isinstance(raw, (str, bytes)) else -1,
                    raw[-100:] if isinstance(raw, (str, bytes)) and len(raw) > 100 else raw,
                )
                return None
        except Exception as exc:
            logger.warning(
                "Failed to parse WeCom payload: error=%s len=%d",
                exc, len(raw) if isinstance(raw, (str, bytes)) else -1,
            )
            return None
        return payload if isinstance(payload, dict) else None

    async def _on_message(self, payload: Dict[str, Any]) -> None:
        body = payload.get("body")
        if not isinstance(body, dict):
            return

        msg_id = str(body.get("msgid") or self._payload_req_id(payload) or uuid.uuid4().hex)
        if self._dedup.is_duplicate(msg_id):
            # Promoted from debug to INFO: dedup misfire (#62860 timing bug —
            # is_duplicate marks at check time, so a msgid redelivered ~5s after
            # a processing exception is dropped for the 300s TTL) is a top
            # suspect for Coral's intermittent group non-reply. At debug level it
            # left zero trace at INFO, so a dropped message looked like it
            # vanished after the FULL-payload dump. Log req_id/sender too so this
            # line correlates with the inbound FULL payload above.
            logger.info(
                "[%s] Duplicate message %s ignored (dedup drop) req_id=%s sender=%r chattype=%r",
                self.name,
                msg_id,
                self._payload_req_id(payload),
                (body.get("from") or {}).get("userid") if isinstance(body.get("from"), dict) else None,
                body.get("chattype"),
            )
            return
        self._remember_reply_req_id(msg_id, self._payload_req_id(payload))

        sender = body.get("from") if isinstance(body.get("from"), dict) else {}
        sender_id = str(sender.get("userid") or "").strip()
        if self._dedup.is_duplicate(msg_id):
            # INFO: a msgid redelivered after a processing exception is dropped for the TTL.
            logger.info("[%s] Duplicate message %s ignored (dedup drop) req_id=%s sender=%r chattype=%r", self.name, msg_id, req_id, sender.get("userid") if sender else None, body.get("chattype"))
            return
        if req_id:
            bounded_put(self._reply_req_ids, msg_id, req_id, DEDUP_MAX_SIZE)
        chat_id = str(body.get("chatid") or sender_id).strip()

        # Diagnostic: log the shape of every inbound callback at INFO so we can
        # see what WeCom actually sends for group messages (chattype value,
        # presence of chatid, msgtype). Group frames may arrive with a
        # chattype other than the literal "group" we test for below.
        logger.info(
            "[%s] Inbound callback: chattype=%r chatid=%r sender=%r msgtype=%r has_chatid=%s",
            self.name,
            body.get("chattype"),
            body.get("chatid"),
            sender_id,
            body.get("msgtype"),
            bool(body.get("chatid")),
        )

        if not chat_id:
            logger.info("[%s] Missing chat id, skipping message; body_keys=%s", self.name, list(body.keys()))
            return
        is_group = str(body.get("chattype") or "").lower() == "group"
        if is_group:
            self._group_chat_ids.add(chat_id)
            if not self._is_group_allowed(chat_id, sender_id):
                logger.info(
                    "[%s] Group message DROPPED by policy: chat=%s sender=%s group_policy=%r "
                    "(set group_policy to 'open' or add to group_allow_from to receive)",
                    self.name, chat_id, sender_id, self._group_policy,
                )
                return
        elif not self._is_dm_intake_allowed(sender_id):
            logger.info("[%s] DM sender %s blocked by policy", self.name, sender_id)
            return
        # Post-policy: cache req_id so sends can fall back to passive reply (required in groups).
        self._remember_chat_req_id(chat_id, req_id)
        text, reply_text = self._extract_text(body)
        if is_group and text:
            text = re.sub(r"^@\S+\s*", "", text).strip()  # "@Bot /approve" -> "/approve"
        media_urls, media_types = await self._extract_media(body)
        message_type = self._derive_message_type(body, text, media_types)
        has_reply_context = bool(reply_text and (text or media_urls))
        if reply_text and not has_reply_context:  # quote-only message: the quote becomes the text
            text = reply_text
        if not text and not media_urls:
            logger.info(
                "[%s] Empty WeCom message skipped: is_group=%s chat=%s msgtype=%r",
                self.name, is_group, chat_id, body.get("msgtype"),
            )
            return
        source = self.build_source(chat_id=chat_id, chat_type="group" if is_group else "dm", user_id=sender_id or None, user_name=sender_id or None,
                                   message_id=msg_id)
        event = MessageEvent(
            text=text, message_type=message_type, source=source, raw_message=payload, message_id=msg_id, media_urls=media_urls, media_types=media_types,
            reply_to_message_id=f"quote:{msg_id}" if has_reply_context else None, reply_to_text=reply_text if has_reply_context else None, timestamp=datetime.now(tz=timezone.utc),
        )

        # Only batch plain text messages — commands, media, etc. dispatch
        # immediately since they won't be split by the WeCom client.
        #
        # Exception: an attachment-ONLY message (media, no text) is held for
        # a short merge window. WeCom clients send "image + text" as TWO
        # separate inbound callbacks — an attachment-only frame followed a few
        # hundred ms later by the text frame. Dispatching the attachment
        # immediately spawns an agent run that the trailing text then
        # "interrupts" (junk "⚡ Interrupting" + "✅" acks). Instead we buffer
        # the attachment on the SAME pending-batch machinery so the following
        # text merges into one event (see _flush_text_batch for the window).
        batch_key = self._text_batch_key(event)
        has_pending_batch = batch_key in self._pending_text_batches
        is_attachment_only = bool(media_urls) and not (text or "").strip()

        if message_type == MessageType.TEXT and (
            self._text_batch_delay_seconds > 0 or has_pending_batch
        ):
            # Route text through the buffer whenever batching is on OR an
            # attachment is already held for this session, so the trailing
            # text always merges instead of dispatching on its own.
            self._enqueue_text_event(event)
        elif is_attachment_only and self._attachment_text_merge_delay_seconds > 0:
            self._enqueue_text_event(event)
        else:
            await self.handle_message(event)

    # ------------------------------------------------------------------
    # Text message aggregation (handles WeCom client-side splits)
    # ------------------------------------------------------------------

    def _text_batch_key(self, event: MessageEvent) -> str:
        """Session-scoped key for text message batching."""
        from gateway.session import build_session_key
        return build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
            profile=self._session_key_profile(event.source),
        )

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Buffer an event and reset the flush timer.

        Two cases share this buffer:

        * WeCom splits a long user message at 4000 chars — the chunks arrive
          within a few hundred milliseconds and are merged into one event.
        * WeCom sends "image + text" as two callbacks (an attachment-only
          frame, then a text frame). The attachment-only frame is buffered
          here first (message_type PHOTO/DOCUMENT/VOICE); when the text frame
          arrives it merges into the same event and the type is promoted to
          TEXT so it dispatches as a single text-with-media turn — matching
          the shape WeCom uses when text+media arrive in one callback.
        """
        key = self._text_batch_key(event)
        existing = self._pending_text_batches.get(key)
        chunk_len = len(event.text or "")
        if existing is None:
            event._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            self._pending_text_batches[key] = event
        else:
            if event.text:
                existing.text = f"{existing.text}\n{event.text}" if existing.text else event.text
            existing._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            # Merge any media that might be attached
            if event.media_urls:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)
            # Once real text joins a buffered attachment-only event, the merged
            # event is a text-with-media turn. Promote the type so downstream
            # dispatch treats it like a normal text message (and inherits the
            # trailing text frame's reply/quote context if the first frame had
            # none).
            if event.text and (event.text or "").strip():
                existing.message_type = MessageType.TEXT
                if event.reply_to_text and not existing.reply_to_text:
                    existing.reply_to_text = event.reply_to_text
                    existing.reply_to_message_id = event.reply_to_message_id

        # Cancel any pending flush and restart the timer
        prior_task = self._pending_text_batch_tasks.get(key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_text_batch_tasks[key] = asyncio.create_task(
            self._flush_text_batch(key)
        )

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text.

        Uses a longer delay when the latest chunk is near WeCom's 4000-char
        split point, since a continuation chunk is almost certain.
        """
        current_task = asyncio.current_task()
        try:
            pending = self._pending_text_batches.get(key)
            last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
            # An attachment-only buffered event (no text yet) waits the
            # attachment/text merge window for a trailing text frame. A text
            # buffered event uses the normal (or split-continuation) delay.
            is_attachment_only = bool(
                pending and pending.media_urls and not (pending.text or "").strip()
            )
            if is_attachment_only:
                delay = self._attachment_text_merge_delay_seconds
            elif last_len >= self._SPLIT_THRESHOLD:
                delay = self._text_batch_split_delay_seconds
            else:
                delay = self._text_batch_delay_seconds
            await asyncio.sleep(delay)
            # Guard against the cancel-delivery race: when the sleep timer
            # fires just before cancel() is called, CPython sets
            # Task._must_cancel but cannot cancel the already-done sleep
            # future, so CancelledError is delivered at the *next* await
            # (handle_message) rather than here.  By that point this task
            # has already popped the merged event, so the superseding task
            # sees an empty batch and silently drops the message.
            # This check is synchronous — no await between the sleep and
            # the pop — so no other coroutine can modify the task registry
            # in between.
            if self._pending_text_batch_tasks.get(key) is not current_task:
                return
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            logger.info(
                "[WeCom] Flushing batch %s (%d chars, %d media)",
                key, len(event.text or ""), len(event.media_urls or []),
            )
            await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    @staticmethod
    def _extract_text(body: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        msgtype = str(body.get("msgtype") or "").lower()
        if msgtype == "mixed":
            items = _dict_or_empty(body, "mixed").get("msg_item")
            text_parts = [_content_of(item, "text") for item in (items if isinstance(items, list) else []) if isinstance(item, dict) and str(item.get("msgtype") or "").lower() == "text"]
        else:  # voice transcript / appmsg attachment title (filename) follow the text; empties drop below
            text_parts = [
                _content_of(body, "text"), _content_of(body, "voice") if msgtype == "voice" else "",
                str(_dict_or_empty(body, "appmsg").get("title") or "").strip() if msgtype == "appmsg" else "",
            ]
        quote = _dict_or_empty(body, "quote")
        quote_type = str(quote.get("msgtype") or "").lower()
        reply_text = _content_of(quote, quote_type) or None if quote_type in ("text", "voice") else None
        return "\n".join(part for part in text_parts if part).strip(), reply_text

    @staticmethod
    def _derive_message_type(body: Dict[str, Any], text: str, media_types: List[str]) -> MessageType:
        if any(mtype.startswith(("application/", "text/")) for mtype in media_types):
            return MessageType.DOCUMENT
        if any(mtype.startswith("image/") for mtype in media_types):
            return MessageType.TEXT if text else MessageType.PHOTO
        if str(body.get("msgtype") or "").lower() == "voice":
            return MessageType.VOICE
        return MessageType.TEXT

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------

    @property
    def enforces_own_access_policy(self) -> bool:
        """WeCom gates DM/group access at intake via dm_policy/group_policy."""
        return True

    def _open_dm_opted_in(self) -> bool:
        # Scoped reads (#93522): the default profile's allow-all flag must
        # not leak into a multiplexed secondary profile's admission gate.
        if (_get_scoped_secret("GATEWAY_ALLOW_ALL_USERS", "") or "").lower() in {"true", "1", "yes"}:
            return True
        return (_get_scoped_secret("WECOM_ALLOW_ALL_USERS", "") or "").lower() in {"true", "1", "yes"}

    def _is_dm_allowed(self, sender_id: str) -> bool:
        if self._dm_policy == "disabled":
            return False
        if self._dm_policy == "allowlist":
            return _entry_matches(self._allow_from, sender_id)
        if self._dm_policy == "open":
            return self._open_dm_opted_in()
        return False

    def _is_dm_intake_allowed(self, sender_id: str) -> bool:
        principal = str(sender_id or "").strip()
        if not principal:
            return False
        if self._dm_policy == "disabled":
            return False
        if self._dm_policy == "allowlist":
            return _entry_matches(self._allow_from, principal)
        if self._dm_policy == "pairing":
            return True
        if self._dm_policy == "open":
            return self._open_dm_opted_in()
        return False

    def _is_group_allowed(self, chat_id: str, sender_id: str) -> bool:
        """Per-group ``groups.<id>.allow_from`` restricts senders on top of the chat-level policy."""
        if not super()._is_group_allowed(chat_id):
            return False
        group_cfg = self._resolve_group_cfg(chat_id)
        sender_allow = _coerce_list(group_cfg.get("allow_from") or group_cfg.get("allowFrom"))
        return _entry_matches(sender_allow, sender_id) if sender_allow else True

    def _resolve_group_cfg(self, chat_id: str) -> Dict[str, Any]:
        """Exact key, then case-insensitive key, then ``"*"``; only dict values count."""
        if not isinstance(self._groups, dict):
            return {}
        lowered = chat_id.lower()
        candidates = (self._groups.get(chat_id), next((v for k, v in self._groups.items() if isinstance(k, str) and k.lower() == lowered and isinstance(v, dict)), None), self._groups.get("*"))
        return next((c for c in candidates if isinstance(c, dict)), {})

    def _remember_chat_req_id(self, chat_id: str, req_id: str) -> None:
        """Cache the most recent inbound req_id per chat.

        Used as a fallback reply target when we need to send into a group
        without an explicit ``reply_to`` — WeCom AI Bots are blocked from
        APP_CMD_SEND in groups and must use APP_CMD_RESPONSE bound to some
        prior req_id. Bounded like _reply_req_ids so long-running gateways
        don't leak memory across many chats.
        """
        normalized_chat_id = str(chat_id or "").strip()
        normalized_req_id = str(req_id or "").strip()
        if not normalized_chat_id or not normalized_req_id:
            return
        self._last_chat_req_ids[normalized_chat_id] = normalized_req_id
        while len(self._last_chat_req_ids) > DEDUP_MAX_SIZE:
            self._last_chat_req_ids.pop(next(iter(self._last_chat_req_ids)))
        # A fresh inbound req_id resurrects the stream channel — drop any
        # stale "stream is dead" marker from prior 846608 responses so the
        # next outbound turn can attempt native streaming again.
        self._stream_expired_chats.discard(normalized_chat_id)
        # A new inbound message starts a new "turn" — allow send_typing to
        # open a fresh stream again (the previous turn's delivery guard is
        # no longer relevant).

    def _resolve_stream_req_id(
        self, chat_id: str, reply_to: Optional[str]
    ) -> Optional[str]:
        """Pick a req_id for a stream reply.

        Precedence: explicit ``reply_to`` (a prior message id we cached) →
        last inbound req_id for this chat → ``None`` (stream impossible).
        """
        req_id = self._reply_req_id_for_message(reply_to)
        if req_id:
            return req_id
        return self._last_chat_req_ids.get(str(chat_id or "").strip()) or None

    def _get_or_create_stream_turn(self, chat_id: str, req_id: str) -> StreamTurn:
        """Get or create a StreamTurn for the given chat and req_id."""
        key = f"{chat_id}:{req_id}"
        if key not in self._stream_turns:
            self._stream_turns[key] = StreamTurn(chat_id, req_id)
        return self._stream_turns[key]

    def _cleanup_stream_turn(self, chat_id: str, req_id: str) -> None:
        """Clean up a StreamTurn after finalization or error."""
        key = f"{chat_id}:{req_id}"
        turn = self._stream_turns.pop(key, None)
        if turn is not None:
            self._cancel_idle_flush(turn)
            self._cancel_keepalive(turn)

    def _cancel_idle_flush(self, turn: StreamTurn) -> None:
        """Cancel a pending idle-flush timer on the turn (no-op if unarmed)."""
        handle = turn.idle_flush_handle
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass
            turn.idle_flush_handle = None

    # ── Stream-level keep-alive (Layer 1) ─────────────────────────────────
    # Structurally mirrors the idle-flush timer: a per-turn asyncio TimerHandle
    # stored on the StreamTurn, cancelled on every turn-exit path.  The only
    # difference from idle-flush is cadence (minutes vs 250ms) and intent
    # (refresh the server's 6-min stream window vs ship a partial buffer).

    def _cancel_keepalive(self, turn: StreamTurn) -> None:
        """Cancel a pending keep-alive timer on the turn (no-op if unarmed)."""
        handle = turn.keepalive_handle
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass
            turn.keepalive_handle = None

    def _arm_keepalive(
        self,
        turn: StreamTurn,
        *,
        turn_id: Optional[str],
    ) -> None:
        """Arm the keep-alive timer (Layer 1) if enabled and not already armed.

        Idempotent — re-arming while one is pending no-ops; a fresh timer is
        only scheduled after the previous one fired or was cancelled.  Skips
        entirely when keep-alive is disabled by config (the default).
        """
        if not self._stream_keepalive_enabled:
            return
        if turn.finalized or turn.expired:
            return
        if turn.keepalive_handle is not None:
            return  # already armed
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # not inside a loop (defensive)
        handle = loop.call_later(
            self._stream_keepalive_interval_seconds,
            self._on_keepalive_fire,
            turn,
            turn_id,
        )
        turn.keepalive_handle = handle

    def _on_keepalive_fire(
        self,
        turn: StreamTurn,
        turn_id: Optional[str],
    ) -> None:
        """Loop callback — dispatch an async keep-alive send without blocking."""
        turn.keepalive_handle = None
        if turn.finalized or turn.expired:
            return
        try:
            asyncio.ensure_future(self._keepalive_send(turn, turn_id))
        except RuntimeError:
            pass

    async def _keepalive_send(
        self,
        turn: StreamTurn,
        turn_id: Optional[str],
    ) -> None:
        """Re-send the accumulated text as a finish=false frame to refresh the
        WeCom server's stream window, then re-arm for the next interval.

        Deliberately conservative (see ANALYSIS §4.2/§5):

        * **Never sends a placeholder.**  When there is no accumulated text yet
          (e.g. a cron turn still fetching data), the tick is skipped and the
          timer is re-armed — we'd rather let Layer 2's clock fallback handle a
          content-less turn than pollute ``last_sent_content`` with a filler
          frame or strand the user on a "still working…" bubble.
        * Reuses ``_send_stream_reply(finish=False)`` with the queue's
          ``skip_if_pending`` semantics, so a heartbeat that races an in-flight
          ack is dropped rather than piling onto the ack queue.
        * On 846604/846608 marks the turn expired and retires it so finalize
          takes the Layer 2 fallback; stops the timer (no re-arm).
        """
        if turn.finalized or turn.expired:
            return
        if turn._intermediate_frames_sent >= MAX_INTERMEDIATE_FRAMES:
            # No room left for intermediate frames; stop keeping alive and let
            # finalize (or Layer 2) run.  Do not re-arm.
            return
        content = turn.accumulated_text or ""
        if not content.strip():
            # Nothing to refresh with yet — skip this tick, re-arm for later.
            self._arm_keepalive(turn, turn_id=turn_id)
            return
        try:
            await self._send_stream_reply(
                turn.req_id,
                turn.stream_id,
                content,
                finish=False,
            )
        except WeComStreamExpiredError:
            turn.expired = True
            self._retire_turn(turn, turn_id)
            self._stream_expired_chats.add(turn.chat_id)
            return
        except Exception as exc:
            logger.debug(
                "[%s] keep-alive send failed (chat=%s, turn=%s): %s",
                self.name, turn.chat_id, turn.stream_id, exc,
            )
            # Transient failure — re-arm and try again next interval.
            self._arm_keepalive(turn, turn_id=turn_id)
            return
        turn._last_frame_sent_at = time.monotonic()
        turn.last_sent_content = content
        # Re-arm for the next interval (guarded internally against
        # finalized/expired).
        self._arm_keepalive(turn, turn_id=turn_id)

    def _retire_turn(self, turn: StreamTurn, turn_id: Optional[str]) -> None:
        """Remove a turn from the registry and cancel BOTH of its timers.

        Single choke point for the "turn is dead" cleanup shared by the
        expired/error paths.  Cancels idle-flush and keep-alive timers before
        popping so neither can fire on a retired turn.
        """
        self._cancel_idle_flush(turn)
        self._cancel_keepalive(turn)
        if turn_id:
            self._stream_turns.pop(f"{turn.chat_id}:{turn_id}", None)
        else:
            self._cleanup_stream_turn(turn.chat_id, turn.req_id)

    def _find_active_turn_for_chat(self, chat_id: str) -> Optional[StreamTurn]:
        """Find the most recent active (non-finalized) turn for a chat."""
        for turn in self._stream_turns.values():
            if turn.chat_id == chat_id and not turn.finalized:
                return turn
        return None

    def _reset_native_stream_state(self) -> None:
        """Legacy method for compatibility. Now a no-op since state is per-turn."""
        # No-op: stream state is now per-turn, not global.
        # Kept for compatibility with existing code that calls this.
        pass

    async def _force_reconnect_on_stale_subscription(self, errcode: int) -> None:
        """Force-close the WS when server rejects our subscription (846609).

        WeCom errcode 846609 means the server no longer considers this WS
        session subscribed — all sends will fail until we reconnect. Rather
        than waiting for the WS to close naturally (can take 2+ minutes of
        timeouts), we proactively close it to trigger _listen_loop's
        reconnect cycle immediately.
        """
        if errcode != STREAM_NOT_SUBSCRIBED_ERRCODE:
            return
        logger.warning(
            "[%s] Got errcode %d (subscription lost) — clearing stale state",
            self.name, errcode,
        )
        # Only invalidate cached req_ids (bound to the dead session).
        # Do NOT close the WS — closing triggers _listen_loop to reconnect,
        # which opens a second WS connection. WeCom only allows one long-lived
        # connection per bot; the server kicks the second one and invalidates
        # the first's session, creating an infinite kick-reconnect loop.
        # The WS will be closed by the server side naturally; _listen_loop
        # handles the reconnect when that happens.
        self._last_chat_req_ids.clear()
        self._reply_req_ids.clear()
        self._reset_native_stream_state()

    def _reply_req_id_for_message(self, reply_to: Optional[str]) -> Optional[str]:
        normalized = str(reply_to or "").strip()
        return None if not normalized or normalized.startswith("quote:") else self._reply_req_ids.get(normalized)

    def _cached_reply_req_id(self, chat_id: str, reply_to: Optional[str]) -> Optional[str]:
        """Explicit reply_to mapping, else the chat's last inbound req_id."""
        return self._reply_req_id_for_message(reply_to) or self._last_chat_req_ids.get(chat_id)

    async def _force_reconnect_on_stale_subscription(self, errcode: int) -> None:
        """On 846609 (subscription lost) drop req_ids bound to the dead session. Do NOT close the
        WS: a second connection gets kicked and invalidates the first (infinite kick loop)."""
        if errcode != STREAM_NOT_SUBSCRIBED_ERRCODE:
            return
        logger.warning("[%s] Got errcode %d (subscription lost) — clearing stale state", self.name, errcode)
        self._last_chat_req_ids.clear()
        self._reply_req_ids.clear()

    @staticmethod
    def _response_error(response: Dict[str, Any]) -> Optional[str]:
        errcode = response.get("errcode", 0)
        return None if errcode in {0, None} else f"WeCom errcode {errcode}: {response.get('errmsg') or 'unknown error'}"

    @classmethod
    def _raise_for_wecom_error(cls, response: Dict[str, Any], operation: str) -> None:
        error = cls._response_error(response)
        if error:
            raise RuntimeError(f"{operation} failed: {error}")

    def _markdown_body(self, content: str) -> Dict[str, Any]:
        return {"msgtype": "markdown", "markdown": {"content": content[:self.MAX_MESSAGE_LENGTH]}}

    async def _send_reply_markdown(self, reply_req_id: str, content: str) -> Dict[str, Any]:
        response = await self._send_reply_request(reply_req_id, self._markdown_body(content))
        self._raise_for_wecom_error(response, "send reply markdown")
        return response

    @staticmethod
    def _truncate_stream_content(content: str, limit: int) -> str:
        """Truncate ``content`` to fit within ``limit`` UTF-8 bytes.

        WeCom enforces a byte-length cap on stream frames; truncating by
        codepoints would still let multi-byte runs blow past the limit.
        """
        encoded = content.encode("utf-8")
        if len(encoded) <= limit:
            return content
        return encoded[:limit].decode("utf-8", errors="ignore")

    async def _send_stream_reply(
        self,
        reply_req_id: str,
        stream_id: str,
        content: str,
        finish: bool = False,
    ) -> Dict[str, Any]:
        """Send a single ``msgtype: "stream"`` frame via aibot_respond_msg.

        Uses the per-req_id reply queue with ack tracking, aligned with the
        official WeCom SDK's replyStreamNonBlocking semantics:

          * **Intermediate frames** (finish=False): sent non-blocking via
            ``_send_reply_queued(skip_if_pending=True)``. If a prior frame's
            ack is still pending, the frame is skipped (cumulative text means
            no information is lost — the next frame carries all content).
          * **Final frame** (finish=True): waits for any pending ack to drain
            before sending, then awaits its own ack. This prevents version
            conflicts (errcode 6000) between the finalize and a concurrent
            intermediate frame.

        Raises :class:`WeComStreamExpiredError` on errcode 846608 so the
        caller can fall back to a proactive markdown send.
        """
        truncated = self._truncate_stream_content(
            content or "", self.MAX_STREAM_CONTENT_LENGTH,
        )
        if len(content or "") != len(truncated):
            logger.warning(
                "[%s] Stream content truncated for stream_id=%s",
                self.name, stream_id,
            )
        body: Dict[str, Any] = {
            "msgtype": "stream",
            "stream": {
                "id": stream_id,
                "finish": bool(finish),
                "content": truncated,
            },
        }

        if not finish:
            # Intermediate frame: non-blocking with pending-skip semantics.
            # If a previous frame's ack is still pending on this req_id,
            # skip this frame entirely (cumulative text guarantees no loss).
            response = await self._send_reply_queued(
                reply_req_id, body, is_final=False, skip_if_pending=True,
            )
            return response

        # Final frame: wait for any pending intermediate ack, then send
        # with ack tracking so we reliably detect 846608/6000.
        response = await self._send_reply_queued(
            reply_req_id, body, is_final=True, skip_if_pending=False,
        )
        errcode = response.get("errcode", 0)
        if errcode in (STREAM_EXPIRED_ERRCODE, STREAM_REQUEST_EXPIRED_ERRCODE):
            # 846608 (stream update window) and 846604 (req_id reply-request
            # window) both mean the reply flow is dead — raise the same
            # expired error so the caller falls back to a proactive send.
            raise WeComStreamExpiredError(
                errcode=errcode, errmsg=str(response.get("errmsg") or ""),
            )
        if errcode == STREAM_VERSION_CONFLICT_ERRCODE:
            # 6000 = version conflict: a newer frame on this stream_id already
            # replaced the bubble. For a finalize frame this means the content
            # is ALREADY on screen (idempotent re-finalize losing the race to a
            # newer version), so treat it as delivered rather than raising —
            # raising here would pop the turn and drop us into a duplicate
            # standalone send(). This is what makes idempotent finalize retry
            # safe: retrying a finalize that already landed returns 6000, which
            # we now absorb instead of turning into a second message.
            logger.info(
                "[%s] finalize hit errcode 6000 (version conflict) — bubble "
                "already replaced by a newer frame; treating as delivered.",
                self.name,
            )
            return response
        self._raise_for_wecom_error(response, "send stream reply")
        return response

    async def _send_reply_media_message(
        self,
        reply_req_id: str,
        media_type: str,
        media_id: str,
    ) -> Dict[str, Any]:
        response = await self._send_reply_request(
            reply_req_id,
            {
                "msgtype": media_type,
                media_type: {"media_id": media_id},
            },
        )
        self._raise_for_wecom_error(response, "send reply media message")
        return response

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send standalone markdown (never touches active streams); serialized per chat for the 30 msgs/min
        limit (846607). ``metadata["is_approval_prompt"]`` uses the control lane."""
        if not chat_id:
            return SendResult(success=False, error="chat_id is required")
        metadata = metadata or {}  # pops mutate the caller's dict on purpose (consumed flags)
        is_control = metadata.pop("is_approval_prompt", False)
        # Approval *confirmations* must not consume the req_id the stream consumer still needs.
        force_proactive = bool(metadata.pop("force_proactive_send", False))
        return await self._enqueue_chat_send(chat_id, lambda: self._send_inner(chat_id, content, reply_to, force_proactive=force_proactive), is_control=is_control)

    async def _send_inner(self, chat_id: str, content: str, reply_to: Optional[str] = None, *, force_proactive: bool = False) -> SendResult:
        """Send under the per-chat queue; force_proactive skips passive reply except in groups."""
        try:
            prepared = await self._prepare_outbound_media(media_source, file_name=file_name)
        except FileNotFoundError as exc:
            return SendResult(success=False, error=str(exc))
        except Exception as exc:
            logger.error("[%s] Failed to prepare outbound media %s: %s", self.name, media_source, exc)
            return SendResult(success=False, error=str(exc))

        if prepared["rejected"]:
            await self._send_followup_markdown(
                chat_id,
                f"⚠️ {prepared['reject_reason']}",
                reply_to=reply_to,
            )
            return SendResult(success=False, error=prepared["reject_reason"])

        reply_req_id = self._reply_req_id_for_message(reply_to)
        if not reply_req_id and chat_id in self._last_chat_req_ids:
            reply_req_id = self._last_chat_req_ids[chat_id]

        # When native streaming was/is active for this chat, media MUST go
        # through the proactive send path (aibot_send_msg), NOT passive reply
        # (aibot_respond_msg). This mirrors the official OpenClaw plugin:
        #   "replyMedia（被动回复）无法覆盖 replyStream 发出的 thinking 流式消息，
        #    因此所有媒体统一走 aibot_send_msg 主动发送。"
        # The reply_req_id is "owned" by the stream — using it for media
        # causes the server to either ignore it or never ack.
        active_turn = self._find_active_turn_for_chat(chat_id)
        if active_turn or chat_id in self._stream_expired_chats:
            reply_req_id = None  # force proactive send

        try:
            upload_result = await self._upload_media_bytes(
                prepared["data"],
                prepared["final_type"],
                prepared["file_name"],
            )
            logger.info("[%s] upload_media_bytes OK: media_id=%s type=%s", self.name, upload_result.get("media_id"), prepared["final_type"])
            if reply_req_id:
                media_response = await self._send_reply_media_message(
                    reply_req_id,
                    prepared["final_type"],
                    upload_result["media_id"],
                )
                logger.info("[%s] send_reply_media OK: %s", self.name, media_response)
            else:
                media_response = await self._send_media_message(
                    chat_id,
                    prepared["final_type"],
                    upload_result["media_id"],
                )
                logger.info("[%s] send_media_message OK: %s", self.name, media_response)
        except asyncio.TimeoutError:
            logger.error("[%s] TIMEOUT in _send_media_source for %s", self.name, media_source)
            return SendResult(success=False, error="Timeout sending media to WeCom")
        except Exception as exc:
            logger.error("[%s] Failed to send media %s: %s", self.name, media_source, exc)
            return SendResult(success=False, error=str(exc))

        caption_result = None
        downgrade_result = None
        if caption:
            caption_result = await self._send_followup_markdown(
                chat_id,
                caption,
                reply_to=reply_to,
            )
        if prepared["downgraded"] and prepared["downgrade_note"]:
            downgrade_result = await self._send_followup_markdown(
                chat_id,
                f"ℹ️ {prepared['downgrade_note']}",
                reply_to=reply_to,
            )

        return SendResult(
            success=True,
            message_id=self._payload_req_id(media_response) or uuid.uuid4().hex[:12],
            raw_response={
                "upload": upload_result,
                "media": media_response,
                "caption": caption_result.raw_response if caption_result else None,
                "caption_error": caption_result.error if caption_result and not caption_result.success else None,
                "downgrade": downgrade_result.raw_response if downgrade_result else None,
                "downgrade_error": downgrade_result.error if downgrade_result and not downgrade_result.success else None,
            },
        )

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send markdown to a WeCom chat.

        Sends content as a standalone message without interfering with any
        active streams. Streams are managed by their creators (typically
        GatewayStreamConsumer) who call send_stream_frame(finalize=True)
        when ready.

        All sends are serialized per chat_id to avoid exceeding WeCom's
        30 msgs/min/chat rate limit (errcode 846607).

        If metadata contains "is_approval_prompt": True, the message is routed
        through the control lane for immediate delivery.
        """
        if not chat_id:
            return SendResult(success=False, error="chat_id is required")

        # Check if this is an approval prompt (should use control lane)
        is_control = False
        force_proactive = False
        if metadata:
            is_control = metadata.pop("is_approval_prompt", False)
            # Explicit opt-in for proactive send: used by approval
            # *confirmation* messages (post-/approve) that must not consume
            # the req_id the stream consumer needs for resumed output.
            # Distinct from is_approval_prompt which only routes to the
            # control lane — the initial approval *request* prompt still
            # uses passive reply (required for groups where APP_CMD_SEND
            # is blocked).
            force_proactive = bool(metadata.pop("force_proactive_send", False))

        return await self._enqueue_chat_send(
            chat_id,
            lambda: self._send_inner(chat_id, content, reply_to, force_proactive=force_proactive),
            is_control=is_control,
        )

    async def _send_inner(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        *,
        force_proactive: bool = False,
    ) -> SendResult:
        """Actual send logic, called under the per-chat lock.

        Sends content as a standalone message. Does NOT close any active
        streams — streams are managed by their creators (GatewayStreamConsumer)
        who call send_stream_frame(finalize=True) when ready.

        This aligns with the official wecom-openclaw-plugin model where
        send() and streaming are independent operations.

        Args:
            force_proactive: When True, always use APP_CMD_SEND instead of
                passive reply. Used for approval confirmations to avoid
                consuming the req_id needed by the post-approval stream.
        """
        try:
            # Directly send the message without touching any active streams.
            # GatewayStreamConsumer manages its own stream lifecycle via
            # send_stream_frame() with turn_id, so send() shouldn't interfere.

            reply_req_id = self._reply_req_id_for_message(reply_to)

            if not reply_req_id and chat_id in self._last_chat_req_ids:
                reply_req_id = self._last_chat_req_ids[chat_id]

            if force_proactive and chat_id not in self._group_chat_ids:
                reply_req_id = None

            if reply_req_id:
                try:
                    response = await self._send_reply_markdown(reply_req_id, content)
                except (asyncio.TimeoutError, RuntimeError) as passive_err:
                    # Passive reply failed (req_id may be stale after WS reconnect).
                    # Fall back to proactive aibot_send_msg which doesn't depend
                    # on any prior req_id.
                    logger.warning(
                        "[%s] Passive reply failed (%s), falling back to proactive send",
                        self.name, passive_err,
                    )
                    response = await self._send_request(
                        APP_CMD_SEND,
                        {
                            "chatid": chat_id,
                            "msgtype": "markdown",
                            "markdown": {"content": content[:self.MAX_MESSAGE_LENGTH]},
                        },
                    )
            else:
                # No req_id available — must use proactive APP_CMD_SEND.
                # Group chats cannot use APP_CMD_SEND (WeCom blocks it),
                # so fail early with a clear error instead of making a
                # doomed network request.
                if chat_id in self._group_chat_ids:
                    logger.warning(
                        "[%s] No cached req_id for group chat %s — "
                        "cannot send (groups require passive reply via req_id)",
                        self.name, chat_id,
                    )
                    return SendResult(
                        success=False,
                        error="No req_id available for group chat (passive reply required)",
                    )
                response = await self._send_request(
                    APP_CMD_SEND,
                    {
                        "chatid": chat_id,
                        "msgtype": "markdown",
                        "markdown": {"content": content[:self.MAX_MESSAGE_LENGTH]},
                    },
                )
        except asyncio.TimeoutError:
            return SendResult(success=False, error="Timeout sending message to WeCom")
        except Exception as exc:
            logger.error("[%s] Send failed: %s", self.name, exc)
            # Detect 846609 (subscription lost) and trigger reconnect so
            # subsequent messages don't fail for 2+ minutes while the dead
            # WS connection lingers.
            exc_str = str(exc)
            if str(STREAM_NOT_SUBSCRIBED_ERRCODE) in exc_str:
                asyncio.ensure_future(
                    self._force_reconnect_on_stale_subscription(STREAM_NOT_SUBSCRIBED_ERRCODE)
                )
            return SendResult(success=False, error=str(exc))

        error = self._response_error(response)
        if error:
            # Also check the response-level errcode for 846609.
            errcode = response.get("errcode", 0)
            if errcode == STREAM_NOT_SUBSCRIBED_ERRCODE:
                asyncio.ensure_future(
                    self._force_reconnect_on_stale_subscription(errcode)
                )
            return SendResult(success=False, error=error)

        # Mark delivered so _keep_typing cannot open an orphan stream after
        # this turn's reply already landed (regardless of which path was taken).
        return SendResult(
            success=True,
            message_id=self._payload_req_id(response) or uuid.uuid4().hex[:12],
            raw_response=response,
        )

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        del metadata

        result = await self._send_media_source(
            chat_id=chat_id,
            media_source=image_url,
            caption=caption,
            reply_to=reply_to,
        )
        if result.success or not self._looks_like_url(image_url):
            return result

        logger.warning("[%s] Falling back to text send for image URL %s: %s", self.name, image_url, result.error)
        fallback_text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id=chat_id, content=fallback_text, reply_to=reply_to)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del kwargs
        return await self._send_media_source(
            chat_id=chat_id,
            media_source=image_path,
            caption=caption,
            reply_to=reply_to,
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
        del kwargs
        logger.info("[%s] send_document called: chat=%s file=%s", self.name, chat_id, file_path)
        return await self._send_media_source(
            chat_id=chat_id,
            media_source=file_path,
            caption=caption,
            file_name=file_name,
            reply_to=reply_to,
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del kwargs
        return await self._send_media_source(
            chat_id=chat_id,
            media_source=audio_path,
            caption=caption,
            reply_to=reply_to,
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del kwargs
        return await self._send_media_source(
            chat_id=chat_id,
            media_source=video_path,
            caption=caption,
            reply_to=reply_to,
        )

    async def send_stream_frame(
        self,
        text: str,
        *,
        finalize: bool = False,
        chat_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Public entry-point for the gateway streaming consumer.

        Native streaming lifecycle (per-turn):
          * **First call** for a turn: resolve req_id, create StreamTurn,
            and send an empty seed frame to trigger WeCom typing animation.
          * **Subsequent calls**: reuse the same StreamTurn's stream_id

        Args:
            **kwargs: Additional platform-specific parameters. Currently supports:
                - turn_id (str): Optional unique identifier for this turn. When
                  provided, the StreamTurn is keyed by (chat_id, turn_id) instead
                  of (chat_id, req_id), preventing concurrent consumers (e.g.,
                  /background, parallel subagents) from interfering with each
                  other. Mirrors official wecom-openclaw-plugin's per-message
                  streamId model.
            and push cumulative text (not deltas) for in-place updates.
          * **finalize=True**: send closing frame and clean up turn state.

        Each turn (chat_id + req_id) maintains independent state, allowing
        concurrent messages without interference (e.g., approval during streaming).

        Returns ``True`` when the frame landed; ``False`` when the
        stream is unavailable (no req_id, expired session, transport
        error). On ``False`` the caller should fall back to
        :meth:`send` to deliver the remaining content as a one-shot
        markdown reply.
        """
        chat = (chat_id or "").strip()
        if not chat:
            logger.warning(
                "[%s] send_stream_frame: chat_id required",
                self.name,
            )
            return False

        # Extract turn_id early to decide whether to check chat-level expired
        turn_id = kwargs.get("turn_id")

        # Chat-level stream expiry only blocks NEW turn creation.
        # Existing turns (identified by turn_id) can continue to finalize
        # even after another turn in the same chat triggered WeComStreamExpiredError.
        # This prevents cross-turn interference in concurrent scenarios.
        if not turn_id and chat in self._stream_expired_chats:
            # No turn_id provided, and chat is expired → block new turn creation
            return False

        if finalize:
            # Finalize frame counts toward 30/min — go through the control queue
            # (high priority) to prevent blocking by normal messages or other streams.
            turn_id = kwargs.get("turn_id")
            return await self._enqueue_chat_send(
                chat,
                lambda: self._send_stream_frame_inner(text, chat=chat, reply_to=reply_to, finalize=True, turn_id=turn_id),
                is_control=True,
            )
        else:
            # Intermediate frames: fire-and-forget, no queue, no rate limit.
            # WeCom does NOT count them toward the 30/min quota.
            turn_id = kwargs.get("turn_id")
            return await self._send_stream_frame_inner(text, chat=chat, reply_to=reply_to, finalize=False, turn_id=turn_id)

    async def _send_stream_frame_inner(
        self,
        text: str,
        *,
        chat: str,
        reply_to: Optional[str] = None,
        finalize: bool = False,
        turn_id: Optional[str] = None,
    ) -> bool:
        """Actual stream frame logic with per-turn state.

        Each turn (identified by chat_id + turn_id OR chat_id + req_id)
        maintains its own stream state. This prevents concurrent messages
        from interfering with each other.

        When turn_id is provided (from GatewayStreamConsumer), the turn is
        keyed by (chat, turn_id) instead of (chat, req_id). This ensures
        concurrent consumers (e.g., /background, parallel subagents) maintain
        independent streams.

        IMPORTANT: Once a turn is created, it locks to its req_id. Even if
        _last_chat_req_ids[chat] changes (e.g., user sends /approve), the
        existing turn continues with its original req_id. This prevents the
        stream from switching to a new req_id mid-turn.
        """
        try:
            # If turn_id is provided, use it to find/create the turn.
            # This is the true per-turn model that prevents concurrent
            # consumers from interfering.
            if turn_id:
                turn_key = f"{chat}:{turn_id}"
                turn = self._stream_turns.get(turn_key)
                if not turn:
                    # finalize=True should NOT create a new turn.
                    # If the turn was already cleaned up (e.g., due to errcode 6000),
                    # the caller should fallback to proactive send() instead of
                    # creating a fresh turn just to finalize it (which would send
                    # another seed + finish, potentially triggering more conflicts).
                    if finalize:
                        logger.debug(
                            "[%s] send_stream_frame: cannot finalize non-existent turn (turn_id=%s, chat=%s)",
                            self.name, turn_id, chat,
                        )
                        return False

                    # First frame for this turn: need to create it.
                    # Check if chat is expired (blocks NEW turn creation).
                    if chat in self._stream_expired_chats:
                        logger.debug(
                            "[%s] send_stream_frame: chat %s is expired, cannot create new turn (turn_id=%s)",
                            self.name, chat, turn_id,
                        )
                        return False

                    # First frame for this turn: resolve req_id and create turn
                    req_id = self._resolve_stream_req_id(chat, reply_to)
                    if not req_id:
                        logger.debug(
                            "[%s] send_stream_frame: no req_id available for chat %s (turn_id=%s)",
                            self.name, chat, turn_id,
                        )
                        return False
                    turn = StreamTurn(chat, req_id)
                    self._stream_turns[turn_key] = turn
                    logger.debug(
                        "[%s] send_stream_frame: created new turn %s (turn_id=%s, req_id=%s) for chat %s",
                        self.name, turn.stream_id, turn_id, req_id, chat,
                    )
            else:
                # Fallback: no turn_id provided (backward compatibility or direct calls).
                # Check if we already have an active turn for this chat.
                # If yes, reuse it (don't resolve req_id again).
                existing_turn = self._find_active_turn_for_chat(chat)
                if existing_turn and not existing_turn.finalized:
                    turn = existing_turn
                    logger.debug(
                        "[%s] send_stream_frame: reusing existing turn %s for chat %s",
                        self.name, turn.stream_id, chat,
                    )
                else:
                    # No active turn, need to create a new one.
                    # Check if chat is expired at the chat level (blocks NEW turn creation).
                    if chat in self._stream_expired_chats:
                        logger.debug(
                            "[%s] send_stream_frame: chat %s is expired, cannot create new turn",
                            self.name, chat,
                        )
                        return False

                    req_id = self._resolve_stream_req_id(chat, reply_to)
                    if not req_id:
                        logger.debug(
                            "[%s] send_stream_frame: no req_id available for chat %s",
                            self.name, chat,
                        )
                        return False
                    turn = self._get_or_create_stream_turn(chat, req_id)
                    logger.debug(
                        "[%s] send_stream_frame: created new turn %s (req_id=%s) for chat %s",
                        self.name, turn.stream_id, req_id, chat,
                    )

            # Check if this turn has expired
            if turn.expired:
                return False

            # First frame for this turn: send seed ONLY if not already seeded.
            # The GatewayStreamConsumer sends the initial empty seed frame itself
            # (stream_consumer.py:461), so we must not duplicate it here.
            # The seeded flag prevents double-seed which causes WeCom errcode 6000
            # (data version conflict).
            if not turn.seeded and not turn.finalized:
                # Seed frame with closed empty <think></think> — matches the
                # official OpenClaw plugin's THINKING_MESSAGE constant.  This
                # tells the WeCom client that a reasoning turn is starting;
                # subsequent frames replace it with cumulative content.
                await self._send_stream_reply(
                    turn.req_id, turn.stream_id,
                    "<think></think>", finish=False,
                )
                turn.seeded = True
                # Stream is now open on the server — arm the keep-alive timer
                # (Layer 1) so long, content-sparse turns refresh the 6-min
                # window.  No-op when keep-alive is disabled by config.
                self._arm_keepalive(turn, turn_id=turn_id)
                # If caller sent empty text (consumer's explicit seed call),
                # we're done — don't send another empty frame below.
                if not text and not finalize:
                    return True

            # Send the frame
            if finalize:
                # ── Layer 2 clock fallback ───────────────────────────────
                # If the stream is older than the safe duration, the finish
                # frame would almost certainly hit 846604/846608.  Decline it
                # up front: mark the turn expired, retire it (cancels both
                # timers), and return False so the gateway consumer's existing
                # fallback send() path delivers the content exactly once — the
                # same contract the WeComStreamExpiredError path already uses
                # (stream_consumer.py rolls back _final_content_delivered on a
                # False finalize).  Zero new uplink frames; safe for groups in
                # the sense that it does not make delivery any worse than the
                # current 846608 fallback.
                #
                # SKIP entirely when Layer 1 keep-alive is enabled: the
                # heartbeat has been refreshing the stream window every
                # STREAM_KEEPALIVE_INTERVAL_SECONDS, so an old stream_age does
                # NOT mean the stream is dead.  Declining a still-live stream on
                # a blind clock read would force the consumer's send() fallback
                # to re-deliver content the intermediate frames already put on
                # screen — the exact duplicate-bubble bug this guards against.
                # If the stream truly HAS expired, the _send_stream_reply(
                # finish=True) below will hit 846604/846608 and raise
                # WeComStreamExpiredError, which the except block turns into the
                # real (finalize-only) fallback.
                if not self._stream_keepalive_enabled:
                    stream_age = time.monotonic() - turn.start_time
                    if stream_age >= self._stream_safe_duration_seconds:
                        logger.info(
                            "[%s] Stream age %.0fs >= safe duration %.0fs for chat "
                            "%s — declining finalize frame, falling back to "
                            "proactive send (Layer 2 clock fallback).",
                            self.name, stream_age,
                            self._stream_safe_duration_seconds, chat,
                        )
                        turn.expired = True
                        self._retire_turn(turn, turn_id)
                        self._stream_expired_chats.add(chat)
                        return False

                self._cancel_idle_flush(turn)
                self._cancel_keepalive(turn)

                # WeCom may silently drop (no ack) a final frame whose content
                # is identical to the preceding intermediate frame — it treats
                # the frame as a duplicate despite the finish flag change.
                # Append a zero-width space to ensure the content differs when
                # the text matches the last ACTUALLY SENT intermediate content.
                final_text = text
                if text and text == turn.last_sent_content:
                    final_text = text + "​"  # zero-width space
                await self._send_stream_reply(
                    turn.req_id,
                    turn.stream_id,
                    final_text,
                    finish=True,
                )
                turn.finalized = True
                # Clean up this turn's state
                # If turn_id was provided, the key is chat:turn_id, otherwise chat:req_id
                if turn_id:
                    turn_key = f"{chat}:{turn_id}"
                    self._stream_turns.pop(turn_key, None)
                else:
                    self._cleanup_stream_turn(chat, turn.req_id)
            else:
                # Fire-and-forget: gateway already decides when to push
                # (pure identity-dedup in stream_consumer.py).  No adapter-
                # side buffering — send immediately when content differs
                # from the last pushed frame.  This removes the _BlockChunker
                # sentence-alignment layer whose "only grow" guard in
                # update() silently dropped frames whenever gateway-side
                # _accumulated was reset (commentary, boundary) and the new
                # cumulative text was shorter than the chunker's high-water
                # mark — the root cause of the "Cla"/"ude" split-bubble bug.
                turn.accumulated_text = text

                if turn._intermediate_frames_sent >= MAX_INTERMEDIATE_FRAMES:
                    # Frame cap reached — drop intermediates, keep accumulating.
                    # The finalize path will drain whatever is left.
                    return True

                # Pure dedup: skip if content is identical to last sent frame.
                if text == turn.last_sent_content:
                    return True

                self._cancel_idle_flush(turn)

                await self._send_stream_reply(
                    turn.req_id,
                    turn.stream_id,
                    text,
                    finish=False,
                )
                turn._last_frame_sent_at = time.monotonic()
                turn._intermediate_frames_sent += 1
                turn.last_sent_content = text

            return True

        except WeComStreamExpiredError:
            # Intermediate frames (finalize=False) are fire-and-forget: a later
            # cumulative frame — or the finalize frame — carries the full text
            # and overwrites whatever this one would have shown.  A transient
            # failure here must NOT flip the turn expired or trip the consumer's
            # send() fallback; doing so re-delivers content the stream will
            # replace anyway (duplicate bubble).  The stream is still alive
            # (keep-alive is refreshing it), so leave the turn intact and report
            # success so the consumer keeps streaming.  Only a FINAL frame's
            # expiry means the screen is genuinely missing this content and the
            # consumer must fall back.
            if not finalize:
                logger.info(
                    "[%s] Intermediate stream frame expired (errcode=%d) for "
                    "chat %s — dropping frame, stream stays live",
                    self.name, STREAM_EXPIRED_ERRCODE, chat,
                )
                return True

            logger.info(
                "[%s] Stream expired (errcode=%d) for chat %s — switching to proactive send",
                self.name, STREAM_EXPIRED_ERRCODE, chat,
            )
            # Mark this specific turn as expired and clean it up
            if 'turn' in locals():
                turn.expired = True
                self._retire_turn(turn, turn_id)

            # Mark the chat as stream-expired to prevent new stream attempts.
            # Other concurrent turns may continue if they're already active.
            self._stream_expired_chats.add(chat)
            return False
        except Exception as exc:
            # Same intermediate/final split as the expired path above: a single
            # intermediate frame failing is transient and self-healing (the next
            # cumulative frame overwrites it), so swallow it and keep the turn
            # (and its keep-alive) alive.  A final-frame failure genuinely leaves
            # the screen short of the answer, so retire the turn and let the
            # consumer's send() fallback deliver it.
            if not finalize:
                logger.info(
                    "[%s] Intermediate stream frame failed (chat=%s): %s — "
                    "dropping frame, stream stays live",
                    self.name, chat, exc,
                )
                return True

            logger.warning(
                "[%s] Stream frame failed (chat=%s): %s",
                self.name, chat, exc,
            )
            # Clean up this turn on error
            if 'turn' in locals():
                self._retire_turn(turn, turn_id)
            return False

    def supports_native_streaming(
        self,
        chat_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Probed by ``GatewayStreamConsumer`` to gate native streaming.

        WeCom AI Bot supports stream frames in both DMs and groups; group
        chats just need a cached inbound ``req_id`` (every group message
        the bot receives populates ``_last_chat_req_ids``, so this is
        effectively always satisfied for actively-used groups).
        """
        del chat_type, metadata
        return True

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """No-op: WeCom typing is handled by the stream consumer seed frame.

        The stream consumer sends an empty seed frame at the start of run(),
        which is what triggers WeCom's typing animation. _keep_typing loops
        are designed for platforms where typing expires (Telegram 5s) — WeCom
        streams stay open indefinitely, so repeated send_typing calls cause
        orphan streams. Delegating entirely to the consumer avoids the race.
        """
        del chat_id, metadata

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "group" if chat_id and chat_id.lower().startswith("group") else "dm"}


_QR_GENERATE_URL = "https://work.weixin.qq.com/ai/qc/generate"
_QR_QUERY_URL = "https://work.weixin.qq.com/ai/qc/query_result"
_QR_CODE_PAGE = "https://work.weixin.qq.com/ai/qc/gen?source=hermes&scode="
_QR_POLL_INTERVAL, _QR_POLL_TIMEOUT = 3, 300  # seconds (poll every 3s, give up after 5 minutes)


def qr_scan_for_bot_info(*, timeout_seconds: int = _QR_POLL_TIMEOUT) -> Optional[Dict[str, str]]:
    """Fetch a WeCom QR code, render it, poll until scanned or timeout; ``{"bot_id", "secret"}`` or None.
    The ``ai/qc/*`` endpoints back the admin console, not the public API, and may change."""
    import urllib.request
    import urllib.parse

    def _get_json(url: str, timeout: int) -> Dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": "HermesAgent/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _fail(log_msg: str, detail: Any, shown: Any) -> None:
        logger.error(log_msg, detail)
        print(f" failed: {shown}")

    print("  Connecting to WeCom...", end="", flush=True)
    try:
        raw = _get_json(f"{_QR_GENERATE_URL}?source=hermes", 15)
    except Exception as exc:
        return _fail("WeCom QR: failed to fetch QR code: %s", exc, exc)
    scode, auth_url = (str((raw.get("data") or {}).get(k) or "").strip() for k in ("scode", "auth_url"))
    if not scode or not auth_url:
        return _fail("WeCom QR: unexpected response format: %s", raw, "unexpected response format")
    print(" done.\n")
    page_url = f"{_QR_CODE_PAGE}{urllib.parse.quote(scode)}"
    try:
        import qrcode as _qrcode
        qr = _qrcode.QRCode()
        qr.add_data(auth_url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        print(f"\n  Scan the QR code above, or open this URL directly:\n  {page_url}")
    except Exception:
        print(f"  Open this URL in WeCom on your phone:\n\n  {page_url}\n")
        print("  Tip: pip install qrcode  to display a scannable QR code here next time")
    print("\n  Fetching configuration results...", end="", flush=True)
    deadline = time.monotonic() + timeout_seconds
    query_url = f"{_QR_QUERY_URL}?scode={urllib.parse.quote(scode)}"
    while time.monotonic() < deadline:
        try:
            result = _get_json(query_url, 10)
            print(".", end="", flush=True)  # progress dot on every poll
        except Exception as exc:
            logger.debug("WeCom QR poll error: %s", exc)
            result = {}
        result_data = result.get("data") or {}
        if str(result_data.get("status") or "").lower() != "success":
            time.sleep(_QR_POLL_INTERVAL)
            continue
        bot_info = result_data.get("bot_info") or {}
        bot_id, secret = str(bot_info.get("botid") or bot_info.get("bot_id") or "").strip(), str(bot_info.get("secret") or "").strip()
        if bot_id and secret:
            print()
            return {"bot_id": bot_id, "secret": secret}
        logger.warning("WeCom QR: scan reported success but bot_info missing or incomplete: %s", result_data)
        print("\n  QR scan reported success but no bot credentials were returned.\n  This usually means the bot was not actually created on the WeCom side.\n  Falling back to manual credential entry.")
        return None
    print(f"\n  QR scan timed out ({timeout_seconds // 60} minutes). Please try again.")
    return None


async def _send_via(adapter, chat_id, message, *, live: bool):
    try:
        result = await adapter.send(chat_id, message)
    except Exception as e:
        return send_error(f"WeCom live adapter send failed: {e}" if live else f"WeCom send failed: {e}")
    if result.success:
        return {"success": True, "platform": "wecom", "chat_id": chat_id, "message_id": result.message_id}
    return send_error(f"WeCom send failed: {result.error}")


async def _standalone_send(
    pconfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """WeCom delivery via live gateway adapter or ephemeral connection.

    Implements the standalone_sender_fn contract. WeCom only allows ONE
    WebSocket connection per bot — opening a second kicks the first. So
    when the gateway is running in-process, we reuse the live adapter.
    Only when running out-of-process (cron separate from gateway) do we
    open an ephemeral connection.
    """
    # Prefer the live gateway adapter to avoid kicking the main connection.
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except Exception:
        runner = None

    if runner is not None:
        from gateway.platforms.base import Platform
        adapter = None
        try:
            adapter = runner.adapters.get(Platform.WECOM)
        except Exception:
            pass
        if adapter is not None:
            try:
                result = await adapter.send(chat_id, message)
                if not result.success:
                    return {"error": f"WeCom send failed: {result.error}"}
                return {
                    "success": True,
                    "platform": "wecom",
                    "chat_id": chat_id,
                    "message_id": result.message_id,
                }
            except Exception as e:
                return {"error": f"WeCom live adapter send failed: {e}"}

    # Fallback: out-of-process — open ephemeral connection.
    if not check_wecom_requirements():
        return send_error("WeCom requirements not met. Need aiohttp + WECOM_BOT_ID/SECRET.")
    try:
        adapter = WeComAdapter(pconfig)
        if not await adapter.connect():
            return send_error(f"WeCom: failed to connect - {getattr(adapter, 'fatal_error_message', None) or 'unknown error'}")
        try:
            return await _send_via(adapter, chat_id, message, live=False)
        finally:
            await adapter.disconnect()
    except Exception as e:
        return send_error(f"WeCom send failed: {e}")


_MANUAL_SETUP_STEPS = (
    "1. Go to WeCom Application → Workspace → Smart Robot -> Create smart robots",
    "2. Select API Mode",
    "3. Copy the Bot ID and Secret from the bot's credentials info",
    "4. The bot connects via WebSocket — no public endpoint needed",
)
# (menu label, env saves, (print level, message)...) per unauthorized-user choice; index 3 = skip
_ACCESS_CHOICES = (
    ("Enable open access (anyone can message the bot)", (("WECOM_DM_POLICY", "open"), ("GATEWAY_ALLOW_ALL_USERS", "true")),
     (("warning", "Open access enabled — anyone can use your bot!"),)),
    ("Use DM pairing (unknown users request access, you approve with 'hermes pairing approve')", (("WECOM_DM_POLICY", "pairing"),),
     (("success", "DM pairing mode — users will receive a code to request access."), ("info", "Approve with: hermes pairing approve <platform> <code>"))),
    ("Disable direct messages", (("WECOM_DM_POLICY", "disabled"),), (("warning", "Direct messages disabled."),)),
    ("Skip for now (bot will deny all users until configured)", (), (("info", "Skipped — configure later with 'hermes gateway setup'"),)),
)


def interactive_setup() -> None:
    from hermes_cli.config import remove_env_value, save_env_value
    from hermes_cli.setup import prompt_choice
    from hermes_cli.cli_output import prompt, print_header, print_info, print_success, print_warning
    from hermes_cli.setup_platforms import declines_reconfigure
    print_header("WeCom (Enterprise WeChat)")
    if declines_reconfigure("WeCom", "Reconfigure WeCom?", "WECOM_BOT_ID"):
        return
    method_idx = prompt_choice("How would you like to set up WeCom?", ["Scan QR code to obtain Bot ID and Secret automatically (recommended)", "Enter existing Bot ID and Secret manually"], 0)
    bot_id = secret = None
    if method_idx == 0:
        try:
            credentials = qr_scan_for_bot_info() or {}
        except KeyboardInterrupt:
            print_warning("WeCom setup cancelled.")
            return
        except Exception as exc:
            print_warning(f"QR scan failed: {exc}")
            credentials = {}
        if credentials:
            bot_id, secret = credentials.get("bot_id", ""), credentials.get("secret", "")
            print_success("✔ QR scan successful! Bot ID and Secret obtained.")
        if not bot_id or not secret:
            print_info("QR scan did not complete. Continuing with manual input.")
            bot_id = secret = None
    if not bot_id or not secret:
        for line in _MANUAL_SETUP_STEPS:
            print_info(line)
        creds = []
        for label, password in (("Bot ID", False), ("Secret", True)):
            creds.append(prompt(label, password=password))
            if not creds[-1]:
                print_warning(f"Skipped — WeCom won't work without a {label}.")
                return
        bot_id, secret = creds
    save_env_value("WECOM_BOT_ID", bot_id)
    save_env_value("WECOM_SECRET", secret)
    print_info("The gateway DENIES all users by default for security.")
    print_info("Enter user IDs to create an allowlist, or leave empty.")
    allowed = prompt("Allowed user IDs (comma-separated, or empty)", password=False)
    if allowed:
        save_env_value("WECOM_ALLOWED_USERS", allowed.replace(" ", ""))
        print_success("Saved — only these users can interact with the bot.")
    else:
        access_idx = prompt_choice("How should unauthorized users be handled?", [label for label, _, _ in _ACCESS_CHOICES], 1)
        _, saves, messages = _ACCESS_CHOICES[access_idx if access_idx in (0, 1, 2) else 3]
        for key, value in saves:
            save_env_value(key, value)
        for level, message in messages:
            {"warning": print_warning, "success": print_success, "info": print_info}[level](message)
    if home := prompt("Home chat ID (optional, for cron/notifications)", password=False).strip():
        save_env_value("WECOM_HOME_CHANNEL", home)
        print_success(f"Home channel set to {home}")
    elif remove_env_value("WECOM_HOME_CHANNEL"):
        print_info("Home channel cleared.")
    print_success("💬 WeCom configured!")


def _is_connected(config) -> bool:
    return bool((getattr(config, "extra", {}) or {}).get("bot_id"))


def _callback_is_connected(config) -> bool:
    """Callback mode: corp_id or a multi-app `apps` block."""
    extra = getattr(config, "extra", {}) or {}
    return bool(extra.get("corp_id") or extra.get("apps"))



def _build_callback_adapter(config):
    from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter
    return WecomCallbackAdapter(config)


def register(ctx) -> None:
    common = dict(install_hint="Run `hermes setup` to install WeCom support.", emoji="💼", allow_update_command=True)
    ctx.register_platform(
        name="wecom", label="WeCom (Enterprise WeChat)", adapter_factory=WeComAdapter, check_fn=check_wecom_requirements,
        is_connected=_is_connected, validate_config=_is_connected, required_env=["WECOM_BOT_ID", "WECOM_SECRET"],
        setup_fn=interactive_setup, allowed_users_env="WECOM_ALLOWED_USERS", allow_all_env="WECOM_ALLOW_ALL_USERS",
        cron_deliver_env_var="WECOM_HOME_CHANNEL", standalone_sender_fn=_standalone_send, max_message_length=4000, **common,
    )
    from plugins.platforms.wecom.callback_adapter import check_wecom_callback_requirements, ensure_wecom_callback_requirements
    ctx.register_platform(
        name="wecom_callback", label="WeCom Callback (self-built apps)", adapter_factory=_build_callback_adapter,
        check_fn=check_wecom_callback_requirements, ensure_deps_fn=ensure_wecom_callback_requirements,
        is_connected=_callback_is_connected, validate_config=_callback_is_connected,
        required_env=["WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET"],
        allowed_users_env="WECOM_CALLBACK_ALLOWED_USERS", allow_all_env="WECOM_CALLBACK_ALLOW_ALL_USERS", **common,
    )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import Path  # noqa: F401,E402
import base64  # noqa: F401,E402
from dataclasses import dataclass  # noqa: F401,E402
from collections import deque  # noqa: F401,E402
import hashlib  # noqa: F401,E402
import mimetypes  # noqa: F401,E402
import os  # noqa: F401,E402
from urllib.parse import unquote  # noqa: F401,E402
from urllib.parse import urlparse  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'ABSOLUTE_MAX_BYTES': ('plugins.platforms.wecom.media', 'ABSOLUTE_MAX_BYTES'),
    'APP_CMD_UPLOAD_MEDIA_CHUNK': ('plugins.platforms.wecom.media', 'APP_CMD_UPLOAD_MEDIA_CHUNK'),
    'APP_CMD_UPLOAD_MEDIA_FINISH': ('plugins.platforms.wecom.media', 'APP_CMD_UPLOAD_MEDIA_FINISH'),
    'APP_CMD_UPLOAD_MEDIA_INIT': ('plugins.platforms.wecom.media', 'APP_CMD_UPLOAD_MEDIA_INIT'),
    'FILE_MAX_BYTES': ('plugins.platforms.wecom.media', 'FILE_MAX_BYTES'),
    'IMAGE_MAX_BYTES': ('plugins.platforms.wecom.media', 'IMAGE_MAX_BYTES'),
    'MAX_INTERMEDIATE_FRAMES': ('plugins.platforms.wecom.streaming', 'MAX_INTERMEDIATE_FRAMES'),
    'MAX_UPLOAD_CHUNKS': ('plugins.platforms.wecom.media', 'MAX_UPLOAD_CHUNKS'),
    'ReplyFrame': ('plugins.platforms.wecom.streaming', 'ReplyFrame'),
    'STREAM_EXPIRED_ERRCODE': ('plugins.platforms.wecom.streaming', 'STREAM_EXPIRED_ERRCODE'),
    'STREAM_REQUEST_EXPIRED_ERRCODE': ('plugins.platforms.wecom.streaming', 'STREAM_REQUEST_EXPIRED_ERRCODE'),
    'STREAM_VERSION_CONFLICT_ERRCODE': ('plugins.platforms.wecom.streaming', 'STREAM_VERSION_CONFLICT_ERRCODE'),
    'UPLOAD_CHUNK_SIZE': ('plugins.platforms.wecom.media', 'UPLOAD_CHUNK_SIZE'),
    'VIDEO_MAX_BYTES': ('plugins.platforms.wecom.media', 'VIDEO_MAX_BYTES'),
    'VOICE_MAX_BYTES': ('plugins.platforms.wecom.media', 'VOICE_MAX_BYTES'),
    'VOICE_SUPPORTED_MIMES': ('plugins.platforms.wecom.media', 'VOICE_SUPPORTED_MIMES'),
    'WeComStreamExpiredError': ('plugins.platforms.wecom.streaming', 'WeComStreamExpiredError'),
    'cache_document_from_bytes_async': ('gateway.platforms.base', 'cache_document_from_bytes_async'),
    'cache_image_from_bytes_async': ('gateway.platforms.base', 'cache_image_from_bytes_async'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
