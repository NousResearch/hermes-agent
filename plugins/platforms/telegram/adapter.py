"""Telegram platform adapter (python-telegram-bot): inbound messages/media/commands, outbound replies."""

import asyncio
import contextlib
import dataclasses
import inspect
import json
import logging
import os
import html as _html
import re
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Optional, Set
from hermes_cli import setup_platforms

logger = logging.getLogger(__name__)

from agent.deadline import run_bounded_async


def _redact_telegram_error_text(error: object) -> str:
    """Redact secrets from Telegram transport errors before logging or returning them."""
    text = "" if error is None else str(error)
    if not text:
        return text
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True)
    except Exception:
        return "<telegram error redacted>"


def _scoped_gate_env(name: str, default: str = "") -> str:
    """Per-profile TELEGRAM_*/GATEWAY_* gate env read (multiplex env is first-writer-wins).

    Under gateway.multiplex_profiles the process env is first-writer-wins (the YAML→env bridge in
    ``_apply_yaml_config``), so a raw ``os.getenv`` can return ANOTHER profile's allowlist (issue #72348,
    Telegram mirror). Reads the active profile's secret scope when installed; falls back to ``os.getenv``
    outside multiplex — identical single-profile behavior.
    """
    try:
        from gateway.authz_mixin import _platform_gate_env
        return _platform_gate_env(name, default)
    except Exception:
        return (os.getenv(name) or default).strip()


def _consume_abandoned_task(task: asyncio.Task) -> None:
    """Observe a detached task's terminal exception to avoid noisy loop logs."""
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("Abandoned Telegram init task failed after timeout", exc_info=True)


async def _await_with_thread_deadline(awaitable, timeout: float, *, on_abandon=None):
    """Wall-clock deadline that survives a blocked loop / cancellation-shielded PTB+httpcore init.

    ``on_abandon`` runs detached so an abandoned initialize() can't leak an httpx pool. Raises
    ``asyncio.TimeoutError`` on expiry (feeds the PTB retry ladder).

    Thin wrapper over :func:`agent.deadline.run_bounded_async` (#85125 Phase 2f) — this adapter's private
    implementation was the ancestor of that primitive and is now consolidated onto it. The unified layer
    keeps every property the 9 call sites here rely on: thread-timer deadline that survives a blocked event
    loop (#63309), abandonment of cancellation-shielded tasks (PTB/httpcore init inside anyio scopes),
    detached best-effort ``on_abandon`` cleanup so an abandoned initialize() can't leak an httpx pool per
    retry attempt, and off-loop stack-dump diagnostics when the loop never processes the expiry.
    """
    result = await run_bounded_async(awaitable, timeout, label="telegram-init", on_abandon=on_abandon)
    if result.timed_out:
        raise asyncio.TimeoutError()
    return result.value


def _iter_exception_graph(error: BaseException) -> "Iterator[BaseException]":
    """Yield ``error`` and every ``__cause__``/``__context__`` ancestor (DFS, cycle-safe) —
    PTB wraps httpx errors, so classifiers must inspect the whole graph."""
    seen: set[int] = set()
    stack: list[BaseException] = [error]
    while stack:
        cur = stack.pop()
        ident = id(cur)
        if ident in seen:
            continue
        seen.add(ident)
        yield cur
        stack.extend(x for x in (getattr(cur, "__cause__", None), getattr(cur, "__context__", None)) if x is not None)


async def _shutdown_abandoned_app(app) -> None:
    """Release a half-built PTB app's httpx transports after an abandoned init: ``app.shutdown()``
    no-ops when ``_initialized`` was never set, so the request transports are closed directly."""
    if app is None:
        return
    try:
        await app.shutdown()
    except Exception:
        logger.debug("Abandoned Telegram app.shutdown() failed", exc_info=True)
    bot = getattr(app, "bot", None)
    for request in (getattr(bot, "_request", None) if bot is not None else None) or ():
        shutdown = getattr(request, "shutdown", None)
        if shutdown is None:
            continue
        try:
            result = shutdown()
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result
        except Exception:
            logger.debug("Abandoned Telegram request shutdown failed", exc_info=True)

try:
    from telegram import Update, Bot, Message, InlineKeyboardButton, InlineKeyboardMarkup
    try:
        from telegram import LinkPreviewOptions
    except ImportError:
        LinkPreviewOptions = None
    from telegram.ext import (
        Application, CommandHandler, CallbackQueryHandler, InlineQueryHandler, MessageHandler as TelegramMessageHandler,
        ContextTypes, TypeHandler, filters)
    from telegram.constants import ParseMode, ChatType
    from telegram.request import HTTPXRequest
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = Bot = Message = InlineKeyboardButton = InlineKeyboardMarkup = Application = Any
    CommandHandler = CallbackQueryHandler = InlineQueryHandler = TypeHandler = TelegramMessageHandler = HTTPXRequest = Any
    LinkPreviewOptions = filters = ParseMode = ChatType = None

    # Mock so ContextTypes.DEFAULT_TYPE annotations don't crash class definition without the lib.
    class _MockContextTypes:
        DEFAULT_TYPE = Any
    ContextTypes = _MockContextTypes

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from gateway.authz_mixin import _coerce_allow_set
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter, MessageEvent, MessageType, ProcessingOutcome, SendResult, classify_send_error,
    cache_image_from_bytes_async, cache_audio_from_bytes_async, cache_video_from_bytes_async, resolve_proxy_url, SUPPORTED_VIDEO_TYPES,
    SUPPORTED_DOCUMENT_TYPES, SUPPORTED_IMAGE_DOCUMENT_TYPES, _TEXT_INJECT_EXTENSIONS, utf16_len)
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id
from plugins.platforms.telegram.telegram_network import (
    SEED_FALLBACK_IPS, TelegramFallbackTransport, discover_fallback_ips, parse_fallback_ip_env, tcp_keepalive_socket_options)
from utils import env_float, env_int

_TELEGRAM_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
# Max seconds a send/edit may sleep inline on a flood-control RetryAfter; longer penalties fail
# closed with ``flood_control:{wait}`` so the caller's retry machinery owns the wait.
# Longer server penalties fail closed with a ``flood_control:{wait}`` SendResult so the caller's retry
# machinery (delivery ledger, streaming fallback) owns the wait instead of the coroutine pinning its worker
# — a 97-minute penalty on the boot path froze inbound on every platform (#91969).
_FLOOD_INLINE_WAIT_CAP_SECS = 5.0


def _flood_cap_result(wait: float) -> "SendResult":
    """The shared fail-closed SendResult for an over-cap flood wait."""
    return SendResult(success=False, error=f"flood_control:{wait}", retry_after=float(wait))


_TELEGRAM_IMAGE_MIME_TO_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}
_TELEGRAM_IMAGE_EXT_TO_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


def _coerce_duration_seconds(value: Any) -> Optional[int]:
    """Round a raw length to whole positive seconds, or None if unusable."""
    try:
        secs = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return secs if secs > 0 else None


def _probe_voice_duration_seconds(path: str) -> Optional[int]:
    """Best-effort whole-second audio length (wave → mutagen → ffprobe; None if unreadable).

    Telegram renders long clips as 0:00 without an explicit duration. Blocking: use ``to_thread``."""
    if os.path.splitext(path)[1].lower() == ".wav":
        try:
            import wave
            with wave.open(path, "rb") as wf:
                rate = wf.getframerate() or 0
                secs = _coerce_duration_seconds(wf.getnframes() / float(rate)) if rate else None
            if secs is not None:
                return secs
        except Exception:
            pass
    try:
        import mutagen
        secs = _coerce_duration_seconds(getattr(getattr(mutagen.File(path), "info", None), "length", None))
        if secs is not None:
            return secs
    except Exception:
        pass
    try:
        import shutil
        import subprocess
        if shutil.which("ffprobe"):
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5)
            if proc.returncode == 0:
                return _coerce_duration_seconds(proc.stdout.strip())
    except Exception:
        pass
    return None


def telegram_deps_present() -> bool:
    """PASSIVE registry ``check_fn``: is python-telegram-bot importable? Never installs
    (``check_telegram_requirements`` is the active ``ensure_deps_fn``).

    Registry ``check_fn`` — called from status displays and config loading, so it must never install
    anything. The ACTIVE lazy-installer (``check_telegram_requirements``) is registered as
    ``ensure_deps_fn`` and runs from ``create_adapter()`` when this returns False (#79812).
    """
    return TELEGRAM_AVAILABLE


def check_telegram_requirements() -> bool:
    """Lazy-install python-telegram-bot if missing, then re-import and rebind the module aliases."""
    global TELEGRAM_AVAILABLE, Update, Bot, Message, InlineKeyboardButton
    global InlineKeyboardMarkup, LinkPreviewOptions, Application
    global CommandHandler, CallbackQueryHandler, InlineQueryHandler, TelegramMessageHandler
    global ContextTypes, filters, ParseMode, ChatType, HTTPXRequest, TypeHandler
    if TELEGRAM_AVAILABLE:
        return True
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("platform.telegram", prompt=False)
    except Exception:
        return False
    try:
        import importlib
        _tg, _ext, _const, _req = (
            importlib.import_module(m) for m in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"))
        Update, Bot, Message, InlineKeyboardButton, InlineKeyboardMarkup = (
            getattr(_tg, n) for n in ("Update", "Bot", "Message", "InlineKeyboardButton", "InlineKeyboardMarkup"))
        LinkPreviewOptions = getattr(_tg, "LinkPreviewOptions", None)
        Application, CommandHandler, CallbackQueryHandler, InlineQueryHandler, TelegramMessageHandler = (
            getattr(_ext, n) for n in ("Application", "CommandHandler", "CallbackQueryHandler", "InlineQueryHandler", "MessageHandler"))
        ContextTypes, filters, TypeHandler = _ext.ContextTypes, _ext.filters, _ext.TypeHandler
        ParseMode, ChatType = _const.ParseMode, _const.ChatType
        HTTPXRequest = _req.HTTPXRequest
    except (ImportError, AttributeError):
        return False
    TELEGRAM_AVAILABLE = True
    return True


# Every char MarkdownV2 requires backslash-escaped outside code spans/fences.
_MDV2_ESCAPE_RE = re.compile(r'([_*\[\]()~`>#\+\-=|{}.!\\])')


def _escape_mdv2(text: str) -> str:
    """Escape Telegram MarkdownV2 special characters with a preceding backslash."""
    return _MDV2_ESCAPE_RE.sub(r'\\\1', text)


def _strip_mdv2(text: str) -> str:
    """Strip MarkdownV2 escapes and formatting markers for the plain-text fallback."""
    cleaned = re.sub(r'\\([_*\[\]()~`>#\+\-=|{}.!\\])', r'\1', text)  # escape backslashes
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)  # **bold** BEFORE MarkdownV2 *bold*
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
    cleaned = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', cleaned)  # italic; word-bounded so snake_case survives
    cleaned = re.sub(r'~([^~]+)~', r'\1', cleaned)  # strikethrough
    cleaned = re.sub(r'\|\|([^|]+)\|\|', r'\1', cleaned)  # spoiler
    return cleaned


_CHUNK_INDICATOR_ON_FENCE_RE = re.compile(r'(?m)^``` (?P<indicator>(?:\\)?\(\d+/\d+(?:\\)?\))$')


def _separate_chunk_indicator_from_fence(text: str) -> str:
    """Move a ``(N/M)`` chunk marker that ``truncate_message()`` appended to a synthesized closing
    fence onto its own line — Telegram rejects ````` \\(1/2\\)`` as a fence."""
    return _CHUNK_INDICATOR_ON_FENCE_RE.sub(r'```\n\g<indicator>', text)


# MarkdownV2 has no table syntax, so pipe tables become bullet groups via convert_table_to_bullets().
from gateway.platforms.helpers import (
    TABLE_SEPARATOR_RE as _TABLE_SEPARATOR_RE, compile_mention_patterns, convert_table_to_bullets as _wrap_markdown_tables)

# Rich-message regions whose internal newlines must stay bare (Telegram renders them natively):
# fenced code blocks OR GFM pipe-table blocks (header row, delimiter row, data rows).
_RICH_PROTECTED_REGION_RE = re.compile(
    r'(?:```[^\n]*\n[\s\S]*?```)'                       # fenced code block
    r'|(?:^[^\n]*\|[^\n]*\n'                            # table header row (has a pipe)
    r'[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*'  # delimiter
    r'(?:\n[^\n]*\|[^\n]*)*)',                          # data rows (newline-led, trailing \n left for prose)
    re.MULTILINE)


def _rich_normalize_linebreaks(text: str) -> str:
    """Convert lone ``\\n`` (a Markdown soft break) to hard breaks for sendRichMessage; ``\\n\\n``,
    fenced code and pipe tables are left untouched."""
    if not text or '\n' not in text:
        return text
    out: list[str] = []
    pos = 0
    for m in _RICH_PROTECTED_REGION_RE.finditer(text):
        out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', text[pos:m.start()]))
        out.append(m.group(0))  # protected region kept verbatim
        pos = m.end()
    out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', text[pos:]))
    return ''.join(out)


# Internal safety bounds (not user knobs): no reconnect/teardown path may hang on a dead CLOSE-WAIT
# socket PTB's polling task is blocked on in epoll.
_UPDATER_STOP_TIMEOUT = 15.0  # `await updater.stop()`, applied identically at every site
_DISCONNECT_STEP_TIMEOUT = 2.0  # other disconnect() steps: short, so a swallowed cancel can't burn the fatal budget
_UPDATER_START_TIMEOUT = 30.0  # start_polling() can hang on a degraded pool after a drain
# Initial connect is unhealthy until getUpdates completes one round trip; bootstrap fails closed so
# GatewayRunner disposes the adapter and retries fresh.
# Per-step bound for disconnect() awaits that are not updater.stop() itself. Kept short so a
# cancellation-swallowing lifecycle/PTB close cannot burn the gateway's whole fatal-handler budget before
# the reconnect queue is useful (#80598). updater.stop() keeps the longer _UPDATER_STOP_TIMEOUT.
# start_polling() can also hang when the connection pool is in a degraded state after
# _drain_polling_connections(), particularly when both primary and fallback Telegram endpoints are
# unreachable. Bounding start_polling() prevents the reconnect ladder from stalling indefinitely and allows
# the heartbeat loop to trigger its own recovery path. Refs: NousResearch/hermes-agent#59614
_INITIAL_POLLING_PROGRESS_TIMEOUT = 60.0
# Bounded drain (shutdown()/initialize() of the getUpdates request) so a wedged socket can't freeze
# _polling_error_task and gate every escalation path behind its in-flight guard.
# shutdown()/initialize() on the getUpdates httpx request close and rebuild the connection pool. When a
# connection is wedged on a stale CLOSE-WAIT socket that close can block forever, hanging
# _drain_polling_connections() and freezing the whole reconnect ladder (the tracked _polling_error_task
# never completes, so every escalation path stays gated behind its in-flight guard). Bound the drain so the
# ladder always advances toward the fatal-restart escalation. Matches _UPDATER_STOP_TIMEOUT. Refs:
# NousResearch/hermes-agent#66377
_DRAIN_TIMEOUT = 15.0
# Wedged-recovery watchdog: healthy worst case is stop + 2x drain + start + 60s backoff ≈ 135s, so
# 300s in flight is unambiguously stuck and the heartbeat force-escalates.
# Every recovery path (the reconnect ladder's re-entry, the pending-update probe, PTB's error callback)
# gates new recovery on ``_polling_error_task.done()``; if that task ever wedges on a hung await that no
# local bound covers, the whole gateway goes silently deaf with nothing retrying. The heartbeat loop
# force-escalates a recovery task that stays in-flight far longer than any healthy ladder attempt could take
# — stop (_UPDATER_STOP_TIMEOUT) + drain (2x_DRAIN_TIMEOUT) + start (_UPDATER_START_TIMEOUT) + max backoff
# (60s) is ~135s, so 300s is unambiguously stuck. See #66377.
_POLLING_ERROR_TASK_STUCK_TIMEOUT = 300.0
_POLLING_PROGRESS_TIMEOUT = 60.0  # generation unhealthy until getUpdates returns; exceeds one idle long-poll
# Telegram answers a long-poll within ~50s; no round-trip for ~3x that while get_me() is healthy and
# nothing is queued means a consumer wedged on a socket that never raises (CLOSE-WAIT behind a route flip).
# Telegram holds a long-poll open for at most ~50s before answering (empty or not), so a healthy idle poller
# completes a getUpdates round-trip well inside this window. If no round-trip has completed for longer than
# this — while get_me() on the general request path stays healthy and no updates are queued server-side —
# the long-poll consumer is wedged on a socket that never raises (CLOSE-WAIT behind a TUN/proxy route flip,
# #92991) and no other probe can see it. ~3x the worst-case poll window leaves ample margin against false
# positives while still recovering within a few heartbeat intervals.
_POLLING_STALL_TIMEOUT = 150.0
# sendVideo transcodes before answering, outlasting the 20s read timeout; also how long a user waits
# to hear the attachment failed, so kept modest.
_MEDIA_SEND_READ_TIMEOUT = 60.0
_POLLING_GENERATION_CONTEXT: ContextVar[Optional[int]] = ContextVar("telegram_polling_generation", default=None)


class _PollingLifecycleAbort(RuntimeError):
    """Internal control flow for polling startup fenced by teardown."""


from .adapter_lifecycle import TelegramLifecycleMixin
from .adapter_delivery import TelegramDeliveryMixin
from .adapter_prompts import TelegramPromptsMixin
from .adapter_media import TelegramMediaMixin
from .adapter_inbound import TelegramInboundMixin
from .adapter_routing import TelegramRoutingMixin


class TelegramAdapter(
    TelegramLifecycleMixin, TelegramDeliveryMixin, TelegramPromptsMixin, TelegramMediaMixin, TelegramInboundMixin, TelegramRoutingMixin,
    BasePlatformAdapter,
):
    """Telegram bot adapter: users/groups, MarkdownV2 replies, forum topics, media."""

    MAX_MESSAGE_LENGTH = 4096
    supports_code_blocks = True  # MarkdownV2 renders fenced code blocks
    splits_long_messages = True  # send() chunks via truncate_message(MAX_MESSAGE_LENGTH)
    RICH_MESSAGE_MAX_CHARS = 32768  # Bot API 10.1 rich cap; above it use legacy chunking
    _SPLIT_THRESHOLD = 4000  # chunk near this length ⇒ a client-side split continuation is almost certain
    MEDIA_GROUP_WAIT_SECONDS = 0.8
    HELD_INBOUND_MAX = 64  # inbound events held across a disconnect window; oldest dropped first
    _GENERAL_TOPIC_THREAD_ID = "1"
    # send() can race a disconnect blip; failing "Not connected" (retryable=False) parks the answer in the
    # delivery ledger until next boot, so wait briefly for _bot (or a replacement adapter) instead.
    _RECONNECT_WAIT_SECONDS = 15.0
    _RECONNECT_POLL_INTERVAL = 0.5

    # edit_message applies MarkdownV2 only on finalize=True; without this flag stream_consumer skips
    # the final edit when raw text is unchanged.
    # Fixes #25710.
    REQUIRES_EDIT_FINALIZE: bool = True
    FALLBACK_ON_FINAL_EDIT_FLOOD: bool = True  # retrying a final edit burns the same flood budget
    RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK: bool = True  # a failed final edit may leave a partial preview

    # Adaptive text-batch ingress ("feels instant"): ≤320 codepoints settle in ~180ms, ≤1024 in ~240ms,
    # longer waits the configured cap; always clamped to ``_text_batch_delay_seconds``.
    _TEXT_BATCH_FAST_LEN = 320
    _TEXT_BATCH_FAST_DELAY_S = 0.18
    _TEXT_BATCH_SHORT_LEN = 1024
    _TEXT_BATCH_SHORT_DELAY_S = 0.24

    @staticmethod
    def _env_float_clamped(name: str, default: float, *, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
        """Read a float env var; non-finite → default; clamp to bounds (safe for asyncio.sleep)."""
        import math
        raw = os.getenv(name)
        try:
            value = float(raw) if raw is not None else float(default)
        except (TypeError, ValueError):
            value = float(default)
        if not math.isfinite(value):
            value = float(default)
        if min_value is not None:
            value = max(value, min_value)
        if max_value is not None:
            value = min(value, max_value)
        return value

    @property
    def _teardown_started(self) -> bool:
        """True once disconnect() fenced polling (tolerates object.__new__ test adapters)."""
        return getattr(self, "_polling_teardown_started", False)

    @property
    def message_len_fn(self):
        """Telegram measures message length in UTF-16 code units."""
        return utf16_len

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.TELEGRAM)
        extra = self.config.extra
        self._app: Optional[Application] = None
        self._bot: Optional[Bot] = None
        self._webhook_mode: bool = False
        self._mention_patterns = self._compile_mention_patterns()
        self._reply_to_mode: str = getattr(config, 'reply_to_mode', 'first') or 'first'
        self._disable_link_previews: bool = self._coerce_bool_extra("disable_link_previews", False)
        # Bot API 10.1 Rich Messages render what MarkdownV2 degrades (tables, task lists, <details>, block
        # math). Opt-in: current clients make rich messages hard to copy as plain text. rich_drafts is a
        # separate opt-in (Desktop can leave rich draft frames overlaid): off keeps native draft transport
        # but skips rich draft rendering; the final reply still lands via sendRichMessage.
        self._rich_messages_enabled: bool = self._coerce_bool_extra("rich_messages", False)
        self._rich_drafts_enabled: bool = self._coerce_bool_extra("rich_drafts", False)
        self._rich_send_disabled = self._rich_draft_disabled = False  # latched after a capability failure
        # Transient sendChatAction failures recur on every keep-typing tick; back off per chat.
        self._telegram_typing_cooldown_until: Dict[str, float] = {}
        self._telegram_typing_cooldown_seconds: float = self._coerce_float_extra(
            "typing_cooldown_seconds", 30.0, min_value=1.0, max_value=300.0)
        # Buffer album/photo bursts into a single MessageEvent instead of self-interrupting turns.
        self._media_batch_delay_seconds = env_float("HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS", 0.8)
        self._pending_photo_batches: Dict[str, MessageEvent] = {}
        self._pending_photo_batch_tasks: Dict[str, asyncio.Task] = {}
        self._media_group_events: Dict[str, MessageEvent] = {}
        self._media_group_tasks: Dict[str, asyncio.Task] = {}
        # Aggregate client-side splits of long messages into one MessageEvent; bounds are conservative
        # for Telegram's ~1 edit/s flood envelope.
        self._text_batch_delay_seconds = self._env_float_clamped(
            "HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS", 0.3, min_value=0.08, max_value=2.0)
        self._text_batch_split_delay_seconds = self._env_float_clamped(
            "HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS", 1.0, min_value=self._text_batch_delay_seconds, max_value=4.0)
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}
        self._drop_delayed_deliveries = False
        # Held across disconnect: PTB advances the offset before our drop-guard runs, so Telegram won't
        # redeliver — dropping is permanent loss (see _hold_inbound_event).
        self._held_inbound_events: List[MessageEvent] = []
        self._held_inbound_redispatch_task: Optional[asyncio.Task] = None
        self._polling_error_task: Optional[asyncio.Task] = None
        self._polling_progress_verifier_task: Optional[asyncio.Task] = None
        self._polling_heartbeat_task: Optional[asyncio.Task] = None
        self._bot_identity_refresh_task: Optional[asyncio.Task] = None
        self._post_connect_task: Optional[asyncio.Task] = None  # command menu + DM topics, off the connect path
        self._polling_conflict_count = self._polling_network_error_count = self._polling_generation = 0
        self._polling_conflict_recovery_generation: Optional[int] = None
        self._polling_progress_event = asyncio.Event()
        self._polling_progress_accepting = self._polling_teardown_started = False
        self._polling_error_callback_ref = None
        # Stall watchdog: generation start and last successful getUpdates (None = unknown).
        # Monotonic timestamps for the polling stall watchdog (#92991): when the current polling generation
        # began, and when the last successful getUpdates round-trip completed.
        self._polling_generation_started_monotonic: Optional[float] = None
        self._polling_last_progress_monotonic: Optional[float] = None
        # Live @username: PTB caches getMe() at initialize() and only rewrites it inside get_me(), so a
        # BotFather rename leaves self._bot.username stale; routing reads _current_bot_username().
        self._bot_username_observed: Optional[str] = None
        # None = never checked. Must NOT be 0.0: compared against time.monotonic(), which on a fresh host
        # starts near zero, so 0.0 would suppress the first refresh for a TTL.
        self._bot_identity_checked_at: Optional[float] = None
        # Consecutive heartbeat probes seeing queued updates the poller isn't consuming (get_me() can't
        # see a wedged getUpdates) / finding the updater stopped with no reconnect in flight; escalate after two.
        self._polling_pending_stuck_count = self._polling_not_running_count = 0
        # Degraded until getUpdates makes progress; while True, send() short-circuits to failure so callers
        # (cron live-adapter branch) fall through to standalone delivery.
        # Consecutive heartbeat probes that saw queued updates the running poller is not consuming. get_me()
        # can't see this — the send path is healthy while the getUpdates consumer is wedged — so the
        # heartbeat also probes get_webhook_info().pending_update_count and escalates to recovery after two
        # consecutive stuck probes (#42909).
        # Consecutive heartbeat probes that found the updater stopped entirely (running=False) while we are
        # in polling mode with no reconnect in flight. Distinct from the wedged-but-running case above: the
        # long-poll task is simply gone, so neither the connectivity probe nor PTB's error_callback ever
        # fires and the gateway silently stops receiving messages with the process still alive (#55769).
        self._send_path_degraded: bool = False
        self._general_request_drain_lock = asyncio.Lock()
        self._dm_topics: Dict[str, int] = {}  # topic_name -> message_thread_id
        self._forum_command_registered: set[int] = set()  # forum chats with commands registered
        self._forum_lock = asyncio.Lock()
        # Status indicator: bot short description "Online"/"Offline" on connect/clean disconnect. Off by
        # default because it mutates the GLOBAL profile; opt in via extra.status_indicator.
        self._status_indicator_enabled: bool = bool(extra.get("status_indicator", False))
        self._status_online_text: str = str(extra.get("status_online", "Online"))
        self._status_offline_text: str = str(extra.get("status_offline", "Offline"))
        self._dm_topics_config: List[Dict[str, Any]] = extra.get("dm_topics", [])
        # chat_ids with DM topics configured (O(1) root-DM ignore check)
        self._dm_topic_chat_ids: Set[str] = {str(e["chat_id"]) for e in self._dm_topics_config if "chat_id" in e}
        # getFile cap: 20MB on the public Bot API, 2GB on a local telegram-bot-api (base_url).
        self._max_doc_bytes: int = 2 * 1024 * 1024 * 1024 if extra.get("base_url") else 20 * 1024 * 1024
        self._model_picker_state: Dict[str, dict] = {}  # per-chat interactive picker state
        self._choice_picker_state: Dict[str, dict] = {}
        self._approval_state: Dict[int, str] = {}  # message_id → session_key
        self._slash_confirm_state: Dict[str, str] = {}  # confirm_id → session_key
        self._clarify_state: Dict[str, str] = {}  # clarify_id → session_key
        # "important" (default): only final responses, approvals and slash confirmations notify;
        # "all": every message notifies (display.platforms.telegram.notifications).
        self._notifications_mode: str = "important"
        # send_or_update_status(): {(chat_id, status_key) -> message_id} so repeat calls edit in place.
        # send_or_update_status() bookkeeping: {(chat_id, status_key) -> bot message_id} Tracks status
        # bubbles owned by this adapter so subsequent calls with the same key edit the same message instead
        # of appending new ones (#30045).
        self._status_message_ids: Dict[tuple, str] = {}
        # Last truncated mid-stream preview per (chat_id, message_id): past the 4096 cap every edit
        # truncates to the SAME text, and resending burns flood budget. Dropped on finalize.
        self._last_overflow_preview: Dict[tuple, str] = {}

    @property
    def send_path_degraded(self) -> bool:
        # True from polling-generation start until the first getUpdates
        # round-trip is proven (_record_polling_progress), and again at every
        # polling-death site. getattr: tests build adapters via object.__new__().
        return bool(getattr(self, "_send_path_degraded", False))

    def _mark_connected(self) -> None:
        self._drop_delayed_deliveries = False
        super()._mark_connected()
        self._schedule_held_inbound_redispatch()  # PTB will not redeliver these events

    def _mark_disconnected(self) -> None:
        self._drop_delayed_deliveries = True
        super()._mark_disconnected()

    def _set_fatal_error(self, code: str, message: str, *, retryable: bool) -> None:
        self._drop_delayed_deliveries = True
        super()._set_fatal_error(code, message, retryable=retryable)
        # Permanent fatal: no reconnect will drain, so discard the hold queue (later holds are refused).
        # Discard the hold queue now and refuse further holds (teardown salvage / late enqueue must not
        # re-populate a queue that can never drain — review #83878).
        if not retryable:
            held = getattr(self, "_held_inbound_events", None)
            n = len(held) if held else 0
            if held:
                held.clear()
            if n:
                logger.warning("[Telegram] Non-retryable fatal (%s); discarding %d held inbound message(s)", code, n)

    def _is_permanent_fatal(self) -> bool:
        """True after non-retryable fatal — holds must discard, not queue."""
        if not getattr(self, "_fatal_error_code", None):
            return False
        return not bool(getattr(self, "_fatal_error_retryable", True))

    def _replacement_telegram_adapter(self) -> Optional["TelegramAdapter"]:
        """Live adapter if the reconnect watcher replaced us in ``runner.adapters`` (an in-flight
        ``send()`` still holds the old instance whose ``_bot`` stays None)."""
        runner = getattr(self, "gateway_runner", None)
        adapters = getattr(runner, "adapters", None) or {}
        live = adapters.get(self.platform)
        if live is not None and live is not self and getattr(live, "_bot", None):
            return live
        return None

    async def _wait_for_reconnection(self) -> bool:
        """Wait for ``_bot`` or a replacement adapter; False on expiry or permanent fatal."""
        if self._bot or self._replacement_telegram_adapter() is not None:
            return True
        if self._is_permanent_fatal():
            return False
        wait_s = float(getattr(self, "_RECONNECT_WAIT_SECONDS", 15.0))
        poll_s = float(getattr(self, "_RECONNECT_POLL_INTERVAL", 0.5))
        logger.info("[%s] Not connected — waiting for reconnection (up to %.0fs)", self.name, wait_s)
        waited = 0.0
        while waited < wait_s:
            await asyncio.sleep(poll_s)
            waited += poll_s
            if self._is_permanent_fatal():
                return False
            if self._bot or self._replacement_telegram_adapter() is not None:
                logger.info("[%s] Reconnected after %.1fs", self.name, waited)
                return True
        logger.warning("[%s] Still not connected after %.0fs", self.name, wait_s)
        return False

    def _should_drop_delayed_delivery(self) -> bool:
        """True once teardown/fatal started: delayed flushes must not dispatch onto a torn-down session.
        Callers must NOT destroy the event (PTB already advanced the offset) — hold and redispatch."""
        return bool(getattr(self, "_drop_delayed_deliveries", False))

    def _schedule_held_inbound_redispatch(self) -> None:
        """Ensure a tracked drain runs when held events exist and delivery is live (no-op while
        down or after permanent fatal; an in-flight drain schedules its own follow-up)."""
        if self._is_permanent_fatal() or self._should_drop_delayed_delivery():
            return
        if not getattr(self, "_held_inbound_events", None):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        prior = getattr(self, "_held_inbound_redispatch_task", None)
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if prior is not None and not prior.done() and prior is not current:
            return
        self._held_inbound_redispatch_task = loop.create_task(self._redispatch_held_inbound(prior=None if prior is current else prior))

    def _hold_inbound_event(self, event: "MessageEvent", *, where: str, schedule: bool = True) -> None:
        """Preserve an inbound event that cannot be dispatched now (PTB already acked the update, so dropping is silent loss).
        Capped, identity-deduped; permanent fatal discards. ``schedule=False`` inside a drain avoids poison-event loops.

        The disconnect drop-guard (#55971) correctly prevents dispatch into a torn-down session. Destroying
        the event is wrong: by the time we reach enqueue/flush, python-telegram-bot has already acked the
        update and advanced the offset — silent permanent loss, no log, no error.
        """
        if self._is_permanent_fatal():
            logger.warning(
                "[Telegram] Discarding inbound under non-retryable fatal (%s, %d chars)", where, len(getattr(event, "text", None) or ""))
            return
        held = getattr(self, "_held_inbound_events", None)
        if held is None:
            self._held_inbound_events = held = []
        if any(existing is event for existing in held):
            return
        max_n = int(getattr(self, "HELD_INBOUND_MAX", 64) or 64)
        while len(held) >= max_n:
            dropped = held.pop(0)
            logger.warning(
                "[Telegram] Held-inbound queue full (%d); dropping oldest (%d chars)", max_n, len(getattr(dropped, "text", None) or ""))
        held.append(event)
        logger.warning(
            "[Telegram] Holding inbound (%s, %d chars, queue=%d)%s", where, len(getattr(event, "text", None) or ""), len(held),
            " - will redispatch on reconnect" if self._should_drop_delayed_delivery() else (" - scheduling redispatch" if schedule else ""))
        # A live-path hold must not orphan the event waiting for a reconnect that never comes.
        if schedule and not self._should_drop_delayed_delivery():
            self._schedule_held_inbound_redispatch()

    def _rehold_from(self, events: list, idx: int, where: str) -> None:
        """Re-hold ``events[idx:]`` without rescheduling (drain interrupted / failed / cancelled)."""
        for rest in events[idx:]:
            self._hold_inbound_event(rest, where=where, schedule=False)

    async def _redispatch_held_inbound(self, prior: Optional[asyncio.Task] = None) -> None:
        """Drain the hold queue after reconnect or a connected-path hold; ``prior`` (previous
        redispatch task) is cancelled+awaited here so ``_mark_connected`` stays synchronous."""
        if prior is not None and prior is not asyncio.current_task() and not prior.done():
            prior.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await prior
        held = getattr(self, "_held_inbound_events", None)
        if self._is_permanent_fatal():
            if held:
                n = len(held)
                held.clear()
                logger.warning("[Telegram] Redispatch aborted; discarded %d held inbound under non-retryable fatal", n)
            return
        if not held:
            return
        # Take ownership atomically; concurrent holds append to the fresh list for a follow-up.
        events = list(held)
        held.clear()
        logger.warning("[Telegram] Redispatching %d held inbound message(s)", len(events))
        allow_followup_schedule = True
        try:
            for idx, event in enumerate(events):
                if self._is_permanent_fatal() or self._should_drop_delayed_delivery():
                    self._rehold_from(events, idx, "redispatch-interrupted")
                    return
                try:
                    await self.handle_message(event)
                except asyncio.CancelledError:
                    self._rehold_from(events, idx, "redispatch-cancelled")
                    raise
                except Exception:
                    # Retryable failure: re-hold but do NOT reschedule now (a poison event would
                    # tight-loop); the next mark_connected/live hold drains.
                    logger.exception(
                        "[Telegram] Failed to redispatch held inbound (%d chars); re-holding", len(getattr(event, "text", None) or ""))
                    self._rehold_from(events, idx, "redispatch-failed")
                    allow_followup_schedule = False
                    return
        finally:
            # Events that arrived mid-drain while still connected need another pass.
            if (
                allow_followup_schedule
                and getattr(self, "_held_inbound_events", None)
                and not self._should_drop_delayed_delivery()
                and not self._is_permanent_fatal()):
                self._schedule_held_inbound_redispatch()

    def _fallback_ips(self) -> list[str]:
        """Return validated fallback IPs from config (populated by _apply_env_overrides)."""
        configured = self.config.extra.get("fallback_ips", []) if getattr(self.config, "extra", None) else []
        if isinstance(configured, str):
            configured = configured.split(",")
        return parse_fallback_ip_env(",".join(str(v) for v in configured) if configured else None)

    @staticmethod
    def _looks_like_polling_conflict(error: Exception) -> bool:
        text = str(error).lower()
        return (
            error.__class__.__name__.lower() == "conflict"
            or "terminated by other getupdates request" in text
            or "another bot instance is running" in text)

    @staticmethod
    def _looks_like_auth_error(error: Exception) -> bool:
        """True for terminal credential failures (InvalidToken, Forbidden) → retryable=False. Type-based
        only, never message text; BadRequest/RetryAfter are transient at connect time."""
        if error.__class__.__name__.lower() in {"invalidtoken", "forbidden"}:
            return True
        try:
            from telegram.error import Forbidden, InvalidToken
            return isinstance(error, (InvalidToken, Forbidden))
        except ImportError:
            return False

    @staticmethod
    def _looks_like_network_error(error: Exception) -> bool:
        """Return True for transient transport failures that warrant reconnect."""
        name = error.__class__.__name__.lower()
        if name in {"badrequest", "invalidtoken", "forbidden", "retryafter"}:
            return False
        if name in {"networkerror", "timedout", "connectionerror"}:
            return True
        try:
            from telegram.error import BadRequest, Forbidden, InvalidToken, NetworkError, RetryAfter, TimedOut
            if isinstance(error, (BadRequest, InvalidToken, Forbidden, RetryAfter)):
                return False
            if isinstance(error, (NetworkError, TimedOut)):
                return True
        except ImportError:
            pass
        return isinstance(error, OSError)

    @staticmethod
    def _exception_graph_matches(error: Exception, name_marker: str, *text_markers: str) -> bool:
        """True when any exception in ``error``'s cause/context graph matches by class name or text."""
        for cur in _iter_exception_graph(error):
            text = str(cur).lower()
            if name_marker in cur.__class__.__name__.lower() or any(m in text for m in text_markers):
                return True
        return False

    @classmethod
    def _looks_like_connect_timeout(cls, error: Exception) -> bool:
        """True when a TimedOut wraps a ConnectTimeout: TCP never connected, so re-sending is safe
        (a plain TimedOut may have reached Telegram and must not be re-sent)."""
        return cls._exception_graph_matches(error, "connecttimeout", "connect timeout", "connect timed out")

    @staticmethod
    def _looks_like_pool_timeout(error: Exception) -> bool:
        """True when a TimedOut wraps ``httpx.PoolTimeout``: PTB says "Request was *not* sent", so
        re-sending cannot duplicate. Matches class AND text to survive rewording."""
        for cur in _iter_exception_graph(error):
            name = cur.__class__.__name__.lower()
            text = str(cur).lower()
            if "pooltimeout" in name or "pool timeout" in text or ("connection pool" in text and "occupied" in text):
                return True
        return False

    def _coerce_bool_extra(self, key: str, default: bool = False) -> bool:
        value = self.config.extra.get(key) if getattr(self.config, "extra", None) else None
        if value is None:
            return default
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
            return default
        return bool(value)

    def _coerce_float_extra(
        self, key: str, default: float, *, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
        value = self.config.extra.get(key) if getattr(self.config, "extra", None) else None
        if value is None:
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if min_value is not None:
            parsed = max(parsed, min_value)
        if max_value is not None:
            parsed = min(parsed, max_value)
        return parsed

    _RICH_DETAILS_RE = re.compile(r"<details\b[^>]*>.*?</details>", re.IGNORECASE | re.DOTALL)
    _RICH_MATH_IN_DETAILS_RE = re.compile(
        r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|"
        r"\\(?:sum|frac|alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|"
        r"int|prod|sqrt|lim|infty|begin\{(?:equation|align|matrix|cases)\}))",
        re.IGNORECASE | re.DOTALL)
    # Hiragana/Katakana, CJK Ext A, CJK Unified, Hangul, CJK Compatibility, CJK ext/compat supplement.
    _RICH_CJK_RE = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\U00020000-\U000323af]")

    # Template attrs for the shared _format_exec_approval core (HTML mode).
    _EA_HEADER = "⚠️ <b>Command Approval Required</b>\n\n"
    _EA_CODE_OPEN = "<pre>"
    _EA_CODE_CLOSE = "</pre>\n\n"
    _EA_SMART_DENY_LINE = "\n\n<b>Smart DENY:</b> owner override applies to this one operation only."
    _EA_CMD_BUDGET = 3800

    _PROVIDER_PAGE_SIZE = 10

    _MODEL_PAGE_SIZE = 8

    # `gt:<verb>` -> (script in ~/.hermes/scripts/gmail-triage/, extra-args, success-label, is_state). The callback
    # `arg` is always the first positional arg. is_state=True keeps the keyboard tappable (sticky sender rule);
    # False strips it on success (per-email one-shot).
    _GT_VERB_DISPATCH = {
        "send":         ("send-draft.sh",      [],         "✓ sent draft",         False),
        "archive":      ("archive.sh",         [],         "✓ archived",           False),
        "draft":        ("draft-blank.sh",     [],         "✓ drafted reply",      False),
        "spam":         ("spam.sh",            [],         "✓ marked spam",        False),
        "mute":         ("mute-add.sh",        ["email"],  "✓ muted",              True),
        "mute-domain":  ("mute-add.sh",        ["domain"], "✓ muted domain",       True),
        "trust":        ("trusted-ops-add.sh", ["email"],  "✓ trusted",            True),
        "trust-domain": ("trusted-ops-add.sh", ["domain"], "✓ trusted domain",     True),
        "vip":          ("vip-add.sh",         ["email"],  "✓ marked VIP",         True),
        "vip-domain":   ("vip-add.sh",         ["domain"], "✓ marked VIP domain",  True)}

    # ── Group mention gating ──────────────────────────────────────────────

    # Decides only whether a FOREIGN @handle is bot-shaped; our own handle is matched by identity, never
    # shape (collectible/Fragment bot usernames need not end in "bot").
    _FOREIGN_BOT_HANDLE_RE = re.compile(r"[a-z0-9_]{2,29}bot", re.IGNORECASE)
    _BOT_IDENTITY_TTL_SECONDS = 300.0  # how long an observed identity is trusted before re-check

    _BOT_IDENTITY_PROBE_TIMEOUT = 15.0

    _CACHED_KIND_TO_MESSAGE_TYPE = {"image": MessageType.PHOTO, "video": MessageType.VIDEO, "audio": MessageType.AUDIO}

    # -- Text message aggregation (handles Telegram client-side splits) --

    # -- Photo batching --

    # -- Message reactions (processing lifecycle) --

    def _reactions_enabled(self) -> bool:
        """Reactions enabled via TELEGRAM_REACTIONS env/config."""
        return os.getenv("TELEGRAM_REACTIONS", "false").lower() not in {"false", "0", "no"}

    async def _set_reaction(self, chat_id: str, message_id: str, emoji: Optional[str]) -> bool:
        """Set a single emoji reaction (``None`` clears all bot-set reactions, the documented Bot API way)."""
        if not self._bot:
            return False
        try:
            await self._bot.set_message_reaction(chat_id=normalize_telegram_chat_id(chat_id), message_id=int(message_id), reaction=emoji)
            return True
        except Exception as e:
            if emoji is None:
                logger.debug("[%s] clear reactions failed: %s", self.name, _redact_telegram_error_text(e))
            else:
                logger.debug("[%s] set_message_reaction failed (%s): %s", self.name, emoji, _redact_telegram_error_text(e))
            return False

    async def _clear_reactions(self, chat_id: str, message_id: str) -> bool:
        """Clear all bot-set reactions."""
        return await self._set_reaction(chat_id, message_id, None)

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add an in-progress reaction when message processing begins."""
        if not self._reactions_enabled():
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if chat_id and message_id:
            await self._set_reaction(chat_id, message_id, "\U0001f440")

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Swap the in-progress reaction for a final success/failure reaction (set_message_reaction
        replaces, not adds); CANCELLED explicitly clears the 👀."""
        if not self._reactions_enabled():
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if not (chat_id and message_id):
            return
        if outcome == ProcessingOutcome.CANCELLED:
            await self._clear_reactions(chat_id, message_id)
        else:
            await self._set_reaction(chat_id, message_id, "\U0001f44d" if outcome == ProcessingOutcome.SUCCESS else "\U0001f44e")


# -- Plugin registration glue: register(ctx) plus the hook implementations (adapter factory, YAML→env/extra
# config, setup wizard, standalone sender).


# ────────────────────────────────────────────────────────────────────────── Plugin migration glue (#41112 /
# #3823) Added when the Telegram adapter (+ its telegram_network satellite) moved from gateway/platforms/
# into this bundled plugin. Mirrors the Discord (#24356) / Slack migrations: a register(ctx) entry point
# plus hook implementations that replace the per-platform core touchpoints (the Platform.TELEGRAM branch in
# gateway/run.py, the telegram_cfg YAML→env/extra block in gateway/config.py, the _setup_telegram wizard +
# _PLATFORMS["telegram"] static dict in hermes_cli/{setup,gateway}.py, and the _send_telegram dispatch in
# tools/send_message_tool.py). Telegram uses the generic token connected check, so no is_connected override
# is needed. ──────────────────────────────────────────────────────────────────────────
def _resolve_notifications_mode() -> str:
    """Notification mode (all/important) from env, else config.yaml display.platforms.telegram.notifications."""
    mode = os.getenv("HERMES_TELEGRAM_NOTIFICATIONS", "")
    if not mode:
        try:
            from gateway.config import load_gateway_config
            from gateway.run import cfg_get
            _raw = cfg_get(load_gateway_config(), "display", "platforms", "telegram", "notifications")
            if _raw not in {None, ""}:
                mode = str(_raw).strip().lower()
        except Exception:
            pass
    mode = mode or "important"
    if mode not in {"all", "important"}:
        logger.warning("Unknown telegram notifications mode '%s', defaulting to 'important' (valid: all, important)", mode)
        mode = "important"
    return mode


def _build_adapter(config):
    """Construct TelegramAdapter and apply the notification mode."""
    adapter = TelegramAdapter(config)
    try:
        adapter._notifications_mode = _resolve_notifications_mode()
    except Exception:
        adapter._notifications_mode = "important"
    return adapter


def _is_connected(config) -> bool:
    """Connected when a bot token is configured (env or PlatformConfig.token); the SDK being importable is
    not enough or the plugin-enable pass would enable Telegram on any machine with it installed."""
    token = getattr(config, "token", None)
    if not token:
        import hermes_cli.gateway as gateway_mod
        token = gateway_mod.get_env_value("TELEGRAM_BOT_TOKEN") or ""
    return bool(str(token).strip())


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """Out-of-process delivery (standalone_sender_fn) so deliver=telegram cron jobs succeed without the
    gateway; delegates to the REST ``_send_telegram`` sender."""
    token = getattr(pconfig, "token", None)
    if not token:
        from agent.secret_scope import get_secret  # profile-scoped: never borrow another profile's token
        token = get_secret("TELEGRAM_BOT_TOKEN", "") or ""
    disable_link_previews = bool(getattr(pconfig, "extra", {}) and pconfig.extra.get("disable_link_previews"))
    from tools.send_message_tool import _send_telegram
    return await _send_telegram(
        token, chat_id, message, media_files=media_files, thread_id=thread_id,
        disable_link_previews=disable_link_previews, force_document=force_document)


def interactive_setup() -> None:
    """Configure Telegram credentials and allowlist via the CLI setup wizard (lazy import)."""
    from hermes_cli import setup as _setup_mod
    setup_platforms._setup_telegram()


def _apply_yaml_config(yaml_cfg: dict, telegram_cfg: dict) -> dict | None:
    """Translate config.yaml telegram: keys into TELEGRAM_* env vars and PlatformConfig.extra. Env vars
    take precedence over YAML. Returns extras to merge into PlatformConfig.extra, or None.

    Implements the apply_yaml_config_fn contract (#24849). Mirrors the legacy telegram_cfg block from
    gateway/config.py::load_gateway_config().
    """
    import json as _json
    extras: dict = {}
    # Under multiplex a secondary profile's authorization gates must NOT hit the process-global env
    # (first-writer-wins would pin them for every profile); they flow via extra/secret scope.
    try:
        # See #72348.
        from agent.secret_scope import current_secret_scope, is_multiplex_active
        _skip_env_bridge = bool(is_multiplex_active() and current_secret_scope() is not None)
    except Exception:
        _skip_env_bridge = False

    def _set_env(env: str, value: str) -> None:
        if not os.getenv(env):
            os.environ[env] = value

    def _bridge_lower(key: str, env: str) -> None:
        if key in telegram_cfg:
            _set_env(env, str(telegram_cfg[key]).lower())

    def _bridge_gate(key: str, env: str, value: Any, *, seed_extra: bool = False) -> None:
        """CSV allowlist gate: list → comma-joined; skipped under multiplex secret scope."""
        if value is None:
            return
        if seed_extra:
            extras.setdefault(key, value)
        if isinstance(value, list):
            value = ",".join(str(v) for v in value)
        if not _skip_env_bridge:
            _set_env(env, str(value))

    if "disable_topic_auto_rename" in telegram_cfg:
        extras.setdefault("disable_topic_auto_rename", telegram_cfg["disable_topic_auto_rename"])
    _effective_rm = telegram_cfg.get("require_mention", yaml_cfg.get("require_mention"))
    if _effective_rm is not None:
        _set_env("TELEGRAM_REQUIRE_MENTION", str(_effective_rm).lower())
    if "mention_patterns" in telegram_cfg:
        _set_env("TELEGRAM_MENTION_PATTERNS", _json.dumps(telegram_cfg["mention_patterns"]))
    for key, env in (
        ("exclusive_bot_mentions", "TELEGRAM_EXCLUSIVE_BOT_MENTIONS"), ("allow_bots", "TELEGRAM_ALLOW_BOTS"),
        ("guest_mode", "TELEGRAM_GUEST_MODE", ), ("observe_unmentioned_group_messages", "TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES")):
        _bridge_lower(key, env)
    # No extras seed for allowed_chats / allowed_topics / group_allowed_chats: the shared-key loop already
    # bridges them with their original type and this merge would clobber it.
    for key, env, seed in (
        ("free_response_chats", "TELEGRAM_FREE_RESPONSE_CHATS", True), ("free_response_topics", "TELEGRAM_FREE_RESPONSE_TOPICS", False),
        ("allowed_chats", "TELEGRAM_ALLOWED_CHATS", False), ("allowed_topics", "TELEGRAM_ALLOWED_TOPICS", False),
        ("ignored_threads", "TELEGRAM_IGNORED_THREADS", True)):
        _bridge_gate(key, env, telegram_cfg.get(key), seed_extra=seed)
    _bridge_lower("reactions", "TELEGRAM_REACTIONS")
    if "proxy_url" in telegram_cfg:
        _set_env("TELEGRAM_PROXY", str(telegram_cfg["proxy_url"]).strip())
    _telegram_extra = telegram_cfg.get("extra") if isinstance(telegram_cfg.get("extra"), dict) else {}
    _telegram_rtm = telegram_cfg["reply_to_mode"] if "reply_to_mode" in telegram_cfg else _telegram_extra.get("reply_to_mode")
    if _telegram_rtm is not None:
        _set_env("TELEGRAM_REPLY_TO_MODE", "off" if _telegram_rtm is False else str(_telegram_rtm).lower())
    _bridge_gate("allow_from", "TELEGRAM_ALLOWED_USERS", telegram_cfg.get("allow_from"))
    _bridge_gate(
        "group_allow_from", "TELEGRAM_GROUP_ALLOWED_USERS", telegram_cfg.get("group_allow_from") or _telegram_extra.get("group_allow_from"))
    _bridge_gate(
        "group_allowed_chats", "TELEGRAM_GROUP_ALLOWED_CHATS",
        telegram_cfg.get("group_allowed_chats") or _telegram_extra.get("group_allowed_chats"))
    for _key in ("guest_mode", "disable_link_previews", "observe_unmentioned_group_messages", "free_response_topics"):
        if _key in telegram_cfg:
            extras.setdefault(_key, telegram_cfg[_key])
    # Pass through telegram-specific extra keys but EXCLUDE generic shared-config keys: _merge_platform_map
    # already applied top-level-over-nested precedence and re-emitting them via dict.update() would undo it.
    _GENERIC_MERGE_KEYS = {
        "reply_prefix", "reply_in_thread", "reply_to_mode", "unauthorized_dm_behavior", "notice_delivery",
        "require_mention", "channel_skill_bindings", "channel_prompts", "gateway_restart_notification", "allow_from",
        "allow_admin_from", "dm_policy", "group_policy"}
    for _k, _v in _telegram_extra.items():
        if _k not in _GENERIC_MERGE_KEYS:
            extras.setdefault(_k, _v)
    return extras or None


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="telegram", label="Telegram", adapter_factory=_build_adapter, check_fn=telegram_deps_present,
        ensure_deps_fn=check_telegram_requirements, is_connected=_is_connected, required_env=["TELEGRAM_BOT_TOKEN"],
        install_hint="Run `hermes setup` to install Telegram support.", setup_fn=interactive_setup, apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="TELEGRAM_ALLOWED_USERS", allow_all_env="TELEGRAM_ALLOW_ALL_USERS", cron_deliver_env_var="TELEGRAM_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send, max_message_length=4096, emoji="✈️", allow_update_command=True)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import threading  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'atomic_replace': ('utils', 'atomic_replace'),
    'cache_document_from_bytes': ('gateway.platforms.base', 'cache_document_from_bytes'),
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
