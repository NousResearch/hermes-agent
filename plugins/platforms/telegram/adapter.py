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


def _callback_identity(value: Any) -> Optional[str]:
    """Normalize a Telegram id for comparison (int on the wire, str in config/state)."""
    return None if value is None else str(value)


@dataclasses.dataclass(frozen=True)
class _ClarifyPrompt:
    """A pending clarify's session key plus the identity of the message that rendered its buttons.

    Telegram message ids are unique within a chat, so ``(chat_id, message_id)`` names this exact
    prompt.  A ``cl:`` callback carrying a matching clarify id but arriving on some *other* message
    cannot be a deliberate tap on this one and must not resolve it (#102957).  ``thread_id`` is
    recorded for the audit line only: it is implied by the message, so gating on it would add false
    rejects (Telegram omits it for a forum's General topic) without narrowing anything.
    """

    session_key: str
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    thread_id: Optional[str] = None


class TelegramAdapter(BasePlatformAdapter):
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
        self._clarify_state: Dict[str, "_ClarifyPrompt"] = {}  # clarify_id → prompt binding (#102957)
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

    def _notification_kwargs(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """In "important" mode return disable_notification=True unless ``metadata["notify"]``."""
        if getattr(self, "_notifications_mode", "important") != "important" or (metadata or {}).get("notify"):
            return {}
        return {"disable_notification": True}

    @staticmethod
    def _normalize_chat_type(chat_type: Any, *, is_forum: bool) -> str:
        """Telegram chat type → gateway chat type (``private``→``dm``, ``supergroup``→forum/group)."""
        normalized = str(chat_type or "dm").strip().lower() or "dm"
        if normalized == "private":
            return "dm"
        if normalized == "supergroup":
            return "forum" if is_forum else "group"
        return normalized

    def _legacy_runner_auth_fn(self):
        """``runner._is_user_authorized`` resolved off the bound handler (bare-adapter tests, direct
        embedding); None under multiplex where the handler is a profile closure."""
        # Resolve through the runner's full auth chain (platform + group allowlists, pairing store,
        # allow-all flags). Prefer the platform-bound callback registered via set_authorization_check: it
        # routes to GatewayRunner._is_user_authorized AND survives multiplex handler wrapping, whereas the
        # bound-handler __self__ lookup is None when the primary handler is a profile closure — which
        # silently dropped the chat allowlist and default-denied allowlisted group members under
        # multiplex_profiles (#87132). Fall back to the bound handler for setups without a registered
        # callback.
        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        auth_fn = getattr(runner, "_is_user_authorized", None)
        return auth_fn if callable(auth_fn) else None

    @staticmethod
    def _env_allowlist_decision(user_id: str) -> Optional[bool]:
        """TELEGRAM_ALLOWED_USERS decision; None when no allowlist is configured."""
        allowed_csv = _scoped_gate_env("TELEGRAM_ALLOWED_USERS").strip()
        if not allowed_csv:
            return None
        allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
        return "*" in allowed_ids or user_id in allowed_ids

    def _is_callback_user_authorized(
        self, user_id: str, *, chat_id: Optional[str] = None, chat_type: Optional[str] = None,
        thread_id: Optional[str] = None, user_name: Optional[str] = None) -> bool:
        """Return whether a Telegram inline-button caller may perform gated actions."""
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return False
        normalized_chat_type = self._normalize_chat_type(chat_type, is_forum=thread_id is not None)
        # Preferred: the auth callback GatewayRunner injects (set_authorization_check) → full
        # _is_user_authorized chain; also works for a multiplexed adapter whose _message_handler is a
        # profile closure. getattr tolerates partially-constructed adapters (object.__new__ in tests).
        if getattr(self, "_authorization_check", None) is not None:
            injected = self._is_sender_authorized(
                normalized_user_id, chat_type=normalized_chat_type, chat_id=str(chat_id or normalized_user_id),
                thread_id=str(thread_id) if thread_id is not None else None)
            if injected is not None:
                return injected
        auth_fn = self._legacy_runner_auth_fn()
        if auth_fn is not None:
            try:
                from gateway.session import SessionSource
                source = SessionSource(
                    platform=Platform.TELEGRAM, chat_id=str(chat_id or normalized_user_id), chat_type=normalized_chat_type,
                    user_id=normalized_user_id, user_name=str(user_name).strip() if user_name else None,
                    thread_id=str(thread_id) if thread_id is not None else None)
                return bool(auth_fn(source))
            except Exception:
                logger.debug(
                    "[Telegram] Falling back to env-only callback auth for user %s", normalized_user_id, exc_info=True)
        decision = self._env_allowlist_decision(normalized_user_id)
        if decision is None:
            # Fail-closed: no allowlist means deny unless GATEWAY_ALLOW_ALL_USERS is set.
            # The runner auth path in _is_user_authorized() handles GATEWAY_ALLOW_ALL_USERS; this fallback
            # must not silently allow everyone (fixes #24457).
            return _scoped_gate_env("GATEWAY_ALLOW_ALL_USERS").lower() in {"true", "1", "yes"}
        return decision

    def _source_from_message_for_auth(self, message: Message):
        """Build the SessionSource the gateway auth path expects; identity comes from ``from_user``,
        falling back to ``sender_chat`` for channel posts so an unauthorized channel can't inject."""
        from gateway.session import SessionSource
        user = getattr(message, "from_user", None)
        chat = getattr(message, "chat", None)
        user_id = str(getattr(user, "id", "")).strip() or None
        # Carry is_bot so the runner's ``*_ALLOW_BOTS`` branch is reachable, as in build_source.
        is_bot = bool(getattr(user, "is_bot", False)) if user is not None else False
        user_name = str(getattr(user, "username", "") or getattr(user, "full_name", "") or "").strip() or None
        if not user_id:  # channel post — authorize the sender chat instead
            sender_chat = getattr(message, "sender_chat", None)
            if sender_chat is not None:
                user_id = str(getattr(sender_chat, "id", "")).strip() or None
                if not user_name:
                    user_name = str(getattr(sender_chat, "title", "") or "").strip() or None
        chat_id = str(getattr(chat, "id", "")).strip() or user_id
        thread_id_raw = getattr(message, "message_thread_id", None)
        is_topic_message = bool(getattr(message, "is_topic_message", False))
        is_forum_group = getattr(chat, "is_forum", False) is True
        chat_type = self._normalize_chat_type(
            getattr(chat, "type", "dm"), is_forum=thread_id_raw is not None and (is_topic_message or is_forum_group))
        thread_id = None
        if thread_id_raw is not None and (
            (chat_type == "forum" and (is_topic_message or is_forum_group)) or (chat_type == "dm" and is_topic_message)):
            thread_id = str(thread_id_raw)
        return SessionSource(
            platform=Platform.TELEGRAM, chat_id=chat_id or "", chat_type=chat_type, user_id=user_id,
            user_name=user_name, thread_id=thread_id, is_bot=is_bot)

    def _source_from_reaction_for_auth(self, update):
        """SessionSource for a ``message_reaction`` update's actor (``user`` or ``actor_chat``).

        Raises ``ValueError`` when actor, chat or message identity is absent so the post-auth boundary fails closed."""
        mr = getattr(update, "message_reaction", None)
        if mr is None:
            raise ValueError("gateway_platform_event source extraction requires a message_reaction update")
        user = getattr(mr, "user", None) or getattr(mr, "actor_chat", None)
        chat = getattr(mr, "chat", None)
        user_id = str(getattr(user, "id", "")).strip() or None
        user_name = str(getattr(user, "username", "") or getattr(user, "full_name", "") or getattr(user, "title", "")).strip() or None
        chat_id = str(getattr(chat, "id", "")).strip() or None
        message_id = getattr(mr, "message_id", None)
        if not user_id or not chat_id or message_id is None or not str(message_id).strip():
            raise ValueError("gateway_platform_event reaction requires actor, chat, and message identities")
        # Reactions carry no message_thread_id; is_forum is the only forum signal.
        chat_type = self._normalize_chat_type(getattr(chat, "type", "dm"), is_forum=getattr(chat, "is_forum", False) is True)
        return self.build_source(
            chat_id=chat_id, chat_type=chat_type, user_id=user_id, user_name=user_name, thread_id=None, message_id=str(message_id))

    def _telegram_auth_env_configured(self) -> bool:
        """Return True when Telegram auth env vars make an early decision safe."""
        keys = (
            "TELEGRAM_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_CHATS",
            "TELEGRAM_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS")
        return any(_scoped_gate_env(key).strip() for key in keys)

    def _should_pass_unauthorized_dm_for_pairing(self, source) -> bool:
        """True when an unauthorized DM must still reach gateway pairing (``unauthorized_dm_behavior``
        resolves to ``pair``, incl. an allowlist plus an explicit platform override)."""
        if source.chat_type != "dm":
            return False
        # Bound-handler ``__self__`` is None under multiplex; ``gateway_runner`` survives that wrapping.
        runner = getattr(getattr(self, "_message_handler", None), "__self__", None) or getattr(self, "gateway_runner", None)
        behavior_fn = getattr(runner, "_get_unauthorized_dm_behavior", None)
        if callable(behavior_fn):
            try:
                profile = getattr(source, "profile", None) or getattr(self, "_owner_profile", None)
                return behavior_fn(Platform.TELEGRAM, profile=profile) == "pair"
            except Exception:
                logger.debug("[Telegram] Failed to resolve unauthorized DM behavior; falling back to adapter-local override", exc_info=True)
        extra = getattr(getattr(self, "config", None), "extra", None) or {}
        return str(extra.get("unauthorized_dm_behavior", "")).strip().lower() == "pair"

    def _is_user_authorized_from_message(self, message: Message) -> bool:
        """Intake auth prefilter, run BEFORE batching/event construction/group observation.

        Only rejects when it can make the same context-aware decision the runner would; unknown DMs pass through when
        there is no allowlist or pairing is the unauthorized-DM behavior."""
        source = self._source_from_message_for_auth(message)
        user_id = source.user_id
        # No identity → service message or channel post without sender_chat; defer to message gating.
        if not user_id:
            return True
        authorized: Optional[bool] = None
        # Adapter-level allow_from (DMs) / group_allow_from (groups) are the sole authority if set.
        adapter_allow_from = self.config.extra.get(
            "group_allow_from" if (source.chat_type or "") in ("group", "forum", "channel") else "allow_from")
        if adapter_allow_from is not None:
            allowed = _coerce_allow_set(adapter_allow_from)
            authorized = user_id in allowed or "*" in allowed
        # Instance-level override only (tests): the class method _is_callback_user_authorized is for
        # inline buttons and must not become a user-id-only shortcut for real messages.
        if authorized is None:
            callback_auth = self.__dict__.get("_is_callback_user_authorized")
            if callable(callback_auth):
                with contextlib.suppress(Exception):
                    authorized = bool(callback_auth(
                        user_id, chat_id=source.chat_id, chat_type=source.chat_type, thread_id=source.thread_id,
                        user_name=source.user_name))
        if authorized is None:
            # Runner's full auth chain; prefer the set_authorization_check callback (survives multiplex
            # handler wrapping, unlike bound-handler __self__).
            auth_fn = self._legacy_runner_auth_fn()
            has_callback = getattr(self, "_authorization_check", None) is not None
            if has_callback or auth_fn is not None:
                # No allowlist → unknown DMs must reach pairing, not be default-denied here.
                if not self._telegram_auth_env_configured():
                    return True
                decision = self._is_sender_authorized(
                    user_id, chat_type=source.chat_type, chat_id=source.chat_id, is_bot=source.is_bot,
                    thread_id=source.thread_id) if has_callback else None
                if decision is not None:
                    authorized = decision
                elif auth_fn is not None:
                    try:
                        authorized = bool(auth_fn(source))
                    except Exception:
                        logger.debug("[Telegram] Falling back to env-only auth for user %s", user_id, exc_info=True)
        if authorized is None:
            authorized = self._env_allowlist_decision(user_id)
            if authorized is None:
                return True
        if authorized:
            return True
        # Unauthorized DM the gateway would pair: forward so pairing can run.
        return self._should_pass_unauthorized_dm_for_pairing(source)

    @classmethod
    def _metadata_thread_id(cls, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        thread_id = (metadata or {}).get("thread_id") or (metadata or {}).get("message_thread_id")
        return str(thread_id) if thread_id is not None else None

    @classmethod
    def _metadata_direct_messages_topic_id(cls, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        topic_id = (metadata or {}).get("direct_messages_topic_id") or (metadata or {}).get("telegram_direct_messages_topic_id")
        return str(topic_id) if topic_id is not None else None

    @classmethod
    def _metadata_reply_to_message_id(cls, metadata: Optional[Dict[str, Any]]) -> Optional[int]:
        reply_to = (metadata or {}).get("telegram_reply_to_message_id")
        return int(reply_to) if reply_to is not None else None

    @staticmethod
    def _dm_topic_fallback(metadata: Optional[Dict[str, Any]]) -> bool:
        """True for Hermes private-chat topic lanes (``telegram_dm_topic_reply_fallback``)."""
        return bool(metadata and metadata.get("telegram_dm_topic_reply_fallback"))

    @classmethod
    def _is_private_dm_topic_send(cls, chat_id: str, thread_id: Optional[str], metadata: Optional[Dict[str, Any]]) -> bool:
        if cls._metadata_direct_messages_topic_id(metadata) is not None:
            return cls._dm_topic_fallback(metadata) and cls._metadata_reply_to_message_id(metadata) is not None
        if metadata and metadata.get("telegram_dm_topic_created_for_send"):
            return False
        return bool(thread_id) and cls._dm_topic_fallback(metadata)

    @staticmethod
    def _dm_topic_missing_anchor_error() -> str:
        return "Telegram DM topic delivery requires a reply anchor; refusing to send outside the requested topic"

    @classmethod
    def _reply_to_message_id_for_send(
        cls, reply_to: Optional[str], metadata: Optional[Dict[str, Any]] = None, reply_to_mode: Optional[str] = None) -> Optional[int]:
        if reply_to:
            return int(reply_to)
        if cls._dm_topic_fallback(metadata) and reply_to_mode != "off":
            return cls._metadata_reply_to_message_id(metadata)
        return None

    @classmethod
    def _thread_kwargs_for_send(
        cls, chat_id: str, thread_id: Optional[str], metadata: Optional[Dict[str, Any]] = None,
        reply_to_message_id: Optional[int] = None, reply_to_mode: Optional[str] = None) -> Dict[str, Any]:
        """Telegram send kwargs for forum and direct-message topic routing.

        Forum topics use ``message_thread_id``; native Bot API DM topics opt in via explicit ``direct_messages_topic_id``
        metadata; Hermes private-chat topic lanes are marked ``telegram_dm_topic_reply_fallback``. Anchor-less synthetic sends
        prefer the Hermes topic's ``message_thread_id`` (the native DM-topic id renders in a different chat lane).
        ``reply_to_mode="off"`` suppresses the anchor but keeps ``message_thread_id``.

        Live replies send the private topic thread id together with a reply anchor. Synthetic/resumed sends
        without an anchor (loop wakeups, background-process notifications, queued follow-ups after a gateway
        restart) prefer the Hermes topic's ``message_thread_id`` so they stay in the active topic lane
        (#87051); ``direct_messages_topic_id`` is only used when no topic thread resolves, since the native
        DM-topic id does not match the Hermes topic lane and can render the message in a different chat
        lane.
        """
        fallback = cls._dm_topic_fallback(metadata)
        if fallback and reply_to_mode != "off":
            if reply_to_message_id is None:
                reply_to_message_id = cls._metadata_reply_to_message_id(metadata)
            if reply_to_message_id is None:
                # Anchor-less synthetic send: prefer the Hermes topic thread id (see docstring).
                # Anchor-less synthetic sends (loop wakeups, watch notifications, restart-resumed
                # follow-ups) must stay in the active topic lane: prefer the Hermes topic thread id when it
                # resolves (#87051). Routing via direct_messages_topic_id here sent these to a different
                # lane than the topic the session runs in.
                thread_message_id = cls._message_thread_id_for_send(thread_id)
                if thread_message_id is not None:
                    return {"message_thread_id": thread_message_id}
                return cls._direct_topic_kwargs(metadata) or {}
        elif not fallback:
            direct_kwargs = cls._direct_topic_kwargs(metadata)
            if direct_kwargs is not None:
                return direct_kwargs
        return {"message_thread_id": cls._message_thread_id_for_send(thread_id)}

    @classmethod
    def _direct_topic_kwargs(cls, metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Native Bot API DM-topic routing kwargs, or None when no ``direct_messages_topic_id``."""
        direct_topic_id = cls._metadata_direct_messages_topic_id(metadata)
        if direct_topic_id is None:
            return None
        return {"message_thread_id": None, "direct_messages_topic_id": int(direct_topic_id)}

    def _thread_kwargs_for_draft(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Routing kwargs for ``sendMessageDraft`` / ``sendRichMessageDraft`` (integer
        ``message_thread_id`` for DM topics — Telegram rejects the raw string ``thread_id``)."""
        kwargs = self._thread_kwargs_for_send(
            chat_id, self._metadata_thread_id(metadata), metadata, reply_to_message_id=self._reply_to_message_id_for_send(None, metadata),
            reply_to_mode=getattr(self, "_reply_to_mode", None))
        return {k: v for k, v in kwargs.items() if v is not None}

    @classmethod
    def _message_thread_id_for_send(cls, thread_id: Optional[str]) -> Optional[int]:
        if not thread_id or str(thread_id) == cls._GENERAL_TOPIC_THREAD_ID:
            return None
        return int(thread_id)

    @classmethod
    def _message_thread_id_for_typing(cls, thread_id: Optional[str]) -> Optional[int]:
        # Deliberately asymmetric with _message_thread_id_for_send: sendMessage rejects message_thread_id=1
        # (forum General), but sendChatAction NEEDS it to place the typing bubble in General.
        return int(thread_id) if thread_id else None

    @staticmethod
    def _is_thread_not_found_error(error: Exception) -> bool:
        return "thread not found" in str(error).lower()

    def _prune_stale_dm_topic_binding(self, chat_id: Any, thread_id: Any, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Drop the stale ``telegram_dm_topic_bindings`` row for a topic Telegram confirmed deleted, else
        ``_recover_telegram_topic_thread_id`` keeps steering inbound to the dead thread. Best-effort.
        Rows are namespaced by profile: the send's ``hermes_profile`` wins over the adapter's stamp.

        Without this prune the recovery logic in ``gateway.run._recover_telegram_topic_thread_id`` keeps
        steering future inbound messages to the dead thread (the bug behind #31501 — tool progress,
        approvals, replies all end up in the wrong place even though the user has moved on to a fresh
        topic). Best-effort: we never raise from a send-fallback path — a failed cleanup must not turn into
        a failed user-facing send.
        Under ``gateway.profile_routes`` the transport adapter may not be the profile that wrote the
        binding, so the send's ``hermes_profile`` metadata wins over the adapter's own profile stamp;
        single-profile bots fall back to ``"default"``. See #76423.
        """
        if chat_id is None or thread_id is None:
            return
        db = getattr(getattr(self, "_session_store", None), "_db", None)
        if db is None or not hasattr(db, "delete_telegram_topic_binding"):
            return
        try:
            profile_name = (metadata or {}).get("hermes_profile") or getattr(self, "_hermes_profile_name", None) or "default"
            removed = db.delete_telegram_topic_binding(chat_id=str(chat_id), thread_id=str(thread_id), profile_name=profile_name)
        except Exception:
            logger.debug(
                "[%s] delete_telegram_topic_binding failed for chat=%s thread=%s — skipping prune",
                self.name, chat_id, thread_id, exc_info=True)
            return
        if removed:
            logger.info(
                "[%s] Pruned stale Telegram DM topic binding chat=%s thread=%s (Bot API: thread not found)", self.name, chat_id, thread_id)

    @staticmethod
    def _is_bad_request_error(error: Exception) -> bool:
        name = error.__class__.__name__.lower()
        if name == "badrequest" or name.endswith("badrequest"):
            return True
        try:
            from telegram.error import BadRequest
            return isinstance(error, BadRequest)
        except ImportError:
            return False

    @classmethod
    def _should_retry_without_dm_topic_reply_anchor(
        cls, error: Exception, metadata: Optional[Dict[str, Any]], reply_to_message_id: Optional[int]) -> bool:
        """True when a DM-topic send should be retried with routing stripped: (1) stale anchor — reply
        target deleted; (2) anchor-less synthetic send whose ``direct_messages_topic_id`` Bot API rejects.

        2. The synthetic-event case (added when #27937 introduced ``direct_messages_topic_id`` fallback for
        sends without an anchor): if Bot API rejects the topic id itself with any BadRequest that mentions
        topic/thread routing, we retry without routing rather than dropping the message.
        """
        if not cls._dm_topic_fallback(metadata) or not cls._is_bad_request_error(error):
            return False
        err_lower = str(error).lower()
        if reply_to_message_id is not None and "message to be replied not found" in err_lower:
            return True
        if not metadata.get("direct_messages_topic_id"):  # topic id rejected → plain DM send
            return False
        topic_markers = (
            "direct_messages_topic", "message thread not found", "thread not found", "topic_closed", "topic_deleted", "topic not found")
        return any(marker in err_lower for marker in topic_markers)

    async def _send_with_dm_topic_reply_anchor_retry(
        self, send_fn: Any, send_kwargs: Dict[str, Any], metadata: Optional[Dict[str, Any]],
        reply_to_message_id: Optional[int], media_label: str, reset_media: Optional[Any] = None) -> Any:
        """Retry stale private-topic media replies once without the topic anchor."""
        try:
            return await send_fn(**send_kwargs)
        except Exception as send_err:
            if not self._should_retry_without_dm_topic_reply_anchor(send_err, metadata, reply_to_message_id):
                raise
            logger.warning(
                "[%s] Reply target deleted for Telegram %s, retrying without reply/topic anchor: %s",
                self.name, media_label, _redact_telegram_error_text(send_err))
            if reset_media is not None:
                reset_media()
            retry_kwargs = dict(send_kwargs)
            retry_kwargs["reply_to_message_id"] = None
            retry_kwargs.pop("message_thread_id", None)
            retry_kwargs.pop("direct_messages_topic_id", None)
            return await send_fn(**retry_kwargs)

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

    def _link_preview_kwargs(self) -> Dict[str, Any]:
        if not getattr(self, "_disable_link_previews", False):
            return {}
        if LinkPreviewOptions is not None:
            return {"link_preview_options": LinkPreviewOptions(is_disabled=True)}
        return {"disable_web_page_preview": True}

    # --- Bot API 10.1 Rich Messages (sendRichMessage): final/new-message replies opportunistically send
    # RAW agent markdown so tables, task lists, <details>, math render natively; legacy MarkdownV2 send()
    # is the fallback. Streaming edits stay on the MarkdownV2 edit path.
    def _content_fits_rich_limits(self, content: str) -> bool:
        """Pre-check the 32,768-char cap only; other rich limits surface as BadRequest (permanent)."""
        return len(content) <= self.RICH_MESSAGE_MAX_CHARS

    def _bot_supports_rich(self) -> bool:
        """True when ``do_api_request`` is an *async* callable (real Bot or AsyncMock); plain MagicMock
        and SimpleNamespace bots resolve False → legacy path."""
        return inspect.iscoroutinefunction(getattr(self._bot, "do_api_request", None))

    _RICH_DETAILS_RE = re.compile(r"<details\b[^>]*>.*?</details>", re.IGNORECASE | re.DOTALL)
    _RICH_MATH_IN_DETAILS_RE = re.compile(
        r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|"
        r"\\(?:sum|frac|alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|"
        r"int|prod|sqrt|lim|infty|begin\{(?:equation|align|matrix|cases)\}))",
        re.IGNORECASE | re.DOTALL)
    # Hiragana/Katakana, CJK Ext A, CJK Unified, Hangul, CJK Compatibility, CJK ext/compat supplement.
    _RICH_CJK_RE = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\U00020000-\U000323af]")

    def _has_telegram_desktop_details_math_crash_shape(self, content: str) -> bool:
        """Math inside <details> crashes Telegram Desktop 6.9.1 (tdesktop#30808); the Bot API accepts
        the payload, so rich delivery must be skipped up front."""
        if not content:
            return False
        return any(self._RICH_MATH_IN_DETAILS_RE.search(block) for block in self._RICH_DETAILS_RE.findall(content))

    def _has_telegram_desktop_cjk_rich_garble_shape(self, content: str) -> bool:
        """True for CJK content: Telegram Mac/Desktop rich rendering leaves overlapping glyphs.

        Telegram Mac/Desktop Bot API 10.1 rich-message rendering currently leaves overlapping draft/overlay
        glyph artifacts for CJK text (#47653). The legacy MarkdownV2 path renders the same text cleanly, so
        skip rich delivery up front until affected clients age out.
        """
        return bool(content and self._RICH_CJK_RE.search(content))

    def _needs_rich_rendering(self, content: str) -> bool:
        """True for constructs MarkdownV2 degrades: pipe tables, task lists, <details>, block math.
        Ordinary replies stay on MarkdownV2 so clients render consistent font weight/spacing.

        The rich endpoint is reserved for constructs where raw markdown materially improves output: pipe
        tables (MarkdownV2 has no table syntax and rewrites them into bullet lists), GFM task lists,
        collapsible ``<details>`` blocks, and block math. Adapted from #45995 (@YonganZhang).
        """
        if not content:
            return False
        if any(_TABLE_SEPARATOR_RE.match(line) for line in content.splitlines()):
            return True
        if re.search(r"(?m)^\s*[-*]\s+\[[ xX]\]\s+", content):
            return True
        if re.search(r"(?m)^<details\b|^</details>|^<summary\b|^</summary>", content):
            return True
        return "$$" in content

    def _rich_delivery_enabled(self) -> bool:
        """Whether rich delivery is allowed (``rich_messages`` opt-in)."""
        return bool(getattr(self, "_rich_messages_enabled", True))

    def _rich_content_ok(self, content: str) -> bool:
        """Shape checks shared by rich sends and rich drafts (non-blank, no Desktop crash/garble
        shapes, under the cap, async-capable bot)."""
        return bool(
            content and content.strip()
            and not self._has_telegram_desktop_details_math_crash_shape(content)
            and not self._has_telegram_desktop_cjk_rich_garble_shape(content)
            and self._content_fits_rich_limits(content)
            and self._bot_supports_rich())

    def _rich_eligible(self, content: str) -> bool:
        """Rich eligibility ignoring ``expect_edits`` (a streamed preview's FINAL edit still upgrades)."""
        return bool(
            self._rich_delivery_enabled()
            and not getattr(self, "_rich_send_disabled", False)
            and content and content.strip()
            and self._needs_rich_rendering(content)
            and self._rich_content_ok(content))

    def _should_attempt_rich(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        return bool(not (metadata or {}).get("expect_edits") and self._rich_eligible(content))

    def prefers_fresh_final_streaming(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Replace a streamed preview with a fresh rich final — DM topics only. Root DMs stay off (a live
        draft has no preview id); DM *topics* degrade to edit-in-place whose MarkdownV2 preview Telegram
        refuses to rich-edit, so a fresh sendRichMessage + delete is the only way to keep native tables.

        Root DMs keep this off (#46206 / #47048): successful draft streaming has no preview ``message_id``,
        so the hook is not consulted, and in-place ``editMessageText.rich_message`` would duplicate a live
        draft turn. Private DM *topics* often reject ``sendMessageDraft``; the consumer then degrades to
        edit-in-place. Telegram rejects a rich edit of that plain MarkdownV2 preview, and the fallback
        formatter permanently turns pipe tables into bullet lists.
        """
        metadata = metadata or {}
        if not (metadata.get("telegram_dm_topic_reply_fallback") or self._metadata_direct_messages_topic_id(metadata)):
            return False
        return self._rich_eligible(content)

    def _rich_transport_available(self) -> bool:
        return bool(
            getattr(self, "_rich_messages_enabled", True) and not getattr(self, "_rich_send_disabled", False) and self._bot_supports_rich())

    def streaming_overflow_limit(self) -> Optional[int]:
        """Let the stream consumer accumulate up to the rich cap so a reply that fits one sendRichMessage
        isn't fragmented at 4,096. None (→ legacy limit) if rich is unavailable."""
        return self.RICH_MESSAGE_MAX_CHARS if self._rich_transport_available() else None

    def _rich_message_payload(self, content: str, *, skip_entity_detection: bool = False) -> Dict[str, Any]:
        """``InputRichMessage`` from RAW markdown — never ``format_message(content)``, whose MarkdownV2
        escaping destroys table pipes."""
        payload: Dict[str, Any] = {"markdown": _rich_normalize_linebreaks(content)}
        if skip_entity_detection:
            payload["skip_entity_detection"] = True
        return payload

    def _is_rich_capability_error(self, exc: Exception) -> bool:
        """True ⇒ the rich endpoint itself is unavailable (old PTB/server); latches rich off.
        Per-message BadRequests (parser/limit) are NOT capability errors."""
        if exc.__class__.__name__.lower() in {"endpointnotfound", "invalidtoken"}:
            return True
        if isinstance(exc, (AttributeError, TypeError, NotImplementedError)) or getattr(exc, "error_code", None) == 404:
            return True
        s = str(exc).lower()
        if ("method" in s or "endpoint" in s) and ("not found" in s or "does not exist" in s):
            return True
        return "no such method" in s

    def _is_rich_fallback_error(self, exc: Exception) -> bool:
        """True ⇒ permanent/capability error ⇒ safe to fall back to legacy. Conservative: anything not
        clearly permanent is transient — the rich request may have reached Telegram (duplicate risk)."""
        if self._is_bad_request_error(exc) or self._is_rich_capability_error(exc):
            return True
        s = str(exc).lower()
        return "unsupported" in s or "not implemented" in s

    def _chunk_reply_routing(
        self, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]], thread_id: Optional[str], index: int) -> tuple:
        """Reply-anchor routing for chunk ``index``: ``(private_dm_topic_send, anchor_off, reply_to_id)``.
        ``anchor_off``: reply_to_mode="off" on the DM-topic fallback path opts into "message_thread_id
        alone is enough" — don't fail loud because the anchor was suppressed by config."""
        metadata_reply_to = self._metadata_reply_to_message_id(metadata)
        private_dm_topic_send = self._is_private_dm_topic_send(chat_id, thread_id, metadata)
        dm_topic_reply_to_off = private_dm_topic_send and self._reply_to_mode == "off" and self._dm_topic_fallback(metadata)
        reply_to_source = reply_to or (str(metadata_reply_to) if private_dm_topic_send and metadata_reply_to is not None else None)
        if private_dm_topic_send:
            should_thread = reply_to_source is not None and self._reply_to_mode != "off"
        else:
            should_thread = self._should_thread_reply(reply_to_source, index)
        reply_to_id = int(reply_to_source) if should_thread and reply_to_source else None
        return private_dm_topic_send, dm_topic_reply_to_off, reply_to_id

    def _compute_single_send_routing(
        self, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]], thread_id: Optional[str]) -> Optional[tuple]:
        """Routing for a single (rich) send — mirrors send()'s index-0 block. Returns ``(reply_to_id,
        thread_kwargs)`` or ``None`` = skip rich, legacy owns the DM-topic fail-loud SendResult."""
        private_dm_topic_send, dm_topic_reply_to_off, reply_to_id = self._chunk_reply_routing(chat_id, reply_to, metadata, thread_id, 0)
        thread_kwargs = self._thread_kwargs_for_send(
            chat_id, thread_id, metadata, reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode)
        # Synthetic/resumed sends via direct_messages_topic_id need no reply anchor.
        if (
            private_dm_topic_send and reply_to_id is None and not dm_topic_reply_to_off
            and not thread_kwargs.get("direct_messages_topic_id")):
            return None
        return reply_to_id, thread_kwargs

    @staticmethod
    def _is_timed_out(exc: Exception) -> bool:
        """PTB ``TimedOut`` (when importable) or a "timed out" message."""
        try:
            from telegram.error import TimedOut as _TimedOut
        except (ImportError, AttributeError):
            _TimedOut = None
        return bool((_TimedOut and isinstance(exc, _TimedOut)) or "timed out" in str(exc).lower())

    def _rich_transient_result(self, exc: Exception, what: str, *, retry_after: Any = None) -> SendResult:
        """SendResult for a transient/unknown rich-API failure (request may have reached Telegram, so the
        caller must NOT legacy-resend); retry semantics mirror legacy send()."""
        safe_error = _redact_telegram_error_text(exc)
        logger.warning("[%s] %s transient failure (no legacy resend): %s", self.name, what, safe_error)
        return SendResult(
            success=False, error=safe_error,
            retryable=(self._looks_like_connect_timeout(exc) or not self._is_timed_out(exc)), retry_after=retry_after)

    @staticmethod
    def _record_rich_sent(chat_id: Any, message_id: Any, content: str) -> None:
        """Index rich content we sent: Telegram won't echo it back in reply_to_message."""
        try:
            from gateway import rich_sent_store
            rich_sent_store.record(str(chat_id), str(message_id), content)
        except Exception:
            pass

    async def _try_send_rich(
        self, chat_id: str, content: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]]) -> Optional[SendResult]:
        """Attempt a single ``sendRichMessage``. Returns a SendResult (success, or a transient failure the
        caller must NOT legacy-resend), or ``None`` = fall back to legacy MarkdownV2."""
        thread_id = self._metadata_thread_id(metadata)
        routing = self._compute_single_send_routing(chat_id, reply_to, metadata, thread_id)
        if routing is None:
            return None
        reply_to_id, thread_kwargs = routing
        payload = self._rich_payload_base(chat_id, content)
        # Only non-None routing keys: direct_messages_topic_id is paired with message_thread_id=None.
        payload.update({k: v for k, v in thread_kwargs.items() if v is not None})
        payload.update(self._notification_kwargs(metadata))
        if reply_to_id is not None:
            # sendRichMessage takes reply_parameters, NOT reply_to_message_id (silently ignored → anchor dropped).
            payload["reply_parameters"] = {"message_id": reply_to_id}
        try:
            # Raw Bot API result: return_type=Message would make PTB deserialize a 10.1 shape it doesn't
            # fully model; a post-delivery parse error ≠ send failure.
            msg = await self._bot.do_api_request("sendRichMessage", api_kwargs=payload)
        except Exception as exc:
            if self._rich_rejected(exc, "sendRichMessage", "MarkdownV2"):
                return None
            # Honor Telegram's flood-control retry_after over the base retry schedule.
            _retry_after = getattr(exc, "retry_after", None)
            if _retry_after is None:
                _m = re.search(r"retry\s+(?:in\s+)?(\d+)", str(exc).lower(), re.IGNORECASE)
                if _m:
                    _retry_after = float(_m.group(1))
            return self._rich_transient_result(exc, "sendRichMessage", retry_after=_retry_after)
        if isinstance(msg, dict):
            message_id = msg.get("message_id")
            if message_id is None:
                message_id = (msg.get("result") or {}).get("message_id")
        else:
            message_id = getattr(msg, "message_id", None)
        if message_id is not None:
            self._record_rich_sent(chat_id, message_id, content)
        return SendResult(success=True, message_id=str(message_id) if message_id is not None else None)

    def _rich_payload_base(self, chat_id: str, content: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"chat_id": normalize_telegram_chat_id(chat_id), "rich_message": self._rich_message_payload(content)}
        if getattr(self, "_disable_link_previews", False):
            payload["link_preview_options"] = {"is_disabled": True}
        return payload

    def _rich_rejected(self, exc: Exception, what: str, fallback: str) -> bool:
        """True for a permanent/capability rich-API failure (caller falls back to legacy); capability
        errors latch rich off so no doomed roundtrip repeats per send."""
        if not self._is_rich_fallback_error(exc):
            return False
        if self._is_rich_capability_error(exc):
            self._rich_send_disabled = True
        logger.debug("[%s] %s rejected (%s) — falling back to %s", self.name, what, _redact_telegram_error_text(exc), fallback)
        return True

    async def _try_edit_rich(
        self, chat_id: str, message_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[SendResult]:
        """Edit a message in place as rich (``editMessageText`` + ``rich_message``) so a streamed preview
        finalizes without send+delete. Same contract as :meth:`_try_send_rich`."""
        # No topic routing on edits: message_thread_id/direct_messages_topic_id make Telegram reject it.
        payload = {**self._rich_payload_base(chat_id, content), "message_id": int(message_id)}
        try:
            await self._bot.do_api_request("editMessageText", api_kwargs=payload)
        except Exception as exc:
            # "Message is not modified" = successful no-op; skip the redundant legacy edit.
            if "not modified" in str(exc).lower():
                if self._is_rich_fallback_error(exc) and self._is_rich_capability_error(exc):
                    self._rich_send_disabled = True
                return SendResult(success=True, message_id=message_id)
            if self._rich_rejected(exc, "rich editMessageText", "MarkdownV2 edit"):
                return None
            return self._rich_transient_result(exc, "rich editMessageText")
        # Mirror the fresh-send index: a streamed final finalized via edit is otherwise never recorded.
        self._record_rich_sent(chat_id, message_id, content)
        return SendResult(success=True, message_id=message_id)

    def _should_attempt_rich_draft(self, content: str) -> bool:
        return bool(
            getattr(self, "_rich_messages_enabled", True)
            and getattr(self, "_rich_drafts_enabled", False)
            and not getattr(self, "_rich_send_disabled", False)
            and not getattr(self, "_rich_draft_disabled", False)
            and self._rich_content_ok(content))

    async def _try_send_rich_draft(self, chat_id: str, draft_id: int, content: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Emit one ``sendRichMessageDraft`` frame; True on success. Frames are ephemeral, so any failure
        returns False and the caller renders the legacy draft; capability failures latch off."""
        payload: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id), "draft_id": int(draft_id), "rich_message": self._rich_message_payload(content)}
        payload.update(self._thread_kwargs_for_draft(chat_id, metadata))
        try:
            return bool(await self._bot.do_api_request("sendRichMessageDraft", api_kwargs=payload))
        except Exception as exc:
            if self._is_rich_capability_error(exc):
                self._rich_draft_disabled = True
                logger.debug(
                    "[%s] sendRichMessageDraft unsupported (%s) — using legacy drafts", self.name, _redact_telegram_error_text(exc))
            else:
                logger.debug(
                    "[%s] sendRichMessageDraft transient failure (%s) — legacy draft this frame", self.name,
                    _redact_telegram_error_text(exc))
            return False

    async def _drain_polling_connections(self) -> None:
        """Reset the httpx pool used for getUpdates polling before a reconnect.

        Half-closed connections (esp. via proxies) occupy pool slots until "Pool timeout: All connections in the connection pool
        are occupied". Only ``_request[0]`` (getUpdates) is reset; the general request stays untouched so concurrent sends are
        never interrupted. Relies on PTB 22.x's private ``(get_updates, general)`` tuple — review on PTB 23+."""
        if not (self._app and self._app.bot):
            return
        try:
            polling_req = self._app.bot._request[0]  # noqa: SLF001
        except Exception:
            return
        # Bounded wall-clock deadline (not asyncio.wait_for): httpcore's pool close runs under
        # AsyncShieldCancellation and a wedged CLOSE-WAIT socket can hang it forever.
        if not await self._bounded_request_step(polling_req.shutdown(), "Polling request shutdown failed/timed out (non-fatal)"):
            # initialize() only rebuilds the client when ``client.is_closed``; an abandoned aclose()
            # leaves it false, so start_polling would reuse the CLOSE-WAIT socket (alive but deaf).
            # Swap in a fresh client before initialize(). See #87057.
            self._orphan_and_rebuild_polling_client(polling_req)
        if await self._bounded_request_step(polling_req.initialize(), "Polling request re-initialize failed/timed out (non-fatal)"):
            logger.debug("[%s] Polling request pool drained before reconnect", self.name)
        else:
            self._orphan_and_rebuild_polling_client(polling_req)

    async def _bounded_request_step(self, awaitable, failure_msg: str) -> bool:
        """Await a request shutdown()/initialize() under ``_DRAIN_TIMEOUT``; False (debug-logged) on failure."""
        try:
            await _await_with_thread_deadline(awaitable, timeout=_DRAIN_TIMEOUT)
            return True
        except Exception:
            logger.debug("[%s] " + failure_msg, self.name, exc_info=True)
            return False

    def _orphan_and_rebuild_polling_client(self, polling_req) -> None:
        """Replace a wedged HTTPXRequest client after a hung aclose(): swap in a fresh client and close
        the old one in a detached, bounded task so it can't block the reconnect ladder.

        PTB's ``HTTPXRequest.initialize()`` only calls ``_build_client()`` when the current client reports
        ``is_closed``. If ``shutdown()`` was abandoned on a CLOSE-WAIT socket, that flag stays false and the
        next ``start_polling()`` reuses the dead getUpdates connection (#87057).
        """
        old = getattr(polling_req, "_client", None)
        build = getattr(polling_req, "_build_client", None)
        if old is None or not callable(build) or getattr(old, "is_closed", True):
            return
        try:
            polling_req._client = build()  # noqa: SLF001
        except Exception:
            logger.debug("[%s] Failed to rebuild polling HTTP client after hung drain", self.name, exc_info=True)
            return
        logger.warning("[%s] Replaced wedged getUpdates HTTP client after drain timeout (likely CLOSE-WAIT socket)", self.name)

        async def _orphan_aclose() -> None:
            try:
                aclose = getattr(old, "aclose", None)
                if not callable(aclose):
                    return
                # Same cancellation-swallowing httpcore scope as shutdown(): wall-clock deadline.
                await _await_with_thread_deadline(aclose(), timeout=_DRAIN_TIMEOUT)
            except Exception:
                logger.debug("[%s] Orphan polling client aclose failed (non-fatal)", self.name, exc_info=True)

        try:
            task = asyncio.ensure_future(_orphan_aclose())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            task.add_done_callback(_consume_abandoned_task)
        except Exception:
            pass

    def _fence_polling(self) -> None:
        """Mark polling closed: no progress accepted, send path degraded."""
        self._polling_progress_accepting = False
        self._send_path_degraded = True

    def _begin_polling_generation(self) -> tuple[int, asyncio.Event]:
        """Start accepting progress for a new getUpdates polling generation."""
        if self._teardown_started:
            self._fence_polling()
            progress = getattr(self, "_polling_progress_event", None)
            if progress is None:
                progress = self._polling_progress_event = asyncio.Event()
            return getattr(self, "_polling_generation", 0), progress
        verifier = getattr(self, "_polling_progress_verifier_task", None)
        if verifier is not None and not verifier.done():
            verifier.cancel()
        self._polling_progress_verifier_task = None
        self._polling_generation = getattr(self, "_polling_generation", 0) + 1
        self._polling_progress_event = asyncio.Event()
        self._polling_progress_accepting = True
        self._send_path_degraded = True
        # Reset stall-watchdog timestamps: no proven progress yet, age measured from here.
        # See #92991.
        self._polling_generation_started_monotonic = time.monotonic()
        self._polling_last_progress_monotonic = None
        return self._polling_generation, self._polling_progress_event

    def _record_polling_progress(self, generation: int) -> None:
        """Record successful getUpdates I/O for the current generation only."""
        if self._teardown_started or not self._polling_progress_accepting or generation != self._polling_generation:
            return
        if not self._polling_progress_event.is_set():
            # First confirmed round-trip resolves the "health pending" line both reconnect paths end on.
            logger.info("[%s] Telegram polling confirmed healthy: getUpdates progressing (generation %d)", self.name, generation)
        self._polling_progress_event.set()
        self._polling_last_progress_monotonic = time.monotonic()
        self._polling_network_error_count = 0
        if generation == self._polling_conflict_recovery_generation:
            self._polling_conflict_recovery_generation = None
        else:
            self._polling_conflict_count = 0
        # First proof getUpdates is flowing for this generation: flip a
        # published "retrying" (degraded connect, reconnect stamp, or the
        # mid-session recovery below) back to "connected" (#101391).
        if self._send_path_degraded and getattr(self, "_running", False) and not self.has_fatal_error:
            self._write_runtime_status_safe(
                "connected", platform_state="connected", error_code=None, error_message=None,
            )
        self._send_path_degraded = False

    def _observe_polling_request_result(self, request, generation, result):
        """Record getUpdates progress from an observed do_request result (purely observational: PTB still
        parses the untouched payload and owns any resulting exception)."""
        status_code, payload = result
        if generation is None or not (200 <= status_code < 300):
            return
        try:
            # The request's own parser keeps health observation in agreement with PTB.
            envelope = request.parse_json_payload(payload)
        except Exception:
            return
        if isinstance(envelope, dict) and envelope.get("ok") is True and "result" in envelope:
            self._record_polling_progress(generation)

    def _instrument_polling_request(self, request):
        """Instrument one dedicated PTB getUpdates request with progress tracking.

        PTB request classes use ``__slots__`` (no ``__dict__`` on 3.13), so re-tag the instance to a thin ``__slots__ = ()``
        subclass overriding ``do_request`` — identical layout makes the ``__class__`` swap legal; works for test doubles too.

        On Python 3.13 their instances no longer carry a ``__dict__`` (the ``AbstractAsyncContextManager``
        MRO stopped yielding one), so ``request.do_request = wrapper`` raises ``AttributeError:
        'HTTPXRequest' object attribute 'do_request' is read-only`` and the whole Telegram connect fails
        (#64482). It only appeared to work on Python 3.12, where those instances still had a ``__dict__``.
        """
        adapter = self

        class _InstrumentedPollingRequest(type(request)):
            __slots__ = ()

            async def do_request(self, *args, **kwargs):
                generation = _POLLING_GENERATION_CONTEXT.get()
                result = await super().do_request(*args, **kwargs)
                adapter._observe_polling_request_result(self, generation, result)
                return result

        request.__class__ = _InstrumentedPollingRequest
        return request

    async def _start_polling_once(
        self, app, *, drop_pending_updates: bool, error_callback, abandon_app_on_timeout: bool = False,
        schedule_verifier: bool = True) -> tuple[int, asyncio.Event]:
        """Start one generation and verify real getUpdates progress. Returns this generation's
        ``(generation, progress_event)`` so readiness-gating callers bind to exactly it."""
        if self._teardown_started:
            raise _PollingLifecycleAbort("Telegram polling teardown started")
        generation, progress = self._begin_polling_generation()
        if not self._polling_progress_accepting:
            raise _PollingLifecycleAbort("Telegram polling teardown started")

        def _generation_error_callback(error: Exception) -> None:
            if self._teardown_started or generation != self._polling_generation or error_callback is None:
                return
            callback_context_token = _POLLING_GENERATION_CONTEXT.set(None)
            try:
                error_callback(error)
            finally:
                _POLLING_GENERATION_CONTEXT.reset(callback_context_token)

        context_token = _POLLING_GENERATION_CONTEXT.set(generation)
        try:
            # asyncio.wait_for can wait forever on httpcore/AnyIO shielded scopes; use the wall-deadline
            # helper and abandon the partial updater (caller rebuilds).
            await _await_with_thread_deadline(
                app.updater.start_polling(
                    allowed_updates=Update.ALL_TYPES, drop_pending_updates=drop_pending_updates, error_callback=_generation_error_callback),
                timeout=_UPDATER_START_TIMEOUT,
                on_abandon=((lambda app=app: _shutdown_abandoned_app(app)) if abandon_app_on_timeout else None))
        finally:
            _POLLING_GENERATION_CONTEXT.reset(context_token)
        if self._teardown_started:
            self._fence_polling()
            raise _PollingLifecycleAbort("Telegram polling teardown started")
        if schedule_verifier:
            self._schedule_polling_progress_verifier(generation, progress)
        return generation, progress

    def _schedule_polling_progress_verifier(self, generation: int, progress: asyncio.Event) -> None:
        """Own exactly one tracked verifier for the current generation."""
        if self._teardown_started:
            self._fence_polling()
            return
        previous = getattr(self, "_polling_progress_verifier_task", None)
        if previous is not None and not previous.done():
            previous.cancel()
        task = asyncio.get_running_loop().create_task(self._verify_polling_after_reconnect(generation, progress))
        self._polling_progress_verifier_task = task
        self._background_tasks.add(task)

        def _clear_finished_verifier(finished: asyncio.Task) -> None:
            self._background_tasks.discard(finished)
            if self._polling_progress_verifier_task is finished:
                self._polling_progress_verifier_task = None

        task.add_done_callback(_clear_finished_verifier)

    def _get_general_request_drain_lock(self) -> asyncio.Lock:
        lock = getattr(self, "_general_request_drain_lock", None)
        if lock is None:
            lock = self._general_request_drain_lock = asyncio.Lock()
        return lock

    async def _drain_general_connections_after_pool_timeout(self) -> None:
        """Reset the general Bot API pool (``_request[1]``) after a confirmed send pool timeout — PTB
        guarantees the request was not sent, so resetting before retrying is safe."""
        bot = getattr(getattr(self, "_app", None), "bot", None)
        if bot is None:
            bot = getattr(self, "_bot", None)
        if bot is None:
            return
        try:
            general_req = bot._request[1]  # noqa: SLF001
        except Exception:
            return
        async with self._get_general_request_drain_lock():
            await self._bounded_request_step(
                general_req.shutdown(), "General request shutdown failed/timed out after pool timeout (non-fatal)")
            if await self._bounded_request_step(
                general_req.initialize(), "General request re-initialize failed/timed out after pool timeout (non-fatal)"):
                logger.warning("[%s] General request pool drained after Telegram pool timeout", self.name)

    def _spawn_polling_recovery(self, loop, coro) -> None:
        """Start ``coro`` as the tracked in-flight recovery task (reentrancy guard)."""
        self._polling_error_task = loop.create_task(coro)
        self._background_tasks.add(self._polling_error_task)
        self._polling_error_task.add_done_callback(self._background_tasks.discard)

    def _recovery_in_flight(self) -> bool:
        return bool(self._polling_error_task and not self._polling_error_task.done())

    def _schedule_polling_recovery(self, error: Exception, *, reason: str) -> None:
        """Schedule background polling recovery without failing gateway startup: a transient bootstrap
        failure degrades only this adapter; the reconnect ladder recovers in the background."""
        if self._teardown_started or self.has_fatal_error:
            return
        if self._recovery_in_flight():
            logger.debug(
                "[%s] Telegram polling recovery already scheduled; ignoring %s: %s", self.name, reason, _redact_telegram_error_text(error))
            return
        self._send_path_degraded = True
        # Polling died mid-session on an adapter that published "connected"
        # at connect time. Without this, gateway_state.json keeps saying
        # connected for as long as the recovery ladder runs (#101391: 11 h).
        if getattr(self, "_running", False):
            self._mark_degraded()
        logger.warning(
            "[%s] Telegram polling degraded (%s); gateway stays alive and will retry. Error: %s", self.name, reason,
            _redact_telegram_error_text(error))
        self._spawn_polling_recovery(asyncio.get_running_loop(), self._handle_polling_network_error(error))

    async def _delete_webhook_best_effort(self, *, require_success: bool = False) -> bool:
        """Clear a stale webhook; ``require_success`` (cold start) raises so GatewayRunner disposes the
        partial adapter, while reconnects recover transient errors in background."""
        if not self._bot:
            return False
        delete_webhook = getattr(self._bot, "delete_webhook", None)
        if not callable(delete_webhook):
            return True
        try:
            # Same shielded-cancellation class as initialize/start_polling: never let it pin connect.
            await _await_with_thread_deadline(delete_webhook(drop_pending_updates=False), timeout=_UPDATER_START_TIMEOUT)
            return True
        except Exception as err:
            if not self._looks_like_network_error(err):
                raise
            if require_success:
                raise OSError("Telegram deleteWebhook did not complete during initial connect") from err
            logger.warning(
                "[%s] deleteWebhook failed with a recoverable network error; continuing to polling so getUpdates/retry can recover: %s",
                self.name, _redact_telegram_error_text(err))
            self._send_path_degraded = True
            return False

    async def _await_cold_start_readiness(self, progress: asyncio.Event, strict_error_event: asyncio.Event, strict_error: list) -> None:
        """Cold start: wait for THIS generation's first getUpdates success or the first polling error;
        raises OSError so GatewayRunner disposes the partial adapter and retries fresh."""
        progress_wait = asyncio.ensure_future(progress.wait())
        error_wait = asyncio.ensure_future(strict_error_event.wait())
        try:
            # Losers are NOT cancelled here; the finally below does it.
            await _await_with_thread_deadline(
                asyncio.wait({progress_wait, error_wait}, return_when=asyncio.FIRST_COMPLETED), timeout=_INITIAL_POLLING_PROGRESS_TIMEOUT)
        except asyncio.TimeoutError as exc:
            raise OSError(
                "Telegram getUpdates made no progress within "
                f"{_INITIAL_POLLING_PROGRESS_TIMEOUT:.0f}s during initial "
                "connect — failing startup so the gateway retries with a fresh adapter (#67498)"
           ) from exc
        finally:
            for fut in (progress_wait, error_wait):
                if not fut.done():
                    fut.cancel()
            await asyncio.gather(progress_wait, error_wait, return_exceptions=True)
        if strict_error and not progress.is_set():
            raise OSError(
                "Telegram polling errored before first getUpdates success during initial connect: "
                f"{_redact_telegram_error_text(strict_error[0])}"
           ) from strict_error[0]
        if not progress.is_set():
            raise OSError("Telegram getUpdates did not become ready during initial connect")

    async def _start_polling_resilient(self, *, drop_pending_updates: bool, error_callback, require_progress: bool = False) -> bool:
        """Start PTB polling; ``require_progress`` (initial connect) demands real readiness. Reconnects
        may recover in background; on cold start a bootstrap failure raises (see _await_cold_start_readiness)."""
        if self._teardown_started:
            return False
        if not (self._app and self._app.updater):
            raise RuntimeError("Telegram application/updater not initialized")
        # Strict cold start: background recovery must not run while the readiness gate waits, else a G1
        # error starts G2 on the same partial app and GatewayRunner never disposes it.
        strict_error: list[BaseException] = []
        strict_error_event = asyncio.Event()
        strict_gate_open = True
        effective_callback = error_callback
        if require_progress:
            loop = asyncio.get_running_loop()

            def _strict_error_callback(error: Exception) -> None:
                # Once the gate closes, delegate so later errors still reach background recovery.
                if not strict_gate_open:
                    if error_callback is not None:
                        error_callback(error)
                    return
                if not strict_error:
                    strict_error.append(error)
                # Called from the polling task; set on the loop to wake the strict waiter.
                loop.call_soon_threadsafe(strict_error_event.set)

            effective_callback = _strict_error_callback
        try:
            # Same watchdog bound as the reconnect ladders; the TimeoutError is an OSError subclass, so
            # the except below classifies it as a network error → background recovery.
            # Same watchdog bound as the reconnect ladders: a wedged httpx connection pool can hang
            # start_polling() forever at bootstrap too (#59614).
            generation, progress = await self._start_polling_once(
                self._app, drop_pending_updates=drop_pending_updates, error_callback=effective_callback,
                abandon_app_on_timeout=require_progress,
                # The strict gate IS the cold-start verifier; a background one would race it.
                schedule_verifier=not require_progress)
            if require_progress:
                await self._await_cold_start_readiness(progress, strict_error_event, strict_error)
                # Readiness proven — close the gate so later errors reach background recovery.
                strict_gate_open = False
                self._polling_error_callback_ref = error_callback
            return True
        except _PollingLifecycleAbort:
            return False
        except Exception as err:
            if self._teardown_started:
                return False
            if require_progress:
                raise
            if self._looks_like_polling_conflict(err):
                logger.warning(
                    "[%s] Telegram polling bootstrap conflict; gateway stays alive while conflict retry runs: %s",
                    self.name, _redact_telegram_error_text(err))
                self._spawn_polling_recovery(asyncio.get_running_loop(), self._handle_polling_conflict(err))
                return False
            if self._looks_like_network_error(err):
                self._schedule_polling_recovery(err, reason="polling bootstrap")
                return False
            raise

    async def _go_fatal_network(self, message: str, log_message: str, *log_args) -> None:
        """Retryable ``telegram_network_error`` fatal + runner handoff (supervisor rebuilds the adapter)."""
        logger.error(log_message, *log_args)
        self._set_fatal_error("telegram_network_error", message, retryable=True)
        await self._handoff_polling_fatal_error()

    async def _stop_updater_or_go_fatal(self, app, what: str) -> bool:
        """Bounded ``updater.stop()`` before a recovery restart; False = went fatal, caller returns.

        Wall-clock deadline, not asyncio.wait_for: a CLOSE-WAIT socket wedges stop() on epoll and PTB/AnyIO shielded cleanup
        hangs wait_for. On timeout the Updater's lifecycle lock may still be held, so rebuild the adapter instead."""
        try:
            if app and app.updater and app.updater.running:
                try:
                    await _await_with_thread_deadline(app.updater.stop(), timeout=_UPDATER_STOP_TIMEOUT)
                except asyncio.TimeoutError:
                    message = (
                        f"Telegram updater.stop() did not finish before the {what} deadline; "
                        "rebuilding the adapter instead of reusing an Updater whose lifecycle lock may still be held.")
                    await self._go_fatal_network(message, "[%s] %s (likely CLOSE-WAIT socket)", self.name, message)
                    return False
        except Exception:
            pass
        return True

    def _restart_polling_in_task(self, coro) -> None:
        """Run a recovery coroutine as the tracked in-flight ``_polling_error_task``."""
        self._polling_error_task = asyncio.get_running_loop().create_task(coro)

    async def _handle_polling_network_error(self, error: Exception) -> None:
        """Reconnect polling after a transient network interruption (NetworkError/TimedOut).

        Host connectivity loss (sleep, WiFi switch, VPN) kills the long-poll silently. Exponential back-off (5s→60s
        cap) up to MAX_NETWORK_RETRIES, then retryable-fatal so the supervisor restarts the gateway."""
        if self._teardown_started or self.has_fatal_error:
            return
        MAX_NETWORK_RETRIES = 10
        BASE_DELAY = 5
        MAX_DELAY = 60
        self._polling_network_error_count += 1
        self._send_path_degraded = True
        attempt = self._polling_network_error_count
        if attempt > MAX_NETWORK_RETRIES:
            message = (
                "Telegram polling could not reconnect after %d network error retries. "
                "Escalating to gateway recovery." % MAX_NETWORK_RETRIES)
            await self._go_fatal_network(message, "[%s] %s Last error: %s", self.name, message, _redact_telegram_error_text(error))
            return
        delay = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
        logger.warning(
            "[%s] Telegram network error (attempt %d/%d), reconnecting in %ds. Error: %s", self.name, attempt,
            MAX_NETWORK_RETRIES, delay, _redact_telegram_error_text(error))
        await asyncio.sleep(delay)
        if self._teardown_started:
            return
        # Stable local ref: a concurrent disconnect() may set self._app = None while we await.
        app = self._app
        # Unguarded stop() on a CLOSE-WAIT socket would leave _polling_error_task perpetually
        # "in-flight" so every probe skips reconnect for hours.
        if not await self._stop_updater_or_go_fatal(app, "network-recovery") or self._teardown_started:
            return
        # start_polling() bootstraps through the *general* pool before getUpdates; a confirmed pool timeout means the request
        # was never sent, so rebuilding that pool is safe. Generic network errors stay polling-only (sends untouched).
        if self._looks_like_pool_timeout(error):
            await self._drain_general_connections_after_pool_timeout()
        if self._teardown_started:
            return
        await self._drain_polling_connections()
        if self._teardown_started:
            return
        try:
            if not app:
                raise RuntimeError("Telegram application was torn down during reconnect")
            await self._start_polling_once(app, drop_pending_updates=False, error_callback=self._polling_error_callback_ref)
            logger.info(
                "[%s] Telegram polling restarted after network error (attempt %d); health pending getUpdates progress", self.name, attempt)
        except _PollingLifecycleAbort:
            return
        except Exception as retry_err:
            if self._teardown_started:
                return
            logger.warning("[%s] Telegram polling reconnect failed: %s", self.name, _redact_telegram_error_text(retry_err))
            # Polling is dead and no more error callbacks will fire — chain the retry ourselves.
            if not self.has_fatal_error and not self._teardown_started:
                task = asyncio.ensure_future(self._handle_polling_network_error(retry_err))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                # The chained retry IS the in-flight recovery: it must replace the reentrancy guard.
                self._polling_error_task = task

    async def _polling_heartbeat_loop(self) -> None:
        """Detect dead Telegram TCP sockets (CLOSE-WAIT) by periodic probing.

        In CLOSE-WAIT epoll still reports the long-poll socket readable and nothing raises, so PTB's
        ``error_callback`` never fires. Probe ``get_me()`` on the *general* path (never the getUpdates pool);
        connect-level failures feed ``_handle_polling_network_error``. Runs for the connection's lifetime, catching
        steady-state wedges the one-shot verifier can't."""
        HEARTBEAT_INTERVAL = 90   # seconds between probes
        PROBE_TIMEOUT = 15        # seconds before declaring the path dead
        # Wedged-recovery watchdog: note when a recovery task is first seen in-flight and force-escalate
        # if the *same* task object still runs past the stuck timeout.
        # Tracked locally so no _polling_error_task assignment site needs to stamp a timestamp: the
        # heartbeat notes when it first observes a given recovery task still in-flight, and force-escalates
        # if the *same* task object is still running after _POLLING_ERROR_TASK_STUCK_TIMEOUT. A healthy
        # ladder attempt completes (task done) or chains to a new task well before then, so a single
        # long-lived task is unambiguously wedged. See #66377.
        stuck_task_ref: Optional[asyncio.Task] = None
        stuck_task_since = 0.0
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                if self._teardown_started or self.has_fatal_error:
                    return
                # A recovery task hung on an unbounded await gates every other recovery path forever
                # (alive but deaf): force retryable-fatal so the reconnector rebuilds the adapter.
                # Independent wedged-recovery watchdog (#66377): if the tracked recovery task has hung (any
                # await no local bound covers), every other recovery path is gated behind it and returns
                # early forever — the gateway stays alive but deaf.
                recovery_task = self._polling_error_task
                if recovery_task is not None and not recovery_task.done():
                    now = time.monotonic()
                    if recovery_task is not stuck_task_ref:
                        stuck_task_ref = recovery_task
                        stuck_task_since = now
                    elif now - stuck_task_since > _POLLING_ERROR_TASK_STUCK_TIMEOUT:
                        stuck_for = now - stuck_task_since
                        logger.error(
                            "[%s] Telegram reconnect task wedged for %.0fs with no ladder progress; forcing retryable-fatal so the gateway "
                            "reconnects instead of staying silently deaf.",
                            self.name, stuck_for)
                        with contextlib.suppress(Exception):
                            recovery_task.cancel()
                        self._set_fatal_error(
                            "telegram_network_error",
                            "Telegram reconnect task wedged for %.0fs; forcing gateway reconnect." % stuck_for,
                            retryable=True)
                        await self._handoff_polling_fatal_error()
                        return
                else:
                    stuck_task_ref = None
                bot = self._app.bot if self._app else None
                if bot is None:
                    continue
                # No get_me() ⇒ not a live polling client (torn down / test double): exit, don't spin.
                if not callable(getattr(bot, "get_me", None)):
                    return
                await asyncio.wait_for(bot.get_me(), PROBE_TIMEOUT)
                # get_me() refreshes PTB's cached bot user: adopt a BotFather rename before routing on it.
                self._bot_identity_checked_at = time.monotonic()
                self._note_bot_username(getattr(bot, "username", None))
                # get_me() OK proves only the send path; a wedged long-poll shows as server-side queue.
                # get_me() succeeded — the general/send request path is healthy. That does NOT prove the
                # getUpdates consumer is alive: PTB can report updater.running=True while the long-poll task
                # is wedged, so DMs queue in the Bot API and never reach handlers (#42909). get_me() is
                # blind to this; get_webhook_info() exposes it via pending_update_count. Escalate only after
                # two consecutive probes see a non-zero queue while we believe we're polling, so a single
                # in-flight update (consumed before the next probe) never trips recovery.
                await self._probe_pending_updates(bot, PROBE_TIMEOUT)
                # An empty queue can't hide a wedge forever: no round-trip past the stall threshold ⇒ dead.
                # Even an empty queue cannot hide a wedged long-poll forever: Telegram answers within ~50s,
                # so a consumer with no successful round-trip past the stall threshold is dead (#92991).
                # Pure local-state check — no Bot API call needed.
                await self._check_polling_stall()
            except asyncio.CancelledError:
                return
            except (asyncio.TimeoutError, OSError) as probe_err:
                self._schedule_polling_recovery(probe_err, reason="heartbeat probe")
            except Exception as probe_err:
                # Non-connectivity errors (e.g. TelegramError 401) aren't CLOSE-WAIT symptoms.
                if self._looks_like_network_error(probe_err):
                    self._schedule_polling_recovery(probe_err, reason="heartbeat probe")

    async def _probe_pending_updates(self, bot, probe_timeout: float) -> None:
        """Detect a wedged or stopped getUpdates consumer via pending_update_count.

        PTB can report ``updater.running`` while the long-poll is stuck; get_me() stays healthy yet DMs queue in the
        Bot API. A stuck queue over two consecutive probes ⇒ dead consumer. Also covers the updater having stopped
        entirely (``running=False``, no reconnect in flight).

        PTB can report ``updater.running == True`` while its long-poll task is silently stuck (e.g. a socket
        that epoll keeps reporting readable on WSL2). ``get_me()`` stays healthy because it uses the general
        request path, so the CLOSE-WAIT heartbeat never fires — yet DMs queue in the Bot API and never reach
        handlers (#42909).
        We detect the stopped updater directly and feed the same ladder (#55769).
        """
        # Polling mode only: in webhook mode Telegram pushes and holds no server-side queue.
        if self._teardown_started or self._webhook_mode:
            return
        # An in-flight reconnect owns recovery — don't double-trigger, and don't misread its brief
        # stop()->start_polling() window (updater.running transiently False) as dead.
        if self._recovery_in_flight():
            self._polling_not_running_count = 0
            return
        updater = getattr(self._app, "updater", None) if self._app else None
        if updater is None:
            self._polling_pending_stuck_count = 0
            return
        if not getattr(updater, "running", False):
            # Long-poll task gone, general-path calls still succeed, so no error_callback/probe ever
            # fires. Debounced over two probes so a just-starting updater never trips it.
            self._polling_pending_stuck_count = 0
            # We are in polling mode with no reconnect in flight, yet PTB's updater has stopped entirely.
            # This is distinct from the wedged-but-running consumer handled below: the long-poll task is
            # gone, get_me()/get_webhook_info() on the general request path still succeed, so no
            # error_callback or connectivity probe ever fires and the gateway silently stops receiving
            # messages while the process stays alive (#55769).
            self._polling_not_running_count += 1
            logger.warning(
                "[%s] Telegram polling heartbeat: updater stopped while in polling mode (stuck probe %d/2)", self.name,
                self._polling_not_running_count)
            if self._polling_not_running_count >= 2:
                self._polling_not_running_count = 0
                self._escalate_stuck_consumer(
                    "[%s] Telegram updater is not running (long-poll task gone); triggering polling restart",
                    "Telegram updater stopped while in polling mode")
            return
        self._polling_not_running_count = 0
        get_webhook_info = getattr(bot, "get_webhook_info", None)
        if not callable(get_webhook_info):
            return
        try:
            info = await asyncio.wait_for(get_webhook_info(), probe_timeout)  # type: ignore[arg-type]
        except (asyncio.TimeoutError, OSError):
            return  # connectivity symptom for the get_me() path, not a stuck-queue signal
        pending = int(getattr(info, "pending_update_count", 0) or 0)
        if pending <= 0:
            self._polling_pending_stuck_count = 0
            return
        self._polling_pending_stuck_count += 1
        logger.warning(
            "[%s] Telegram polling heartbeat: %d update(s) queued but not consumed (stuck probe %d/2)", self.name,
            pending, self._polling_pending_stuck_count)
        if self._polling_pending_stuck_count >= 2:
            self._polling_pending_stuck_count = 0
            self._escalate_stuck_consumer(
                "[%s] getUpdates consumer appears wedged (queue not draining); triggering polling restart",
                "getUpdates consumer wedged: pending updates not draining")

    def _escalate_stuck_consumer(self, log_message: str, reason: str) -> None:
        """Second consecutive stuck probe: restart polling via the network-error ladder (unless tearing down)."""
        if self._teardown_started:
            return
        logger.warning(log_message, self.name)
        self._polling_error_task = asyncio.get_running_loop().create_task(self._handle_polling_network_error(RuntimeError(reason)))

    async def _check_polling_stall(self) -> None:
        """Watchdog the last successful getUpdates round-trip: a long-poll can wedge without raising
        (CLOSE-WAIT after a route flip) while every other probe stays blind; no round-trip for
        ``_POLLING_STALL_TIMEOUT`` ⇒ escalate through the bounded reconnect ladder.

        See #92991.
        """
        if self._webhook_mode or self._teardown_started or self.has_fatal_error or self._recovery_in_flight():
            return
        now = time.monotonic()
        last_progress = getattr(self, "_polling_last_progress_monotonic", None)
        generation_started = getattr(self, "_polling_generation_started_monotonic", None)
        if last_progress is not None:
            stalled_for = now - last_progress
        elif generation_started is not None:
            # No round-trip yet this generation: fallback for when the one-shot verifier could not run.
            stalled_for = now - generation_started
        else:
            return
        if stalled_for <= _POLLING_STALL_TIMEOUT:
            return
        logger.error(
            "[%s] Telegram polling stalled: no getUpdates progress for %.0fs "
            "(generation %d). Rebuilding the long-poll consumer through the reconnect ladder instead of staying silently deaf.",
            self.name, stalled_for, getattr(self, "_polling_generation", 0))
        self._spawn_polling_recovery(
            asyncio.get_running_loop(),
            self._handle_polling_network_error(
                RuntimeError("getUpdates made no progress for %.0fs (polling stall watchdog)" % stalled_for)))

    def _verifier_stale(self, generation: int, progress: asyncio.Event) -> bool:
        """True when a verifier's generation no longer matters (progressed, fatal, replaced, torn down)."""
        return (
            self._teardown_started or progress.is_set() or self.has_fatal_error
            or not self._polling_progress_accepting or generation != self._polling_generation
            or progress is not self._polling_progress_event)

    async def _verify_polling_after_reconnect(self, generation: Optional[int] = None, progress: Optional[asyncio.Event] = None) -> None:
        """Require getUpdates progress, using getMe only to classify failure: a general-path getMe
        success cannot heal polling health. Connectivity failures enter the guarded recovery ladder."""
        PROBE_TIMEOUT = 10
        if self._teardown_started:
            return
        if generation is None:
            generation = self._polling_generation
        if progress is None:
            progress = self._polling_progress_event
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(progress.wait(), timeout=_POLLING_PROGRESS_TIMEOUT)
        if self._verifier_stale(generation, progress):
            return
        app = self._app
        if not (app and app.updater and app.updater.running):
            logger.warning("[%s] Updater made no getUpdates progress and is not running", self.name)
            self._schedule_polling_recovery(
                RuntimeError("Updater not running after polling progress deadline"),
                reason="polling progress verifier: updater not running")
            return
        try:
            await asyncio.wait_for(app.bot.get_me(), PROBE_TIMEOUT)
        except Exception as probe_err:
            if self._verifier_stale(generation, progress):
                return
            if not self._looks_like_network_error(probe_err):
                logger.warning(
                    "[%s] Polling progress verifier hit a non-connectivity error (not retrying): %s", self.name,
                    _redact_telegram_error_text(probe_err))
                return
            logger.warning(
                "[%s] Polling progress verifier connectivity probe failed: %s", self.name, _redact_telegram_error_text(probe_err))
            self._schedule_polling_recovery(probe_err, reason="polling progress verifier connectivity failure")
            return
        if self._verifier_stale(generation, progress):
            return
        self._schedule_polling_recovery(
            RuntimeError("getUpdates made no progress before verifier deadline"),
            reason="polling progress verifier: general path healthy but getUpdates stalled")

    def _disarm_ptb_retry_loop(self) -> None:
        """Synchronously stop PTB's internal polling retry loop.

        PTB's ``network_retry_loop`` calls our ``error_callback`` *synchronously* on a 409 Conflict then polls again; our
        callback only schedules async recovery, so two sessions overlap and Telegram 409s on a ~31s cadence. Setting PTB's
        private ``stop_event`` makes its loop exit on the next tick; ``updater.stop()`` + drain + ``start_polling()`` then build
        a fresh one. Best-effort across PTB spellings. Deliberately NOT flipping ``updater._running``: stop() raises when
        already False, which would skip the real teardown and poison the next start."""
        updater = getattr(self._app, "updater", None) if self._app else None
        if updater is None:
            return
        for attr in ("_Updater__polling_task_stop_event", "_polling_task_stop_event"):
            stop_event = getattr(updater, attr, None)
            if isinstance(stop_event, asyncio.Event):
                if not stop_event.is_set():
                    stop_event.set()
                    logger.debug("[%s] Disarmed PTB polling retry loop via %s", self.name, attr)
                return
        logger.debug(
            "[%s] Could not disarm PTB polling retry loop (stop_event not found on this PTB version); falling back to async stop()",
            self.name)

    async def _handle_polling_conflict(self, error: Exception) -> None:
        """Recover a 409 Conflict: the previous gateway process was killed but Telegram holds its
        getUpdates session ~30s. Stop, wait (growing delay), drain, restart — MAX_CONFLICT_RETRIES
        times before going fatal; a failed retry must never return silently (limbo)."""
        if self._teardown_started:
            return
        if self.has_fatal_error and self.fatal_error_code == "telegram_polling_conflict":
            return
        self._polling_conflict_count += 1
        MAX_CONFLICT_RETRIES = 5
        # 15s, 25s, 35s, 45s, 55s — clears Telegram's ~30s session window without hammering the API.
        RETRY_DELAY = 10 + (self._polling_conflict_count * 10)  # seconds
        if self._polling_conflict_count <= MAX_CONFLICT_RETRIES:
            logger.warning(
                "[%s] Telegram polling conflict (%d/%d) — previous session still "
                "held open on Telegram's servers. Waiting %ds for it to expire. Error: %s",
                self.name, self._polling_conflict_count, MAX_CONFLICT_RETRIES,
                RETRY_DELAY, _redact_telegram_error_text(error))
            # Stop the updater before sleeping (no-op if PTB raised before running was set).
            if not await self._stop_updater_or_go_fatal(self._app, "conflict-retry"):
                return
            await asyncio.sleep(RETRY_DELAY)
            if self._teardown_started:
                return
            await self._drain_polling_connections()
            if self._teardown_started:
                return
            # Stable local ref: a concurrent disconnect() may null self._app across the awaits above.
            app = self._app
            # Capture a stable local reference: self._app can be reassigned to None by a concurrent
            # disconnect() while we're suspended across the awaits above (same race #55992 fixed on the
            # network path). Re-reading self._app after that point would raise AttributeError deep inside
            # start_polling instead of failing fast here, where the except below reschedules or escalates to
            # fatal.
            expected_generation = self._polling_generation + 1
            if not app:
                raise RuntimeError("Telegram application was torn down during conflict reconnect")
            # drop_pending_updates=True makes Telegram terminate any other getUpdates session for this
            # token (zombie or our own prior retry); without it each retry is immediately 409'd.
            # The competing session is either a zombie from the previous gateway process (whose long-poll
            # hasn't expired server-side yet) or our own previous retry's still-expiring session. Without
            # this, each retry starts a new getUpdates session that immediately gets 409'd by the previous
            # one, creating the very conflict we are trying to recover from (#75017).
            self._polling_conflict_recovery_generation = expected_generation
            try:
                await self._start_polling_once(app, drop_pending_updates=True, error_callback=self._polling_error_callback_ref)
                logger.info(
                    "[%s] Telegram polling restarted after conflict retry %d/%d; health pending getUpdates progress",
                    self.name, self._polling_conflict_count, MAX_CONFLICT_RETRIES)
                return
            except _PollingLifecycleAbort:
                return
            except Exception as retry_err:
                if self._teardown_started:
                    return
                logger.warning(
                    "[%s] Telegram polling retry %d/%d failed: %s. Scheduling next attempt.", self.name,
                    self._polling_conflict_count, MAX_CONFLICT_RETRIES, _redact_telegram_error_text(retry_err))
                # Never return silently: alive-and-"connected" with no polling is limbo.
                if self._polling_conflict_count < MAX_CONFLICT_RETRIES and not self._teardown_started:
                    # get_running_loop(): get_event_loop() raises on 3.10+ from PTB's callback context.
                    self._restart_polling_in_task(self._handle_polling_conflict(retry_err))
                    return
                # Fall through to fatal on the last retry.
            finally:
                if self._polling_conflict_recovery_generation == expected_generation:
                    self._polling_conflict_recovery_generation = None
        if self._teardown_started:
            return
        # Retries exhausted — fatal so the runner surfaces it and the user knows to act.
        message = (
            "Telegram polling could not recover after %d retries (%ds total wait). "
            "The previous gateway session is still held open on Telegram's servers, "
            "or another process is using the same bot token. To recover: ensure no other Hermes or OpenClaw instance is running "
            "with this token, then restart the gateway with 'hermes gateway restart'."
            % (MAX_CONFLICT_RETRIES, sum(10 + i * 10 for i in range(1, MAX_CONFLICT_RETRIES + 1))))
        logger.error("[%s] %s Original error: %s", self.name, message, _redact_telegram_error_text(error))
        # Snapshot whether WE transition to fatal: a concurrent retry task suspended past the entry
        # guard reaches this branch too. Only the first transition notifies.
        _already_fatal = self.has_fatal_error and self.fatal_error_code == "telegram_polling_conflict"
        self._set_fatal_error("telegram_polling_conflict", message, retryable=False)
        try:
            if self._app and self._app.updater:
                await _await_with_thread_deadline(self._app.updater.stop(), timeout=_UPDATER_STOP_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("[%s] updater.stop() timed out after exhausting conflict retries (likely CLOSE-WAIT socket); proceeding to fatal notify", self.name)
        except Exception as stop_error:
            logger.warning(
                "[%s] Failed stopping Telegram updater after exhausting conflict retries: %s", self.name, stop_error,
                exc_info=True,
            )
        if not _already_fatal:
            await self._handoff_polling_fatal_error()

    async def _handoff_polling_fatal_error(self) -> None:
        """Notify the runner without letting child teardown cancel this owner: ``disconnect()`` cancels
        the tracked recovery/heartbeat tasks, so release only the current owner from its field."""
        current_task = asyncio.current_task()
        if self._polling_error_task is current_task:
            self._polling_error_task = None
        if getattr(self, "_polling_heartbeat_task", None) is current_task:
            self._polling_heartbeat_task = None
        await self._notify_fatal_error()

    async def _create_dm_topic(
        self, chat_id: int, name: str, icon_color: Optional[int] = None, icon_custom_emoji_id: Optional[str] = None) -> Optional[int]:
        """Create a forum topic in a private (DM) chat (Bot API 9.4+); message_thread_id or None."""
        if not self._bot:
            return None
        try:
            kwargs: Dict[str, Any] = {"chat_id": chat_id, "name": name}
            if icon_color is not None:
                kwargs["icon_color"] = icon_color
            if icon_custom_emoji_id:
                kwargs["icon_custom_emoji_id"] = icon_custom_emoji_id
            topic = await self._bot.create_forum_topic(**kwargs)
            thread_id = topic.message_thread_id
            logger.info("[%s] Created DM topic '%s' in chat %s -> thread_id=%s", self.name, name, chat_id, thread_id)
            return thread_id
        except Exception as e:
            error_text = str(e).lower()
            # Telegram has no "list topics" API: an existing topic is mapped from incoming messages.
            if "topic_name_duplicate" in error_text or "already" in error_text:
                logger.info(
                    "[%s] DM topic '%s' already exists in chat %s (will be mapped from incoming messages)", self.name, name, chat_id)
            elif "not a forum" in error_text or "forums_disabled" in error_text:
                logger.warning(
                    "[%s] Cannot create DM topic '%s' in chat %s: Topics mode is not enabled. "
                    "The user must open the DM with this bot in Telegram, tap the bot name "
                    "at the top, and enable 'Topics' in chat settings before topics can be created.",
                    self.name, name, chat_id)
            else:
                logger.warning(
                    "[%s] Failed to create DM topic '%s' in chat %s: %s", self.name, name, chat_id, _redact_telegram_error_text(e))
            return None

    async def create_handoff_thread(self, parent_chat_id: str, name: str) -> Optional[str]:
        """Create a forum topic for a session handoff; ``message_thread_id`` as str, or None."""
        try:
            chat_id_int = int(parent_chat_id)
        except (TypeError, ValueError):
            return None
        thread_id = await self._create_dm_topic(chat_id_int, name=name)
        return str(thread_id) if thread_id else None

    async def ensure_dm_topic(self, chat_id: str, topic_name: str, force_create: bool = False) -> Optional[str]:
        """Return a private DM topic thread id, creating and persisting it if needed."""
        name = str(topic_name or "").strip()
        if not name:
            return None
        try:
            chat_id_int = int(chat_id)
        except (TypeError, ValueError):
            return None
        cache_key = f"{chat_id_int}:{name}"
        cached = self._dm_topics.get(cache_key)
        if cached and not force_create:
            return str(cached)
        topic_conf: Optional[Dict[str, Any]] = None
        chat_entry: Optional[Dict[str, Any]] = None
        for entry in self._dm_topics_config:
            if str(entry.get("chat_id")) != str(chat_id_int):
                continue
            chat_entry = entry
            topic_conf = next((c for c in entry.get("topics", []) if c.get("name") == name), None)
            break
        if topic_conf and topic_conf.get("thread_id") and not force_create:
            thread_id = int(topic_conf["thread_id"])
            self._dm_topics[cache_key] = thread_id
            return str(thread_id)
        if chat_entry is None:
            chat_entry = {"chat_id": chat_id_int, "topics": []}
            self._dm_topics_config.append(chat_entry)
        if topic_conf is None:
            topic_conf = {"name": name}
            chat_entry.setdefault("topics", []).append(topic_conf)
        thread_id = await self._create_dm_topic(
            chat_id_int, name=name, icon_color=topic_conf.get("icon_color"), icon_custom_emoji_id=topic_conf.get("icon_custom_emoji_id"))
        if not thread_id:
            return None
        topic_conf["thread_id"] = thread_id
        self._dm_topics[cache_key] = int(thread_id)
        self._persist_dm_topic_thread_id(chat_id_int, name, int(thread_id), replace_existing=force_create)
        return str(thread_id)

    async def rename_dm_topic(self, chat_id: int, thread_id: int, name: str) -> None:
        """Rename a forum topic in a private (DM) chat."""
        if not self._bot:
            return
        try:
            chat_id_arg = int(chat_id)
        except (TypeError, ValueError):
            chat_id_arg = chat_id
        await self._bot.edit_forum_topic(chat_id=chat_id_arg, message_thread_id=int(thread_id), name=name)
        logger.info("[%s] Renamed DM topic in chat %s thread_id=%s -> '%s'", self.name, chat_id, thread_id, name)

    def _persist_dm_topic_thread_id(self, chat_id: int, topic_name: str, thread_id: int, replace_existing: bool = False) -> None:
        """Save a newly created thread_id back into config.yaml so it survives restarts."""
        try:
            from hermes_constants import get_hermes_home
            config_path = get_hermes_home() / "config.yaml"
            if not config_path.exists():
                logger.warning("[%s] Config file not found at %s, cannot persist thread_id", self.name, config_path)
                return
            from hermes_cli.config import atomic_config_write, read_user_config_raw
            config = read_user_config_raw(config_path)
            # platforms.telegram.extra.dm_topics — create the path for topics not predeclared in config.yaml.
            dm_topics = config.setdefault("platforms", {}).setdefault("telegram", {}).setdefault("extra", {}).setdefault("dm_topics", [])
            changed = False
            matching_chat_entry = None
            for chat_entry in dm_topics:
                try:
                    if int(chat_entry.get("chat_id", 0)) != int(chat_id):
                        continue
                except (TypeError, ValueError):
                    continue
                matching_chat_entry = chat_entry
                topics = chat_entry.setdefault("topics", [])
                t = next((t for t in topics if t.get("name") == topic_name), None)
                if t is None:
                    topics.append({"name": topic_name, "thread_id": thread_id})
                    changed = True
                elif (replace_existing or not t.get("thread_id")) and t.get("thread_id") != thread_id:
                    t["thread_id"] = thread_id
                    changed = True
                break
            if matching_chat_entry is None:
                dm_topics.append({"chat_id": chat_id, "topics": [{"name": topic_name, "thread_id": thread_id}]})
                changed = True
            if changed:
                atomic_config_write(config_path, config, default_flow_style=False, sort_keys=False)
                logger.info("[%s] Persisted thread_id=%s for topic '%s' in config.yaml", self.name, thread_id, topic_name)
        except Exception as e:
            logger.warning("[%s] Failed to persist thread_id to config: %s", self.name, e, exc_info=True)

    async def _setup_dm_topics(self) -> None:
        """Load or create configured DM topics: ``extra['dm_topics']`` is ``[{"chat_id", "topics": [{"name",
        "icon_color", "thread_id"?, "skill"?}]}]``; persisted thread_ids are cached without an API call."""
        for chat_entry in self._dm_topics_config or ():
            chat_id = chat_entry.get("chat_id")
            topics = chat_entry.get("topics", [])
            if not chat_id or not topics:
                continue
            logger.info("[%s] Setting up %d DM topic(s) for chat %s", self.name, len(topics), chat_id)
            for topic_conf in topics:
                topic_name = topic_conf.get("name")
                if not topic_name:
                    continue
                cache_key = f"{chat_id}:{topic_name}"
                existing_thread_id = topic_conf.get("thread_id")
                if existing_thread_id:
                    self._dm_topics[cache_key] = int(existing_thread_id)
                    logger.info("[%s] DM topic loaded from config: %s -> thread_id=%s", self.name, cache_key, existing_thread_id)
                    continue
                thread_id = await self._create_dm_topic(
                    chat_id=normalize_telegram_chat_id(chat_id), name=topic_name, icon_color=topic_conf.get("icon_color"),
                    icon_custom_emoji_id=topic_conf.get("icon_custom_emoji_id"))
                if not thread_id:
                    continue
                self._dm_topics[cache_key] = thread_id
                logger.info("[%s] DM topic cached: %s -> thread_id=%s", self.name, cache_key, thread_id)
                self._persist_dm_topic_thread_id(int(chat_id), topic_name, thread_id)
                # Seed message: Telegram's client hides empty topics until they contain one.
                try:
                    await self._bot.send_message(
                        chat_id=normalize_telegram_chat_id(chat_id), message_thread_id=thread_id, text=f"\U0001f4cc {topic_name}")
                except Exception as seed_err:
                    logger.debug("[%s] Could not send seed message to topic '%s': %s", self.name, topic_name, seed_err)

    async def _bot_identity_refresh_loop(self) -> None:
        """Keep the cached @username fresh in webhook mode (no heartbeat calls ``get_me()`` there)."""
        while True:
            try:
                await asyncio.sleep(self._BOT_IDENTITY_TTL_SECONDS)
                if self._teardown_started or self.has_fatal_error:
                    return
                await self._refresh_bot_identity(force=True)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("[%s] Telegram identity refresh loop iteration failed", self.name, exc_info=True)

    def _start_post_connect_housekeeping(self) -> None:
        """Kick off deferred post-connect housekeeping; idempotent while a task is still running."""
        task = self._post_connect_task
        if task and not task.done():
            return
        self._post_connect_task = asyncio.ensure_future(self._run_post_connect_housekeeping())

    async def _register_command_menu(self) -> None:
        """Register the command menu (from COMMAND_REGISTRY) in every scope — Telegram picks the
        narrowest matching one per chat type; forum topics are handled lazily by _ensure_forum_commands."""
        from telegram import BotCommand, BotCommandScopeAllPrivateChats, BotCommandScopeAllGroupChats, BotCommandScopeDefault
        from hermes_cli.commands_platforms import telegram_menu_commands, telegram_menu_max_commands
        if not self._bot:
            return
        # Telegram allows 100 commands but has an undocumented ~4KB payload limit; default cap 60.
        max_commands = telegram_menu_max_commands()
        menu_commands, hidden_count = telegram_menu_commands(max_commands=max_commands)
        bot_commands = [BotCommand(name, desc) for name, desc in menu_commands]
        for scope_cls in (BotCommandScopeDefault, BotCommandScopeAllPrivateChats, BotCommandScopeAllGroupChats):
            scope_name = getattr(scope_cls, "__name__", str(scope_cls))
            try:
                await self._bot.set_my_commands(bot_commands, scope=scope_cls())
                logger.info("[%s] set_my_commands OK for scope %s (%d cmds)", self.name, scope_name, len(bot_commands))
            except Exception as scope_err:
                logger.warning("[%s] set_my_commands FAILED for scope %s: %s", self.name, scope_name, scope_err)
        if hidden_count:
            logger.info(
                "[%s] Telegram menu: %d commands registered, %d hidden (over %d limit). Use /commands for full list.",
                self.name, len(menu_commands), hidden_count, max_commands)

    async def _run_post_connect_housekeeping(self) -> None:
        """Command menu, status indicator and DM topics off the connect path; every step is non-fatal.

        DM topics — all off the connect path so a slow Bot API call cannot blow the gateway connect timeout
        (#46298).
        """
        try:
            try:
                await self._register_command_menu()
            except Exception as e:
                logger.warning(
                    "[%s] Could not register Telegram command menu: %s", self.name, _redact_telegram_error_text(e), exc_info=True)
            with contextlib.suppress(Exception):
                await self._set_status_indicator(online=True)
            try:
                await self._setup_dm_topics()
            except Exception as topics_err:
                logger.warning("[%s] DM topics setup failed (non-fatal): %s", self.name, topics_err, exc_info=True)
        except asyncio.CancelledError:
            raise
        finally:
            if self._post_connect_task is asyncio.current_task():
                self._post_connect_task = None

    async def _on_platform_update(self, update, context) -> None:
        """Catch-all PTB handler (group 99) firing ``gateway_platform_event`` per inbound update with a
        stable envelope (no raw SDK objects) and an internal auth source. Never raises into PTB."""
        handler: Optional[Callable[[Dict[str, Any], Any], Awaitable[None]]] = getattr(self, "_platform_event_handler", None)
        if handler is None:
            return
        try:
            from hermes_cli.lifecycle import has_hook
            if not has_hook("gateway_platform_event"):
                return
            event = self._normalize_platform_event(update)
        except Exception:
            logger.debug("[%s] gateway_platform_event normalize error", self.name, exc_info=True)
            return
        if event is None:
            return
        # The gateway-owned boundary runs the full profile-scoped auth chain before plugin dispatch.
        try:
            source = self._source_for_platform_event_auth(update)
            await handler(event, source)
        except Exception:
            logger.debug("[%s] gateway_platform_event dispatch error", self.name, exc_info=True)

    def _source_for_platform_event_auth(self, update):
        """Route a supported update to its event-specific auth-source extractor (reactor / editor);
        raises ``ValueError`` for updates without one so the boundary fails closed."""
        if getattr(update, "message_reaction", None) is not None:
            return self._source_from_reaction_for_auth(update)
        edited = getattr(update, "edited_message", None)
        if edited is not None:
            source = self._source_from_message_for_auth(edited)
            # Tolerates missing identities for pairing-flow callers; this boundary must not.
            if not source.user_id or not source.chat_id:
                raise ValueError("gateway_platform_event message_edited requires editor and chat identities")
            return source
        raise ValueError("gateway_platform_event source extraction has no extractor for this update type")

    def _normalize_platform_event(self, update) -> Optional[Dict[str, Any]]:
        """Map a PTB update to a ``{platform, event_type, payload}`` envelope (hooks.md contracts), or
        ``None`` for types without one."""
        if getattr(update, "message_reaction", None) is not None:
            return self._normalize_reaction_event(update)
        if getattr(update, "edited_message", None) is not None:
            return self._normalize_message_edited_event(update)
        return None

    @staticmethod
    def _is_id_like(value: Any) -> bool:
        return not isinstance(value, bool) and isinstance(value, (str, int))

    def _normalize_reaction_event(self, update) -> Optional[Dict[str, Any]]:
        """``message_reaction`` → ``reaction`` event: emojis (unicode), custom_emoji_ids, chat_id,
        message_id, thread_id (always None — reactions carry none)."""
        mr = getattr(update, "message_reaction", None)
        if mr is None:
            return None
        chat = getattr(mr, "chat", None)
        new_reaction = getattr(mr, "new_reaction", None) or []
        if not isinstance(new_reaction, (list, tuple)):
            return None
        chat_id = getattr(chat, "id", None) if chat is not None else None
        message_id = getattr(mr, "message_id", None)
        if not self._is_id_like(chat_id) or not self._is_id_like(message_id):
            return None
        emojis: List[str] = []
        custom_emoji_ids: List[str] = []
        for r in new_reaction[:64]:
            emoji = getattr(r, "emoji", None)
            if isinstance(emoji, str) and emoji:
                emojis.append(emoji[:64])
            custom_id = getattr(r, "custom_emoji_id", None)
            if self._is_id_like(custom_id):
                custom_emoji_ids.append(str(custom_id)[:128])
        return {
            "platform": "telegram",
            "event_type": "reaction",
            "payload": {
                "emojis": emojis, "custom_emoji_ids": custom_emoji_ids, "chat_id": str(chat_id)[:128],
                "message_id": str(message_id)[:128], "thread_id": None},
        }

    def _normalize_message_edited_event(self, update) -> Optional[Dict[str, Any]]:
        """``edited_message`` → ``message_edited`` event (v1, additive): chat_id, message_id, thread_id
        (forum topic), text (edited text or caption, bounded), edited_at (ISO 8601 UTC or None)."""
        message = getattr(update, "edited_message", None)
        if message is None:
            return None
        chat = getattr(message, "chat", None)
        chat_id = getattr(chat, "id", None) if chat is not None else None
        message_id = getattr(message, "message_id", None)
        if not self._is_id_like(chat_id) or not self._is_id_like(message_id):
            return None
        text = getattr(message, "text", None) or getattr(message, "caption", None)
        if not isinstance(text, str):
            text = None
        thread_id = None
        thread_id_raw = getattr(message, "message_thread_id", None)
        if self._is_id_like(thread_id_raw) and bool(getattr(message, "is_topic_message", False)):
            thread_id = str(thread_id_raw)[:128]
        edited_at = None
        edit_date = getattr(message, "edit_date", None)
        try:
            if edit_date is not None and hasattr(edit_date, "isoformat"):
                edited_at = str(edit_date.isoformat())[:64]
        except Exception:
            edited_at = None
        return {
            "platform": "telegram",
            "event_type": "message_edited",
            "payload": {
                "chat_id": str(chat_id)[:128], "message_id": str(message_id)[:128], "thread_id": thread_id,
                "text": text[:8192] if text is not None else None, "edited_at": edited_at},
        }

    def _register_handlers(self, app) -> None:
        """Register every PTB handler on ``app`` (initial connect and the transient-init rebuild)."""
        app.add_handler(TelegramMessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text_message))
        app.add_handler(TelegramMessageHandler(filters.COMMAND, self._handle_command))
        app.add_handler(TelegramMessageHandler(
            filters.LOCATION | getattr(filters, "VENUE", filters.LOCATION), self._handle_location_message))
        app.add_handler(TelegramMessageHandler(
            filters.PHOTO | filters.VIDEO | filters.AUDIO | filters.VOICE | filters.Document.ALL | filters.Sticker.ALL,
            self._handle_media_message))
        app.add_handler(CallbackQueryHandler(self._handle_callback_query))
        # Inline command picker; inert until the owner enables inline mode via BotFather /setinline.
        app.add_handler(InlineQueryHandler(self._handle_inline_query))
        # gateway_platform_event observer: group 99 observes alongside, never displaces, core handlers.
        app.add_handler(TypeHandler(Update, self._on_platform_update), group=99)

    async def _build_ptb_requests(self) -> tuple:
        """Build the (general, getUpdates) HTTPXRequest pair: fallback-IP transport, explicit proxy, or
        direct DNS; the getUpdates request is instrumented for polling-progress tracking."""
        # PTB's pool_timeout=1s default trips "Pool timeout" on flaky networks; safer defaults + env overrides.
        request_kwargs = {
            "connection_pool_size": env_int("HERMES_TELEGRAM_HTTP_POOL_SIZE", 512),
            "pool_timeout": env_float("HERMES_TELEGRAM_HTTP_POOL_TIMEOUT", 8.0),
            "connect_timeout": env_float("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", 10.0),
            "read_timeout": env_float("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", 20.0),
            "write_timeout": env_float("HERMES_TELEGRAM_HTTP_WRITE_TIMEOUT", 20.0),
            # PTB routes file requests to media_write_timeout; httpx budgets it per socket write (stall
            # tolerance, not bandwidth), so 60s rides out congested-link buffer stalls.
            "media_write_timeout": 60.0,
        }
        # CLOSE_WAIT fd leak: PTB's httpx.AsyncClient has no keepalive tuning; inject platform_httpx_limits()
        # while preserving PTB's max_connections (httpx_kwargs is spread last, so `limits` here wins).
        # CLOSE_WAIT fd leak (#31599, same class as #18451): PTB's HTTPXRequest builds the underlying
        # httpx.AsyncClient with `limits = httpx.Limits(max_connections=connection_pool_size)` and *no*
        # keepalive tuning, so httpx's default keepalive_expiry=5.0 applies. Behind an HTTP proxy
        # (Cloudflare Warp etc.) a peer-initiated FIN can sit in CLOSE_WAIT longer than that, leaking fds in
        # the general request pool (_request[1]) which _drain_polling_connections never resets.
        from gateway.platforms._http_client_limits import platform_httpx_limits
        _base_limits = platform_httpx_limits()
        if _base_limits is not None:
            import httpx as _httpx
            _pool_limits = _httpx.Limits(
                max_connections=request_kwargs["connection_pool_size"],
                max_keepalive_connections=_base_limits.max_keepalive_connections, keepalive_expiry=_base_limits.keepalive_expiry)
            # A long-poll is continuously active, so keepalive expiry can't protect it from a server-side
            # close: never hand getUpdates a pooled socket from a previous poll.
            _updates_limits = _httpx.Limits(
                max_connections=request_kwargs["connection_pool_size"], max_keepalive_connections=0,
                keepalive_expiry=_base_limits.keepalive_expiry)
        else:  # pragma: no cover — httpx always present alongside PTB
            _pool_limits = _updates_limits = None

        def _with_limits(httpx_kwargs: Optional[dict] = None) -> dict:
            """Merge tuned limits into httpx client kwargs (proxy/direct branches only; the fallback-IP
            branch must pass limits straight into the transport — httpx ignores client `limits` then)."""
            kwargs = dict(httpx_kwargs or {})
            if _pool_limits is not None and "limits" not in kwargs:
                kwargs["limits"] = _pool_limits
            return kwargs

        disable_fallback = os.getenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "").strip().lower() in {"1", "true", "yes", "on"}
        fallback_ips = [] if disable_fallback else self._fallback_ips()
        if not fallback_ips and not disable_fallback:
            discovery_timeout = self._env_float_clamped("HERMES_TELEGRAM_FALLBACK_DISCOVERY_TIMEOUT", 5.0, min_value=0.0)
            logger.warning("[%s] Discovering Telegram API fallback IPs via DNS-over-HTTPS…", self.name)
            try:
                fallback_ips = await _await_with_thread_deadline(discover_fallback_ips(), timeout=discovery_timeout)
            except Exception as exc:
                logger.warning(
                    "[%s] Telegram fallback-IP discovery failed after %.0fs; "
                    "using seed IPv4 Telegram API IPs so a blackholed IPv6 hostname path cannot hang initialize() (#87015): %s",
                    self.name, discovery_timeout, _redact_telegram_error_text(exc))
                fallback_ips = list(SEED_FALLBACK_IPS)
            else:
                logger.info("[%s] Auto-discovered Telegram fallback IPs: %s", self.name, ", ".join(fallback_ips))
        proxy_url = resolve_proxy_url("TELEGRAM_PROXY", target_hosts=["api.telegram.org", *fallback_ips])

        def _pair(general_httpx: dict, updates_httpx: dict, **extra) -> tuple:
            return (HTTPXRequest(**request_kwargs, **extra, httpx_kwargs=general_httpx),
                    HTTPXRequest(**request_kwargs, **extra, httpx_kwargs=updates_httpx))

        if fallback_ips and not proxy_url and not disable_fallback:
            logger.info("[%s] Telegram fallback IPs active: %s", self.name, ", ".join(fallback_ips))
            # Separate request/update pools reduce contention during polling reconnect + bootstrap calls.
            _transport_kwargs: dict = {"socket_options": tcp_keepalive_socket_options()}
            # Keep request/update pools separate to reduce contention during polling reconnect + bot API
            # bootstrap/delete_webhook calls. httpx ignores the client-level `limits` kwarg when a custom
            # `transport` is supplied (#58790). Unlike the proxy/direct branches (which inject limits at the
            # client level via `_with_limits`), this branch MUST pass the tuned limits directly into
            # TelegramFallbackTransport so its inner AsyncHTTPTransport instances honour keepalive_expiry —
            # do not route this through `_with_limits`, httpx would discard it.
            if _pool_limits is not None:
                _transport_kwargs["limits"] = _pool_limits
            _updates_transport_kwargs = dict(_transport_kwargs)
            if _updates_limits is not None:
                _updates_transport_kwargs["limits"] = _updates_limits
            request, get_updates_request = _pair(
                {"transport": TelegramFallbackTransport(fallback_ips, **_transport_kwargs)},
                {"transport": TelegramFallbackTransport(fallback_ips, **_updates_transport_kwargs)})
        elif proxy_url:
            logger.info("[%s] Proxy detected; passing explicitly to HTTPXRequest: %s", self.name, proxy_url)
            request, get_updates_request = _pair(_with_limits(), {"limits": _updates_limits}, proxy=proxy_url)
        else:
            if disable_fallback:
                logger.info("[%s] Telegram fallback-IP transport disabled via env", self.name)
            request, get_updates_request = _pair(_with_limits(), {"limits": _updates_limits})
        return request, self._instrument_polling_request(get_updates_request)

    async def _initialize_app_with_retries(self, builder) -> None:
        """``app.initialize()`` with a bounded retry ladder; rebuilds ``self._app``/``self._bot`` from
        ``builder`` after each failed attempt; OSError when the per-attempt or total watchdog expires."""
        _max_connect = 8
        _init_timeout = env_float("HERMES_TELEGRAM_INIT_TIMEOUT", 30.0)  # per attempt
        # Total watchdog: bounds the whole connect loop even if the retry loop silently stalls.
        _total_deadline = asyncio.get_running_loop().time() + _init_timeout * _max_connect + 120.0
        _timed_out = f"Telegram initialization timed out after {_max_connect} attempts ({_init_timeout:.0f}s each)"
        for _attempt in range(_max_connect):
            rebuild_app = False
            try:
                if asyncio.get_running_loop().time() >= _total_deadline:
                    raise OSError(
                        f"{_timed_out} — total connect watchdog deadline ({_init_timeout * _max_connect + 120.0:.0f}s) exceeded. "
                        f"Check network connectivity to api.telegram.org or set HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT / "
                        f"HERMES_TELEGRAM_INIT_TIMEOUT to a lower value.")
                logger.warning("[%s] Connecting to Telegram (attempt %d/%d)…", self.name, _attempt + 1, _max_connect)
                # On timeout the (possibly shielded) initialize() task is abandoned; release the half-built
                # app's httpx client so it isn't leaked across the ladder.
                await _await_with_thread_deadline(
                    self._app.initialize(), timeout=_init_timeout, on_abandon=lambda app=self._app: _shutdown_abandoned_app(app))
                break
            except asyncio.TimeoutError:
                rebuild_app = True
                if _attempt >= _max_connect - 1:
                    raise OSError(
                        f"{_timed_out}. Check network connectivity to api.telegram.org "
                        f"or set HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT to a lower value.")
                wait = min(2 ** _attempt, 15)
                logger.warning(
                    "[%s] Connect attempt %d/%d timed out after %.0fs — retrying in %ds", self.name, _attempt + 1,
                    _max_connect, _init_timeout, wait)
                await asyncio.sleep(wait)
            except Exception as init_err:
                # OSError always retries; anything else only when it looks like a network error.
                rebuild_app = True
                if (not isinstance(init_err, OSError) and not self._looks_like_network_error(init_err)) or _attempt >= _max_connect - 1:
                    raise
                wait = min(2 ** _attempt, 15)
                logger.warning(
                    "[%s] Connect attempt %d/%d failed: %s — retrying in %ds", self.name, _attempt + 1, _max_connect, init_err, wait)
                await asyncio.sleep(wait)
            except BaseException:
                # CancelledError etc.: log for the operator, then reraise. LAST so the Exception handlers win.
                logger.warning(
                    "[%s] Connect attempt %d/%d interrupted by %s — propagating", self.name, _attempt + 1, _max_connect,
                    "CancelledError" if isinstance(sys.exc_info()[1], asyncio.CancelledError) else type(sys.exc_info()[1]).__name__)
                raise
            finally:
                # A failed attempt may leave the app half-initialized: rebuild a fresh Application from the
                # same builder for the next attempt and discard the old one.
                if rebuild_app and _attempt < _max_connect - 1:
                    old_app = self._app
                    self._app = builder.build()
                    self._bot = self._app.bot
                    self._register_handlers(self._app)  # keep core and observer handlers in lockstep
                    with contextlib.suppress(Exception):
                        await _shutdown_abandoned_app(old_app)

    async def _start_webhook_mode(self, webhook_url: str, *, is_reconnect: bool) -> None:
        """Start PTB's webhook server (Telegram pushes updates; lets cloud platforms auto-wake suspended
        machines). SECURITY: TELEGRAM_WEBHOOK_SECRET is REQUIRED — without it the endpoint accepts forged
        updates (GHSA-3vpc-7q5r-276h); refuse to start rather than run fail-open."""
        webhook_port = env_int("TELEGRAM_WEBHOOK_PORT", 8443)
        # Default "" → tornado listens on IPv4 + IPv6; "0.0.0.0" is unreachable on IPv6-only networks.
        webhook_host = (os.getenv("TELEGRAM_WEBHOOK_HOST", "").strip() or str((self.config.extra or {}).get("webhook_host") or "").strip())
        # Profile-scoped read; only an UNSCOPED read under multiplex falls back to process env.
        from agent.secret_scope import UnscopedSecretError, get_secret
        try:
            webhook_secret = (get_secret("TELEGRAM_WEBHOOK_SECRET") or "").strip()
        except UnscopedSecretError:
            webhook_secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
        if not webhook_secret:
            raise RuntimeError(
                "TELEGRAM_WEBHOOK_SECRET is required when TELEGRAM_WEBHOOK_URL is set. Without it, the "
                "webhook endpoint accepts forged updates from anyone who can reach it — see "
                "https://github.com/NousResearch/hermes-agent/security/advisories/GHSA-3vpc-7q5r-276h.\n\n"
                "Generate a secret and set it in your .env:\n  export TELEGRAM_WEBHOOK_SECRET=\"$(openssl rand -hex 32)\"\n\n"
                "Then register it with Telegram when setting the webhook via setWebhook's secret_token parameter.")
        from urllib.parse import urlparse
        webhook_path = urlparse(webhook_url).path or "/telegram"
        await self._app.updater.start_webhook(
            listen=webhook_host, port=webhook_port, url_path=webhook_path, webhook_url=webhook_url,
            secret_token=webhook_secret, allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=not is_reconnect,  # push-based ⇒ practically a no-op; mirrors polling
       )
        self._webhook_mode = True
        self._polling_progress_accepting = False
        self._send_path_degraded = False
        logger.info(
            "[%s] Webhook server listening on %s:%d%s", self.name, webhook_host or "* (all interfaces, IPv4+IPv6)",
            webhook_port, webhook_path)

    async def _start_polling_mode(self, *, is_reconnect: bool) -> None:
        """Clear any stale webhook and start resilient long polling."""
        # Best-effort: a transient Bot API error must not fail gateway startup — degrade to recovery.
        await self._delete_webhook_best_effort(require_success=not is_reconnect)
        loop = asyncio.get_running_loop()

        def _polling_error_callback(error: Exception) -> None:
            if self._teardown_started or self._recovery_in_flight():
                return
            if self._looks_like_polling_conflict(error):
                # Stop PTB's network_retry_loop synchronously BEFORE scheduling async recovery, else PTB's
                # retry and our stop->restart overlap and produce a fresh 409.
                self._disarm_ptb_retry_loop()
                self._spawn_polling_recovery(loop, self._handle_polling_conflict(error))
            elif self._looks_like_network_error(error):
                logger.warning("[%s] Telegram network _redact_telegram_error_text(error), scheduling reconnect: %s", self.name, error)
                self._spawn_polling_recovery(loop, self._handle_polling_network_error(error))
            else:
                logger.error("[%s] Telegram polling _redact_telegram_error_text(error): %s", self.name, error, exc_info=True)

        self._polling_error_callback_ref = _polling_error_callback  # reused by _handle_polling_conflict
        polling_started = await self._start_polling_resilient(
            # Cold first boot drops the stale Bot API queue; a watcher reconnect preserves it.
            drop_pending_updates=not is_reconnect, error_callback=_polling_error_callback, require_progress=not is_reconnect)
        if not polling_started:
            logger.warning(
                "[%s] Connected in degraded Telegram mode: gateway is alive, polling will be retried in the background", self.name)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect via long polling, or a webhook server if ``TELEGRAM_WEBHOOK_URL`` is set.

        ``is_reconnect``: False = cold boot (drop the stale Bot API queue); True = watcher reconnect (preserve queued
        updates, else every message sent during the outage is lost). Webhook env: TELEGRAM_WEBHOOK_URL,
        TELEGRAM_WEBHOOK_PORT (8443), TELEGRAM_WEBHOOK_HOST, TELEGRAM_WEBHOOK_SECRET."""
        # Explicit connect() is the only operation allowed to reopen polling after a completed teardown.
        self._polling_teardown_started = False
        self._webhook_mode = False  # re-evaluated on every explicit connection
        if not TELEGRAM_AVAILABLE:
            logger.error("[%s] python-telegram-bot not installed. Run: pip install python-telegram-bot", self.name)
            self._set_fatal_error("missing_dependency", "python-telegram-bot not installed", retryable=False)
            return False
        if not self.config.token:
            logger.error("[%s] No bot token configured", self.name)
            self._set_fatal_error("missing_credentials", "No bot token configured", retryable=False)
            return False
        try:
            if not self._acquire_platform_lock('telegram-bot-token', self.config.token, 'Telegram bot token'):
                return False
            builder = Application.builder().token(self.config.token)
            custom_base_url = self.config.extra.get("base_url")
            if custom_base_url:
                builder = builder.base_url(custom_base_url)
                builder = builder.base_file_url(self.config.extra.get("base_file_url", custom_base_url))
                logger.info("[%s] Using custom Telegram base_url: %s", self.name, custom_base_url)
            # Local-mode telegram-bot-api returns absolute server-side file paths; PTB needs local_mode=True
            # so download_*() reads from disk instead of a 404ing HTTP GET.
            if self.config.extra.get("local_mode"):
                builder = builder.local_mode(True)
                logger.info("[%s] Using Telegram local_mode (read files from disk)", self.name)
            request, get_updates_request = await self._build_ptb_requests()
            builder = builder.request(request).get_updates_request(get_updates_request)
            self._app = builder.build()
            self._bot = self._app.bot
            # Plugin PTB handlers go BEFORE core: PTB dispatches the first matching handler per group.
            self._wire_plugin_handlers(self._app)
            self._register_handlers(self._app)
            await self._initialize_app_with_retries(builder)
            await self._app.start()
            webhook_url = os.getenv("TELEGRAM_WEBHOOK_URL", "").strip()
            if webhook_url:
                await self._start_webhook_mode(webhook_url, is_reconnect=is_reconnect)
            else:
                await self._start_polling_mode(is_reconnect=is_reconnect)
            self._mark_connected()
            # WARNING, not INFO: "Connecting…" above is WARNING and reaches the terminal; an INFO success
            # line made healthy startups look stalled at "attempt 1/8".
            logger.warning("[%s] Connected to Telegram (%s mode)", self.name, "webhook" if self._webhook_mode else "polling")
            # Heartbeat only in polling mode: webhook mode has no long-poll socket to wedge in CLOSE-WAIT.
            # WARNING, not INFO: the "Connecting to Telegram (attempt N/8)…" line above is emitted at
            # WARNING and reaches the terminal (the gateway's default stderr handler is WARNING-only), but
            # this success line was INFO and went to the log file only. A healthy startup therefore looked
            # permanently stalled at "attempt 1/8" on the console — the logging illusion in #90835. Both
            # sides of the connect transition must share a terminal-visible level so a real hang is the
            # *absence* of this line, not ambiguity.
            if not self._webhook_mode:
                self._restart_task_attr("_polling_heartbeat_task", self._polling_heartbeat_loop())
            # Seed the live identity from PTB's initialize() cache; polling rides the heartbeat's get_me(),
            # webhook mode gets a low-frequency refresh loop (else a BotFather rename breaks routing).
            self._note_bot_username(getattr(self._bot, "username", None))
            self._bot_identity_checked_at = time.monotonic()
            if self._webhook_mode:
                self._restart_task_attr("_bot_identity_refresh_task", self._bot_identity_refresh_loop())
            # Command menu / DM topics / status indicator can stall for some tokens: defer to a cancellable
            # task so one slow call can't sink the (gateway-timed) connect while transport is live.
            # Command-menu registration, DM-topic setup, and the status indicator each make Bot API calls
            # that can stall for certain tokens. Running them here — inside the connect() coroutine that the
            # gateway wraps in a connect timeout — means one slow call blows the whole connect and the
            # adapter never comes up, even though polling/webhook is already live (#46298).
            self._start_post_connect_housekeeping()
            return True
        except Exception as e:
            self._release_platform_lock()
            safe_error = _redact_telegram_error_text(e)
            # Classify by exception TYPE (never message text): auth failures can never self-heal, so
            # marking them retryable put agents into a silent eternal reconnect loop.
            if self._looks_like_auth_error(e):
                message = (
                    f"Telegram bot token rejected: {safe_error}. "
                    "The token is invalid or was revoked — generate a new one "
                    "with @BotFather and update TELEGRAM_BOT_TOKEN.")
                self._set_fatal_error("telegram_auth_error", message, retryable=False)
            else:
                self._set_fatal_error("telegram_connect_error", f"Telegram startup failed: {safe_error}", retryable=True)
            logger.error("[%s] Failed to connect to Telegram: %s", self.name, safe_error)
            return False

    async def _set_status_indicator(self, online: bool) -> None:
        """Set the bot's short description to the online/offline text (closest Bot API surface to
        presence). No-op unless ``extra.status_indicator``; failures are debug-logged."""
        if not getattr(self, "_status_indicator_enabled", False):
            return
        bot = self._bot
        if bot is None:
            return
        text = (self._status_online_text if online else self._status_offline_text)[:120]  # Telegram cap
        try:
            await bot.set_my_short_description(short_description=text)
            logger.info("[%s] Set bot status indicator to %r", self.name, text)
        except Exception as e:
            logger.debug("[%s] Failed to set bot status indicator to %r: %s", self.name, text, _redact_telegram_error_text(e))

    @staticmethod
    def _collect_live_tasks(candidates, current_task) -> list:
        """Unique, unfinished tasks from ``candidates`` excluding ``current_task`` (so teardown never cancels itself)."""
        seen: set[int] = set()
        out: list[asyncio.Task] = []
        for task in candidates:
            if not task or task.done() or task is current_task or id(task) in seen:
                continue
            seen.add(id(task))
            out.append(task)
        return out

    def _clear_task_attrs_except(self, current_task, *attrs: str) -> None:
        for attr in attrs:
            if getattr(self, attr, None) is not current_task:
                setattr(self, attr, None)

    async def _cancel_pending_delivery_tasks(self) -> None:
        """Cancel every delayed-delivery task family before disconnect completes (media-group, photo-batch, text-batch flushes plus
        polling recovery all sit behind ``asyncio.sleep()`` and would dispatch ``handle_message`` into a torn-down session)."""
        current_task = asyncio.current_task()
        pending_tasks = self._collect_live_tasks(
            [
                *self._media_group_tasks.values(), *self._pending_photo_batch_tasks.values(), *self._pending_text_batch_tasks.values(),
                getattr(self, "_polling_error_task", None), getattr(self, "_polling_progress_verifier_task", None),
                # Hold-queue redispatch must be cancellable+awaitable on teardown too.
                getattr(self, "_held_inbound_redispatch_task", None),
           ],
            current_task)
        awaitable_tasks = [t for t in pending_tasks if asyncio.isfuture(t) or asyncio.iscoroutine(t)]
        # Hold-queue redispatch must be cancellable+awaitable on teardown so it cannot dispatch
        # handle_message into a torn-down session (same lifecycle rule teknium called out on #72037 for
        # shielded flush dispatch).
        for task in pending_tasks:
            task.cancel()
        if awaitable_tasks:
            await asyncio.gather(*awaitable_tasks, return_exceptions=True)
        # Salvage buffered inbound events before clearing maps — unless permanent fatal, where no
        # reconnect can drain and hold would re-orphan them.
        if self._is_permanent_fatal():
            n_pending = len(self._pending_text_batches) + len(self._pending_photo_batches) + len(self._media_group_events)
            if n_pending:
                logger.warning("[Telegram] Non-retryable fatal teardown; discarding %d pending inbound batch(es)", n_pending)
        else:
            for events, where in (
                (self._pending_text_batches, "text-batch-teardown"), (self._pending_photo_batches, "photo-batch-teardown"),
                (self._media_group_events, "media-group-teardown")):
                for event in list(events.values()):
                    self._hold_inbound_event(event, where=where)
        for d in (
            self._media_group_tasks, self._media_group_events, self._pending_photo_batch_tasks,
            self._pending_photo_batches, self._pending_text_batch_tasks, self._pending_text_batches):
            d.clear()
        self._clear_task_attrs_except(
            current_task, "_polling_error_task", "_polling_progress_verifier_task", "_held_inbound_redispatch_task")

    async def _await_disconnect_step(self, awaitable, timeout: float, step: str) -> bool:
        """Await one disconnect step; detach on timeout so teardown advances (``wait_for`` would wait for a
        PTB close that swallows ``CancelledError`` on a half-dead socket). Abandoned tasks are observed.

        ``asyncio.wait_for`` cancels an overdue child but then waits for it to exit. Detach at the deadline
        and continue — the abandoned task is observed via ``_consume_abandoned_task``. See #80598.
        """
        task = asyncio.ensure_future(awaitable)
        try:
            done, _pending = await asyncio.wait({task}, timeout=timeout if timeout > 0 else None)
        except asyncio.CancelledError:
            # asyncio.wait does NOT cancel its futures when itself cancelled; don't orphan the inner task.
            task.cancel()
            # Mirror the pattern used by GatewayRunner._await_adapter_cleanup_with_timeout. See #80598.
            task.add_done_callback(_consume_abandoned_task)
            raise
        if task in done:
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return True
        task.cancel()
        task.add_done_callback(_consume_abandoned_task)
        logger.warning("[%s] %s timed out after %.1fs during disconnect; continuing teardown", self.name, step, timeout)
        return False

    def _restart_task_attr(self, attr: str, coro) -> None:
        """Cancel any live task stored at ``self.<attr>`` and start ``coro`` in its place."""
        prior = getattr(self, attr, None)
        if prior and not prior.done():
            prior.cancel()
        setattr(self, attr, asyncio.ensure_future(coro))

    async def _cancel_task_attr(self, attr: str, label: str) -> None:
        """Cancel + bounded-await the task stored at ``self.<attr>`` (may be missing: object.__new__ tests), then clear it."""
        task = getattr(self, attr, None)
        if task and not task.done():
            task.cancel()
            await self._await_disconnect_step(task, _DISCONNECT_STEP_TIMEOUT, label)
        setattr(self, attr, None)

    async def disconnect(self) -> None:
        """Stop polling/webhook, cancel pending delayed deliveries, and disconnect."""
        # Mark disconnected first so the drop guard short-circuits any flush that wins the race.
        self._mark_disconnected()
        self._polling_teardown_started = True
        self._polling_progress_accepting = False
        self._polling_generation = getattr(self, "_polling_generation", 0) + 1
        self._polling_progress_event = asyncio.Event()
        self._send_path_degraded = True
        # Release the bot-token lock immediately so a wedged close cannot block the reconnect watcher.
        # The rest of teardown is best-effort against a half-dead transport. See #80598.
        self._release_platform_lock()
        # Cancel and await both polling lifecycle owners right after the fence, before any other teardown
        # await lets them start a new generation.
        current_task = asyncio.current_task()
        lifecycle_tasks = self._collect_live_tasks(
            [getattr(self, "_polling_error_task", None), getattr(self, "_polling_progress_verifier_task", None)], current_task)
        for task in lifecycle_tasks:
            task.cancel()
        lifecycle_tasks = [t for t in lifecycle_tasks if asyncio.isfuture(t) or asyncio.iscoroutine(t)]
        if lifecycle_tasks:
            await self._await_disconnect_step(
                asyncio.gather(*lifecycle_tasks, return_exceptions=True), _DISCONNECT_STEP_TIMEOUT, "lifecycle-task cancel")
        self._clear_task_attrs_except(current_task, "_polling_error_task", "_polling_progress_verifier_task")
        # Cancellation callbacks may have run while awaited; the fence stays authoritative.
        self._polling_progress_accepting = False
        self._send_path_degraded = True
        # Cancel deferred post-connect housekeeping so it cannot fire into a half-torn-down bot client.
        # Cancel deferred post-connect housekeeping (command-menu / DM-topic / status-indicator Bot API
        # calls) so it cannot fire into a half-torn-down bot client (#46298). getattr guards the
        # object.__new__ test pattern where __init__ (which sets this attr) is never called.
        post_connect_task = getattr(self, "_post_connect_task", None)
        if post_connect_task and not post_connect_task.done():
            post_connect_task.cancel()
            await self._await_disconnect_step(
                asyncio.gather(post_connect_task, return_exceptions=True), _DISCONNECT_STEP_TIMEOUT, "post-connect cancel")
        self._post_connect_task = None
        # Cancel the heartbeat (and webhook-mode identity loop) before tearing down the app.
        await self._cancel_task_attr("_polling_heartbeat_task", "heartbeat cancel")
        await self._cancel_task_attr("_bot_identity_refresh_task", "identity-refresh cancel")
        # Mark the bot "Offline" while its HTTP client is still alive. Opt-in, non-fatal.
        with contextlib.suppress(Exception):
            await self._await_disconnect_step(self._set_status_indicator(online=False), _DISCONNECT_STEP_TIMEOUT, "status-indicator update")
        await self._await_disconnect_step(self._cancel_pending_delivery_tasks(), _DISCONNECT_STEP_TIMEOUT, "pending-delivery cancel")
        if self._app:
            try:
                # Bounded: a CLOSE-WAIT socket can wedge updater.stop() forever; fall through on timeout.
                if self._app.updater and self._app.updater.running:
                    try:
                        await self._await_disconnect_step(self._app.updater.stop(), _UPDATER_STOP_TIMEOUT, "updater.stop()")
                    except Exception as stop_error:
                        logger.warning(
                            "[%s] updater.stop() failed during disconnect: %s", self.name, _redact_telegram_error_text(stop_error))
                # app.stop()/shutdown() can also block on a half-dead httpx pool.
                # Detach-on-timeout so disconnect always returns (#80598).
                if self._app.running:
                    await self._await_disconnect_step(self._app.stop(), _DISCONNECT_STEP_TIMEOUT, "app.stop()")
                await self._await_disconnect_step(self._app.shutdown(), _DISCONNECT_STEP_TIMEOUT, "app.shutdown()")
            except Exception as e:
                logger.warning("[%s] Error during Telegram disconnect: %s", self.name, _redact_telegram_error_text(e))
        self._app = None
        self._bot = None
        logger.info("[%s] Disconnected from Telegram", self.name)

    def _should_thread_reply(self, reply_to: Optional[str], chunk_index: int) -> bool:
        """Whether this chunk (0 = first) should reply-thread to ``reply_to``, per reply_to_mode."""
        if not reply_to:
            return False
        mode = self._reply_to_mode
        if mode == "off":
            return False
        if mode == "all":
            return True
        return chunk_index == 0  # "first" (default)

    @staticmethod
    def _telegram_error_types() -> tuple:
        """``(NetworkError, BadRequest, TimedOut)`` from PTB, with import-failure fallbacks
        (``OSError``, ``None``, ``None``) so send() still classifies without the SDK."""
        try:
            from telegram.error import NetworkError as _NetErr
        except ImportError:
            _NetErr = OSError  # type: ignore[misc,assignment]
        try:
            from telegram.error import BadRequest as _BadReq
        except ImportError:
            _BadReq = None  # type: ignore[assignment,misc]
        try:
            from telegram.error import TimedOut as _TimedOut
        except (ImportError, AttributeError):
            _TimedOut = None  # type: ignore[assignment,misc]
        return _NetErr, _BadReq, _TimedOut

    async def _send_chunk_markdown_or_plain(self, chunk: str, send_kwargs: Dict[str, Any]):
        """MarkdownV2 first; on a parse/markdown rejection resend as stripped plain text."""
        try:
            return await self._bot.send_message(text=chunk, parse_mode=ParseMode.MARKDOWN_V2, **send_kwargs)
        except Exception as md_error:
            if "parse" in str(md_error).lower() or "markdown" in str(md_error).lower():
                logger.warning("[%s] MarkdownV2 parse failed, falling back to plain text: %s", self.name, md_error)
                return await self._bot.send_message(text=_strip_mdv2(chunk), parse_mode=None, **send_kwargs)
            raise

    async def _send_chunk_with_retries(
        self, chat_id: str, chunk: str, index: int, reply_to: Optional[str], metadata: Optional[Dict[str, Any]],
        thread_id: Optional[str], used_thread_fallback: bool, error_types: tuple):
        """Deliver one chunk: routing, up to 3 attempts, thread-not-found / deleted-anchor / flood handling.

        Returns ``(msg, used_thread_fallback)`` on success or a ``SendResult`` to return verbatim (fail-loud DM-topic
        cases, flood cap); raises anything the caller's classifier should see."""
        _NetErr, _BadReq, _TimedOut = error_types
        retried_thread_not_found = False
        private_dm_topic_send, dm_topic_reply_to_off, reply_to_id = self._chunk_reply_routing(chat_id, reply_to, metadata, thread_id, index)
        if private_dm_topic_send and reply_to_id is None and not dm_topic_reply_to_off:
            return SendResult(success=False, error=self._dm_topic_missing_anchor_error(), retryable=False)
        thread_kwargs = self._thread_kwargs_for_send(
            chat_id, thread_id, metadata, reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode)
        if used_thread_fallback and thread_kwargs.get("message_thread_id") is not None:
            thread_kwargs = dict(thread_kwargs)
            thread_kwargs["message_thread_id"] = None
        effective_thread_id = thread_kwargs.get("message_thread_id")
        for _send_attempt in range(3):
            try:
                send_kwargs = {
                    "chat_id": normalize_telegram_chat_id(chat_id), "reply_to_message_id": reply_to_id, **thread_kwargs,
                    **self._link_preview_kwargs(), **self._notification_kwargs(metadata)}
                return await self._send_chunk_markdown_or_plain(chunk, send_kwargs), used_thread_fallback
            except _NetErr as send_err:
                # BadRequest subclasses NetworkError in PTB but is permanent; handle specific cases.
                if _BadReq and isinstance(send_err, _BadReq):
                    if self._is_thread_not_found_error(send_err) and effective_thread_id is not None:
                        if private_dm_topic_send or (metadata and metadata.get("telegram_dm_topic_created_for_send")):
                            return SendResult(success=False, error=str(send_err), retryable=False)
                        # One-off "thread not found" flakes recover on immediate retry: same thread_id once.
                        if not retried_thread_not_found:
                            retried_thread_not_found = True
                            logger.warning("[%s] Thread %s not found, retrying once with same thread_id", self.name, effective_thread_id)
                            continue
                        # Thread is genuinely gone: retry without it and prune the stale binding.
                        logger.warning("[%s] Thread %s not found, retrying without message_thread_id", self.name, effective_thread_id)
                        self._prune_stale_dm_topic_binding(chat_id, effective_thread_id, metadata=metadata)
                        used_thread_fallback = True
                        effective_thread_id = None
                        thread_kwargs = {"message_thread_id": None}
                        continue
                    if "message to be replied not found" in str(send_err).lower() and reply_to_id is not None:
                        safe_send_error = _redact_telegram_error_text(send_err)
                        if private_dm_topic_send:
                            return SendResult(success=False, error=safe_send_error, retryable=False)
                        # Reply target deleted; private-topic fallback sends drop anchor + topic id together.
                        logger.warning("[%s] Reply target deleted, retrying without reply_to: %s", self.name, safe_send_error)
                        reply_to_id = None
                        if self._dm_topic_fallback(metadata):
                            thread_kwargs = {}
                        else:
                            thread_kwargs = self._thread_kwargs_for_send(
                                chat_id, thread_id, metadata, reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode)
                        effective_thread_id = thread_kwargs.get("message_thread_id")
                        continue
                    raise  # other BadRequest errors are permanent
                # TimedOut also subclasses NetworkError: a generic timeout may have reached Telegram (don't
                # retry); a wrapped ConnectTimeout or an httpx pool timeout is safe to retry.
                is_pool_timeout = self._looks_like_pool_timeout(send_err)
                if (
                    _TimedOut and isinstance(send_err, _TimedOut)
                    and not self._looks_like_connect_timeout(send_err) and not is_pool_timeout):
                    raise
                if is_pool_timeout:
                    await self._drain_general_connections_after_pool_timeout()
                if _send_attempt >= 2:
                    raise
                wait = 2 ** _send_attempt
                logger.warning("[%s] Network error on send (attempt %d/3), retrying in %ds: %s",
                               self.name, _send_attempt + 1, wait, _redact_telegram_error_text(send_err))
                await asyncio.sleep(wait)
            except Exception as send_err:
                retry_after = getattr(send_err, "retry_after", None)
                if retry_after is not None or "retry after" in str(send_err).lower():
                    wait = float(retry_after) if retry_after is not None else 1.0
                    safe_send_error = _redact_telegram_error_text(send_err)
                    # Never sleep a long server RetryAfter verbatim — it once pinned send() for 97 minutes.
                    # Mirror the edit path: a RetryAfter past a few seconds is not something to hold this
                    # coroutine open for. Sleeping the server value verbatim pinned send() for 97 minutes in
                    # production and froze inbound on every platform when it ran on the gateway boot path
                    # (#91969).
                    if wait > _FLOOD_INLINE_WAIT_CAP_SECS:
                        logger.warning(
                            "[%s] Telegram flood control on send (retry_after=%.1fs > %.0fs); failing closed instead of sleeping: %s",
                            self.name, wait, _FLOOD_INLINE_WAIT_CAP_SECS, safe_send_error)
                        return _flood_cap_result(wait)
                    if _send_attempt < 2:
                        logger.warning(
                            "[%s] Telegram flood control on send (attempt %d/3), retrying in %.1fs: %s", self.name,
                            _send_attempt + 1, wait, safe_send_error)
                        await asyncio.sleep(wait)
                        continue
                raise

    async def _retrigger_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> None:
        """Re-arm typing after an intermediate send (Telegram clears it when a message lands). Skipped on
        the FINAL reply (``metadata["notify"]``): the refresh loop is gone and no API cancels the bubble."""
        if (metadata or {}).get("notify"):
            return
        with contextlib.suppress(Exception):
            await self.send_typing(chat_id, metadata=metadata)

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a message to a Telegram chat."""
        if not self._bot:
            live = self._replacement_telegram_adapter()
            if live is not None:
                return await live.send(chat_id, content, reply_to, metadata)
            if self._is_permanent_fatal() or not await self._wait_for_reconnection():
                return SendResult(success=False, error="Not connected", retryable=not self._is_permanent_fatal())
            live = self._replacement_telegram_adapter()
            if not self._bot and live is not None:
                return await live.send(chat_id, content, reply_to, metadata)
            if not self._bot:
                return SendResult(success=False, error="Not connected", retryable=True)
        # getattr() — tests build adapters via object.__new__() (no __init__).
        if getattr(self, "_send_path_degraded", False):
            return SendResult(success=False, error="send_path_degraded", retryable=True)
        # Skip whitespace-only text to prevent Telegram 400 empty-text errors.
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)
        error_types = self._telegram_error_types()
        try:
            # Bot API 10.1 rich fast-path; falls through to legacy MarkdownV2 on permanent/capability
            # errors or DM-topic skips; returns directly on success or transient failure (no legacy resend).
            if self._should_attempt_rich(content, metadata=metadata):
                rich_result = await self._try_send_rich(chat_id, content, reply_to, metadata)
                if rich_result is not None:
                    if rich_result.success:
                        await self._retrigger_typing(chat_id, metadata)
                    return rich_result
            chunks = self.truncate_message(self.format_message(content), self.MAX_MESSAGE_LENGTH, len_fn=utf16_len)
            if len(chunks) > 1:
                # truncate_message appends a raw " (1/2)" suffix; escape the MarkdownV2-special parentheses.
                chunks = [
                    _separate_chunk_indicator_from_fence(re.sub(r" \((\d+)/(\d+)\)$", r" \\(\1/\2\\)", chunk))
                    for chunk in chunks
               ]
            message_ids = []
            thread_id = self._metadata_thread_id(metadata)
            requested_thread_id = self._message_thread_id_for_send(thread_id)
            used_thread_fallback = False
            for i, chunk in enumerate(chunks):
                outcome = await self._send_chunk_with_retries(
                    chat_id, chunk, i, reply_to, metadata, thread_id, used_thread_fallback, error_types)
                if isinstance(outcome, SendResult):
                    return outcome
                msg, used_thread_fallback = outcome
                message_ids.append(str(msg.message_id))
            await self._retrigger_typing(chat_id, metadata)
            return SendResult(
                success=True, message_id=message_ids[0] if message_ids else None,
                raw_response={
                    "message_ids": message_ids, "requested_thread_id": requested_thread_id, "thread_fallback": used_thread_fallback})
        except Exception as e:
            safe_error = _redact_telegram_error_text(e)
            logger.error("[%s] Failed to send Telegram message: %s", self.name, safe_error)
            err_str = str(e).lower()
            error_kind = classify_send_error(e)
            # Content exceeded 4096 chars: fail so the stream consumer enters fallback mode.
            if "message_too_long" in err_str or "too long" in err_str:
                logger.debug("[%s] send() content too long, falling back to new-message continuation", self.name)
                return SendResult(success=False, error="message_too_long", error_kind="too_long")
            # TimedOut may have reached Telegram — non-retryable so _send_with_retry() doesn't re-send,
            # except a wrapped ConnectTimeout or an httpx pool timeout (safe to re-send).
            _to = error_types[2]
            is_timeout = (_to and isinstance(e, _to)) or "timed out" in err_str
            return SendResult(
                success=False, error=safe_error,
                retryable=(self._looks_like_connect_timeout(e) or self._looks_like_pool_timeout(e) or not is_timeout),
                error_kind=error_kind)

    async def send_or_update_status(
        self, chat_id: str, status_key: str, content: str, *, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a status message, or edit the previous one with the same ``(chat_id, status_key)``; if the
        edit fails (deleted, too old, …) the cached id is dropped and a fresh message is sent.

        Issue #30045: progress/status callbacks (context-pressure, lifecycle, compression, etc.) used to
        append a fresh bubble on every call. With this method, the first call sends and the message id is
        remembered; subsequent calls with the same (chat_id, status_key) edit that same message in place.
        """
        key = (str(chat_id), str(status_key))
        cached_id = self._status_message_ids.get(key)
        if cached_id is not None:
            result = await self.edit_message(chat_id, cached_id, content, finalize=True, metadata=metadata)
            if result.success:
                if result.message_id:
                    self._status_message_ids[key] = str(result.message_id)
                return result
            self._status_message_ids.pop(key, None)
        result = await self.send(chat_id, content, metadata=metadata)
        if result.success and result.message_id:
            self._status_message_ids[key] = str(result.message_id)
        return result

    async def _edit_text(self, chat_id: str, message_id: str, text: str, parse_mode: Any = None) -> None:
        """``editMessageText`` with normalized ids; ``parse_mode=None`` sends plain text."""
        kwargs: Dict[str, Any] = {"chat_id": normalize_telegram_chat_id(chat_id), "message_id": int(message_id), "text": text}
        if parse_mode is not None:
            kwargs["parse_mode"] = parse_mode
        await self._bot.edit_message_text(**kwargs)

    async def _edit_markdown_or_plain(self, chat_id: str, message_id: str, formatted: str, plain: str, warn_fmt: str) -> bool:
        """MarkdownV2 edit with plain-text fallback. Returns True on a "not modified" no-op (caller may
        skip further work); the fallback edit's exceptions propagate."""
        try:
            await self._edit_text(chat_id, message_id, formatted, ParseMode.MARKDOWN_V2)
        except Exception as fmt_err:
            if "not modified" in str(fmt_err).lower():
                return True
            logger.warning(warn_fmt, self.name, _redact_telegram_error_text(fmt_err))
            await self._edit_text(chat_id, message_id, plain)
        return False

    async def edit_message(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool = False, metadata: Optional[Dict[str, Any]] = None,
   ) -> SendResult:
        """Edit a previously sent Telegram message.

        Telegram caps a message at 4096 UTF-16 codeunits. Streaming replies that outgrow it must NOT be truncated
        silently nor fail (the consumer would re-send a duplicate): edit with the first chunk, send the rest as
        continuations, and return the final chunk's id as the next edit target."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        # Rich finalize (Bot API 10.1): edit the preview IN PLACE via rich_message — no fresh send + delete.
        # Before the 4,096 pre-flight because the rich cap is 32,768; falls back to legacy on rejection.
        # Rich finalize (Bot API 10.1): when the completed content has constructs the legacy MarkdownV2 edit
        # degrades (tables → bullet lists, task lists, <details>, block math) and rich is available, edit
        # the preview IN PLACE via editMessageText's rich_message param. No fresh send + delete → no
        # duplicate preview (the problem #46206 reverted the fresh-final path for). Attempted before the
        # 4,096 overflow pre-flight because the rich text cap is 32,768 — a rich table that exceeds the
        # MarkdownV2 limit must not be split into legacy chunks. Falls back to the legacy edit path
        # (overflow split included) on capability/permanent rejection.
        if finalize and self._rich_eligible(content):
            rich_result = await self._try_edit_rich(chat_id, message_id, content, metadata=metadata)
            if rich_result is not None:
                return rich_result
        # Pre-flight: over-limit content is split-and-delivered on finalize; mid-stream we truncate instead
        # (splitting moves the edit target to a continuation → infinite duplication loop).
        # Pre-flight: if content already exceeds the limit, split-and-deliver without round-tripping a
        # doomed edit. During streaming (finalize=False) we truncate instead of splitting — splitting
        # creates continuation messages whose IDs become the new edit target, and on the next token chunk
        # the full accumulated text is re-edited into the continuation, triggering another split → infinite
        # duplication loop (#48648).
        _preview_key = (str(chat_id), str(message_id))
        _saturated_preview = False
        if finalize:
            self._last_overflow_preview.pop(_preview_key, None)  # the final edit always delivers full content
        if utf16_len(content) > self.MAX_MESSAGE_LENGTH:
            if finalize:
                return await self._edit_overflow_split(chat_id, message_id, content, finalize=finalize, metadata=metadata)
            content = self._truncate_stream_overflow_preview(content)
            _saturated_preview = True
            # Saturated-preview dedup: past the cap every progressive edit truncates to the same text;
            # re-sending is a visual no-op that still burns flood budget (200s+ penalties).
            if self._last_overflow_preview.get(_preview_key) == content:
                return SendResult(success=True, message_id=message_id)
        elif not finalize:
            # Content shrank back under the cap — clear stale saturation state so dedup can't mask an edit.
            self._last_overflow_preview.pop(_preview_key, None)
        try:
            if not finalize:
                await self._edit_text(chat_id, message_id, content)
                if _saturated_preview:
                    self._last_overflow_preview[_preview_key] = content
                return SendResult(success=True, message_id=message_id)
            await self._edit_markdown_or_plain(
                chat_id, message_id, self.format_message(content), _strip_mdv2(content) if content else content,
                "[%s] MarkdownV2 edit failed, falling back to plain text: %s")
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            err_str = str(e).lower()
            if "not modified" in err_str:
                return SendResult(success=True, message_id=message_id)
            # Reactive split: MarkdownV2 escapes can inflate the payload past the limit even when raw text fit.
            if "message_too_long" in err_str or "too long" in err_str:
                logger.debug(
                    "[%s] edit_message overflow (%d UTF-16 > %d), splitting", self.name, utf16_len(content), self.MAX_MESSAGE_LENGTH)
                if finalize:
                    return await self._edit_overflow_split(chat_id, message_id, content, finalize=finalize, metadata=metadata)
                # Mid-stream: truncate and retry instead of splitting (saturated-preview dedup as above).
                # See #48648.
                truncated = self._truncate_stream_overflow_preview(content)
                if self._last_overflow_preview.get(_preview_key) == truncated:
                    return SendResult(success=True, message_id=message_id)
                await self._edit_text(chat_id, message_id, truncated)
                self._last_overflow_preview[_preview_key] = truncated
                return SendResult(success=True, message_id=message_id)
            # Flood control: short waits retry inline; long waits fail immediately so streaming falls back
            # to a normal final send instead of a clipped partial.
            retry_after = getattr(e, "retry_after", None)
            if retry_after is not None or "retry after" in err_str:
                wait = retry_after if retry_after else 1.0
                logger.warning("[%s] Telegram flood control, waiting %.1fs", self.name, wait)
                if wait > _FLOOD_INLINE_WAIT_CAP_SECS:
                    return _flood_cap_result(wait)
                await asyncio.sleep(wait)
                try:
                    await self._edit_text(chat_id, message_id, content)
                    return SendResult(success=True, message_id=message_id)
                except Exception as retry_err:
                    safe_retry_error = _redact_telegram_error_text(retry_err)
                    logger.error("[%s] Edit retry failed after flood wait: %s", self.name, safe_retry_error)
                    return SendResult(success=False, error=safe_retry_error)
            safe_error = _redact_telegram_error_text(e)
            # Transient network errors must not permanently disable progress-message editing.
            _transient_markers = (
                "connecterror", "connect error", "connection error", "networkerror", "network error", "timed out", "readtimeout",
                "writetimeout", "server disconnected", "temporarily unavailable", "temporary failure", "httpx")
            if any(m in err_str for m in _transient_markers):
                logger.warning("[%s] Transient network error editing message %s (will retry): %s", self.name, message_id, safe_error)
                return SendResult(success=False, error=safe_error, retryable=True)
            logger.error("[%s] Failed to edit Telegram message %s: %s", self.name, message_id, safe_error)
            return SendResult(success=False, error=safe_error)

    def _truncate_stream_overflow_preview(self, content: str) -> str:
        """One-message preview for oversized streaming edits (edits must keep targeting the original id;
        final edits use ``_edit_overflow_split``).

        Splitting a mid-stream preview creates continuation messages and moves the active message id, so the
        next accumulated-token edit repeats the overflow cycle (#48648). Final edits still use
        ``_edit_overflow_split`` to deliver the complete response.
        """
        return self.truncate_message(content, self.MAX_MESSAGE_LENGTH, len_fn=utf16_len)[0]

    async def _send_overflow_continuation(
        self, chat_id: str, chunk: str, reply_to_id: Optional[int], thread_kwargs: Dict[str, Any],
        thread_id: Optional[str], metadata: Optional[Dict[str, Any]], finalize: bool):
        """Send one continuation chunk (MarkdownV2 then plain on finalize; raw when streaming); drops the
        reply anchor once on 'reply message not found'. Returns the sent message or None."""
        base = {**self._link_preview_kwargs(), **self._notification_kwargs(metadata)}
        for use_markdown in (True, False) if finalize else (False,):
            try:
                if use_markdown:
                    text = _separate_chunk_indicator_from_fence(self.format_message(chunk))
                else:
                    # Degrade to stripped text on finalize (raw ** / ``` would render literally); previews stay raw.
                    text = _strip_mdv2(chunk) if finalize else chunk
                return await self._bot.send_message(
                    chat_id=normalize_telegram_chat_id(chat_id), text=text, parse_mode=ParseMode.MARKDOWN_V2 if use_markdown else None,
                    reply_to_message_id=reply_to_id, **thread_kwargs, **base)
            except Exception as send_err:
                if "reply message not found" in str(send_err).lower():
                    # Private DM topic fallback needs anchor + topic id together; forum topics keep thread id.
                    retry_thread_kwargs = (
                        {} if self._dm_topic_fallback(metadata)
                        else self._thread_kwargs_for_send(chat_id, thread_id, metadata, reply_to_message_id=None))
                    try:
                        return await self._bot.send_message(
                            chat_id=normalize_telegram_chat_id(chat_id), text=_strip_mdv2(chunk) if finalize else chunk,
                            **retry_thread_kwargs, **base)
                    except Exception as _retry_err:
                        logger.warning(
                            "[%s] Overflow continuation no-reply retry failed: %s", self.name, _redact_telegram_error_text(_retry_err))
                        return None
                if use_markdown:
                    continue  # try plain text on next loop iteration
                logger.warning("[%s] Overflow continuation send failed: %s", self.name, _redact_telegram_error_text(send_err))
                return None
        return None

    async def _edit_overflow_split(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Split an oversized edit across the existing message + continuations: edit ``message_id`` with
        chunk 1, send the rest as replies to the previous chunk, return ``message_id=<last-chunk-id>`` so
        the consumer keeps editing the newest message. ``success=False`` only if the first-chunk edit fails."""
        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH, len_fn=utf16_len)
        if len(chunks) <= 1:
            chunks = [content]  # defensive: a single chunk just edits normally
        first_chunk = chunks[0]
        try:
            if finalize:
                await self._edit_markdown_or_plain(
                    chat_id, message_id, _separate_chunk_indicator_from_fence(self.format_message(first_chunk)), _strip_mdv2(first_chunk),
                    "[%s] Overflow split: MarkdownV2 first-chunk edit failed, falling back to plain text: %s")
            else:
                await self._edit_text(chat_id, message_id, first_chunk)
        except Exception as e:
            if "not modified" not in str(e).lower():  # identical first chunk still sends continuations
                logger.error("[%s] Overflow split: first-chunk edit failed: %s", self.name, _redact_telegram_error_text(e), exc_info=True)
                return SendResult(success=False, error=_redact_telegram_error_text(e))
        # Continuations call self._bot.send_message directly to skip self.send's pre-chunking.
        continuation_ids: list[str] = []
        delivered_chunks = [first_chunk]
        prev_id = message_id
        thread_id = self._metadata_thread_id(metadata)
        for chunk in chunks[1:]:
            reply_to_id = int(prev_id) if prev_id else None
            thread_kwargs = self._thread_kwargs_for_send(chat_id, thread_id, metadata, reply_to_message_id=reply_to_id)
            sent_msg = await self._send_overflow_continuation(chat_id, chunk, reply_to_id, thread_kwargs, thread_id, metadata, finalize)
            if sent_msg is None:
                # Partial delivery: do NOT report success — the consumer would treat it as final delivery.
                logger.warning("[%s] Overflow split: stopped at %d/%d chunks delivered", self.name, 1 + len(continuation_ids), len(chunks))
                delivered_prefix = "".join(re.sub(r" \(\d+/\d+\)$", "", delivered) for delivered in delivered_chunks)
                return SendResult(
                    success=False, message_id=prev_id, error="overflow_continuation_failed", retryable=True,
                    raw_response={
                        "partial_overflow": True, "delivered_chunks": 1 + len(continuation_ids),
                        "total_chunks": len(chunks), "last_message_id": prev_id, "delivered_prefix": delivered_prefix,
                        "continuation_message_ids": tuple(continuation_ids)},
                    continuation_message_ids=tuple(continuation_ids))
            new_id = str(getattr(sent_msg, "message_id", "")) or prev_id
            continuation_ids.append(new_id)
            delivered_chunks.append(chunk)
            prev_id = new_id
        last_id = continuation_ids[-1] if continuation_ids else message_id
        logger.debug("[%s] Overflow split delivered %d chunks; last_id=%s", self.name, 1 + len(continuation_ids), last_id)
        return SendResult(success=True, message_id=last_id, continuation_message_ids=tuple(continuation_ids))

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a bot-posted message (Bot API allows it within 48h); failures are non-fatal.

        Used by the stream consumer's fresh-final cleanup path (ported from openclaw/openclaw#72038) to
        remove long-lived preview messages after sending the completed reply as a fresh message. Telegram's
        Bot API ``deleteMessage`` works for bot-posted messages in the last 48 hours. Failures are non-fatal
        — the caller leaves the preview in place and logs at debug level.
        """
        if not self._bot:
            return False
        try:
            await self._bot.delete_message(chat_id=normalize_telegram_chat_id(chat_id), message_id=int(message_id))
            return True
        except Exception as e:
            logger.debug("[%s] Failed to delete Telegram message %s: %s", self.name, message_id, _redact_telegram_error_text(e))
            return False

    def supports_draft_streaming(self, chat_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """sendMessageDraft works for private chats only (Bot API 9.5) and needs PTB >= 22.6; groups and
        older installs use the edit-based path. ``rich_drafts`` controls draft *format*, not availability."""
        if not self._bot or not hasattr(self._bot, "send_message_draft"):
            return False
        return (chat_type or "").lower() in {"dm", "private"}

    async def send_draft(self, chat_id: str, draft_id: int, content: str, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Stream a partial message via ``sendRichMessageDraft`` (when rich is enabled and supported) else
        ``sendMessageDraft``; reusing ``draft_id`` animates the preview. The caller sends the final text."""
        if not self._bot:
            return SendResult(success=False, error="not_connected")
        # Rich draft fast-path; any failure degrades to the plain draft below. Drafts have no message_id.
        if self._should_attempt_rich_draft(content) and await self._try_send_rich_draft(chat_id, draft_id, content, metadata):
            return SendResult(success=True, message_id=None)
        if not hasattr(self._bot, "send_message_draft"):
            return SendResult(success=False, error="api_unavailable")
        # Drafts share the regular-send UTF-16 length contract.
        text = content if len(
            content) <= self.MAX_MESSAGE_LENGTH else self.truncate_message(content, self.MAX_MESSAGE_LENGTH, len_fn=utf16_len)[0]
        # Same MarkdownV2 conversion as ``send`` (MarkdownV2 then plain) so the draft doesn't snap at the end. Exception: a Rich
        # final with rich drafts disabled previews raw — the legacy formatter would turn pipe tables into bullets.
        plain_rich_preview = bool(
            getattr(self, "_rich_messages_enabled", False) and not getattr(self, "_rich_drafts_enabled", False)
            and self._needs_rich_rendering(text))
        draft_thread_kwargs = self._thread_kwargs_for_draft(chat_id, metadata)
        for use_markdown in ((False,) if plain_rich_preview else (True, False)):
            kwargs: Dict[str, Any] = {
                "chat_id": normalize_telegram_chat_id(chat_id), "draft_id": int(draft_id),
                "text": self.format_message(text) if use_markdown else text}
            if use_markdown:
                kwargs["parse_mode"] = ParseMode.MARKDOWN_V2
            kwargs.update(draft_thread_kwargs)
            try:
                if await self._bot.send_message_draft(**kwargs):
                    return SendResult(success=True, message_id=None)
                return SendResult(success=False, error="draft_rejected")
            except Exception as e:
                # MarkdownV2 parse failure → retry once as plain text; anything else returns to the caller,
                # which falls back to edit-based streaming for this response.
                if use_markdown and self._is_bad_request_error(e):
                    logger.debug(
                        "[%s] sendMessageDraft MarkdownV2 rejected, retrying as plain text (chat=%s draft_id=%s): %s",
                        self.name, chat_id, draft_id, _redact_telegram_error_text(e))
                    continue
                logger.debug("[%s] sendMessageDraft failed (chat=%s draft_id=%s): %s", self.name, chat_id, draft_id, e)
                return SendResult(success=False, error=_redact_telegram_error_text(e))
        return SendResult(success=False, error="draft_rejected")

    async def _send_message_with_thread_fallback(self, **kwargs):
        """Send a control-style message (approval prompts, pickers), retrying once without
        message_thread_id on 'Message thread not found' (stale thread_id); ``send`` has its own.

        Used for control-style sends (approval prompts, model picker, update prompts) that can carry a stale
        thread_id from a DM reply chain. The streaming send loop has its own equivalent (PR #3390) at the
        body of ``send``; this helper applies the same retry pattern to the non-streaming control paths.
        """
        if not self._bot:
            raise RuntimeError("Not connected")
        message_thread_id = kwargs.get("message_thread_id")
        try:
            return await self._bot.send_message(**kwargs)
        except Exception as send_err:
            if (message_thread_id is not None and self._is_bad_request_error(send_err) and self._is_thread_not_found_error(send_err)):
                logger.warning(
                    "[%s] Thread %s not found for control message, retrying without message_thread_id", self.name, message_thread_id)
                # Same prune as the streaming send path; control sends carry no gateway metadata.
                self._prune_stale_dm_topic_binding(kwargs.get("chat_id"), message_thread_id)
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("message_thread_id", None)
                return await self._bot.send_message(**retry_kwargs)
            raise

    async def _send_control_message(
        self, chat_id: str, text: str, *, parse_mode: Any, thread_id: Optional[str], metadata: Optional[Dict[str, Any]],
        reply_markup: Any = None, reply_to_mode: Optional[str] = None):
        """Send a control-style message (prompt/picker) with topic routing + thread fallback."""
        reply_to_id = self._reply_to_message_id_for_send(None, metadata, reply_to_mode=reply_to_mode)
        kwargs: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id), "text": text, "parse_mode": parse_mode, **self._link_preview_kwargs()}
        if reply_markup is not None:
            kwargs["reply_markup"] = reply_markup
        kwargs["reply_to_message_id"] = reply_to_id
        kwargs.update(self._thread_kwargs_for_send(
            chat_id, thread_id, metadata, reply_to_message_id=reply_to_id, reply_to_mode=reply_to_mode))
        return await self._send_message_with_thread_fallback(**kwargs)

    async def _send_prompt(self, what: str, chat_id: str, metadata: Optional[Dict[str, Any]], build, *,
                           parse_mode: Any = None, thread_id: Any = None, reply_to_mode: Any = None) -> SendResult:
        """Shared control-prompt shell: not-connected guard, ``build()`` → ``(text, keyboard, on_sent)`` (or a
        SendResult to return as-is), routed send, state hook, redacted failure log."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            built = build()
            if isinstance(built, SendResult):
                return built
            text, keyboard, on_sent = built
            msg = await self._send_control_message(
                chat_id, text, parse_mode=parse_mode if parse_mode is not None else ParseMode.MARKDOWN_V2,
                reply_markup=keyboard, thread_id=thread_id, metadata=metadata, reply_to_mode=reply_to_mode)
            if on_sent is not None:
                on_sent(msg)
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] %s failed: %s", self.name, what, _redact_telegram_error_text(e))
            return SendResult(success=False, error=_redact_telegram_error_text(e))

    @staticmethod
    def _rows_of_two(buttons: list) -> list:
        """2-per-row layout keeps labels readable on mobile (a 4-button row truncates)."""
        return [buttons[i:i + 2] for i in range(0, len(buttons), 2)]

    async def send_update_prompt(
        self, chat_id: str, prompt: str, default: str = "", session_key: str = "", metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an inline-keyboard Yes/No prompt for the gateway ``/update`` watcher."""
        def build():
            default_hint = f" (default: {default})" if default else ""
            text = self.format_message(f"⚕ *Update needs your input:*\n\n{prompt}{default_hint}")
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("✓ Yes", callback_data="update_prompt:y"),
                InlineKeyboardButton("✗ No", callback_data="update_prompt:n")]])
            return text, keyboard, None
        return await self._send_prompt(
            "send_update_prompt", chat_id, metadata, build, thread_id=self._metadata_thread_id(metadata), reply_to_mode=self._reply_to_mode)

    # Template attrs for the shared _format_exec_approval core (HTML mode).
    _EA_HEADER = "⚠️ <b>Command Approval Required</b>\n\n"
    _EA_CODE_OPEN = "<pre>"
    _EA_CODE_CLOSE = "</pre>\n\n"
    _EA_SMART_DENY_LINE = "\n\n<b>Smart DENY:</b> owner override applies to this one operation only."
    _EA_CMD_BUDGET = 3800

    def _ea_escape(self, text: str) -> str:
        return _html.escape(text)

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str, description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None, allow_permanent: bool = True, allow_session: bool = True,
        smart_denied: bool = False) -> SendResult:
        """Send an inline-keyboard approval prompt; buttons call ``resolve_gateway_approval()`` like the
        text ``/approve`` flow."""
        def build():
            text = self._format_exec_approval(command, description, smart_denied)
            # Short monotonic ids in callback_data map back to session_key.
            import itertools
            if not hasattr(self, "_approval_counter"):
                self._approval_counter = itertools.count(1)
            approval_id = next(self._approval_counter)
            buttons = [InlineKeyboardButton("✅ Allow Once", callback_data=f"ea:once:{approval_id}")]
            if not smart_denied and allow_session:
                buttons.append(InlineKeyboardButton("✅ Session", callback_data=f"ea:session:{approval_id}"))
                if allow_permanent:
                    buttons.append(InlineKeyboardButton("✅ Always", callback_data=f"ea:always:{approval_id}"))
            buttons.append(InlineKeyboardButton("❌ Deny", callback_data=f"ea:deny:{approval_id}"))
            return text, InlineKeyboardMarkup(
                self._rows_of_two(buttons)), lambda msg: self._approval_state.__setitem__(approval_id, session_key)
        return await self._send_prompt(
            "send_exec_approval", chat_id, metadata, build, parse_mode=ParseMode.HTML,
            thread_id=self._metadata_thread_id(metadata), reply_to_mode=self._reply_to_mode)

    async def send_slash_confirm(
        self, chat_id: str, title: str, message: str, session_key: str, confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Render a three-button slash-command confirmation prompt."""
        def build():
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Approve Once", callback_data=f"sc:once:{confirm_id}"),
                    InlineKeyboardButton("🔒 Always Approve", callback_data=f"sc:always:{confirm_id}")],
                [InlineKeyboardButton("❌ Cancel", callback_data=f"sc:cancel:{confirm_id}")],
           ])
            preview = self.format_message(self._truncate_preview(message, 3800))
            return preview, keyboard, lambda msg: self._slash_confirm_state.__setitem__(confirm_id, session_key)
        return await self._send_prompt(
            "send_slash_confirm", chat_id, metadata, build, thread_id=self._metadata_thread_id(metadata), reply_to_mode=self._reply_to_mode)

    async def send_clarify(
        self, chat_id: str, question: str, choices: Optional[list], clarify_id: str, session_key: str,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Render a clarify prompt: numbered buttons per choice plus "✏️ Other (type answer)" (flips to
        text-capture mode); without choices, plain question and the gateway text-intercept captures."""
        def build():
            text = f"❓ {_html.escape(question)}"
            keyboard = None
            if choices:
                # Full option text in the body (mobile truncates button labels); buttons keep numeric labels.
                text += "\n\n" + "\n".join(f"{i + 1}. {_html.escape(str(c))}" for i, c in enumerate(choices))
                # Telegram caps callback_data at 64 bytes; keep "cl:<id>:<idx>" short.
                rows = [[InlineKeyboardButton(str(idx + 1), callback_data=f"cl:{clarify_id}:{idx}")] for idx in range(len(choices))]
                rows.append([InlineKeyboardButton("✏️ Other (type answer)", callback_data=f"cl:{clarify_id}:other")])
                keyboard = InlineKeyboardMarkup(rows)
            return text, keyboard, lambda msg: self._bind_clarify_prompt(clarify_id, session_key, msg)
        return await self._send_prompt(
            "send_clarify", chat_id, metadata, build, parse_mode=ParseMode.HTML, thread_id=self._metadata_thread_id(metadata))

    @staticmethod
    def _provider_get_label():
        try:
            from hermes_cli.providers import get_label
        except ImportError:
            def get_label(slug):
                return slug
        return get_label

    async def send_model_picker(
        self, chat_id: str, providers: list, current_model: str, current_provider: str, session_key: str,
        on_model_selected, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an inline-keyboard model picker: provider → model drill-down, edited in place."""
        def build():
            keyboard, provider_page_info = self._build_provider_keyboard(providers, 0)
            text = self.format_message(
                self._provider_list_text(current_model, self._provider_get_label()(current_provider), provider_page_info)
            )

            def _remember(msg):
                self._model_picker_state[str(chat_id)] = {
                    "msg_id": msg.message_id, "providers": providers, "session_key": session_key, "on_model_selected": on_model_selected,
                    "current_model": current_model, "current_provider": current_provider, "provider_page": 0}
            return text, keyboard, _remember
        return await self._send_prompt(
            "send_model_picker", chat_id, metadata, build, thread_id=metadata.get("thread_id") if metadata else None,
            reply_to_mode=self._reply_to_mode)

    _PROVIDER_PAGE_SIZE = 10

    async def send_choice_picker(
        self, chat_id: str, title: str, choices: list, session_key: str, on_choice_selected,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Flat inline-keyboard picker (one tap → one value) for /reasoning, /fast, etc. Each choice dict:
        ``{"value": str, "label": str, "is_current": bool}``."""
        def build():
            buttons = []
            for i, choice in enumerate(choices):
                label = str(choice.get("label") or choice.get("value") or "")
                if choice.get("is_current"):
                    label = f"✓ {label}"
                buttons.append(InlineKeyboardButton(label, callback_data=f"cp:{i}"))
            if not buttons:
                return SendResult(success=False, error="No choices")
            keyboard = InlineKeyboardMarkup(self._rows_of_two(buttons))

            def _remember(msg):
                self._choice_picker_state[str(chat_id)] = {
                    "msg_id": msg.message_id, "choices": choices, "session_key": session_key, "on_choice_selected": on_choice_selected}
            return self.format_message(title), keyboard, _remember
        return await self._send_prompt(
            "send_choice_picker", chat_id, metadata, build, thread_id=metadata.get("thread_id") if metadata else None,
            reply_to_mode=self._reply_to_mode)

    async def _edit_result_text(self, query, result_text: str) -> None:
        """Replace a picker message with ``result_text`` (MarkdownV2, then plain, then give up), keyboard removed."""
        try:
            await query.edit_message_text(text=self.format_message(result_text), parse_mode=ParseMode.MARKDOWN_V2, reply_markup=None)
        except Exception:
            with contextlib.suppress(Exception):
                await query.edit_message_text(text=result_text, parse_mode=None, reply_markup=None)

    async def _handle_choice_picker_callback(self, query, data: str, chat_id: str) -> None:
        """Handle choice picker button taps (cp:<index>)."""
        state = self._choice_picker_state.get(chat_id)
        if not state:
            await query.answer(text="Picker expired — run the command again.")
            return
        # Same auth gate as approval buttons: strangers in a shared group must not flip session state.
        if not await self._callback_authorized(query, self._callback_ctx(query), "⛔ You are not authorized to change this setting."):
            return
        try:
            choice = state["choices"][int(data[3:])]
        except (ValueError, IndexError):
            await query.answer(text="Invalid selection.")
            return
        callback = state.get("on_choice_selected")
        if not callback:
            await query.answer(text="Picker expired.")
            return
        try:
            result_text = await callback(chat_id, str(choice.get("value") or ""))
        except Exception as exc:
            logger.error("Choice picker selection failed: %s", exc)
            result_text = f"Error applying selection: {exc}"
        await self._edit_result_text(query, result_text)
        await query.answer()
        self._choice_picker_state.pop(chat_id, None)

    _MODEL_PAGE_SIZE = 8

    @staticmethod
    def _provider_button(p: dict) -> "InlineKeyboardButton":
        count = p.get("total_models", len(p.get("models", [])))
        label = f"{p['name']} ({count})"
        if p.get("is_current"):
            label = f"✓ {label}"
        return InlineKeyboardButton(label, callback_data=f"mp:{p['slug']}")

    @staticmethod
    def _picker_nav_row(page: int, total_pages: int, prefix: str) -> list:
        """``◀ Prev | n/N | Next ▶`` row (``prefix`` = ``mpv``/``mg`` page callback)."""
        nav: list = []
        if page > 0:
            nav.append(InlineKeyboardButton("◀ Prev", callback_data=f"{prefix}:{page - 1}"))
        nav.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="mx:noop"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton("Next ▶", callback_data=f"{prefix}:{page + 1}"))
        return nav

    @staticmethod
    def _picker_back_cancel_row() -> list:
        return [InlineKeyboardButton("◀ Back", callback_data="mb"), InlineKeyboardButton("✗ Cancel", callback_data="mx")]

    def _paged_keyboard(self, buttons: list, page_meta: dict, nav_prefix: str, tail_row: list) -> tuple:
        rows = self._rows_of_two(buttons)
        if page_meta["total_pages"] > 1:
            rows.append(self._picker_nav_row(page_meta["page"], page_meta["total_pages"], nav_prefix))
        rows.append(tail_row)
        return InlineKeyboardMarkup(rows), page_meta["page_info"]

    def _build_provider_keyboard(self, providers: list, page: int = 0) -> tuple:
        """Paginated top-level provider keyboard folding provider families (Kimi/Moonshot, MiniMax, xAI…)
        into one ``mpg:<gid>`` button via the shared ``group_providers`` fold; singles are ``mp:<slug>``."""
        try:
            from hermes_cli.models_catalog_static import group_providers
        except Exception:
            group_providers = None
        by_slug = {p.get("slug"): p for p in providers}
        buttons: list = []
        if group_providers is not None:
            for row in group_providers([p.get("slug") for p in providers]):
                if row["kind"] == "group":
                    members = [by_slug[m] for m in row["members"] if m in by_slug]
                    count = sum(m.get("total_models", len(m.get("models", []))) for m in members)
                    label = f"{row['label']} ▸ ({count})"
                    if any(m.get("is_current") for m in members):
                        label = f"✓ {label}"
                    buttons.append(InlineKeyboardButton(label, callback_data=f"mpg:{row['group_id']}"))
                else:
                    p = by_slug.get(row["slug"])
                    if p is not None:
                        buttons.append(self._provider_button(p))
        else:
            buttons = [self._provider_button(p) for p in providers]
        page_buttons, page_meta = self._format_choice_page(buttons, page, self._PROVIDER_PAGE_SIZE)
        return self._paged_keyboard(page_buttons, page_meta, "mpv", [InlineKeyboardButton("✗ Cancel", callback_data="mx")])

    def _build_model_keyboard(self, models: list, page: int) -> tuple:
        """Build paginated model buttons. Returns (keyboard, page_info_text)."""
        page_models, page_meta = self._format_choice_page(models, page, self._MODEL_PAGE_SIZE)
        start = page_meta["start"]
        buttons: list = []
        for i, model_id in enumerate(page_models):
            short = model_id.split("/")[-1] if "/" in model_id else model_id
            if len(short) > 38:
                short = short[:35] + "..."
            buttons.append(InlineKeyboardButton(short, callback_data=f"mm:{start + i}"))
        return self._paged_keyboard(buttons, page_meta, "mg", self._picker_back_cancel_row())

    async def _picker_edit(self, query, text_md: str, keyboard) -> None:
        """Re-render the picker message in place (MarkdownV2) and ack the tap."""
        await query.edit_message_text(text=self.format_message(text_md), parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard)
        await query.answer()

    async def _picker_show_models(self, query, state: dict, page: int) -> None:
        """Render the model page for the provider currently selected in ``state``."""
        models = state.get("model_list", [])
        state["model_page"] = page
        keyboard, page_info = self._build_model_keyboard(models, page)
        pname = state.get("selected_provider_name", "")
        provider_slug = state.get("selected_provider", "")
        provider = next((p for p in state["providers"] if p["slug"] == provider_slug), None)
        total = provider.get("total_models", len(models)) if provider else len(models)
        shown = len(models)
        extra = f"\n_{total - shown} more available — type `/model <name>` directly_" if total > shown else ""
        await self._picker_edit(query, f"⚙ *Model Configuration*\n\nProvider: *{pname}*{page_info}\nSelect a model:{extra}", keyboard)

    @staticmethod
    def _provider_list_text(current_model: str, provider_label: str, page_info: str) -> str:
        return (f"⚙ *Model Configuration*\n\nCurrent model: `{current_model or 'unknown'}`\n"
                f"Provider: {provider_label}\n\nSelect a provider:{page_info}")

    async def _picker_show_providers(self, query, state: dict, page: int, get_label) -> None:
        """Render the (folded, paginated) provider list."""
        keyboard, provider_page_info = self._build_provider_keyboard(state["providers"], page)
        try:
            provider_label = get_label(state["current_provider"])
        except Exception:
            provider_label = state["current_provider"]
        await self._picker_edit(query, self._provider_list_text(state["current_model"], provider_label, provider_page_info), keyboard)

    async def _picker_selection(self, query, state: dict, raw_idx: str) -> Optional[tuple]:
        """Resolve ``mm:``/``mc:`` index → ``(idx, model_id, provider_slug, callback)``; answers + None on error."""
        try:
            idx = int(raw_idx)
        except ValueError:
            await query.answer(text="Invalid selection.")
            return None
        model_list = state.get("model_list", [])
        if idx < 0 or idx >= len(model_list):
            await query.answer(text="Invalid model index.")
            return None
        callback = state.get("on_model_selected")
        if not callback:
            await query.answer(text="Picker expired.")
            return None
        return idx, model_list[idx], state.get("selected_provider", ""), callback

    async def _picker_switch(self, query, chat_id: str, model_id: str, provider_slug: str, callback) -> None:
        """Perform the model switch, render the result, and drop the picker state."""
        switch_failed = False
        try:
            result_text = await callback(chat_id, model_id, provider_slug)
        except Exception as exc:
            logger.error("Model picker switch failed: %s", exc)
            result_text = f"Error switching model: {exc}"
            switch_failed = True
        await self._edit_result_text(query, result_text)
        await query.answer(text="Switch failed." if switch_failed else "Model switched!")
        self._model_picker_state.pop(chat_id, None)

    @staticmethod
    async def _parse_page(query, raw: str) -> Optional[int]:
        try:
            return int(raw)
        except ValueError:
            await query.answer(text="Invalid page.")
            return None

    async def _handle_model_picker_callback(self, query, data: str, chat_id: str) -> None:
        """Handle model picker callbacks (mp:/mpg:/mpv:/mm:/mc:/mb/mx/mg:)."""
        state = self._model_picker_state.get(chat_id)
        if not state:
            await query.answer(text="Picker expired — use /model again.")
            return
        get_label = self._provider_get_label()
        if data.startswith("mp:"):  # provider selected: show model buttons (page 0)
            provider_slug = data[3:]
            provider = next((p for p in state["providers"] if p["slug"] == provider_slug), None)
            if not provider:
                await query.answer(text="Provider not found.")
                return
            state["selected_provider"] = provider_slug
            state["selected_provider_name"] = provider.get("name", provider_slug)
            state["model_list"] = provider.get("models", [])
            await self._picker_show_models(query, state, 0)
        elif data.startswith("mg:"):  # model page navigation
            page = await self._parse_page(query, data[3:])
            if page is not None:
                await self._picker_show_models(query, state, page)
        elif data.startswith("mpv:"):  # provider page navigation
            page = await self._parse_page(query, data[4:])
            if page is not None:
                state["provider_page"] = page
                await self._picker_show_providers(query, state, page, get_label)
        elif data.startswith("mc:"):  # expensive model confirmed: perform the switch
            sel = await self._picker_selection(query, state, data[3:])
            if sel is not None:
                _idx, model_id, provider_slug, callback = sel
                await self._picker_switch(query, chat_id, model_id, provider_slug, callback)
        elif data.startswith("mm:"):  # model selected: warn if expensive, else perform the switch
            sel = await self._picker_selection(query, state, data[3:])
            if sel is None:
                return
            idx, model_id, provider_slug, callback = sel
            try:
                from hermes_cli.model_selection_guards import combined_selection_warning
                # Pricing lookup may hit models.dev on a cache miss — keep it off the event loop.
                warning = await asyncio.to_thread(combined_selection_warning, model_id, provider=provider_slug)
            except Exception:
                warning = None
            if warning is not None:
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("Switch anyway", callback_data=f"mc:{idx}")], self._picker_back_cancel_row()])
                await query.edit_message_text(
                    text=self.format_message(f"⚠ *{warning.title}*\n\n{warning.message}"),
                    parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard)
                await query.answer(text="Confirm model selection")
                return
            await self._picker_switch(query, chat_id, model_id, provider_slug, callback)
        elif data.startswith("mpg:"):  # provider group selected: show member providers
            group_id = data[4:]
            try:
                from hermes_cli.models_catalog_static import PROVIDER_GROUPS
                _label, _desc, member_slugs = PROVIDER_GROUPS.get(group_id, ("", "", []))
            except Exception:
                _label, member_slugs = "", []
            by_slug = {p["slug"]: p for p in state["providers"]}
            members = [by_slug[m] for m in member_slugs if m in by_slug]
            if not members:
                await query.answer(text="Group not found.")
                return
            rows = self._rows_of_two([self._provider_button(p) for p in members])
            rows.append(self._picker_back_cancel_row())
            await self._picker_edit(
                query, f"⚙ *Model Configuration*\n\nProvider family: *{_label or group_id}*\n\nSelect a provider:",
                InlineKeyboardMarkup(rows))
        elif data == "mb":  # back to provider list (folds groups)
            await self._picker_show_providers(query, state, int(state.get("provider_page", 0) or 0), get_label)
        elif data == "mx":
            self._model_picker_state.pop(chat_id, None)
            await query.edit_message_text(text="Model selection cancelled.", reply_markup=None)
            await query.answer()
        else:
            await query.answer()  # e.g. page-counter button "mx:noop"

    async def _notify_clarify_expired(self, query, user_display: str) -> None:
        """Tell the user a clarify tap arrived too late (entry evicted or gateway restarted) — otherwise
        the tap leaves a misleading ✓ the agent never sees."""
        with contextlib.suppress(Exception):
            await query.answer(text="⚠️ This prompt expired — please /retry.")
        await self._edit_html_quiet(
            query, f"❓ {_html.escape(query.message.text or '')}\n\n<i>⚠️ This question expired or the session reset — please /retry.</i>")

    @staticmethod
    async def _edit_html_quiet(query, text: str) -> None:
        """HTML edit with the keyboard removed; failures ignored (non-fatal)."""
        with contextlib.suppress(Exception):
            await query.edit_message_text(text=text, parse_mode=ParseMode.HTML, reply_markup=None)

    async def _edit_md_quiet(self, query, text_md: str) -> None:
        """MarkdownV2 edit with the keyboard removed; failures ignored (non-fatal)."""
        with contextlib.suppress(Exception):
            await query.edit_message_text(text=self.format_message(text_md), parse_mode=ParseMode.MARKDOWN_V2, reply_markup=None)

    async def _handle_inline_query(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Answer ``@botname <query>`` with a searchable command/skill picker (the ``/`` menu is capped at
        60 slots). Results are computed per keystroke, 50 per page; tapping sends ``/cmd`` text as the
        user, so dispatch flows through the normal command path. Inline queries arrive from ANY chat, so
        unauthorized users get an empty list (the skill catalog is not leaked)."""
        inline_query = getattr(update, "inline_query", None)
        if inline_query is None:
            return
        from_user = getattr(inline_query, "from_user", None)
        user_id = str(getattr(from_user, "id", "") or "").strip()
        try:
            # No chat context on inline queries — authorize on user identity alone, DM-shaped.
            authorized = bool(user_id) and self._is_callback_user_authorized(
                user_id, chat_id=user_id, chat_type="private", user_name=getattr(from_user, "username", None))
        except Exception:
            logger.debug("[%s] inline picker auth check failed", self.name, exc_info=True)
            authorized = False
        if not authorized:
            try:
                from plugins.platforms.telegram.inline_picker import CACHE_TIME_SECONDS as _deny_cache
                await inline_query.answer([], cache_time=_deny_cache, is_personal=True)
            except Exception:
                logger.debug("[%s] inline picker empty answer failed", self.name, exc_info=True)
            return
        try:
            from telegram import InlineQueryResultArticle, InputTextMessageContent
            from plugins.platforms.telegram.inline_picker import CACHE_TIME_SECONDS as _CACHE, build_inline_results
            results, next_offset = build_inline_results(
                getattr(inline_query, "query", "") or "", offset=getattr(inline_query, "offset", "") or "")
            articles = [
                InlineQueryResultArticle(
                    id=r["id"], title=r["title"], description=r["description"],
                    input_message_content=InputTextMessageContent(r["message_text"]))
                for r in results
           ]
            # is_personal: catalogs differ per user (auth, disabled skills) — never share cached pages.
            await inline_query.answer(articles, cache_time=_CACHE, is_personal=True, next_offset=next_offset)
        except Exception:
            logger.debug("[%s] inline picker answer failed", self.name, exc_info=True)

    @staticmethod
    def _callback_ctx(query) -> Dict[str, Any]:
        """Chat/thread/user context of a button tap, for the callback auth gate."""
        query_message = getattr(query, "message", None)
        query_chat = getattr(query_message, "chat", None)
        return {
            "chat_id": getattr(query_message, "chat_id", None), "chat_type": getattr(query_chat, "type", None),
            "thread_id": getattr(query_message, "message_thread_id", None), "user_name": getattr(query.from_user, "first_name", None)}

    async def _callback_authorized(self, query, cb: Dict[str, Any], denial_text: str) -> bool:
        """Gate a button tap on the callback allowlist; answers ``denial_text`` when refused."""
        if self._is_callback_user_authorized(
            str(getattr(query.from_user, "id", "")), chat_id=cb["chat_id"],
            chat_type=str(cb["chat_type"]) if cb["chat_type"] is not None else None,
            thread_id=str(cb["thread_id"]) if cb["thread_id"] is not None else None, user_name=cb["user_name"]):
            return True
        await query.answer(text=denial_text)
        return False

    async def _handle_callback_query(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Dispatch inline keyboard button clicks on the callback_data prefix."""
        query = update.callback_query
        if not query or not query.data:
            return
        data = query.data
        cb = self._callback_ctx(query)
        # Model picker / generic choice picker (/reasoning, /fast) need a chat id.
        for prefixes, handler in (
            (("mp:", "mpg:", "mpv:", "mm:", "mc:", "mb", "mx", "mg:"), self._handle_model_picker_callback),
            (("cp:",), self._handle_choice_picker_callback)):
            if data.startswith(prefixes):
                chat_id = str(query.message.chat_id) if query.message else None
                if chat_id:
                    await handler(query, data, chat_id)
                return
        for prefix, handler in (
            ("gt:", self._handle_gmail_triage_callback), ("ea:", self._handle_exec_approval_callback),
            ("sc:", self._handle_slash_confirm_callback), ("cl:", self._handle_clarify_callback),
            ("update_prompt:", self._handle_update_prompt_callback)):
            if data.startswith(prefix):
                await handler(query, data, cb)
                return

    async def _claim_callback_state(self, query, cb: Dict[str, Any], state: dict, key, denial: str, resolved: str, *, pop: bool = True):
        """Auth-gate a button tap, then claim its pending entry; None (after answering) when refused or expired."""
        if not await self._callback_authorized(query, cb, denial):
            return None
        session_key = state.pop(key, None) if pop else state.get(key)
        if not session_key:
            await query.answer(text=resolved)
        return session_key

    async def _handle_exec_approval_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """``ea:<choice>:<approval_id>`` — resolve a pending exec approval."""
        parts = data.split(":", 2)
        if len(parts) != 3:
            return
        choice = parts[1]  # once, session, always, deny
        try:
            approval_id = int(parts[2])
        except (ValueError, IndexError):
            await query.answer(text="Invalid approval data.")
            return
        session_key = await self._claim_callback_state(
            query, cb, self._approval_state, approval_id, "⛔ You are not authorized to approve commands.",
            "This approval has already been resolved.")
        if not session_key:
            return
        user_display = getattr(query.from_user, "first_name", "User")
        # Resolve FIRST (unblocks the agent thread), render after: a tap landing after the wait timed out
        # (count == 0) must NOT claim "Approved" — the command was already denied.
        try:
            # Rendering happens after so the message reflects what actually occurred: a tap that lands after
            # the approval wait timed out (count == 0) must NOT claim "Approved" — the command was already
            # denied and will not run (#63501 regression follow-up: 60s waits made stale taps common).
            from tools.approval import resolve_gateway_approval
            count = resolve_gateway_approval(session_key, choice)
            logger.info(
                "Telegram button resolved %d approval(s) for session %s (choice=%s, user=%s)", count, session_key, choice, user_display)
        except Exception as exc:
            logger.error("Failed to resolve gateway approval from Telegram button: %s", exc)
            count = 0
        if count:
            label_map = {
                "once": "✅ Approved once", "session": "✅ Approved for session", "always": "✅ Approved permanently", "deny": "❌ Denied",
            }
            label = label_map.get(choice, "Resolved")
            edit_text = f"{label} by {user_display}"
        else:
            label = "⌛ Approval expired"
            edit_text = f"{label} — no command was waiting. It already timed out (and was denied) or was resolved elsewhere."
        await query.answer(text=label)
        await self._edit_md_quiet(query, edit_text)
        # Typing was paused when the approval was sent; the text /approve and /deny paths resume it too.
        if count and cb["chat_id"] is not None:
            self.resume_typing_for_chat(str(cb["chat_id"]))

    async def _handle_slash_confirm_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """``sc:<choice>:<confirm_id>`` — resolve a slash-command confirmation."""
        parts = data.split(":", 2)
        if len(parts) != 3:
            return
        choice = parts[1]  # once, always, cancel
        confirm_id = parts[2]
        session_key = await self._claim_callback_state(
            query, cb, self._slash_confirm_state, confirm_id, "⛔ You are not authorized to answer this prompt.",
            "This prompt has already been resolved.")
        if not session_key:
            return
        label_map = {"once": "✅ Approved once", "always": "🔒 Always approve", "cancel": "❌ Cancelled"}
        user_display = getattr(query.from_user, "first_name", "User")
        label = label_map.get(choice, "Resolved")
        await query.answer(text=label)
        await self._edit_md_quiet(query, f"{label} by {user_display}")
        # The runner stored a handler keyed by session_key; run it and send any returned text as a follow-up.
        try:
            from tools import slash_confirm as _slash_confirm_mod
            result_text = await _slash_confirm_mod.resolve(session_key, confirm_id, choice)
            if result_text and query.message:
                # Inherit the prompt's topic: forums use message_thread_id; private DM-topic lanes need
                # both the topic id and the prompt reply anchor.
                thread_id = getattr(query.message, "message_thread_id", None)
                chat_type = getattr(getattr(query.message, "chat", None), "type", None)
                prompt_message_id = getattr(query.message, "message_id", None)
                send_kwargs: Dict[str, Any] = {
                    "chat_id": int(query.message.chat_id), "text": self.format_message(result_text),
                    "parse_mode": ParseMode.MARKDOWN_V2, **self._link_preview_kwargs()}
                is_private_chat = str(getattr(chat_type, "value", chat_type)).lower() in {
                    "private", str(ChatType.PRIVATE).lower(), str(getattr(ChatType.PRIVATE, "value", ChatType.PRIVATE)).lower()}
                if thread_id is not None:
                    meta: Dict[str, Any] = {"thread_id": str(thread_id)}
                    reply_to_id = None
                    if is_private_chat and prompt_message_id is not None:
                        reply_to_id = send_kwargs["reply_to_message_id"] = int(prompt_message_id)
                        meta["telegram_dm_topic_reply_fallback"] = True
                    send_kwargs.update(self._thread_kwargs_for_send(
                        str(
                            query.message.chat_id
                        ), str(thread_id), meta, reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode))
                await self._send_message_with_thread_fallback(**send_kwargs)
        except Exception as exc:
            logger.error("[%s] slash-confirm callback failed: %s", self.name, exc, exc_info=True)

    def _bind_clarify_prompt(self, clarify_id: str, session_key: str, msg: Any) -> None:
        """Register a pending clarify against the message that carries its buttons (#102957)."""
        self._clarify_state[clarify_id] = _ClarifyPrompt(
            session_key=session_key, chat_id=_callback_identity(getattr(msg, "chat_id", None)),
            message_id=_callback_identity(getattr(msg, "message_id", None)),
            thread_id=_callback_identity(getattr(msg, "message_thread_id", None)))

    async def _clarify_callback_matches_prompt(self, query, clarify_id: str, prompt: "_ClarifyPrompt") -> bool:
        """Fail closed unless the tap came from the message that rendered this clarify (#102957).

        The allowlist gate upstream of this proves only that the caller may answer *some* prompt here — it
        cannot tell a deliberate tap from a callback that arrived on another message.  On a mismatch the
        entry is left pending so the real prompt stays answerable, and the rejection is logged with enough
        non-secret provenance (callback-query id, immutable caller id, both message identities) to tell a
        stale tap from a mismatched one after the fact.
        """
        if prompt.message_id is None:
            # Nothing to match against: the send path reported no message id.  Only a Bot API response can
            # reach this branch — no caller-controlled input does — so treat it as unverifiable, not hostile.
            logger.warning(
                "[%s] Telegram clarify callback unverifiable — no prompt message bound (id=%s)", self.name, clarify_id)
            return True
        message = getattr(query, "message", None)
        actual_chat_id = _callback_identity(getattr(message, "chat_id", None))
        actual_message_id = _callback_identity(getattr(message, "message_id", None))
        if actual_chat_id == prompt.chat_id and actual_message_id == prompt.message_id:
            return True
        logger.warning(
            "[%s] Telegram clarify callback rejected — prompt binding mismatch (id=%s, expected chat=%s message=%s "
            "thread=%s, got chat=%s message=%s thread=%s, callback_query=%s, caller_id=%s)",
            self.name, clarify_id, prompt.chat_id, prompt.message_id, prompt.thread_id, actual_chat_id,
            actual_message_id, _callback_identity(getattr(message, "message_thread_id", None)),
            getattr(query, "id", None), _callback_identity(getattr(query.from_user, "id", None)))
        with contextlib.suppress(Exception):
            await query.answer(text="⛔ This button belongs to a different prompt.")
        return False

    async def _handle_clarify_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """``cl:<clarify_id>:<idx|other>`` — resolve a clarify prompt or flip to text capture."""
        parts = data.split(":", 2)
        if len(parts) != 3:
            return
        clarify_id = parts[1]
        choice_token = parts[2]
        prompt = await self._claim_callback_state(
            query, cb, self._clarify_state, clarify_id, "⛔ You are not authorized to answer this prompt.",
            "This prompt has already been resolved.", pop=False)
        if not prompt:
            return
        if not await self._clarify_callback_matches_prompt(query, clarify_id, prompt):
            return
        user_display = getattr(query.from_user, "first_name", "User")
        if choice_token == "other":
            # Flip to text-capture: the gateway's text-intercept resolves the clarify with the next message.
            # Do NOT pop _clarify_state yet — still needed if the entry gets cleared by something else.
            flipped = False
            try:
                from tools.clarify_gateway import mark_awaiting_text
                flipped = mark_awaiting_text(clarify_id)
            except Exception as exc:
                logger.warning("[%s] mark_awaiting_text failed: %s", self.name, exc)
            if not flipped:
                # Entry evicted / gateway restarted — a typed answer would go nowhere.
                self._clarify_state.pop(clarify_id, None)
                await self._notify_clarify_expired(query, user_display)
                return
            await query.answer(text="✏️ Type your answer in the chat.")
            await self._edit_html_quiet(
                query, f"❓ {query.message.text or ''}\n\n<i>Awaiting typed response from {_html.escape(user_display)}…</i>")
            return
        # Numeric choice → resolve immediately with the chosen text
        try:
            idx = int(choice_token)
        except (ValueError, TypeError):
            await query.answer(text="Invalid choice.")
            return
        resolved_text: Optional[str] = None
        try:
            from tools.clarify_gateway import _entries as _clarify_entries  # type: ignore
            entry = _clarify_entries.get(clarify_id)
            if entry and entry.choices and 0 <= idx < len(entry.choices):
                resolved_text = entry.choices[idx]
        except Exception:
            resolved_text = None
        if resolved_text is None:
            # Race (timeout / session reset): echo the index so the agent sees an intentional response.
            resolved_text = f"choice {idx + 1}"
        self._clarify_state.pop(clarify_id, None)
        try:
            from tools.clarify_gateway import resolve_gateway_clarify
            resolved = resolve_gateway_clarify(clarify_id, resolved_text)
        except Exception as exc:
            logger.error("[%s] resolve_gateway_clarify failed: %s", self.name, exc)
            resolved = False
        if resolved:
            await query.answer(text=f"✓ {resolved_text[:60]}")
            await self._edit_html_quiet(
                query, f"❓ {_html.escape(query.message.text or '')}\n\n<b>{_html.escape(user_display)}:</b> {_html.escape(resolved_text)}")
            logger.info(
                "Telegram clarify button resolved (id=%s, choice=%r, user=%s, caller_id=%s, callback_query=%s, "
                "chat=%s, message=%s)", clarify_id, resolved_text, user_display,
                _callback_identity(getattr(query.from_user, "id", None)), getattr(query, "id", None),
                prompt.chat_id, prompt.message_id)
        else:
            # Entry evicted / gateway restarted between ask and tap.
            await self._notify_clarify_expired(query, user_display)
            logger.warning("Telegram clarify button: resolve_gateway_clarify returned False (id=%s)", clarify_id)

    async def _handle_update_prompt_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """``update_prompt:<y|n>`` — forward the answer to the update process."""
        answer = data.split(":", 1)[1]  # "y" or "n"
        if not await self._callback_authorized(query, cb, "⛔ You are not authorized to answer update prompts."):
            return
        await query.answer(text=f"Sent '{answer}' to the update process.")
        await self._edit_md_quiet(query, f"⚕ Update prompt answered: *{'Yes' if answer == 'y' else 'No'}*")
        try:
            from hermes_constants import get_hermes_home
            response_path = get_hermes_home() / ".update_response"
            tmp = response_path.with_suffix(".tmp")
            tmp.write_text(answer, encoding="utf-8")
            tmp.replace(response_path)
            logger.info("Telegram update prompt answered '%s' by user %s", answer, getattr(query.from_user, "id", "unknown"))
        except Exception as exc:
            logger.error("Failed to write update response from callback: %s", exc)

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

    async def _handle_gmail_triage_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """Dispatch a gmail-triage inline-button callback (gt:verb:arg)."""
        parts = data.split(":", 2)
        if len(parts) != 3:
            await query.answer(text="Invalid gmail-triage data.")
            return
        verb, arg = parts[1], parts[2]
        if not await self._callback_authorized(query, cb, "⛔ You are not authorized to act on this email."):
            return
        entry = self._GT_VERB_DISPATCH.get(verb)
        if not entry:
            await query.answer(text=f"Unknown verb: {verb}")
            return
        script_name, extra_args, success_label, is_state_verb = entry
        script_path = _Path.home() / ".hermes" / "scripts" / "gmail-triage" / script_name
        if not script_path.exists():
            await query.answer(text=f"❌ {script_name} missing")
            logger.error("[%s] gmail-triage script missing: %s", self.name, script_path)
            return
        success = False
        try:
            proc = await asyncio.create_subprocess_exec(
                str(script_path), arg, *extra_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode == 0:
                label = success_label
                success = True
                logger.info("[%s] gmail-triage callback ok: verb=%s arg=%s", self.name, verb, arg)
            else:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                last_line = stderr_text.splitlines()[-1] if stderr_text else f"exit {proc.returncode}"
                label = f"❌ {verb} failed: {last_line[:80]}"
                logger.error(
                    "[%s] gmail-triage callback failed: verb=%s arg=%s rc=%s stderr=%s", self.name, verb, arg, proc.returncode, stderr_text)
        except asyncio.TimeoutError:
            label = f"❌ {verb} timed out"
            logger.error("[%s] gmail-triage callback timed out: verb=%s arg=%s", self.name, verb, arg)
        except Exception as exc:
            label = f"❌ {verb} error: {exc}"
            logger.error("[%s] gmail-triage callback exception: verb=%s arg=%s err=%s", self.name, verb, arg, exc, exc_info=True)
        await query.answer(text=label)
        if not success:
            return
        original_text = (query.message.text or "") if query.message else ""
        appended = f"{original_text}\n— {label} by {getattr(query.from_user, 'first_name', 'User')}"
        # Sticky state verbs keep the keyboard so further actions can stack; one-shots strip it (can't fire twice).
        with contextlib.suppress(Exception):
            await query.edit_message_text(text=appended, **({} if is_state_verb else {"reply_markup": None}))

    def _missing_media_path_error(self, label: str, path: str) -> str:
        """File-not-found error for MEDIA delivery; /workspace-style paths often exist only in the sandbox."""
        error = f"{label} file not found: {path}"
        if path.startswith(("/workspace/", "/output/", "/outputs/")):
            error += (
                " (path may only exist inside the Docker sandbox. "
                "Bind-mount a host directory and emit the host-visible path in MEDIA: for gateway file delivery.)")
        return error

    def _telegram_media_too_large_note(self, label: str, file_size: Any, max_bytes: int) -> str:
        limit_mb = max(1, max_bytes // (1024 * 1024))
        try:
            size_text = f"{int(file_size or 0) / (1024 * 1024):.1f} MB"
        except (TypeError, ValueError):
            size_text = "unknown size"
        return f"[Telegram {label} skipped: file size {size_text} exceeds the {limit_mb} MB limit. Ask the user to send a smaller file.]"

    @staticmethod
    def _int_or_zero(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _telegram_media_size_allowed(self, source: Any, label: str) -> tuple[bool, Optional[str]]:
        """Validate Telegram media size before downloading into memory."""
        max_bytes = int(getattr(self, "_max_doc_bytes", 20 * 1024 * 1024) or 20 * 1024 * 1024)
        size = self._int_or_zero(getattr(source, "file_size", None))
        if size <= 0 or size <= max_bytes:
            return True, None
        return False, self._telegram_media_too_large_note(label, size, max_bytes)

    def _media_send_kwargs(
        self, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]]) -> tuple[Optional[int], Dict[str, Any]]:
        """Return ``(reply_to_id, base_kwargs)`` shared by every native media send."""
        reply_to_id = self._reply_to_message_id_for_send(reply_to, metadata, reply_to_mode=self._reply_to_mode)
        thread_kwargs = self._thread_kwargs_for_send(
            chat_id, self._metadata_thread_id(metadata), metadata, reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode)
        return reply_to_id, {
            "chat_id": normalize_telegram_chat_id(chat_id), "reply_to_message_id": reply_to_id,
            "read_timeout": _MEDIA_SEND_READ_TIMEOUT, **thread_kwargs, **self._notification_kwargs(metadata)}

    async def _send_media(
        self, send_fn: Any, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]], media_label: str,
        reset_media: Optional[Any] = None, **media_kwargs: Any) -> Any:
        """Send one native media payload with thread routing + DM-topic anchor retry."""
        reply_to_id, kwargs = self._media_send_kwargs(chat_id, reply_to, metadata)
        return await self._send_with_dm_topic_reply_anchor_retry(
            send_fn, {**kwargs, **media_kwargs}, metadata, reply_to_id, media_label, reset_media=reset_media)

    @staticmethod
    def _caption_1024(caption: Optional[str]) -> Optional[str]:
        return caption[:1024] if caption else None

    async def _send_voice_bubble(self, audio_file, chat_id, reply_to, metadata, caption, duration_secs):
        """sendVoice with caption variants: MarkdownV2 when it fits 1024 chars, plain fallback when the
        Bot API rejects the entities; anything else is a real error."""
        # Render caption markdown (#32029): auto-TTS captions carry the agent's markdown reply, which showed
        # literal *asterisks* and [links](...) without a parse_mode. Format to MarkdownV2 when it fits the
        # 1024-char caption cap; fall back to the raw text (previous behaviour) when formatting would
        # overflow or the Bot API rejects the entities.
        _caption_variants: List[tuple] = []
        if caption:
            try:
                _formatted_caption = self.format_message(caption)
                if utf16_len(_formatted_caption) <= 1024:
                    _caption_variants.append((_formatted_caption, ParseMode.MARKDOWN_V2))
            except Exception:
                logger.debug("[%s] voice caption MarkdownV2 formatting failed; sending plain caption", self.name, exc_info=True)
            _caption_variants.append((caption[:1024], None))
        else:
            _caption_variants.append((None, None))
        _last_parse_error: Optional[Exception] = None
        for _cap_text, _cap_parse_mode in _caption_variants:
            try:
                return await self._send_media(
                    self._bot.send_voice, chat_id, reply_to, metadata, "voice", reset_media=lambda: audio_file.seek(0),
                    voice=audio_file, caption=_cap_text, parse_mode=_cap_parse_mode, duration=duration_secs)
            except Exception as _cap_error:
                err = str(_cap_error).lower()
                if _cap_parse_mode is not None and ("parse" in err or "entit" in err):
                    logger.warning(
                        "[%s] voice caption MarkdownV2 rejected, retrying plain: %s", self.name, _redact_telegram_error_text(_cap_error))
                    _last_parse_error = _cap_error
                    audio_file.seek(0)
                    continue
                raise
        raise _last_parse_error or RuntimeError("Telegram send_voice failed for all caption variants")

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send audio as a native Telegram voice message or audio file."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        _transcoded_voice_path: Optional[str] = None
        try:
            if not os.path.exists(audio_path):
                return SendResult(success=False, error=self._missing_media_path_error("Audio", audio_path))
            # sendVoice only accepts Ogg/Opus: an explicit voice-bubble request (is_voice) transcodes via
            # ffmpeg; otherwise route by extension (.mp3/.m4a → sendAudio, others → document).
            if kwargs.get("is_voice") and os.path.splitext(audio_path)[1].lower() not in (".ogg", ".opus"):
                from gateway.platforms.base import transcode_to_ogg_opus
                _transcoded_voice_path = await asyncio.to_thread(transcode_to_ogg_opus, audio_path)
                if _transcoded_voice_path:
                    audio_path = _transcoded_voice_path
                else:
                    logger.warning(
                        "[%s] voice transcode unavailable for %s — sending original format (install ffmpeg for voice bubbles)",
                        self.name, os.path.basename(audio_path))
            # Telegram drops duration for long clips (~5 min+, shows 0:00).
            _duration_secs = await asyncio.to_thread(_probe_voice_duration_seconds, audio_path)
            with open(audio_path, "rb") as audio_file:
                ext = os.path.splitext(audio_path)[1].lower()
                if ext in {".ogg", ".opus"}:  # round playable voice bubble
                    msg = await self._send_voice_bubble(audio_file, chat_id, reply_to, metadata, caption, _duration_secs)
                elif ext in {".mp3", ".m4a"}:  # Bot API sendAudio only accepts MP3 / M4A
                    msg = await self._send_media(
                        self._bot.send_audio, chat_id, reply_to, metadata, "audio", reset_media=lambda: audio_file.seek(0),
                        audio=audio_file, caption=self._caption_1024(caption), duration=_duration_secs)
                else:  # formats Telegram can't play natively (.wav, .flac, ...)
                    return await self.send_document(
                        chat_id=chat_id, file_path=audio_path, caption=caption, reply_to=reply_to, metadata=metadata)
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.error(
                "[%s] Failed to send Telegram voice/audio, falling back to base adapter: %s", self.name,
                _redact_telegram_error_text(e), exc_info=True)
            return await super().send_voice(chat_id, audio_path, caption, reply_to, metadata=metadata)
        finally:
            if _transcoded_voice_path:
                with contextlib.suppress(OSError):
                    os.unlink(_transcoded_voice_path)

    async def send_multiple_images(
        self, chat_id: str, images: List[tuple], metadata: Optional[Dict[str, Any]] = None, human_delay: float = 0.0) -> None:
        """Send images as Telegram albums (``send_media_group``, 10 per chunk). Animated GIFs can't join a
        media group (need ``send_animation``) so they go via the base per-image path, as does a failed chunk."""
        if not self._bot or not images:
            return
        try:
            from telegram import InputMediaPhoto
        except Exception as exc:  # pragma: no cover - missing SDK
            logger.warning("[%s] InputMediaPhoto unavailable, falling back to per-image send: %s", self.name, exc)
            await super().send_multiple_images(chat_id, images, metadata, human_delay)
            return
        is_anim = lambda url: not url.startswith("file://") and self._is_animation_url(url)  # noqa: E731
        animations = [img for img in images if is_anim(img[0])]
        photos = [img for img in images if not is_anim(img[0])]
        if animations:
            await super().send_multiple_images(chat_id, animations, metadata, human_delay=human_delay)
        if not photos:
            return
        from urllib.parse import unquote as _unquote
        CHUNK = 10  # Telegram's album limit
        chunks = [photos[i:i + CHUNK] for i in range(0, len(photos), CHUNK)]
        for chunk_idx, chunk in enumerate(chunks):
            if human_delay > 0 and chunk_idx > 0:
                await asyncio.sleep(human_delay)
            media: List[Any] = []
            opened_files: List[Any] = []
            try:
                for image_url, alt_text in chunk:
                    source: Any = image_url
                    if image_url.startswith("file://"):
                        local_path = _unquote(image_url[7:])
                        if not os.path.exists(local_path):
                            logger.warning("[%s] Skipping missing image in media group: %s", self.name, local_path)
                            continue
                        source = open(local_path, "rb")
                        opened_files.append(source)
                    media.append(InputMediaPhoto(media=source, caption=self._caption_1024(alt_text)))
                if not media:
                    continue
                logger.info("[%s] Sending media group of %d photo(s) (chunk %d/%d)", self.name, len(media), chunk_idx + 1, len(chunks))
                reply_to_id, send_kwargs = self._media_send_kwargs(chat_id, None, metadata)

                def _reset_opened_files() -> None:
                    for fh in opened_files:
                        with contextlib.suppress(Exception):
                            fh.seek(0)

                await self._send_with_dm_topic_reply_anchor_retry(
                    self._bot.send_media_group, {**send_kwargs, "media": media}, metadata, reply_to_id,
                    "media group", reset_media=_reset_opened_files)
            except Exception as e:
                logger.warning(
                    "[%s] send_media_group failed (chunk %d/%d), falling back to per-image: %s", self.name,
                    chunk_idx + 1, len(chunks), _redact_telegram_error_text(e), exc_info=True)
                await super().send_multiple_images(chat_id, chunk, metadata, human_delay=human_delay)
            finally:
                for fh in opened_files:
                    with contextlib.suppress(Exception):
                        fh.close()

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a local image file natively as a Telegram photo."""
        async def _photo_failed(e: Exception) -> SendResult:
            error_str = str(e)
            # Dimension errors are expected for valid images Telegram refuses as photos → INFO.
            if "Photo_invalid_dimensions" in error_str or "PHOTO_INVALID_DIMENSIONS" in error_str:
                logger.info("[%s] Image dimensions exceed Telegram photo limits, sending as document: %s", self.name, image_path)
            else:
                logger.warning(
                    "[%s] Failed to send Telegram local image as photo, trying document fallback: %s", self.name,
                    _redact_telegram_error_text(e), exc_info=True)
            # Document has no dimension limit (50MB only); if even that fails, base adapter text.
            try:
                return await self.send_document(
                    chat_id=chat_id, file_path=image_path, caption=caption, file_name=os.path.basename(image_path),
                    reply_to=reply_to, metadata=metadata)
            except Exception as doc_err:
                logger.error(
                    "[%s] Failed to send Telegram local image as document, falling back to base adapter: %s",
                    self.name, doc_err, exc_info=True)
                return await super(TelegramAdapter, self).send_image_file(chat_id, image_path, caption, reply_to, metadata=metadata)
        return await self._send_local_file(
            "Image", image_path, chat_id, reply_to, metadata, "photo",
            lambda f: {"photo": f, "caption": self._caption_1024(caption)}, _photo_failed)

    async def _send_local_file(
        self, label: str, path: str, chat_id, reply_to, metadata, media_key: str, build_kwargs, on_error,
    ) -> SendResult:
        """Shared shell for native local-file sends: existence check, open, send with routing, then
        ``await on_error(exc)`` on any failure. ``build_kwargs(f)`` supplies the media kwargs."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            if not os.path.exists(path):
                return SendResult(success=False, error=self._missing_media_path_error(label, path))
            with open(path, "rb") as f:
                msg = await self._send_media(
                    getattr(self._bot, f"send_{media_key}"), chat_id, reply_to, metadata, media_key,
                    reset_media=lambda: f.seek(0), **build_kwargs(f))
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            return await on_error(e)

    async def _warn_then(self, media_key: str, e: Exception, fallback) -> SendResult:
        logger.warning("[%s] Failed to send %s: %s", self.name, media_key, _redact_telegram_error_text(e))
        return await fallback

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a document/file natively as a Telegram file attachment."""
        return await self._send_local_file(
            "File", file_path, chat_id, reply_to, metadata, "document",
            lambda f: {"document": f, "filename": file_name or os.path.basename(file_path), "caption": self._caption_1024(caption)},
            lambda e: self._warn_then(
                "document", e, super(
                    TelegramAdapter, self,
                ).send_document(chat_id, file_path, caption, file_name, reply_to, metadata=metadata)))

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a video natively as a Telegram video message."""
        return await self._send_local_file(
            "Video", video_path, chat_id, reply_to, metadata, "video",
            lambda f: {"video": f, "caption": self._caption_1024(caption)},
            lambda e: self._warn_then(
                "video", e, super(TelegramAdapter, self).send_video(chat_id, video_path, caption, reply_to, metadata=metadata),
            ))

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a URL image as a Telegram photo: URL send (<5MB) → download+upload (≤10MB) → base text."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        from tools.url_safety import is_safe_url
        if not is_safe_url(image_url):
            logger.warning("[%s] Blocked unsafe image URL (SSRF protection)", self.name)
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)
        photo_caption = self._caption_1024(caption)
        try:
            msg = await self._send_media(
                self._bot.send_photo, chat_id, reply_to, metadata, "URL photo", photo=image_url, caption=photo_caption)
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning(
                "[%s] URL-based send_photo failed, trying file upload: %s", self.name, _redact_telegram_error_text(e), exc_info=True)
            try:
                from gateway.platforms.base import _ssrf_redirect_guard
                from tools.url_safety import create_ssrf_safe_async_client
                async with create_ssrf_safe_async_client(timeout=30.0, event_hooks={"response": [_ssrf_redirect_guard]}) as client:
                    resp = await client.get(image_url)
                    resp.raise_for_status()
                    image_data = resp.content
                msg = await self._send_media(
                    self._bot.send_photo, chat_id, reply_to, metadata, "uploaded photo", photo=image_data, caption=photo_caption)
                return SendResult(success=True, message_id=str(msg.message_id))
            except Exception as e2:
                logger.error("[%s] File upload send_photo also failed: %s", self.name, e2, exc_info=True)
                return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)

    async def send_animation(
        self, chat_id: str, animation_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an animated GIF natively as a Telegram animation (auto-plays inline)."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            msg = await self._send_media(
                self._bot.send_animation, chat_id, reply_to, metadata, "animation", animation=animation_url,
                caption=self._caption_1024(caption))
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.error(
                "[%s] Failed to send Telegram animation, falling back to photo: %s", self.name,
                _redact_telegram_error_text(e), exc_info=True)
            return await self.send_image(chat_id, animation_url, caption, reply_to, metadata=metadata)

    @staticmethod
    def _is_transient_typing_error(exc: Exception) -> bool:
        """Return True for Telegram typing errors worth cooling down."""
        if getattr(exc, "retry_after", None) is not None:
            return True
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
            return True
        text = str(exc).lower()
        if any(marker in text for marker in ("too many requests", "rate limit", "timed out", "timeout", "temporar")):
            return True
        return isinstance(exc, (OSError, TimeoutError, ConnectionError, asyncio.TimeoutError))

    def _record_typing_cooldown(self, chat_id: str, exc: Exception) -> None:
        """Suppress Telegram typing refreshes for this chat after transient failures."""
        if not hasattr(self, "_telegram_typing_cooldown_until"):
            self._telegram_typing_cooldown_until = {}
        retry_after = getattr(exc, "retry_after", None)
        try:
            delay = float(retry_after) if retry_after is not None else self._telegram_typing_cooldown_seconds
        except (TypeError, ValueError):
            delay = self._telegram_typing_cooldown_seconds
        self._telegram_typing_cooldown_until[str(chat_id)] = asyncio.get_running_loop().time() + max(1.0, min(delay, 300.0))

    def _typing_in_cooldown(self, chat_id: str) -> bool:
        if not hasattr(self, "_telegram_typing_cooldown_until"):
            self._telegram_typing_cooldown_until = {}
            self._telegram_typing_cooldown_seconds = 30.0
        until = self._telegram_typing_cooldown_until.get(str(chat_id))
        if until is None:
            return False
        if asyncio.get_running_loop().time() < until:
            return True
        self._telegram_typing_cooldown_until.pop(str(chat_id), None)
        return False

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Send typing indicator."""
        if not self._bot or self._typing_in_cooldown(chat_id):
            return
        _is_dm_topic: bool = False
        message_thread_id: Optional[int] = None

        async def _action(**kw) -> None:
            await self._bot.send_chat_action(chat_id=normalize_telegram_chat_id(chat_id), action="typing", **kw)
            self._telegram_typing_cooldown_until.pop(str(chat_id), None)
        try:
            _is_dm_topic = self._dm_topic_fallback(metadata)
            message_thread_id = self._message_thread_id_for_typing(self._metadata_thread_id(metadata))
            await _action(message_thread_id=message_thread_id)
        except Exception as e:
            # DM topic lanes: Telegram may reject message_thread_id — retry without it so the indicator at
            # least appears in the main DM view.
            if _is_dm_topic and message_thread_id is not None:
                try:
                    await _action()
                    return
                except Exception as fallback_exc:
                    if self._is_transient_typing_error(fallback_exc):
                        self._record_typing_cooldown(chat_id, fallback_exc)
            elif self._is_transient_typing_error(e):
                self._record_typing_cooldown(chat_id, e)
            logger.debug("[%s] Failed to send Telegram typing indicator: %s", self.name, _redact_telegram_error_text(e), exc_info=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a Telegram chat."""
        if not self._bot:
            return {"name": "Unknown", "type": "dm"}
        try:
            chat = await self._bot.get_chat(normalize_telegram_chat_id(chat_id))
            chat_type = "dm"
            if chat.type == ChatType.GROUP:
                chat_type = "group"
            elif chat.type == ChatType.SUPERGROUP:
                chat_type = "forum" if chat.is_forum else "group"
            elif chat.type == ChatType.CHANNEL:
                chat_type = "channel"
            return {
                "name": chat.title or chat.full_name or str(chat_id), "type": chat_type, "username": chat.username,
                "is_forum": getattr(chat, "is_forum", False)}
        except Exception as e:
            logger.error("[%s] Failed to get Telegram chat info for %s: %s", self.name, chat_id, _redact_telegram_error_text(
                e), exc_info=True)
            return {"name": str(chat_id), "type": "dm", "error": str(e)}

    def format_message(self, content: str) -> str:
        """Convert standard markdown to Telegram MarkdownV2: code is stashed behind placeholders first (never
        modified), markdown constructs become MarkdownV2 syntax, everything else is escaped."""
        if not content:
            return content
        placeholders: dict = {}
        counter = [0]

        def _ph(value: str) -> str:
            """Stash *value* behind a placeholder token that survives escaping."""
            key = f"\x00PH{counter[0]}\x00"
            counter[0] += 1
            placeholders[key] = value
            return key

        def _ph_wrap(open_: str, close: str):
            return lambda m: _ph(f"{open_}{_escape_mdv2(m.group(1))}{close}")

        # 0) GFM pipe tables → Telegram-friendly row groups, before the MarkdownV2 conversions.
        text = _wrap_markdown_tables(content)
        # 1) Protect fenced code blocks; per MarkdownV2 spec \ and ` inside pre/code must be escaped.
        def _protect_fenced(m):
            raw = m.group(0)
            open_end = raw.index('\n') + 1 if '\n' in raw[3:] else 3  # opening ``` (+ optional language)
            body = raw[open_end:][:-3].replace('\\', '\\\\').replace('`', '\\`')
            return _ph(raw[:open_end] + body + '```')

        text = re.sub(r'(```(?:[^\n]*\n)?[\s\S]*?```)', _protect_fenced, text)
        # 2) Protect inline code; escape \ inside it per MarkdownV2 spec.
        text = re.sub(r'(`[^`]+`)', lambda m: _ph(m.group(0).replace('\\', '\\\\')), text)
        # 3) Links: escape display text; inside the URL only ')' and '\' need escaping.
        def _convert_link(m):
            url = m.group(2).replace('\\', '\\\\').replace(')', '\\)')
            return _ph(f'[{_escape_mdv2(m.group(1))}]({url})')

        text = re.sub(r'\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)', _convert_link, text)
        # 4) Headers (## Title) → bold *Title*, stripping redundant ** inside the header
        def _convert_header(m):
            inner = re.sub(r'\*\*(.+?)\*\*', r'\1', m.group(1).strip())
            return _ph(f'*{_escape_mdv2(inner)}*')

        text = re.sub(r'^#{1,6}\s+(.+)$', _convert_header, text, flags=re.MULTILINE)
        # 5) Bold **text** → *text*; 6) Italic *text* → _text_ ([^*\n]+ keeps matches on one line, or *
        # bullet lists corrupt); 7) Strikethrough ~~text~~ → ~text~; 8) Spoiler ||text|| kept as-is.
        text = re.sub(r'\*\*(.+?)\*\*', _ph_wrap('*', '*'), text)
        text = re.sub(r'\*([^*\n]+)\*', _ph_wrap('_', '_'), text)
        text = re.sub(r'~~(.+?)~~', _ph_wrap('~', '~'), text)
        text = re.sub(r'\|\|(.+?)\|\|', _ph_wrap('||', '||'), text)
        # 9) Blockquotes: protect leading > from escaping; expandable quotes (**> starts, trailing || ends).
        def _convert_blockquote(m):
            prefix, content = m.group(1), m.group(2)  # prefix: >, >>, >>>, **>, **>> …
            if prefix.startswith('**') and content.endswith('||'):
                return _ph(f'{prefix} {_escape_mdv2(content[:-2])}||')
            return _ph(f'{prefix} {_escape_mdv2(content)}')

        text = re.sub(r'^((?:\*\*)?>{1,3}) (.+)$', _convert_blockquote, text, flags=re.MULTILINE)
        # 10) Escape remaining special characters in plain text
        text = _escape_mdv2(text)
        # 11) Restore placeholders in reverse insertion order so nested placeholders resolve.
        for key in reversed(list(placeholders.keys())):
            text = text.replace(key, placeholders[key])
        # 12) Safety net: escape bare ( ) { } that slipped through, but never inside ``` or ` spans.
        _safe_parts = []
        for _idx, _seg in enumerate(re.split(r'(```[\s\S]*?```|`[^`]+`)', text)):
            if _idx % 2 == 1:
                _safe_parts.append(_seg)  # inside code — untouched
            else:
                _safe_parts.append(re.sub(r'[(){}]', lambda m, _seg=_seg: TelegramAdapter._escape_bare_bracket(m, _seg), _seg))
        return ''.join(_safe_parts)

    @staticmethod
    def _escape_bare_bracket(m, seg: str) -> str:
        """Escape a bare ( ) { } unless it is already escaped or delimits a ``[text](url)`` link."""
        s = m.start()
        ch = m.group(0)
        if s > 0 and seg[s - 1] == '\\':  # already escaped
            return ch
        if ch == '(' and s > 0 and seg[s - 1] == ']':  # opens a link [text](url)
            return ch
        if ch == ')':  # closes a link URL? walk back matching depth
            before = seg[:s]
            if '](http' in before or '](' in before:
                depth = 0
                for j in range(s - 1, max(s - 2000, -1), -1):
                    if seg[j] == '(':
                        depth -= 1
                        if depth < 0:
                            if j > 0 and seg[j - 1] == ']':
                                return ch
                            break
                    elif seg[j] == ')':
                        depth += 1
        return '\\' + ch

    # ── Group mention gating ──────────────────────────────────────────────

    def _extra_bool(self, key: str, env_name: str, default: str, *fallback_keys: str) -> bool:
        """Boolean gate from ``config.extra[key]`` (then ``fallback_keys``), else env var."""
        configured = self.config.extra.get(key)
        for alt in fallback_keys:
            if configured is None:
                configured = self.config.extra.get(alt)
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return os.getenv(env_name, default).lower() in {"true", "1", "yes", "on"}

    def _extra_str_set(self, key: str, env_name: str) -> set[str]:
        """Comma/list allowlist from ``config.extra[key]``, else the profile-scoped env var."""
        raw = self.config.extra.get(key)
        if raw is None:
            raw = _scoped_gate_env(env_name)
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _telegram_require_mention(self) -> bool:
        """Return whether group chats should require an explicit bot trigger."""
        return self._extra_bool("require_mention", "TELEGRAM_REQUIRE_MENTION", "false")

    def _telegram_observe_unmentioned_group_messages(self) -> bool:
        """Store skipped unmentioned group messages as context (observe chatter, dispatch only when addressed)."""
        return self._extra_bool(
            "observe_unmentioned_group_messages", "TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES", "false",
            "ingest_unmentioned_group_messages")

    def _telegram_guest_mode(self) -> bool:
        """Return whether non-allowlisted groups may trigger via direct @mention."""
        return self._extra_bool("guest_mode", "TELEGRAM_GUEST_MODE", "false")

    def _telegram_exclusive_bot_mentions(self) -> bool:
        """Return whether explicit @...bot mentions exclusively route group messages."""
        return self._extra_bool("exclusive_bot_mentions", "TELEGRAM_EXCLUSIVE_BOT_MENTIONS", "true")

    def _telegram_free_response_chats(self) -> set[str]:
        return self._extra_str_set("free_response_chats", "TELEGRAM_FREE_RESPONSE_CHATS")

    def _telegram_free_response_topics(self) -> set[str]:
        """Topic-level free-response entries as ``<chat_id>:<thread_id>`` (General topic = ``1``)."""
        return self._extra_str_set("free_response_topics", "TELEGRAM_FREE_RESPONSE_TOPICS")

    def _telegram_is_free_response_topic(self, message: Message) -> bool:
        """True when the message's chat/topic pair is in ``free_response_topics``."""
        topics = self._telegram_free_response_topics()
        if not topics:
            return False
        chat_id = self._chat_id_str(message)
        if not chat_id:
            return False
        return f"{chat_id}:{self._topic_id_or_general(self._effective_message_thread_id(message))}" in topics

    def _telegram_allowed_chats(self) -> set[str]:
        """Group chat IDs the bot responds in (non-empty: others need ``guest_mode`` + @mention; DMs never
        filtered; empty = no restriction)."""
        return self._extra_str_set("allowed_chats", "TELEGRAM_ALLOWED_CHATS")

    def _telegram_group_allowed_chats(self) -> set[str]:
        """Return Telegram chats authorized at group scope."""
        return self._extra_str_set("group_allowed_chats", "TELEGRAM_GROUP_ALLOWED_CHATS")

    def _telegram_observe_allowed_chats(self) -> set[str]:
        """Chats where observed group context may use a shared source: ``group_allowed_chats`` ∩
        ``allowed_chats`` (when set)."""
        group_allowed = self._telegram_group_allowed_chats()
        if not group_allowed:
            return set()
        response_allowed = self._telegram_allowed_chats()
        return group_allowed & response_allowed if response_allowed else group_allowed

    def _telegram_allowed_topics(self) -> set[str]:
        """Forum topic IDs this bot handles (non-empty: other topics ignored; DMs never filtered; missing
        ``message_thread_id`` == General topic ``1``)."""
        return self._extra_str_set("allowed_topics", "TELEGRAM_ALLOWED_TOPICS")

    def _telegram_ignored_threads(self) -> set[int]:
        raw = self.config.extra.get("ignored_threads")
        if raw is None:
            raw = _scoped_gate_env("TELEGRAM_IGNORED_THREADS")
        ignored: set[int] = set()
        for value in (raw if isinstance(raw, list) else str(raw).split(",")):
            text = str(value).strip()
            if not text:
                continue
            try:
                ignored.add(int(text))
            except (TypeError, ValueError):
                logger.warning("[%s] Ignoring invalid Telegram thread id: %r", self.name, value)
        return ignored

    def _compile_mention_patterns(self) -> List[re.Pattern]:
        """Compile optional regex wake-word patterns for group triggers."""
        patterns = self.config.extra.get("mention_patterns")
        if patterns is None:
            raw = os.getenv("TELEGRAM_MENTION_PATTERNS", "").strip()
            if raw:
                try:
                    loaded = json.loads(raw)
                except Exception:
                    loaded = [part.strip() for part in raw.splitlines() if part.strip()]
                    if not loaded:
                        loaded = [part.strip() for part in raw.split(",") if part.strip()]
                patterns = loaded
        if patterns is None:
            return []  # before touching ``self.name``: tests build bare adapters via object.__new__
        return compile_mention_patterns(patterns, log_prefix=self.name, platform_label="telegram", display_label="Telegram", logger_=logger)

    @staticmethod
    def _chat_type_str(chat) -> str:
        """PTB enum or plain-string ``chat.type`` → bare lowercase name (``supergroup``)."""
        return str(getattr(chat, "type", "")).split(".")[-1].lower() if chat else ""

    @staticmethod
    def _chat_id_str(message) -> str:
        return str(getattr(getattr(message, "chat", None), "id", ""))

    @classmethod
    def _topic_id_or_general(cls, thread_id) -> str:
        return str(thread_id) if thread_id is not None else cls._GENERAL_TOPIC_THREAD_ID

    def _is_group_chat(self, message: Message) -> bool:
        chat = getattr(message, "chat", None)
        return bool(chat) and self._chat_type_str(chat) in {"group", "supergroup"}

    @classmethod
    def _effective_message_thread_id(cls, message: Message) -> Optional[str]:
        """Routable thread id: forum General-topic messages arrive with ``message_thread_id=None`` but
        Telegram addresses that topic as ``1``; plain group/DM replies carry a reply-UI anchor that is NOT
        a routing id. Gating, skill binding and outbound routing must all agree on this value."""
        chat = getattr(message, "chat", None)
        chat_type = cls._chat_type_str(chat)
        raw = getattr(message, "message_thread_id", None)
        is_topic_message = bool(getattr(message, "is_topic_message", False))
        is_group = chat_type in ("group", "supergroup")
        is_forum_group = is_group and getattr(chat, "is_forum", False) is True
        if raw is not None:
            if is_forum_group or (is_group and is_topic_message) or (chat_type == "private" and is_topic_message):
                return str(raw)
            return None
        return cls._GENERAL_TOPIC_THREAD_ID if is_forum_group else None

    # Decides only whether a FOREIGN @handle is bot-shaped; our own handle is matched by identity, never
    # shape (collectible/Fragment bot usernames need not end in "bot").
    _FOREIGN_BOT_HANDLE_RE = re.compile(r"[a-z0-9_]{2,29}bot", re.IGNORECASE)
    _BOT_IDENTITY_TTL_SECONDS = 300.0  # how long an observed identity is trusted before re-check

    def _current_bot_username(self) -> str:
        """This bot's live @username (lowercased, no ``@``): the last observed handle beats PTB's
        ``get_me()`` cache, which keeps a stale handle after a BotFather rename."""
        observed = getattr(self, "_bot_username_observed", None)
        if observed:
            return observed
        return (getattr(self._bot, "username", None) or "").lstrip("@").lower()

    def _note_bot_username(self, username: Optional[str]) -> None:
        """Record the bot's current @username, logging real renames."""
        handle = (username or "").lstrip("@").lower()
        if not handle:
            return
        previous = getattr(self, "_bot_username_observed", None)
        if previous == handle:
            return
        self._bot_username_observed = handle
        self._bot_identity_checked_at = time.monotonic()
        if previous:
            logger.info(
                "[%s] Telegram bot username changed: @%s -> @%s (mention routing now follows the new handle)", self.name, previous, handle)

    def _observe_bot_identity_from_message(self, message: Message) -> None:
        """Learn our own handle from a message Telegram says we authored (own messages and
        ``reply_to_message``); only trusted when the user id matches this bot."""
        bot_id = getattr(self._bot, "id", None)
        if bot_id is None:
            return
        for candidate in (getattr(message, "from_user", None), getattr(getattr(message, "reply_to_message", None), "from_user", None)):
            if candidate is not None and getattr(candidate, "id", None) == bot_id:
                self._note_bot_username(getattr(candidate, "username", None))

    def _bot_identity_is_fresh(self) -> bool:
        """True when identity was re-read within the TTL. ``None`` (never checked) is always stale — do
        not fold it into ``0.0``: monotonic clocks have an arbitrary epoch."""
        checked_at = getattr(self, "_bot_identity_checked_at", None)
        return checked_at is not None and (time.monotonic() - checked_at) < self._BOT_IDENTITY_TTL_SECONDS

    async def _refresh_bot_identity(self, *, force: bool = False) -> None:
        """Re-read bot identity when the cache may be stale (``get_me()`` rewrites PTB's ``Bot._bot_user``
        in place). Best-effort: a failed probe keeps the last known handle."""
        bot = self._bot
        if bot is None or not callable(getattr(bot, "get_me", None)):
            return
        if not force and self._bot_identity_is_fresh():
            return
        try:
            me = await asyncio.wait_for(bot.get_me(), self._BOT_IDENTITY_PROBE_TIMEOUT)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "[%s] Telegram identity refresh failed (keeping @%s): %s", self.name, self._current_bot_username() or "unknown", exc)
            return
        self._bot_identity_checked_at = time.monotonic()
        self._note_bot_username(getattr(me, "username", None))

    _BOT_IDENTITY_PROBE_TIMEOUT = 15.0

    def _is_reply_to_bot(self, message: Message) -> bool:
        if not self._bot or not getattr(message, "reply_to_message", None):
            return False
        reply_user = getattr(message.reply_to_message, "from_user", None)
        return bool(reply_user and getattr(reply_user, "id", None) == getattr(self._bot, "id", None))

    @staticmethod
    def _entity_sources(message: Message):
        """``(text, entities)`` pairs for the message text and caption."""
        yield getattr(message, "text", None) or "", getattr(message, "entities", None) or []
        yield getattr(message, "caption", None) or "", getattr(message, "caption_entities", None) or []

    @staticmethod
    def _entity_type(entity) -> str:
        return str(getattr(entity, "type", "")).split(".")[-1].lower()

    @classmethod
    def _entity_span(cls, source_text: str, entity) -> Optional[str]:
        """The entity's text, or None when its offsets are unusable."""
        # Telegram's official group-disambiguation form for slash commands (``/cmd@botname``) is emitted as
        # a single ``bot_command`` entity covering the whole span — there is no accompanying ``mention``
        # entity. Treat it as a direct address to this bot when the ``@botname`` suffix matches. This is the
        # form Telegram's own command menu autocomplete produces in groups, so dropping it at the mention
        # gate would break /new, /reset, /help, ... for every group that has ``require_mention`` enabled
        # (#15415).
        offset = int(getattr(entity, "offset", -1))
        length = int(getattr(entity, "length", 0))
        if offset < 0 or length <= 0:
            return None
        return cls._telegram_entity_text(source_text, offset, length)

    @classmethod
    def _extract_bot_mention_usernames(cls, message: Message, self_username: str = "") -> set[str]:
        """Explicit bot usernames mentioned in text/captions: foreign handles count only when bot-shaped
        (``...bot``), ``self_username`` opts our OWN handle in regardless of shape. Entity mentions are
        authoritative; the raw-text fallback is deliberately narrow."""
        mentioned_bot_usernames: set[str] = set()
        own = (self_username or "").lstrip("@").lower()

        def _is_bot_handle(handle: str) -> bool:
            if not handle:
                return False
            if own and handle == own:
                return True
            return bool(cls._FOREIGN_BOT_HANDLE_RE.fullmatch(handle))

        for source_text, entities in cls._entity_sources(message):
            for entity in entities:
                entity_type = cls._entity_type(entity)
                if entity_type not in {"mention", "bot_command"}:
                    continue
                entity_text = cls._entity_span(source_text, entity)
                if entity_text is None:
                    continue
                entity_text = entity_text.strip()
                if entity_type == "mention":
                    handle = entity_text.lstrip("@").lower()
                    if _is_bot_handle(handle):
                        mentioned_bot_usernames.add(handle)
                    continue
                # /cmd@botname is one bot_command entity; its suffix is an explicit bot address.
                at_index = entity_text.find("@")
                if at_index < 0:
                    continue
                command_target = entity_text[at_index + 1:].strip().lower()
                if _is_bot_handle(command_target):
                    mentioned_bot_usernames.add(command_target)
        # Entity-less fallback only: if Telegram supplied entities, trust them (no URL/code rescue).
        for raw_text, entities in cls._entity_sources(message):
            if not raw_text or entities:
                continue
            for match in re.finditer(r"(?i)(?<![A-Za-z0-9_`/])@([A-Za-z0-9_]{2,31})\b", raw_text):
                handle = match.group(1).lower()
                if _is_bot_handle(handle):
                    mentioned_bot_usernames.add(handle)
        return mentioned_bot_usernames

    @staticmethod
    def _telegram_entity_text(source_text: str, offset: int, length: int) -> str:
        """Return a Telegram entity span using UTF-16 code-unit offsets."""
        if offset < 0 or length <= 0:
            return ""
        try:
            return source_text.encode("utf-16-le")[offset * 2:(offset + length) * 2].decode("utf-16-le")
        except UnicodeDecodeError:
            return ""

    def _message_mentions_bot(self, message: Message) -> bool:
        if not self._bot:
            return False
        bot_username = self._current_bot_username()
        bot_id = getattr(self._bot, "id", None)
        expected = f"@{bot_username}" if bot_username else None
        # Server-side MessageEntity values are authoritative: raw substrings like "foo@hermes_bot.example"
        # or handles inside URLs/code are not mentions.
        for source_text, entities in self._entity_sources(message):
            for entity in entities:
                entity_type = self._entity_type(entity)
                if entity_type == "mention" and expected:
                    span = self._entity_span(source_text, entity)
                    if span is not None and span.strip().lower() == expected:
                        return True
                elif entity_type == "text_mention":
                    user = getattr(entity, "user", None)
                    if user and getattr(user, "id", None) == bot_id:
                        return True
                elif entity_type == "bot_command" and expected:
                    # ``/cmd@botname`` (what the group command menu produces) must count as a direct address.
                    command_text = self._entity_span(source_text, entity)
                    if command_text is None:
                        continue
                    at_index = command_text.find("@")
                    if at_index >= 0 and command_text[at_index:].strip().lower() == expected:
                        return True
        if bot_username:
            return bot_username in self._extract_bot_mention_usernames(message, bot_username)
        return False

    def _schedule_bot_identity_recheck(self) -> None:
        """Fire a TTL-guarded identity refresh in the background when routing is about to discard a
        message naming other bots but not us (the symptom of a stale handle after a rename)."""
        existing = getattr(self, "_bot_identity_refresh_task", None)
        if (existing is not None and not existing.done()) or self._bot_identity_is_fresh():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._refresh_bot_identity())
        self._bot_identity_refresh_task = task
        tracked = getattr(self, "_background_tasks", None)
        if isinstance(tracked, set):
            tracked.add(task)
            task.add_done_callback(tracked.discard)

    def _explicit_bot_mentions_exclude_self(self, message: Message) -> bool:
        """True when explicit bot handles target other bots, not this one (``@bot3 hi @bot4`` must not
        wake ``@bot1`` via reply/wake-word fallbacks)."""
        if not self._bot:
            return False
        bot_username = self._current_bot_username()
        if not bot_username:
            return False
        mentioned_bot_usernames = self._extract_bot_mention_usernames(message, bot_username)
        excludes_self = bool(mentioned_bot_usernames) and bot_username not in mentioned_bot_usernames
        if excludes_self:
            # Either truly for another bot, or our handle is stale after a rename — re-check out of band.
            self._schedule_bot_identity_recheck()
        return excludes_self

    def _message_matches_mention_patterns(self, message: Message) -> bool:
        if not self._mention_patterns:
            return False
        return any(
            pattern.search(candidate)
            for candidate in (getattr(message, "text", None), getattr(message, "caption", None)) if candidate
            for pattern in self._mention_patterns)

    def _is_guest_mention(self, message: Message) -> bool:
        """Guest-mode bypass: explicit bot mention (caller already verified group chat)."""
        return self._telegram_guest_mode() and self._message_mentions_bot(message)

    def _clean_bot_trigger_text(self, text: Optional[str]) -> Optional[str]:
        bot_username = self._current_bot_username()
        if not text or not bot_username:
            return text
        cleaned = re.sub(rf"(?i)@{re.escape(bot_username)}\b[,:\-]*\s*", "", text).strip()
        return cleaned or text

    def _topic_gates_pass(self, thread_id, *, warn_non_numeric: bool) -> Optional[bool]:
        """``allowed_topics`` / ``ignored_threads`` gates; False = blocked, None = undecided."""
        allowed_topics = self._telegram_allowed_topics()
        if allowed_topics and self._topic_id_or_general(thread_id) not in allowed_topics:
            return False
        if thread_id is not None:
            try:
                if int(thread_id) in self._telegram_ignored_threads():
                    return False
            except (TypeError, ValueError):
                if not warn_non_numeric:
                    return False
                logger.warning("[%s] Ignoring non-numeric Telegram message_thread_id: %r", self.name, thread_id)
        return None

    def _should_observe_unmentioned_group_message(self, message: Message) -> bool:
        """Return True when a group message should be stored but not dispatched."""
        if self._is_own_message(message) or not self._telegram_observe_unmentioned_group_messages() or not self._is_group_chat(message):
            return False
        if self._topic_gates_pass(getattr(message, "message_thread_id", None), warn_non_numeric=False) is False:
            return False
        chat_id_str = self._chat_id_str(message)
        if self._telegram_exclusive_bot_mentions() and self._explicit_bot_mentions_exclude_self(message):
            return False
        # Observed context is shared at chat/topic scope, so require an explicit chat allowlist.
        allowed = self._telegram_observe_allowed_chats()
        if not allowed or chat_id_str not in allowed:
            return False
        # Only observe messages the require_mention gate would skip.
        if chat_id_str in self._telegram_free_response_chats() or self._telegram_is_free_response_topic(message):
            return False
        if not self._telegram_require_mention() or self._is_reply_to_bot(message) or self._message_mentions_bot(message):
            return False
        return not self._message_matches_mention_patterns(message)

    def _telegram_group_observe_shared_source(self, source):
        """Return a chat/topic-scoped source for observed Telegram group context."""
        return dataclasses.replace(source, user_id=None, user_name=None, user_id_alt=None)

    def _telegram_group_observe_attributed_text(self, event: MessageEvent) -> str:
        user_id = event.source.user_id or "unknown"
        return f"[{event.source.user_name or user_id}|{user_id}]\n{event.text or ''}"

    def _telegram_group_observe_channel_prompt(self) -> str:
        username = self._current_bot_username() or "unknown"
        bot_id = getattr(getattr(self, "_bot", None), "id", None) or "unknown"
        return (
            "You are handling a Telegram group chat message.\n"
            f"- Your identity: user_id={bot_id}, @-mention name in this group=@{username}\n"
            "- observed Telegram group context may be provided in a separate context-only block "
            "before the current message; it is not necessarily addressed to you.\n"
            "- Treat only the current new message as a request explicitly directed at you, "
            "and use observed context only when the current message asks for it.")

    def _apply_telegram_group_observe_attribution(self, event: MessageEvent) -> MessageEvent:
        """Align triggered group turns with observed-history attribution."""
        if not self._telegram_observe_unmentioned_group_messages():
            return event
        raw_message = getattr(event, "raw_message", None)
        if not raw_message or not self._is_group_chat(raw_message):
            return event
        allowed = self._telegram_observe_allowed_chats()
        if not allowed or self._chat_id_str(raw_message) not in allowed:
            return event
        observe_prompt = self._telegram_group_observe_channel_prompt()
        channel_prompt = f"{event.channel_prompt}\n\n{observe_prompt}" if event.channel_prompt else observe_prompt
        if event.message_type == MessageType.COMMAND:
            # Commands keep the original source (user_id) so _check_slash_access can identify the sender.
            return dataclasses.replace(event, channel_prompt=channel_prompt)
        return dataclasses.replace(
            event, text=self._telegram_group_observe_attributed_text(event),
            source=self._telegram_group_observe_shared_source(event.source), channel_prompt=channel_prompt)

    def _media_message_type(self, msg: Message) -> MessageType:
        """Classify a Telegram media message into a MessageType (first present attachment wins)."""
        for attr, mtype in (
            ("sticker", MessageType.STICKER), ("photo", MessageType.PHOTO), ("video", MessageType.VIDEO),
            ("audio", MessageType.AUDIO), ("voice", MessageType.VOICE)):
            if getattr(msg, attr):
                return mtype
        return MessageType.DOCUMENT

    _CACHED_KIND_TO_MESSAGE_TYPE = {"image": MessageType.PHOTO, "video": MessageType.VIDEO, "audio": MessageType.AUDIO}

    async def _download_observed_media(self, msg: Any, what: str):
        """Download ``msg``'s attachment into the media cache (bounded by ``_max_doc_bytes``). Returns ``(status, cached)``:
        ``"none"``, ``"oversized"`` (cached = raw file_size), ``"failed"``, ``"unreadable"`` or ``"ok"``."""
        from gateway.platforms.base import cache_media_bytes_async
        source, filename, mime, kind = self._observed_media_source(msg)
        if source is None:
            return "none", None
        file_size = getattr(source, "file_size", None)
        if not (0 < self._int_or_zero(file_size) <= getattr(self, "_max_doc_bytes", 20 * 1024 * 1024)):
            return "oversized", file_size
        try:
            file_obj = await source.get_file()
            data = bytes(await file_obj.download_as_bytearray())
            if not filename:
                filename = os.path.basename(getattr(file_obj, "file_path", "") or "")
            cached = await cache_media_bytes_async(data, filename=filename, mime_type=mime, default_kind=kind)
        except Exception as exc:
            logger.warning("[Telegram] Failed to cache %s: %s", what, _redact_telegram_error_text(exc), exc_info=True)
            return "failed", None
        if cached is None:
            return "unreadable", None
        return "ok", cached

    async def _cache_observed_media(self, msg: Message, event: MessageEvent) -> None:
        """Cache an unmentioned group attachment and annotate the observed text; oversized or unsupported
        attachments are noted in the transcript without downloading."""
        status, cached = await self._download_observed_media(msg, "observed group media")
        if status == "oversized":
            limit_mb = getattr(self, "_max_doc_bytes", 20 * 1024 * 1024) // (1024 * 1024)
            event.text = self._append_observed_note(
                event.text, f"[Observed Telegram attachment too large or unverifiable. Maximum: {limit_mb} MB.]")
            logger.info("[Telegram] Observed group attachment skipped (size=%s)", cached)
            return
        if status == "unreadable":  # only images that fail validation reach here
            event.text = self._append_observed_note(event.text, "[Observed Telegram attachment could not be read, not cached.]")
            return
        if status == "ok":
            event.media_urls = []
            event.media_types = []
            self._attach_cached(event, cached, cached.context_note(), "[Telegram] Cached observed group %s at %s")

    def _attach_cached(self, event: MessageEvent, cached, note: str, log_fmt: str) -> None:
        """Append a cached attachment to the event (message type follows the kind only for the first one)."""
        event.media_urls.append(cached.path)
        event.media_types.append(cached.media_type)
        if len(event.media_urls) == 1 and cached.kind in self._CACHED_KIND_TO_MESSAGE_TYPE:
            event.message_type = self._CACHED_KIND_TO_MESSAGE_TYPE[cached.kind]
        event.text = self._append_observed_note(event.text, note)
        logger.info(log_fmt, cached.kind, cached.path)

    async def _cache_replied_media(self, msg: Any, event: MessageEvent) -> None:
        """Cache media from the message this turn replies to, if any."""
        reply_msg = getattr(msg, "reply_to_message", None)
        if reply_msg is None:
            return
        status, cached = await self._download_observed_media(reply_msg, "replied-to media")
        if status == "ok":
            self._attach_cached(
                event, cached, f"[Replied-to {cached.kind} '{cached.display_name}' saved at: {cached.path}]",
                "[Telegram] Cached replied-to %s at %s")

    def _observed_media_source(self, msg: Message):
        """Return (telegram_file_source, filename, mime, default_kind) or Nones."""
        if msg.photo:
            return msg.photo[-1], "", "", "image"
        if msg.video:
            return msg.video, "", "video/mp4", "video"
        if msg.voice:
            return msg.voice, "voice.ogg", "audio/ogg", "audio"
        if msg.audio:
            return msg.audio, getattr(msg.audio, "file_name", "") or "", "", "audio"
        if msg.document:
            doc = msg.document
            return doc, doc.file_name or "", (doc.mime_type or "").lower(), None
        return None, "", "", None

    @staticmethod
    def _append_observed_note(existing: Optional[str], note: str) -> str:
        if not note:
            return existing or ""
        return f"{existing}\n\n{note}" if existing else note

    async def _surface_media_cache_failure(
        self, msg: Message, event: MessageEvent, kind: str, exc: Exception, display_name: Optional[str] = None) -> None:
        """Surface a failed media download to BOTH the user (reply asking to retry) and the agent (observed
        note) — otherwise the turn dispatches silently with empty media_urls.

        This (1) replies to the user in Telegram so they know to retry, and (2) appends an agent-visible
        notice to event.text via the existing observed-note channel so the agent knows an attachment was
        attempted and failed — never a silent empty turn. No new event fields (the structured-event refactor
        is out of scope per #23045).
        """
        named = f" ({display_name})" if display_name else ""
        try:
            await msg.reply_text(
                f"\u26a0\ufe0f Couldn't download your {kind}{named} ({exc.__class__.__name__}). Please try sending it again.")
        except Exception as reply_err:
            logger.warning("[Telegram] Failed to notify user about %s cache failure: %s", kind, reply_err, exc_info=True)
        event.text = self._append_observed_note(
            event.text,
            f"[The user attempted to send a {kind}{named} but it could not be downloaded ({exc.__class__.__name__}); they have been asked to retry.]",
       )

    def _observe_unmentioned_group_message(
        self, message: Message, msg_type: MessageType, update_id: Optional[int] = None, event: Optional[MessageEvent] = None) -> None:
        """Append skipped group chatter to the target session without dispatching."""
        store = getattr(self, "_session_store", None)
        if not store:
            return
        adapter_name = getattr(self, "name", "telegram")
        try:
            event = event or self._build_message_event(message, msg_type, update_id=update_id)
            session_entry = store.get_or_create_session(self._telegram_group_observe_shared_source(event.source))
            entry = {
                "role": "user", "content": self._telegram_group_observe_attributed_text(event),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(), "observed": True}
            if event.message_id:
                entry["message_id"] = str(event.message_id)
            store.append_to_transcript(session_entry.session_id, entry)
            logger.info(
                "[%s] Telegram group message observed (no bot trigger): chat=%s from=%s", adapter_name,
                getattr(getattr(message, "chat", None), "id", "unknown"), event.source.user_id or "unknown")
        except Exception as exc:
            logger.warning("[%s] Failed to observe Telegram group message: %s", adapter_name, exc)

    def _is_own_message(self, message: Message) -> bool:
        """True when sent by this bot itself (echoed getUpdates must not count as incoming unread)."""
        if not self._bot:
            return False
        from_user = getattr(message, "from_user", None)
        if from_user is None:
            return False
        bot_id = getattr(self._bot, "id", None)
        user_id = getattr(from_user, "id", None)
        return bot_id is not None and user_id is not None and bot_id == user_id

    def _should_process_message(self, message: Message, *, is_command: bool = False) -> bool:
        """Apply Telegram group trigger rules: DMs unrestricted; group messages pass ``allowed_chats`` (hard gate; only
        the ``guest_mode`` @mention bypass crosses it) and then any of free_response chat/topic, ``require_mention``
        off, reply to the bot, @mention (incl. ``/cmd@botname``), or a wake-word match."""
        # Learn the live handle BEFORE any mention gate routes on it, then drop our own echoed messages.
        # Filter out the bot's own messages (returned by getUpdates in some environments like
        # groups/supergroups where the bot can see its own messages). Without this, outbound messages are
        # counted as incoming unread in the Hermes inbox (#52363). Otherwise a BotFather rename leaves the
        # stale handle in place and the exclusive-mention gate reads a message addressed to us as one
        # addressed to some other bot.
        self._observe_bot_identity_from_message(message)
        if self._is_own_message(message):
            return False
        if not self._is_group_chat(message):
            return True
        thread_id = self._effective_message_thread_id(message)
        if self._topic_gates_pass(thread_id, warn_non_numeric=True) is False:
            return False
        chat_id_str = self._chat_id_str(message)
        if self._telegram_exclusive_bot_mentions() and self._explicit_bot_mentions_exclude_self(message):
            return False
        # Resolve once; _message_mentions_bot is not re-called below in guest mode.
        guest_mention = self._is_guest_mention(message)
        # allowed_chats whitelist: outside chats pass only via the guest-mode explicit mention.
        allowed = self._telegram_allowed_chats()
        if allowed and chat_id_str not in allowed:
            return guest_mention
        if guest_mention or chat_id_str in self._telegram_free_response_chats() or self._telegram_is_free_response_topic(message):
            return True
        if not self._telegram_require_mention() or self._is_reply_to_bot(message):
            return True
        if not self._telegram_guest_mode() and self._message_mentions_bot(message):
            return True
        return self._message_matches_mention_patterns(message)

    async def _ensure_forum_commands(self, message) -> None:
        """Lazy-register bot commands for forum supergroups (topics don't inherit AllGroupChats scope;
        Telegram resolves via BotCommandScopeChat)."""
        async with self._forum_lock:
            try:
                chat = getattr(message, "chat", None)
                if not chat or not getattr(chat, "is_forum", False):
                    return
                chat_id = int(chat.id)
                if chat_id in self._forum_command_registered:
                    return
                from telegram import BotCommand, BotCommandScopeChat
                from hermes_cli.commands_platforms import telegram_menu_commands, telegram_menu_max_commands
                menu_commands, _ = telegram_menu_commands(max_commands=telegram_menu_max_commands())
                bot_commands = [BotCommand(name, desc) for name, desc in menu_commands]
                await self._bot.set_my_commands(bot_commands, scope=BotCommandScopeChat(chat_id=chat_id))
                self._forum_command_registered.add(chat_id)
                logger.info("[%s] Lazy-registered %d commands for forum chat %s", self.name, len(bot_commands), chat_id)
            except Exception as e:
                logger.warning("[%s] Forum command lazy-registration failed: %s", self.name, _redact_telegram_error_text(e))

    def _effective_update_message(self, update: Update) -> Optional[Message]:
        """Message-like payload for normal messages and channel posts (``update.channel_post``)."""
        return getattr(update, "effective_message", None) or getattr(update, "message", None)

    def _log_blocked_user(self, msg, *, level=logging.WARNING, what: str = "unauthorized user") -> None:
        logger.log(
            level, "[Telegram] Blocked %s %s in chat %s", what, getattr(getattr(msg, "from_user", None), "id", None),
            getattr(getattr(msg, "chat", None), "id", None))

    def _gate_or_observe(self, msg, update, msg_type: MessageType) -> bool:
        """Group trigger gate; observes unmentioned chatter when configured. True = proceed."""
        if self._should_process_message(msg):
            return True
        if self._should_observe_unmentioned_group_message(msg):
            self._observe_unmentioned_group_message(msg, msg_type, update_id=update.update_id)
        return False

    async def _build_triggered_event(self, msg, update, msg_type: MessageType) -> MessageEvent:
        """Event for an addressed text/command: trigger text cleaned, replied-to media cached, attribution applied."""
        event = self._build_message_event(msg, msg_type, update_id=update.update_id)
        event.text = self._clean_bot_trigger_text(event.text)
        await self._cache_replied_media(msg, event)
        return self._apply_telegram_group_observe_attribution(event)

    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text; buffers client-split chunks into one MessageEvent."""
        msg = self._effective_update_message(update)
        if not msg or not msg.text:
            return
        # Auth check first: blocked users must not reach batching, the observed transcript, or the agent.
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg)
            return
        if not self._gate_or_observe(msg, update, MessageType.TEXT):
            return
        await self._ensure_forum_commands(update.message)
        self._enqueue_text_event(await self._build_triggered_event(msg, update, MessageType.TEXT))

    async def _handle_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming command messages."""
        msg = self._effective_update_message(update)
        if not msg or not msg.text:
            return
        if not self._should_process_message(msg, is_command=True):
            return
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg)
            return
        await self._ensure_forum_commands(msg)
        event = await self._build_triggered_event(msg, update, MessageType.COMMAND)
        # A >4096-char command paste arrives as a near-limit COMMAND chunk plus TEXT continuations; dispatching
        # immediately would orphan them. Near-limit commands go through text batching.
        if len(event.text or "") >= self._SPLIT_THRESHOLD:
            self._enqueue_text_event(event)
            return
        await self.handle_message(event)

    async def _handle_location_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming location/venue pin messages."""
        msg = self._effective_update_message(update)
        if not msg:
            return
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg)
            return
        if not self._gate_or_observe(msg, update, MessageType.LOCATION):
            return
        venue = getattr(msg, "venue", None)
        location = getattr(venue, "location", None) if venue else getattr(msg, "location", None)
        if not location:
            return
        lat = getattr(location, "latitude", None)
        lon = getattr(location, "longitude", None)
        if lat is None or lon is None:
            return
        parts = ["[The user shared a location pin.]"]
        if venue:
            title = getattr(venue, "title", None)
            address = getattr(venue, "address", None)
            if title:
                parts.append(f"Venue: {title}")
            if address:
                parts.append(f"Address: {address}")
        parts += [
            f"latitude: {lat}", f"longitude: {lon}", f"Map: https://www.google.com/maps/search/?api=1&query={lat},{lon}",
            "Ask what they'd like to find nearby (restaurants, cafes, etc.) and any preferences."]
        event = self._build_message_event(msg, MessageType.LOCATION, update_id=update.update_id)
        event.text = "\n".join(parts)
        await self.handle_message(self._apply_telegram_group_observe_attribution(event))

    # -- Text message aggregation (handles Telegram client-side splits) --

    def _text_batch_key(self, event: MessageEvent) -> str:
        """Session-scoped batching key; topic recovery first so DM-topic batches coalesce on the recovered lane."""
        self._apply_topic_recovery(event)
        return super()._text_batch_key(event)

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Buffer a text chunk, or hold it while delayed delivery must be dropped."""
        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="text-enqueue")
            return
        super()._enqueue_text_event(event)

    async def _flush_buffered(self, pending: dict, tasks: dict, key: str, delay: float, where: str, log_fn=None) -> None:
        """Shared delayed-flush body: sleep, pop, hold if teardown started, else dispatch. A cancel after
        the pop but before durable dispatch re-holds the event (never lose it)."""
        current_task = asyncio.current_task()
        event = None
        try:
            await asyncio.sleep(delay)
            event = pending.pop(key, None)
            if not event:
                return
            if self._should_drop_delayed_delivery():
                self._hold_inbound_event(event, where=f"{where}-flush")
                event = None
                return
            if log_fn is not None:
                log_fn(event)
            await self.handle_message(event)
            event = None
        except asyncio.CancelledError:
            if event is not None:
                self._hold_inbound_event(event, where=f"{where}-flush-cancelled")
            raise
        finally:
            if tasks.get(key) is current_task:
                tasks.pop(key, None)

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text."""
        # Adaptive delay: near-split-point last chunk → long delay (continuation almost certain);
        # short/medium totals → capped fast delays; else configured cap (all min()'d with the operator cap).
        pending = self._pending_text_batches.get(key)
        last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
        total_len = len(getattr(pending, "text", "") or "") if pending else 0
        if last_len >= self._SPLIT_THRESHOLD:
            delay = self._text_batch_split_delay_seconds
        elif total_len <= self._TEXT_BATCH_FAST_LEN:
            delay = min(self._text_batch_delay_seconds, self._TEXT_BATCH_FAST_DELAY_S)
        elif total_len <= self._TEXT_BATCH_SHORT_LEN:
            delay = min(self._text_batch_delay_seconds, self._TEXT_BATCH_SHORT_DELAY_S)
        else:
            delay = self._text_batch_delay_seconds
        await self._flush_buffered(
            self._pending_text_batches, self._pending_text_batch_tasks, key, delay, "text",
            lambda ev: logger.info("[Telegram] Flushing text batch %s (%d chars)", key, len(ev.text or "")))

    # -- Photo batching --

    def _photo_batch_key(self, event: MessageEvent, msg: Message) -> str:
        """Return a batching key for Telegram photos/albums."""
        from gateway.session import build_session_key
        session_key = build_session_key(
            event.source, group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
            profile=self._session_key_profile(event.source))
        media_group_id = getattr(msg, "media_group_id", None)
        return f"{session_key}:album:{media_group_id}" if media_group_id else f"{session_key}:photo-burst"

    async def _flush_photo_batch(self, batch_key: str) -> None:
        """Send a buffered photo burst/album as a single MessageEvent."""
        await self._flush_buffered(
            self._pending_photo_batches, self._pending_photo_batch_tasks, batch_key, self._media_batch_delay_seconds, "photo",
            lambda ev: logger.info("[Telegram] Flushing photo batch %s with %d image(s)", batch_key, len(ev.media_urls)))

    def _merge_into_pending(self, pending: dict, key: str, event: MessageEvent) -> None:
        """Merge ``event`` into ``pending[key]`` (media + caption) or seed it."""
        existing = pending.get(key)
        if existing is None:
            pending[key] = event
            return
        existing.media_urls.extend(event.media_urls)
        existing.media_types.extend(event.media_types)
        if event.text:
            existing.text = self._merge_caption(existing.text, event.text)

    def _enqueue_photo_event(self, batch_key: str, event: MessageEvent) -> None:
        """Merge photo events into a pending batch and schedule flush."""
        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="photo-enqueue")
            return
        self._merge_into_pending(self._pending_photo_batches, batch_key, event)
        prior_task = self._pending_photo_batch_tasks.get(batch_key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_photo_batch_tasks[batch_key] = asyncio.create_task(self._flush_photo_batch(batch_key))

    async def _route_photo_event(self, msg, event: MessageEvent) -> None:
        """Album items debounce on media_group_id; singles go through the photo burst batcher."""
        media_group_id = getattr(msg, "media_group_id", None)
        if media_group_id:
            await self._queue_media_group_event(str(media_group_id), event)
        else:
            self._enqueue_photo_event(self._photo_batch_key(event, msg), event)

    @staticmethod
    def _ext_from_path(file_path: Optional[str], candidates, default: str) -> str:
        """First extension in ``candidates`` that ``file_path`` ends with (case-insensitive), else default."""
        if file_path:
            lowered = file_path.lower()
            for candidate in candidates:
                if lowered.endswith(candidate):
                    return candidate
        return default

    async def _cache_inbound_av(self, msg, event: MessageEvent, source: Any, label: str, kind: str, ext: str, mime: str) -> bool:
        """Download a voice/audio/video attachment into the local cache. Returns True when the event was
        already dispatched (oversized attachment), so the caller must return."""
        try:
            allowed, note = self._telegram_media_size_allowed(source, label)
            if not allowed:
                event.text = self._append_observed_note(event.text, note or "")
                logger.info("[Telegram] Skipped oversized user %s (size=%s)", kind, getattr(source, "file_size", None))
                await self.handle_message(event)
                return True
            file_obj = await source.get_file()
            data = await file_obj.download_as_bytearray()
            if kind == "video":
                ext = self._ext_from_path(getattr(file_obj, "file_path", None), SUPPORTED_VIDEO_TYPES, ext)
                cached_path = await cache_video_from_bytes_async(bytes(data), ext=ext)
                mime = SUPPORTED_VIDEO_TYPES.get(ext, "video/mp4")
            else:
                cached_path = await cache_audio_from_bytes_async(bytes(data), ext=ext)
            event.media_urls = [cached_path]
            event.media_types = [mime]
            logger.info("[Telegram] Cached user %s at %s", kind, cached_path)
        except Exception as e:
            logger.warning("[Telegram] Failed to cache %s: %s", kind, _redact_telegram_error_text(e), exc_info=True)
            await self._surface_media_cache_failure(msg, event, label, e)
        return False

    async def _dispatch_with_text(self, event: MessageEvent, text: str) -> bool:
        """Replace the event text with a user-facing note and dispatch it; returns True (handled)."""
        event.text = text
        await self.handle_message(event)
        return True

    @staticmethod
    def _set_cached_media(event: MessageEvent, path: str, mime: str, mtype: MessageType, log_fmt: str) -> None:
        event.media_urls = [path]
        event.media_types = [mime]
        event.message_type = mtype
        logger.info(log_fmt, path)

    async def _cache_inbound_document(self, msg, event: MessageEvent) -> bool:
        """Cache a document attachment (image → photo path, video, else generic media + text injection).
        Returns True when the event was already dispatched/routed so the caller must return."""
        doc = msg.document
        try:
            original_filename = doc.file_name or ""
            ext = os.path.splitext(original_filename)[1].lower() if original_filename else ""
            doc_mime = (doc.mime_type or "").lower()  # some clients send "IMAGE/PNG"
            if not ext and doc_mime:
                ext = _TELEGRAM_IMAGE_MIME_TO_EXT.get(doc_mime, "")
                if not ext:
                    ext = {v: k for k, v in SUPPORTED_DOCUMENT_TYPES.items()}.get(doc_mime, "")
            display = original_filename or doc_mime or ext or 'unknown'
            # Size check before the image branch so image documents can't bypass the limit.
            if not doc.file_size or doc.file_size > self._max_doc_bytes:
                logger.info("[Telegram] Document too large: %s bytes", doc.file_size)
                return await self._dispatch_with_text(
                    event, f"The document is too large or its size could not be verified. Maximum: {self._max_doc_bytes // (1024 * 1024)} MB.")
            # Screenshots/photos sent as documents take the image cache + batching path.
            if ext in _TELEGRAM_IMAGE_EXTENSIONS or doc_mime.startswith("image/"):
                file_obj = await doc.get_file()
                image_bytes = await file_obj.download_as_bytearray()
                image_ext = ext if ext in _TELEGRAM_IMAGE_EXTENSIONS else _TELEGRAM_IMAGE_MIME_TO_EXT.get(doc_mime, ".jpg")
                try:
                    cached_path = await cache_image_from_bytes_async(bytes(image_bytes), ext=image_ext)
                except ValueError as e:
                    logger.warning("[Telegram] Failed to cache image document: %s", _redact_telegram_error_text(e), exc_info=True)
                    return await self._dispatch_with_text(event, f"Image document '{display}' could not be read as an image.")
                self._set_cached_media(
                    event, cached_path, doc_mime if doc_mime.startswith(
                        "image/"
                    ) else _TELEGRAM_IMAGE_EXT_TO_MIME.get(image_ext, "image/jpeg"),
                    MessageType.PHOTO, "[Telegram] Cached user image-document at %s")
                await self._route_photo_event(msg, event)
                return True
            if not ext and doc.mime_type:
                ext = {v: k for k, v in SUPPORTED_VIDEO_TYPES.items()}.get(doc.mime_type, "")
            if not ext and doc.mime_type:
                # .jpg and .jpeg both map to image/jpeg; keep the first ext seen.
                image_mime_to_ext: dict[str, str] = {}
                for _ext, _mime in SUPPORTED_IMAGE_DOCUMENT_TYPES.items():
                    image_mime_to_ext.setdefault(_mime, _ext)
                ext = image_mime_to_ext.get(doc.mime_type, "")
            if ext in SUPPORTED_VIDEO_TYPES:
                file_obj = await doc.get_file()
                video_bytes = await file_obj.download_as_bytearray()
                self._set_cached_media(
                    event, await cache_video_from_bytes_async(bytes(video_bytes), ext=ext), SUPPORTED_VIDEO_TYPES[ext], MessageType.VIDEO,
                    "[Telegram] Cached user video document at %s")
                await self.handle_message(event)
                return True
            # Any file type is accepted (authorization is the gate, not the extension); unknown types get
            # application/octet-stream. Image documents already returned above.
            file_obj = await doc.get_file()
            raw_bytes = bytes(await file_obj.download_as_bytearray())
            from gateway.platforms.base import cache_media_bytes_async
            cached = await cache_media_bytes_async(raw_bytes, filename=original_filename or f"document{ext or '.bin'}", mime_type=doc_mime)
            if cached is None:
                return await self._dispatch_with_text(event, f"Document '{display}' could not be cached.")
            event.media_urls = [cached.path]
            event.media_types = [cached.media_type]
            if cached.kind == "audio":
                event.message_type = MessageType.AUDIO
            logger.info("[Telegram] Cached user %s at %s (%s)", cached.kind, cached.path, cached.media_type)
            # Inject text-readable content (≤100 KB). Gate on extension/MIME, NOT a blind UTF-8 decode:
            # PDF/zip/docx have decodable ASCII headers. Binary files are surfaced as a cached path only.
            MAX_TEXT_INJECT_BYTES = 100 * 1024
            _is_text = ext in _TEXT_INJECT_EXTENSIONS or (doc_mime or "").startswith("text/")
            if _is_text and len(raw_bytes) <= MAX_TEXT_INJECT_BYTES:
                try:
                    text_content = raw_bytes.decode("utf-8")
                    display_name = re.sub(r'[^\w.\- ]', '_', original_filename or f"document{ext or '.txt'}")
                    injection = f"[Content of {display_name}]:\n{text_content}"
                    event.text = f"{injection}\n\n{event.text}" if event.text else injection
                except UnicodeDecodeError:
                    pass  # binary — agent has the cached path
        except Exception as e:
            logger.warning("[Telegram] Failed to cache document: %s", _redact_telegram_error_text(e), exc_info=True)
            await self._surface_media_cache_failure(msg, event, "attachment", e, display_name=getattr(doc, "file_name", None) or None)
        return False

    async def _handle_media_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming media messages, downloading images to local cache."""
        msg = update.message
        if not msg:
            return
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg, level=logging.INFO, what="media from unauthorized user")
            return
        if not self._should_process_message(msg):
            if self._should_observe_unmentioned_group_message(msg):
                _event = self._build_message_event(msg, self._media_message_type(msg), update_id=update.update_id)
                if msg.caption:
                    _event.text = self._clean_bot_trigger_text(msg.caption)
                await self._cache_observed_media(msg, _event)
                self._observe_unmentioned_group_message(msg, _event.message_type, update_id=update.update_id, event=_event)
            return
        event = self._build_message_event(msg, self._media_message_type(msg), update_id=update.update_id)
        if msg.caption:
            event.text = self._clean_bot_trigger_text(msg.caption)
        # Stickers: _handle_sticker overwrites event.text with its vision description, so observe attribution must run after it.
        if msg.sticker:
            await self._handle_sticker(msg, event)
            await self.handle_message(self._apply_telegram_group_observe_attribution(event))
            return
        event = self._apply_telegram_group_observe_attribution(event)
        # Cache photo locally: Telegram's file URLs expire (~1 hour) before vision may run.
        if msg.photo:
            try:
                file_obj = await msg.photo[-1].get_file()  # PhotoSize list sorted by size; largest last
                image_bytes = await file_obj.download_as_bytearray()
                ext = self._ext_from_path(file_obj.file_path, [".png", ".webp", ".gif", ".jpeg", ".jpg"], ".jpg")
                self._set_cached_media(
                    event, await cache_image_from_bytes_async(bytes(image_bytes), ext=ext), f"image/{ext.lstrip('.')}", event.message_type,
                    "[Telegram] Cached user photo at %s")
                await self._route_photo_event(msg, event)
                return
            except Exception as e:
                logger.warning("[Telegram] Failed to cache photo: %s", _redact_telegram_error_text(e), exc_info=True)
                await self._surface_media_cache_failure(msg, event, "photo", e)
        # Voice/audio cached for STT transcription; video for vision.
        if msg.voice:
            if await self._cache_inbound_av(msg, event, msg.voice, "voice message", "voice", ".ogg", "audio/ogg"):
                return
        elif msg.audio:
            if await self._cache_inbound_av(msg, event, msg.audio, "audio file", "audio", ".mp3", "audio/mp3"):
                return
        elif msg.video:
            if await self._cache_inbound_av(msg, event, msg.video, "video file", "video", ".mp4", "video/mp4"):
                return
        elif msg.document and await self._cache_inbound_document(msg, event):
            return
        media_group_id = getattr(msg, "media_group_id", None)
        if media_group_id:
            await self._queue_media_group_event(str(media_group_id), event)
            return
        await self.handle_message(event)

    async def _queue_media_group_event(self, media_group_id: str, event: MessageEvent) -> None:
        """Debounce album items (shared media_group_id) into one MessageEvent so the second image isn't
        treated as a new message interrupting the first."""
        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="media-group-enqueue")
            return
        self._merge_into_pending(self._media_group_events, media_group_id, event)
        prior_task = self._media_group_tasks.get(media_group_id)
        if prior_task:
            prior_task.cancel()
        self._media_group_tasks[media_group_id] = asyncio.create_task(self._flush_media_group_event(media_group_id))

    async def _flush_media_group_event(self, media_group_id: str) -> None:
        await self._flush_buffered(
            self._media_group_events, self._media_group_tasks, media_group_id, self.MEDIA_GROUP_WAIT_SECONDS, "media-group")

    async def _handle_sticker(self, msg: Message, event: "MessageEvent") -> None:
        """Describe a sticker via vision, cached by file_unique_id; animated/video stickers get an emoji placeholder."""
        from gateway.sticker_cache import (
            get_cached_description, cache_sticker_description, build_sticker_injection,
            build_animated_sticker_injection, STICKER_VISION_PROMPT)
        sticker = msg.sticker
        emoji = sticker.emoji or ""
        set_name = sticker.set_name or ""
        if sticker.is_animated or sticker.is_video:
            event.text = build_animated_sticker_injection(emoji)
            return
        cached = get_cached_description(sticker.file_unique_id)
        if cached:
            event.text = build_sticker_injection(cached["description"], cached.get("emoji", emoji), cached.get("set_name", set_name))
            logger.info("[Telegram] Sticker cache hit: %s", sticker.file_unique_id)
            return
        fallback = f"a sticker with emoji {emoji}" if emoji else "a sticker"
        try:
            file_obj = await sticker.get_file()
            image_bytes = await file_obj.download_as_bytearray()
            cached_path = await cache_image_from_bytes_async(bytes(image_bytes), ext=".webp")
            logger.info("[Telegram] Analyzing sticker at %s", cached_path)
            from tools.vision_tools import vision_analyze_tool
            result = json.loads(await vision_analyze_tool(image_url=cached_path, user_prompt=STICKER_VISION_PROMPT))
            if result.get("success"):
                description = result.get("analysis", "a sticker")
                cache_sticker_description(sticker.file_unique_id, description, emoji, set_name)
                event.text = build_sticker_injection(description, emoji, set_name)
            else:
                event.text = build_sticker_injection(fallback, emoji, set_name)
        except Exception as e:
            logger.warning("[Telegram] Sticker analysis error: %s", _redact_telegram_error_text(e), exc_info=True)
            event.text = build_sticker_injection(fallback, emoji, set_name)

    def _reload_dm_topics_from_config(self) -> None:
        """Re-read dm_topics from config.yaml so externally created topics work without restart."""
        try:
            from hermes_cli.config import load_config_readonly  # canonical loader: managed overlay + ${VAR}
            dm_topics = load_config_readonly().get("platforms", {}).get("telegram", {}).get("extra", {}).get("dm_topics", [])
            if not dm_topics:
                self._dm_topics_config = []
                self._dm_topic_chat_ids = set()
                return
            self._dm_topics_config = dm_topics
            self._dm_topic_chat_ids = {str(chat_entry["chat_id"]) for chat_entry in dm_topics if "chat_id" in chat_entry}
            for chat_entry in dm_topics:
                cid = chat_entry.get("chat_id")
                if not cid:
                    continue
                for t in chat_entry.get("topics", []):
                    tid = t.get("thread_id")
                    name = t.get("name")
                    if tid and name and f"{cid}:{name}" not in self._dm_topics:
                        self._dm_topics[f"{cid}:{name}"] = int(tid)
                        logger.info("[%s] Hot-loaded DM topic from config: %s -> thread_id=%s", self.name, f"{cid}:{name}", tid)
        except Exception as e:
            logger.debug("[%s] Failed to reload dm_topics from config: %s", self.name, e)

    def _get_dm_topic_info(self, chat_id: str, thread_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the DM topic config dict (name, skill, ...) for this thread_id, or None."""
        if not thread_id:
            return None
        thread_id_int = int(thread_id)

        def _lookup() -> Optional[Dict[str, Any]]:
            for key, cached_tid in self._dm_topics.items():
                if cached_tid == thread_id_int and key.startswith(f"{chat_id}:"):
                    topic_name = key.split(":", 1)[1]
                    for chat_entry in self._dm_topics_config:
                        if str(chat_entry.get("chat_id")) == chat_id:
                            for t in chat_entry.get("topics", []):
                                if t.get("name") == topic_name:
                                    return t
                    return {"name": topic_name}
            return None

        found = _lookup()
        if found is not None:
            return found
        self._reload_dm_topics_from_config()  # cache miss — topics may have been added externally
        return _lookup()

    def _cache_dm_topic_from_message(self, chat_id: str, thread_id: str, topic_name: str) -> None:
        """Cache a thread_id -> topic_name mapping discovered from an incoming message."""
        cache_key = f"{chat_id}:{topic_name}"
        if cache_key not in self._dm_topics:
            self._dm_topics[cache_key] = int(thread_id)
            logger.info("[%s] Cached DM topic from message: %s -> thread_id=%s", self.name, cache_key, thread_id)

    @classmethod
    def _flatten_rich_inline_text(cls, value: Any) -> str:
        """Best-effort plaintext flattener for Bot API rich-message inline nodes."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(cls._flatten_rich_inline_text(item) for item in value)
        if isinstance(value, dict):
            for key in ("text", "children"):
                if value.get(key) is not None:
                    return cls._flatten_rich_inline_text(value[key])
        return ""

    @classmethod
    def _flatten_rich_blocks(cls, blocks: Any) -> str:
        """Best-effort plaintext flattener for Bot API rich-message blocks."""
        if not isinstance(blocks, list):
            return ""
        lines: List[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "list":
                for item in block.get("items", []):
                    if not isinstance(item, dict):
                        continue
                    item_lines = cls._flatten_rich_blocks(item.get("blocks")).splitlines()
                    if not item_lines:
                        continue
                    label = item.get("label")
                    lines.append(f"{label} {item_lines[0]}".strip() if label else item_lines[0])
                    lines.extend(item_lines[1:])
                continue
            text = cls._flatten_rich_inline_text(block.get("text"))
            if text:
                lines.extend(text.splitlines())
        return "\n".join(line.rstrip() for line in lines if line)

    @classmethod
    def _extract_rich_reply_text(cls, reply_to_message: Any) -> Optional[str]:
        """Return plaintext echoed by Telegram's rich_message reply payload."""
        try:
            getter = getattr(getattr(reply_to_message, "api_kwargs", None), "get", None)
            if not callable(getter):
                return None
            rich_getter = getattr(getter("rich_message"), "get", None)
            if not callable(rich_getter):
                return None
            return cls._flatten_rich_blocks(rich_getter("blocks")).strip() or None
        except Exception:
            return None

    def _resolve_topic_binding(self, message: Message, chat_type: str, thread_id_str: Optional[str]) -> tuple:
        """Return ``(chat_topic, topic_skill)`` for a DM topic or bound forum topic (else Nones)."""
        chat = message.chat
        chat_topic = None
        topic_skill = None
        if chat_type == "dm" and thread_id_str:
            topic_info = self._get_dm_topic_info(str(chat.id), thread_id_str)
            if topic_info:
                chat_topic = topic_info.get("name")
                topic_skill = topic_info.get("skill")
            # forum_topic_created service messages also reveal topic names
            if hasattr(message, "forum_topic_created") and message.forum_topic_created:
                created_name = message.forum_topic_created.name
                if created_name:
                    self._cache_dm_topic_from_message(str(chat.id), thread_id_str, created_name)
                    if not chat_topic:
                        chat_topic = created_name
        elif chat_type == "group" and thread_id_str:
            # Forum topic skill binding via config.extra['group_topics']; accepts both
            # [{"chat_id": ..., "topics": [...]}] and legacy {"-100...": [{"thread_id": 12}]}.
            group_topics_config = self.config.extra.get("group_topics", [])
            if isinstance(group_topics_config, dict):
                group_topics_iter = [{"chat_id": cfg_chat_id, "topics": topics} for cfg_chat_id, topics in group_topics_config.items()]
            elif isinstance(group_topics_config, list):
                group_topics_iter = [entry for entry in group_topics_config if isinstance(entry, dict)]
            else:
                group_topics_iter = []
            for chat_entry in group_topics_iter:
                if str(chat_entry.get("chat_id", "")) != str(chat.id):
                    continue
                topics = chat_entry.get("topics", [])
                for topic in (topics if isinstance(topics, list) else []):
                    if not isinstance(topic, dict):
                        continue
                    tid = topic.get("thread_id")
                    if tid is not None and str(tid) == thread_id_str:
                        chat_topic = topic.get("name")
                        topic_skill = topic.get("skill")
                        break
                break
        return chat_topic, topic_skill

    def _reply_context(self, message: Message) -> tuple:
        """``(reply_to_id, reply_to_text)`` for the replied-to message: Telegram's native partial quote
        first, then text/caption, rich echo, then the sent index."""
        if not message.reply_to_message:
            return None, None
        reply_to_id = str(message.reply_to_message.message_id)
        quote = getattr(message, "quote", None)
        quote_text = getattr(quote, "text", None) if quote is not None else None
        if quote_text:
            return reply_to_id, quote_text
        reply_to_text = message.reply_to_message.text or message.reply_to_message.caption or None
        if not reply_to_text:
            reply_to_text = self._extract_rich_reply_text(message.reply_to_message)
        if not reply_to_text:
            try:
                from gateway import rich_sent_store
                reply_to_text = rich_sent_store.lookup(str(message.chat.id), reply_to_id)
            except Exception:
                # Extract reply context if this message is a reply. Prefer Telegram's native partial quote
                # (message.quote, TextQuote) so a user replying to a single selected substring of a prior
                # multi-section message doesn't get the whole replied-to message injected into the agent's
                # context — which can cause the agent to act on unrelated actionable-looking text the user
                # didn't quote (#22619). Fall back to the full replied-to message text / caption when no
                # native quote is present.
                reply_to_text = None
        return reply_to_id, reply_to_text

    def _build_message_event(self, message: Message, msg_type: MessageType, update_id: Optional[int] = None) -> MessageEvent:
        """Build a MessageEvent from a Telegram message. ``update_id`` lets ``/restart`` record the
        triggering offset so the new gateway process advances past it."""
        chat = message.chat
        user = message.from_user
        telegram_chat_type = self._chat_type_str(chat)  # str() so PTB enums and plain-string mocks both work
        chat_type = "group" if telegram_chat_type in {"group", "supergroup"} else ("channel" if telegram_chat_type == "channel" else "dm")
        # Shared normalizer so gating and session routing agree (reply-UI anchors dropped, General → "1").
        # Resolve routable thread id for DM topics and forum group topics via the shared normalizer, so
        # gating and session routing agree on one value. Only real topic/forum messages keep a thread id;
        # ordinary reply-UI anchors are dropped (they are not durable session threads and sends against them
        # hit 'Message thread not found', #3206), while forum General-topic messages
        # (message_thread_id=None) normalize to the General-topic id so replies route back to General
        # (#22423).
        thread_id_str = self._effective_message_thread_id(message)
        chat_topic, topic_skill = self._resolve_topic_binding(message, chat_type, thread_id_str)
        has_full_name = hasattr(chat, "full_name")
        if user:
            user_name = user.full_name
        elif has_full_name and chat_type == "dm":
            user_name = chat.full_name
        else:
            user_name = chat.title if chat_type == "channel" else None
        source = self.build_source(
            chat_id=str(chat.id), chat_name=chat.title or (chat.full_name if has_full_name else None), chat_type=chat_type,
            user_id=(str(user.id) if user else (str(chat.id) if chat_type in {"dm", "channel"} else None)),
            user_name=user_name, thread_id=thread_id_str, chat_topic=chat_topic, message_id=str(message.message_id),
            is_bot=bool(getattr(user, "is_bot", False)) if user else False)
        reply_to_id, reply_to_text = self._reply_context(message)
        from gateway.platforms.base import resolve_channel_prompt  # per-channel/topic ephemeral prompt
        _chat_id_str = str(chat.id)
        return MessageEvent(
            text=message.text or "", message_type=msg_type, source=source, raw_message=message,
            message_id=str(message.message_id), platform_update_id=update_id,
            reply_to_message_id=reply_to_id, reply_to_text=reply_to_text, auto_skill=topic_skill,
            channel_prompt=resolve_channel_prompt(self.config.extra, thread_id_str or _chat_id_str, _chat_id_str if thread_id_str else None),
            timestamp=message.date)

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
