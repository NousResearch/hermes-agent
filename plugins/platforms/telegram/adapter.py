"""
Telegram platform adapter.

Uses python-telegram-bot library for:
- Receiving messages from users/groups
- Sending responses back
- Handling media and commands
"""

import asyncio
import dataclasses
import inspect
import json
import logging
import os
import html as _html
import re
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)


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


def _consume_abandoned_task(task: asyncio.Task) -> None:
    """Observe a detached task's terminal exception to avoid noisy loop logs."""
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("Abandoned Telegram init task failed after timeout", exc_info=True)


async def _await_with_thread_deadline(awaitable, timeout: float, *, on_abandon=None):
    """Await with a wall-clock deadline that does not depend on loop timers.

    ``asyncio.wait_for`` schedules its timeout on the event loop and then waits
    for cancellation to propagate.  PTB/httpcore initialization can sit inside
    cancellation-shielded anyio scopes, so a timed-out initialize() may never
    hand control back to the retry ladder under some supervisors.  This helper
    lets a daemon ``threading.Timer`` wake the loop and, on timeout, abandons
    the shielded task instead of awaiting cancellation completion.

    ``on_abandon`` (optional) is a zero-arg callable returning an awaitable that
    is scheduled as a detached best-effort cleanup when the task is abandoned on
    timeout.  The abandoned initialize() may leave a half-built httpx client /
    connection pool open (it never completed and we do not await its
    cancellation), so the caller uses this to shut that state down and avoid
    leaking a pool per retry attempt.  Cleanup runs detached and its own errors
    are swallowed, so it can never re-block the retry ladder.
    """
    task = asyncio.ensure_future(awaitable)
    loop = asyncio.get_running_loop()
    deadline = loop.create_future()

    def _mark_expired() -> None:
        if not deadline.done():
            deadline.set_result(None)

    def _expire_from_thread() -> None:
        loop.call_soon_threadsafe(_mark_expired)

    timer = threading.Timer(max(timeout, 0.0), _expire_from_thread)
    timer.daemon = True
    timer.start()
    try:
        done, _ = await asyncio.wait(
            {task, deadline},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            if not deadline.done():
                deadline.cancel()
            return await task

        task.cancel()
        task.add_done_callback(_consume_abandoned_task)
        if on_abandon is not None:
            # Detached best-effort cleanup: close the half-built app's httpx
            # client/pool so an abandoned attempt can't leak sockets across the
            # retry ladder. Detached + exception-observed so it never re-blocks
            # or re-hangs the ladder we are trying to advance.
            cleanup = asyncio.ensure_future(_run_abandon_cleanup(on_abandon))
            cleanup.add_done_callback(_consume_abandoned_task)
        raise asyncio.TimeoutError()
    finally:
        timer.cancel()


async def _run_abandon_cleanup(on_abandon) -> None:
    """Run the abandonment cleanup coroutine, swallowing any failure.

    Wrapped so a cleanup that itself hangs or raises cannot surface as an
    unhandled task error or block anything — it is fully fire-and-forget.
    """
    try:
        result = on_abandon()
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            await result
    except Exception:
        logger.debug("Abandoned Telegram init cleanup failed", exc_info=True)


async def _shutdown_abandoned_app(app) -> None:
    """Release a half-built PTB app's httpx transports after init was abandoned.

    ``Application.shutdown()`` / ``Bot.shutdown()`` are gated on the app's
    ``_initialized`` / ``_requests_initialized`` flags, which a wedged
    ``initialize()`` (the case this whole path exists for) may never have set —
    so calling only ``app.shutdown()`` no-ops and leaks the connection pool it
    was meant to close.  ``HTTPXRequest`` builds its ``httpx.AsyncClient``
    eagerly in its constructor and its ``shutdown()`` gates only on
    ``client.is_closed``, so closing the request transports directly releases
    the pool regardless of PTB init state.  We try the clean path first, then
    fall back to the transports.  All best-effort and swallowed.
    """
    if app is None:
        return
    try:
        await app.shutdown()
    except Exception:
        logger.debug("Abandoned Telegram app.shutdown() failed", exc_info=True)
    # Directly close the underlying request transports (bypasses PTB's
    # init-gated shutdown so the eagerly-built httpx pool is released even when
    # the abandoned initialize() never flipped _initialized).
    bot = getattr(app, "bot", None)
    requests = getattr(bot, "_request", None) if bot is not None else None
    if not requests:
        return
    for request in requests:
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
        Application,
        CommandHandler,
        CallbackQueryHandler,
        MessageHandler as TelegramMessageHandler,
        ContextTypes,
        filters,
    )
    from telegram.constants import ParseMode, ChatType
    from telegram.request import HTTPXRequest
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = Any
    Bot = Any
    Message = Any
    InlineKeyboardButton = Any
    InlineKeyboardMarkup = Any
    LinkPreviewOptions = None
    Application = Any
    CommandHandler = Any
    CallbackQueryHandler = Any
    TelegramMessageHandler = Any
    HTTPXRequest = Any
    filters = None
    ParseMode = None
    ChatType = None

    # Mock ContextTypes so type annotations using ContextTypes.DEFAULT_TYPE
    # don't crash during class definition when the library isn't installed.
    class _MockContextTypes:
        DEFAULT_TYPE = Any
    ContextTypes = _MockContextTypes

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    classify_send_error,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_video_from_bytes,
    cache_document_from_bytes,
    resolve_proxy_url,
    SUPPORTED_VIDEO_TYPES,
    SUPPORTED_DOCUMENT_TYPES,
    SUPPORTED_IMAGE_DOCUMENT_TYPES,
    _TEXT_INJECT_EXTENSIONS,
    utf16_len,
)
from plugins.platforms.telegram.telegram_ids import (
    normalize_telegram_chat_id,
)
from plugins.platforms.telegram.telegram_network import (
    TelegramFallbackTransport,
    discover_fallback_ips,
    parse_fallback_ip_env,
)
from utils import atomic_replace, env_float, env_int

_TELEGRAM_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_TELEGRAM_IMAGE_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_TELEGRAM_IMAGE_EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def check_telegram_requirements() -> bool:
    """Check if Telegram dependencies are available.

    If python-telegram-bot is missing, attempts to lazy-install it via
    ``tools.lazy_deps.ensure("platform.telegram")``. After a successful
    install, re-imports the SDK and flips ``TELEGRAM_AVAILABLE`` to True
    so the adapter's class-level type aliases get rebound.
    """
    global TELEGRAM_AVAILABLE, Update, Bot, Message, InlineKeyboardButton
    global InlineKeyboardMarkup, LinkPreviewOptions, Application
    global CommandHandler, CallbackQueryHandler, TelegramMessageHandler
    global ContextTypes, filters, ParseMode, ChatType, HTTPXRequest
    if TELEGRAM_AVAILABLE:
        return True
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("platform.telegram", prompt=False)
    except Exception:
        return False
    try:
        from telegram import Update as _Update, Bot as _Bot, Message as _Message
        from telegram import InlineKeyboardButton as _IKB, InlineKeyboardMarkup as _IKM
        try:
            from telegram import LinkPreviewOptions as _LPO
        except ImportError:
            _LPO = None
        from telegram.ext import (
            Application as _App, CommandHandler as _CH,
            CallbackQueryHandler as _CQH,
            MessageHandler as _MH,
            ContextTypes as _CT, filters as _filters,
        )
        from telegram.constants import ParseMode as _PM, ChatType as _CtT
        from telegram.request import HTTPXRequest as _HR
    except ImportError:
        return False
    Update = _Update
    Bot = _Bot
    Message = _Message
    InlineKeyboardButton = _IKB
    InlineKeyboardMarkup = _IKM
    LinkPreviewOptions = _LPO
    Application = _App
    CommandHandler = _CH
    CallbackQueryHandler = _CQH
    TelegramMessageHandler = _MH
    ContextTypes = _CT
    filters = _filters
    ParseMode = _PM
    ChatType = _CtT
    HTTPXRequest = _HR
    TELEGRAM_AVAILABLE = True
    return True


# Matches every character that MarkdownV2 requires to be backslash-escaped
# when it appears outside a code span or fenced code block.
_MDV2_ESCAPE_RE = re.compile(r'([_*\[\]()~`>#\+\-=|{}.!\\])')


def _escape_mdv2(text: str) -> str:
    """Escape Telegram MarkdownV2 special characters with a preceding backslash."""
    return _MDV2_ESCAPE_RE.sub(r'\\\1', text)


def _strip_mdv2(text: str) -> str:
    """Strip MarkdownV2 escape backslashes to produce clean plain text.

    Also removes MarkdownV2 formatting markers so the fallback
    doesn't show stray syntax characters from format_message conversion.
    """
    # Remove escape backslashes before special characters
    cleaned = re.sub(r'\\([_*\[\]()~`>#\+\-=|{}.!\\])', r'\1', text)
    # Remove standard markdown bold (**text** → text) BEFORE MarkdownV2 bold
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    # Remove MarkdownV2 bold markers that format_message converted from **bold**
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
    # Remove MarkdownV2 italic markers that format_message converted from *italic*
    # Use word boundary (\b) to avoid breaking snake_case like my_variable_name
    cleaned = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', cleaned)
    # Remove MarkdownV2 strikethrough markers (~text~ → text)
    cleaned = re.sub(r'~([^~]+)~', r'\1', cleaned)
    # Remove MarkdownV2 spoiler markers (||text|| → text)
    cleaned = re.sub(r'\|\|([^|]+)\|\|', r'\1', cleaned)
    return cleaned


_CHUNK_INDICATOR_ON_FENCE_RE = re.compile(
    r'(?m)^``` (?P<indicator>(?:\\)?\(\d+/\d+(?:\\)?\))$'
)


def _separate_chunk_indicator_from_fence(text: str) -> str:
    """Move ``(N/M)`` chunk markers off Telegram code-fence lines.

    ``truncate_message()`` appends chunk indicators to the end of a chunk. When
    the chunk had to close an in-progress fenced code block, that creates a
    line like ````` \\(1/2\\)`` after MarkdownV2 escaping. Telegram does not
    treat that as a clean closing fence, so it can reject MarkdownV2 and fall
    back to plain text. Put the indicator on its own line immediately after the
    closing fence.
    """
    return _CHUNK_INDICATOR_ON_FENCE_RE.sub(r'```\n\g<indicator>', text)


# ---------------------------------------------------------------------------
# Markdown table → Telegram-friendly row groups
# ---------------------------------------------------------------------------
# Telegram's MarkdownV2 has no table syntax — '|' is just an escaped literal,
# so pipe tables render as noisy backslash-pipe text with no alignment.
# The shared convert_table_to_bullets() in gateway.platforms.helpers handles
# the full conversion (detection + rendering); Telegram just calls it.

from gateway.platforms.helpers import (
    TABLE_SEPARATOR_RE as _TABLE_SEPARATOR_RE,
    convert_table_to_bullets as _wrap_markdown_tables,
)


# ---------------------------------------------------------------------------
# Rich-message newline normalization
# ---------------------------------------------------------------------------

# Matches a protected region whose internal newlines must stay bare in the
# rich-message path: a fenced code block (```...```) OR a GFM pipe-table block
# (a header row, a delimiter row of dashes/pipes, then any pipe data rows).
# Telegram renders both natively, so injecting Markdown hard breaks inside them
# would corrupt the code block / table.
_RICH_PROTECTED_REGION_RE = re.compile(
    r'(?:```[^\n]*\n[\s\S]*?```)'                       # fenced code block
    r'|(?:^[^\n]*\|[^\n]*\n'                            # table header row (has a pipe)
    r'[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*'  # delimiter
    r'(?:\n[^\n]*\|[^\n]*)*)',                          # data rows (newline-led, trailing \n left for prose)
    re.MULTILINE,
)


def _rich_normalize_linebreaks(text: str) -> str:
    """Convert single ``\\n`` to Markdown hard breaks for the rich-message path.

    Standard Markdown treats a lone ``\\n`` as whitespace (soft break), so
    Bot API 10.1 ``sendRichMessage`` collapses multi-line content — e.g.
    slash-command lists joined with ``"\\n".join(lines)`` — into a single
    paragraph.  Adding two trailing spaces before each single newline
    forces a hard line break (``<br>``) in the rendered output.

    Paragraph breaks (``\\n\\n``), fenced code blocks, and GFM pipe-table
    blocks are left untouched: tables render natively in the rich path and a
    hard break injected into a row separator would corrupt the table.
    """
    if not text or '\n' not in text:
        return text

    out: list[str] = []
    # Split off protected regions (fenced code OR table blocks) and only inject
    # hard breaks in the prose between them. Boundary newlines are handled by
    # the original single-\n regex, which sees each prose run as a whole string.
    pos = 0
    for m in _RICH_PROTECTED_REGION_RE.finditer(text):
        prose = text[pos:m.start()]
        out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', prose))
        out.append(m.group(0))  # protected region kept verbatim
        pos = m.end()
    tail = text[pos:]
    out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', tail))
    return ''.join(out)


# Watchdog bound for `await updater.stop()`. When the underlying TCP socket is
# in CLOSE-WAIT the PTB polling task is blocked on epoll on the dead socket and
# never wakes, so an unguarded stop() hangs indefinitely and wedges the whole
# reconnect/teardown ladder. This is an internal safety bound (not a user knob),
# applied identically at every stop() site so no path can hang on a dead socket.
_UPDATER_STOP_TIMEOUT = 15.0
# start_polling() can also hang when the connection pool is in a degraded state
# after _drain_polling_connections(), particularly when both primary and fallback
# Telegram endpoints are unreachable. Bounding start_polling() prevents the
# reconnect ladder from stalling indefinitely and allows the heartbeat loop to
# trigger its own recovery path. Refs: NousResearch/hermes-agent#59614
_UPDATER_START_TIMEOUT = 30.0


class TelegramAdapter(BasePlatformAdapter):
    """
    Telegram bot adapter.

    Handles:
    - Receiving messages from users and groups
    - Sending responses with Telegram markdown
    - Forum topics (thread_id support)
    - Media messages
    """

    # Telegram message limits
    MAX_MESSAGE_LENGTH = 4096
    supports_code_blocks = True  # Telegram MarkdownV2 renders fenced code blocks
    splits_long_messages = True  # send() chunks via truncate_message(MAX_MESSAGE_LENGTH)
    # Bot API 10.1 Rich Messages cap the raw markdown/html text at 32,768
    # UTF-8 characters. Content above this is sent via the legacy chunking path.
    RICH_MESSAGE_MAX_CHARS = 32768
    # Backwards-compatible alias for tests/external callers that referenced the
    # initial implementation name. The API limit is character-based, not bytes.
    RICH_MESSAGE_MAX_BYTES = RICH_MESSAGE_MAX_CHARS
    # Threshold for detecting Telegram client-side message splits.
    # When a chunk is near this limit, a continuation is almost certain.
    _SPLIT_THRESHOLD = 4000
    MEDIA_GROUP_WAIT_SECONDS = 0.8
    _GENERAL_TOPIC_THREAD_ID = "1"

    # Telegram's edit_message applies MarkdownV2 formatting only on the
    # finalize=True path.  Without this flag, stream_consumer._send_or_edit
    # short-circuits when the raw text is unchanged between the last streamed
    # edit and the final edit, skipping the plain-text → MarkdownV2 conversion.
    # Fixes #25710.
    REQUIRES_EDIT_FINALIZE: bool = True
    # Retrying a turn-final edit consumes more of the same Telegram flood
    # budget while the completed answer remains undelivered. Move directly to
    # the final fallback path instead.
    FALLBACK_ON_FINAL_EDIT_FLOOD: bool = True
    # A failed final edit can leave Telegram clients with only a partial or
    # non-durable preview. Commit empty-tail fallbacks as a fresh final message
    # instead of trusting the preview as completed delivery.
    RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK: bool = True

    # Adaptive text-batch ingress: short messages need a tighter delay so the
    # first token reaches the agent fast.  Numbers tuned for "feels instant":
    # ≤320 codepoints (one short paragraph) settles in ~180ms; ≤1024
    # (a normal paragraph) in ~240ms; longer waits the configured cap.
    # Always clamped to ``_text_batch_delay_seconds`` so an operator can lower
    # the cap further via env var.
    _TEXT_BATCH_FAST_LEN = 320
    _TEXT_BATCH_FAST_DELAY_S = 0.18
    _TEXT_BATCH_SHORT_LEN = 1024
    _TEXT_BATCH_SHORT_DELAY_S = 0.24

    @staticmethod
    def _env_float_clamped(
        name: str,
        default: float,
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        """Read a float env var, reject non-finite values, and clamp to bounds.

        Guarantees the returned value is a finite number usable directly in
        ``asyncio.sleep()`` and similar APIs that reject NaN / Inf.
        """
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
    def message_len_fn(self):
        """Telegram measures message length in UTF-16 code units."""
        return utf16_len

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.TELEGRAM)
        self._app: Optional[Application] = None
        self._bot: Optional[Bot] = None
        self._webhook_mode: bool = False
        self._mention_patterns = self._compile_mention_patterns()
        self._reply_to_mode: str = getattr(config, 'reply_to_mode', 'first') or 'first'
        self._disable_link_previews: bool = self._coerce_bool_extra("disable_link_previews", False)
        # Bot API 10.1 Rich Messages: render constructs the legacy MarkdownV2
        # path degrades (tables → bullet lists, task lists, <details>, block
        # math) via sendRichMessage / editMessageText's rich_message param using
        # the raw agent markdown. Disabled by default so Telegram messages stay
        # easy to copy as plain text; users can opt in for richer rendering on
        # clients that accept but render rich messages poorly via
        # platforms.telegram.extra.rich_messages: true.  Keep this opt-in:
        # current Telegram clients can make rich messages difficult to copy
        # as plain text, which is worse than degraded table/task-list rendering
        # for command snippets and mobile handoffs.
        self._rich_messages_enabled: bool = self._coerce_bool_extra("rich_messages", False)
        # Rich draft previews use a separate opt-in. Telegram macOS / Desktop
        # can leave Bot API 10.1 rich draft frames visually overlaid until the
        # chat is redrawn, while final rich messages remain useful.
        self._rich_drafts_enabled: bool = self._coerce_bool_extra("rich_drafts", False)
        # Latched off after a capability failure on sendRichMessage /
        # sendRichMessageDraft (e.g. older python-telegram-bot without the
        # endpoint) so later sends skip the doomed rich attempt entirely.
        self._rich_send_disabled: bool = False
        self._rich_draft_disabled: bool = False
        # Transient Telegram sendChatAction failures (network blips, 429/5xx)
        # can happen on every keep-typing tick while the agent is waiting on a
        # long model call. Back off per chat so a short Telegram-side outage
        # does not spam the API/logs or burn the keep-typing budget.
        self._telegram_typing_cooldown_until: Dict[str, float] = {}
        self._telegram_typing_cooldown_seconds: float = self._coerce_float_extra(
            "typing_cooldown_seconds",
            30.0,
            min_value=1.0,
            max_value=300.0,
        )
        # Buffer rapid/album photo updates so Telegram image bursts are handled
        # as a single MessageEvent instead of self-interrupting multiple turns.
        self._media_batch_delay_seconds = env_float("HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS", 0.8)
        self._pending_photo_batches: Dict[str, MessageEvent] = {}
        self._pending_photo_batch_tasks: Dict[str, asyncio.Task] = {}
        self._media_group_events: Dict[str, MessageEvent] = {}
        self._media_group_tasks: Dict[str, asyncio.Task] = {}
        # Buffer rapid text messages so Telegram client-side splits of long
        # messages are aggregated into a single MessageEvent.  Lower defaults
        # (0.3s / 1.0s instead of 0.6s / 2.0s) let short replies stream
        # without a noticeable wait — combined with the adaptive fast-path
        # in ``_calc_text_batch_delay`` below, ≤320-codepoint replies settle
        # in ~180ms.  All bounds are conservative for Telegram's
        # ~1 edit/s flood envelope.
        self._text_batch_delay_seconds = self._env_float_clamped(
            "HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS",
            0.3,
            min_value=0.08,
            max_value=2.0,
        )
        self._text_batch_split_delay_seconds = self._env_float_clamped(
            "HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS",
            1.0,
            min_value=self._text_batch_delay_seconds,
            max_value=4.0,
        )
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}
        self._drop_delayed_deliveries = False
        self._polling_error_task: Optional[asyncio.Task] = None
        self._polling_conflict_count: int = 0
        self._polling_network_error_count: int = 0
        self._polling_error_callback_ref = None
        self._polling_heartbeat_task: Optional[asyncio.Task] = None
        # Consecutive heartbeat probes that saw queued updates the running
        # poller is not consuming. get_me() can't see this — the send path is
        # healthy while the getUpdates consumer is wedged — so the heartbeat
        # also probes get_webhook_info().pending_update_count and escalates to
        # recovery after two consecutive stuck probes (#42909).
        self._polling_pending_stuck_count: int = 0
        # Consecutive heartbeat probes that found the updater stopped entirely
        # (running=False) while we are in polling mode with no reconnect in
        # flight. Distinct from the wedged-but-running case above: the long-poll
        # task is simply gone, so neither the connectivity probe nor PTB's
        # error_callback ever fires and the gateway silently stops receiving
        # messages with the process still alive (#55769).
        self._polling_not_running_count: int = 0
        # After sustained reconnect storms the PTB httpx pool can return
        # SendResult(success=True) for sends that never actually transmit.
        # _handle_polling_network_error sets this; _verify_polling_after_reconnect
        # clears it once getMe() confirms the Bot client is healthy.
        # While True, send() short-circuits to a failure so callers
        # (cron live-adapter branch) fall through to standalone delivery.
        self._send_path_degraded: bool = False
        self._general_request_drain_lock = asyncio.Lock()
        # DM Topics: map of topic_name -> message_thread_id (populated at startup)
        self._dm_topics: Dict[str, int] = {}
        # Track forum chats where we've already registered bot commands
        self._forum_command_registered: set[int] = set()
        # Lock per la registrazione sicura dei comandi nei forum supergroup
        self._forum_lock = asyncio.Lock()
        # Status indicator: when enabled, the bot's short description (the line
        # shown under its name in the profile) is set to "Online" on connect and
        # "Offline" on clean disconnect, so users can tell whether the gateway is
        # up. Telegram bots have no real presence/online dot (that's a user-account
        # feature), so the short description is the closest available surface.
        # Off by default — this mutates the bot's GLOBAL profile, visible to all
        # users. Opt in via gateway config: extra.status_indicator: true, or set
        # custom strings via extra.status_online / extra.status_offline.
        self._status_indicator_enabled: bool = bool(
            self.config.extra.get("status_indicator", False)
        )
        self._status_online_text: str = str(
            self.config.extra.get("status_online", "Online")
        )
        self._status_offline_text: str = str(
            self.config.extra.get("status_offline", "Offline")
        )
        # DM Topics config from extra.dm_topics
        self._dm_topics_config: List[Dict[str, Any]] = self.config.extra.get("dm_topics", [])
        # Precomputed chat_ids that have DM topics configured (for O(1) root-DM ignore check)
        self._dm_topic_chat_ids: Set[str] = {
            str(e["chat_id"]) for e in self._dm_topics_config if "chat_id" in e
        }
        # Document size cap. Telegram's public Bot API caps getFile at 20MB; a
        # locally-hosted telegram-bot-api server (configured via extra.base_url)
        # raises that to 2GB, so the presence of base_url is the opt-in.
        self._max_doc_bytes: int = (
            2 * 1024 * 1024 * 1024
            if self.config.extra.get("base_url")
            else 20 * 1024 * 1024
        )
        # Interactive model picker state per chat
        self._model_picker_state: Dict[str, dict] = {}
        # Approval button state: message_id → session_key
        self._approval_state: Dict[int, str] = {}
        # Slash-confirm button state: confirm_id → session_key (for /reload-mcp
        # and any other slash-confirm prompts; see GatewayRunner._request_slash_confirm).
        self._slash_confirm_state: Dict[str, str] = {}
        # Clarify button state: clarify_id → session_key (for the clarify tool's
        # multiple-choice prompts; see GatewayRunner clarify_callback wiring).
        self._clarify_state: Dict[str, str] = {}
        # Notification mode for message sends.
        # "important" — only final responses, approvals, and slash confirmations
        #               trigger notifications; tool progress, streaming, status
        #               messages are delivered silently via disable_notification.
        #               This is the default — Telegram users found per-tool-call
        #               push notifications too noisy.
        # "all"       — every message triggers a push notification (legacy
        #               behavior; opt-in via display.platforms.telegram.notifications).
        self._notifications_mode: str = "important"
        # send_or_update_status() bookkeeping: {(chat_id, status_key) -> bot message_id}
        # Tracks status bubbles owned by this adapter so subsequent calls with the
        # same key edit the same message instead of appending new ones (#30045).
        self._status_message_ids: Dict[tuple, str] = {}
        # Last truncated mid-stream preview delivered per (chat_id, message_id).
        # Once an oversized streaming edit saturates at the 4096 preview cap,
        # every subsequent progressive edit truncates to the SAME text; sending
        # it again is a no-op that still burns Telegram's flood budget (~1
        # edit/0.8s × the rest of the stream ⇒ flood control with 200s+
        # penalties, hanging final delivery). Dedup here so a saturated preview
        # goes quiet until finalize. Bounded: entries are dropped on finalize.
        self._last_overflow_preview: Dict[tuple, str] = {}
        # Background task that runs post-connect housekeeping (command-menu
        # registration + DM-topic setup) off the connect path so a slow Bot
        # API call (e.g. a set_my_commands stall for certain tokens) cannot
        # blow the gateway's connect timeout (#46298).
        self._post_connect_task: Optional[asyncio.Task] = None

    def _mark_connected(self) -> None:
        self._drop_delayed_deliveries = False
        super()._mark_connected()

    def _mark_disconnected(self) -> None:
        self._drop_delayed_deliveries = True
        super()._mark_disconnected()

    def _set_fatal_error(self, code: str, message: str, *, retryable: bool) -> None:
        self._drop_delayed_deliveries = True
        super()._set_fatal_error(code, message, retryable=retryable)

    def _should_drop_delayed_delivery(self) -> bool:
        """True once teardown/fatal-error started — delayed flushes must drop.

        Buffered text/photo/media-group flushes sit behind an asyncio.sleep().
        If disconnect wins the race, dispatching them spawns an agent on a
        torn-down session, producing stale/duplicate deliveries.
        """
        return bool(getattr(self, "_drop_delayed_deliveries", False))

    def _notification_kwargs(
        self, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return disable_notification kwargs when the adapter is in silent mode.

        In "important" mode, all message sends are silently delivered
        (disable_notification=True) unless the caller explicitly requests a
        notification by setting ``metadata["notify"] = True``.
        """
        if getattr(self, "_notifications_mode", "important") != "important":
            return {}
        if (metadata or {}).get("notify"):
            return {}
        return {"disable_notification": True}

    def _is_callback_user_authorized(
        self,
        user_id: str,
        *,
        chat_id: Optional[str] = None,
        chat_type: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> bool:
        """Return whether a Telegram inline-button caller may perform gated actions."""
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return False

        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        auth_fn = getattr(runner, "_is_user_authorized", None)
        if callable(auth_fn):
            try:
                from gateway.session import SessionSource

                normalized_chat_type = str(chat_type or "dm").strip().lower() or "dm"
                if normalized_chat_type == "private":
                    normalized_chat_type = "dm"
                elif normalized_chat_type == "supergroup":
                    normalized_chat_type = "forum" if thread_id is not None else "group"

                source = SessionSource(
                    platform=Platform.TELEGRAM,
                    chat_id=str(chat_id or normalized_user_id),
                    chat_type=normalized_chat_type,
                    user_id=normalized_user_id,
                    user_name=str(user_name).strip() if user_name else None,
                    thread_id=str(thread_id) if thread_id is not None else None,
                )
                return bool(auth_fn(source))
            except Exception:
                logger.debug(
                    "[Telegram] Falling back to env-only callback auth for user %s",
                    normalized_user_id,
                    exc_info=True,
                )

        allowed_csv = os.getenv("TELEGRAM_ALLOWED_USERS", "").strip()
        if not allowed_csv:
            # Fail-closed: no allowlist means deny by default.
            # The runner auth path in _is_user_authorized() handles
            # GATEWAY_ALLOW_ALL_USERS; this fallback must not silently
            # allow everyone (fixes #24457).
            return os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}
        allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
        return "*" in allowed_ids or normalized_user_id in allowed_ids

    def _source_from_message_for_auth(self, message: Message):
        """Build the same Telegram source shape the gateway auth path expects.

        Resolves the identity to authorize from ``from_user`` for normal
        messages, falling back to ``sender_chat`` for channel posts (which
        carry no ``from_user``) so a removed/unauthorized channel cannot
        inject content via the broadcast path either.
        """
        from gateway.session import SessionSource

        user = getattr(message, "from_user", None)
        chat = getattr(message, "chat", None)
        user_id = str(getattr(user, "id", "")).strip() or None
        user_name = (
            str(getattr(user, "username", "") or getattr(user, "full_name", "") or "").strip()
            or None
        )
        # Channel posts have no from_user — authorize the sender chat instead.
        if not user_id:
            sender_chat = getattr(message, "sender_chat", None)
            if sender_chat is not None:
                user_id = str(getattr(sender_chat, "id", "")).strip() or None
                if not user_name:
                    user_name = (
                        str(getattr(sender_chat, "title", "") or "").strip() or None
                    )

        chat_id = str(getattr(chat, "id", "")).strip() or user_id
        chat_type = str(getattr(chat, "type", "dm")).strip().lower() or "dm"
        if chat_type == "private":
            chat_type = "dm"
        elif chat_type == "supergroup":
            thread_id_raw = getattr(message, "message_thread_id", None)
            is_topic_message = bool(getattr(message, "is_topic_message", False))
            is_forum_group = getattr(chat, "is_forum", False) is True
            chat_type = (
                "forum"
                if thread_id_raw is not None and (is_topic_message or is_forum_group)
                else "group"
            )

        thread_id = None
        thread_id_raw = getattr(message, "message_thread_id", None)
        if thread_id_raw is not None:
            is_topic_message = bool(getattr(message, "is_topic_message", False))
            is_forum_group = getattr(chat, "is_forum", False) is True
            if chat_type == "forum" and (is_topic_message or is_forum_group):
                thread_id = str(thread_id_raw)
            elif chat_type == "dm" and is_topic_message:
                thread_id = str(thread_id_raw)

        return SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id or "",
            chat_type=chat_type,
            user_id=user_id,
            user_name=user_name,
            thread_id=thread_id,
        )

    def _telegram_auth_env_configured(self) -> bool:
        """Return True when Telegram auth env vars make an early decision safe."""
        keys = (
            "TELEGRAM_ALLOWED_USERS",
            "TELEGRAM_GROUP_ALLOWED_USERS",
            "TELEGRAM_GROUP_ALLOWED_CHATS",
            "TELEGRAM_ALLOW_ALL_USERS",
            "GATEWAY_ALLOWED_USERS",
            "GATEWAY_ALLOW_ALL_USERS",
        )
        return any(os.getenv(key, "").strip() for key in keys)

    def _is_user_authorized_from_message(self, message: Message) -> bool:
        """Check if the sender of a Telegram message is authorized.

        Intake prefilter that runs BEFORE text batching, event construction,
        and unmentioned-group observation, so a removed/unauthorized user
        cannot inject prompt content into the agent path or the observed
        transcript (fixes #40863). It only rejects when it can make the same
        context-aware decision the runner would make. Unknown DMs with no
        allowlist still pass through so the normal pairing flow can run.
        """
        source = self._source_from_message_for_auth(message)
        user_id = source.user_id
        # No identity at all → genuine group service message (pin, delete,
        # new_chat_members, etc.). Defer to the cold path. Channel posts
        # without sender_chat already resolved to None above and fall here;
        # they carry no authorizable identity, so let the normal
        # _should_process_message gating handle them.
        if not user_id:
            return True

        # Adapter-level allow_from: when set, it is the sole authority.
        adapter_allow_from = self.config.extra.get("allow_from")
        if adapter_allow_from is not None:
            allowed = {str(u).strip() for u in adapter_allow_from if str(u).strip()}
            return user_id in allowed or "*" in allowed

        # Test/custom injection only. The class method named
        # _is_callback_user_authorized is for inline button callbacks and must
        # not be treated as a user-id-only shortcut for real messages — only
        # honor an instance-level override (set in tests).
        callback_auth = self.__dict__.get("_is_callback_user_authorized")
        if callable(callback_auth):
            try:
                return bool(
                    callback_auth(
                        user_id,
                        chat_id=source.chat_id,
                        chat_type=source.chat_type,
                        thread_id=source.thread_id,
                        user_name=source.user_name,
                    )
                )
            except Exception:
                pass

        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        auth_fn = getattr(runner, "_is_user_authorized", None)
        if callable(auth_fn):
            # Only make an early decision via the runner when an allowlist
            # actually exists; otherwise unknown DMs must reach the pairing
            # flow rather than being default-denied here.
            if not self._telegram_auth_env_configured():
                return True
            try:
                return bool(auth_fn(source))
            except Exception:
                logger.debug(
                    "[Telegram] Falling back to env-only auth for user %s",
                    user_id,
                    exc_info=True,
                )

        allowed_csv = os.getenv("TELEGRAM_ALLOWED_USERS", "").strip()
        if not allowed_csv:
            return True
        allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
        return "*" in allowed_ids or user_id in allowed_ids

    @classmethod
    def _metadata_thread_id(cls, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if not metadata:
            return None
        thread_id = metadata.get("thread_id") or metadata.get("message_thread_id")
        return str(thread_id) if thread_id is not None else None

    @classmethod
    def _metadata_direct_messages_topic_id(cls, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if not metadata:
            return None
        topic_id = metadata.get("direct_messages_topic_id") or metadata.get("telegram_direct_messages_topic_id")
        return str(topic_id) if topic_id is not None else None

    @classmethod
    def _metadata_reply_to_message_id(cls, metadata: Optional[Dict[str, Any]]) -> Optional[int]:
        if not metadata:
            return None
        reply_to = metadata.get("telegram_reply_to_message_id")
        return int(reply_to) if reply_to is not None else None

    @classmethod
    def _is_private_dm_topic_send(
        cls,
        chat_id: str,
        thread_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> bool:
        if cls._metadata_direct_messages_topic_id(metadata) is not None:
            return bool(
                metadata
                and metadata.get("telegram_dm_topic_reply_fallback")
                and cls._metadata_reply_to_message_id(metadata) is not None
            )
        if metadata and metadata.get("telegram_dm_topic_created_for_send"):
            return False
        return bool(
            thread_id
            and metadata
            and metadata.get("telegram_dm_topic_reply_fallback")
        )

    @staticmethod
    def _dm_topic_missing_anchor_error() -> str:
        return "Telegram DM topic delivery requires a reply anchor; refusing to send outside the requested topic"

    @classmethod
    def _reply_to_message_id_for_send(
        cls,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
        reply_to_mode: Optional[str] = None,
    ) -> Optional[int]:
        if reply_to:
            return int(reply_to)
        if metadata and metadata.get("telegram_dm_topic_reply_fallback"):
            if reply_to_mode == "off":
                return None
            return cls._metadata_reply_to_message_id(metadata)
        return None

    @classmethod
    def _thread_kwargs_for_send(
        cls,
        chat_id: str,
        thread_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
        reply_to_message_id: Optional[int] = None,
        reply_to_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return Telegram send kwargs for forum and direct-message topic routing.

        Supergroup/forum topics use ``message_thread_id``. True Bot API Direct
        Messages topics can opt in with explicit ``direct_messages_topic_id``
        metadata. Hermes-created private-chat topic lanes are marked with
        ``telegram_dm_topic_reply_fallback``. Live replies send the private
        topic thread id together with a reply anchor; synthetic/resumed sends
        without an anchor use ``direct_messages_topic_id`` when metadata has it.
        ``message_thread_id`` alone can render outside the visible lane.

        When ``reply_to_mode`` is ``"off"``, the reply anchor is suppressed for
        DM topic fallback sends while preserving the ``message_thread_id`` so
        the message still lands in the correct topic.
        """
        if metadata and metadata.get("telegram_dm_topic_reply_fallback"):
            if reply_to_mode == "off":
                return {"message_thread_id": cls._message_thread_id_for_send(thread_id)}
            if reply_to_message_id is None:
                reply_to_message_id = cls._metadata_reply_to_message_id(metadata)
            if reply_to_message_id is None:
                direct_topic_id = cls._metadata_direct_messages_topic_id(metadata)
                if direct_topic_id is not None:
                    return {
                        "message_thread_id": None,
                        "direct_messages_topic_id": int(direct_topic_id),
                    }
                return {}
            return {"message_thread_id": cls._message_thread_id_for_send(thread_id)}
        direct_topic_id = cls._metadata_direct_messages_topic_id(metadata)
        if direct_topic_id is not None:
            return {
                "message_thread_id": None,
                "direct_messages_topic_id": int(direct_topic_id),
            }
        return {"message_thread_id": cls._message_thread_id_for_send(thread_id)}

    @classmethod
    def _message_thread_id_for_send(cls, thread_id: Optional[str]) -> Optional[int]:
        if not thread_id or str(thread_id) == cls._GENERAL_TOPIC_THREAD_ID:
            return None
        return int(thread_id)

    @classmethod
    def _message_thread_id_for_typing(cls, thread_id: Optional[str]) -> Optional[int]:
        # Asymmetric with _message_thread_id_for_send on purpose. Telegram's
        # sendMessage and sendChatAction treat thread id "1" (the forum General
        # topic) differently: sends reject message_thread_id=1 and must omit it,
        # but sendChatAction needs message_thread_id=1 to place the typing
        # bubble in the General topic (omitting it hides the bubble entirely
        # from the client's view of that topic). Preserve the real id here —
        # sends still map "1" → None via _message_thread_id_for_send.
        if not thread_id:
            return None
        return int(thread_id)

    @staticmethod
    def _is_thread_not_found_error(error: Exception) -> bool:
        return "thread not found" in str(error).lower()

    def _prune_stale_dm_topic_binding(
        self, chat_id: Any, thread_id: Any,
    ) -> None:
        """Drop the stale ``telegram_dm_topic_bindings`` row for a
        topic Telegram has confirmed deleted.

        Without this prune the recovery logic in
        ``gateway.run._recover_telegram_topic_thread_id`` keeps
        steering future inbound messages to the dead thread (the
        bug behind #31501 — tool progress, approvals, replies all
        end up in the wrong place even though the user has moved
        on to a fresh topic).  Best-effort: we never raise from a
        send-fallback path — a failed cleanup must not turn into a
        failed user-facing send.
        """
        if chat_id is None or thread_id is None:
            return
        store = getattr(self, "_session_store", None)
        if store is None:
            return
        db = getattr(store, "_db", None)
        if db is None or not hasattr(db, "delete_telegram_topic_binding"):
            return
        try:
            removed = db.delete_telegram_topic_binding(
                chat_id=str(chat_id), thread_id=str(thread_id),
            )
        except Exception:
            logger.debug(
                "[%s] delete_telegram_topic_binding failed for "
                "chat=%s thread=%s — skipping prune",
                self.name, chat_id, thread_id, exc_info=True,
            )
            return
        if removed:
            logger.info(
                "[%s] Pruned stale Telegram DM topic binding "
                "chat=%s thread=%s (Bot API: thread not found)",
                self.name, chat_id, thread_id,
            )

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
        cls,
        error: Exception,
        metadata: Optional[Dict[str, Any]],
        reply_to_message_id: Optional[int],
    ) -> bool:
        """True when a DM-topic send should be retried with routing stripped.

        Two cases trigger the retry:

        1. The original anchor-stale case — the reply target was deleted, so
           Bot API returns "message to be replied not found". The retry drops
           the reply anchor and the topic id together.

        2. The synthetic-event case (added when #27937 introduced
           ``direct_messages_topic_id`` fallback for sends without an anchor):
           if Bot API rejects the topic id itself with any BadRequest that
           mentions topic/thread routing, we retry without routing rather
           than dropping the message.
        """
        if not (metadata and metadata.get("telegram_dm_topic_reply_fallback")):
            return False
        if not cls._is_bad_request_error(error):
            return False
        err_lower = str(error).lower()
        if reply_to_message_id is not None and "message to be replied not found" in err_lower:
            return True
        # Synthetic / resumed sends route via ``direct_messages_topic_id``
        # instead of a reply anchor. If Telegram rejects the topic id, fall
        # back to a plain DM send.
        if metadata.get("direct_messages_topic_id"):
            topic_markers = (
                "direct_messages_topic",
                "message thread not found",
                "thread not found",
                "topic_closed",
                "topic_deleted",
                "topic not found",
            )
            if any(marker in err_lower for marker in topic_markers):
                return True
        return False

    async def _send_with_dm_topic_reply_anchor_retry(
        self,
        send_fn: Any,
        send_kwargs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
        reply_to_message_id: Optional[int],
        media_label: str,
        reset_media: Optional[Any] = None,
    ) -> Any:
        """Retry stale private-topic media replies once without the topic anchor."""
        try:
            return await send_fn(**send_kwargs)
        except Exception as send_err:
            if not self._should_retry_without_dm_topic_reply_anchor(
                send_err,
                metadata,
                reply_to_message_id,
            ):
                raise
            logger.warning(
                "[%s] Reply target deleted for Telegram %s, "
                "retrying without reply/topic anchor: %s",
                self.name,
                media_label,
                send_err,
            )
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
            or "another bot instance is running" in text
        )

    @staticmethod
    def _looks_like_network_error(error: Exception) -> bool:
        """Return True for transient transport failures that warrant reconnect."""
        name = error.__class__.__name__.lower()
        if name in {"badrequest", "invalidtoken", "forbidden", "retryafter"}:
            return False
        if name in {"networkerror", "timedout", "connectionerror"}:
            return True
        try:
            from telegram.error import (
                BadRequest,
                Forbidden,
                InvalidToken,
                NetworkError,
                RetryAfter,
                TimedOut,
            )
            if isinstance(error, (BadRequest, InvalidToken, Forbidden, RetryAfter)):
                return False
            if isinstance(error, (NetworkError, TimedOut)):
                return True
        except ImportError:
            pass
        return isinstance(error, OSError)

    @staticmethod
    def _looks_like_connect_timeout(error: Exception) -> bool:
        """Return True when a Telegram TimedOut wraps a connect-timeout.

        A plain Telegram TimedOut may mean the request reached Telegram and
        should not be re-sent. A ConnectTimeout means the TCP connection was
        never established, so retrying is safe and prevents silent drops.
        """
        seen: set[int] = set()
        stack: list[BaseException] = [error]
        while stack:
            cur = stack.pop()
            ident = id(cur)
            if ident in seen:
                continue
            seen.add(ident)
            name = cur.__class__.__name__.lower()
            text = str(cur).lower()
            if "connecttimeout" in name or "connect timeout" in text or "connect timed out" in text:
                return True
            cause = getattr(cur, "__cause__", None)
            context = getattr(cur, "__context__", None)
            if cause is not None:
                stack.append(cause)
            if context is not None:
                stack.append(context)
        return False

    @staticmethod
    def _looks_like_pool_timeout(error: Exception) -> bool:
        """Return True when a Telegram TimedOut wraps an httpx pool timeout.

        PTB converts ``httpx.PoolTimeout`` into ``telegram.error.TimedOut`` with
        a message that explicitly states the request was *not* sent
        (``"Pool timeout: All connections in the connection pool are occupied.
        Request was *not* sent to Telegram."``). Because the request never left
        the process, re-sending is safe and cannot duplicate -- the opposite of
        a generic TimedOut, which may have reached Telegram. We match the
        wrapped ``httpx.PoolTimeout`` class as well as the message string so the
        check survives PTB message-wording changes.
        """
        seen: set[int] = set()
        stack: list[BaseException] = [error]
        while stack:
            cur = stack.pop()
            ident = id(cur)
            if ident in seen:
                continue
            seen.add(ident)
            name = cur.__class__.__name__.lower()
            text = str(cur).lower()
            if "pooltimeout" in name or "pool timeout" in text or (
                "connection pool" in text and "occupied" in text
            ):
                return True
            cause = getattr(cur, "__cause__", None)
            context = getattr(cur, "__context__", None)
            if cause is not None:
                stack.append(cause)
            if context is not None:
                stack.append(context)
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
        self,
        key: str,
        default: float,
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
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

    # ------------------------------------------------------------------
    # Bot API 10.1 Rich Messages (sendRichMessage)
    #
    # Final / new-message replies opportunistically use sendRichMessage with
    # the RAW agent markdown so richer constructs (tables, task lists,
    # collapsible details, math, ...) render natively. The legacy MarkdownV2
    # send() path stays as the fallback for unsupported/oversized content and
    # older PTB/clients. Streaming edits stay on Hermes' existing MarkdownV2
    # edit path for now; finalization can re-send as rich and delete the stale
    # preview until rich_message edit support is wired directly.
    # ------------------------------------------------------------------
    def _content_fits_rich_limits(self, content: str) -> bool:
        """Cheap pre-check for the one hard rich limit we can count locally.

        Only the 32,768 UTF-8 character text cap is enforced here. Other Bot API
        rich limits (500 blocks, 16 nesting levels, 20 table columns, ...) are
        not pre-counted; if exceeded Telegram returns a BadRequest, which
        :meth:`_is_rich_fallback_error` classifies as permanent so the send
        degrades to the legacy chunking path.
        """
        return len(content) <= self.RICH_MESSAGE_MAX_CHARS

    def _bot_supports_rich(self) -> bool:
        """True when the bound bot can issue raw ``sendRichMessage`` calls.

        Gates on ``do_api_request`` being an *async* callable. The real
        ``telegram.Bot.do_api_request`` is a coroutine function; test doubles
        that opt into rich set it to an ``AsyncMock`` (also a coroutine
        function). Plain ``MagicMock`` bots expose a *sync* auto-child and
        ``SimpleNamespace`` bots lack the attribute entirely — both resolve to
        ``False`` here, so the legacy path is used unchanged.
        """
        return inspect.iscoroutinefunction(getattr(self._bot, "do_api_request", None))

    _RICH_DETAILS_RE = re.compile(r"<details\b[^>]*>.*?</details>", re.IGNORECASE | re.DOTALL)
    _RICH_MATH_IN_DETAILS_RE = re.compile(
        r"(\$\$.*?\$\$|"
        r"\\\[.*?\\\]|"
        r"\\\(.*?\\\)|"
        r"\\(?:sum|frac|alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|"
        r"int|prod|sqrt|lim|infty|begin\{(?:equation|align|matrix|cases)\}))",
        re.IGNORECASE | re.DOTALL,
    )
    _RICH_CJK_RE = re.compile(
        "["
        "\u3040-\u30ff"  # Hiragana, Katakana
        "\u3400-\u4dbf"  # CJK Extension A
        "\u4e00-\u9fff"  # CJK Unified Ideographs
        "\uac00-\ud7af"  # Hangul syllables
        "\uf900-\ufaff"  # CJK Compatibility Ideographs
        "\U00020000-\U000323af"  # CJK extensions and compatibility supplement
        "]"
    )

    def _has_telegram_desktop_details_math_crash_shape(self, content: str) -> bool:
        """Return True for rich-message details+math content that crashes TDesktop.

        Telegram Desktop 6.9.1 can crash while rendering Bot API 10.1 rich
        messages containing math inside a collapsible details block
        (telegramdesktop/tdesktop#30808). The Bot API accepts the payload, so
        Hermes must skip rich delivery up front and use the legacy MarkdownV2
        path until affected Desktop clients age out.
        """
        if not content:
            return False
        for details_block in self._RICH_DETAILS_RE.findall(content):
            if self._RICH_MATH_IN_DETAILS_RE.search(details_block):
                return True
        return False

    def _has_telegram_desktop_cjk_rich_garble_shape(self, content: str) -> bool:
        """Return True for CJK content that current TDesktop rich drafts garble.

        Telegram Mac/Desktop Bot API 10.1 rich-message rendering currently
        leaves overlapping draft/overlay glyph artifacts for CJK text (#47653).
        The legacy MarkdownV2 path renders the same text cleanly, so skip rich
        delivery up front until affected clients age out.
        """
        return bool(content and self._RICH_CJK_RE.search(content))

    def _needs_rich_rendering(self, content: str) -> bool:
        """Return True for markdown constructs that the legacy path degrades.

        Keep ordinary replies on the pre-rich MarkdownV2 path so Telegram
        clients render a consistent font weight/spacing. The rich endpoint is
        reserved for constructs where raw markdown materially improves output:
        pipe tables (MarkdownV2 has no table syntax and rewrites them into
        bullet lists), GFM task lists, collapsible ``<details>`` blocks, and
        block math.  Adapted from #45995 (@YonganZhang).
        """
        if not content:
            return False
        if any(_TABLE_SEPARATOR_RE.match(line) for line in content.splitlines()):
            return True
        if re.search(r"(?m)^\s*[-*]\s+\[[ xX]\]\s+", content):
            return True
        if re.search(r"(?m)^<details\b|^</details>|^<summary\b|^</summary>", content):
            return True
        if "$$" in content:
            return True
        return False

    def _content_is_pipe_table_primary(self, content: str) -> bool:
        """True when pipe tables are the only rich construct in *content*.

        Tables are auto-routed to ``sendRichMessage`` even when the full
        ``rich_messages`` opt-in is off — MarkdownV2 has no table syntax and
        the legacy path rewrites them into bullet lists, which reads like a
        regression when users enable Telegram Topics and expect native tables.
        Task lists, ``<details>``, and block math still require the full opt-in.
        """
        if not content or not any(
            _TABLE_SEPARATOR_RE.match(line) for line in content.splitlines()
        ):
            return False
        if re.search(r"(?m)^\s*[-*]\s+\[[ xX]\]\s+", content):
            return False
        if re.search(r"(?m)^<details\b|^</details>|^<summary\b|^</summary>", content):
            return False
        if "$$" in content:
            return False
        return True

    def _rich_delivery_enabled(self, content: str) -> bool:
        """Whether rich delivery is allowed for this payload."""
        return bool(
            getattr(self, "_rich_messages_enabled", True)
            or self._content_is_pipe_table_primary(content)
        )

    def _rich_eligible(self, content: str) -> bool:
        """Capability/content eligibility for rich, ignoring ``expect_edits``.

        Shared core of :meth:`_should_attempt_rich` minus the per-call
        ``expect_edits`` metadata gate.  The rich EDIT-finalize path
        (:meth:`_try_edit_rich`) needs this: a streamed preview is sent with
        ``expect_edits=True`` to stay on the editable path mid-stream, but the
        FINAL edit should still upgrade to rich when the content warrants it.
        """
        return bool(
            self._rich_delivery_enabled(content)
            and not getattr(self, "_rich_send_disabled", False)
            and content
            and content.strip()
            and self._needs_rich_rendering(content)
            and not self._has_telegram_desktop_details_math_crash_shape(content)
            and not self._has_telegram_desktop_cjk_rich_garble_shape(content)
            and self._content_fits_rich_limits(content)
            and self._bot_supports_rich()
        )

    def _should_attempt_rich(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        return bool(
            not (metadata or {}).get("expect_edits")
            and self._rich_eligible(content)
        )

    def prefers_fresh_final_streaming(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Whether to replace a streamed preview with a fresh rich final.

        Disabled for Telegram. The fresh-final path briefly shows two copies of
        the final answer, then deletes the streaming preview after the rich send
        succeeds — it looks like duplicate delivery at the end of every streamed
        turn (the reason #46206 reverted it).  Rich finalize is instead handled
        by editing the existing preview in place via Bot API 10.1's
        ``editMessageText`` ``rich_message`` parameter (see
        :meth:`_try_edit_rich`), so no fresh re-send / delete is needed.
        """
        return False

    def streaming_overflow_limit(self) -> Optional[int]:
        """Allow the stream consumer to accumulate up to the rich-message cap
        before splitting, so a reply that fits one ``sendRichMessage`` /
        ``sendRichMessageDraft`` isn't fragmented at the 4,096 MarkdownV2 limit.

        Gated on the same rich capability as the send path (minus the
        content-length check — raising that cap is the whole point): rich not
        latched off and the bot exposes an async ``do_api_request``.  Returns
        ``None`` (→ legacy 4,096 limit) when rich isn't available, so non-rich
        streams split exactly as before.
        """
        if (
            getattr(self, "_rich_messages_enabled", True)
            and not getattr(self, "_rich_send_disabled", False)
            and self._bot_supports_rich()
        ):
            return self.RICH_MESSAGE_MAX_CHARS
        return None

    def _rich_message_payload(
        self, content: str, *, skip_entity_detection: bool = False
    ) -> Dict[str, Any]:
        """Build the ``InputRichMessage`` object from RAW markdown.

        Never pass ``format_message(content)`` here — that converts to
        MarkdownV2 and would escape/destroy rich syntax like table pipes.

        Single newlines are normalized to Markdown hard breaks so that
        multi-line content (slash-command lists, etc.) renders correctly
        in the rich-message path.  See ``_rich_normalize_linebreaks``.
        """
        payload: Dict[str, Any] = {"markdown": _rich_normalize_linebreaks(content)}
        if skip_entity_detection:
            payload["skip_entity_detection"] = True
        return payload

    def _is_rich_capability_error(self, exc: Exception) -> bool:
        """True ⇒ the rich endpoint itself is unavailable (old PTB/server).

        These latch rich off for the rest of the adapter's life — retrying is
        pointless and would cost a failed roundtrip on every send. Per-message
        rejections (BadRequest from a parser/limit issue) are NOT capability
        errors: the next message may be fine.
        """
        name = exc.__class__.__name__.lower()
        if name in {"endpointnotfound", "invalidtoken"}:
            return True
        if isinstance(exc, (AttributeError, TypeError, NotImplementedError)):
            return True
        if getattr(exc, "error_code", None) == 404:
            return True
        s = str(exc).lower()
        if ("method" in s or "endpoint" in s) and (
            "not found" in s or "does not exist" in s
        ):
            return True
        return "no such method" in s

    def _is_rich_fallback_error(self, exc: Exception) -> bool:
        """True ⇒ permanent/capability error ⇒ safe to fall back to legacy.

        Conservative on purpose: only clearly-permanent failures (BadRequest,
        capability errors, unknown/unsupported endpoint) qualify. Everything
        else is treated as transient — the rich request may have reached
        Telegram, so we must NOT legacy-resend and risk a duplicate.
        """
        if self._is_bad_request_error(exc):
            return True
        if self._is_rich_capability_error(exc):
            return True
        s = str(exc).lower()
        return "unsupported" in s or "not implemented" in s

    def _compute_single_send_routing(
        self,
        chat_id: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
        thread_id: Optional[str],
    ) -> Optional[tuple]:
        """Routing for a single (rich) send — mirrors send()'s index-0 block.

        Returns ``(reply_to_id, thread_kwargs)``, or ``None`` to signal "skip
        rich, let the legacy path handle it" — used for the DM-topic fail-loud
        case so the legacy path stays the single source of the refuse result.
        """
        metadata_reply_to = self._metadata_reply_to_message_id(metadata)
        private_dm_topic_send = self._is_private_dm_topic_send(chat_id, thread_id, metadata)
        dm_topic_reply_to_off = (
            private_dm_topic_send
            and self._reply_to_mode == "off"
            and bool(metadata and metadata.get("telegram_dm_topic_reply_fallback"))
        )
        reply_to_source = reply_to or (
            str(metadata_reply_to)
            if private_dm_topic_send and metadata_reply_to is not None
            else None
        )
        if private_dm_topic_send:
            should_thread = reply_to_source is not None and self._reply_to_mode != "off"
        else:
            should_thread = self._should_thread_reply(reply_to_source, 0)
        reply_to_id = int(reply_to_source) if should_thread and reply_to_source else None
        thread_kwargs = self._thread_kwargs_for_send(
            chat_id,
            thread_id,
            metadata,
            reply_to_message_id=reply_to_id,
            reply_to_mode=self._reply_to_mode,
        )
        if private_dm_topic_send and reply_to_id is None and not dm_topic_reply_to_off:
            # Refusing to send outside the requested DM topic — defer to the
            # legacy path, which returns the canonical fail-loud SendResult.
            # Exception: synthetic/resumed topic sends that route via
            # ``direct_messages_topic_id`` do not need a reply anchor.
            if not thread_kwargs.get("direct_messages_topic_id"):
                return None
        return reply_to_id, thread_kwargs

    async def _try_send_rich(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[SendResult]:
        """Attempt a single ``sendRichMessage`` send.

        Returns a :class:`SendResult` (success, or a transient failure that the
        caller must NOT legacy-resend), or ``None`` to signal "fall back to the
        legacy MarkdownV2 path" (permanent/capability error or DM-topic skip).
        """
        thread_id = self._metadata_thread_id(metadata)
        routing = self._compute_single_send_routing(chat_id, reply_to, metadata, thread_id)
        if routing is None:
            return None
        reply_to_id, thread_kwargs = routing

        payload: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id),
            "rich_message": self._rich_message_payload(content),
        }
        # Only forward non-None routing keys: when direct_messages_topic_id is
        # present _thread_kwargs_for_send pairs it with message_thread_id=None,
        # which must not be sent as a stray field on the raw endpoint.
        payload.update({k: v for k, v in thread_kwargs.items() if v is not None})
        payload.update(self._notification_kwargs(metadata))
        if getattr(self, "_disable_link_previews", False):
            payload["link_preview_options"] = {"is_disabled": True}
        if reply_to_id is not None:
            # Spec: sendRichMessage takes reply_parameters (ReplyParameters
            # object), NOT the legacy reply_to_message_id scalar. Unknown
            # params are silently ignored by the Bot API, so the scalar would
            # quietly drop the reply anchor instead of erroring.
            payload["reply_parameters"] = {"message_id": reply_to_id}

        try:
            # Take the raw Bot API result (dict under real PTB). Passing
            # return_type=Message would make PTB deserialize a Bot API 10.1
            # response shape it does not fully model yet; a post-delivery parse
            # error must not be mistaken for a sendable failure.
            msg = await self._bot.do_api_request(
                "sendRichMessage", api_kwargs=payload
            )
        except Exception as exc:
            if self._is_rich_fallback_error(exc):
                if self._is_rich_capability_error(exc):
                    # Endpoint missing (old PTB/server) — latch rich off so
                    # every later send doesn't pay a doomed extra roundtrip.
                    self._rich_send_disabled = True
                logger.debug(
                    "[%s] sendRichMessage rejected (%s) — falling back to MarkdownV2",
                    self.name, exc,
                )
                return None
            # Transient / network / unknown: the request may have reached
            # Telegram. Do NOT legacy-resend (duplicate risk); surface a
            # failure with retry semantics mirroring the legacy send() except.
            err_str = str(exc).lower()
            try:
                from telegram.error import TimedOut as _TimedOut
            except (ImportError, AttributeError):
                _TimedOut = None
            is_timeout = (_TimedOut and isinstance(exc, _TimedOut)) or "timed out" in err_str
            is_connect_timeout = self._looks_like_connect_timeout(exc)
            # Extract server-requested retry_after for flood control so the
            # base retry layer honors Telegram's backoff instead of its own
            # short exponential schedule.
            _retry_after = getattr(exc, "retry_after", None)
            if _retry_after is None:
                import re as _re
                _m = _re.search(r"retry\s+(?:in\s+)?(\d+)", err_str, _re.IGNORECASE)
                if _m:
                    _retry_after = float(_m.group(1))
            safe_error = _redact_telegram_error_text(exc)
            logger.warning(
                "[%s] sendRichMessage transient failure (no legacy resend): %s",
                self.name, safe_error,
            )
            return SendResult(
                success=False,
                error=safe_error,
                retryable=(is_connect_timeout or not is_timeout),
                retry_after=_retry_after,
            )

        message_id = None
        if isinstance(msg, dict):
            message_id = msg.get("message_id")
            if message_id is None:
                message_id = (msg.get("result") or {}).get("message_id")
        else:
            message_id = getattr(msg, "message_id", None)
        if message_id is not None:
            # Telegram won't echo rich content in reply_to_message, so remember
            # what we sent — replies to this message resolve via this index.
            try:
                from gateway import rich_sent_store
                rich_sent_store.record(str(chat_id), str(message_id), content)
            except Exception:
                pass
        return SendResult(
            success=True,
            message_id=str(message_id) if message_id is not None else None,
        )

    async def _try_edit_rich(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SendResult]:
        """Edit an existing message in place as a rich message (Bot API 10.1).

        Uses ``editMessageText`` with the ``rich_message`` parameter so a
        streamed preview can finalize as rich (tables/task lists/details/math)
        WITHOUT a fresh send + delete — no duplicate preview.  Mirrors
        :meth:`_try_send_rich`'s error contract:

        - success → ``SendResult(success=True, message_id=...)``
        - permanent / capability error → ``None`` (caller falls back to the
          legacy MarkdownV2 edit; capability errors latch rich off)
        - transient / unknown → ``SendResult(success=False)`` with retry
          semantics (the message may already be edited; do NOT legacy-resend)
        """
        payload: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id),
            "message_id": int(message_id),
            "rich_message": self._rich_message_payload(content),
        }
        thread_id = self._metadata_thread_id(metadata)
        thread_kwargs = self._thread_kwargs_for_send(
            chat_id,
            thread_id,
            metadata,
            reply_to_message_id=None,
            reply_to_mode=self._reply_to_mode,
        )
        payload.update({k: v for k, v in thread_kwargs.items() if v is not None})
        if getattr(self, "_disable_link_previews", False):
            payload["link_preview_options"] = {"is_disabled": True}
        try:
            # Raw Bot API result; do not request return_type=Message (PTB does
            # not fully model the 10.1 response shape yet — a post-edit parse
            # error must not be mistaken for a failed edit).
            await self._bot.do_api_request("editMessageText", api_kwargs=payload)
        except Exception as exc:
            if self._is_rich_fallback_error(exc):
                if self._is_rich_capability_error(exc):
                    self._rich_send_disabled = True
                # "Message is not modified" — content identical to the current
                # rich message; treat as a successful no-op so the caller does
                # not fall through to a redundant legacy edit.
                if "not modified" in str(exc).lower():
                    return SendResult(success=True, message_id=message_id)
                logger.debug(
                    "[%s] rich editMessageText rejected (%s) — falling back to MarkdownV2 edit",
                    self.name, exc,
                )
                return None
            if "not modified" in str(exc).lower():
                return SendResult(success=True, message_id=message_id)
            err_str = str(exc).lower()
            try:
                from telegram.error import TimedOut as _TimedOut
            except (ImportError, AttributeError):
                _TimedOut = None
            is_timeout = (_TimedOut and isinstance(exc, _TimedOut)) or "timed out" in err_str
            is_connect_timeout = self._looks_like_connect_timeout(exc)
            safe_error = _redact_telegram_error_text(exc)
            logger.warning(
                "[%s] rich editMessageText transient failure (no legacy resend): %s",
                self.name, safe_error,
            )
            return SendResult(
                success=False,
                error=safe_error,
                retryable=(is_connect_timeout or not is_timeout),
            )
        # Telegram won't echo rich content for messages that predate the bot's
        # first rich send, so mirror the fresh-send index here too: a streamed
        # final finalized via editMessageText is otherwise never recorded, and
        # replies to it would have no native echo to recover from.
        try:
            from gateway import rich_sent_store
            rich_sent_store.record(str(chat_id), str(message_id), content)
        except Exception:
            pass
        return SendResult(success=True, message_id=message_id)

    def _should_attempt_rich_draft(self, content: str) -> bool:
        return bool(
            getattr(self, "_rich_messages_enabled", True)
            and getattr(self, "_rich_drafts_enabled", False)
            and not getattr(self, "_rich_send_disabled", False)
            and not getattr(self, "_rich_draft_disabled", False)
            and content
            and content.strip()
            and not self._has_telegram_desktop_details_math_crash_shape(content)
            and not self._has_telegram_desktop_cjk_rich_garble_shape(content)
            and self._content_fits_rich_limits(content)
            and self._bot_supports_rich()
        )

    async def _try_send_rich_draft(
        self,
        chat_id: str,
        draft_id: int,
        content: str,
        metadata: Optional[Dict[str, Any]],
    ) -> bool:
        """Emit one ``sendRichMessageDraft`` preview frame; True on success.

        Draft frames are ephemeral and overwritten by the next frame / the
        final ``sendRichMessage``, so a duplicate or lost rich draft is
        harmless — any failure simply returns False and the caller renders the
        legacy plain-text draft. A permanent/capability failure additionally
        latches ``_rich_draft_disabled`` so later frames skip the rich attempt.
        """
        payload: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id),
            "draft_id": int(draft_id),
            "rich_message": self._rich_message_payload(content),
        }
        thread_id = self._metadata_thread_id(metadata)
        if thread_id is not None:
            payload["message_thread_id"] = int(thread_id)
        try:
            ok = await self._bot.do_api_request("sendRichMessageDraft", api_kwargs=payload)
            return bool(ok)
        except Exception as exc:
            if self._is_rich_capability_error(exc):
                self._rich_draft_disabled = True
                logger.debug(
                    "[%s] sendRichMessageDraft unsupported (%s) — using legacy drafts",
                    self.name, exc,
                )
            else:
                logger.debug(
                    "[%s] sendRichMessageDraft transient failure (%s) — legacy draft this frame",
                    self.name, exc,
                )
            return False

    async def _drain_polling_connections(self) -> None:
        """Reset the httpx connection pool used for getUpdates polling.

        Network errors (especially through proxies like sing-box) can leave
        httpx connections in a half-closed state that still occupy pool slots.
        After enough reconnect cycles the pool fills up entirely, causing
        ``Pool timeout: All connections in the connection pool are occupied.``

        We reset ONLY ``_request[0]`` (the getUpdates request) — the general
        request (``_request[1]``) is left untouched so concurrent
        ``send_message`` / ``edit_message`` calls are never interrupted.

        Implementation note: accesses ``Bot._request[0]`` which is the
        get-updates ``BaseRequest`` in the PTB 22.x internal tuple
        ``(get_updates_request, general_request)``.  There is no public
        accessor for the polling request; review if upgrading to PTB 23+.
        """
        if not (self._app and self._app.bot):
            return
        try:
            # PTB 22.x: _request is a (get_updates, general) tuple;
            # no public accessor exists for the polling request.
            polling_req = self._app.bot._request[0]  # noqa: SLF001
        except Exception:
            return
        try:
            await polling_req.shutdown()
        except Exception:
            logger.debug(
                "[%s] Polling request shutdown failed (non-fatal)",
                self.name, exc_info=True,
            )
        try:
            await polling_req.initialize()
            logger.debug(
                "[%s] Polling request pool drained before reconnect", self.name
            )
        except Exception:
            logger.debug(
                "[%s] Polling request re-initialize failed (non-fatal)",
                self.name, exc_info=True,
            )

    def _get_general_request_drain_lock(self) -> asyncio.Lock:
        lock = getattr(self, "_general_request_drain_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._general_request_drain_lock = lock
        return lock

    async def _drain_general_connections_after_pool_timeout(self) -> None:
        """Reset the Bot API request pool after a confirmed send pool timeout.

        ``send_message`` uses PTB's general request pool (``_request[1]``).
        When httpx reports that this pool is exhausted, PTB says the request
        was not sent, so it is safe to reset the wedged pool before retrying.
        """
        bot = getattr(getattr(self, "_app", None), "bot", None)
        if bot is None:
            bot = getattr(self, "_bot", None)
        if bot is None:
            return
        try:
            # PTB 22.x: _request is (get_updates_request, general_request).
            general_req = bot._request[1]  # noqa: SLF001
        except Exception:
            return
        async with self._get_general_request_drain_lock():
            try:
                await general_req.shutdown()
            except Exception:
                logger.debug(
                    "[%s] General request shutdown failed after pool timeout (non-fatal)",
                    self.name, exc_info=True,
                )
            try:
                await general_req.initialize()
                logger.warning(
                    "[%s] General request pool drained after Telegram pool timeout",
                    self.name,
                )
            except Exception:
                logger.debug(
                    "[%s] General request re-initialize failed after pool timeout (non-fatal)",
                    self.name, exc_info=True,
                )

    def _schedule_polling_recovery(self, error: Exception, *, reason: str) -> None:
        """Schedule polling recovery without failing gateway startup.

        A Telegram bootstrap failure (deleteWebhook / initial start_polling)
        caused by a transient network error should degrade only the Telegram
        adapter: the gateway process stays alive and the existing reconnect
        ladder (``_handle_polling_network_error``) recovers in the background.
        """
        if self.has_fatal_error:
            return
        if self._polling_error_task and not self._polling_error_task.done():
            logger.debug(
                "[%s] Telegram polling recovery already scheduled; ignoring %s: %s",
                self.name, reason, error,
            )
            return
        self._send_path_degraded = True
        logger.warning(
            "[%s] Telegram polling degraded (%s); gateway stays alive and will retry. Error: %s",
            self.name, reason, error,
        )
        loop = asyncio.get_running_loop()
        self._polling_error_task = loop.create_task(self._handle_polling_network_error(error))
        self._background_tasks.add(self._polling_error_task)
        self._polling_error_task.add_done_callback(self._background_tasks.discard)

    async def _delete_webhook_best_effort(self) -> bool:
        """Clear any stale webhook, but never fail polling on a network error.

        Returns True when the webhook was cleared (or there was nothing to do)
        and False when a transient network error was swallowed so bootstrap can
        continue to polling; the reconnect ladder recovers from there.
        """
        if not self._bot:
            return False
        delete_webhook = getattr(self._bot, "delete_webhook", None)
        if not callable(delete_webhook):
            return True
        try:
            await delete_webhook(drop_pending_updates=False)
            return True
        except Exception as err:
            if self._looks_like_network_error(err):
                logger.warning(
                    "[%s] deleteWebhook failed with a recoverable network error; "
                    "continuing to polling so getUpdates/retry can recover: %s",
                    self.name, err,
                )
                self._send_path_degraded = True
                return False
            raise

    async def _start_polling_resilient(self, *, drop_pending_updates: bool, error_callback) -> bool:
        """Start PTB polling; on a transient bootstrap failure, recover in background.

        Returns True when polling started, False when a transient conflict or
        network error was scheduled for background recovery instead of raising
        (keeping the gateway process alive).
        """
        if not (self._app and self._app.updater):
            raise RuntimeError("Telegram application/updater not initialized")
        try:
            # Same watchdog bound as the reconnect ladders: a wedged httpx
            # connection pool can hang start_polling() forever at bootstrap
            # too (#59614). A propagating TimeoutError is a builtins
            # TimeoutError (OSError subclass), so the except below classifies
            # it via _looks_like_network_error and schedules background
            # recovery instead of blocking connect() indefinitely.
            await asyncio.wait_for(
                self._app.updater.start_polling(
                    allowed_updates=Update.ALL_TYPES,
                    drop_pending_updates=drop_pending_updates,
                    error_callback=error_callback,
                ),
                timeout=_UPDATER_START_TIMEOUT,
            )
            return True
        except Exception as err:
            if self._looks_like_polling_conflict(err):
                logger.warning(
                    "[%s] Telegram polling bootstrap conflict; gateway stays alive "
                    "while conflict retry runs: %s",
                    self.name, err,
                )
                loop = asyncio.get_running_loop()
                self._polling_error_task = loop.create_task(self._handle_polling_conflict(err))
                self._background_tasks.add(self._polling_error_task)
                self._polling_error_task.add_done_callback(self._background_tasks.discard)
                return False
            if self._looks_like_network_error(err):
                self._schedule_polling_recovery(err, reason="polling bootstrap")
                return False
            raise

    async def _handle_polling_network_error(self, error: Exception) -> None:
        """Reconnect polling after a transient network interruption.

        Triggered by NetworkError/TimedOut in the polling error callback, which
        happen when the host loses connectivity (Mac sleep, WiFi switch, VPN
        reconnect, etc.).  The gateway process stays alive but the long-poll
        connection silently dies; without this handler the bot never recovers.

        Strategy: exponential back-off (5s, 10s, 20s, 40s, 60s cap) up to
        MAX_NETWORK_RETRIES attempts, then mark the adapter retryable-fatal so
        the supervisor restarts the gateway process.
        """
        if self.has_fatal_error:
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
                "Restarting gateway." % MAX_NETWORK_RETRIES
            )
            logger.error("[%s] %s Last error: %s", self.name, message, _redact_telegram_error_text(error))
            self._set_fatal_error("telegram_network_error", message, retryable=True)
            await self._notify_fatal_error()
            return

        delay = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
        safe_error = _redact_telegram_error_text(error)
        logger.warning(
            "[%s] Telegram network error (attempt %d/%d), reconnecting in %ds. Error: %s",
            self.name, attempt, MAX_NETWORK_RETRIES, delay, safe_error,
        )
        await asyncio.sleep(delay)

        # Capture a stable local reference: self._app can be reassigned to None
        # by a concurrent disconnect() while we're suspended across the awaits
        # below, and re-reading self._app after that point would silently swap
        # in None mid-sequence instead of failing fast in one place.
        app = self._app

        try:
            if app and app.updater and app.updater.running:
                try:
                    # Guard stop() with a timeout: when the underlying TCP
                    # connection is in CLOSE-WAIT the PTB polling task is
                    # blocked on epoll on the dead socket and never wakes up,
                    # so an unguarded stop() hangs indefinitely.  The result
                    # is that _polling_error_task stays alive-but-blocked
                    # forever, every subsequent heartbeat probe sees it as
                    # "in-flight" and skips triggering a new reconnect, and
                    # the gateway silently drops messages for hours.
                    # Bounding stop() lets the reconnect ladder always advance.
                    # Refs: NousResearch/hermes-agent#58270
                    await asyncio.wait_for(app.updater.stop(), timeout=_UPDATER_STOP_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(
                        "[%s] updater.stop() timed out during network-error "
                        "reconnect (likely CLOSE-WAIT socket); forcing drain "
                        "and restart without clean stop",
                        self.name,
                    )
        except Exception:
            pass

        await self._drain_polling_connections()

        try:
            if not app:
                raise RuntimeError("Telegram application was torn down during reconnect")
            # Guard start_polling() with a timeout: when the connection pool is
            # in a degraded state (e.g., after _drain_polling_connections()), the
            # httpx client may hold a stale socket that neither connects nor times
            # out within PTB's internal flow. Bounding start_polling() prevents
            # the reconnect ladder from stalling indefinitely and allows the
            # heartbeat loop to trigger its own recovery path.
            # Refs: NousResearch/hermes-agent#59614
            try:
                await asyncio.wait_for(
                    app.updater.start_polling(
                        allowed_updates=Update.ALL_TYPES,
                        drop_pending_updates=False,
                        error_callback=self._polling_error_callback_ref,
                    ),
                    timeout=_UPDATER_START_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "start_polling() timed out — connection pool may be wedged"
                )
            logger.info(
                "[%s] Telegram polling resumed after network error (attempt %d)",
                self.name, attempt,
            )
            self._polling_network_error_count = 0
            # start_polling() succeeding IS the recovery signal: the long-poll
            # connection is live again, so clear the degraded flag immediately
            # rather than blocking all outbound sends for the full
            # HEARTBEAT_PROBE_DELAY window. The deferred probe below is a
            # defensive re-check — if it later detects a silent wedge (PTB
            # running=True but consumer task dead) it re-enters the ladder,
            # which re-sets _send_path_degraded. Without this clear here, a
            # clean reconnect leaves the flag stuck True until the 60s probe
            # (or forever, if the probe is never scheduled), blocking the send
            # path even though the bot has fully recovered. See #35205.
            self._send_path_degraded = False
            # start_polling() returning is necessary but not sufficient:
            # PTB's Updater can be left in a state where `running` is True
            # but the underlying long-poll task is wedged on a stale httpx
            # connection and never makes progress. No error_callback fires
            # in that state, so the reconnect ladder won't advance on its
            # own. Schedule a deferred probe to detect the wedge and
            # re-enter the ladder if needed.
            if not self.has_fatal_error:
                probe = asyncio.ensure_future(self._verify_polling_after_reconnect())
                self._background_tasks.add(probe)
                probe.add_done_callback(self._background_tasks.discard)
        except Exception as retry_err:
            safe_retry_error = _redact_telegram_error_text(retry_err)
            logger.warning("[%s] Telegram polling reconnect failed: %s", self.name, safe_retry_error)
            # start_polling failed — polling is dead and no further error
            # callbacks will fire, so schedule the next retry ourselves.
            if not self.has_fatal_error:
                task = asyncio.ensure_future(
                    self._handle_polling_network_error(retry_err)
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                # This chained retry IS the in-flight recovery attempt — it
                # must replace the reentrancy guard, otherwise the heartbeat
                # loop, the pending-updates probe, and the PTB error callback
                # all see _polling_error_task as "done" and can each start a
                # second, concurrent recovery for the same outage.
                self._polling_error_task = task

    async def _polling_heartbeat_loop(self) -> None:
        """Detect dead Telegram TCP sockets (CLOSE-WAIT) by periodic probing.

        PTB's long-poll task blocks on epoll waiting for Telegram to push an
        update.  When the underlying TCP connection enters CLOSE-WAIT (the remote
        sent a FIN but the httpx pool has not yet noticed), epoll still reports
        the socket as readable and no exception is raised — so PTB's
        ``error_callback`` never fires and the gateway silently stops receiving
        messages.

        This loop probes ``get_me()`` every ``HEARTBEAT_INTERVAL`` seconds on the
        *general* request path (not the getUpdates pool), so a healthy long-poll
        waiting for the 30-second Telegram window is never interrupted.  On any
        connect-level failure the loop hands off to
        ``_handle_polling_network_error`` — the same path triggered by PTB's own
        ``error_callback`` — which drains the dead pool and restarts polling.

        Unlike ``_verify_polling_after_reconnect`` (a one-shot probe scheduled
        only after an explicit reconnect), this loop runs for the full lifetime
        of the polling connection, so it catches a socket that wedges during
        steady-state operation without any prior error event.
        """
        HEARTBEAT_INTERVAL = 90   # seconds between probes
        PROBE_TIMEOUT = 15        # seconds before declaring the path dead

        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                if self.has_fatal_error:
                    return
                bot = self._app.bot if self._app else None
                if bot is None:
                    continue
                # A real PTB Bot always exposes get_me(); if it's absent the
                # app isn't a live polling client (e.g. torn down or a test
                # double), so there is nothing to probe — exit rather than spin.
                if not callable(getattr(bot, "get_me", None)):
                    return
                await asyncio.wait_for(bot.get_me(), PROBE_TIMEOUT)
                # get_me() succeeded — the general/send request path is healthy.
                # That does NOT prove the getUpdates consumer is alive: PTB can
                # report updater.running=True while the long-poll task is wedged,
                # so DMs queue in the Bot API and never reach handlers (#42909).
                # get_me() is blind to this; get_webhook_info() exposes it via
                # pending_update_count. Escalate only after two consecutive
                # probes see a non-zero queue while we believe we're polling, so
                # a single in-flight update (consumed before the next probe)
                # never trips recovery.
                await self._probe_pending_updates(bot, PROBE_TIMEOUT)
            except asyncio.CancelledError:
                return
            except (asyncio.TimeoutError, OSError) as probe_err:
                self._schedule_polling_recovery(probe_err, reason="heartbeat probe")
            except Exception as probe_err:
                if self._looks_like_network_error(probe_err):
                    self._schedule_polling_recovery(probe_err, reason="heartbeat probe")
                    continue
                # Non-connectivity errors (e.g. TelegramError 401) are not
                # CLOSE-WAIT symptoms — let PTB's own handlers surface them.
                pass

    async def _probe_pending_updates(self, bot, probe_timeout: float) -> None:
        """Detect a wedged getUpdates consumer via pending_update_count.

        PTB can report ``updater.running == True`` while its long-poll task is
        silently stuck (e.g. a socket that epoll keeps reporting readable on
        WSL2). ``get_me()`` stays healthy because it uses the general request
        path, so the CLOSE-WAIT heartbeat never fires — yet DMs queue in the
        Bot API and never reach handlers (#42909).

        ``get_webhook_info().pending_update_count`` is the one signal that
        exposes this: a growing/stuck queue while we believe we're polling means
        the consumer is dead. We only escalate after two consecutive stuck
        probes so a single update that's simply in-flight between probes does
        not trip a needless recovery. Recovery reuses
        ``_handle_polling_network_error`` — the same ladder PTB's own
        ``error_callback`` feeds — so no new restart machinery is introduced.

        This also covers the harsher case where the updater has stopped
        entirely (``running=False``) with no reconnect in flight: the long-poll
        task is gone rather than wedged, so even ``get_webhook_info`` can't
        report a queue against a live consumer. We detect the stopped updater
        directly and feed the same ladder (#55769).
        """
        # Only meaningful in polling mode; in webhook mode Telegram pushes
        # updates and holds no server-side queue.
        if self._webhook_mode:
            return
        # A reconnect already in flight owns recovery — don't double-trigger,
        # and don't misread its brief stop()->start_polling() window (where
        # updater.running is transiently False) as a dead updater below.
        if self._polling_error_task and not self._polling_error_task.done():
            self._polling_not_running_count = 0
            return
        updater = getattr(self._app, "updater", None) if self._app else None
        if updater is None:
            self._polling_pending_stuck_count = 0
            return
        if not getattr(updater, "running", False):
            # We are in polling mode with no reconnect in flight, yet PTB's
            # updater has stopped entirely. This is distinct from the
            # wedged-but-running consumer handled below: the long-poll task is
            # gone, get_me()/get_webhook_info() on the general request path
            # still succeed, so no error_callback or connectivity probe ever
            # fires and the gateway silently stops receiving messages while the
            # process stays alive (#55769). Escalate through the same reconnect
            # ladder as a wedged consumer, debounced over two consecutive probes
            # so a just-starting updater never trips it.
            self._polling_pending_stuck_count = 0
            self._polling_not_running_count += 1
            logger.warning(
                "[%s] Telegram polling heartbeat: updater stopped while in "
                "polling mode (stuck probe %d/2)",
                self.name, self._polling_not_running_count,
            )
            if self._polling_not_running_count >= 2:
                self._polling_not_running_count = 0
                logger.warning(
                    "[%s] Telegram updater is not running (long-poll task "
                    "gone); triggering polling restart",
                    self.name,
                )
                loop = asyncio.get_running_loop()
                self._polling_error_task = loop.create_task(
                    self._handle_polling_network_error(
                        RuntimeError("Telegram updater stopped while in polling mode")
                    )
                )
            return
        self._polling_not_running_count = 0
        get_webhook_info = getattr(bot, "get_webhook_info", None)
        if not callable(get_webhook_info):
            return
        try:
            info = await asyncio.wait_for(get_webhook_info(), probe_timeout)  # type: ignore[arg-type]
        except (asyncio.TimeoutError, OSError):
            # A failed probe is a connectivity symptom the get_me() path or the
            # outer handler will catch; don't treat it as a stuck-queue signal.
            return
        pending = int(getattr(info, "pending_update_count", 0) or 0)
        if pending <= 0:
            self._polling_pending_stuck_count = 0
            return
        self._polling_pending_stuck_count += 1
        logger.warning(
            "[%s] Telegram polling heartbeat: %d update(s) queued but not "
            "consumed (stuck probe %d/2)",
            self.name, pending, self._polling_pending_stuck_count,
        )
        if self._polling_pending_stuck_count >= 2:
            self._polling_pending_stuck_count = 0
            logger.warning(
                "[%s] getUpdates consumer appears wedged (queue not draining); "
                "triggering polling restart",
                self.name,
            )
            loop = asyncio.get_running_loop()
            self._polling_error_task = loop.create_task(
                self._handle_polling_network_error(
                    RuntimeError("getUpdates consumer wedged: pending updates not draining")
                )
            )

    async def _verify_polling_after_reconnect(self) -> None:
        """Heartbeat probe scheduled after a successful reconnect.

        PTB's Updater can survive a botched stop()+start_polling() cycle
        with `running=True` but a wedged consumer task. No error callback
        fires, so the reconnect ladder doesn't advance on its own. This
        probe detects the wedge by:

        1. Sleeping HEARTBEAT_PROBE_DELAY so a healthy long-poll has time
           to complete at least one cycle.
        2. Verifying `Updater.running` is still True.
        3. Probing the bot endpoint with a tight asyncio timeout. A
           wedged httpx pool fails this probe; a healthy one returns
           well under the timeout.

        On any failure, re-enter the reconnect ladder so the existing
        MAX_NETWORK_RETRIES path can ultimately escalate to fatal-error.
        """
        HEARTBEAT_PROBE_DELAY = 60
        PROBE_TIMEOUT = 10

        await asyncio.sleep(HEARTBEAT_PROBE_DELAY)

        if self.has_fatal_error:
            return
        if not (self._app and self._app.updater and self._app.updater.running):
            logger.warning(
                "[%s] Updater not running %ds after reconnect — treating as wedged",
                self.name, HEARTBEAT_PROBE_DELAY,
            )
            await self._handle_polling_network_error(
                RuntimeError("Updater not running after reconnect heartbeat")
            )
            return

        try:
            await asyncio.wait_for(self._app.bot.get_me(), PROBE_TIMEOUT)
            self._send_path_degraded = False
        except Exception as probe_err:
            logger.warning(
                "[%s] Polling heartbeat probe failed %ds after reconnect: %s",
                self.name, HEARTBEAT_PROBE_DELAY, probe_err,
            )
            await self._handle_polling_network_error(probe_err)

    def _disarm_ptb_retry_loop(self) -> None:
        """Synchronously stop PTB's internal polling retry loop.

        PTB wraps ``getUpdates`` in ``network_retry_loop`` with
        ``max_retries=-1`` (retry forever).  When a ``TelegramError`` (including
        a 409 ``Conflict``) fires, that loop calls our ``error_callback``
        *synchronously*, then sleeps and re-checks ``while is_running()`` before
        polling again.  Our ``error_callback`` only schedules an async recovery
        task (``loop.create_task(...)``) and returns immediately, so PTB's loop
        keeps polling while our handler concurrently runs
        ``stop -> sleep -> start_polling``.  The two polling sessions overlap and
        Telegram returns a fresh 409 — a self-inflicted conflict loop on a
        ~31s cadence.

        The loop is wired with ``is_running=lambda: updater.running`` and a
        private ``stop_event`` (``do_action`` races that event and returns the
        moment it is set).  Setting that event *synchronously inside the
        callback* — before it returns — makes PTB's loop exit on its own next
        tick instead of racing our recovery.  Our async handler then performs
        the real ``await updater.stop()`` (idempotent) followed by
        drain + ``start_polling()``, which builds a fresh ``stop_event`` so the
        restart is not poisoned.

        Best-effort and defensive: PTB names the attribute differently across
        versions (``_Updater__polling_task_stop_event`` via name-mangling), so
        we probe for both spellings.  If neither is found we do nothing and
        fall back to the prior behaviour (async ``updater.stop()`` racing PTB) —
        i.e. we never make things worse than before.

        We deliberately do NOT fall back to flipping ``updater._running``:
        ``stop()`` raises ``RuntimeError`` when ``running`` is already False and
        our recovery handler guards its ``stop()`` call on ``running``, so
        clearing the flag here would skip the real teardown and leave PTB's
        stop_event uncleared — poisoning the subsequent ``start_polling()``.
        The stop_event lever leaves ``_running`` True, so the handler's
        ``await updater.stop()`` still runs, drains the polling task, and clears
        the event for a clean restart.
        """
        updater = getattr(self._app, "updater", None) if self._app else None
        if updater is None:
            return
        # Preferred (and only) lever: PTB's polling stop_event. Name-mangled on
        # Updater, so probe both the mangled and unmangled spellings.
        for attr in (
            "_Updater__polling_task_stop_event",
            "_polling_task_stop_event",
        ):
            stop_event = getattr(updater, attr, None)
            if isinstance(stop_event, asyncio.Event):
                if not stop_event.is_set():
                    stop_event.set()
                    logger.debug(
                        "[%s] Disarmed PTB polling retry loop via %s",
                        self.name, attr,
                    )
                return
        logger.debug(
            "[%s] Could not disarm PTB polling retry loop "
            "(stop_event not found on this PTB version); "
            "falling back to async stop()",
            self.name,
        )

    async def _handle_polling_conflict(self, error: Exception) -> None:
        if self.has_fatal_error and self.fatal_error_code == "telegram_polling_conflict":
            return
        # Transient 409 Conflict errors arise when the previous gateway process
        # has been killed (e.g. during `hermes update` or `--replace` handoffs)
        # but its long-poll connection hasn't yet expired on Telegram's servers.
        # Telegram holds open getUpdates sessions for up to ~30s after the
        # client disconnects, so a new gateway starting immediately will receive
        # a 409 until that server-side session expires.
        #
        # Strategy: stop the local updater, wait long enough for Telegram's
        # server-side session to expire (RETRY_DELAY grows with each attempt),
        # drain the connection pool, then restart polling.  We attempt this
        # MAX_CONFLICT_RETRIES times before declaring a fatal error.
        #
        # Crucially, a failed retry must NOT leave polling in an ambiguous
        # state.  If start_polling() raises, the updater is neither running
        # nor fatal — messages are silently dropped.  We schedule another
        # retry attempt instead of returning silently, and only escalate to
        # fatal after all retries are exhausted.
        self._polling_conflict_count += 1

        MAX_CONFLICT_RETRIES = 5
        # Delay grows with each attempt: 15s, 25s, 35s, 45s, 55s.
        # Telegram server-side getUpdates sessions typically expire within
        # 30s; the increasing back-off ensures we clear that window without
        # hammering the API on fast-restart loops.
        RETRY_DELAY = 10 + (self._polling_conflict_count * 10)  # seconds

        if self._polling_conflict_count <= MAX_CONFLICT_RETRIES:
            logger.warning(
                "[%s] Telegram polling conflict (%d/%d) — previous session still "
                "held open on Telegram's servers. Waiting %ds for it to expire. "
                "Error: %s",
                self.name, self._polling_conflict_count, MAX_CONFLICT_RETRIES,
                RETRY_DELAY, error,
            )
            # Stop the local updater cleanly before sleeping.  If it's already
            # stopped (e.g. PTB raised before updater.running was set) this is
            # a no-op.  Bounded with a timeout for the same reason as the
            # network-error path: a CLOSE-WAIT socket can wedge stop() on epoll
            # forever, which would stall the conflict-retry ladder.
            try:
                if self._app and self._app.updater and self._app.updater.running:
                    try:
                        await asyncio.wait_for(self._app.updater.stop(), timeout=_UPDATER_STOP_TIMEOUT)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[%s] updater.stop() timed out during conflict "
                            "retry (likely CLOSE-WAIT socket); continuing",
                            self.name,
                        )
            except Exception:
                pass

            await asyncio.sleep(RETRY_DELAY)
            await self._drain_polling_connections()

            # Capture a stable local reference: self._app can be reassigned to
            # None by a concurrent disconnect() while we're suspended across
            # the awaits above (same race #55992 fixed on the network path).
            # Re-reading self._app after that point would raise
            # AttributeError deep inside start_polling instead of failing fast
            # here, where the except below reschedules or escalates to fatal.
            app = self._app
            try:
                if not app:
                    raise RuntimeError("Telegram application was torn down during conflict reconnect")
                # Same watchdog bound as the network-error ladder: an
                # exhausted pool hangs start_polling() on the conflict path
                # identically (#59614). Timeout converts to RuntimeError so
                # the except below logs a readable message and schedules the
                # next conflict attempt instead of wedging attempt N forever.
                try:
                    await asyncio.wait_for(
                        app.updater.start_polling(
                            allowed_updates=Update.ALL_TYPES,
                            drop_pending_updates=False,
                            error_callback=self._polling_error_callback_ref,
                        ),
                        timeout=_UPDATER_START_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError(
                        "start_polling() timed out — connection pool may be wedged"
                    )
                logger.info(
                    "[%s] Telegram polling resumed after conflict retry %d/%d",
                    self.name, self._polling_conflict_count, MAX_CONFLICT_RETRIES,
                )
                self._polling_conflict_count = 0  # reset counter on success
                return
            except Exception as retry_err:
                logger.warning(
                    "[%s] Telegram polling retry %d/%d failed: %s. "
                    "Scheduling next attempt.",
                    self.name, self._polling_conflict_count, MAX_CONFLICT_RETRIES,
                    retry_err,
                )
                # Schedule the next retry rather than returning silently.
                # Returning here without either restarting polling or setting
                # a fatal error leaves the adapter in a limbo state: the
                # gateway process is alive and reports "connected" but
                # no messages are received or sent.
                if self._polling_conflict_count < MAX_CONFLICT_RETRIES:
                    # We are inside a running coroutine, so the running loop is
                    # guaranteed to exist. asyncio.get_event_loop() is deprecated
                    # and raises "RuntimeError: There is no current event loop in
                    # thread 'MainThread'" on Python 3.10+ when invoked from a
                    # context without an attached loop (which can happen when PTB
                    # dispatches this error callback). Use get_running_loop().
                    loop = asyncio.get_running_loop()
                    self._polling_error_task = loop.create_task(
                        self._handle_polling_conflict(retry_err)
                    )
                    return
                # Fall through to fatal on the last retry.

        # Exhausted all retries — declare a fatal error so the gateway
        # runner can surface this clearly and the user knows to act.
        message = (
            "Telegram polling could not recover after %d retries (%ds total wait). "
            "The previous gateway session is still held open on Telegram's servers, "
            "or another process is using the same bot token. "
            "To recover: ensure no other Hermes or OpenClaw instance is running "
            "with this token, then restart the gateway with 'hermes gateway restart'."
            % (MAX_CONFLICT_RETRIES, sum(10 + i * 10 for i in range(1, MAX_CONFLICT_RETRIES + 1)))
        )
        logger.error(
            "[%s] %s Original error: %s",
            self.name, message, error,
        )
        # Snapshot whether we are the call that actually transitions to fatal.
        # A concurrent retry task scheduled by an earlier conflict may already
        # be suspended past the entry guard; once _set_fatal_error flips the
        # flag, adding an await below (the bounded stop()) yields the loop and
        # lets that task reach this branch too — double-notifying the fatal
        # handler.  Only the first transition notifies.
        _already_fatal = (
            self.has_fatal_error
            and self.fatal_error_code == "telegram_polling_conflict"
        )
        self._set_fatal_error("telegram_polling_conflict", message, retryable=False)
        try:
            if self._app and self._app.updater:
                await asyncio.wait_for(self._app.updater.stop(), timeout=_UPDATER_STOP_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] updater.stop() timed out after exhausting conflict "
                "retries (likely CLOSE-WAIT socket); proceeding to fatal notify",
                self.name,
            )
        except Exception as stop_error:
            logger.warning(
                "[%s] Failed stopping Telegram updater after exhausting conflict retries: %s",
                self.name, stop_error, exc_info=True,
            )
        if not _already_fatal:
            await self._notify_fatal_error()

    async def _create_dm_topic(
        self,
        chat_id: int,
        name: str,
        icon_color: Optional[int] = None,
        icon_custom_emoji_id: Optional[str] = None,
    ) -> Optional[int]:
        """Create a forum topic in a private (DM) chat.

        Uses Bot API 9.4's createForumTopic which now works for 1-on-1 chats.
        Returns the message_thread_id on success, None on failure.
        """
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
            logger.info(
                "[%s] Created DM topic '%s' in chat %s -> thread_id=%s",
                self.name, name, chat_id, thread_id,
            )
            return thread_id
        except Exception as e:
            error_text = str(e).lower()
            # If topic already exists, try to find it via getForumTopicIconStickers
            # or we just log and skip — Telegram doesn't provide a "list topics" API
            if "topic_name_duplicate" in error_text or "already" in error_text:
                logger.info(
                    "[%s] DM topic '%s' already exists in chat %s (will be mapped from incoming messages)",
                    self.name, name, chat_id,
                )
            elif "not a forum" in error_text or "forums_disabled" in error_text:
                logger.warning(
                    "[%s] Cannot create DM topic '%s' in chat %s: Topics mode is not enabled. "
                    "The user must open the DM with this bot in Telegram, tap the bot name "
                    "at the top, and enable 'Topics' in chat settings before topics can be created.",
                    self.name, name, chat_id,
                )
            else:
                logger.warning(
                    "[%s] Failed to create DM topic '%s' in chat %s: %s",
                    self.name, name, chat_id, e,
                )
            return None

    async def create_handoff_thread(
        self,
        parent_chat_id: str,
        name: str,
    ) -> Optional[str]:
        """Create a forum topic for a session handoff.

        Works for DM topics (Bot API 9.4+, requires user to enable Topics
        in their chat with the bot) and forum supergroups. Returns the
        ``message_thread_id`` as a string, or ``None`` on failure.
        """
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
            for candidate in entry.get("topics", []):
                if candidate.get("name") == name:
                    topic_conf = candidate
                    break
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
            chat_id_int,
            name=name,
            icon_color=topic_conf.get("icon_color"),
            icon_custom_emoji_id=topic_conf.get("icon_custom_emoji_id"),
        )
        if not thread_id:
            return None

        topic_conf["thread_id"] = thread_id
        self._dm_topics[cache_key] = int(thread_id)
        self._persist_dm_topic_thread_id(chat_id_int, name, int(thread_id), replace_existing=force_create)
        return str(thread_id)

    async def rename_dm_topic(
        self,
        chat_id: int,
        thread_id: int,
        name: str,
    ) -> None:
        """Rename a forum topic in a private (DM) chat."""
        if not self._bot:
            return
        try:
            chat_id_arg = int(chat_id)
        except (TypeError, ValueError):
            chat_id_arg = chat_id
        await self._bot.edit_forum_topic(
            chat_id=chat_id_arg,
            message_thread_id=int(thread_id),
            name=name,
        )
        logger.info(
            "[%s] Renamed DM topic in chat %s thread_id=%s -> '%s'",
            self.name, chat_id, thread_id, name,
        )

    def _persist_dm_topic_thread_id(
        self,
        chat_id: int,
        topic_name: str,
        thread_id: int,
        replace_existing: bool = False,
    ) -> None:
        """Save a newly created thread_id back into config.yaml so it persists across restarts."""
        try:
            from hermes_constants import get_hermes_home
            config_path = get_hermes_home() / "config.yaml"
            if not config_path.exists():
                logger.warning("[%s] Config file not found at %s, cannot persist thread_id", self.name, config_path)
                return

            import yaml as _yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = _yaml.safe_load(f) or {}

            # Navigate to platforms.telegram.extra.dm_topics, creating the path
            # when a named delivery target asks us to create a topic that was
            # not predeclared in config.yaml.
            platforms = config.setdefault("platforms", {})
            telegram_config = platforms.setdefault("telegram", {})
            extra = telegram_config.setdefault("extra", {})
            dm_topics = extra.setdefault("dm_topics", [])

            changed = False
            matching_chat_entry = None
            for chat_entry in dm_topics:
                try:
                    chat_matches = int(chat_entry.get("chat_id", 0)) == int(chat_id)
                except (TypeError, ValueError):
                    chat_matches = False
                if not chat_matches:
                    continue
                matching_chat_entry = chat_entry
                for t in chat_entry.setdefault("topics", []):
                    if t.get("name") == topic_name:
                        if replace_existing or not t.get("thread_id"):
                            if t.get("thread_id") != thread_id:
                                t["thread_id"] = thread_id
                                changed = True
                        break
                else:
                    chat_entry.setdefault("topics", []).append(
                        {"name": topic_name, "thread_id": thread_id}
                    )
                    changed = True
                break

            if matching_chat_entry is None:
                dm_topics.append({
                    "chat_id": chat_id,
                    "topics": [{"name": topic_name, "thread_id": thread_id}],
                })
                changed = True

            if changed:
                from hermes_cli.config import atomic_config_write

                atomic_config_write(
                    config_path,
                    config,
                    default_flow_style=False,
                    sort_keys=False,
                )
                logger.info(
                    "[%s] Persisted thread_id=%s for topic '%s' in config.yaml",
                    self.name, thread_id, topic_name,
                )
        except Exception as e:
            logger.warning("[%s] Failed to persist thread_id to config: %s", self.name, e, exc_info=True)

    async def _setup_dm_topics(self) -> None:
        """Load or create configured DM topics for specified chats.

        Reads config.extra['dm_topics'] — a list of dicts:
        [
            {
                "chat_id": 123456789,
                "topics": [
                    {"name": "General", "icon_color": 7322096, "thread_id": 100},
                    {"name": "Accessibility Auditor", "icon_color": 9367192, "skill": "accessibility-auditor"}
                ]
            }
        ]

        If a topic already has a thread_id in the config (persisted from a previous
        creation), it is loaded into the cache without calling createForumTopic.
        Only topics without a thread_id are created via the API, and their thread_id
        is then saved back to config.yaml for future restarts.
        """
        if not self._dm_topics_config:
            return

        for chat_entry in self._dm_topics_config:
            chat_id = chat_entry.get("chat_id")
            topics = chat_entry.get("topics", [])
            if not chat_id or not topics:
                continue

            logger.info(
                "[%s] Setting up %d DM topic(s) for chat %s",
                self.name, len(topics), chat_id,
            )

            for topic_conf in topics:
                topic_name = topic_conf.get("name")
                if not topic_name:
                    continue

                cache_key = f"{chat_id}:{topic_name}"

                # If thread_id is already persisted in config, just load into cache
                existing_thread_id = topic_conf.get("thread_id")
                if existing_thread_id:
                    self._dm_topics[cache_key] = int(existing_thread_id)
                    logger.info(
                        "[%s] DM topic loaded from config: %s -> thread_id=%s",
                        self.name, cache_key, existing_thread_id,
                    )
                    continue

                # No persisted thread_id — create the topic via API
                icon_color = topic_conf.get("icon_color")
                icon_emoji = topic_conf.get("icon_custom_emoji_id")

                thread_id = await self._create_dm_topic(
                    chat_id=normalize_telegram_chat_id(chat_id),
                    name=topic_name,
                    icon_color=icon_color,
                    icon_custom_emoji_id=icon_emoji,
                )

                if thread_id:
                    self._dm_topics[cache_key] = thread_id
                    logger.info(
                        "[%s] DM topic cached: %s -> thread_id=%s",
                        self.name, cache_key, thread_id,
                    )
                    # Persist thread_id to config so we don't recreate on next restart
                    self._persist_dm_topic_thread_id(int(chat_id), topic_name, thread_id)

                    # Send a seed message so the topic is visible in Telegram's client.
                    # Empty topics are hidden by the client UI until they contain a message.
                    try:
                        await self._bot.send_message(
                            chat_id=normalize_telegram_chat_id(chat_id),
                            message_thread_id=thread_id,
                            text=f"\U0001f4cc {topic_name}",
                        )
                    except Exception as seed_err:
                        logger.debug(
                            "[%s] Could not send seed message to topic '%s': %s",
                            self.name, topic_name, seed_err,
                        )

    def _start_post_connect_housekeeping(self) -> None:
        """Kick off deferred post-connect housekeeping in the background.

        Idempotent: if a previous housekeeping task is still running (e.g. a
        rapid reconnect), it is left in place rather than double-scheduled.
        """
        task = self._post_connect_task
        if task and not task.done():
            return
        self._post_connect_task = asyncio.ensure_future(
            self._run_post_connect_housekeeping()
        )

    async def _run_post_connect_housekeeping(self) -> None:
        """Register the command menu, surface the status indicator, and set up
        DM topics — all off the connect path so a slow Bot API call cannot blow
        the gateway connect timeout (#46298). Every step is non-fatal."""
        try:
            # Register bot commands so Telegram shows a hint menu when users type /
            # List is derived from the central COMMAND_REGISTRY — adding a new
            # gateway command there automatically adds it to the Telegram menu.
            try:
                from telegram import (
                    BotCommand,
                    BotCommandScopeAllPrivateChats,
                    BotCommandScopeAllGroupChats,
                    BotCommandScopeDefault,
                )
                from hermes_cli.commands import telegram_menu_commands, telegram_menu_max_commands
                if not self._bot:
                    return
                # Telegram allows up to 100 commands but has an undocumented
                # payload size limit (~4KB total).  Hermes defaults to 60 to
                # keep built-ins plus common skill commands visible while
                # staying under the threshold; users can tune the cap via
                # platforms.telegram.extra.command_menu.
                max_commands = telegram_menu_max_commands()
                menu_commands, hidden_count = telegram_menu_commands(max_commands=max_commands)
                bot_commands = [BotCommand(name, desc) for name, desc in menu_commands]
                # Register for all scopes independently — Telegram picks the
                # narrowest matching scope per chat type (forum topics fall
                # through to AllGroupChats or Default).
                for scope_cls in (BotCommandScopeDefault, BotCommandScopeAllPrivateChats, BotCommandScopeAllGroupChats):
                    scope_name = getattr(scope_cls, "__name__", str(scope_cls))
                    try:
                        await self._bot.set_my_commands(bot_commands, scope=scope_cls())
                        logger.info("[%s] set_my_commands OK for scope %s (%d cmds)", self.name, scope_name, len(bot_commands))
                    except Exception as scope_err:
                        logger.warning("[%s] set_my_commands FAILED for scope %s: %s", self.name, scope_name, scope_err)
                # Forum topics don't inherit AllGroupChats — Telegram resolves
                # commands via BotCommandScopeChat(chat_id) for forum groups.
                # Lazy registration happens in _ensure_forum_commands on first
                # message from a forum topic (see _handle_text_message).
                if hidden_count:
                    logger.info(
                        "[%s] Telegram menu: %d commands registered, %d hidden (over %d limit). Use /commands for full list.",
                        self.name, len(menu_commands), hidden_count, max_commands,
                    )
            except Exception as e:
                logger.warning(
                    "[%s] Could not register Telegram command menu: %s",
                    self.name,
                    e,
                    exc_info=True,
                )

            # Surface the gateway as "Online" in the bot's short description
            # (opt-in via extra.status_indicator). Non-fatal.
            try:
                await self._set_status_indicator(online=True)
            except Exception:
                pass

            # Set up DM topics (Bot API 9.4 — Private Chat Topics)
            # Runs after connection is established so the bot can call createForumTopic.
            # Failures here are non-fatal — the bot works fine without topics.
            try:
                await self._setup_dm_topics()
            except Exception as topics_err:
                logger.warning(
                    "[%s] DM topics setup failed (non-fatal): %s",
                    self.name, topics_err, exc_info=True,
                )
        except asyncio.CancelledError:
            raise
        finally:
            if self._post_connect_task is asyncio.current_task():
                self._post_connect_task = None

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to Telegram via polling or webhook.

        By default, uses long polling (outbound connection to Telegram).
        If ``TELEGRAM_WEBHOOK_URL`` is set, starts an HTTP webhook server
        instead.  Webhook mode is useful for cloud deployments (Fly.io,
        Railway) where inbound HTTP can wake a suspended machine.

        ``is_reconnect`` distinguishes a cold first boot (False — drop any
        stale Bot API queue) from a watcher reconnect after a prolonged
        outage (True — preserve the updates Telegram queued while the bot
        was offline, otherwise every message sent during the outage is
        silently lost). The in-process network-error ladder and the
        409-conflict handler already pass ``drop_pending_updates=False``
        for the same reason; bootstrap follows suit on the reconnect path.

        Env vars for webhook mode::

            TELEGRAM_WEBHOOK_URL    Public HTTPS URL (e.g. https://app.fly.dev/telegram)
            TELEGRAM_WEBHOOK_PORT   Local listen port (default 8443)
            TELEGRAM_WEBHOOK_SECRET Secret token for update verification
        """
        if not TELEGRAM_AVAILABLE:
            logger.error(
                "[%s] python-telegram-bot not installed. Run: pip install python-telegram-bot",
                self.name,
            )
            return False
        
        if not self.config.token:
            logger.error("[%s] No bot token configured", self.name)
            return False
        
        try:
            if not self._acquire_platform_lock('telegram-bot-token', self.config.token, 'Telegram bot token'):
                return False

            # Build the application
            builder = Application.builder().token(self.config.token)
            custom_base_url = self.config.extra.get("base_url")
            if custom_base_url:
                builder = builder.base_url(custom_base_url)
                builder = builder.base_file_url(
                    self.config.extra.get("base_file_url", custom_base_url)
                )
                logger.info(
                    "[%s] Using custom Telegram base_url: %s",
                    self.name, custom_base_url,
                )
            # In local-mode telegram-bot-api, file_path is an absolute path on the
            # server's filesystem rather than a relative HTTP path. PTB needs
            # local_mode=True so download_*() reads from disk instead of issuing
            # an HTTP GET that would 404. Requires that the same path is
            # readable by the Hermes process (shared mount, same machine, etc.).
            if self.config.extra.get("local_mode"):
                builder = builder.local_mode(True)
                logger.info("[%s] Using Telegram local_mode (read files from disk)", self.name)

            # PTB defaults (pool_timeout=1s) are too aggressive on flaky networks and
            # can trigger "Pool timeout: All connections in the connection pool are occupied"
            # during reconnect/bootstrap. Use safer defaults and allow env overrides.
            def _env_int(name: str, default: int) -> int:
                try:
                    return int(os.getenv(name, str(default)))
                except (TypeError, ValueError):
                    return default

            def _env_float(name: str, default: float) -> float:
                try:
                    return float(os.getenv(name, str(default)))
                except (TypeError, ValueError):
                    return default

            request_kwargs = {
                "connection_pool_size": _env_int("HERMES_TELEGRAM_HTTP_POOL_SIZE", 512),
                "pool_timeout": _env_float("HERMES_TELEGRAM_HTTP_POOL_TIMEOUT", 8.0),
                "connect_timeout": _env_float("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", 10.0),
                "read_timeout": _env_float("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", 20.0),
                "write_timeout": _env_float("HERMES_TELEGRAM_HTTP_WRITE_TIMEOUT", 20.0),
            }

            # CLOSE_WAIT fd leak (#31599, same class as #18451): PTB's
            # HTTPXRequest builds the underlying httpx.AsyncClient with
            # `limits = httpx.Limits(max_connections=connection_pool_size)`
            # and *no* keepalive tuning, so httpx's default
            # keepalive_expiry=5.0 applies. Behind an HTTP proxy (Cloudflare
            # Warp etc.) a peer-initiated FIN can sit in CLOSE_WAIT longer
            # than that, leaking fds in the general request pool (_request[1])
            # which _drain_polling_connections never resets. Wire the shared
            # platform_httpx_limits() helper into the httpx client so idle
            # keepalive sockets drain aggressively, while preserving PTB's
            # max_connections (= connection_pool_size). httpx_kwargs is spread
            # last into PTB's client kwargs, so `limits` here wins.
            from gateway.platforms._http_client_limits import platform_httpx_limits

            _base_limits = platform_httpx_limits()
            if _base_limits is not None:
                import httpx as _httpx

                _pool_limits = _httpx.Limits(
                    max_connections=request_kwargs["connection_pool_size"],
                    max_keepalive_connections=_base_limits.max_keepalive_connections,
                    keepalive_expiry=_base_limits.keepalive_expiry,
                )
            else:  # pragma: no cover — httpx always present alongside PTB
                _pool_limits = None

            def _with_limits(httpx_kwargs: Optional[dict] = None) -> dict:
                """Merge tuned keepalive limits into httpx client kwargs.

                Used by the proxy and direct-DNS branches, where httpx honours
                the client-level ``limits`` kwarg. A caller-supplied ``limits``
                is left untouched; otherwise the CLOSE_WAIT-safe limits are
                injected. The fallback-IP branch does NOT use this helper — see
                the ``_transport_kwargs`` note below for why.
                """
                kwargs = dict(httpx_kwargs or {})
                if _pool_limits is not None and "limits" not in kwargs:
                    kwargs["limits"] = _pool_limits
                return kwargs

            disable_fallback = (os.getenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "").strip().lower() in {"1", "true", "yes", "on"})
            fallback_ips = self._fallback_ips()
            if not fallback_ips:
                logger.warning("[%s] Discovering Telegram API fallback IPs via DNS-over-HTTPS…", self.name)
                fallback_ips = await discover_fallback_ips()
                logger.info(
                    "[%s] Auto-discovered Telegram fallback IPs: %s",
                    self.name,
                    ", ".join(fallback_ips),
                )

            proxy_targets = ["api.telegram.org", *fallback_ips]
            proxy_url = resolve_proxy_url("TELEGRAM_PROXY", target_hosts=proxy_targets)
            if fallback_ips and not proxy_url and not disable_fallback:
                logger.info(
                    "[%s] Telegram fallback IPs active: %s",
                    self.name,
                    ", ".join(fallback_ips),
                )
                # Keep request/update pools separate to reduce contention during
                # polling reconnect + bot API bootstrap/delete_webhook calls.
                # httpx ignores the client-level `limits` kwarg when a custom
                # `transport` is supplied (#58790).  Unlike the proxy/direct
                # branches (which inject limits at the client level via
                # `_with_limits`), this branch MUST pass the tuned limits
                # directly into TelegramFallbackTransport so its inner
                # AsyncHTTPTransport instances honour keepalive_expiry — do not
                # route this through `_with_limits`, httpx would discard it.
                _transport_kwargs: dict = {}
                if _pool_limits is not None:
                    _transport_kwargs["limits"] = _pool_limits
                request = HTTPXRequest(
                    **request_kwargs,
                    httpx_kwargs={
                        "transport": TelegramFallbackTransport(
                            fallback_ips, **_transport_kwargs
                        )
                    },
                )
                get_updates_request = HTTPXRequest(
                    **request_kwargs,
                    httpx_kwargs={
                        "transport": TelegramFallbackTransport(
                            fallback_ips, **_transport_kwargs
                        )
                    },
                )
            elif proxy_url:
                logger.info("[%s] Proxy detected; passing explicitly to HTTPXRequest: %s", self.name, proxy_url)
                request = HTTPXRequest(
                    **request_kwargs, proxy=proxy_url, httpx_kwargs=_with_limits()
                )
                get_updates_request = HTTPXRequest(
                    **request_kwargs, proxy=proxy_url, httpx_kwargs=_with_limits()
                )
            else:
                if disable_fallback:
                    logger.info("[%s] Telegram fallback-IP transport disabled via env", self.name)
                request = HTTPXRequest(**request_kwargs, httpx_kwargs=_with_limits())
                get_updates_request = HTTPXRequest(
                    **request_kwargs, httpx_kwargs=_with_limits()
                )

            builder = builder.request(request).get_updates_request(get_updates_request)
            self._app = builder.build()
            self._bot = self._app.bot
            
            # Register handlers
            self._app.add_handler(TelegramMessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._handle_text_message
            ))
            self._app.add_handler(TelegramMessageHandler(
                filters.COMMAND,
                self._handle_command
            ))
            self._app.add_handler(TelegramMessageHandler(
                filters.LOCATION | getattr(filters, "VENUE", filters.LOCATION),
                self._handle_location_message
            ))
            self._app.add_handler(TelegramMessageHandler(
                filters.PHOTO | filters.VIDEO | filters.AUDIO | filters.VOICE | filters.Document.ALL | filters.Sticker.ALL,
                self._handle_media_message
            ))
            # Handle inline keyboard button callbacks (update prompts)
            self._app.add_handler(CallbackQueryHandler(self._handle_callback_query))
            
            # Start polling — retry initialize() for transient TLS resets.
            # Each attempt is capped by _init_timeout so a single unreachable
            # fallback-IP chain can't block startup indefinitely.
            _max_connect = 8
            _init_timeout = _env_float("HERMES_TELEGRAM_INIT_TIMEOUT", 30.0)
            for _attempt in range(_max_connect):
                try:
                    logger.warning(
                        "[%s] Connecting to Telegram (attempt %d/%d)…",
                        self.name, _attempt + 1, _max_connect,
                    )
                    await _await_with_thread_deadline(
                        self._app.initialize(),
                        timeout=_init_timeout,
                        # On timeout the initialize() task is abandoned without
                        # awaiting its cancellation (it may be wedged in a
                        # shielded scope). Best-effort release the half-built
                        # app's httpx client/connection pool so it isn't leaked
                        # across the retry ladder (mirrors the client-close-on-
                        # timeout pattern in agent/auxiliary_client.py).
                        on_abandon=lambda app=self._app: _shutdown_abandoned_app(app),
                    )
                    break
                except asyncio.TimeoutError:
                    if _attempt < _max_connect - 1:
                        wait = min(2 ** _attempt, 15)
                        logger.warning(
                            "[%s] Connect attempt %d/%d timed out after %.0fs — retrying in %ds",
                            self.name, _attempt + 1, _max_connect, _init_timeout, wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise OSError(
                            f"Telegram initialization timed out after {_max_connect} attempts "
                            f"({_init_timeout:.0f}s each). Check network connectivity to api.telegram.org "
                            f"or set HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT to a lower value."
                        )
                except OSError as init_err:
                    if _attempt < _max_connect - 1:
                        wait = min(2 ** _attempt, 15)
                        logger.warning(
                            "[%s] Connect attempt %d/%d failed: %s — retrying in %ds",
                            self.name, _attempt + 1, _max_connect, init_err, wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise
                except Exception as init_err:
                    if not self._looks_like_network_error(init_err):
                        raise
                    if _attempt < _max_connect - 1:
                        wait = min(2 ** _attempt, 15)
                        logger.warning(
                            "[%s] Connect attempt %d/%d failed: %s — retrying in %ds",
                            self.name, _attempt + 1, _max_connect, init_err, wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise
            await self._app.start()

            # Decide between webhook and polling mode
            webhook_url = os.getenv("TELEGRAM_WEBHOOK_URL", "").strip()

            if webhook_url:
                # ── Webhook mode ─────────────────────────────────────
                # Telegram pushes updates to our HTTP endpoint.  This
                # enables cloud platforms (Fly.io, Railway) to auto-wake
                # suspended machines on inbound HTTP traffic.
                #
                # SECURITY: TELEGRAM_WEBHOOK_SECRET is REQUIRED. Without it,
                # python-telegram-bot passes secret_token=None and the
                # webhook endpoint accepts any HTTP POST — attackers can
                # inject forged updates as if from Telegram. Refuse to
                # start rather than silently run in fail-open mode.
                # See GHSA-3vpc-7q5r-276h.
                webhook_port = env_int("TELEGRAM_WEBHOOK_PORT", 8443)
                webhook_secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
                if not webhook_secret:
                    raise RuntimeError(
                        "TELEGRAM_WEBHOOK_SECRET is required when "
                        "TELEGRAM_WEBHOOK_URL is set. Without it, the "
                        "webhook endpoint accepts forged updates from "
                        "anyone who can reach it — see "
                        "https://github.com/NousResearch/hermes-agent/"
                        "security/advisories/GHSA-3vpc-7q5r-276h.\n\n"
                        "Generate a secret and set it in your .env:\n"
                        "  export TELEGRAM_WEBHOOK_SECRET=\"$(openssl rand -hex 32)\"\n\n"
                        "Then register it with Telegram when setting the "
                        "webhook via setWebhook's secret_token parameter."
                    )
                from urllib.parse import urlparse
                webhook_path = urlparse(webhook_url).path or "/telegram"

                await self._app.updater.start_webhook(
                    listen="0.0.0.0",
                    port=webhook_port,
                    url_path=webhook_path,
                    webhook_url=webhook_url,
                    secret_token=webhook_secret,
                    allowed_updates=Update.ALL_TYPES,
                    # Webhooks are push-based — Telegram does not hold a
                    # server-side getUpdates queue, so this flag is a no-op
                    # in practice. Mirror the polling path's reconnect
                    # semantics for consistency.
                    drop_pending_updates=not is_reconnect,
                )
                self._webhook_mode = True
                logger.info(
                    "[%s] Webhook server listening on 0.0.0.0:%d%s",
                    self.name, webhook_port, webhook_path,
                )
            else:
                # ── Polling mode (default) ───────────────────────────
                # Clear any stale webhook first so polling doesn't inherit a
                # previous webhook registration and silently stop receiving
                # updates. Best-effort: a transient Bot API network error here
                # must not fail gateway startup — degrade to background polling
                # recovery instead.
                await self._delete_webhook_best_effort()

                loop = asyncio.get_running_loop()

                def _polling_error_callback(error: Exception) -> None:
                    if self._polling_error_task and not self._polling_error_task.done():
                        return
                    if self._looks_like_polling_conflict(error):
                        # Synchronously stop PTB's internal network_retry_loop
                        # BEFORE scheduling our async recovery task.  PTB calls
                        # this callback synchronously inside its loop and then
                        # keeps polling on its own; if we only schedule a task
                        # here, PTB's retry and our stop->restart overlap and
                        # produce a fresh 409.  Disarming the loop now makes it
                        # exit on its next tick so recovery owns polling alone.
                        self._disarm_ptb_retry_loop()
                        self._polling_error_task = loop.create_task(self._handle_polling_conflict(error))
                        self._background_tasks.add(self._polling_error_task)
                        self._polling_error_task.add_done_callback(self._background_tasks.discard)
                    elif self._looks_like_network_error(error):
                        logger.warning("[%s] Telegram network error, scheduling reconnect: %s", self.name, error)
                        self._polling_error_task = loop.create_task(self._handle_polling_network_error(error))
                        self._background_tasks.add(self._polling_error_task)
                        self._polling_error_task.add_done_callback(self._background_tasks.discard)
                    else:
                        logger.error("[%s] Telegram polling error: %s", self.name, error, exc_info=True)

                # Store reference for retry use in _handle_polling_conflict
                self._polling_error_callback_ref = _polling_error_callback

                polling_started = await self._start_polling_resilient(
                    # On a cold first boot drop the stale Bot API queue; on a
                    # watcher reconnect after an outage preserve it so messages
                    # sent while the bot was offline are delivered (#46621).
                    drop_pending_updates=not is_reconnect,
                    error_callback=_polling_error_callback,
                )
                if not polling_started:
                    logger.warning(
                        "[%s] Connected in degraded Telegram mode: gateway is alive, "
                        "polling will be retried in the background",
                        self.name,
                    )
            
            self._mark_connected()
            mode = "webhook" if self._webhook_mode else "polling"
            logger.warning("[%s] Connected to Telegram (%s mode)", self.name, mode)

            # Start the persistent heartbeat loop in polling mode. Webhook mode
            # receives updates via incoming pushes — there is no long-poll
            # socket to wedge in CLOSE-WAIT, so the loop is not needed there.
            if not self._webhook_mode:
                if self._polling_heartbeat_task and not self._polling_heartbeat_task.done():
                    self._polling_heartbeat_task.cancel()
                self._polling_heartbeat_task = asyncio.ensure_future(
                    self._polling_heartbeat_loop()
                )

            # Command-menu registration, DM-topic setup, and the status
            # indicator each make Bot API calls that can stall for certain
            # tokens. Running them here — inside the connect() coroutine that
            # the gateway wraps in a connect timeout — means one slow call
            # blows the whole connect and the adapter never comes up, even
            # though polling/webhook is already live (#46298). Defer them to a
            # cancellable background task so connect() returns as soon as the
            # transport is up.
            self._start_post_connect_housekeeping()

            return True
            
        except Exception as e:
            self._release_platform_lock()
            safe_error = _redact_telegram_error_text(e)
            message = f"Telegram startup failed: {safe_error}"
            self._set_fatal_error("telegram_connect_error", message, retryable=True)
            logger.error("[%s] Failed to connect to Telegram: %s", self.name, safe_error)
            return False

    async def _set_status_indicator(self, online: bool) -> None:
        """Set the bot's short description to the online/offline status text.

        The short description is the line shown under the bot's name in its
        profile. It is the closest Bot API surface to a presence indicator —
        bots have no real online/offline dot (that's a user-account feature).

        No-op unless ``extra.status_indicator`` is enabled. Best-effort: any
        failure is logged at debug and swallowed so it never blocks connect or
        disconnect. The default (no language_code) description applies to every
        user who doesn't have a language-specific one set.
        """
        if not getattr(self, "_status_indicator_enabled", False):
            return
        bot = self._bot
        if bot is None:
            return
        text = self._status_online_text if online else self._status_offline_text
        # Telegram caps short_description at 120 chars.
        text = text[:120]
        try:
            await bot.set_my_short_description(short_description=text)
            logger.info("[%s] Set bot status indicator to %r", self.name, text)
        except Exception as e:
            logger.debug(
                "[%s] Failed to set bot status indicator to %r: %s",
                self.name, text, e,
            )

    async def _cancel_pending_delivery_tasks(self) -> None:
        """Cancel every delayed-delivery task family before disconnect completes.

        Covers media-group, photo-batch and text-batch flush tasks plus the
        polling-error recovery task. Each sits behind an ``asyncio.sleep()``;
        if teardown leaves them running they dispatch ``handle_message`` into a
        torn-down session. Skips the current task so the coroutine driving
        teardown does not cancel itself.
        """
        current_task = asyncio.current_task()
        pending_tasks: list[asyncio.Task] = []
        awaitable_tasks: list[asyncio.Task] = []
        seen: set[int] = set()

        def collect(task: Optional[asyncio.Task]) -> None:
            if not task or task.done() or task is current_task:
                return
            marker = id(task)
            if marker in seen:
                return
            seen.add(marker)
            pending_tasks.append(task)
            if asyncio.isfuture(task) or asyncio.iscoroutine(task):
                awaitable_tasks.append(task)

        for task in list(self._media_group_tasks.values()):
            collect(task)
        for task in list(self._pending_photo_batch_tasks.values()):
            collect(task)
        for task in list(self._pending_text_batch_tasks.values()):
            collect(task)
        collect(self._polling_error_task)

        for task in pending_tasks:
            task.cancel()
        if awaitable_tasks:
            await asyncio.gather(*awaitable_tasks, return_exceptions=True)

        self._media_group_tasks.clear()
        self._media_group_events.clear()
        self._pending_photo_batch_tasks.clear()
        self._pending_photo_batches.clear()
        self._pending_text_batch_tasks.clear()
        self._pending_text_batches.clear()
        if self._polling_error_task is not current_task:
            self._polling_error_task = None

    async def disconnect(self) -> None:
        """Stop polling/webhook, cancel pending delayed deliveries, and disconnect."""
        # Mark disconnected first so the drop guard short-circuits any flush
        # that wins the race against teardown and prevents new delayed tasks
        # from being scheduled by late update handlers.
        self._mark_disconnected()

        # Cancel deferred post-connect housekeeping (command-menu / DM-topic /
        # status-indicator Bot API calls) so it cannot fire into a half-torn-down
        # bot client (#46298). getattr guards the object.__new__ test pattern
        # where __init__ (which sets this attr) is never called.
        post_connect_task = getattr(self, "_post_connect_task", None)
        if post_connect_task and not post_connect_task.done():
            post_connect_task.cancel()
            await asyncio.gather(post_connect_task, return_exceptions=True)
        self._post_connect_task = None

        # Cancel the heartbeat before tearing down the app so the probe task
        # cannot fire get_me() into a half-shutdown bot client.
        if self._polling_heartbeat_task and not self._polling_heartbeat_task.done():
            self._polling_heartbeat_task.cancel()
            try:
                await self._polling_heartbeat_task
            except asyncio.CancelledError:
                pass
        self._polling_heartbeat_task = None

        # Mark the bot "Offline" in its short description while the bot's HTTP
        # client is still alive (before app shutdown closes it). Opt-in via
        # extra.status_indicator. Non-fatal. This is the clean-shutdown path;
        # a hard crash leaves the last-known status, which is the expected
        # limitation of a profile-text indicator.
        try:
            await self._set_status_indicator(online=False)
        except Exception:
            pass

        await self._cancel_pending_delivery_tasks()

        if self._app:
            try:
                # Only stop the updater if it's running.  Bounded with a
                # timeout: a CLOSE-WAIT socket can wedge stop() on epoll
                # indefinitely, which would hang disconnect() (and any
                # gateway shutdown/restart waiting on it) forever.  On timeout
                # we fall through to app.stop()/shutdown() to force teardown.
                if self._app.updater and self._app.updater.running:
                    try:
                        await asyncio.wait_for(self._app.updater.stop(), timeout=_UPDATER_STOP_TIMEOUT)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[%s] updater.stop() timed out during disconnect "
                            "(likely CLOSE-WAIT socket); forcing app shutdown",
                            self.name,
                        )
                if self._app.running:
                    await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                logger.warning(
                    "[%s] Error during Telegram disconnect: %s",
                    self.name, _redact_telegram_error_text(e),
                )
        self._release_platform_lock()

        self._app = None
        self._bot = None
        logger.info("[%s] Disconnected from Telegram", self.name)

    def _should_thread_reply(self, reply_to: Optional[str], chunk_index: int) -> bool:
        """Determine if this message chunk should thread to the original message.

        Args:
            reply_to: The original message ID to reply to
            chunk_index: Index of this chunk (0 = first chunk)

        Returns:
            True if this chunk should be threaded to the original message
        """
        if not reply_to:
            return False
        mode = self._reply_to_mode
        if mode == "off":
            return False
        elif mode == "all":
            return True
        else:  # "first" (default)
            return chunk_index == 0

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Send a message to a Telegram chat."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        # getattr() — tests build adapters via object.__new__() (no __init__).
        if getattr(self, "_send_path_degraded", False):
            return SendResult(success=False, error="send_path_degraded", retryable=True)

        # Skip whitespace-only text to prevent Telegram 400 empty-text errors.
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)
        
        try:
            # Bot API 10.1 rich fast-path: send the raw agent markdown via
            # sendRichMessage so tables/task lists/etc. render natively. Falls
            # through to the legacy MarkdownV2 path on permanent/capability
            # errors or DM-topic routing skips; returns directly on success or
            # on a transient failure (which must NOT be legacy-resent).
            if self._should_attempt_rich(content, metadata=metadata):
                rich_result = await self._try_send_rich(chat_id, content, reply_to, metadata)
                if rich_result is not None:
                    if rich_result.success:
                        # Re-trigger typing like the legacy success path does,
                        # but ONLY for intermediate sends. On the final reply
                        # (metadata["notify"]) the gateway has already torn down
                        # the typing refresh loop; re-arming Telegram's ~5s timer
                        # here would leave the "...typing" bubble lingering after
                        # the answer (no Bot API call cancels it). See #48678.
                        if not (metadata or {}).get("notify"):
                            try:
                                await self.send_typing(chat_id, metadata=metadata)
                            except Exception:
                                pass  # Typing failures are non-fatal
                    return rich_result

            # Format and split message if needed
            formatted = self.format_message(content)
            chunks = self.truncate_message(
                formatted, self.MAX_MESSAGE_LENGTH, len_fn=utf16_len,
            )
            if len(chunks) > 1:
                # truncate_message appends a raw " (1/2)" suffix. Escape the
                # MarkdownV2-special parentheses so Telegram doesn't reject the
                # chunk and fall back to plain text.
                chunks = [
                    _separate_chunk_indicator_from_fence(
                        re.sub(r" \((\d+)/(\d+)\)$", r" \\(\1/\2\\)", chunk)
                    )
                    for chunk in chunks
                ]
            
            message_ids = []
            thread_id = self._metadata_thread_id(metadata)
            requested_thread_id = self._message_thread_id_for_send(thread_id)
            used_thread_fallback = False
            
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

            for i, chunk in enumerate(chunks):
                retried_thread_not_found = False
                metadata_reply_to = self._metadata_reply_to_message_id(metadata)
                private_dm_topic_send = self._is_private_dm_topic_send(chat_id, thread_id, metadata)
                # reply_to_mode="off" on the existing telegram_dm_topic_reply_fallback path
                # is an explicit user opt-in to "message_thread_id alone is enough" (PR #23994
                # / commit 21a15b671). Honor it — don't fail loud just because the anchor was
                # suppressed by config. The new fail-loud contract only applies when the caller
                # didn't ask for the anchor to be dropped.
                dm_topic_reply_to_off = (
                    private_dm_topic_send
                    and self._reply_to_mode == "off"
                    and bool(metadata and metadata.get("telegram_dm_topic_reply_fallback"))
                )
                reply_to_source = reply_to or (
                    str(metadata_reply_to) if private_dm_topic_send and metadata_reply_to is not None else None
                )
                if private_dm_topic_send:
                    should_thread = (
                        reply_to_source is not None
                        and self._reply_to_mode != "off"
                    )
                else:
                    should_thread = self._should_thread_reply(reply_to_source, i)
                reply_to_id = int(reply_to_source) if should_thread and reply_to_source else None
                if private_dm_topic_send and reply_to_id is None and not dm_topic_reply_to_off:
                    return SendResult(
                        success=False,
                        error=self._dm_topic_missing_anchor_error(),
                        retryable=False,
                    )
                thread_kwargs = self._thread_kwargs_for_send(
                    chat_id,
                    thread_id,
                    metadata,
                    reply_to_message_id=reply_to_id,
                    reply_to_mode=self._reply_to_mode,
                )
                if used_thread_fallback and thread_kwargs.get("message_thread_id") is not None:
                    thread_kwargs = dict(thread_kwargs)
                    thread_kwargs["message_thread_id"] = None
                effective_thread_id = thread_kwargs.get("message_thread_id")

                msg = None
                for _send_attempt in range(3):
                    try:
                        # Try Markdown first, fall back to plain text if it fails
                        try:
                            msg = await self._bot.send_message(
                                chat_id=normalize_telegram_chat_id(chat_id),
                                text=chunk,
                                parse_mode=ParseMode.MARKDOWN_V2,
                                reply_to_message_id=reply_to_id,
                                **thread_kwargs,
                                **self._link_preview_kwargs(),
                                **self._notification_kwargs(metadata),
                            )
                        except Exception as md_error:
                            # Markdown parsing failed, try plain text
                            if "parse" in str(md_error).lower() or "markdown" in str(md_error).lower():
                                logger.warning("[%s] MarkdownV2 parse failed, falling back to plain text: %s", self.name, md_error)
                                plain_chunk = _strip_mdv2(chunk)
                                msg = await self._bot.send_message(
                                    chat_id=normalize_telegram_chat_id(chat_id),
                                    text=plain_chunk,
                                    parse_mode=None,
                                    reply_to_message_id=reply_to_id,
                                    **thread_kwargs,
                                    **self._link_preview_kwargs(),
                                    **self._notification_kwargs(metadata),
                                )
                            else:
                                raise
                        break  # success
                    except _NetErr as send_err:
                        # BadRequest is a subclass of NetworkError in
                        # python-telegram-bot but represents permanent errors
                        # (not transient network issues). Detect and handle
                        # specific cases instead of blindly retrying.
                        if _BadReq and isinstance(send_err, _BadReq):
                            if self._is_thread_not_found_error(send_err) and effective_thread_id is not None:
                                if private_dm_topic_send or (metadata and metadata.get("telegram_dm_topic_created_for_send")):
                                    return SendResult(
                                        success=False,
                                        error=str(send_err),
                                        retryable=False,
                                    )
                                # Telegram has been observed to return a
                                # one-off "thread not found" that recovers on
                                # an immediate retry (transient flake — see
                                # test_send_retries_transient_thread_not_found_before_fallback).
                                # Try the same thread_id once without sleeping
                                # before falling back to a plain send.
                                if not retried_thread_not_found:
                                    retried_thread_not_found = True
                                    logger.warning(
                                        "[%s] Thread %s not found, retrying once with same thread_id",
                                        self.name, effective_thread_id,
                                    )
                                    continue
                                # Second failure: the thread is genuinely gone.
                                # Retry without ``message_thread_id`` so the
                                # message still reaches the chat, and prune
                                # the stale binding so future inbound
                                # messages aren't redirected back to it
                                # (#31501).
                                logger.warning(
                                    "[%s] Thread %s not found, retrying without message_thread_id",
                                    self.name, effective_thread_id,
                                )
                                self._prune_stale_dm_topic_binding(
                                    chat_id, effective_thread_id,
                                )
                                used_thread_fallback = True
                                effective_thread_id = None
                                thread_kwargs = {"message_thread_id": None}
                                continue
                            err_lower = str(send_err).lower()
                            if "message to be replied not found" in err_lower and reply_to_id is not None:
                                if private_dm_topic_send:
                                    safe_send_error = _redact_telegram_error_text(send_err)
                                    return SendResult(
                                        success=False,
                                        error=safe_send_error,
                                        retryable=False,
                                    )
                                # Original message was deleted before we
                                # could reply. For private-topic fallback
                                # sends, message_thread_id is only valid with
                                # the reply anchor, so drop both together.
                                safe_send_error = _redact_telegram_error_text(send_err)
                                logger.warning(
                                    "[%s] Reply target deleted, retrying without reply_to: %s",
                                    self.name, safe_send_error,
                                )
                                reply_to_id = None
                                if metadata and metadata.get("telegram_dm_topic_reply_fallback"):
                                    thread_kwargs = {}
                                    effective_thread_id = None
                                else:
                                    thread_kwargs = self._thread_kwargs_for_send(
                                        chat_id,
                                        thread_id,
                                        metadata,
                                        reply_to_message_id=reply_to_id,
                                        reply_to_mode=self._reply_to_mode,
                                    )
                                    effective_thread_id = thread_kwargs.get("message_thread_id")
                                continue
                            # Other BadRequest errors are permanent — don't retry
                            raise
                        # TimedOut is also a subclass of NetworkError. A
                        # generic timeout may have reached Telegram, so don't
                        # retry; a wrapped ConnectTimeout means no connection
                        # was established, so retrying is safe. A pool timeout
                        # (httpx pool exhausted) is explicitly "not sent to
                        # Telegram" -- retrying through the loop is safe and
                        # prevents silent drops when the pool frees up.
                        is_pool_timeout = self._looks_like_pool_timeout(send_err)
                        if (
                            _TimedOut
                            and isinstance(send_err, _TimedOut)
                            and not self._looks_like_connect_timeout(send_err)
                            and not is_pool_timeout
                        ):
                            raise
                        if is_pool_timeout:
                            await self._drain_general_connections_after_pool_timeout()
                        if _send_attempt < 2:
                            wait = 2 ** _send_attempt
                            safe_send_error = _redact_telegram_error_text(send_err)
                            logger.warning("[%s] Network error on send (attempt %d/3), retrying in %ds: %s",
                                           self.name, _send_attempt + 1, wait, safe_send_error)
                            await asyncio.sleep(wait)
                        else:
                            raise
                    except Exception as send_err:
                        retry_after = getattr(send_err, "retry_after", None)
                        if retry_after is not None or "retry after" in str(send_err).lower():
                            if _send_attempt < 2:
                                wait = float(retry_after) if retry_after is not None else 1.0
                                safe_send_error = _redact_telegram_error_text(send_err)
                                logger.warning(
                                    "[%s] Telegram flood control on send (attempt %d/3), retrying in %.1fs: %s",
                                    self.name,
                                    _send_attempt + 1,
                                    wait,
                                    safe_send_error,
                                )
                                await asyncio.sleep(wait)
                                continue
                        raise
                message_ids.append(str(msg.message_id))

            # Re-trigger typing indicator after sending a message.
            # Telegram clears the typing state when a new message is delivered,
            # so without this the "...typing" bubble disappears mid-response
            # (especially noticeable when the agent sends intermediate progress
            # messages like "Checking:" before running tools).
            # Skip this on the FINAL reply (metadata["notify"]): the gateway has
            # already cancelled the typing refresh loop by the time the final
            # send returns, so re-arming Telegram's ~5s timer here would leave
            # the indicator lingering after the answer with nothing to cancel
            # it (Telegram exposes no stop-typing API). See #48678.
            if not (metadata or {}).get("notify"):
                try:
                    await self.send_typing(chat_id, metadata=metadata)
                except Exception:
                    pass  # Typing failures are non-fatal

            return SendResult(
                success=True,
                message_id=message_ids[0] if message_ids else None,
                raw_response={
                    "message_ids": message_ids,
                    "requested_thread_id": requested_thread_id,
                    "thread_fallback": used_thread_fallback,
                },
            )
            
        except Exception as e:
            safe_error = _redact_telegram_error_text(e)
            logger.error("[%s] Failed to send Telegram message: %s", self.name, safe_error)
            err_str = str(e).lower()
            error_kind = classify_send_error(e)
            # Message too long — content exceeded 4096 chars. Return failure so
            # stream consumer enters fallback mode and sends the remainder.
            if "message_too_long" in err_str or "too long" in err_str:
                logger.debug(
                    "[%s] send() content too long, falling back to new-message continuation",
                    self.name,
                )
                return SendResult(success=False, error="message_too_long", error_kind="too_long")
            # TimedOut usually means the request may have reached Telegram —
            # mark as non-retryable so _send_with_retry() doesn't re-send.
            # Exceptions: a wrapped ConnectTimeout (no connection established)
            # and an httpx pool timeout (request explicitly not sent) -- both
            # are safe to re-send and must not be silently dropped.
            _to = locals().get("_TimedOut")
            is_timeout = (_to and isinstance(e, _to)) or "timed out" in err_str
            is_connect_timeout = self._looks_like_connect_timeout(e)
            is_pool_timeout = self._looks_like_pool_timeout(e)
            return SendResult(
                success=False,
                error=safe_error,
                retryable=(is_connect_timeout or is_pool_timeout or not is_timeout),
                error_kind=error_kind,
            )

    async def send_or_update_status(
        self,
        chat_id: str,
        status_key: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a status message, or edit the previous one with the same key.

        Issue #30045: progress/status callbacks (context-pressure, lifecycle,
        compression, etc.) used to append a fresh bubble on every call. With
        this method, the first call sends and the message id is remembered;
        subsequent calls with the same (chat_id, status_key) edit that same
        message in place. If the edit fails (message deleted, too old, etc.)
        we drop the cached id and send fresh.
        """
        key = (str(chat_id), str(status_key))
        cached_id = self._status_message_ids.get(key)
        if cached_id is not None:
            result = await self.edit_message(
                chat_id, cached_id, content, finalize=True, metadata=metadata,
            )
            if result.success:
                if result.message_id:
                    self._status_message_ids[key] = str(result.message_id)
                return result
            # Edit failed — clear the cached id and fall through to a fresh send.
            self._status_message_ids.pop(key, None)
        result = await self.send(chat_id, content, metadata=metadata)
        if result.success and result.message_id:
            self._status_message_ids[key] = str(result.message_id)
        return result

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Edit a previously sent Telegram message.

        Telegram caps single-message text at 4096 UTF-16 codeunits.  Streaming
        replies that grow past this limit must NOT be silently truncated and
        must NOT return failure (the consumer would re-send and create a
        duplicate).  Instead this method split-and-delivers: edit the
        existing message with the first chunk and send the rest as
        continuation messages, returning the final chunk's id so subsequent
        edits target the most recent visible message.
        """
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        # Rich finalize (Bot API 10.1): when the completed content has
        # constructs the legacy MarkdownV2 edit degrades (tables → bullet
        # lists, task lists, <details>, block math) and rich is available,
        # edit the preview IN PLACE via editMessageText's rich_message param.
        # No fresh send + delete → no duplicate preview (the problem #46206
        # reverted the fresh-final path for).  Attempted before the 4,096
        # overflow pre-flight because the rich text cap is 32,768 — a rich
        # table that exceeds the MarkdownV2 limit must not be split into legacy
        # chunks.  Falls back to the legacy edit path (overflow split included)
        # on capability/permanent rejection.
        if finalize and self._rich_eligible(content):
            rich_result = await self._try_edit_rich(
                chat_id, message_id, content, metadata=metadata,
            )
            if rich_result is not None:
                return rich_result

        # Pre-flight: if content already exceeds the limit, split-and-deliver
        # without round-tripping a doomed edit.  During streaming
        # (finalize=False) we truncate instead of splitting — splitting creates
        # continuation messages whose IDs become the new edit target, and on
        # the next token chunk the full accumulated text is re-edited into the
        # continuation, triggering another split → infinite duplication loop
        # (#48648).  The full content is delivered when finalize=True.
        _preview_key = (str(chat_id), str(message_id))
        _saturated_preview = False
        if finalize:
            # Any saturation state for this message is finished with — the
            # final edit always delivers real (full) content.
            self._last_overflow_preview.pop(_preview_key, None)
        if utf16_len(content) > self.MAX_MESSAGE_LENGTH:
            if finalize:
                return await self._edit_overflow_split(
                    chat_id, message_id, content, finalize=finalize, metadata=metadata,
                )
            content = self._truncate_stream_overflow_preview(content)
            _saturated_preview = True
            # Saturated-preview dedup: past the cap, every progressive edit
            # truncates to the same text. Re-sending it is a visual no-op that
            # still burns flood budget (Telegram counts the request and answers
            # "message is not modified"). ~1 edit/0.8s for the rest of a long
            # stream trips flood control (200s+ penalties) and hangs the final
            # delivery. Skip silently until finalize.
            if self._last_overflow_preview.get(_preview_key) == content:
                return SendResult(success=True, message_id=message_id)
        elif not finalize:
            # Content shrank back under the cap (segment break / new message
            # id) — clear stale saturation state so dedup can't mask a real
            # edit later.
            self._last_overflow_preview.pop(_preview_key, None)

        try:
            if not finalize:
                await self._bot.edit_message_text(
                    chat_id=normalize_telegram_chat_id(chat_id),
                    message_id=int(message_id),
                    text=content,
                )
                if _saturated_preview:
                    self._last_overflow_preview[_preview_key] = content
                return SendResult(success=True, message_id=message_id)

            formatted = self.format_message(content)
            try:
                await self._bot.edit_message_text(
                    chat_id=normalize_telegram_chat_id(chat_id),
                    message_id=int(message_id),
                    text=formatted,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
            except Exception as fmt_err:
                # "Message is not modified" is a no-op, not an error
                if "not modified" in str(fmt_err).lower():
                    return SendResult(success=True, message_id=message_id)
                # Fallback: strip MarkdownV2 escapes and retry as clean plain text
                safe_format_error = _redact_telegram_error_text(fmt_err)
                logger.warning(
                    "[%s] MarkdownV2 edit failed, falling back to plain text: %s",
                    self.name,
                    safe_format_error,
                )
                _plain = _strip_mdv2(content) if content else content
                await self._bot.edit_message_text(
                    chat_id=normalize_telegram_chat_id(chat_id),
                    message_id=int(message_id),
                    text=_plain,
                )
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            err_str = str(e).lower()
            # "Message is not modified" — content identical, treat as success
            if "not modified" in err_str:
                return SendResult(success=True, message_id=message_id)
            # Reactive split-and-deliver: parse_mode formatting can inflate
            # the payload past the limit even when the raw text was under
            # (e.g. MarkdownV2 escapes).  Same fix as the pre-flight path.
            if "message_too_long" in err_str or "too long" in err_str:
                logger.debug(
                    "[%s] edit_message overflow (%d UTF-16 > %d), splitting",
                    self.name, utf16_len(content), self.MAX_MESSAGE_LENGTH,
                )
                if finalize:
                    return await self._edit_overflow_split(
                        chat_id, message_id, content, finalize=finalize, metadata=metadata,
                    )
                # Mid-stream: truncate and retry instead of splitting (#48648).
                truncated = self._truncate_stream_overflow_preview(content)
                if self._last_overflow_preview.get(_preview_key) == truncated:
                    # Saturated-preview dedup (see pre-flight path above).
                    return SendResult(success=True, message_id=message_id)
                await self._bot.edit_message_text(
                    chat_id=normalize_telegram_chat_id(chat_id),
                    message_id=int(message_id),
                    text=truncated,
                )
                self._last_overflow_preview[_preview_key] = truncated
                return SendResult(success=True, message_id=message_id)
            # Flood control / RetryAfter — short waits are retried inline,
            # long waits return a failure immediately so streaming can fall back
            # to a normal final send instead of leaving a truncated partial.
            retry_after = getattr(e, "retry_after", None)
            if retry_after is not None or "retry after" in err_str:
                wait = retry_after if retry_after else 1.0
                logger.warning(
                    "[%s] Telegram flood control, waiting %.1fs",
                    self.name, wait,
                )
                if wait > 5.0:
                    return SendResult(
                        success=False,
                        error=f"flood_control:{wait}",
                        retry_after=float(wait),
                    )
                await asyncio.sleep(wait)
                try:
                    await self._bot.edit_message_text(
                        chat_id=normalize_telegram_chat_id(chat_id),
                        message_id=int(message_id),
                        text=content,
                    )
                    return SendResult(success=True, message_id=message_id)
                except Exception as retry_err:
                    safe_retry_error = _redact_telegram_error_text(retry_err)
                    logger.error(
                        "[%s] Edit retry failed after flood wait: %s",
                        self.name, safe_retry_error,
                    )
                    return SendResult(success=False, error=safe_retry_error)
            # Transient network errors (ConnectError, timeouts, server
            # disconnects) should not permanently disable progress-message
            # editing.  Mark the result retryable so the caller knows it
            # can keep trying on the next update cycle.
            _transient_markers = (
                "connecterror",
                "connect error",
                "connection error",
                "networkerror",
                "network error",
                "timed out",
                "readtimeout",
                "writetimeout",
                "server disconnected",
                "temporarily unavailable",
                "temporary failure",
                "httpx",
            )
            _is_transient = any(m in err_str for m in _transient_markers)
            if _is_transient:
                safe_error = _redact_telegram_error_text(e)
                logger.warning(
                    "[%s] Transient network error editing message %s (will retry): %s",
                    self.name,
                    message_id,
                    safe_error,
                )
                return SendResult(success=False, error=safe_error, retryable=True)
            safe_error = _redact_telegram_error_text(e)
            logger.error(
                "[%s] Failed to edit Telegram message %s: %s",
                self.name,
                message_id,
                safe_error,
            )
            return SendResult(success=False, error=safe_error)

    def _truncate_stream_overflow_preview(self, content: str) -> str:
        """Return a one-message preview for oversized streaming edits.

        Streaming edits must keep targeting the original message. Splitting a
        mid-stream preview creates continuation messages and moves the active
        message id, so the next accumulated-token edit repeats the overflow
        cycle (#48648). Final edits still use ``_edit_overflow_split`` to
        deliver the complete response.
        """
        return self.truncate_message(
            content,
            self.MAX_MESSAGE_LENGTH,
            len_fn=utf16_len,
        )[0]

    async def _edit_overflow_split(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Split an oversized edit across the existing message + continuations.

        Edit the original ``message_id`` with chunk 1 (with the platform's
        usual ``(1/N)`` suffix preserved), then send the remaining chunks as
        new messages threaded as replies to the previous chunk so the user
        sees them grouped.  Returns ``SendResult(success=True,
        message_id=<last-chunk-id>, continuation_message_ids=(...))`` so the
        stream consumer can keep editing the most recent visible message
        and the gateway has full visibility into every message id we put on
        screen.

        Falls back to ``SendResult(success=False)`` only if even the first-
        chunk edit fails — that's a real adapter problem, not an overflow.
        """
        chunks = self.truncate_message(
            content, self.MAX_MESSAGE_LENGTH, len_fn=utf16_len,
        )
        if len(chunks) <= 1:
            # Defensive: shouldn't happen given the caller's pre-flight, but
            # if truncate_message returned a single chunk just edit normally.
            chunks = [content]

        # Step 1 — edit the existing message with the first chunk.
        first_chunk = chunks[0]
        try:
            if finalize:
                # Use format_message + parse_mode for the final chunk;
                # mirror edit_message's main happy-path.
                formatted = _separate_chunk_indicator_from_fence(
                    self.format_message(first_chunk)
                )
                try:
                    await self._bot.edit_message_text(
                        chat_id=normalize_telegram_chat_id(chat_id),
                        message_id=int(message_id),
                        text=formatted,
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )
                except Exception as fmt_err:
                    if "not modified" not in str(fmt_err).lower():
                        logger.warning(
                            "[%s] Overflow split: MarkdownV2 first-chunk edit "
                            "failed, falling back to plain text: %s",
                            self.name, fmt_err,
                        )
                        await self._bot.edit_message_text(
                            chat_id=normalize_telegram_chat_id(chat_id),
                            message_id=int(message_id),
                            text=_strip_mdv2(first_chunk),
                        )
            else:
                await self._bot.edit_message_text(
                    chat_id=normalize_telegram_chat_id(chat_id),
                    message_id=int(message_id),
                    text=first_chunk,
                )
        except Exception as e:
            err_str = str(e).lower()
            if "not modified" in err_str:
                # First chunk identical to current text — fall through to
                # send continuations.
                pass
            else:
                logger.error(
                    "[%s] Overflow split: first-chunk edit failed: %s",
                    self.name, e, exc_info=True,
                )
                return SendResult(success=False, error=str(e))

        # Step 2 — send each remaining chunk as a continuation message,
        # threaded as a reply to the previous so the user sees them as a
        # contiguous block.  We call self._bot.send_message directly so the
        # continuation skips ``self.send``'s own pre-chunking pass (chunks
        # are already correctly sized).  Best-effort MarkdownV2 with plain
        # fallback, mirroring send().
        continuation_ids: list[str] = []
        delivered_chunks = [first_chunk]
        prev_id = message_id
        thread_id = self._metadata_thread_id(metadata)
        for chunk in chunks[1:]:
            sent_msg = None
            reply_to_id = int(prev_id) if prev_id else None
            thread_kwargs = self._thread_kwargs_for_send(
                chat_id,
                thread_id,
                metadata,
                reply_to_message_id=reply_to_id,
            )
            for use_markdown in (True, False) if finalize else (False,):
                try:
                    if use_markdown:
                        text = _separate_chunk_indicator_from_fence(
                            self.format_message(chunk)
                        )
                    else:
                        # Plain attempt: on finalize the MarkdownV2 attempt
                        # failed, so degrade to clean stripped text, never
                        # the raw chunk (raw ** / ``` markers would render
                        # literally); streaming previews stay raw.
                        text = _strip_mdv2(chunk) if finalize else chunk
                    sent_msg = await self._bot.send_message(
                        chat_id=normalize_telegram_chat_id(chat_id),
                        text=text,
                        parse_mode=ParseMode.MARKDOWN_V2 if use_markdown else None,
                        reply_to_message_id=reply_to_id,
                        **thread_kwargs,
                        **self._link_preview_kwargs(),
                        **self._notification_kwargs(metadata),
                    )
                    break
                except Exception as send_err:
                    if "reply message not found" in str(send_err).lower():
                        # Drop the reply anchor and try again.  Private DM
                        # topic fallback needs the anchor and topic id together;
                        # forum topics can still safely keep message_thread_id.
                        retry_thread_kwargs = (
                            {}
                            if metadata and metadata.get("telegram_dm_topic_reply_fallback")
                            else self._thread_kwargs_for_send(
                                chat_id, thread_id, metadata, reply_to_message_id=None
                            )
                        )
                        try:
                            sent_msg = await self._bot.send_message(
                                chat_id=normalize_telegram_chat_id(chat_id),
                                text=_strip_mdv2(chunk) if finalize else chunk,
                                **retry_thread_kwargs,
                                **self._link_preview_kwargs(),
                                **self._notification_kwargs(metadata),
                            )
                            break
                        except Exception as _retry_err:
                            logger.warning(
                                "[%s] Overflow continuation no-reply retry failed: %s",
                                self.name, _retry_err,
                            )
                            sent_msg = None
                            break
                    if use_markdown:
                        # try plain text on next loop iteration
                        continue
                    logger.warning(
                        "[%s] Overflow continuation send failed: %s",
                        self.name, send_err,
                    )
                    sent_msg = None
                    break
            if sent_msg is None:
                # Continuation failed — the user has chunk 1 + however many
                # continuations succeeded, but NOT the full response.  Do not
                # report success: the stream consumer treats a successful edit
                # as final delivery on got_done, which would suppress fallback
                # delivery and leave the Telegram topic clipped after the last
                # delivered chunk.
                logger.warning(
                    "[%s] Overflow split: stopped at %d/%d chunks delivered",
                    self.name, 1 + len(continuation_ids), len(chunks),
                )
                delivered_prefix = "".join(
                    re.sub(r" \(\d+/\d+\)$", "", delivered)
                    for delivered in delivered_chunks
                )
                return SendResult(
                    success=False,
                    message_id=prev_id,
                    error="overflow_continuation_failed",
                    retryable=True,
                    raw_response={
                        "partial_overflow": True,
                        "delivered_chunks": 1 + len(continuation_ids),
                        "total_chunks": len(chunks),
                        "last_message_id": prev_id,
                        "delivered_prefix": delivered_prefix,
                        "continuation_message_ids": tuple(continuation_ids),
                    },
                    continuation_message_ids=tuple(continuation_ids),
                )
            new_id = str(getattr(sent_msg, "message_id", "")) or prev_id
            continuation_ids.append(new_id)
            delivered_chunks.append(chunk)
            prev_id = new_id

        last_id = continuation_ids[-1] if continuation_ids else message_id
        logger.debug(
            "[%s] Overflow split delivered %d chunks; last_id=%s",
            self.name, 1 + len(continuation_ids), last_id,
        )
        return SendResult(
            success=True,
            message_id=last_id,
            continuation_message_ids=tuple(continuation_ids),
        )

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a previously sent Telegram message.

        Used by the stream consumer's fresh-final cleanup path (ported
        from openclaw/openclaw#72038) to remove long-lived preview
        messages after sending the completed reply as a fresh message.
        Telegram's Bot API ``deleteMessage`` works for bot-posted
        messages in the last 48 hours.  Failures are non-fatal — the
        caller leaves the preview in place and logs at debug level.
        """
        if not self._bot:
            return False
        try:
            await self._bot.delete_message(
                chat_id=normalize_telegram_chat_id(chat_id),
                message_id=int(message_id),
            )
            return True
        except Exception as e:
            logger.debug(
                "[%s] Failed to delete Telegram message %s: %s",
                self.name, message_id, e,
            )
            return False

    def supports_draft_streaming(
        self,
        chat_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Telegram supports sendMessageDraft for private chats only.

        Bot API 9.5 (March 2026) opened ``sendMessageDraft`` to all bots
        unconditionally for private (DM) chats.  Groups, supergroups, and
        channels still rely on the edit-based path.

        We additionally require ``self._bot`` to expose ``send_message_draft``
        (added to python-telegram-bot in 22.6); older PTB installs gracefully
        fall back to the edit path even on DMs.
        """
        if not self._bot or not hasattr(self._bot, "send_message_draft"):
            return False
        return (chat_type or "").lower() in {"dm", "private"}

    async def send_draft(
        self,
        chat_id: str,
        draft_id: int,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Stream a partial message via Telegram's native draft API.

        Uses ``sendRichMessageDraft`` (Bot API 10.1) with the raw markdown when
        rich messages are enabled and supported, otherwise the plain-text
        ``sendMessageDraft``. The Bot API animates the preview when the same
        ``draft_id`` is reused across consecutive calls in the same chat.  When
        the response finishes, the caller sends the final text via the normal
        ``send`` path; the draft preview clears naturally on the client
        (Telegram has no Bot API to "promote" a draft to a real message — the
        final ``sendMessage``/``sendRichMessage`` is what the user receives in
        their history).
        """
        if not self._bot:
            return SendResult(success=False, error="not_connected")

        # Rich draft fast-path (Bot API 10.1 sendRichMessageDraft): render the
        # streaming preview with the same raw markdown the final
        # sendRichMessage will persist, so the animated draft matches the final
        # message. Any failure degrades to the legacy plain-text draft below.
        if self._should_attempt_rich_draft(content):
            if await self._try_send_rich_draft(chat_id, draft_id, content, metadata):
                # Drafts have no message_id; report success without one.
                return SendResult(success=True, message_id=None)

        if not hasattr(self._bot, "send_message_draft"):
            return SendResult(success=False, error="api_unavailable")

        # Trim to the same UTF-16 budget the platform enforces on regular
        # sends.  Drafts have the same length contract as messages.
        text = content if len(content) <= self.MAX_MESSAGE_LENGTH else \
            self.truncate_message(content, self.MAX_MESSAGE_LENGTH, len_fn=utf16_len)[0]

        thread_id = self._metadata_thread_id(metadata)

        # Apply the same MarkdownV2 conversion the regular ``send`` path uses
        # so the animated draft preview renders with identical formatting to
        # the final message.  Without this, the draft streams as raw text and
        # the final ``sendMessage`` (which DOES use MarkdownV2) snaps into
        # formatted output, producing a jarring visual shift at the end of the
        # response.  We try MarkdownV2 first and fall back to plain text if a
        # malformed escape would be rejected — mirroring the (True, False)
        # retry the streaming send loop uses — so a single bad token never
        # kills draft streaming for the whole response.
        for use_markdown in (True, False):
            kwargs: Dict[str, Any] = {
                "chat_id": normalize_telegram_chat_id(chat_id),
                "draft_id": int(draft_id),
                "text": self.format_message(text) if use_markdown else text,
            }
            if use_markdown:
                kwargs["parse_mode"] = ParseMode.MARKDOWN_V2
            if thread_id is not None:
                kwargs["message_thread_id"] = thread_id

            try:
                ok = await self._bot.send_message_draft(**kwargs)
                if ok:
                    # Drafts have no message_id; we report success without one
                    # so the caller knows the animation frame landed.
                    return SendResult(success=True, message_id=None)
                return SendResult(success=False, error="draft_rejected")
            except Exception as e:
                # A MarkdownV2 parse failure (BadRequest "can't parse entities")
                # is recoverable: retry once as plain text.  Any other failure
                # (chat doesn't allow drafts, transient hiccup) — or a failure
                # on the plain-text attempt — propagates to the caller, which
                # treats it as "fall back to edit-based for this response".
                if use_markdown and self._is_bad_request_error(e):
                    logger.debug(
                        "[%s] sendMessageDraft MarkdownV2 rejected, retrying "
                        "as plain text (chat=%s draft_id=%s): %s",
                        self.name, chat_id, draft_id, e,
                    )
                    continue
                logger.debug(
                    "[%s] sendMessageDraft failed (chat=%s draft_id=%s): %s",
                    self.name, chat_id, draft_id, e,
                )
                return SendResult(success=False, error=str(e))

        return SendResult(success=False, error="draft_rejected")

    async def _send_message_with_thread_fallback(self, **kwargs):
        """Send a Telegram message, retrying once without message_thread_id
        if Telegram returns 'Message thread not found'.

        Used for control-style sends (approval prompts, model picker,
        update prompts) that can carry a stale thread_id from a DM
        reply chain.  The streaming send loop has its own equivalent
        (PR #3390) at the body of ``send``; this helper applies the
        same retry pattern to the non-streaming control paths.
        """
        if not self._bot:
            raise RuntimeError("Not connected")

        message_thread_id = kwargs.get("message_thread_id")
        try:
            return await self._bot.send_message(**kwargs)
        except Exception as send_err:
            if (
                message_thread_id is not None
                and self._is_bad_request_error(send_err)
                and self._is_thread_not_found_error(send_err)
            ):
                logger.warning(
                    "[%s] Thread %s not found for control message, retrying without message_thread_id",
                    self.name,
                    message_thread_id,
                )
                # Same prune as the streaming send path — the
                # control-message retry tells us the topic is gone,
                # so the binding row in state.db must go too
                # (#31501).
                self._prune_stale_dm_topic_binding(
                    kwargs.get("chat_id"), message_thread_id,
                )
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("message_thread_id", None)
                return await self._bot.send_message(**retry_kwargs)
            raise

    async def send_update_prompt(
        self, chat_id: str, prompt: str, default: str = "",
        session_key: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an inline-keyboard update prompt (Yes / No buttons).

        Used by the gateway ``/update`` watcher when ``hermes update --gateway``
        needs user input (stash restore, config migration).
        """
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            default_hint = f" (default: {default})" if default else ""
            text = self.format_message(f"⚕ *Update needs your input:*\n\n{prompt}{default_hint}")
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✓ Yes", callback_data="update_prompt:y"),
                    InlineKeyboardButton("✗ No", callback_data="update_prompt:n"),
                ]
            ])
            thread_id = self._metadata_thread_id(metadata)
            reply_to_id = self._reply_to_message_id_for_send(None, metadata, reply_to_mode=self._reply_to_mode)
            msg = await self._send_message_with_thread_fallback(
                chat_id=normalize_telegram_chat_id(chat_id),
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard,
                reply_to_message_id=reply_to_id,
                **self._thread_kwargs_for_send(
                    chat_id,
                    thread_id,
                    metadata,
                    reply_to_message_id=reply_to_id,
                    reply_to_mode=self._reply_to_mode
                ),
                **self._link_preview_kwargs(),
            )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] send_update_prompt failed: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
        allow_permanent: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        """Send an inline-keyboard approval prompt with interactive buttons.

        The buttons call ``resolve_gateway_approval()`` to unblock the waiting
        agent thread — same mechanism as the text ``/approve`` flow.
        """
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        try:
            cmd_preview = command[:3800] + "..." if len(command) > 3800 else command
            text = (
                f"⚠️ <b>Command Approval Required</b>\n\n"
                f"<pre>{_html.escape(cmd_preview)}</pre>\n\n"
                f"Reason: {_html.escape(description)}"
            )
            if smart_denied:
                text += "\n\n<b>Smart DENY:</b> owner override applies to this one operation only."

            # Resolve thread context for thread replies
            thread_id = self._metadata_thread_id(metadata)

            # We'll use the message_id as part of callback_data to look up session_key
            # Send a placeholder first, then update — or use a counter.
            # Simpler: use a monotonic counter to generate short IDs.
            import itertools
            if not hasattr(self, "_approval_counter"):
                self._approval_counter = itertools.count(1)
            approval_id = next(self._approval_counter)

            buttons = [
                InlineKeyboardButton("✅ Allow Once", callback_data=f"ea:once:{approval_id}")
            ]
            if not smart_denied:
                buttons.append(
                    InlineKeyboardButton("✅ Session", callback_data=f"ea:session:{approval_id}")
                )
                if allow_permanent:
                    buttons.append(
                        InlineKeyboardButton("✅ Always", callback_data=f"ea:always:{approval_id}")
                    )
            buttons.append(InlineKeyboardButton("❌ Deny", callback_data=f"ea:deny:{approval_id}"))
            keyboard = InlineKeyboardMarkup([buttons])

            kwargs: Dict[str, Any] = {
                "chat_id": normalize_telegram_chat_id(chat_id),
                "text": text,
                "parse_mode": ParseMode.HTML,
                "reply_markup": keyboard,
                **self._link_preview_kwargs(),
            }
            reply_to_id = self._reply_to_message_id_for_send(None, metadata, reply_to_mode=self._reply_to_mode)
            kwargs["reply_to_message_id"] = reply_to_id
            kwargs.update(
                self._thread_kwargs_for_send(
                    chat_id,
                    thread_id,
                    metadata,
                    reply_to_message_id=reply_to_id,
                    reply_to_mode=self._reply_to_mode
                )
            )

            msg = await self._send_message_with_thread_fallback(**kwargs)

            # Store session_key keyed by approval_id for the callback handler
            self._approval_state[approval_id] = session_key

            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] send_exec_approval failed: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_slash_confirm(
        self, chat_id: str, title: str, message: str, session_key: str,
        confirm_id: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Render a three-button slash-command confirmation prompt."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        try:
            preview = self.format_message(message if len(message) <= 3800 else message[:3800] + "...")

            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Approve Once", callback_data=f"sc:once:{confirm_id}"),
                    InlineKeyboardButton("🔒 Always Approve", callback_data=f"sc:always:{confirm_id}"),
                ],
                [
                    InlineKeyboardButton("❌ Cancel", callback_data=f"sc:cancel:{confirm_id}"),
                ],
            ])

            thread_id = self._metadata_thread_id(metadata)
            kwargs: Dict[str, Any] = {
                "chat_id": normalize_telegram_chat_id(chat_id),
                "text": preview,
                "parse_mode": ParseMode.MARKDOWN_V2,
                "reply_markup": keyboard,
                **self._link_preview_kwargs(),
            }
            reply_to_id = self._reply_to_message_id_for_send(None, metadata, reply_to_mode=self._reply_to_mode)
            kwargs["reply_to_message_id"] = reply_to_id
            kwargs.update(
                self._thread_kwargs_for_send(
                    chat_id,
                    thread_id,
                    metadata,
                    reply_to_message_id=reply_to_id,
                    reply_to_mode=self._reply_to_mode
                )
            )

            msg = await self._send_message_with_thread_fallback(**kwargs)
            self._slash_confirm_state[confirm_id] = session_key
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] send_slash_confirm failed: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Render a clarify prompt with one inline button per choice.

        Multi-choice mode (``choices`` non-empty): renders one button per
        option plus a final "✏️ Other (type answer)" button.  Picking the
        "Other" button flips the entry into text-capture mode so the next
        message becomes the response.

        Open-ended mode (``choices`` empty): renders the question as plain
        text — no buttons.  The next message in the session is captured by
        the gateway's text-intercept and resolves the clarify.
        """
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        try:
            text = f"❓ {_html.escape(question)}"
            thread_id = self._metadata_thread_id(metadata)

            if choices:
                # Render full option text in the message body so mobile
                # users can read long choices that would be truncated in
                # inline button labels.  Buttons keep short numeric labels
                # (1, 2, …, Other) to avoid Telegram truncation.
                option_lines = "\n".join(
                    f"{i + 1}. {_html.escape(str(c))}"
                    for i, c in enumerate(choices)
                )
                text += f"\n\n{option_lines}"

            kwargs: Dict[str, Any] = {
                "chat_id": normalize_telegram_chat_id(chat_id),
                "text": text,
                "parse_mode": ParseMode.HTML,
                **self._link_preview_kwargs(),
            }

            if choices:
                # Telegram caps callback_data at 64 bytes; keep "cl:<id>:<idx>"
                # short.
                rows = []
                for idx in range(len(choices)):
                    rows.append([
                        InlineKeyboardButton(
                            str(idx + 1),
                            callback_data=f"cl:{clarify_id}:{idx}",
                        )
                    ])
                rows.append([
                    InlineKeyboardButton(
                        "✏️ Other (type answer)",
                        callback_data=f"cl:{clarify_id}:other",
                    )
                ])
                kwargs["reply_markup"] = InlineKeyboardMarkup(rows)

            reply_to_id = self._reply_to_message_id_for_send(None, metadata)
            kwargs["reply_to_message_id"] = reply_to_id
            kwargs.update(
                self._thread_kwargs_for_send(
                    chat_id,
                    thread_id,
                    metadata,
                    reply_to_message_id=reply_to_id,
                )
            )

            msg = await self._send_message_with_thread_fallback(**kwargs)
            self._clarify_state[clarify_id] = session_key
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] send_clarify failed: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def send_model_picker(
        self,
        chat_id: str,
        providers: list,
        current_model: str,
        current_provider: str,
        session_key: str,
        on_model_selected,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an interactive inline-keyboard model picker.

        Two-step drill-down: provider selection → model selection.
        Edits the same message in-place as the user navigates.
        """
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        try:
            from hermes_cli.providers import get_label
        except ImportError:
            def get_label(slug):
                return slug

        try:
            # Build provider buttons — folds provider groups (display only).
            keyboard, provider_page_info = self._build_provider_keyboard(providers, 0)

            provider_label = get_label(current_provider)
            text = self.format_message(
                (
                    f"⚙ *Model Configuration*\n\n"
                    f"Current model: `{current_model or 'unknown'}`\n"
                    f"Provider: {provider_label}\n\n"
                    f"Select a provider:{provider_page_info}"
                )
            )

            thread_id = metadata.get("thread_id") if metadata else None
            reply_to_id = self._reply_to_message_id_for_send(None, metadata, reply_to_mode=self._reply_to_mode)
            msg = await self._send_message_with_thread_fallback(
                chat_id=normalize_telegram_chat_id(chat_id),
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard,
                reply_to_message_id=reply_to_id,
                **self._thread_kwargs_for_send(
                    chat_id,
                    thread_id,
                    metadata,
                    reply_to_message_id=reply_to_id,
                    reply_to_mode=self._reply_to_mode
                ),
                **self._link_preview_kwargs(),
            )

            # Store picker state keyed by chat_id
            self._model_picker_state[str(chat_id)] = {
                "msg_id": msg.message_id,
                "providers": providers,
                "session_key": session_key,
                "on_model_selected": on_model_selected,
                "current_model": current_model,
                "current_provider": current_provider,
                "provider_page": 0,
            }

            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] send_model_picker failed: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    _PROVIDER_PAGE_SIZE = 10
    _MODEL_PAGE_SIZE = 8

    def _build_provider_keyboard(self, providers: list, page: int = 0) -> tuple:
        """Build the paginated top-level provider keyboard, folding groups.

        Provider families (Kimi/Moonshot, MiniMax, xAI Grok, ...) collapse to
        a single ``mpg:<gid>`` button; tapping it drills into a member
        sub-keyboard. Single providers (and groups with only one authenticated
        member) render as direct ``mp:<slug>`` buttons. Grouping mirrors the
        CLI ``hermes model`` picker via the shared ``group_providers`` fold,
        so all surfaces stay consistent.
        """
        try:
            from hermes_cli.models import group_providers
        except Exception:
            group_providers = None

        by_slug = {p.get("slug"): p for p in providers}

        def _provider_button(p):
            count = p.get("total_models", len(p.get("models", [])))
            label = f"{p['name']} ({count})"
            if p.get("is_current"):
                label = f"✓ {label}"
            return InlineKeyboardButton(label, callback_data=f"mp:{p['slug']}")

        buttons: list = []
        if group_providers is not None:
            for row in group_providers([p.get("slug") for p in providers]):
                if row["kind"] == "group":
                    members = [by_slug[m] for m in row["members"] if m in by_slug]
                    count = sum(
                        m.get("total_models", len(m.get("models", []))) for m in members
                    )
                    label = f"{row['label']} ▸ ({count})"
                    if any(m.get("is_current") for m in members):
                        label = f"✓ {label}"
                    buttons.append(
                        InlineKeyboardButton(label, callback_data=f"mpg:{row['group_id']}")
                    )
                else:
                    p = by_slug.get(row["slug"])
                    if p is not None:
                        buttons.append(_provider_button(p))
        else:
            for p in providers:
                buttons.append(_provider_button(p))

        page_size = self._PROVIDER_PAGE_SIZE
        total = len(buttons)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(0, min(page, total_pages - 1))

        start = page * page_size
        end = min(start + page_size, total)
        page_buttons = buttons[start:end]

        rows = [page_buttons[i : i + 2] for i in range(0, len(page_buttons), 2)]

        if total_pages > 1:
            nav: list = []
            if page > 0:
                nav.append(InlineKeyboardButton("◀ Prev", callback_data=f"mpv:{page - 1}"))
            nav.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="mx:noop"))
            if page < total_pages - 1:
                nav.append(InlineKeyboardButton("Next ▶", callback_data=f"mpv:{page + 1}"))
            rows.append(nav)

        rows.append([InlineKeyboardButton("✗ Cancel", callback_data="mx")])

        page_info = f" ({start + 1}–{end} of {total})" if total_pages > 1 else ""
        return InlineKeyboardMarkup(rows), page_info

    def _build_model_keyboard(self, models: list, page: int) -> tuple:
        """Build paginated model buttons. Returns (keyboard, page_info_text)."""
        page_size = self._MODEL_PAGE_SIZE
        total = len(models)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(0, min(page, total_pages - 1))

        start = page * page_size
        end = min(start + page_size, total)
        page_models = models[start:end]

        buttons: list = []
        for i, model_id in enumerate(page_models):
            abs_idx = start + i
            short = model_id.split("/")[-1] if "/" in model_id else model_id
            if len(short) > 38:
                short = short[:35] + "..."
            buttons.append(
                InlineKeyboardButton(short, callback_data=f"mm:{abs_idx}")
            )

        rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]

        # Pagination row (if needed)
        if total_pages > 1:
            nav: list = []
            if page > 0:
                nav.append(InlineKeyboardButton("◀ Prev", callback_data=f"mg:{page - 1}"))
            nav.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="mx:noop"))
            if page < total_pages - 1:
                nav.append(InlineKeyboardButton("Next ▶", callback_data=f"mg:{page + 1}"))
            rows.append(nav)

        rows.append([
            InlineKeyboardButton("◀ Back", callback_data="mb"),
            InlineKeyboardButton("✗ Cancel", callback_data="mx"),
        ])

        page_info = f" ({start + 1}–{end} of {total})" if total_pages > 1 else ""
        return InlineKeyboardMarkup(rows), page_info

    async def _handle_model_picker_callback(
        self, query, data: str, chat_id: str
    ) -> None:
        """Handle model picker inline keyboard callbacks (mp:/mm:/mc:/mb:/mx:/mg:)."""
        state = self._model_picker_state.get(chat_id)
        if not state:
            await query.answer(text="Picker expired — use /model again.")
            return

        try:
            from hermes_cli.providers import get_label
        except ImportError:
            def get_label(slug):
                return slug

        if data.startswith("mp:"):
            # --- Provider selected: show model buttons (page 0) ---
            provider_slug = data[3:]
            provider = next(
                (p for p in state["providers"] if p["slug"] == provider_slug),
                None,
            )
            if not provider:
                await query.answer(text="Provider not found.")
                return

            models = provider.get("models", [])
            state["selected_provider"] = provider_slug
            state["selected_provider_name"] = provider.get("name", provider_slug)
            state["model_list"] = models
            state["model_page"] = 0

            keyboard, page_info = self._build_model_keyboard(models, 0)

            pname = provider.get("name", provider_slug)
            total = provider.get("total_models", len(models))
            shown = len(models)
            extra = f"\n_{total - shown} more available — type `/model <name>` directly_" if total > shown else ""

            await query.edit_message_text(
                text=self.format_message(
                    (
                        f"⚙ *Model Configuration*\n\n"
                        f"Provider: *{pname}*{page_info}\n"
                        f"Select a model:{extra}"
                    )
                ),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard,
            )
            await query.answer()

        elif data.startswith("mg:"):
            # --- Page navigation ---
            try:
                page = int(data[3:])
            except ValueError:
                await query.answer(text="Invalid page.")
                return

            models = state.get("model_list", [])
            state["model_page"] = page

            keyboard, page_info = self._build_model_keyboard(models, page)

            pname = state.get("selected_provider_name", "")
            provider_slug = state.get("selected_provider", "")
            provider = next(
                (p for p in state["providers"] if p["slug"] == provider_slug),
                None,
            )
            total = provider.get("total_models", len(models)) if provider else len(models)
            shown = len(models)
            extra = f"\n_{total - shown} more available — type `/model <name>` directly_" if total > shown else ""

            await query.edit_message_text(
                text=self.format_message(
                    (
                        f"⚙ *Model Configuration*\n\n"
                        f"Provider: *{pname}*{page_info}\n"
                        f"Select a model:{extra}"
                    )
                ),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard,
            )
            await query.answer()

        elif data.startswith("mpv:"):
            # --- Provider page navigation ---
            try:
                page = int(data[4:])
            except ValueError:
                await query.answer(text="Invalid page.")
                return

            state["provider_page"] = page
            keyboard, provider_page_info = self._build_provider_keyboard(
                state["providers"], page
            )

            try:
                provider_label = get_label(state["current_provider"])
            except Exception:
                provider_label = state["current_provider"]

            await query.edit_message_text(
                text=self.format_message(
                    (
                        f"⚙ *Model Configuration*\n\n"
                        f"Current model: `{state['current_model'] or 'unknown'}`\n"
                        f"Provider: {provider_label}\n\n"
                        f"Select a provider:{provider_page_info}"
                    )
                ),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard,
            )
            await query.answer()

        elif data.startswith("mc:"):
            # --- Expensive model confirmed: perform the switch ---
            try:
                idx = int(data[3:])
            except ValueError:
                await query.answer(text="Invalid selection.")
                return

            model_list = state.get("model_list", [])
            if idx < 0 or idx >= len(model_list):
                await query.answer(text="Invalid model index.")
                return

            model_id = model_list[idx]
            provider_slug = state.get("selected_provider", "")
            callback = state.get("on_model_selected")

            if not callback:
                await query.answer(text="Picker expired.")
                return

            switch_failed = False
            try:
                result_text = await callback(chat_id, model_id, provider_slug)
            except Exception as exc:
                logger.error("Model picker switch failed: %s", exc)
                result_text = f"Error switching model: {exc}"
                switch_failed = True

            try:
                await query.edit_message_text(
                    text=self.format_message(result_text),
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=None,
                )
            except Exception:
                try:
                    await query.edit_message_text(
                        text=result_text,
                        parse_mode=None,
                        reply_markup=None,
                    )
                except Exception:
                    pass
            await query.answer(
                text="Switch failed." if switch_failed else "Model switched!"
            )
            self._model_picker_state.pop(chat_id, None)

        elif data.startswith("mm:"):
            # --- Model selected: perform the switch ---
            try:
                idx = int(data[3:])
            except ValueError:
                await query.answer(text="Invalid selection.")
                return

            model_list = state.get("model_list", [])
            if idx < 0 or idx >= len(model_list):
                await query.answer(text="Invalid model index.")
                return

            model_id = model_list[idx]
            provider_slug = state.get("selected_provider", "")
            callback = state.get("on_model_selected")

            if not callback:
                await query.answer(text="Picker expired.")
                return

            try:
                from hermes_cli.model_cost_guard import expensive_model_warning

                # Pricing lookup can hit models.dev / a /models endpoint on a
                # cache miss — keep it off the event loop.
                warning = await asyncio.to_thread(
                    expensive_model_warning,
                    model_id,
                    provider=provider_slug,
                )
            except Exception:
                warning = None
            if warning is not None:
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("Switch anyway", callback_data=f"mc:{idx}")],
                    [
                        InlineKeyboardButton("◀ Back", callback_data="mb"),
                        InlineKeyboardButton("✗ Cancel", callback_data="mx"),
                    ],
                ])
                await query.edit_message_text(
                    text=self.format_message(
                        f"⚠ *Expensive Model Warning*\n\n{warning.message}"
                    ),
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=keyboard,
                )
                await query.answer(text="Confirm expensive model")
                return

            switch_failed = False
            try:
                result_text = await callback(chat_id, model_id, provider_slug)
            except Exception as exc:
                logger.error("Model picker switch failed: %s", exc)
                result_text = f"Error switching model: {exc}"
                switch_failed = True

            # Edit message to show confirmation, remove buttons
            try:
                await query.edit_message_text(
                    text=self.format_message(result_text),
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=None,
                )
            except Exception:
                # Markdown parse failure — retry as plain text
                try:
                    await query.edit_message_text(
                        text=result_text,
                        parse_mode=None,
                        reply_markup=None,
                    )
                except Exception:
                    pass
            await query.answer(
                text="Switch failed." if switch_failed else "Model switched!"
            )

            # Clean up state
            self._model_picker_state.pop(chat_id, None)

        elif data.startswith("mpg:"):
            # --- Provider group selected: show member providers ---
            group_id = data[4:]
            try:
                from hermes_cli.models import PROVIDER_GROUPS
                _label, _desc, member_slugs = PROVIDER_GROUPS.get(group_id, ("", "", []))
            except Exception:
                _label, member_slugs = "", []

            by_slug = {p["slug"]: p for p in state["providers"]}
            members = [by_slug[m] for m in member_slugs if m in by_slug]
            if not members:
                await query.answer(text="Group not found.")
                return

            buttons = []
            for p in members:
                count = p.get("total_models", len(p.get("models", [])))
                label = f"{p['name']} ({count})"
                if p.get("is_current"):
                    label = f"✓ {label}"
                buttons.append(
                    InlineKeyboardButton(label, callback_data=f"mp:{p['slug']}")
                )
            rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
            rows.append([
                InlineKeyboardButton("◀ Back", callback_data="mb"),
                InlineKeyboardButton("✗ Cancel", callback_data="mx"),
            ])
            keyboard = InlineKeyboardMarkup(rows)

            await query.edit_message_text(
                text=self.format_message(
                    (
                        f"⚙ *Model Configuration*\n\n"
                        f"Provider family: *{_label or group_id}*\n\n"
                        f"Select a provider:"
                    )
                ),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard,
            )
            await query.answer()

        elif data == "mb":
            # --- Back to provider list (folds groups) ---
            page = int(state.get("provider_page", 0) or 0)
            keyboard, provider_page_info = self._build_provider_keyboard(
                state["providers"], page
            )

            try:
                provider_label = get_label(state["current_provider"])
            except Exception:
                provider_label = state["current_provider"]

            await query.edit_message_text(
                text=self.format_message(
                    (
                        f"⚙ *Model Configuration*\n\n"
                        f"Current model: `{state['current_model'] or 'unknown'}`\n"
                        f"Provider: {provider_label}\n\n"
                        f"Select a provider:{provider_page_info}"
                    )
                ),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard,
            )
            await query.answer()

        elif data == "mx":
            # --- Cancel ---
            self._model_picker_state.pop(chat_id, None)
            await query.edit_message_text(
                text="Model selection cancelled.",
                reply_markup=None,
            )
            await query.answer()

        else:
            # Catch-all (e.g. page counter button "mx:noop")
            await query.answer()

    async def _notify_clarify_expired(self, query, user_display: str) -> None:
        """Tell the user a clarify tap arrived too late to be delivered.

        Fires when the clarify entry was evicted by ``clarify_timeout`` or the
        gateway restarted between asking and the tap. In both cases the agent
        thread is no longer waiting, so the tap would otherwise leave a
        misleading ✓ (or an "awaiting typed response" prompt) on a button the
        agent never receives.
        """
        try:
            await query.answer(text="⚠️ This prompt expired — please /retry.")
        except Exception:
            pass
        try:
            await query.edit_message_text(
                text=(
                    f"❓ {_html.escape(query.message.text or '')}\n\n"
                    "<i>⚠️ This question expired or the session reset — please /retry.</i>"
                ),
                parse_mode=ParseMode.HTML,
                reply_markup=None,
            )
        except Exception:
            pass

    async def _handle_callback_query(
        self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"
    ) -> None:
        """Handle inline keyboard button clicks."""
        query = update.callback_query
        if not query or not query.data:
            return
        data = query.data
        query_message = getattr(query, "message", None)
        query_chat_id = getattr(query_message, "chat_id", None)
        query_chat = getattr(query_message, "chat", None)
        query_chat_type = getattr(query_chat, "type", None)
        query_thread_id = getattr(query_message, "message_thread_id", None)
        query_user_name = getattr(query.from_user, "first_name", None)

        # --- Model picker callbacks ---
        if data.startswith(("mp:", "mpg:", "mpv:", "mm:", "mc:", "mb", "mx", "mg:")):
            chat_id = str(query.message.chat_id) if query.message else None
            if chat_id:
                await self._handle_model_picker_callback(query, data, chat_id)
            return

        # --- Gmail-triage callbacks (gt:verb:arg) ---
        if data.startswith("gt:"):
            await self._handle_gmail_triage_callback(
                query,
                data,
                query_chat_id=query_chat_id,
                query_chat_type=query_chat_type,
                query_thread_id=query_thread_id,
                query_user_name=query_user_name,
            )
            return

        # --- Exec approval callbacks (ea:choice:id) ---
        if data.startswith("ea:"):
            parts = data.split(":", 2)
            if len(parts) == 3:
                choice = parts[1]  # once, session, always, deny
                try:
                    approval_id = int(parts[2])
                except (ValueError, IndexError):
                    await query.answer(text="Invalid approval data.")
                    return

                # Only authorized users may click approval buttons.
                caller_id = str(getattr(query.from_user, "id", ""))
                if not self._is_callback_user_authorized(
                    caller_id,
                    chat_id=query_chat_id,
                    chat_type=str(query_chat_type) if query_chat_type is not None else None,
                    thread_id=str(query_thread_id) if query_thread_id is not None else None,
                    user_name=query_user_name,
                ):
                    await query.answer(text="⛔ You are not authorized to approve commands.")
                    return

                session_key = self._approval_state.pop(approval_id, None)
                if not session_key:
                    await query.answer(text="This approval has already been resolved.")
                    return

                # Map choice to human-readable label
                label_map = {
                    "once": "✅ Approved once",
                    "session": "✅ Approved for session",
                    "always": "✅ Approved permanently",
                    "deny": "❌ Denied",
                }
                user_display = getattr(query.from_user, "first_name", "User")
                label = label_map.get(choice, "Resolved")

                await query.answer(text=label)

                # Edit message to show decision, remove buttons
                try:
                    await query.edit_message_text(
                        text=self.format_message(f"{label} by {user_display}"),
                        parse_mode=ParseMode.MARKDOWN_V2,
                        reply_markup=None,
                    )
                except Exception:
                    pass  # non-fatal if edit fails

                # Resolve the approval — unblocks the agent thread
                try:
                    from tools.approval import resolve_gateway_approval
                    count = resolve_gateway_approval(session_key, choice)
                    logger.info(
                        "Telegram button resolved %d approval(s) for session %s (choice=%s, user=%s)",
                        count, session_key, choice, user_display,
                    )
                except Exception as exc:
                    logger.error("Failed to resolve gateway approval from Telegram button: %s", exc)
                    count = 0

                # Resume the typing indicator — paused when the approval was
                # sent (gateway/run.py).  The text /approve and /deny paths
                # call resume_typing_for_chat here too; without it, typing
                # stays paused for the rest of the turn after an inline
                # button click.
                if count and query_chat_id is not None:
                    self.resume_typing_for_chat(str(query_chat_id))
            return

        # --- Slash-confirm callbacks (sc:choice:confirm_id) ---
        if data.startswith("sc:"):
            parts = data.split(":", 2)
            if len(parts) == 3:
                choice = parts[1]  # once, always, cancel
                confirm_id = parts[2]

                caller_id = str(getattr(query.from_user, "id", ""))
                if not self._is_callback_user_authorized(
                    caller_id,
                    chat_id=query_chat_id,
                    chat_type=str(query_chat_type) if query_chat_type is not None else None,
                    thread_id=str(query_thread_id) if query_thread_id is not None else None,
                    user_name=query_user_name,
                ):
                    await query.answer(text="⛔ You are not authorized to answer this prompt.")
                    return

                session_key = self._slash_confirm_state.pop(confirm_id, None)
                if not session_key:
                    await query.answer(text="This prompt has already been resolved.")
                    return

                label_map = {
                    "once": "✅ Approved once",
                    "always": "🔒 Always approve",
                    "cancel": "❌ Cancelled",
                }
                user_display = getattr(query.from_user, "first_name", "User")
                label = label_map.get(choice, "Resolved")

                await query.answer(text=label)

                try:
                    await query.edit_message_text(
                        text=self.format_message(f"{label} by {user_display}"),
                        parse_mode=ParseMode.MARKDOWN_V2,
                        reply_markup=None,
                    )
                except Exception:
                    pass

                # Resolve via the module-level primitive.  The runner stored
                # a handler keyed by session_key; we run it on the event
                # loop and (if it returns a string) send it as a follow-up
                # message in the same chat.
                try:
                    from tools import slash_confirm as _slash_confirm_mod
                    result_text = await _slash_confirm_mod.resolve(
                        session_key, confirm_id, choice,
                    )
                    if result_text and query.message:
                        # Inherit the prompt message's topic. Supergroup forums
                        # use message_thread_id; Telegram private DM-topic lanes
                        # need both the private topic id and the prompt reply anchor.
                        thread_id = getattr(query.message, "message_thread_id", None)
                        chat = getattr(query.message, "chat", None)
                        chat_type = getattr(chat, "type", None)
                        prompt_message_id = getattr(query.message, "message_id", None)
                        send_kwargs: Dict[str, Any] = {
                            "chat_id": int(query.message.chat_id),
                            "text": self.format_message(result_text),
                            "parse_mode": ParseMode.MARKDOWN_V2,
                            **self._link_preview_kwargs(),
                        }
                        chat_type_value = getattr(chat_type, "value", chat_type)
                        is_private_chat = str(chat_type_value).lower() in {
                            "private",
                            str(ChatType.PRIVATE).lower(),
                            str(getattr(ChatType.PRIVATE, "value", ChatType.PRIVATE)).lower(),
                        }
                        if thread_id is not None and is_private_chat and prompt_message_id is not None:
                            reply_to_id = int(prompt_message_id)
                            send_kwargs["reply_to_message_id"] = reply_to_id
                            send_kwargs.update(
                                self._thread_kwargs_for_send(
                                    str(query.message.chat_id),
                                    str(thread_id),
                                    {
                                        "thread_id": str(thread_id),
                                        "telegram_dm_topic_reply_fallback": True,
                                    },
                                    reply_to_message_id=reply_to_id,
                                    reply_to_mode=self._reply_to_mode
                                )
                            )
                        elif thread_id is not None:
                            send_kwargs.update(
                                self._thread_kwargs_for_send(
                                    str(query.message.chat_id),
                                    str(thread_id),
                                    {"thread_id": str(thread_id)},
                                    reply_to_mode=self._reply_to_mode
                                )
                            )
                        await self._send_message_with_thread_fallback(**send_kwargs)
                except Exception as exc:
                    logger.error("[%s] slash-confirm callback failed: %s", self.name, exc, exc_info=True)
            return

        # --- Clarify callbacks (cl:clarify_id:idx | cl:clarify_id:other) ---
        if data.startswith("cl:"):
            parts = data.split(":", 2)
            if len(parts) == 3:
                clarify_id = parts[1]
                choice_token = parts[2]

                caller_id = str(getattr(query.from_user, "id", ""))
                if not self._is_callback_user_authorized(
                    caller_id,
                    chat_id=query_chat_id,
                    chat_type=str(query_chat_type) if query_chat_type is not None else None,
                    thread_id=str(query_thread_id) if query_thread_id is not None else None,
                    user_name=query_user_name,
                ):
                    await query.answer(text="⛔ You are not authorized to answer this prompt.")
                    return

                session_key = self._clarify_state.get(clarify_id)
                if not session_key:
                    await query.answer(text="This prompt has already been resolved.")
                    return

                user_display = getattr(query.from_user, "first_name", "User")

                if choice_token == "other":
                    # Flip into text-capture mode and tell the user to type
                    # their answer.  The gateway's text-intercept will pick
                    # up the next message in this session and resolve the
                    # clarify.  Do NOT pop _clarify_state yet — we still
                    # need it if the user is slow to respond and the entry
                    # is cleared by something else.
                    flipped = False
                    try:
                        from tools.clarify_gateway import mark_awaiting_text
                        flipped = mark_awaiting_text(clarify_id)
                    except Exception as exc:
                        logger.warning("[%s] mark_awaiting_text failed: %s", self.name, exc)

                    if not flipped:
                        # Entry evicted (clarify_timeout) or gateway restarted
                        # between ask and tap — a typed answer would go nowhere.
                        self._clarify_state.pop(clarify_id, None)
                        await self._notify_clarify_expired(query, user_display)
                        return

                    await query.answer(text="✏️ Type your answer in the chat.")
                    try:
                        await query.edit_message_text(
                            text=f"❓ {query.message.text or ''}\n\n<i>Awaiting typed response from {_html.escape(user_display)}…</i>",
                            parse_mode=ParseMode.HTML,
                            reply_markup=None,
                        )
                    except Exception:
                        pass
                    return

                # Numeric choice → resolve immediately with the chosen text
                try:
                    idx = int(choice_token)
                except (ValueError, TypeError):
                    await query.answer(text="Invalid choice.")
                    return

                # Look up the choice text from the entry registered in the
                # clarify primitive.  Fall back to the index if the entry
                # has been cleaned up (race with timeout / session reset).
                resolved_text: Optional[str] = None
                try:
                    from tools.clarify_gateway import _entries as _clarify_entries  # type: ignore
                    entry = _clarify_entries.get(clarify_id)
                    if entry and entry.choices and 0 <= idx < len(entry.choices):
                        resolved_text = entry.choices[idx]
                except Exception:
                    resolved_text = None

                if resolved_text is None:
                    # Race: entry vanished. Echo the index as a number so
                    # the agent at least sees an intentional response
                    # rather than nothing.
                    resolved_text = f"choice {idx + 1}"

                # Pop state and resolve
                self._clarify_state.pop(clarify_id, None)
                try:
                    from tools.clarify_gateway import resolve_gateway_clarify
                    resolved = resolve_gateway_clarify(clarify_id, resolved_text)
                except Exception as exc:
                    logger.error("[%s] resolve_gateway_clarify failed: %s", self.name, exc)
                    resolved = False

                if resolved:
                    await query.answer(text=f"✓ {resolved_text[:60]}")
                    try:
                        await query.edit_message_text(
                            text=f"❓ {_html.escape(query.message.text or '')}\n\n<b>{_html.escape(user_display)}:</b> {_html.escape(resolved_text)}",
                            parse_mode=ParseMode.HTML,
                            reply_markup=None,
                        )
                    except Exception:
                        pass
                    logger.info(
                        "Telegram clarify button resolved (id=%s, choice=%r, user=%s)",
                        clarify_id, resolved_text, user_display,
                    )
                else:
                    # Entry evicted (clarify_timeout) or gateway restarted
                    # between ask and tap — surface this instead of leaving a
                    # misleading ✓ on a button the agent will never receive.
                    await self._notify_clarify_expired(query, user_display)
                    logger.warning(
                        "Telegram clarify button: resolve_gateway_clarify returned False (id=%s)",
                        clarify_id,
                    )
            return

        # --- Update prompt callbacks ---
        if not data.startswith("update_prompt:"):
            return
        answer = data.split(":", 1)[1]  # "y" or "n"
        caller_id = str(getattr(query.from_user, "id", ""))
        if not self._is_callback_user_authorized(
            caller_id,
            chat_id=query_chat_id,
            chat_type=str(query_chat_type) if query_chat_type is not None else None,
            thread_id=str(query_thread_id) if query_thread_id is not None else None,
            user_name=query_user_name,
        ):
            await query.answer(text="⛔ You are not authorized to answer update prompts.")
            return
        await query.answer(text=f"Sent '{answer}' to the update process.")
        # Edit the message to show the choice and remove buttons
        label = "Yes" if answer == "y" else "No"
        try:
            await query.edit_message_text(
                text=self.format_message(f"⚕ Update prompt answered: *{label}*"),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=None,
            )
        except Exception:
            pass  # non-fatal if edit fails
        # Write the response file
        try:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
            response_path = home / ".update_response"
            tmp = response_path.with_suffix(".tmp")
            tmp.write_text(answer)
            tmp.replace(response_path)
            logger.info("Telegram update prompt answered '%s' by user %s",
                        answer, getattr(query.from_user, "id", "unknown"))
        except Exception as exc:
            logger.error("Failed to write update response from callback: %s", exc)

    # Maps `gt:<verb>` -> (script-name, extra-args, success-label, is_state).
    # Scripts live in ~/.hermes/scripts/gmail-triage/. `arg` from the callback
    # data is always passed as the first positional arg.
    # is_state=True means the verb is a sticky sender-rule change (mute, trust,
    # vip) that should leave the keyboard tappable for follow-on actions.
    # is_state=False is a per-email one-shot (send, archive, draft, spam) that
    # strips the keyboard on success.
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
        "vip-domain":   ("vip-add.sh",         ["domain"], "✓ marked VIP domain",  True),
    }

    async def _handle_gmail_triage_callback(
        self,
        query,
        data: str,
        *,
        query_chat_id,
        query_chat_type,
        query_thread_id,
        query_user_name,
    ) -> None:
        """Dispatch a gmail-triage inline-button callback (gt:verb:arg)."""
        parts = data.split(":", 2)
        if len(parts) != 3:
            await query.answer(text="Invalid gmail-triage data.")
            return
        verb, arg = parts[1], parts[2]

        caller_id = str(getattr(query.from_user, "id", ""))
        if not self._is_callback_user_authorized(
            caller_id,
            chat_id=query_chat_id,
            chat_type=str(query_chat_type) if query_chat_type is not None else None,
            thread_id=str(query_thread_id) if query_thread_id is not None else None,
            user_name=query_user_name,
        ):
            await query.answer(text="⛔ You are not authorized to act on this email.")
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

        cmd = [str(script_path), arg, *extra_args]
        success = False
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=60,
            )
            if proc.returncode == 0:
                label = success_label
                success = True
                logger.info(
                    "[%s] gmail-triage callback ok: verb=%s arg=%s",
                    self.name, verb, arg,
                )
            else:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                last_line = stderr_text.splitlines()[-1] if stderr_text else f"exit {proc.returncode}"
                label = f"❌ {verb} failed: {last_line[:80]}"
                logger.error(
                    "[%s] gmail-triage callback failed: verb=%s arg=%s rc=%s stderr=%s",
                    self.name, verb, arg, proc.returncode, stderr_text,
                )
        except asyncio.TimeoutError:
            label = f"❌ {verb} timed out"
            logger.error("[%s] gmail-triage callback timed out: verb=%s arg=%s", self.name, verb, arg)
        except Exception as exc:
            label = f"❌ {verb} error: {exc}"
            logger.error(
                "[%s] gmail-triage callback exception: verb=%s arg=%s err=%s",
                self.name, verb, arg, exc, exc_info=True,
            )

        await query.answer(text=label)
        if not success:
            return

        user_display = getattr(query.from_user, "first_name", "User")
        original_text = (query.message.text or "") if query.message else ""
        appended = f"{original_text}\n— {label} by {user_display}"
        try:
            if is_state_verb:
                # Sticky state change: append confirmation, KEEP keyboard so
                # the user can stack further actions on this email.
                await query.edit_message_text(text=appended)
            else:
                # Per-email one-shot: strip keyboard so the action can't fire twice.
                await query.edit_message_text(text=appended, reply_markup=None)
        except Exception:
            pass

    def _missing_media_path_error(self, label: str, path: str) -> str:
        """Build an actionable file-not-found error for gateway MEDIA delivery.

        Paths like /workspace/... or /output/... often only exist inside the
        Docker sandbox, while the gateway process runs on the host.
        """
        error = f"{label} file not found: {path}"
        if path.startswith(("/workspace/", "/output/", "/outputs/")):
            error += (
                " (path may only exist inside the Docker sandbox. "
                "Bind-mount a host directory and emit the host-visible "
                "path in MEDIA: for gateway file delivery.)"
            )
        return error

    def _telegram_media_too_large_note(self, label: str, file_size: Any, max_bytes: int) -> str:
        limit_mb = max(1, max_bytes // (1024 * 1024))
        try:
            size_mb = int(file_size or 0) / (1024 * 1024)
            size_text = f"{size_mb:.1f} MB"
        except (TypeError, ValueError):
            size_text = "unknown size"
        return (
            f"[Telegram {label} skipped: file size {size_text} exceeds the "
            f"{limit_mb} MB limit. Ask the user to send a smaller file.]"
        )

    def _telegram_media_size_allowed(self, source: Any, label: str) -> tuple[bool, Optional[str]]:
        """Validate Telegram media size before downloading into memory."""
        max_bytes = int(getattr(self, "_max_doc_bytes", 20 * 1024 * 1024) or 20 * 1024 * 1024)
        file_size = getattr(source, "file_size", None)
        try:
            size = int(file_size or 0)
        except (TypeError, ValueError):
            size = 0
        if size <= 0:
            return True, None
        if size <= max_bytes:
            return True, None
        return False, self._telegram_media_too_large_note(label, size, max_bytes)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send audio as a native Telegram voice message or audio file."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        
        try:
            if not os.path.exists(audio_path):
                return SendResult(success=False, error=self._missing_media_path_error("Audio", audio_path))
            
            with open(audio_path, "rb") as audio_file:
                ext = os.path.splitext(audio_path)[1].lower()
                # .ogg / .opus files -> send as voice (round playable bubble)
                if ext in {".ogg", ".opus"}:
                    _voice_thread = self._metadata_thread_id(metadata)
                    reply_to_id = self._reply_to_message_id_for_send(reply_to, metadata, reply_to_mode=self._reply_to_mode)
                    voice_thread_kwargs = self._thread_kwargs_for_send(
                        chat_id,
                        _voice_thread,
                        metadata,
                        reply_to_message_id=reply_to_id,
                        reply_to_mode=self._reply_to_mode
                    )
                    msg = await self._send_with_dm_topic_reply_anchor_retry(
                        self._bot.send_voice,
                        {
                            "chat_id": normalize_telegram_chat_id(chat_id),
                            "voice": audio_file,
                            "caption": caption[:1024] if caption else None,
                            "reply_to_message_id": reply_to_id,
                            **voice_thread_kwargs,
                            **self._notification_kwargs(metadata),
                        },
                        metadata,
                        reply_to_id,
                        "voice",
                        reset_media=lambda: audio_file.seek(0),
                    )
                elif ext in {".mp3", ".m4a"}:
                    # Telegram's Bot API sendAudio only accepts MP3 / M4A.
                    _audio_thread = self._metadata_thread_id(metadata)
                    reply_to_id = self._reply_to_message_id_for_send(reply_to, metadata, reply_to_mode=self._reply_to_mode)
                    audio_thread_kwargs = self._thread_kwargs_for_send(
                        chat_id,
                        _audio_thread,
                        metadata,
                        reply_to_message_id=reply_to_id,
                        reply_to_mode=self._reply_to_mode
                    )
                    msg = await self._send_with_dm_topic_reply_anchor_retry(
                        self._bot.send_audio,
                        {
                            "chat_id": normalize_telegram_chat_id(chat_id),
                            "audio": audio_file,
                            "caption": caption[:1024] if caption else None,
                            "reply_to_message_id": reply_to_id,
                            **audio_thread_kwargs,
                            **self._notification_kwargs(metadata),
                        },
                        metadata,
                        reply_to_id,
                        "audio",
                        reset_media=lambda: audio_file.seek(0),
                    )
                else:
                    # Formats Telegram can't play natively (.wav, .flac, ...)
                    # — fall back to document delivery instead of raising.
                    return await self.send_document(
                        chat_id=chat_id,
                        file_path=audio_path,
                        caption=caption,
                        reply_to=reply_to,
                        metadata=metadata,
                    )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.error(
                "[%s] Failed to send Telegram voice/audio, falling back to base adapter: %s",
                self.name,
                e,
                exc_info=True,
            )
            return await super().send_voice(chat_id, audio_path, caption, reply_to, metadata=metadata)

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[tuple],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send a batch of images natively via Telegram's media group API.

        Telegram's ``send_media_group`` bundles up to 10 photos/videos into
        a single album. Larger batches are chunked. Animated GIFs cannot
        go into a media group (they require ``send_animation``), so they
        are peeled off and sent individually via the base default path.

        URL-based photos go into the group directly; local files are
        opened as byte streams. On failure the whole batch falls back to
        the base adapter's per-image loop.
        """
        if not self._bot:
            return
        if not images:
            return

        try:
            from telegram import InputMediaPhoto
        except Exception as exc:  # pragma: no cover - missing SDK
            logger.warning(
                "[%s] InputMediaPhoto unavailable, falling back to per-image send: %s",
                self.name, exc,
            )
            await super().send_multiple_images(chat_id, images, metadata, human_delay)
            return

        # Peel off animations — they need send_animation, not send_media_group
        animations: List[tuple] = []
        photos: List[tuple] = []
        for image_url, alt_text in images:
            if not image_url.startswith("file://") and self._is_animation_url(image_url):
                animations.append((image_url, alt_text))
            else:
                photos.append((image_url, alt_text))

        # Animations: route through the base default (per-image send_animation)
        if animations:
            await super().send_multiple_images(
                chat_id, animations, metadata, human_delay=human_delay,
            )

        if not photos:
            return

        from urllib.parse import unquote as _unquote
        _thread = self._metadata_thread_id(metadata)

        # Chunk into groups of 10 (Telegram's album limit)
        CHUNK = 10
        chunks = [photos[i:i + CHUNK] for i in range(0, len(photos), CHUNK)]

        for chunk_idx, chunk in enumerate(chunks):
            if human_delay > 0 and chunk_idx > 0:
                await asyncio.sleep(human_delay)

            media: List[Any] = []
            opened_files: List[Any] = []
            try:
                for image_url, alt_text in chunk:
                    caption = alt_text[:1024] if alt_text else None
                    if image_url.startswith("file://"):
                        local_path = _unquote(image_url[7:])
                        if not os.path.exists(local_path):
                            logger.warning(
                                "[%s] Skipping missing image in media group: %s",
                                self.name, local_path,
                            )
                            continue
                        fh = open(local_path, "rb")
                        opened_files.append(fh)
                        media.append(InputMediaPhoto(media=fh, caption=caption))
                    else:
                        media.append(InputMediaPhoto(media=image_url, caption=caption))

                if not media:
                    continue

                logger.info(
                    "[%s] Sending media group of %d photo(s) (chunk %d/%d)",
                    self.name, len(media), chunk_idx + 1, len(chunks),
                )
                reply_to_id = self._reply_to_message_id_for_send(None, metadata, reply_to_mode=self._reply_to_mode)
                thread_kwargs = self._thread_kwargs_for_send(
                    chat_id,
                    _thread,
                    metadata,
                    reply_to_message_id=reply_to_id,
                    reply_to_mode=self._reply_to_mode
                )

                def _reset_opened_files() -> None:
                    for fh in opened_files:
                        try:
                            fh.seek(0)
                        except Exception:
                            pass

                await self._send_with_dm_topic_reply_anchor_retry(
                    self._bot.send_media_group,
                    {
                        "chat_id": normalize_telegram_chat_id(chat_id),
                        "media": media,
                        "reply_to_message_id": reply_to_id,
                        **thread_kwargs,
                        **self._notification_kwargs(metadata),
                    },
                    metadata,
                    reply_to_id,
                    "media group",
                    reset_media=_reset_opened_files,
                )
            except Exception as e:
                logger.warning(
                    "[%s] send_media_group failed (chunk %d/%d), falling back to per-image: %s",
                    self.name, chunk_idx + 1, len(chunks), e,
                    exc_info=True,
                )
                # Fallback: send each photo in this chunk individually
                await super().send_multiple_images(
                    chat_id, chunk, metadata, human_delay=human_delay,
                )
            finally:
                for fh in opened_files:
                    try:
                        fh.close()
                    except Exception:
                        pass

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file natively as a Telegram photo."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        try:
            if not os.path.exists(image_path):
                return SendResult(success=False, error=self._missing_media_path_error("Image", image_path))

            _thread = self._metadata_thread_id(metadata)
            reply_to_id = self._reply_to_message_id_for_send(reply_to, metadata, reply_to_mode=self._reply_to_mode)
            thread_kwargs = self._thread_kwargs_for_send(
                chat_id,
                _thread,
                metadata,
                reply_to_message_id=reply_to_id,
                reply_to_mode=self._reply_to_mode
            )
            with open(image_path, "rb") as image_file:
                msg = await self._send_with_dm_topic_reply_anchor_retry(
                    self._bot.send_photo,
                    {
                        "chat_id": normalize_telegram_chat_id(chat_id),
                        "photo": image_file,
                        "caption": caption[:1024] if caption else None,
                        "reply_to_message_id": reply_to_id,
                        **thread_kwargs,
                        **self._notification_kwargs(metadata),
                    },
                    metadata,
                    reply_to_id,
                    "photo",
                    reset_media=lambda: image_file.seek(0),
                )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            error_str = str(e)
            # Dimension-related errors are the expected case for valid image
            # files that Telegram just refuses as photos (screenshots, extreme
            # aspect ratios). Log at INFO because the document fallback is
            # the correct path. Any other send_photo failure also falls back
            # to document (rate limits, corrupt file markers, format edge
            # cases), but at WARNING because it's unexpected and worth
            # surfacing in logs.
            is_dim_error = (
                "Photo_invalid_dimensions" in error_str
                or "PHOTO_INVALID_DIMENSIONS" in error_str
            )
            if is_dim_error:
                logger.info(
                    "[%s] Image dimensions exceed Telegram photo limits, "
                    "sending as document: %s",
                    self.name,
                    image_path,
                )
            else:
                logger.warning(
                    "[%s] Failed to send Telegram local image as photo, "
                    "trying document fallback: %s",
                    self.name,
                    e,
                    exc_info=True,
                )
            # Fallback to sending as document (file) — no dimension limit,
            # only 50MB size limit. If even that fails, fall back to the
            # base adapter's text-only "Image: /path" rendering.
            try:
                return await self.send_document(
                    chat_id=chat_id,
                    file_path=image_path,
                    caption=caption,
                    file_name=os.path.basename(image_path),
                    reply_to=reply_to,
                    metadata=metadata,
                )
            except Exception as doc_err:
                logger.error(
                    "[%s] Failed to send Telegram local image as document, "
                    "falling back to base adapter: %s",
                    self.name,
                    doc_err,
                    exc_info=True,
                )
                return await super().send_image_file(chat_id, image_path, caption, reply_to, metadata=metadata)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a document/file natively as a Telegram file attachment."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        try:
            if not os.path.exists(file_path):
                return SendResult(success=False, error=self._missing_media_path_error("File", file_path))

            display_name = file_name or os.path.basename(file_path)
            _thread = self._metadata_thread_id(metadata)
            reply_to_id = self._reply_to_message_id_for_send(reply_to, metadata, reply_to_mode=self._reply_to_mode)
            thread_kwargs = self._thread_kwargs_for_send(
                chat_id,
                _thread,
                metadata,
                reply_to_message_id=reply_to_id,
                reply_to_mode=self._reply_to_mode
            )

            with open(file_path, "rb") as f:
                msg = await self._send_with_dm_topic_reply_anchor_retry(
                    self._bot.send_document,
                    {
                        "chat_id": normalize_telegram_chat_id(chat_id),
                        "document": f,
                        "filename": display_name,
                        "caption": caption[:1024] if caption else None,
                        "reply_to_message_id": reply_to_id,
                        **thread_kwargs,
                        **self._notification_kwargs(metadata),
                    },
                    metadata,
                    reply_to_id,
                    "document",
                    reset_media=lambda: f.seek(0),
                )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning(
                "[%s] Failed to send document: %s",
                self.name, _redact_telegram_error_text(e),
            )
            return await super().send_document(chat_id, file_path, caption, file_name, reply_to, metadata=metadata)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video natively as a Telegram video message."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        try:
            if not os.path.exists(video_path):
                return SendResult(success=False, error=self._missing_media_path_error("Video", video_path))

            _thread = self._metadata_thread_id(metadata)
            reply_to_id = self._reply_to_message_id_for_send(reply_to, metadata, reply_to_mode=self._reply_to_mode)
            thread_kwargs = self._thread_kwargs_for_send(
                chat_id,
                _thread,
                metadata,
                reply_to_message_id=reply_to_id,
                reply_to_mode=self._reply_to_mode
            )
            with open(video_path, "rb") as f:
                msg = await self._send_with_dm_topic_reply_anchor_retry(
                    self._bot.send_video,
                    {
                        "chat_id": normalize_telegram_chat_id(chat_id),
                        "video": f,
                        "caption": caption[:1024] if caption else None,
                        "reply_to_message_id": reply_to_id,
                        **thread_kwargs,
                        **self._notification_kwargs(metadata),
                    },
                    metadata,
                    reply_to_id,
                    "video",
                    reset_media=lambda: f.seek(0),
                )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning(
                "[%s] Failed to send video: %s",
                self.name, _redact_telegram_error_text(e),
            )
            return await super().send_video(chat_id, video_path, caption, reply_to, metadata=metadata)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image natively as a Telegram photo.
        
        Tries URL-based send first (fast, works for <5MB images).
        Falls back to downloading and uploading as file (supports up to 10MB).
        """
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        from tools.url_safety import is_safe_url
        if not is_safe_url(image_url):
            logger.warning("[%s] Blocked unsafe image URL (SSRF protection)", self.name)
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)

        try:
            # Telegram can send photos directly from URLs (up to ~5MB)
            _photo_thread = self._metadata_thread_id(metadata)
            reply_to_id = self._reply_to_message_id_for_send(reply_to, metadata, reply_to_mode=self._reply_to_mode)
            photo_thread_kwargs = self._thread_kwargs_for_send(
                chat_id,
                _photo_thread,
                metadata,
                reply_to_message_id=reply_to_id,
                reply_to_mode=self._reply_to_mode
            )
            msg = await self._send_with_dm_topic_reply_anchor_retry(
                self._bot.send_photo,
                {
                    "chat_id": normalize_telegram_chat_id(chat_id),
                    "photo": image_url,
                    "caption": caption[:1024] if caption else None,
                    "reply_to_message_id": reply_to_id,
                    **photo_thread_kwargs,
                    **self._notification_kwargs(metadata),
                },
                metadata,
                reply_to_id,
                "URL photo",
            )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning(
                "[%s] URL-based send_photo failed, trying file upload: %s",
                self.name,
                e,
                exc_info=True,
            )
            # Fallback: download and upload as file (supports up to 10MB)
            try:
                import httpx
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(image_url)
                    resp.raise_for_status()
                    image_data = resp.content

                upload_thread_kwargs = self._thread_kwargs_for_send(
                    chat_id,
                    _photo_thread,
                    metadata,
                    reply_to_message_id=reply_to_id,
                    reply_to_mode=self._reply_to_mode
                )
                msg = await self._send_with_dm_topic_reply_anchor_retry(
                    self._bot.send_photo,
                    {
                        "chat_id": normalize_telegram_chat_id(chat_id),
                        "photo": image_data,
                        "caption": caption[:1024] if caption else None,
                        "reply_to_message_id": reply_to_id,
                        **upload_thread_kwargs,
                        **self._notification_kwargs(metadata),
                    },
                    metadata,
                    reply_to_id,
                    "uploaded photo",
                )
                return SendResult(success=True, message_id=str(msg.message_id))
            except Exception as e2:
                logger.error(
                    "[%s] File upload send_photo also failed: %s",
                    self.name,
                    e2,
                    exc_info=True,
                )
                # Final fallback: send URL as text
                return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an animated GIF natively as a Telegram animation (auto-plays inline)."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        
        try:
            _anim_thread = self._metadata_thread_id(metadata)
            reply_to_id = self._reply_to_message_id_for_send(reply_to, metadata, reply_to_mode=self._reply_to_mode)
            animation_thread_kwargs = self._thread_kwargs_for_send(
                chat_id,
                _anim_thread,
                metadata,
                reply_to_message_id=reply_to_id,
                reply_to_mode=self._reply_to_mode
            )
            msg = await self._send_with_dm_topic_reply_anchor_retry(
                self._bot.send_animation,
                {
                    "chat_id": normalize_telegram_chat_id(chat_id),
                    "animation": animation_url,
                    "caption": caption[:1024] if caption else None,
                    "reply_to_message_id": reply_to_id,
                    **animation_thread_kwargs,
                    **self._notification_kwargs(metadata),
                },
                metadata,
                reply_to_id,
                "animation",
            )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.error(
                "[%s] Failed to send Telegram animation, falling back to photo: %s",
                self.name,
                e,
                exc_info=True,
            )
            # Fallback: try as a regular photo
            return await self.send_image(chat_id, animation_url, caption, reply_to, metadata=metadata)

    @staticmethod
    def _is_transient_typing_error(exc: Exception) -> bool:
        """Return True for Telegram typing errors worth cooling down."""
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            return True

        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
            return True

        text = str(exc).lower()
        if any(marker in text for marker in ("too many requests", "rate limit", "timed out", "timeout", "temporar")):
            return True
        if isinstance(exc, (OSError, TimeoutError, ConnectionError, asyncio.TimeoutError)):
            return True
        return False

    def _record_typing_cooldown(self, chat_id: str, exc: Exception) -> None:
        """Suppress Telegram typing refreshes for this chat after transient failures."""
        if not hasattr(self, "_telegram_typing_cooldown_until"):
            self._telegram_typing_cooldown_until = {}
        loop = asyncio.get_running_loop()
        retry_after = getattr(exc, "retry_after", None)
        try:
            delay = float(retry_after) if retry_after is not None else self._telegram_typing_cooldown_seconds
        except (TypeError, ValueError):
            delay = self._telegram_typing_cooldown_seconds
        delay = max(1.0, min(delay, 300.0))
        self._telegram_typing_cooldown_until[str(chat_id)] = loop.time() + delay

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
        try:
            _typing_thread = self._metadata_thread_id(metadata)
            _is_dm_topic = bool(metadata and metadata.get("telegram_dm_topic_reply_fallback"))
            message_thread_id = self._message_thread_id_for_typing(_typing_thread)
            await self._bot.send_chat_action(
                chat_id=normalize_telegram_chat_id(chat_id),
                action="typing",
                message_thread_id=message_thread_id,
            )
            self._telegram_typing_cooldown_until.pop(str(chat_id), None)
        except Exception as e:
            # For DM topic lanes, Telegram may reject message_thread_id.
            # Fall back to sending typing without thread_id so the typing
            # indicator at least appears in the main DM view.
            if _is_dm_topic and message_thread_id is not None:
                try:
                    await self._bot.send_chat_action(
                        chat_id=normalize_telegram_chat_id(chat_id),
                        action="typing",
                    )
                    self._telegram_typing_cooldown_until.pop(str(chat_id), None)
                    return
                except Exception as fallback_exc:
                    if self._is_transient_typing_error(fallback_exc):
                        self._record_typing_cooldown(chat_id, fallback_exc)
            elif self._is_transient_typing_error(e):
                self._record_typing_cooldown(chat_id, e)
            # Typing failures are non-fatal; log at debug level only.
            logger.debug(
                "[%s] Failed to send Telegram typing indicator: %s",
                self.name,
                e,
                exc_info=True,
            )

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
                chat_type = "group"
                if chat.is_forum:
                    chat_type = "forum"
            elif chat.type == ChatType.CHANNEL:
                chat_type = "channel"
            
            return {
                "name": chat.title or chat.full_name or str(chat_id),
                "type": chat_type,
                "username": chat.username,
                "is_forum": getattr(chat, "is_forum", False),
            }
        except Exception as e:
            logger.error(
                "[%s] Failed to get Telegram chat info for %s: %s",
                self.name,
                chat_id,
                e,
                exc_info=True,
            )
            return {"name": str(chat_id), "type": "dm", "error": str(e)}

    def format_message(self, content: str) -> str:
        """
        Convert standard markdown to Telegram MarkdownV2 format.

        Protected regions (code blocks, inline code) are extracted first so
        their contents are never modified.  Standard markdown constructs
        (headers, bold, italic, links) are translated to MarkdownV2 syntax,
        and all remaining special characters are escaped.
        """
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

        text = content

        # 0) Rewrite GFM-style pipe tables into Telegram-friendly row groups
        #    before the normal MarkdownV2 conversions run.
        text = _wrap_markdown_tables(text)

        # 1) Protect fenced code blocks (``` ... ```)
        #    Per MarkdownV2 spec, \ and ` inside pre/code must be escaped.
        def _protect_fenced(m):
            raw = m.group(0)
            # Split off opening ``` (with optional language) and closing ```
            open_end = raw.index('\n') + 1 if '\n' in raw[3:] else 3
            opening = raw[:open_end]
            body_and_close = raw[open_end:]
            body = body_and_close[:-3]
            body = body.replace('\\', '\\\\').replace('`', '\\`')
            return _ph(opening + body + '```')

        text = re.sub(
            r'(```(?:[^\n]*\n)?[\s\S]*?```)',
            _protect_fenced,
            text,
        )

        # 2) Protect inline code (`...`)
        #    Escape \ inside inline code per MarkdownV2 spec.
        text = re.sub(
            r'(`[^`]+`)',
            lambda m: _ph(m.group(0).replace('\\', '\\\\')),
            text,
        )

        # 3) Convert markdown links – escape the display text; inside the URL
        #    only ')' and '\' need escaping per the MarkdownV2 spec.
        def _convert_link(m):
            display = _escape_mdv2(m.group(1))
            url = m.group(2).replace('\\', '\\\\').replace(')', '\\)')
            return _ph(f'[{display}]({url})')

        text = re.sub(r'\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)', _convert_link, text)

        # 4) Convert markdown headers (## Title) → bold *Title*
        def _convert_header(m):
            inner = m.group(1).strip()
            # Strip redundant bold markers that may appear inside a header
            inner = re.sub(r'\*\*(.+?)\*\*', r'\1', inner)
            return _ph(f'*{_escape_mdv2(inner)}*')

        text = re.sub(
            r'^#{1,6}\s+(.+)$', _convert_header, text, flags=re.MULTILINE
        )

        # 5) Convert bold: **text** → *text* (MarkdownV2 bold)
        text = re.sub(
            r'\*\*(.+?)\*\*',
            lambda m: _ph(f'*{_escape_mdv2(m.group(1))}*'),
            text,
        )

        # 6) Convert italic: *text* (single asterisk) → _text_ (MarkdownV2 italic)
        #    [^*\n]+ prevents matching across newlines (which would corrupt
        #    bullet lists using * markers and multi-line content).
        text = re.sub(
            r'\*([^*\n]+)\*',
            lambda m: _ph(f'_{_escape_mdv2(m.group(1))}_'),
            text,
        )

        # 7) Convert strikethrough: ~~text~~ → ~text~ (MarkdownV2)
        text = re.sub(
            r'~~(.+?)~~',
            lambda m: _ph(f'~{_escape_mdv2(m.group(1))}~'),
            text,
        )

        # 8) Convert spoiler: ||text|| → ||text|| (protect from | escaping)
        text = re.sub(
            r'\|\|(.+?)\|\|',
            lambda m: _ph(f'||{_escape_mdv2(m.group(1))}||'),
            text,
        )

        # 9) Convert blockquotes: > at line start → protect > from escaping
        #    Handle both regular blockquotes (> text) and expandable blockquotes
        #    (Telegram MarkdownV2: **> for expandable start, || to end the quote)
        def _convert_blockquote(m):
            prefix = m.group(1)  # >, >>, >>>, **>, or **>> etc.
            content = m.group(2)
            # Check if content ends with || (expandable blockquote end marker)
            # In this case, preserve the trailing || unescaped for Telegram
            if prefix.startswith('**') and content.endswith('||'):
                return _ph(f'{prefix} {_escape_mdv2(content[:-2])}||')
            return _ph(f'{prefix} {_escape_mdv2(content)}')

        text = re.sub(
            r'^((?:\*\*)?>{1,3}) (.+)$',
            _convert_blockquote,
            text,
            flags=re.MULTILINE,
        )

        # 10) Escape remaining special characters in plain text
        text = _escape_mdv2(text)

        # 11) Restore placeholders in reverse insertion order so that
        #    nested references (a placeholder inside another) resolve correctly.
        for key in reversed(list(placeholders.keys())):
            text = text.replace(key, placeholders[key])

        # 12) Safety net: escape unescaped ( ) { } that slipped through
        #     placeholder processing.  Split the text into code/non-code
        #     segments so we never touch content inside ``` or ` spans.
        _code_split = re.split(r'(```[\s\S]*?```|`[^`]+`)', text)
        _safe_parts = []
        for _idx, _seg in enumerate(_code_split):
            if _idx % 2 == 1:
                # Inside code span/block — leave untouched
                _safe_parts.append(_seg)
            else:
                # Outside code — escape bare ( ) { }
                def _esc_bare(m, _seg=_seg):
                    s = m.start()
                    ch = m.group(0)
                    # Already escaped
                    if s > 0 and _seg[s - 1] == '\\':
                        return ch
                    # ( that opens a MarkdownV2 link [text](url)
                    if ch == '(' and s > 0 and _seg[s - 1] == ']':
                        return ch
                    # ) that closes a link URL
                    if ch == ')':
                        before = _seg[:s]
                        if '](http' in before or '](' in before:
                            # Check depth
                            depth = 0
                            for j in range(s - 1, max(s - 2000, -1), -1):
                                if _seg[j] == '(':
                                    depth -= 1
                                    if depth < 0:
                                        if j > 0 and _seg[j - 1] == ']':
                                            return ch
                                        break
                                elif _seg[j] == ')':
                                    depth += 1
                    return '\\' + ch
                _safe_parts.append(re.sub(r'[(){}]', _esc_bare, _seg))
        text = ''.join(_safe_parts)

        return text

    # ── Group mention gating ──────────────────────────────────────────────

    def _telegram_require_mention(self) -> bool:
        """Return whether group chats should require an explicit bot trigger."""
        configured = self.config.extra.get("require_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return os.getenv("TELEGRAM_REQUIRE_MENTION", "false").lower() in {"true", "1", "yes", "on"}

    def _telegram_observe_unmentioned_group_messages(self) -> bool:
        """Return whether skipped unmentioned group messages are stored as context.

        When enabled with ``require_mention``, Telegram matches the Yuanbao /
        OpenClaw-style group UX: observe ordinary group chatter in the session
        transcript, but only dispatch the agent when the bot is explicitly
        addressed.
        """
        configured = self.config.extra.get("observe_unmentioned_group_messages")
        if configured is None:
            configured = self.config.extra.get("ingest_unmentioned_group_messages")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return os.getenv("TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES", "false").lower() in {"true", "1", "yes", "on"}

    def _telegram_guest_mode(self) -> bool:
        """Return whether non-allowlisted groups may trigger via direct @mention."""
        configured = self.config.extra.get("guest_mode")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return os.getenv("TELEGRAM_GUEST_MODE", "false").lower() in {"true", "1", "yes", "on"}

    def _telegram_exclusive_bot_mentions(self) -> bool:
        """Return whether explicit @...bot mentions exclusively route group messages."""
        configured = self.config.extra.get("exclusive_bot_mentions")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return os.getenv("TELEGRAM_EXCLUSIVE_BOT_MENTIONS", "true").lower() in {"true", "1", "yes", "on"}

    def _telegram_free_response_chats(self) -> set[str]:
        raw = self.config.extra.get("free_response_chats")
        if raw is None:
            raw = os.getenv("TELEGRAM_FREE_RESPONSE_CHATS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _telegram_allowed_chats(self) -> set[str]:
        """Return the whitelist of group/supergroup chat IDs the bot will respond in.

        When non-empty, group messages from chats NOT in this set are
        silently ignored unless ``guest_mode`` is enabled and the bot is
        explicitly @mentioned.  DMs are never filtered.
        Empty set means no restriction (fully backward compatible).
        """
        raw = self.config.extra.get("allowed_chats")
        if raw is None:
            raw = os.getenv("TELEGRAM_ALLOWED_CHATS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _telegram_group_allowed_chats(self) -> set[str]:
        """Return Telegram chats authorized at group scope."""
        raw = self.config.extra.get("group_allowed_chats")
        if raw is None:
            raw = os.getenv("TELEGRAM_GROUP_ALLOWED_CHATS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _telegram_observe_allowed_chats(self) -> set[str]:
        """Chats where observed group context may use a shared source.

        ``group_allowed_chats`` is the gateway authorization allowlist for
        user-less group sources.  ``allowed_chats`` remains an optional response
        gate; when set, observed context must satisfy both lists.
        """
        group_allowed = self._telegram_group_allowed_chats()
        if not group_allowed:
            return set()
        response_allowed = self._telegram_allowed_chats()
        if response_allowed:
            return group_allowed & response_allowed
        return group_allowed

    def _telegram_allowed_topics(self) -> set[str]:
        """Return the whitelist of Telegram forum topic IDs this bot handles.

        When non-empty, group/supergroup messages from other topics are
        silently ignored. DMs are never filtered by topic. Telegram may omit
        ``message_thread_id`` for the forum General topic, so ``None`` is
        treated as topic ``1`` for matching purposes.
        """
        raw = self.config.extra.get("allowed_topics")
        if raw is None:
            raw = os.getenv("TELEGRAM_ALLOWED_TOPICS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _telegram_ignored_threads(self) -> set[int]:
        raw = self.config.extra.get("ignored_threads")
        if raw is None:
            raw = os.getenv("TELEGRAM_IGNORED_THREADS", "")

        if isinstance(raw, list):
            values = raw
        else:
            values = str(raw).split(",")

        ignored: set[int] = set()
        for value in values:
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
            return []
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            logger.warning(
                "[%s] telegram mention_patterns must be a list or string; got %s",
                self.name,
                type(patterns).__name__,
            )
            return []

        compiled: List[re.Pattern] = []
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                logger.warning("[%s] Invalid Telegram mention pattern %r: %s", self.name, pattern, exc)
        if compiled:
            logger.info("[%s] Loaded %d Telegram mention pattern(s)", self.name, len(compiled))
        return compiled

    def _is_group_chat(self, message: Message) -> bool:
        chat = getattr(message, "chat", None)
        if not chat:
            return False
        chat_type = str(getattr(chat, "type", "")).split(".")[-1].lower()
        return chat_type in {"group", "supergroup"}

    @classmethod
    def _effective_message_thread_id(cls, message: Message) -> Optional[str]:
        """Return the routable thread id for a Telegram message.

        Forum supergroup messages posted in the General topic arrive with
        ``message_thread_id=None`` while Telegram itself addresses that topic
        as thread id ``1``.  Ordinary replies are the opposite footgun:
        Telegram populates ``message_thread_id`` with a reply-UI anchor id on
        plain group/DM replies, but those ids are not topic/session routing
        ids and must not be treated as such.  Gating, skill binding, and
        outbound routing must all agree on the same normalized value.
        """
        chat = getattr(message, "chat", None)
        chat_type = str(getattr(chat, "type", "")).split(".")[-1].lower() if chat else ""
        raw = getattr(message, "message_thread_id", None)
        is_topic_message = bool(getattr(message, "is_topic_message", False))
        is_forum_group = chat_type in ("group", "supergroup") and getattr(chat, "is_forum", False) is True
        if raw is not None:
            if is_forum_group or (chat_type in ("group", "supergroup") and is_topic_message):
                return str(raw)
            if chat_type == "private" and is_topic_message:
                return str(raw)
            return None
        if is_forum_group:
            return cls._GENERAL_TOPIC_THREAD_ID
        return None

    def _is_reply_to_bot(self, message: Message) -> bool:
        if not self._bot or not getattr(message, "reply_to_message", None):
            return False
        reply_user = getattr(message.reply_to_message, "from_user", None)
        return bool(reply_user and getattr(reply_user, "id", None) == getattr(self._bot, "id", None))

    @staticmethod
    def _extract_bot_mention_usernames(message: Message) -> set[str]:
        """Extract explicit Telegram bot usernames mentioned in text/captions.

        Telegram bot usernames are 5-32 characters and must end in "bot".
        Entity mentions are authoritative. The raw-text fallback is intentionally narrow so
        entity-less mobile/client variants still work without treating email
        addresses or arbitrary substrings as bot mentions.
        """
        mentioned_bot_usernames: set[str] = set()

        def _iter_sources():
            yield getattr(message, "text", None) or "", getattr(message, "entities", None) or []
            yield getattr(message, "caption", None) or "", getattr(message, "caption_entities", None) or []

        for source_text, entities in _iter_sources():
            for entity in entities:
                entity_type = str(getattr(entity, "type", "")).split(".")[-1].lower()
                if entity_type not in {"mention", "bot_command"}:
                    continue
                offset = int(getattr(entity, "offset", -1))
                length = int(getattr(entity, "length", 0))
                if offset < 0 or length <= 0:
                    continue

                entity_text = source_text[offset:offset + length].strip()
                if entity_type == "mention":
                    handle = entity_text.lstrip("@").lower()
                    if re.fullmatch(r"[a-z0-9_]{2,29}bot", handle, re.IGNORECASE):
                        mentioned_bot_usernames.add(handle)
                    continue

                # Telegram emits /cmd@botname as one bot_command entity, not as
                # a separate mention entity. Treat that suffix as an explicit
                # bot address for exclusive multi-bot routing even when the
                # group has require_mention/free-response disabled.
                at_index = entity_text.find("@")
                if at_index < 0:
                    continue
                command_target = entity_text[at_index + 1:].strip().lower()
                if re.fullmatch(r"[a-z0-9_]{2,29}bot", command_target, re.IGNORECASE):
                    mentioned_bot_usernames.add(command_target)

        # Entity-less fallback for older/client-specific updates. If Telegram
        # supplied entities for a source, trust them and do not regex-rescue
        # malformed/URL/code spans that the server did not mark as mentions.
        for raw_text, entities in _iter_sources():
            if not raw_text or entities:
                continue
            for match in re.finditer(r"(?i)(?<![A-Za-z0-9_`/])@([A-Za-z0-9_]{2,29}bot)\b", raw_text):
                mentioned_bot_usernames.add(match.group(1).lower())

        return mentioned_bot_usernames

    def _message_mentions_bot(self, message: Message) -> bool:
        if not self._bot:
            return False

        bot_username = (getattr(self._bot, "username", None) or "").lstrip("@").lower()
        bot_id = getattr(self._bot, "id", None)
        expected = f"@{bot_username}" if bot_username else None

        def _iter_sources():
            yield getattr(message, "text", None) or "", getattr(message, "entities", None) or []
            yield getattr(message, "caption", None) or "", getattr(message, "caption_entities", None) or []

        # Telegram parses mentions server-side and emits MessageEntity objects
        # (type=mention for @username, type=text_mention for @FirstName targeting
        # a user without a public username). Those entities are authoritative:
        # raw substring matches like "foo@hermes_bot.example" are not mentions
        # (bug #12545). Entities also correctly handle @handles inside URLs, code
        # blocks, and quoted text, where a regex scan would over-match.
        for source_text, entities in _iter_sources():
            for entity in entities:
                entity_type = str(getattr(entity, "type", "")).split(".")[-1].lower()
                if entity_type == "mention" and expected:
                    offset = int(getattr(entity, "offset", -1))
                    length = int(getattr(entity, "length", 0))
                    if offset < 0 or length <= 0:
                        continue
                    if source_text[offset:offset + length].strip().lower() == expected:
                        return True
                elif entity_type == "text_mention":
                    user = getattr(entity, "user", None)
                    if user and getattr(user, "id", None) == bot_id:
                        return True
                elif entity_type == "bot_command" and expected:
                    # Telegram's official group-disambiguation form for slash
                    # commands (``/cmd@botname``) is emitted as a single
                    # ``bot_command`` entity covering the whole span — there
                    # is no accompanying ``mention`` entity. Treat it as a
                    # direct address to this bot when the ``@botname`` suffix
                    # matches. This is the form Telegram's own command menu
                    # autocomplete produces in groups, so dropping it at the
                    # mention gate would break /new, /reset, /help, ... for
                    # every group that has ``require_mention`` enabled (#15415).
                    offset = int(getattr(entity, "offset", -1))
                    length = int(getattr(entity, "length", 0))
                    if offset < 0 or length <= 0:
                        continue
                    command_text = source_text[offset:offset + length]
                    at_index = command_text.find("@")
                    if at_index < 0:
                        continue
                    if command_text[at_index:].strip().lower() == expected:
                        return True
        if bot_username and re.fullmatch(r"[a-z0-9_]{2,29}bot", bot_username, re.IGNORECASE):
            return bot_username in self._extract_bot_mention_usernames(message)
        return False

    def _explicit_bot_mentions_exclude_self(self, message: Message) -> bool:
        """Return True when explicit bot handles target other bots, not this one.

        Telegram groups can contain several Hermes bot profiles. A message like
        ``@bot3 hi @bot4`` must not wake ``@bot1`` through reply/wake-word
        fallbacks. Treat explicit bot-handle mentions as an exclusive routing
        hint: if at least one @...bot username is present and none matches this
        adapter's own bot username, this adapter should ignore the message.

        MessageEntity values are preferred, but some Telegram clients expose
        selected bot handles as plain text in group messages. The raw-text
        fallback is intentionally limited to usernames ending in "bot", which
        Telegram requires for bot accounts.
        """
        if not self._bot:
            return False

        bot_username = (getattr(self._bot, "username", None) or "").lstrip("@").lower()
        if not bot_username:
            return False

        mentioned_bot_usernames = self._extract_bot_mention_usernames(message)
        return bool(mentioned_bot_usernames) and bot_username not in mentioned_bot_usernames

    def _message_matches_mention_patterns(self, message: Message) -> bool:
        if not self._mention_patterns:
            return False
        for candidate in (getattr(message, "text", None), getattr(message, "caption", None)):
            if not candidate:
                continue
            for pattern in self._mention_patterns:
                if pattern.search(candidate):
                    return True
        return False

    def _is_guest_mention(self, message: Message) -> bool:
        """Return True for the narrow guest-mode bypass: explicit bot mention.

        The caller (:meth:`_should_process_message`) has already verified
        the message is a group chat, so that check is not repeated here.
        """
        return self._telegram_guest_mode() and self._message_mentions_bot(message)

    def _clean_bot_trigger_text(self, text: Optional[str]) -> Optional[str]:
        if not text or not self._bot or not getattr(self._bot, "username", None):
            return text
        username = re.escape(self._bot.username)
        cleaned = re.sub(rf"(?i)@{username}\b[,:\-]*\s*", "", text).strip()
        return cleaned or text

    def _should_observe_unmentioned_group_message(self, message: Message) -> bool:
        """Return True when a group message should be stored but not dispatched."""
        if self._is_own_message(message):
            return False
        if not self._telegram_observe_unmentioned_group_messages():
            return False
        if not self._is_group_chat(message):
            return False

        thread_id = getattr(message, "message_thread_id", None)
        allowed_topics = self._telegram_allowed_topics()
        if allowed_topics:
            topic_id = str(thread_id) if thread_id is not None else self._GENERAL_TOPIC_THREAD_ID
            if topic_id not in allowed_topics:
                return False

        if thread_id is not None:
            try:
                if int(thread_id) in self._telegram_ignored_threads():
                    return False
            except (TypeError, ValueError):
                return False

        chat_id_str = str(getattr(getattr(message, "chat", None), "id", ""))
        if self._telegram_exclusive_bot_mentions() and self._explicit_bot_mentions_exclude_self(message):
            return False

        allowed = self._telegram_observe_allowed_chats()
        # Observed context is shared at chat/topic scope so a later trigger from
        # another user can see it.  Require an explicit chat allowlist; that
        # keeps shared observed history limited to operator-approved groups and
        # lets gateway authorization pass even after the shared session source
        # drops the per-sender user_id.
        if not allowed or chat_id_str not in allowed:
            return False

        # Only observe messages skipped by the require_mention gate.  If the
        # message would be processed normally, let the dispatcher handle it;
        # if require_mention is disabled, every group message is a request.
        if chat_id_str in self._telegram_free_response_chats():
            return False
        if not self._telegram_require_mention():
            return False
        if self._is_reply_to_bot(message):
            return False
        if self._message_mentions_bot(message):
            return False
        if self._message_matches_mention_patterns(message):
            return False
        return True

    def _telegram_group_observe_shared_source(self, source):
        """Return a chat/topic-scoped source for observed Telegram group context."""
        return dataclasses.replace(source, user_id=None, user_name=None, user_id_alt=None)

    def _telegram_group_observe_attributed_text(self, event: MessageEvent) -> str:
        user_id = event.source.user_id or "unknown"
        sender = event.source.user_name or user_id
        return f"[{sender}|{user_id}]\n{event.text or ''}"

    def _telegram_group_observe_channel_prompt(self) -> str:
        username = getattr(getattr(self, "_bot", None), "username", None) or "unknown"
        bot_id = getattr(getattr(self, "_bot", None), "id", None) or "unknown"
        return (
            "You are handling a Telegram group chat message.\n"
            f"- Your identity: user_id={bot_id}, @-mention name in this group=@{username}\n"
            "- observed Telegram group context may be provided in a separate context-only block "
            "before the current message; it is not necessarily addressed to you.\n"
            "- Treat only the current new message as a request explicitly directed at you, "
            "and use observed context only when the current message asks for it."
        )

    def _apply_telegram_group_observe_attribution(self, event: MessageEvent) -> MessageEvent:
        """Align triggered group turns with observed-history attribution."""
        if not self._telegram_observe_unmentioned_group_messages():
            return event
        raw_message = getattr(event, "raw_message", None)
        if not raw_message or not self._is_group_chat(raw_message):
            return event
        chat_id_str = str(getattr(getattr(raw_message, "chat", None), "id", ""))
        allowed = self._telegram_observe_allowed_chats()
        if not allowed or chat_id_str not in allowed:
            return event
        shared_source = self._telegram_group_observe_shared_source(event.source)
        observe_prompt = self._telegram_group_observe_channel_prompt()
        channel_prompt = f"{event.channel_prompt}\n\n{observe_prompt}" if event.channel_prompt else observe_prompt
        if event.message_type == MessageType.COMMAND:
            return dataclasses.replace(
                event,
                source=shared_source,
                channel_prompt=channel_prompt,
            )
        return dataclasses.replace(
            event,
            text=self._telegram_group_observe_attributed_text(event),
            source=shared_source,
            channel_prompt=channel_prompt,
        )

    def _media_message_type(self, msg: Message) -> MessageType:
        """Classify a Telegram media message into a MessageType."""
        if msg.sticker:
            return MessageType.STICKER
        if msg.photo:
            return MessageType.PHOTO
        if msg.video:
            return MessageType.VIDEO
        if msg.audio:
            return MessageType.AUDIO
        if msg.voice:
            return MessageType.VOICE
        return MessageType.DOCUMENT

    async def _cache_observed_media(self, msg: Message, event: MessageEvent) -> None:
        """Cache an unmentioned group attachment and annotate the observed text.

        Passive group traffic, so downloads are bounded by the same
        ``_max_doc_bytes`` limit as the addressed document path. Oversized or
        unsupported attachments are noted in the transcript without downloading.
        """
        from gateway.platforms.base import cache_media_bytes

        source, filename, mime, kind = self._observed_media_source(msg)
        if source is None:
            return

        max_bytes = getattr(self, "_max_doc_bytes", 20 * 1024 * 1024)
        file_size = getattr(source, "file_size", None)
        try:
            size = int(file_size or 0)
        except (TypeError, ValueError):
            size = 0
        if not (0 < size <= max_bytes):
            limit_mb = max_bytes // (1024 * 1024)
            event.text = self._append_observed_note(
                event.text,
                f"[Observed Telegram attachment too large or unverifiable. Maximum: {limit_mb} MB.]",
            )
            logger.info("[Telegram] Observed group attachment skipped (size=%s)", file_size)
            return

        try:
            file_obj = await source.get_file()
            data = bytes(await file_obj.download_as_bytearray())
            if not filename:
                filename = os.path.basename(getattr(file_obj, "file_path", "") or "")
            cached = cache_media_bytes(data, filename=filename, mime_type=mime, default_kind=kind)
        except Exception as exc:
            logger.warning("[Telegram] Failed to cache observed group media: %s", exc, exc_info=True)
            return

        if cached is None:
            # Only reachable for images that fail validation now — any other
            # file type is always cached (authorization is the gate, not the
            # extension).
            event.text = self._append_observed_note(
                event.text, "[Observed Telegram attachment could not be read, not cached.]"
            )
            return

        event.media_urls = [cached.path]
        event.media_types = [cached.media_type]
        if cached.kind == "image":
            event.message_type = MessageType.PHOTO
        elif cached.kind == "video":
            event.message_type = MessageType.VIDEO
        event.text = self._append_observed_note(event.text, cached.context_note())
        logger.info("[Telegram] Cached observed group %s at %s", cached.kind, cached.path)

    async def _cache_replied_media(self, msg: Any, event: MessageEvent) -> None:
        """Cache media from the message this turn replies to, if any."""
        from gateway.platforms.base import cache_media_bytes

        reply_msg = getattr(msg, "reply_to_message", None)
        if reply_msg is None:
            return
        source, filename, mime, kind = self._observed_media_source(reply_msg)
        if source is None:
            return

        max_bytes = getattr(self, "_max_doc_bytes", 20 * 1024 * 1024)
        file_size = getattr(source, "file_size", None)
        try:
            size = int(file_size or 0)
        except (TypeError, ValueError):
            size = 0
        if not (0 < size <= max_bytes):
            return

        try:
            file_obj = await source.get_file()
            data = bytes(await file_obj.download_as_bytearray())
            if not filename:
                filename = os.path.basename(getattr(file_obj, "file_path", "") or "")
            cached = cache_media_bytes(data, filename=filename, mime_type=mime, default_kind=kind)
        except Exception as exc:
            logger.warning("[Telegram] Failed to cache replied-to media: %s", exc, exc_info=True)
            return

        if cached is None:
            return

        event.media_urls.append(cached.path)
        event.media_types.append(cached.media_type)
        if len(event.media_urls) == 1:
            if cached.kind == "image":
                event.message_type = MessageType.PHOTO
            elif cached.kind == "video":
                event.message_type = MessageType.VIDEO
        event.text = self._append_observed_note(
            event.text,
            f"[Replied-to {cached.kind} '{cached.display_name}' saved at: {cached.path}]",
        )
        logger.info("[Telegram] Cached replied-to %s at %s", cached.kind, cached.path)

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
        if not existing:
            return note
        return f"{existing}\n\n{note}"

    async def _surface_media_cache_failure(
        self,
        msg: Message,
        event: MessageEvent,
        kind: str,
        exc: Exception,
        display_name: Optional[str] = None,
    ) -> None:
        """Surface a failed media download/cache on BOTH ends instead of swallowing it.

        When download_as_bytearray()/cache_*_from_bytes() raises (typically a
        transient httpx.ConnectError to Telegram's CDN), the attachment never
        made it into event.media_urls. Without this, the handler falls through
        and dispatches an empty turn: the user thinks the file was delivered,
        the agent sees nothing, and the only record is a buried log warning.

        This (1) replies to the user in Telegram so they know to retry, and
        (2) appends an agent-visible notice to event.text via the existing
        observed-note channel so the agent knows an attachment was attempted
        and failed — never a silent empty turn. No new event fields (the
        structured-event refactor is out of scope per #23045).
        """
        named = f" ({display_name})" if display_name else ""
        try:
            await msg.reply_text(
                f"\u26a0\ufe0f Couldn't download your {kind}{named} "
                f"({exc.__class__.__name__}). Please try sending it again."
            )
        except Exception as reply_err:
            logger.warning(
                "[Telegram] Failed to notify user about %s cache failure: %s",
                kind,
                reply_err,
                exc_info=True,
            )
        agent_note = (
            f"[The user attempted to send a {kind}{named} but it could not be "
            f"downloaded ({exc.__class__.__name__}); they have been asked to retry.]"
        )
        event.text = self._append_observed_note(event.text, agent_note)

    def _observe_unmentioned_group_message(
        self,
        message: Message,
        msg_type: MessageType,
        update_id: Optional[int] = None,
        event: Optional[MessageEvent] = None,
    ) -> None:
        """Append skipped group chatter to the target session without dispatching."""
        store = getattr(self, "_session_store", None)
        if not store:
            return
        try:
            event = event or self._build_message_event(message, msg_type, update_id=update_id)
            shared_source = self._telegram_group_observe_shared_source(event.source)
            session_entry = store.get_or_create_session(shared_source)
            entry = {
                "role": "user",
                "content": self._telegram_group_observe_attributed_text(event),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "observed": True,
            }
            if event.message_id:
                entry["message_id"] = str(event.message_id)
            store.append_to_transcript(session_entry.session_id, entry)
            adapter_name = getattr(self, "name", "telegram")
            logger.info(
                "[%s] Telegram group message observed (no bot trigger): chat=%s from=%s",
                adapter_name,
                getattr(getattr(message, "chat", None), "id", "unknown"),
                event.source.user_id or "unknown",
            )
        except Exception as exc:
            adapter_name = getattr(self, "name", "telegram")
            logger.warning("[%s] Failed to observe Telegram group message: %s", adapter_name, exc)

    def _is_own_message(self, message: Message) -> bool:
        """Return True when the message was sent by this bot itself.

        In some Telegram environments (groups, supergroups where the bot can
        see its own messages), getUpdates returns the bot's own outgoing
        messages as updates.  These must be filtered out so they are not
        counted as incoming unread messages in the Hermes inbox.
        """
        if not self._bot:
            return False
        from_user = getattr(message, "from_user", None)
        if from_user is None:
            return False
        bot_id = getattr(self._bot, "id", None)
        user_id = getattr(from_user, "id", None)
        return bot_id is not None and user_id is not None and bot_id == user_id

    def _should_process_message(self, message: Message, *, is_command: bool = False) -> bool:
        """Apply Telegram group trigger rules.

        DMs remain unrestricted. Group/supergroup messages are accepted when:
        - the chat passes the ``allowed_chats`` whitelist (when set), or
          ``guest_mode`` is enabled and the bot is explicitly mentioned
        - the chat is explicitly allowlisted in ``free_response_chats``
        - ``require_mention`` is disabled
        - the message replies to the bot
        - the bot is @mentioned
        - the text/caption matches a configured regex wake-word pattern

        When ``allowed_chats`` is non-empty, it remains a hard gate except for
        the narrow ``guest_mode`` bypass: group/supergroup messages that
        explicitly @mention this bot. Replies and regex wake words do not bypass
        ``allowed_chats``. When ``require_mention`` is enabled, slash commands are not given
        special treatment — they must pass the same mention/reply checks
        as any other group message.  Users can still trigger commands via
        the Telegram bot menu (``/command@botname``) or by explicitly
        mentioning the bot (``@botname /command``), both of which are
        recognised as mentions by :meth:`_message_mentions_bot`.
        """
        # Filter out the bot's own messages (returned by getUpdates in some
        # environments like groups/supergroups where the bot can see its own
        # messages).  Without this, outbound messages are counted as incoming
        # unread in the Hermes inbox (#52363).
        if self._is_own_message(message):
            return False

        if not self._is_group_chat(message):
            return True

        thread_id = self._effective_message_thread_id(message)
        allowed_topics = self._telegram_allowed_topics()
        if allowed_topics:
            topic_id = str(thread_id) if thread_id is not None else self._GENERAL_TOPIC_THREAD_ID
            if topic_id not in allowed_topics:
                return False

        # Check ignored_threads first — applies to both groups and DM topics
        if thread_id is not None:
            try:
                if int(thread_id) in self._telegram_ignored_threads():
                    return False
            except (TypeError, ValueError):
                logger.warning("[%s] Ignoring non-numeric Telegram message_thread_id: %r", self.name, thread_id)

        if not self._is_group_chat(message):
            # Root DM (non-topic): ignore if ignore_root_dm is configured
            if thread_id is None and self.config.extra.get("ignore_root_dm", False):
                chat_id = str(getattr(getattr(message, "chat", None), "id", ""))
                if not is_command and chat_id in self._dm_topic_chat_ids:
                    return False
            return True

        chat_id_str = str(getattr(getattr(message, "chat", None), "id", ""))

        if self._telegram_exclusive_bot_mentions() and self._explicit_bot_mentions_exclude_self(message):
            return False

        # Resolve guest-mode mention bypass once so _message_mentions_bot
        # is not called redundantly in the normal flow below.
        guest_mention = self._is_guest_mention(message)

        # allowed_chats check (whitelist). When set, group messages from chats
        # outside the whitelist are ignored unless guest_mode permits this
        # exact message as an explicit direct mention. DMs are excluded above.
        allowed = self._telegram_allowed_chats()
        if allowed and chat_id_str not in allowed:
            return guest_mention

        if guest_mention:
            return True
        if chat_id_str in self._telegram_free_response_chats():
            return True
        if not self._telegram_require_mention():
            return True
        if self._is_reply_to_bot(message):
            return True
        # When guest_mode is True, _is_guest_mention already called
        # _message_mentions_bot above — skip the redundant second call.
        if not self._telegram_guest_mode() and self._message_mentions_bot(message):
            return True
        return self._message_matches_mention_patterns(message)

    async def _ensure_forum_commands(self, message) -> None:
        """Lazy-register bot commands for forum supergroups.

        Forum topics don't inherit AllGroupChats scope — Telegram resolves
        via BotCommandScopeChat(chat_id).  Register on first message so the
        command menu works in topic views.
        """
        async with self._forum_lock:
            try:
                chat = getattr(message, "chat", None)
                if not chat or not getattr(chat, "is_forum", False):
                    return
                chat_id = int(chat.id)
                if chat_id in self._forum_command_registered:
                    return
                from telegram import BotCommand, BotCommandScopeChat
                from hermes_cli.commands import telegram_menu_commands, telegram_menu_max_commands
                menu_commands, _ = telegram_menu_commands(max_commands=telegram_menu_max_commands())
                bot_commands = [BotCommand(name, desc) for name, desc in menu_commands]
                await self._bot.set_my_commands(bot_commands, scope=BotCommandScopeChat(chat_id=chat_id))
                self._forum_command_registered.add(chat_id)
                logger.info("[%s] Lazy-registered %d commands for forum chat %s", self.name, len(bot_commands), chat_id)
            except Exception as e:
                logger.warning("[%s] Forum command lazy-registration failed: %s", self.name, e)

    def _effective_update_message(self, update: Update) -> Optional[Message]:
        """Return the message-like payload for normal messages and channel posts.

        Telegram exposes channel broadcasts as ``update.channel_post`` rather
        than ``update.message``.  MessageHandler filters can still dispatch
        those updates, so handlers must use ``effective_message`` to avoid
        consuming channel posts without ever building a gateway event.
        """
        return getattr(update, "effective_message", None) or getattr(update, "message", None)

    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages.

        Telegram clients split long messages into multiple updates.  Buffer
        rapid successive text messages from the same user/chat and aggregate
        them into a single MessageEvent before dispatching.
        """
        msg = self._effective_update_message(update)
        if not msg or not msg.text:
            return
        # Early user-level auth check: reject unauthorized users before any
        # text batching, observe-buffer persistence, event building, or response
        # generation. This prevents removed/blocked users from injecting prompts
        # into the agent path or the observed transcript context (#40863).
        if not self._is_user_authorized_from_message(msg):
            logger.warning(
                "[Telegram] Blocked unauthorized user %s in chat %s",
                getattr(getattr(msg, "from_user", None), "id", None),
                getattr(getattr(msg, "chat", None), "id", None),
            )
            return
        if not self._should_process_message(msg):
            if self._should_observe_unmentioned_group_message(msg):
                self._observe_unmentioned_group_message(msg, MessageType.TEXT, update_id=update.update_id)
            return
        await self._ensure_forum_commands(update.message)

        event = self._build_message_event(msg, MessageType.TEXT, update_id=update.update_id)
        event.text = self._clean_bot_trigger_text(event.text)
        await self._cache_replied_media(msg, event)
        event = self._apply_telegram_group_observe_attribution(event)
        self._enqueue_text_event(event)

    async def _handle_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming command messages."""
        msg = self._effective_update_message(update)
        if not msg or not msg.text:
            return
        if not self._should_process_message(msg, is_command=True):
            return
        if not self._is_user_authorized_from_message(msg):
            logger.warning(
                "[Telegram] Blocked unauthorized user %s in chat %s",
                getattr(getattr(msg, "from_user", None), "id", None),
                getattr(getattr(msg, "chat", None), "id", None),
            )
            return
        await self._ensure_forum_commands(msg)

        event = self._build_message_event(msg, MessageType.COMMAND, update_id=update.update_id)
        event.text = self._clean_bot_trigger_text(event.text)
        await self._cache_replied_media(msg, event)
        event = self._apply_telegram_group_observe_attribution(event)
        await self.handle_message(event)

    async def _handle_location_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming location/venue pin messages."""
        msg = self._effective_update_message(update)
        if not msg:
            return
        if not self._is_user_authorized_from_message(msg):
            logger.warning(
                "[Telegram] Blocked unauthorized user %s in chat %s",
                getattr(getattr(msg, "from_user", None), "id", None),
                getattr(getattr(msg, "chat", None), "id", None),
            )
            return
        if not self._should_process_message(msg):
            if self._should_observe_unmentioned_group_message(msg):
                self._observe_unmentioned_group_message(msg, MessageType.LOCATION, update_id=update.update_id)
            return

        venue = getattr(msg, "venue", None)
        location = getattr(venue, "location", None) if venue else getattr(msg, "location", None)

        if not location:
            return

        lat = getattr(location, "latitude", None)
        lon = getattr(location, "longitude", None)
        if lat is None or lon is None:
            return

        # Build a text message with coordinates and context
        parts = ["[The user shared a location pin.]"]
        if venue:
            title = getattr(venue, "title", None)
            address = getattr(venue, "address", None)
            if title:
                parts.append(f"Venue: {title}")
            if address:
                parts.append(f"Address: {address}")
        parts.append(f"latitude: {lat}")
        parts.append(f"longitude: {lon}")
        parts.append(f"Map: https://www.google.com/maps/search/?api=1&query={lat},{lon}")
        parts.append("Ask what they'd like to find nearby (restaurants, cafes, etc.) and any preferences.")

        event = self._build_message_event(msg, MessageType.LOCATION, update_id=update.update_id)
        event.text = "\n".join(parts)
        event = self._apply_telegram_group_observe_attribution(event)
        await self.handle_message(event)

    # ------------------------------------------------------------------
    # Text message aggregation (handles Telegram client-side splits)
    # ------------------------------------------------------------------

    def _text_batch_key(self, event: MessageEvent) -> str:
        """Session-scoped key for text message batching.

        Applies the installed topic-recovery hook first so DM-topic batches
        coalesce on (and dispatch to) the recovered lane rather than the
        raw inbound ``message_thread_id`` Telegram may have attached.
        """
        from gateway.session import build_session_key
        self._apply_topic_recovery(event)
        return build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Buffer a text event and reset the flush timer.

        When Telegram splits a long user message into multiple updates,
        they arrive within a few hundred milliseconds.  This method
        concatenates them and waits for a short quiet period before
        dispatching the combined message.
        """
        if self._should_drop_delayed_delivery():
            logger.debug("[Telegram] Dropping text batch enqueue after disconnect started")
            return

        key = self._text_batch_key(event)
        existing = self._pending_text_batches.get(key)
        chunk_len = len(event.text or "")
        if existing is None:
            event._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            self._pending_text_batches[key] = event
        else:
            # Append text from the follow-up chunk
            if event.text:
                existing.text = f"{existing.text}\n{event.text}" if existing.text else event.text
            existing._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            # Merge any media that might be attached
            if event.media_urls:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)

        # Cancel any pending flush and restart the timer
        prior_task = self._pending_text_batch_tasks.get(key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_text_batch_tasks[key] = asyncio.create_task(
            self._flush_text_batch(key)
        )

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text.

        Uses a longer delay when the latest chunk is near Telegram's 4096-char
        split point, since a continuation chunk is almost certain.
        """
        current_task = asyncio.current_task()
        try:
            # Adaptive delay tiers:
            #  - last chunk ≥ _SPLIT_THRESHOLD: a continuation is almost
            #    certain → wait the longer split delay.
            #  - total accumulated text ≤ _TEXT_BATCH_FAST_LEN (~320 cp):
            #    short message → cap delay at _TEXT_BATCH_FAST_DELAY_S
            #    so the agent sees the text near-instantly.
            #  - total ≤ _TEXT_BATCH_SHORT_LEN (~1024 cp):
            #    medium → cap at _TEXT_BATCH_SHORT_DELAY_S.
            #  - otherwise: use the configured cap.
            # Tiers compose with operator overrides via the env-var-driven
            # ``_text_batch_delay_seconds`` (e.g. an operator who sets the
            # cap below 0.18s gets that lower number on every tier).
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
            await asyncio.sleep(delay)
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            if self._should_drop_delayed_delivery():
                logger.debug("[Telegram] Dropping text batch flush after disconnect started")
                return
            logger.info(
                "[Telegram] Flushing text batch %s (%d chars)",
                key, len(event.text or ""),
            )
            await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    # ------------------------------------------------------------------
    # Photo batching
    # ------------------------------------------------------------------

    def _photo_batch_key(self, event: MessageEvent, msg: Message) -> str:
        """Return a batching key for Telegram photos/albums."""
        from gateway.session import build_session_key
        session_key = build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )
        media_group_id = getattr(msg, "media_group_id", None)
        if media_group_id:
            return f"{session_key}:album:{media_group_id}"
        return f"{session_key}:photo-burst"

    async def _flush_photo_batch(self, batch_key: str) -> None:
        """Send a buffered photo burst/album as a single MessageEvent."""
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(self._media_batch_delay_seconds)
            event = self._pending_photo_batches.pop(batch_key, None)
            if not event:
                return
            if self._should_drop_delayed_delivery():
                logger.debug("[Telegram] Dropping photo batch flush after disconnect started")
                return
            logger.info("[Telegram] Flushing photo batch %s with %d image(s)", batch_key, len(event.media_urls))
            await self.handle_message(event)
        finally:
            if self._pending_photo_batch_tasks.get(batch_key) is current_task:
                self._pending_photo_batch_tasks.pop(batch_key, None)

    def _enqueue_photo_event(self, batch_key: str, event: MessageEvent) -> None:
        """Merge photo events into a pending batch and schedule flush."""
        if self._should_drop_delayed_delivery():
            logger.debug("[Telegram] Dropping photo batch enqueue after disconnect started")
            return

        existing = self._pending_photo_batches.get(batch_key)
        if existing is None:
            self._pending_photo_batches[batch_key] = event
        else:
            existing.media_urls.extend(event.media_urls)
            existing.media_types.extend(event.media_types)
            if event.text:
                existing.text = self._merge_caption(existing.text, event.text)

        prior_task = self._pending_photo_batch_tasks.get(batch_key)
        if prior_task and not prior_task.done():
            prior_task.cancel()

        self._pending_photo_batch_tasks[batch_key] = asyncio.create_task(self._flush_photo_batch(batch_key))

    async def _handle_media_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming media messages, downloading images to local cache."""
        if not update.message:
            return
        if not self._is_user_authorized_from_message(update.message):
            logger.info(
                "[Telegram] Blocked media from unauthorized user %s in chat %s",
                getattr(getattr(update.message, "from_user", None), "id", None),
                getattr(getattr(update.message, "chat", None), "id", None),
            )
            return
        if not self._should_process_message(update.message):
            if self._should_observe_unmentioned_group_message(update.message):
                _m = update.message
                _observe_type = self._media_message_type(_m)
                _event = self._build_message_event(_m, _observe_type, update_id=update.update_id)
                if _m.caption:
                    _event.text = self._clean_bot_trigger_text(_m.caption)
                await self._cache_observed_media(_m, _event)
                self._observe_unmentioned_group_message(
                    _m, _event.message_type, update_id=update.update_id, event=_event
                )
            return

        msg = update.message

        msg_type = self._media_message_type(msg)

        event = self._build_message_event(msg, msg_type, update_id=update.update_id)
        
        # Add caption as text
        if msg.caption:
            event.text = self._clean_bot_trigger_text(msg.caption)
        
        # Handle stickers: describe via vision tool with caching
        if msg.sticker:
            await self._handle_sticker(msg, event)
            event = self._apply_telegram_group_observe_attribution(event)
            await self.handle_message(event)
            return

        # Apply observe attribution after caption is set; sticker is handled above
        # because _handle_sticker overwrites event.text with its vision description.
        event = self._apply_telegram_group_observe_attribution(event)

        # Download photo to local image cache so the vision tool can access it
        # even after Telegram's ephemeral file URLs expire (~1 hour).
        if msg.photo:
            try:
                # msg.photo is a list of PhotoSize sorted by size; take the largest
                photo = msg.photo[-1]
                file_obj = await photo.get_file()
                # Download the image bytes directly into memory
                image_bytes = await file_obj.download_as_bytearray()
                # Determine extension from the file path if available
                ext = ".jpg"
                if file_obj.file_path:
                    for candidate in [".png", ".webp", ".gif", ".jpeg", ".jpg"]:
                        if file_obj.file_path.lower().endswith(candidate):
                            ext = candidate
                            break
                # Save to local cache (for vision tool access)
                cached_path = cache_image_from_bytes(bytes(image_bytes), ext=ext)
                event.media_urls = [cached_path]
                event.media_types = [f"image/{ext.lstrip('.')}" ]
                logger.info("[Telegram] Cached user photo at %s", cached_path)
                media_group_id = getattr(msg, "media_group_id", None)
                if media_group_id:
                    await self._queue_media_group_event(str(media_group_id), event)
                else:
                    batch_key = self._photo_batch_key(event, msg)
                    self._enqueue_photo_event(batch_key, event)
                return

            except Exception as e:
                logger.warning("[Telegram] Failed to cache photo: %s", e, exc_info=True)
                await self._surface_media_cache_failure(msg, event, "photo", e)

        # Download voice/audio messages to cache for STT transcription
        if msg.voice:
            try:
                allowed, note = self._telegram_media_size_allowed(msg.voice, "voice message")
                if not allowed:
                    event.text = self._append_observed_note(event.text, note or "")
                    logger.info("[Telegram] Skipped oversized user voice (size=%s)", getattr(msg.voice, "file_size", None))
                    await self.handle_message(event)
                    return
                file_obj = await msg.voice.get_file()
                audio_bytes = await file_obj.download_as_bytearray()
                cached_path = cache_audio_from_bytes(bytes(audio_bytes), ext=".ogg")
                event.media_urls = [cached_path]
                event.media_types = ["audio/ogg"]
                logger.info("[Telegram] Cached user voice at %s", cached_path)
            except Exception as e:
                logger.warning("[Telegram] Failed to cache voice: %s", e, exc_info=True)
                await self._surface_media_cache_failure(msg, event, "voice message", e)
        elif msg.audio:
            try:
                allowed, note = self._telegram_media_size_allowed(msg.audio, "audio file")
                if not allowed:
                    event.text = self._append_observed_note(event.text, note or "")
                    logger.info("[Telegram] Skipped oversized user audio (size=%s)", getattr(msg.audio, "file_size", None))
                    await self.handle_message(event)
                    return
                file_obj = await msg.audio.get_file()
                audio_bytes = await file_obj.download_as_bytearray()
                cached_path = cache_audio_from_bytes(bytes(audio_bytes), ext=".mp3")
                event.media_urls = [cached_path]
                event.media_types = ["audio/mp3"]
                logger.info("[Telegram] Cached user audio at %s", cached_path)
            except Exception as e:
                logger.warning("[Telegram] Failed to cache audio: %s", e, exc_info=True)
                await self._surface_media_cache_failure(msg, event, "audio file", e)

        elif msg.video:
            try:
                allowed, note = self._telegram_media_size_allowed(msg.video, "video file")
                if not allowed:
                    event.text = self._append_observed_note(event.text, note or "")
                    logger.info("[Telegram] Skipped oversized user video (size=%s)", getattr(msg.video, "file_size", None))
                    await self.handle_message(event)
                    return
                file_obj = await msg.video.get_file()
                video_bytes = await file_obj.download_as_bytearray()
                ext = ".mp4"
                if getattr(file_obj, "file_path", None):
                    for candidate in SUPPORTED_VIDEO_TYPES:
                        if file_obj.file_path.lower().endswith(candidate):
                            ext = candidate
                            break
                cached_path = cache_video_from_bytes(bytes(video_bytes), ext=ext)
                event.media_urls = [cached_path]
                event.media_types = [SUPPORTED_VIDEO_TYPES.get(ext, "video/mp4")]
                logger.info("[Telegram] Cached user video at %s", cached_path)
            except Exception as e:
                logger.warning("[Telegram] Failed to cache video: %s", e, exc_info=True)
                await self._surface_media_cache_failure(msg, event, "video file", e)

        # Download document files to cache for agent processing
        elif msg.document:
            doc = msg.document
            try:
                # Determine file extension
                ext = ""
                original_filename = doc.file_name or ""
                if original_filename:
                    _, ext = os.path.splitext(original_filename)
                    ext = ext.lower()

                # Normalize mime_type for robust comparisons (some clients send
                # uppercase like "IMAGE/PNG").
                doc_mime = (doc.mime_type or "").lower()

                # If no extension from filename, reverse-lookup from MIME type
                if not ext and doc_mime:
                    ext = _TELEGRAM_IMAGE_MIME_TO_EXT.get(doc_mime, "")
                    if not ext:
                        mime_to_ext = {v: k for k, v in SUPPORTED_DOCUMENT_TYPES.items()}
                        ext = mime_to_ext.get(doc_mime, "")

                # Check file size early so image documents cannot bypass the
                # document size limit by taking the image path.
                if not doc.file_size or doc.file_size > self._max_doc_bytes:
                    limit_mb = self._max_doc_bytes // (1024 * 1024)
                    event.text = (
                        "The document is too large or its size could not be verified. "
                        f"Maximum: {limit_mb} MB."
                    )
                    logger.info("[Telegram] Document too large: %s bytes", doc.file_size)
                    await self.handle_message(event)
                    return

                # Telegram may deliver screenshots/photos as documents. If the
                # payload is actually an image, route it through the image cache
                # and batching path instead of rejecting it as a document.
                if ext in _TELEGRAM_IMAGE_EXTENSIONS or doc_mime.startswith("image/"):
                    file_obj = await doc.get_file()
                    image_bytes = await file_obj.download_as_bytearray()
                    image_ext = ext if ext in _TELEGRAM_IMAGE_EXTENSIONS else _TELEGRAM_IMAGE_MIME_TO_EXT.get(doc_mime, ".jpg")
                    try:
                        cached_path = cache_image_from_bytes(bytes(image_bytes), ext=image_ext)
                    except ValueError as e:
                        logger.warning("[Telegram] Failed to cache image document: %s", e, exc_info=True)
                        event.text = (
                            f"Image document '{original_filename or doc_mime or ext or 'unknown'}' "
                            "could not be read as an image."
                        )
                        await self.handle_message(event)
                        return

                    event.message_type = MessageType.PHOTO
                    event.media_urls = [cached_path]
                    event.media_types = [doc_mime if doc_mime.startswith("image/") else _TELEGRAM_IMAGE_EXT_TO_MIME.get(image_ext, "image/jpeg")]
                    logger.info("[Telegram] Cached user image-document at %s", cached_path)

                    media_group_id = getattr(msg, "media_group_id", None)
                    if media_group_id:
                        await self._queue_media_group_event(str(media_group_id), event)
                    else:
                        batch_key = self._photo_batch_key(event, msg)
                        self._enqueue_photo_event(batch_key, event)
                    return

                if not ext and doc.mime_type:
                    video_mime_to_ext = {v: k for k, v in SUPPORTED_VIDEO_TYPES.items()}
                    ext = video_mime_to_ext.get(doc.mime_type, "")

                if not ext and doc.mime_type:
                    # SUPPORTED_IMAGE_DOCUMENT_TYPES has duplicate values (.jpg + .jpeg
                    # both map to image/jpeg); keep the first ext we encounter.
                    image_mime_to_ext: dict[str, str] = {}
                    for _ext, _mime in SUPPORTED_IMAGE_DOCUMENT_TYPES.items():
                        image_mime_to_ext.setdefault(_mime, _ext)
                    ext = image_mime_to_ext.get(doc.mime_type, "")

                if ext in SUPPORTED_VIDEO_TYPES:
                    file_obj = await doc.get_file()
                    video_bytes = await file_obj.download_as_bytearray()
                    cached_path = cache_video_from_bytes(bytes(video_bytes), ext=ext)
                    event.media_urls = [cached_path]
                    event.media_types = [SUPPORTED_VIDEO_TYPES[ext]]
                    event.message_type = MessageType.VIDEO
                    logger.info("[Telegram] Cached user video document at %s", cached_path)
                    await self.handle_message(event)
                    return

                # NOTE: image-document handling is performed earlier in this
                # function (ext in _TELEGRAM_IMAGE_EXTENSIONS or image/* mime),
                # which returns before reaching here.  Any subsequent
                # ext-in-SUPPORTED_IMAGE_DOCUMENT_TYPES branch would be dead
                # code — the extension sets are identical.

                # Download and cache. Any file type is accepted — authorization
                # to message the agent is the gate, not the file extension.
                # Known types keep their precise MIME; unknown types are tagged
                # application/octet-stream so the agent reaches for terminal tools.
                file_obj = await doc.get_file()
                doc_bytes = await file_obj.download_as_bytearray()
                raw_bytes = bytes(doc_bytes)
                cached_path = cache_document_from_bytes(raw_bytes, original_filename or f"document{ext or '.bin'}")
                mime_type = SUPPORTED_DOCUMENT_TYPES.get(ext) or doc.mime_type or "application/octet-stream"
                event.media_urls = [cached_path]
                event.media_types = [mime_type]
                logger.info("[Telegram] Cached user document at %s (%s)", cached_path, mime_type)

                # For text-readable files, inject content into event.text (capped
                # at 100 KB). Gate on a text-like extension/MIME — NOT a blind
                # UTF-8 decode, since binary formats (PDF/zip/docx) can have
                # decodable ASCII headers. Binary files are surfaced as a cached
                # path only (run.py emits a path-pointing context note).
                MAX_TEXT_INJECT_BYTES = 100 * 1024
                _is_text = ext in _TEXT_INJECT_EXTENSIONS or (doc_mime or "").startswith("text/")
                if _is_text and len(raw_bytes) <= MAX_TEXT_INJECT_BYTES:
                    try:
                        text_content = raw_bytes.decode("utf-8")
                        display_name = original_filename or f"document{ext or '.txt'}"
                        display_name = re.sub(r'[^\w.\- ]', '_', display_name)
                        injection = f"[Content of {display_name}]:\n{text_content}"
                        if event.text:
                            event.text = f"{injection}\n\n{event.text}"
                        else:
                            event.text = injection
                    except UnicodeDecodeError:
                        # Binary file — agent has the cached path and can use
                        # terminal/read_file against it. No inline injection.
                        pass

            except Exception as e:
                logger.warning("[Telegram] Failed to cache document: %s", e, exc_info=True)
                await self._surface_media_cache_failure(
                    msg, event, "attachment", e,
                    display_name=getattr(doc, "file_name", None) or None,
                )

        media_group_id = getattr(msg, "media_group_id", None)
        if media_group_id:
            await self._queue_media_group_event(str(media_group_id), event)
            return

        await self.handle_message(event)

    async def _queue_media_group_event(self, media_group_id: str, event: MessageEvent) -> None:
        """Buffer Telegram media-group items so albums arrive as one logical event.

        Telegram delivers albums as multiple updates with a shared media_group_id.
        If we forward each item immediately, the gateway thinks the second image is a
        new user message and interrupts the first. We debounce briefly and merge the
        attachments into a single MessageEvent.
        """
        if self._should_drop_delayed_delivery():
            logger.debug("[Telegram] Dropping media group enqueue after disconnect started")
            return

        existing = self._media_group_events.get(media_group_id)
        if existing is None:
            self._media_group_events[media_group_id] = event
        else:
            existing.media_urls.extend(event.media_urls)
            existing.media_types.extend(event.media_types)
            if event.text:
                existing.text = self._merge_caption(existing.text, event.text)

        prior_task = self._media_group_tasks.get(media_group_id)
        if prior_task:
            prior_task.cancel()

        self._media_group_tasks[media_group_id] = asyncio.create_task(
            self._flush_media_group_event(media_group_id)
        )

    async def _flush_media_group_event(self, media_group_id: str) -> None:
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(self.MEDIA_GROUP_WAIT_SECONDS)
            event = self._media_group_events.pop(media_group_id, None)
            if event is not None:
                if self._should_drop_delayed_delivery():
                    logger.debug("[Telegram] Dropping media group flush after disconnect started")
                    return
                await self.handle_message(event)
        except asyncio.CancelledError:
            return
        finally:
            if self._media_group_tasks.get(media_group_id) is current_task:
                self._media_group_tasks.pop(media_group_id, None)

    async def _handle_sticker(self, msg: Message, event: "MessageEvent") -> None:
        """
        Describe a Telegram sticker via vision analysis, with caching.

        For static stickers (WEBP), we download, analyze with vision, and cache
        the description by file_unique_id. For animated/video stickers, we inject
        a placeholder noting the emoji.
        """
        from gateway.sticker_cache import (
            get_cached_description,
            cache_sticker_description,
            build_sticker_injection,
            build_animated_sticker_injection,
            STICKER_VISION_PROMPT,
        )

        sticker = msg.sticker
        emoji = sticker.emoji or ""
        set_name = sticker.set_name or ""

        # Animated and video stickers can't be analyzed as static images
        if sticker.is_animated or sticker.is_video:
            event.text = build_animated_sticker_injection(emoji)
            return

        # Check the cache first
        cached = get_cached_description(sticker.file_unique_id)
        if cached:
            event.text = build_sticker_injection(
                cached["description"], cached.get("emoji", emoji), cached.get("set_name", set_name)
            )
            logger.info("[Telegram] Sticker cache hit: %s", sticker.file_unique_id)
            return

        # Cache miss -- download and analyze
        try:
            file_obj = await sticker.get_file()
            image_bytes = await file_obj.download_as_bytearray()
            cached_path = cache_image_from_bytes(bytes(image_bytes), ext=".webp")
            logger.info("[Telegram] Analyzing sticker at %s", cached_path)

            from tools.vision_tools import vision_analyze_tool
            result_json = await vision_analyze_tool(
                image_url=cached_path,
                user_prompt=STICKER_VISION_PROMPT,
            )
            result = json.loads(result_json)

            if result.get("success"):
                description = result.get("analysis", "a sticker")
                cache_sticker_description(sticker.file_unique_id, description, emoji, set_name)
                event.text = build_sticker_injection(description, emoji, set_name)
            else:
                # Vision failed -- use emoji as fallback
                event.text = build_sticker_injection(
                    f"a sticker with emoji {emoji}" if emoji else "a sticker",
                    emoji, set_name,
                )
        except Exception as e:
            logger.warning("[Telegram] Sticker analysis error: %s", e, exc_info=True)
            event.text = build_sticker_injection(
                f"a sticker with emoji {emoji}" if emoji else "a sticker",
                emoji, set_name,
            )

    def _reload_dm_topics_from_config(self) -> None:
        """Re-read dm_topics from config.yaml and load any new thread_ids into cache.

        This allows topics created externally (e.g. by the agent via API) to be
        recognized without a gateway restart.
        """
        try:
            from hermes_constants import get_hermes_home
            config_path = get_hermes_home() / "config.yaml"
            if not config_path.exists():
                return

            import yaml as _yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = _yaml.safe_load(f) or {}

            dm_topics = (
                config.get("platforms", {})
                .get("telegram", {})
                .get("extra", {})
                .get("dm_topics", [])
            )
            if not dm_topics:
                # Clear both config and precomputed set when all topics are removed
                self._dm_topics_config = []
                self._dm_topic_chat_ids = set()
                return

            # Update in-memory config and cache any new thread_ids
            self._dm_topics_config = dm_topics
            # Rebuild the chat_id set for O(1) root-DM ignore lookup
            self._dm_topic_chat_ids = {
                str(chat_entry["chat_id"]) for chat_entry in dm_topics if "chat_id" in chat_entry
            }
            for chat_entry in dm_topics:
                cid = chat_entry.get("chat_id")
                if not cid:
                    continue
                for t in chat_entry.get("topics", []):
                    tid = t.get("thread_id")
                    name = t.get("name")
                    if tid and name:
                        cache_key = f"{cid}:{name}"
                        if cache_key not in self._dm_topics:
                            self._dm_topics[cache_key] = int(tid)
                            logger.info(
                                "[%s] Hot-loaded DM topic from config: %s -> thread_id=%s",
                                self.name, cache_key, tid,
                            )
        except Exception as e:
            logger.debug("[%s] Failed to reload dm_topics from config: %s", self.name, e)

    def _get_dm_topic_info(self, chat_id: str, thread_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Look up DM topic config by chat_id and thread_id.

        Returns the topic config dict (name, skill, etc.) if this thread_id
        matches a known DM topic, or None.
        """
        if not thread_id:
            return None

        thread_id_int = int(thread_id)

        # Check cached topics first (created by us or loaded at startup)
        for key, cached_tid in self._dm_topics.items():
            if cached_tid == thread_id_int and key.startswith(f"{chat_id}:"):
                topic_name = key.split(":", 1)[1]
                # Find the full config for this topic
                for chat_entry in self._dm_topics_config:
                    if str(chat_entry.get("chat_id")) == chat_id:
                        for t in chat_entry.get("topics", []):
                            if t.get("name") == topic_name:
                                return t
                return {"name": topic_name}

        # Not in cache — hot-reload config in case topics were added externally
        self._reload_dm_topics_from_config()

        # Check cache again after reload
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

    def _cache_dm_topic_from_message(self, chat_id: str, thread_id: str, topic_name: str) -> None:
        """Cache a thread_id -> topic_name mapping discovered from an incoming message."""
        cache_key = f"{chat_id}:{topic_name}"
        if cache_key not in self._dm_topics:
            self._dm_topics[cache_key] = int(thread_id)
            logger.info(
                "[%s] Cached DM topic from message: %s -> thread_id=%s",
                self.name, cache_key, thread_id,
            )

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
            text = value.get("text")
            if text is not None:
                return cls._flatten_rich_inline_text(text)
            children = value.get("children")
            if children is not None:
                return cls._flatten_rich_inline_text(children)
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

            block_type = block.get("type")
            if block_type == "list":
                for item in block.get("items", []):
                    if not isinstance(item, dict):
                        continue
                    item_text = cls._flatten_rich_blocks(item.get("blocks"))
                    if not item_text:
                        continue
                    label = item.get("label")
                    item_lines = item_text.splitlines()
                    if not item_lines:
                        continue
                    first_line = item_lines[0]
                    if label:
                        first_line = f"{label} {first_line}".strip()
                    lines.append(first_line)
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
            api_kwargs = getattr(reply_to_message, "api_kwargs", None)
            getter = getattr(api_kwargs, "get", None)
            if not callable(getter):
                return None
            rich_message = getter("rich_message")
            rich_getter = getattr(rich_message, "get", None)
            if not callable(rich_getter):
                return None
            text = cls._flatten_rich_blocks(rich_getter("blocks")).strip()
            return text or None
        except Exception:
            return None

    def _build_message_event(
        self,
        message: Message,
        msg_type: MessageType,
        update_id: Optional[int] = None,
    ) -> MessageEvent:
        """Build a MessageEvent from a Telegram message.

        ``update_id`` is the ``Update.update_id`` from PTB; passing it through
        lets ``/restart`` record the triggering offset so the new gateway
        process can advance past it (prevents ``/restart`` being re-delivered
        when PTB's graceful-shutdown ACK fails).
        """
        chat = message.chat
        user = message.from_user
        
        # Determine chat type.  Normalize through ``str`` so tests/mocks and
        # python-telegram-bot enum values both work (``ChatType.CHANNEL`` is
        # string-like, but mocks often provide plain strings).
        telegram_chat_type = str(getattr(chat, "type", "")).split(".")[-1].lower()
        chat_type = "dm"
        if telegram_chat_type in {"group", "supergroup"}:
            chat_type = "group"
        elif telegram_chat_type == "channel":
            chat_type = "channel"

        # Resolve routable thread id for DM topics and forum group topics via
        # the shared normalizer, so gating and session routing agree on one
        # value. Only real topic/forum messages keep a thread id; ordinary
        # reply-UI anchors are dropped (they are not durable session threads
        # and sends against them hit 'Message thread not found', #3206), while
        # forum General-topic messages (message_thread_id=None) normalize to
        # the General-topic id so replies route back to General (#22423).
        thread_id_str = self._effective_message_thread_id(message)
        chat_topic = None
        topic_skill = None

        if chat_type == "dm" and thread_id_str:
            topic_info = self._get_dm_topic_info(str(chat.id), thread_id_str)
            if topic_info:
                chat_topic = topic_info.get("name")
                topic_skill = topic_info.get("skill")

            # Also check forum_topic_created service message for topic discovery
            if hasattr(message, "forum_topic_created") and message.forum_topic_created:
                created_name = message.forum_topic_created.name
                if created_name:
                    self._cache_dm_topic_from_message(str(chat.id), thread_id_str, created_name)
                    if not chat_topic:
                        chat_topic = created_name

        elif chat_type == "group" and thread_id_str:
            # Group/supergroup forum topic skill binding via config.extra['group_topics'].
            # Accept both supported shapes:
            #   [{"chat_id": "-100...", "topics": [...]}]
            # and legacy/operator-edited mapping shape:
            #   {"-100...": [{"thread_id": 12, ...}]}
            group_topics_config = self.config.extra.get("group_topics", [])
            if isinstance(group_topics_config, dict):
                group_topics_iter = [
                    {"chat_id": cfg_chat_id, "topics": topics}
                    for cfg_chat_id, topics in group_topics_config.items()
                ]
            elif isinstance(group_topics_config, list):
                group_topics_iter = [
                    entry for entry in group_topics_config if isinstance(entry, dict)
                ]
            else:
                group_topics_iter = []
            for chat_entry in group_topics_iter:
                if str(chat_entry.get("chat_id", "")) == str(chat.id):
                    topics = chat_entry.get("topics", [])
                    if not isinstance(topics, list):
                        topics = []
                    for topic in topics:
                        if not isinstance(topic, dict):
                            continue
                        tid = topic.get("thread_id")
                        if tid is not None and str(tid) == thread_id_str:
                            chat_topic = topic.get("name")
                            topic_skill = topic.get("skill")
                            break
                    break

        # Build source
        source = self.build_source(
            chat_id=str(chat.id),
            chat_name=chat.title or (chat.full_name if hasattr(chat, "full_name") else None),
            chat_type=chat_type,
            user_id=(
                str(user.id)
                if user
                else (str(chat.id) if chat_type in {"dm", "channel"} else None)
            ),
            user_name=(
                user.full_name
                if user
                else (
                    chat.full_name
                    if hasattr(chat, "full_name") and chat_type == "dm"
                    else (chat.title if chat_type == "channel" else None)
                )
            ),
            thread_id=thread_id_str,
            chat_topic=chat_topic,
            message_id=str(message.message_id),
            is_bot=bool(getattr(user, "is_bot", False)) if user else False,
        )
        
        # Extract reply context if this message is a reply.
        # Prefer Telegram's native partial quote (message.quote, TextQuote)
        # so a user replying to a single selected substring of a prior
        # multi-section message doesn't get the whole replied-to message
        # injected into the agent's context — which can cause the agent
        # to act on unrelated actionable-looking text the user didn't
        # quote (#22619). Fall back to the full replied-to message text
        # / caption when no native quote is present.
        reply_to_id = None
        reply_to_text = None
        if message.reply_to_message:
            reply_to_id = str(message.reply_to_message.message_id)
            quote = getattr(message, "quote", None)
            quote_text = getattr(quote, "text", None) if quote is not None else None
            if quote_text:
                reply_to_text = quote_text
            else:
                reply_to_text = (
                    message.reply_to_message.text
                    or message.reply_to_message.caption
                    or None
                )
                if not reply_to_text:
                    # Prefer Telegram's native rich-message echo when present;
                    # keep the local send-time index only as a fallback for
                    # older/unrecoverable reply payloads.
                    reply_to_text = self._extract_rich_reply_text(message.reply_to_message)
                if not reply_to_text:
                    try:
                        from gateway import rich_sent_store
                        reply_to_text = rich_sent_store.lookup(
                            str(chat.id), reply_to_id
                        )
                    except Exception:
                        reply_to_text = None

        # Per-channel/topic ephemeral prompt
        from gateway.platforms.base import resolve_channel_prompt
        _chat_id_str = str(chat.id)
        _channel_prompt = resolve_channel_prompt(
            self.config.extra,
            thread_id_str or _chat_id_str,
            _chat_id_str if thread_id_str else None,
        )

        return MessageEvent(
            text=message.text or "",
            message_type=msg_type,
            source=source,
            raw_message=message,
            message_id=str(message.message_id),
            platform_update_id=update_id,
            reply_to_message_id=reply_to_id,
            reply_to_text=reply_to_text,
            auto_skill=topic_skill,
            channel_prompt=_channel_prompt,
            timestamp=message.date,
        )

    # ── Message reactions (processing lifecycle) ──────────────────────────

    def _reactions_enabled(self) -> bool:
        """Check if message reactions are enabled via config/env."""
        return os.getenv("TELEGRAM_REACTIONS", "false").lower() not in {"false", "0", "no"}

    async def _set_reaction(self, chat_id: str, message_id: str, emoji: str) -> bool:
        """Set a single emoji reaction on a Telegram message."""
        if not self._bot:
            return False
        try:
            await self._bot.set_message_reaction(
                chat_id=normalize_telegram_chat_id(chat_id),
                message_id=int(message_id),
                reaction=emoji,
            )
            return True
        except Exception as e:
            logger.debug("[%s] set_message_reaction failed (%s): %s", self.name, emoji, e)
            return False

    async def _clear_reactions(self, chat_id: str, message_id: str) -> bool:
        """Clear all reactions from a Telegram message.

        Calling ``set_message_reaction`` with ``reaction=None`` (or an empty
        sequence) is the documented Bot API way to remove all bot-set
        reactions on a message — equivalent to Bot API 10.0's
        ``deleteMessageReaction`` but supported in PTB 22.6 already.
        """
        if not self._bot:
            return False
        try:
            await self._bot.set_message_reaction(
                chat_id=normalize_telegram_chat_id(chat_id),
                message_id=int(message_id),
                reaction=None,
            )
            return True
        except Exception as e:
            logger.debug("[%s] clear reactions failed: %s", self.name, e)
            return False

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add an in-progress reaction when message processing begins."""
        if not self._reactions_enabled():
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if chat_id and message_id:
            await self._set_reaction(chat_id, message_id, "\U0001f440")

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Swap the in-progress reaction for a final success/failure reaction.

        Unlike Discord (additive reactions), Telegram's set_message_reaction
        replaces all existing reactions in one call — no remove step needed.

        On CANCELLED outcomes (e.g. the user runs ``/stop``, or a session is
        interrupted mid-flight), we explicitly clear the 👀 in-progress
        reaction so it doesn't linger on the user's message indefinitely.
        Without this clear, the only way to remove the 👀 was to wait for
        another agent run to swap it to 👍/👎 — which never happens if the
        cancellation was the last activity in the chat.
        """
        if not self._reactions_enabled():
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if not (chat_id and message_id):
            return
        if outcome == ProcessingOutcome.CANCELLED:
            await self._clear_reactions(chat_id, message_id)
        else:
            await self._set_reaction(
                chat_id,
                message_id,
                "\U0001f44d" if outcome == ProcessingOutcome.SUCCESS else "\U0001f44e",
            )


# ──────────────────────────────────────────────────────────────────────────
# Plugin migration glue (#41112 / #3823)
#
# Added when the Telegram adapter (+ its telegram_network satellite) moved from
# gateway/platforms/ into this bundled plugin. Mirrors the Discord (#24356) /
# Slack migrations: a register(ctx) entry point plus hook implementations that
# replace the per-platform core touchpoints (the Platform.TELEGRAM branch in
# gateway/run.py, the telegram_cfg YAML→env/extra block in gateway/config.py,
# the _setup_telegram wizard + _PLATFORMS["telegram"] static dict in
# hermes_cli/{setup,gateway}.py, and the _send_telegram dispatch in
# tools/send_message_tool.py).  Telegram uses the generic token connected
# check, so no is_connected override is needed.
# ──────────────────────────────────────────────────────────────────────────


def _resolve_notifications_mode() -> str:
    """Resolve the Telegram notification mode (all/important) from env or
    config.yaml display.platforms.telegram.notifications, defaulting to
    'important'.  Mirrors the post-construction logic that used to live in
    gateway/run.py::_create_adapter()."""
    mode = os.getenv("HERMES_TELEGRAM_NOTIFICATIONS", "")
    if not mode:
        try:
            from gateway.config import load_gateway_config
            from gateway.run import cfg_get
            _gw_cfg = load_gateway_config()
            _raw = cfg_get(_gw_cfg, "display", "platforms", "telegram", "notifications")
            if _raw not in {None, ""}:
                mode = str(_raw).strip().lower()
        except Exception:
            pass
    mode = mode or "important"
    if mode not in {"all", "important"}:
        logger.warning(
            "Unknown telegram notifications mode '%s', defaulting to 'important' "
            "(valid: all, important)", mode,
        )
        mode = "important"
    return mode


def _build_adapter(config):
    """Factory wrapper that constructs TelegramAdapter and applies the
    notification mode (preserving the gateway/run.py post-construction step)."""
    adapter = TelegramAdapter(config)
    try:
        adapter._notifications_mode = _resolve_notifications_mode()
    except Exception:
        adapter._notifications_mode = "important"
    return adapter


def _is_connected(config) -> bool:
    """Telegram is connected when a bot token is configured.

    check_telegram_requirements() only verifies the python-telegram-bot SDK is
    importable, NOT that a token is set — so without this is_connected the
    registry-driven plugin-enable pass in gateway/config.py would enable
    Telegram on any machine that merely has the SDK installed. Gate on the
    token (env or PlatformConfig.token), matching the generic token check
    Telegram had as a built-in.
    """
    token = getattr(config, "token", None)
    if not token:
        import hermes_cli.gateway as gateway_mod
        token = gateway_mod.get_env_value("TELEGRAM_BOT_TOKEN") or ""
    return bool(str(token).strip())


async def _standalone_send(
    pconfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """Out-of-process Telegram delivery. Delegates to the standalone
    ``_send_telegram`` REST sender in tools/send_message_tool.py (which already
    handles chunking-agnostic single sends, threads, media, retries, and
    parse-mode fallback). Implements the standalone_sender_fn contract so
    deliver=telegram cron jobs succeed when cron runs separately from the
    gateway."""
    token = getattr(pconfig, "token", None) or os.getenv("TELEGRAM_BOT_TOKEN", "")
    disable_link_previews = bool(
        getattr(pconfig, "extra", {}) and pconfig.extra.get("disable_link_previews")
    )
    from tools.send_message_tool import _send_telegram
    return await _send_telegram(
        token,
        chat_id,
        message,
        media_files=media_files,
        thread_id=thread_id,
        disable_link_previews=disable_link_previews,
        force_document=force_document,
    )


def interactive_setup() -> None:
    """Configure Telegram bot credentials and allowlist.

    Delegates to the existing CLI setup helpers (managed-bot QR onboarding,
    token validation, allowlist capture) via lazy import so the full wizard
    behavior is preserved without duplicating ~150 lines. Replaces the
    _PLATFORMS["telegram"] static dict dispatch in hermes_cli/gateway.py.
    """
    from hermes_cli import setup as _setup_mod
    _setup_mod._setup_telegram()


def _apply_yaml_config(yaml_cfg: dict, telegram_cfg: dict) -> dict | None:
    """Translate config.yaml telegram: keys into TELEGRAM_* env vars and
    PlatformConfig.extra entries.

    Implements the apply_yaml_config_fn contract (#24849). Mirrors the legacy
    telegram_cfg block from gateway/config.py::load_gateway_config(). Env vars
    take precedence over YAML. Returns a dict of extras to merge into
    PlatformConfig.extra (disable_topic_auto_rename + runtime flags), or None.
    """
    import json as _json
    extras: dict = {}

    if "disable_topic_auto_rename" in telegram_cfg:
        extras.setdefault("disable_topic_auto_rename", telegram_cfg["disable_topic_auto_rename"])

    _effective_rm = telegram_cfg.get("require_mention", yaml_cfg.get("require_mention"))
    if _effective_rm is not None and not os.getenv("TELEGRAM_REQUIRE_MENTION"):
        os.environ["TELEGRAM_REQUIRE_MENTION"] = str(_effective_rm).lower()
    if "mention_patterns" in telegram_cfg and not os.getenv("TELEGRAM_MENTION_PATTERNS"):
        os.environ["TELEGRAM_MENTION_PATTERNS"] = _json.dumps(telegram_cfg["mention_patterns"])
    if "exclusive_bot_mentions" in telegram_cfg and not os.getenv("TELEGRAM_EXCLUSIVE_BOT_MENTIONS"):
        os.environ["TELEGRAM_EXCLUSIVE_BOT_MENTIONS"] = str(telegram_cfg["exclusive_bot_mentions"]).lower()
    if "allow_bots" in telegram_cfg and not os.getenv("TELEGRAM_ALLOW_BOTS"):
        os.environ["TELEGRAM_ALLOW_BOTS"] = str(telegram_cfg["allow_bots"]).lower()
    if "guest_mode" in telegram_cfg and not os.getenv("TELEGRAM_GUEST_MODE"):
        os.environ["TELEGRAM_GUEST_MODE"] = str(telegram_cfg["guest_mode"]).lower()
    if "observe_unmentioned_group_messages" in telegram_cfg and not os.getenv("TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES"):
        os.environ["TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES"] = str(telegram_cfg["observe_unmentioned_group_messages"]).lower()
    frc = telegram_cfg.get("free_response_chats")
    if frc is not None and not os.getenv("TELEGRAM_FREE_RESPONSE_CHATS"):
        if isinstance(frc, list):
            frc = ",".join(str(v) for v in frc)
        os.environ["TELEGRAM_FREE_RESPONSE_CHATS"] = str(frc)
    ac = telegram_cfg.get("allowed_chats")
    if ac is not None and not os.getenv("TELEGRAM_ALLOWED_CHATS"):
        if isinstance(ac, list):
            ac = ",".join(str(v) for v in ac)
        os.environ["TELEGRAM_ALLOWED_CHATS"] = str(ac)
    allowed_topics = telegram_cfg.get("allowed_topics")
    if allowed_topics is not None and not os.getenv("TELEGRAM_ALLOWED_TOPICS"):
        if isinstance(allowed_topics, list):
            allowed_topics = ",".join(str(v) for v in allowed_topics)
        os.environ["TELEGRAM_ALLOWED_TOPICS"] = str(allowed_topics)
    ignored_threads = telegram_cfg.get("ignored_threads")
    if ignored_threads is not None and not os.getenv("TELEGRAM_IGNORED_THREADS"):
        if isinstance(ignored_threads, list):
            ignored_threads = ",".join(str(v) for v in ignored_threads)
        os.environ["TELEGRAM_IGNORED_THREADS"] = str(ignored_threads)
    if "reactions" in telegram_cfg and not os.getenv("TELEGRAM_REACTIONS"):
        os.environ["TELEGRAM_REACTIONS"] = str(telegram_cfg["reactions"]).lower()
    if "proxy_url" in telegram_cfg and not os.getenv("TELEGRAM_PROXY"):
        os.environ["TELEGRAM_PROXY"] = str(telegram_cfg["proxy_url"]).strip()
    _telegram_extra = telegram_cfg.get("extra") if isinstance(telegram_cfg.get("extra"), dict) else {}
    _telegram_rtm = (
        telegram_cfg["reply_to_mode"] if "reply_to_mode" in telegram_cfg
        else _telegram_extra.get("reply_to_mode")
    )
    if _telegram_rtm is not None and not os.getenv("TELEGRAM_REPLY_TO_MODE"):
        _rtm_str = "off" if _telegram_rtm is False else str(_telegram_rtm).lower()
        os.environ["TELEGRAM_REPLY_TO_MODE"] = _rtm_str
    allowed_users = telegram_cfg.get("allow_from")
    if allowed_users is not None and not os.getenv("TELEGRAM_ALLOWED_USERS"):
        if isinstance(allowed_users, list):
            allowed_users = ",".join(str(v) for v in allowed_users)
        os.environ["TELEGRAM_ALLOWED_USERS"] = str(allowed_users)
    group_allowed_users = telegram_cfg.get("group_allow_from")
    if group_allowed_users is not None and not os.getenv("TELEGRAM_GROUP_ALLOWED_USERS"):
        if isinstance(group_allowed_users, list):
            group_allowed_users = ",".join(str(v) for v in group_allowed_users)
        os.environ["TELEGRAM_GROUP_ALLOWED_USERS"] = str(group_allowed_users)
    group_allowed_chats = telegram_cfg.get("group_allowed_chats")
    if group_allowed_chats is not None and not os.getenv("TELEGRAM_GROUP_ALLOWED_CHATS"):
        if isinstance(group_allowed_chats, list):
            group_allowed_chats = ",".join(str(v) for v in group_allowed_chats)
        os.environ["TELEGRAM_GROUP_ALLOWED_CHATS"] = str(group_allowed_chats)
    for _key in ("guest_mode", "disable_link_previews", "observe_unmentioned_group_messages"):
        if _key in telegram_cfg:
            extras.setdefault(_key, telegram_cfg[_key])
    # Pass through telegram-specific extra keys (e.g. base_url proxy override),
    # but EXCLUDE the generic shared-config keys that _merge_platform_map in
    # gateway/config.py already merges with correct top-level-over-nested
    # precedence. The apply_yaml_config_fn dispatch merges our return via
    # dict.update() (clobber), so re-emitting those generic keys here would
    # undo that precedence (top-level losing to a nested-fallback block).
    _GENERIC_MERGE_KEYS = {
        "reply_prefix", "reply_in_thread", "reply_to_mode",
        "unauthorized_dm_behavior", "notice_delivery", "require_mention",
        "channel_skill_bindings", "channel_prompts", "gateway_restart_notification",
        "allow_from", "allow_admin_from", "dm_policy", "group_policy",
    }
    for _k, _v in _telegram_extra.items():
        if _k not in _GENERIC_MERGE_KEYS:
            extras.setdefault(_k, _v)

    return extras or None


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="telegram",
        label="Telegram",
        adapter_factory=_build_adapter,
        check_fn=check_telegram_requirements,
        is_connected=_is_connected,
        required_env=["TELEGRAM_BOT_TOKEN"],
        install_hint="pip install 'hermes-agent[telegram]'",
        setup_fn=interactive_setup,
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="TELEGRAM_ALLOWED_USERS",
        allow_all_env="TELEGRAM_ALLOW_ALL_USERS",
        cron_deliver_env_var="TELEGRAM_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=4096,
        emoji="✈️",
        allow_update_command=True,
    )
