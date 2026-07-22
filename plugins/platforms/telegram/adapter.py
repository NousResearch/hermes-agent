import asyncio
import dataclasses
import faulthandler
import inspect
import json
import logging
import os
import html as _html
import re
import threading
import time
from contextvars import ContextVar
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


# Grace period after the wall-clock deadline fires: if the event loop still
# hasn't processed the expiry callback by then, the loop thread itself is
# blocked in a synchronous call â€” the exact state in which every asyncio-based
# timeout (including this helper's own expiry hand-off) goes silent, so the
# gateway hangs at "attempt 1/8" with no further output (#63309).
_LOOP_BLOCKED_DUMP_GRACE = 5.0


def _dump_loop_blocked_diagnostics(timeout: float, grace: float) -> None:
    """Emit diagnostics from the deadline timer thread when the loop is stuck.

    Runs OFF the event loop, so it works precisely when the loop cannot. The
    faulthandler dump names the frame the loop thread is blocked in â€” the one
    piece of information #63309-class hangs otherwise never surface.
    """
    logger.warning(
        "[Telegram] init deadline (%.0fs) expired but the event loop has not "
        "processed the expiry after a further %.0fs â€” the loop thread appears "
        "BLOCKED in a synchronous call, which is why no timeout fires (#63309). "
        "Dumping all thread stacks to stderr to identify the blocking frame.",
        timeout,
        grace,
    )
    try:
        faulthandler.dump_traceback(all_threads=True)
    except Exception:
        logger.debug("faulthandler traceback dump failed", exc_info=True)


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
    # Set the moment the loop actually runs the expiry callback (or the helper
    # exits normally). threading.Event so the watchdog thread can read it
    # without touching asyncio state from off-loop.
    loop_processed_expiry = threading.Event()

    def _mark_expired() -> None:
        loop_processed_expiry.set()
        if not deadline.done():
            deadline.set_result(None)

    def _expire_from_thread() -> None:
        loop.call_soon_threadsafe(_mark_expired)

    def _watchdog_check() -> None:
        # The deadline fired _LOOP_BLOCKED_DUMP_GRACE ago but the loop never
        # ran _mark_expired: the loop thread is stuck in a synchronous call.
        # Diagnose from this thread â€” the loop can't.
        if not loop_processed_expiry.is_set():
            _dump_loop_blocked_diagnostics(timeout, _LOOP_BLOCKED_DUMP_GRACE)

    timer = threading.Timer(max(timeout, 0.0), _expire_from_thread)
    timer.daemon = True
    timer.start()
    watchdog = threading.Timer(
        max(timeout, 0.0) + _LOOP_BLOCKED_DUMP_GRACE, _watchdog_check
    )
    watchdog.daemon = True
    watchdog.start()
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
        watchdog.cancel()
        # cancel() cannot stop a Timer whose callback is already running;
        # setting the event closes that race so a completed await can never
        # be misreported as a blocked loop.
        loop_processed_expiry.set()


async def _run_abandon_cleanup(on_abandon) -> None:
    """Run the abandonment cleanup coroutine, swallowing any failure.

    Wrapped so a cleanup that itself hangs or raises cannot surface as an
    unhandled task error or block anything â€” it is fully fire-and-forget.
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
    ``initialize()`` (the case this whole path exists for) may never have set â€”
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

from gateway.authz_mixin import _coerce_allow_set
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

def _coerce_duration_seconds(value: Any) -> Optional[int]:
    """Round a raw length to whole positive seconds, or None if unusable."""
    try:
        secs = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return secs if secs > 0 else None


def _probe_voice_duration_seconds(path: str) -> Optional[int]:
    """Best-effort audio length in whole seconds for outgoing voice/audio.

    Telegram only auto-derives a clip's duration from container metadata for
    short recordings; longer ones (roughly 5 min+) are sent with duration 0
    and render as ``0:00`` in the player. We read the length locally and pass
    it explicitly so the bubble shows the real time.

    Mirrors ``gateway.run._probe_audio_duration``: stdlib ``wave`` for WAV,
    then mutagen for OGG/Opus/MP3/M4A metadata, then an ``ffprobe`` fallback.
    All three are optional â€” when none can read the file we return ``None``
    and the caller omits ``duration``, falling back to Telegram's own
    (possibly absent) metadata, i.e. the prior behavior. Blocking (mutagen
    read + ffprobe subprocess), so call it via ``asyncio.to_thread``.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".wav":
        try:
            import wave

            with wave.open(path, "rb") as wf:
                rate = wf.getframerate() or 0
                if rate:
                    secs = _coerce_duration_seconds(wf.getnframes() / float(rate))
                    if secs is not None:
                        return secs
        except Exception:
    
            logger.debug("Silently suppressed exception", exc_info=True)
            pass

    try:
        import mutagen

        audio = mutagen.File(path)
        secs = _coerce_duration_seconds(
            getattr(getattr(audio, "info", None), "length", None)
        )
        if secs is not None:
            return secs
    except Exception:
    
        logger.debug("Silently suppressed exception", exc_info=True)
        pass

    try:
        import shutil
        import subprocess

        if shutil.which("ffprobe"):
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0:
                return _coerce_duration_seconds(proc.stdout.strip())
    except Exception:
    
        logger.debug("Silently suppressed exception", exc_info=True)
        pass

    return None


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
    # Remove standard markdown bold (**text** â†’ text) BEFORE MarkdownV2 bold
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    # Remove MarkdownV2 bold markers that format_message converted from **bold**
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
    # Remove MarkdownV2 italic markers that format_message converted from *italic*
    # Use word boundary (\b) to avoid breaking snake_case like my_variable_name
    cleaned = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', cleaned)
    # Remove MarkdownV2 strikethrough markers (~text~ â†’ text)
    cleaned = re.sub(r'~([^~]+)~', r'\1', cleaned)
    # Remove MarkdownV2 spoiler markers (||text|| â†’ text)
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
# Markdown table â†’ Telegram-friendly row groups
# ---------------------------------------------------------------------------
# Telegram's MarkdownV2 has no table syntax â€” '|' is just an escaped literal,
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
    Bot API 10.1 ``sendRichMessage`` collapses multi-line content â€” e.g.
    slash-command lists joined with ``"\\n".join(lines)`` â€” into a single
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
# shutdown()/initialize() on the getUpdates httpx request close and rebuild the
# connection pool. When a connection is wedged on a stale CLOSE-WAIT socket that
# close can block forever, hanging _drain_polling_connections() and freezing the
# whole reconnect ladder (the tracked _polling_error_task never completes, so
# every escalation path stays gated behind its in-flight guard). Bound the drain
# so the ladder always advances toward the fatal-restart escalation. Matches
# _UPDATER_STOP_TIMEOUT. Refs: NousResearch/hermes-agent#66377
_DRAIN_TIMEOUT = 15.0
# Cause-agnostic wedged-recovery watchdog (#66377). Every recovery path (the
# reconnect ladder's re-entry, the pending-update probe, PTB's error callback)
# gates new recovery on ``_polling_error_task.done()``; if that task ever wedges
# on a hung await that no local bound covers, the whole gateway goes silently
# deaf with nothing retrying. The heartbeat loop force-escalates a recovery task
# that stays in-flight far longer than any healthy ladder attempt could take â€”
# stop (_UPDATER_STOP_TIMEOUT) + drain (2x_DRAIN_TIMEOUT) + start
# (_UPDATER_START_TIMEOUT) + max backoff (60s) is ~135s, so 300s is
# unambiguously stuck.
_POLLING_ERROR_TASK_STUCK_TIMEOUT = 300.0
# A generation is not healthy until the dedicated getUpdates request returns
# successfully. This exceeds a normal long-poll cycle for healthy idle bots.
_POLLING_PROGRESS_TIMEOUT = 60.0
_POLLING_GENERATION_CONTEXT: ContextVar[Optional[int]] = ContextVar(
    "telegram_polling_generation", default=None
)


class _PollingLifecycleAbort(RuntimeError):
    """Internal control flow for polling startup fenced by teardown."""


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
    # edit and the final edit, skipping the plain-text â†’ MarkdownV2 conversion.
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

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.TELEGRAM)
        self._app: Optional[Application] = None
        self._bot: Optional[Bot] = None
        self._webhook_mode: bool = False
        self._mention_patterns = self._compile_mention_patterns()
        self._reply_to_mode: str = getattr(config, 'reply_to_mode', 'first') or 'first'
        self._disable_link_previews: bool = self._coerce_bool_extra("disable_link_previews", False)
        # Bot API 10.1 Rich Messages: render constructs the legacy MarkdownV2
        # path degrades (tables â†’ bullet lists, task lists, <details>, block
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
        # #69444: Configurable cutoff for long Rich Messages.
        # Telegram API allows 32,768 chars, but some clients render partially
        # when documents are long and block-heavy. Legacy MarkdownV2 chunking
        # (4,096 chars) is more reliable for long documents.
        self._rich_message_max_chars: int = self._coerce_int_extra(
            "rich_message_max_chars", self.RICH_MESSAGE_MAX_CHARS
        )
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
        )

        # Batching/Ingress
        # Delay before processing the first text message in a batch, to allow
        # related media or further text to arrive (particularly for pasted or
        # bulk-forwarded content).
        self._text_batch_delay_seconds: float = self._coerce_float_extra(
            "text_batch_delay_seconds",
            0.6,
            min_value=0.0,
            max_value=5.0,
        )
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}
        self._drop_delayed_deliveries = False
        self._polling_error_task: Optional[asyncio.Task] = None
        self._polling_conflict_count: int = 0
        self._polling_network_error_count: int = 0
        self._polling_generation: int = 0
        self._polling_progress_event = asyncio.Event()
        self._polling_progress_accepting: bool = False
        self._polling_progress_verifier_task: Optional[asyncio.Task] = None
        self._polling_teardown_started: bool = False
        self._polling_error_callback_ref = None
        self._polling_heartbeat_thread: Optional[threading.Thread] = None
        # Consecutive heartbeat probes that saw queued updates the running
        # poller is not consuming. get_me() can't see this â€” the send path is
        # healthy while the getUpdates consumer is wedged â€” so the heartbeat
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
        # A polling generation stays degraded until the dedicated getUpdates
        # request makes successful progress. start_polling() return and getMe()
        # success on the general request path are not polling-health signals.
        # While True, send() short-circuits to a failure so callers
        # (cron live-adapter branch) fall through to standalone delivery.
        self._send_path_degraded: bool = False

    async def initialize(self) -> None:
        """Start the Telegram bot and register message handlers."""
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot not installed")

        self._app = Application.builder().token(self.config.token).build()
        self._bot = self._app.bot
        
        # Add handlers
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("help", self._handle_help))
        self._app.add_handler(
            TelegramMessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        self._app.add_handler(CallbackQueryHandler(self._handle_callback_query))

        await self._app.initialize()
        await self._app.start()
        
        logger.info("[%s] Telegram adapter initialized", self.name)

    async def shutdown(self) -> None:
        """Stop the Telegram bot and release resources."""
        if self._app:
            await self._app.stop()
            await self._app.shutdown()
            self._app = None
            self._bot = None
        logger.info("[%s] Telegram adapter shut down", self.name)

    def _content_fits_rich_limits(self, content: str) -> bool:
        """Cheap pre-check for the one hard rich limit we can count locally.

        Only the 32,768 UTF-8 character text cap is enforced here. Other Bot API
        rich limits (500 blocks, 16 nesting levels, 20 table columns, ...) are
        not pre-counted; if exceeded Telegram returns a BadRequest, which
        :meth:`_is_rich_fallback_error` classifies as permanent so the send
        degrades to the legacy chunking path.
        """
        return len(content) <= self._rich_message_max_chars

    async def send(
        self,
        chat_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message to a Telegram chat."""
        if not self._bot:
            return SendResult(success=False, error="Bot not initialized")

        try:
            # Check for rich message eligibility
            if self._should_attempt_rich(content, metadata):
                try:
                    return await self._send_rich(chat_id, content, metadata)
                except Exception as e:
                    logger.debug("[%s] Rich send failed, falling back: %s", self.name, e)

            # Standard delivery path
            message = await self._bot.send_message(
                chat_id=chat_id,
                text=self.format_message(content),
                parse_mode=ParseMode.MARKDOWN_V2,
            )
            return SendResult(success=True, message_id=str(message.message_id))
        except Exception as e:
            logger.error("[%s] Failed to send message: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    def _should_attempt_rich(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Determine if rich delivery should be attempted for the content."""
        if not self._rich_messages_enabled or self._rich_send_disabled:
            return False
        
        # Check if message requires rich rendering and fits limits
        return (
            self._needs_rich_rendering(content) and 
            self._content_fits_rich_limits(content)
        )

    def _needs_rich_rendering(self, content: str) -> bool:
        """Return True for markdown constructs that the legacy path degrades."""
        if not content:
            return False
        # Table detection
        if any(_TABLE_SEPARATOR_RE.match(line) for line in content.splitlines()):
            return True
        # GFM task list detection
        if re.search(r"(?m)^\s*[-*]\s+\[[ xX]\]\s+", content):
            return True
        # Collapsible details detection
        if re.search(r"(?m)^<details\b|^</details>|^<summary\b|^</summary>", content):
            return True
        # Block math detection
        if "$$" in content:
            return True
        return False

    async def _send_rich(
        self, chat_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Issue a sendRichMessage call via the bot's custom request path."""
        # Bot API 10.1 Rich Messages newline normalization
        text = _rich_normalize_linebreaks(content)
        
        # Note: In a real implementation, this would call the custom endpoint.
        # This mock demonstrates the logic flow.
        message = await self._bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=None, # Rich rendering handles formatting
        )
        return SendResult(success=True, message_id=str(message.message_id))

    def format_message(self, content: str) -> str:
        """Convert agent markdown to Telegram MarkdownV2."""
        # Wrap tables in bullets for legacy rendering
        text = _wrap_markdown_tables(content)
        # Escape for MarkdownV2
        return _escape_mdv2(text)

    # ... remaining Telegram handlers and helpers ...
