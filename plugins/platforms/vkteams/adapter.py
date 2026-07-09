"""VK Teams platform adapter (Hermes plugin).

Talks to the VK Teams Bot API (https://teams.vk.com/botapi/) — the
Telegram-like HTTP bot API descended from the ICQ New / Mail.ru Agent
bot API.  Receives updates via long polling (``GET /events/get``; the
API has no webhooks) and sends via plain HTTP calls.  No external SDK —
only httpx, which is already a Hermes dependency.

This adapter ships as a Hermes platform plugin under
``plugins/platforms/vkteams/``.  The Hermes plugin loader scans the
directory at startup, calls :func:`register`, and the platform becomes
available to ``gateway/run.py`` and ``tools/send_message_tool`` through
the registry — no edits to core files required.

Configuration in config.yaml::

    platforms:
      vkteams:
        enabled: true
        extra:
          token: "001.0123456789.0123456789:1000000"
          api_base: "https://myteam.mail.ru/bot/v1"   # or on-premise URL
          parse_mode: "HTML"                          # recommended; MarkdownV2 | none
          poll_time: 25

Environment variables (env wins over config.yaml ``extra``):

    VKTEAMS_BOT_TOKEN          Bot token from Metabot /newbot (required)
    VKTEAMS_API_BASE           API base URL. SaaS default:
                               https://myteam.mail.ru/bot/v1. Every
                               on-premise installation has its own —
                               send /start to Metabot to discover it.
    VKTEAMS_PARSE_MODE         MarkdownV2 (code default) / HTML (recommended) /
                               none. Prefer HTML: VK Teams' MarkdownV2 parser
                               rejects valid inline code / lone ``_ * ~`` with
                               "Format error"; HTML escapes only ``& < >``.
    VKTEAMS_POLL_TIME          Long-poll hold time in seconds (default 25)
    VKTEAMS_ALLOWED_USERS      Comma-separated allowlist of user IDs/emails
    VKTEAMS_ALLOW_ALL_USERS    Allow any user — dev only
    VKTEAMS_HOME_CHANNEL       Default chat for cron / notification delivery
    VKTEAMS_HOME_CHANNEL_NAME  Human label for the home channel

Platform quirks this adapter accounts for:

  * The bot token travels as a ``token`` query parameter in EVERY
    request, so it can leak through proxy logs and exception messages
    that embed the request URL.  All log output is routed through
    :func:`_redact_token`.
  * Rate limits: ~30 msg/s to private dialogs but only 1 msg/s into any
    single group chat.  Mid-stream ``edit_message`` calls are throttled
    per chat (group chat IDs end with ``@chat.agent``) and the
    ``Retry-After`` header on ``ratelimit`` errors is honored.
  * No per-message length limit is documented — only a 60 KB
    query-string cap.  A conservative 4096-char limit (matching the
    Telegram-family lineage) keeps requests well under it.
  * A bot cannot initiate a dialog: the user must message it first (or
    the bot must be a member of the target group).  Cron delivery to a
    cold chat fails server-side — document VKTEAMS_HOME_CHANNEL
    accordingly.
  * No threads, reactions, or media albums in the Bot API.
"""

import asyncio
import json
import logging
import mimetypes
import os
import re
import time
from datetime import datetime, timezone
from itertools import count as _itercount
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
    cache_image_from_url,
    cache_video_from_bytes,
)

logger = logging.getLogger(__name__)


DEFAULT_API_BASE = "https://myteam.mail.ru/bot/v1"
# No documented per-message cap; 4096 matches the Telegram-family lineage
# and keeps GET query strings far below the API's 60 KB request cap.
MAX_MESSAGE_LENGTH = 4096
DEFAULT_POLL_TIME_S = 25
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 1000
# Group/channel chat IDs look like "685000000@chat.agent"; private
# counterparts are bare emails or numeric IDs (documented chatId format).
GROUP_CHAT_SUFFIX = "@chat.agent"
# Group chats allow only 1 msg/s; DMs allow 30/s. Mid-stream edits are
# throttled below these so streaming can't trip "ratelimit" errors.
GROUP_EDIT_MIN_INTERVAL_S = 1.1
DM_EDIT_MIN_INTERVAL_S = 0.35
# sendActions "typing" expires after 10s server-side; base _keep_typing
# refreshes every 2s — resending every tick is pure waste, so the adapter
# only re-sends after this many seconds.
TYPING_RESEND_INTERVAL_S = 6.0
# Only these containers render as a playable voice bubble via sendVoice.
VOICE_EXTS = {".aac", ".ogg", ".m4a"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # documented request-body cap
# Inline-button prompts resolved by typing (/approve, /cancel) never tap a
# button, so their adapter-side state is never popped. Evict entries older
# than this and cap the dicts so a long-lived gateway can't leak them.
PROMPT_STATE_TTL_SECONDS = 3600
PROMPT_STATE_MAX_SIZE = 512


class _FatalPollError(Exception):
    """Raised when the event loop hit an unrecoverable error (bad token)."""


def _redact_token(text: str, token: str) -> str:
    """Mask the bot token in *text* (URLs in httpx errors embed it)."""
    if not text:
        return text
    if token:
        text = text.replace(token, "***")
    # Defense in depth for tokens that differ from the configured one
    # (e.g. a URL captured before a config reload).
    return re.sub(r"token=[^&\s'\"]+", "token=***", text)


# Characters VK Teams MarkdownV2 treats as style delimiters — the only ones
# that may carry a backslash escape.  This is DELIBERATELY narrower than
# Telegram's set: VK Teams' parser rejects an escape before a character it
# does not consider special (e.g. ``\.`` ``\-`` ``\!``) and fails the WHOLE
# message with "Format error".  Telegram requires escaping ``. ! - = + # | { }``;
# VK Teams does not — and NOT escaping ``-`` / ``.`` also lets line-start list
# markers (``- item`` / ``1. item``) render as real lists.  The set below is
# exactly the inline style delimiters from the VK Teams tutorial
# (https://teams.vk.com/botapi/tutorial/#Text_Format): bold ``*``, italic /
# underline ``_``, strikethrough ``~``, code `` ` ``, link ``[ ] ( )``, plus
# the escape character ``\`` itself.
_MDV2_ESCAPE_RE = re.compile(r'([\\_*~`\[\]()])')


def _escape_mdv2(text: str) -> str:
    """Escape VK Teams MarkdownV2 style-delimiter characters."""
    return _MDV2_ESCAPE_RE.sub(r'\\\1', text)


def _strip_mdv2(text: str) -> str:
    """Strip MarkdownV2 escapes/markers to produce clean plain text.

    Used as the fallback when the server rejects formatted text, so the
    user never sees stray escape backslashes or marker characters.
    """
    cleaned = re.sub(r'\\([_*\[\]()~`>#\+\-=|{}.!\\])', r'\1', text)
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
    # Word boundaries avoid breaking snake_case identifiers.
    cleaned = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', cleaned)
    cleaned = re.sub(r'~([^~]+)~', r'\1', cleaned)
    return cleaned


def _escape_html(text: str) -> str:
    """Escape the only three characters special to VK Teams HTML mode.

    Unlike MarkdownV2 (which has a large, under-documented escape set and
    rejects lone style delimiters even inside code), HTML mode is robust:
    ``&`` must be escaped first so the ``<``/``>`` replacements don't double
    up on the entity ampersands.
    """
    return (
        text.replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
    )


_HTML_TAG_RE = re.compile(r'<[^>]+>')


def _strip_html(text: str) -> str:
    """Reduce VK Teams HTML back to clean plain text for the fallback path.

    Removes tags, then un-escapes entities (``&amp;`` last so a literal
    ``&lt;`` in the source doesn't get turned into a tag-looking ``<``).
    """
    cleaned = _HTML_TAG_RE.sub('', text)
    cleaned = (
        cleaned.replace('&lt;', '<')
               .replace('&gt;', '>')
               .replace('&quot;', '"')
               .replace('&#39;', "'")
               .replace('&amp;', '&')
    )
    return cleaned


_TABLE_ROW_RE = re.compile(r'^\s*\|.+\|\s*$')
_TABLE_SEPARATOR_RE = re.compile(r'^\s*\|(?:\s*:?-+:?\s*\|)+\s*$')


def _wrap_markdown_tables(text: str) -> str:
    """Wrap GFM pipe tables in code fences so columns stay aligned.

    VK Teams MarkdownV2 has no table construct; left inline, pipe rows
    get their ``|`` and ``-`` characters escaped into unreadable soup.
    A monospace block preserves the columnar layout.
    """
    lines = text.split('\n')
    out: List[str] = []
    i = 0
    while i < len(lines):
        if (
            i + 1 < len(lines)
            and _TABLE_ROW_RE.match(lines[i])
            and _TABLE_SEPARATOR_RE.match(lines[i + 1])
        ):
            block = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                block.append(lines[i])
                i += 1
            out.append('```')
            out.extend(block)
            out.append('```')
            continue
        out.append(lines[i])
        i += 1
    return '\n'.join(out)


def _guess_mime(filename: str, api_type: str = "") -> str:
    """Best-effort MIME type from a filename, falling back to the API's
    coarse ``type`` field (image/video/audio)."""
    mime, _ = mimetypes.guess_type(filename or "")
    if mime:
        return mime
    return {
        "image": "image/jpeg",
        "video": "video/mp4",
        "audio": "audio/ogg",
    }.get(api_type, "application/octet-stream")


def _resolve_token(config: Optional[PlatformConfig]) -> str:
    """Token resolution order: config.yaml extra > PlatformConfig.token > env."""
    extra = getattr(config, "extra", {}) or {}
    return (
        extra.get("token")
        or getattr(config, "token", None)
        or os.getenv("VKTEAMS_BOT_TOKEN", "")
    ).strip()


def _resolve_api_base(config: Optional[PlatformConfig]) -> str:
    extra = getattr(config, "extra", {}) or {}
    return (
        extra.get("api_base")
        or os.getenv("VKTEAMS_API_BASE", "")
        or DEFAULT_API_BASE
    ).strip().rstrip("/")


def check_requirements() -> bool:
    """httpx is the only dependency (already shipped with Hermes)."""
    return HTTPX_AVAILABLE


def validate_config(config) -> bool:
    """Validate that the configured VK Teams platform has a token set."""
    return bool(_resolve_token(config))


def is_connected(config) -> bool:
    """Check whether VK Teams is configured (env or config.yaml)."""
    return bool(_resolve_token(config))


class VKTeamsAdapter(BasePlatformAdapter):
    """VK Teams Bot API adapter.

    Long-polls ``/events/get`` for updates and sends via the query-string
    HTTP API (multipart POST for file uploads). No external SDK — httpx only.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    supports_code_blocks = True
    # send() chunks oversized content itself via truncate_message(), so the
    # DeliveryRouter must not pre-truncate.
    splits_long_messages = True

    def __init__(self, config: PlatformConfig):
        platform = Platform("vkteams")
        super().__init__(config=config, platform=platform)

        extra = config.extra or {}
        self._token: str = _resolve_token(config)
        self._api_base: str = _resolve_api_base(config)
        self._parse_mode: str = (
            extra.get("parse_mode")
            or os.getenv("VKTEAMS_PARSE_MODE", "MarkdownV2")
        ).strip()
        if self._parse_mode.lower() in ("", "none", "plain"):
            self._parse_mode = ""
        try:
            self._poll_time: int = int(
                extra.get("poll_time")
                or os.getenv("VKTEAMS_POLL_TIME", DEFAULT_POLL_TIME_S)
            )
        except (TypeError, ValueError):
            self._poll_time = DEFAULT_POLL_TIME_S
        self._poll_time = max(1, min(self._poll_time, 60))

        # Callback-button authorization mirrors the gateway's inbound
        # allowlist (merged from the same env vars via authz_mixin).
        allow_all = os.getenv("VKTEAMS_ALLOW_ALL_USERS", "").strip().lower()
        self._allow_all_users: bool = allow_all in ("1", "true", "yes")
        raw_allow = (
            os.getenv("VKTEAMS_ALLOWED_USERS", "")
            or ",".join(extra.get("allowed_users", []) or [])
        )
        self._allowed_users = {
            u.strip().lower() for u in raw_allow.split(",") if u.strip()
        }

        self._poll_task: Optional[asyncio.Task] = None
        self._http_client: Optional["httpx.AsyncClient"] = None
        # /events/get cursor. The server never re-delivers events below it,
        # so it must survive reconnects and NEVER reset back to 0.
        self._last_event_id: int = 0
        self._bot_user_id: Optional[str] = None

        # Message deduplication: msg_id -> timestamp
        self._seen_messages: Dict[str, float] = {}

        # Inline-button state: short id -> (session_key, created_monotonic).
        # See telegram plugin for the shared ea:/sc:/cl: callback convention.
        # The timestamp drives TTL eviction (prompts answered by text never
        # tap a button, so entries would otherwise live forever).
        self._approval_counter = _itercount(1)
        self._approval_state: Dict[int, Tuple[str, float]] = {}
        self._slash_confirm_state: Dict[str, Tuple[str, float]] = {}
        self._clarify_state: Dict[str, Tuple[str, float]] = {}

        # Mid-stream edit throttle state: chat_id -> monotonic timestamp.
        self._last_edit_at: Dict[str, float] = {}
        # Saturated-overflow preview dedup: (chat_id, message_id) -> text.
        self._last_overflow_preview: Dict[Tuple[str, str], str] = {}
        # Typing resend throttle: chat_id -> monotonic timestamp.
        self._last_typing_at: Dict[str, float] = {}

    # -- Small helpers --------------------------------------------------------

    def _redact(self, text: Any) -> str:
        return _redact_token(str(text), self._token)

    @staticmethod
    def _is_group_chat(chat_id: str) -> bool:
        return str(chat_id).endswith(GROUP_CHAT_SUFFIX)

    def _should_reply(self, chunk_index: int) -> bool:
        mode = getattr(self.config, "reply_to_mode", "first")
        if mode == "off":
            return False
        if mode == "all":
            return True
        return chunk_index == 0

    @staticmethod
    def _prune_prompt_state(state: Dict[Any, Tuple[str, float]]) -> None:
        """Drop timed-out (and, if oversized, oldest) prompt-state entries.

        Prompts resolved by typing ``/approve`` etc. never tap a button, so
        their entries are never popped by the callback handler.  Pruning on
        every new registration keeps these dicts bounded.
        """
        now = time.monotonic()
        for key in [k for k, (_, ts) in state.items() if now - ts > PROMPT_STATE_TTL_SECONDS]:
            state.pop(key, None)
        if len(state) > PROMPT_STATE_MAX_SIZE:
            for key in sorted(state, key=lambda k: state[k][1])[: len(state) - PROMPT_STATE_MAX_SIZE]:
                state.pop(key, None)

    def _register_prompt_state(
        self, state: Dict[Any, Tuple[str, float]], key: Any, session_key: str,
    ) -> None:
        state[key] = (session_key, time.monotonic())
        self._prune_prompt_state(state)

    # -- Low-level API client --------------------------------------------------

    async def _api_get(
        self,
        method: str,
        params: Dict[str, Any],
        *,
        timeout: float = 20.0,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[float]]:
        """Call a Bot API method with query params.

        Returns ``(data, error, retry_after)``.  ``data`` is the parsed
        JSON body when the server answered ``ok: true``; otherwise
        ``error`` carries a redacted human-readable description and
        ``retry_after`` the server-requested delay (rate limits only).
        """
        if not self._http_client:
            return None, "HTTP client not initialized", None
        url = f"{self._api_base}/{method}"
        query = {"token": self._token}
        for key, value in params.items():
            if value is not None:
                query[key] = value
        try:
            resp = await self._http_client.get(url, params=query, timeout=timeout)
        except httpx.TimeoutException:
            return None, f"timeout calling {method}", None
        except Exception as e:
            return None, self._redact(e), None

        retry_after: Optional[float] = None
        try:
            retry_after = float(resp.headers.get("Retry-After", ""))
        except (TypeError, ValueError):
            retry_after = None

        if resp.status_code == 401:
            return None, "unauthorized (401) — check VKTEAMS_BOT_TOKEN", None
        if resp.status_code >= 300:
            return (
                None,
                f"HTTP {resp.status_code} from {method}: {self._redact(resp.text[:200])}",
                retry_after,
            )
        try:
            data = resp.json()
        except Exception:
            return None, f"non-JSON response from {method}", None
        if not isinstance(data, dict):
            return None, f"unexpected response shape from {method}", None
        if data.get("ok") is False:
            description = str(data.get("description") or "unknown error")
            return None, self._redact(description), retry_after
        return data, None, retry_after

    @staticmethod
    def _classify_error(error: str) -> Tuple[bool, str]:
        """Map a VK Teams error description to (retryable, error_kind).

        The API has no documented error-code taxonomy — only
        ``{ok: false, description}`` — so classification is heuristic
        string matching (hardened empirically against a live server).
        """
        lowered = (error or "").lower()
        if "ratelimit" in lowered or "rate limit" in lowered or "too many" in lowered:
            return True, "rate_limited"
        if "timeout" in lowered or "network" in lowered or "connect" in lowered \
                or "disconnected" in lowered or "temporarily" in lowered:
            return True, "transient"
        if "unauthorized" in lowered or "permission" in lowered or "forbidden" in lowered:
            return False, "forbidden"
        if "not found" in lowered or "chat not found" in lowered:
            return False, "not_found"
        if "format" in lowered or "parse" in lowered or "markdown" in lowered:
            return False, "bad_format"
        if "too long" in lowered or "length" in lowered:
            return False, "too_long"
        return False, "unknown"

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Validate the token and start the long-poll loop.

        The server keeps undelivered events queued against the
        ``lastEventId`` cursor, so reconnects (``is_reconnect=True``)
        naturally resume from where the previous poll stopped — the
        cursor is intentionally NOT reset here.
        """
        if not HTTPX_AVAILABLE:
            logger.warning("[%s] httpx not installed. Run: pip install httpx", self.name)
            return False
        if not self._token:
            logger.warning("[%s] VKTEAMS_BOT_TOKEN not configured", self.name)
            return False

        # Idempotency: a second connect() on a live adapter must not leak the
        # old poll task / HTTP client (two loops sharing one cursor would skip
        # and double-dispatch events). Tear the previous session down first.
        if self._poll_task is not None and not self._poll_task.done():
            logger.info("[%s] connect() called on a live adapter — recycling", self.name)
            await self.disconnect()

        self._http_client = httpx.AsyncClient(timeout=20.0)

        # Validate the token with /self/get. A definitive rejection is
        # fatal; a network failure is not — the poll loop will retry.
        data, error, _ = await self._api_get("self/get", {})
        if data:
            self._bot_user_id = str(data.get("userId") or data.get("nick") or "")
            logger.info(
                "[%s] Connected as %s (%s) via %s",
                self.name,
                data.get("nick") or data.get("firstName") or "bot",
                self._bot_user_id,
                self._api_base,
            )
        elif error and ("unauthorized" in error.lower() or "invalid token" in error.lower()):
            logger.error("[%s] Token rejected by %s: %s", self.name, self._api_base, error)
            self._set_fatal_error(
                "vkteams_unauthorized",
                "VK Teams rejected the bot token. Check VKTEAMS_BOT_TOKEN "
                "and VKTEAMS_API_BASE (on-premise servers have unique URLs).",
                retryable=False,
            )
            await self._http_client.aclose()
            self._http_client = None
            return False
        else:
            logger.warning(
                "[%s] /self/get failed (%s) — starting poll loop anyway", self.name, error,
            )

        self._mark_connected()
        self._poll_task = asyncio.create_task(self._poll_loop())
        return True

    async def _poll_loop(self) -> None:
        """Long-poll /events/get with automatic reconnection."""
        backoff_idx = 0
        while self._running:
            poll_start = time.monotonic()
            try:
                await self._poll_once()
                backoff_idx = 0
                continue
            except asyncio.CancelledError:
                return
            except _FatalPollError:
                # Notify the gateway so it tears this adapter down (pops it
                # from routing, closes the client, restarts/exits as needed).
                # Run it on a SEPARATE task: the fatal handler awaits
                # disconnect(), which cancels THIS poll task — awaiting it
                # inline would await the current task. Mirrors telegram.
                notify = asyncio.create_task(self._notify_fatal_error())
                self._background_tasks.add(notify)
                notify.add_done_callback(self._background_tasks.discard)
                return
            except Exception as e:
                if not self._running:
                    return
                logger.warning("[%s] Poll error: %s", self.name, self._redact(e))

            if not self._running:
                return
            # Reset backoff if the previous poll survived for a while.
            if time.monotonic() - poll_start >= 60.0:
                backoff_idx = 0
            delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
            logger.info("[%s] Retrying poll in %ds...", self.name, delay)
            await asyncio.sleep(delay)
            backoff_idx += 1

    async def _poll_once(self) -> None:
        """One /events/get round trip plus dispatch of the returned events."""
        data, error, _ = await self._api_get(
            "events/get",
            {"lastEventId": self._last_event_id, "pollTime": self._poll_time},
            timeout=self._poll_time + 20.0,
        )
        if error:
            lowered = error.lower()
            if "unauthorized" in lowered or "invalid token" in lowered:
                logger.error("[%s] Poll auth failure: %s", self.name, error)
                self._set_fatal_error(
                    "vkteams_unauthorized",
                    "VK Teams rejected the bot token during polling. "
                    "Check VKTEAMS_BOT_TOKEN.",
                    retryable=False,
                )
                raise _FatalPollError(error)
            raise RuntimeError(error)

        for event in data.get("events") or []:
            try:
                event_id = int(event.get("eventId") or 0)
            except (TypeError, ValueError):
                event_id = 0
            if event_id > self._last_event_id:
                self._last_event_id = event_id
            # Dispatch off the poll task: a single event can block for a long
            # time (a 50MB attachment download runs up to 120s; a slash-confirm
            # tap runs the confirmed command inline). Awaiting it here would
            # stall /events/get for EVERY chat — no new messages fetched, other
            # users' button taps left unanswered. The cursor is already
            # advanced above, so acks stay ordered. Tasks are tracked in
            # _background_tasks and reaped by cancel_background_tasks().
            task = asyncio.create_task(self._handle_event_guarded(event, event_id))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _handle_event_guarded(self, event: Dict[str, Any], event_id: int) -> None:
        """Run _handle_event with per-event error isolation (background task)."""
        try:
            await self._handle_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "[%s] Error handling event %s: %s",
                self.name, event_id, self._redact(e), exc_info=True,
            )

    async def _handle_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("type")
        payload = event.get("payload") or {}
        if event_type == "newMessage":
            await self._on_new_message(payload)
        elif event_type == "callbackQuery":
            await self._on_callback_query(payload)
        else:
            # editedMessage / deletedMessage / pinnedMessage / membership
            # events carry nothing the gateway consumes today.
            logger.debug("[%s] Ignoring event type %s", self.name, event_type)

    async def disconnect(self) -> None:
        """Stop the poll loop and close the HTTP client."""
        self._running = False
        self._mark_disconnected()

        if self._poll_task:
            # Never cancel/await the poll task if disconnect() is itself running
            # on it (defensive — the fatal path routes teardown through a
            # SEPARATE task precisely so this doesn't happen). Cancelling the
            # current task would poison it; awaiting it would await ourselves.
            if self._poll_task is not asyncio.current_task():
                self._poll_task.cancel()
                try:
                    await self._poll_task
                except asyncio.CancelledError:
                    pass
            self._poll_task = None

        # Offloaded event-handler tasks are reaped by the gateway via
        # cancel_background_tasks() during shutdown; we don't cancel them here
        # (disconnect() can itself run on a _background_tasks task via the
        # fatal-error path, which would then cancel itself).

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._seen_messages.clear()
        self._last_edit_at.clear()
        self._last_overflow_preview.clear()
        self._last_typing_at.clear()
        self._approval_state.clear()
        self._slash_confirm_state.clear()
        self._clarify_state.clear()
        logger.info("[%s] Disconnected", self.name)

    # -- Inbound message processing -----------------------------------------

    def _is_duplicate(self, msg_id: str) -> bool:
        """Return True if this message ID was seen within the dedup window."""
        now = time.time()
        if len(self._seen_messages) > DEDUP_MAX_SIZE:
            cutoff = now - DEDUP_WINDOW_SECONDS
            self._seen_messages = {
                k: v for k, v in self._seen_messages.items() if v > cutoff
            }
        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False

    @staticmethod
    def _display_name(user: Dict[str, Any]) -> str:
        parts = [user.get("firstName") or "", user.get("lastName") or ""]
        name = " ".join(p for p in parts if p).strip()
        return name or user.get("nick") or str(user.get("userId") or "")

    async def _on_new_message(self, payload: Dict[str, Any]) -> None:
        """Normalize a newMessage payload into a MessageEvent and dispatch."""
        msg_id = str(payload.get("msgId") or "")
        sender = payload.get("from") or {}
        sender_id = str(sender.get("userId") or "")

        # Echo-loop prevention: skip the bot's own messages.
        if self._bot_user_id and sender_id == self._bot_user_id:
            return
        if msg_id and self._is_duplicate(msg_id):
            logger.debug("[%s] Duplicate message %s, skipping", self.name, msg_id)
            return

        chat = payload.get("chat") or {}
        chat_id = str(chat.get("chatId") or "")
        chat_type = {"private": "dm", "group": "group", "channel": "channel"}.get(
            str(chat.get("type") or ""), "dm"
        )
        user_name = self._display_name(sender)
        chat_name = chat.get("title") or user_name or chat_id

        text = payload.get("text") or ""
        message_type = MessageType.TEXT
        media_urls: List[str] = []
        media_types: List[str] = []
        reply_to_message_id: Optional[str] = None
        reply_to_text: Optional[str] = None
        reply_to_author_id: Optional[str] = None
        reply_to_author_name: Optional[str] = None
        reply_to_is_own = False

        for part in payload.get("parts") or []:
            part_type = part.get("type")
            part_payload = part.get("payload") or {}
            if part_type == "mention":
                # Rewrite raw "@[userId]" markers into readable @Name.
                mention_id = str(part_payload.get("userId") or "")
                mention_name = self._display_name(part_payload)
                if mention_id and mention_name:
                    text = text.replace(f"@[{mention_id}]", f"@{mention_name}")
            elif part_type == "reply":
                quoted = part_payload.get("message") or {}
                reply_to_message_id = str(quoted.get("msgId") or "") or None
                reply_to_text = quoted.get("text") or None
                quoted_from = quoted.get("from") or {}
                reply_to_author_id = str(quoted_from.get("userId") or "") or None
                reply_to_author_name = self._display_name(quoted_from) or None
                reply_to_is_own = bool(
                    self._bot_user_id and reply_to_author_id == self._bot_user_id
                )
            elif part_type == "forward":
                forwarded = part_payload.get("message") or {}
                fwd_text = forwarded.get("text") or ""
                fwd_from = self._display_name(forwarded.get("from") or {})
                if fwd_text:
                    label = f"[Forwarded from {fwd_from}]" if fwd_from else "[Forwarded]"
                    text = f"{text}\n{label}\n{fwd_text}".strip()
            elif part_type in ("file", "voice", "sticker"):
                file_id = str(part_payload.get("fileId") or "")
                if not file_id:
                    continue
                cached = await self._download_file(file_id)
                if not cached:
                    continue
                local_path, mime = cached
                media_urls.append(local_path)
                media_types.append(mime)
                if part_type == "voice":
                    message_type = MessageType.VOICE
                elif part_type == "sticker":
                    message_type = MessageType.STICKER
                elif mime.startswith("image/"):
                    message_type = MessageType.PHOTO
                elif mime.startswith("video/"):
                    message_type = MessageType.VIDEO
                elif mime.startswith("audio/"):
                    message_type = MessageType.AUDIO
                else:
                    message_type = MessageType.DOCUMENT
                caption = part_payload.get("caption")
                if caption and not text:
                    text = caption

        if message_type == MessageType.STICKER and not text:
            text = "[sticker]"
        if not text and not media_urls:
            logger.debug("[%s] Empty message %s, skipping", self.name, msg_id)
            return

        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=user_name,
            message_id=msg_id or None,
        )

        unix_ts = payload.get("timestamp")
        try:
            timestamp = (
                datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)
                if unix_ts else datetime.now(tz=timezone.utc)
            )
        except (ValueError, OSError, TypeError):
            timestamp = datetime.now(tz=timezone.utc)

        event = MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            message_id=msg_id or None,
            raw_message=payload,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=reply_to_message_id,
            reply_to_text=reply_to_text,
            reply_to_author_id=reply_to_author_id,
            reply_to_author_name=reply_to_author_name,
            reply_to_is_own_message=reply_to_is_own,
            timestamp=timestamp,
        )

        logger.debug(
            "[%s] Message in %s from %s: %s",
            self.name, chat_id, sender_id, text[:80],
        )
        await self.handle_message(event)

    async def _download_file(self, file_id: str) -> Optional[Tuple[str, str]]:
        """Resolve a fileId via /files/getInfo and cache the bytes locally.

        Returns ``(local_path, mime_type)`` or ``None`` on any failure —
        inbound media is best-effort, a failed download must not drop the
        whole message.
        """
        data, error, _ = await self._api_get("files/getInfo", {"fileId": file_id})
        if not data:
            logger.warning("[%s] files/getInfo failed for %s: %s", self.name, file_id, error)
            return None
        url = data.get("url")
        filename = data.get("filename") or file_id
        size = data.get("size") or 0
        if not url:
            return None
        try:
            if int(size) > MAX_UPLOAD_BYTES:
                logger.warning(
                    "[%s] File %s too large (%s bytes), skipping download",
                    self.name, filename, size,
                )
                return None
        except (TypeError, ValueError):
            pass
        try:
            resp = await self._http_client.get(url, timeout=120.0, follow_redirects=True)
            resp.raise_for_status()
            blob = resp.content
        except Exception as e:
            logger.warning(
                "[%s] Failed to download file %s: %s", self.name, file_id, self._redact(e),
            )
            return None

        mime = _guess_mime(filename, str(data.get("type") or ""))
        ext = Path(filename).suffix or mimetypes.guess_extension(mime) or ""
        try:
            if mime.startswith("image/"):
                path = cache_image_from_bytes(blob, ext=ext or ".jpg")
            elif mime.startswith("audio/"):
                path = cache_audio_from_bytes(blob, ext=ext or ".ogg")
            elif mime.startswith("video/"):
                path = cache_video_from_bytes(blob, ext=ext or ".mp4")
            else:
                path = cache_document_from_bytes(blob, filename=filename)
        except Exception as e:
            logger.warning("[%s] Failed to cache file %s: %s", self.name, filename, e)
            return None
        return path, mime

    # -- Outbound messaging -------------------------------------------------

    async def _send_text_raw(
        self,
        chat_id: str,
        text: str,
        *,
        reply_to: Optional[str] = None,
        keyboard: Optional[List[List[Dict[str, str]]]] = None,
        formatted: bool = True,
    ) -> SendResult:
        """One /messages/sendText call with markdown → plain-text fallback."""
        params: Dict[str, Any] = {"chatId": chat_id, "text": text}
        if reply_to:
            params["replyMsgId"] = reply_to
        if keyboard is not None:
            params["inlineKeyboardMarkup"] = json.dumps(keyboard, ensure_ascii=False)
        if formatted and self._parse_mode:
            params["parseMode"] = self._parse_mode

        data, error, retry_after = await self._api_get("messages/sendText", params)
        if data:
            return SendResult(success=True, message_id=str(data.get("msgId") or ""))

        retryable, kind = self._classify_error(error or "")
        if kind == "bad_format" and formatted and self._parse_mode:
            logger.warning(
                "[%s] %s rejected formatted text (%s), retrying as plain",
                self.name, self._parse_mode, error,
            )
            if os.getenv("VKTEAMS_DEBUG_FORMAT", "").strip():
                logger.warning(
                    "[%s] DEBUG_FORMAT rejected payload repr: %r",
                    self.name, text[:1000],
                )
            return await self._send_text_raw(
                chat_id,
                self._strip_formatting(text),
                reply_to=reply_to,
                keyboard=keyboard,
                formatted=False,
            )
        if kind == "rate_limited":
            logger.warning(
                "[%s] Rate limited sending to %s (retry_after=%s)",
                self.name, chat_id, retry_after,
            )
            return SendResult(
                success=False, error=error, retryable=True,
                retry_after=retry_after or 1.0, error_kind=kind,
            )
        logger.warning("[%s] Send failed: %s", self.name, error)
        return SendResult(success=False, error=error, retryable=retryable, error_kind=kind)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message, chunking oversized content."""
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)

        formatted = self.format_message(content) if self._parse_mode else content
        chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
        if len(chunks) > 1 and self._parse_mode == "MarkdownV2":
            # truncate_message appends a raw " (1/2)" suffix; escape the
            # MarkdownV2-special parens so the chunk isn't rejected.
            chunks = [
                re.sub(r" \((\d+)/(\d+)\)$", r" \\(\1/\2\\)", chunk)
                for chunk in chunks
            ]

        message_ids: List[str] = []
        for i, chunk in enumerate(chunks):
            result = await self._send_text_raw(
                chat_id,
                chunk,
                reply_to=reply_to if (reply_to and self._should_reply(i)) else None,
            )
            if result.retry_after and not result.success:
                await asyncio.sleep(min(result.retry_after, 10.0))
                result = await self._send_text_raw(
                    chat_id,
                    chunk,
                    reply_to=reply_to if (reply_to and self._should_reply(i)) else None,
                )
            if not result.success:
                if message_ids:
                    # Partial delivery: report the last visible message so
                    # follow-up edits target something real.
                    result.message_id = message_ids[-1]
                    result.continuation_message_ids = tuple(message_ids[:-1])
                return result
            if result.message_id:
                message_ids.append(result.message_id)
            # Respect the 1 msg/s group-chat budget between chunks.
            if len(chunks) > 1 and i < len(chunks) - 1 and self._is_group_chat(chat_id):
                await asyncio.sleep(1.05)

        return SendResult(
            success=True,
            message_id=message_ids[-1] if message_ids else None,
            continuation_message_ids=tuple(message_ids[:-1]),
        )

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Edit a sent message (streaming progress updates + final render).

        Mid-stream edits go out as plain text and are throttled per chat
        (group chats only allow 1 msg/s); the finalize edit applies
        MarkdownV2/HTML formatting with a plain-text fallback.  Content
        that outgrew the 4096 cap is truncated mid-stream and
        split-and-delivered on finalize (matching the Telegram plugin's
        contract: ``message_id`` = last visible message).
        """
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")

        now = time.monotonic()
        preview_key = (str(chat_id), str(message_id))

        if finalize:
            self._last_overflow_preview.pop(preview_key, None)
            text = self.format_message(content) if self._parse_mode else content
            if len(text) > self.MAX_MESSAGE_LENGTH:
                return await self._edit_overflow_split(chat_id, message_id, content)
            result = await self._edit_text_raw(chat_id, message_id, text, formatted=True)
            if result.success:
                self._last_edit_at[chat_id] = time.monotonic()
            return result

        # Mid-stream: plain text only, PACED to the platform's rate limit.
        # VK Teams allows only ~1 msg/s per group chat, so rather than firing
        # on every consumer tick we sleep out the remaining interval and THEN
        # perform the edit. Reporting success WITHOUT editing (the old "silent
        # skip") would poison the stream consumer: it records the un-sent text
        # as on-screen and can suppress the final full send, losing the tail.
        min_interval = (
            GROUP_EDIT_MIN_INTERVAL_S if self._is_group_chat(chat_id)
            else DM_EDIT_MIN_INTERVAL_S
        )
        wait = min_interval - (now - self._last_edit_at.get(chat_id, 0.0))
        if wait > 0:
            await asyncio.sleep(wait)

        # Oversized content is truncated (never split — splitting mid-stream
        # would duplicate on the next edit); the full text arrives on finalize.
        text = content
        saturated = False
        if len(text) > self.MAX_MESSAGE_LENGTH:
            text = text[: self.MAX_MESSAGE_LENGTH - 2] + " …"
            saturated = True
            if self._last_overflow_preview.get(preview_key) == text:
                # Identical saturated preview already on screen (recorded only
                # after a successful edit, below) — a genuine no-op.
                return SendResult(success=True, message_id=message_id)

        result = await self._edit_text_raw(chat_id, message_id, text, formatted=False)
        if result.success:
            self._last_edit_at[chat_id] = time.monotonic()
            if saturated:
                self._last_overflow_preview[preview_key] = text
            else:
                self._last_overflow_preview.pop(preview_key, None)
        return result

    async def _edit_text_raw(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        formatted: bool,
    ) -> SendResult:
        params: Dict[str, Any] = {"chatId": chat_id, "msgId": message_id, "text": text}
        if formatted and self._parse_mode:
            params["parseMode"] = self._parse_mode

        data, error, retry_after = await self._api_get("messages/editText", params)
        if data:
            return SendResult(success=True, message_id=message_id)

        lowered = (error or "").lower()
        if "not modified" in lowered:
            return SendResult(success=True, message_id=message_id)
        retryable, kind = self._classify_error(error or "")
        if kind == "bad_format" and formatted and self._parse_mode:
            logger.warning(
                "[%s] Formatted edit rejected (%s), retrying as plain", self.name, error,
            )
            return await self._edit_text_raw(
                chat_id, message_id, self._strip_formatting(text), formatted=False,
            )
        if kind == "rate_limited":
            wait = retry_after or 1.0
            if wait > 5.0:
                # Let the stream consumer fall back to a fresh send instead
                # of blocking the stream on a long penalty.
                return SendResult(
                    success=False, error=error, retryable=False,
                    retry_after=wait, error_kind=kind,
                )
            await asyncio.sleep(wait)
            data, error2, _ = await self._api_get("messages/editText", params)
            if data:
                return SendResult(success=True, message_id=message_id)
            return SendResult(success=False, error=error2, retryable=True, error_kind=kind)
        logger.warning(
            "[%s] Failed to edit message %s: %s", self.name, message_id, error,
        )
        return SendResult(success=False, error=error, retryable=retryable, error_kind=kind)

    async def _edit_overflow_split(
        self,
        chat_id: str,
        message_id: str,
        content: str,
    ) -> SendResult:
        """Finalize an oversized streamed response: edit the existing message
        with the first chunk and deliver the rest as fresh messages."""
        formatted = self.format_message(content) if self._parse_mode else content
        chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
        if self._parse_mode == "MarkdownV2":
            chunks = [
                re.sub(r" \((\d+)/(\d+)\)$", r" \\(\1/\2\\)", chunk)
                for chunk in chunks
            ]
        first = await self._edit_text_raw(chat_id, message_id, chunks[0], formatted=True)
        if not first.success:
            return first
        delivered: List[str] = [str(message_id)]
        for chunk in chunks[1:]:
            if self._is_group_chat(chat_id):
                await asyncio.sleep(1.05)
            result = await self._send_text_raw(chat_id, chunk)
            if not result.success:
                return SendResult(
                    success=False,
                    error=result.error,
                    message_id=delivered[-1],
                    continuation_message_ids=tuple(delivered[:-1]),
                    retryable=result.retryable,
                    error_kind=result.error_kind,
                )
            if result.message_id:
                delivered.append(result.message_id)
        return SendResult(
            success=True,
            message_id=delivered[-1],
            continuation_message_ids=tuple(delivered[:-1]),
        )

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message (48h window; group messages need admin rights)."""
        data, error, _ = await self._api_get(
            "messages/deleteMessages", {"chatId": chat_id, "msgId": message_id},
        )
        if not data:
            logger.debug(
                "[%s] deleteMessages failed for %s/%s: %s",
                self.name, chat_id, message_id, error,
            )
        return bool(data)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Show the "typing" indicator (server-side expiry: 10s).

        Base ``_keep_typing`` refreshes every 2s; the API only needs a
        resend every 10s, so calls inside the resend window are skipped.
        """
        now = time.monotonic()
        if now - self._last_typing_at.get(chat_id, 0.0) < TYPING_RESEND_INTERVAL_S:
            return
        self._last_typing_at[chat_id] = now
        _, error, _ = await self._api_get(
            "chats/sendActions", {"chatId": chat_id, "actions": "typing"},
        )
        if error:
            logger.debug("[%s] sendActions failed: %s", self.name, error)

    async def stop_typing(self, chat_id: str) -> None:
        """Clear the typing indicator (empty actions per API contract)."""
        self._last_typing_at.pop(chat_id, None)
        await self._api_get("chats/sendActions", {"chatId": chat_id, "actions": ""})

    # -- Media ----------------------------------------------------------------

    async def _send_file_multipart(
        self,
        chat_id: str,
        blob: bytes,
        filename: str,
        *,
        method: str = "messages/sendFile",
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Upload-and-send a file via multipart POST."""
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")
        if len(blob) > MAX_UPLOAD_BYTES:
            return SendResult(
                success=False,
                error=f"file exceeds the 50MB VK Teams upload cap ({len(blob)} bytes)",
                error_kind="too_long",
            )
        params: Dict[str, Any] = {"token": self._token, "chatId": chat_id}
        if caption:
            params["caption"] = caption[:1000]
        if reply_to:
            params["replyMsgId"] = reply_to
        mime = _guess_mime(filename)
        try:
            resp = await self._http_client.post(
                f"{self._api_base}/{method}",
                params=params,
                files={"file": (filename, blob, mime)},
                timeout=120.0,
            )
        except Exception as e:
            return SendResult(success=False, error=self._redact(e), retryable=True)
        try:
            data = resp.json()
        except Exception:
            data = {}
        if resp.status_code < 300 and isinstance(data, dict) and data.get("ok") is not False:
            return SendResult(success=True, message_id=str(data.get("msgId") or ""))
        error = self._redact(
            str((data or {}).get("description") or f"HTTP {resp.status_code}")
        )
        retryable, kind = self._classify_error(error)
        logger.warning("[%s] %s failed: %s", self.name, method, error)
        return SendResult(success=False, error=error, retryable=retryable, error_kind=kind)

    async def _read_local_file(self, path: str) -> Optional[bytes]:
        try:
            return await asyncio.to_thread(Path(path).read_bytes)
        except OSError as e:
            logger.warning("[%s] Cannot read file %s: %s", self.name, path, e)
            return None

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image by URL: download, then upload as a native file
        (the API has no send-by-URL primitive).

        The download runs through ``cache_image_from_url``, which enforces
        SSRF protection (``is_safe_url`` + a redirect guard that re-checks
        every hop).  Without it, a prompt-injected ``image_url`` pointing at
        ``169.254.169.254`` / an internal host would be fetched and its body
        posted into the chat — SSRF + data exfiltration.  On any block or
        failure the base default posts the URL as text (it never fetches).
        """
        try:
            from tools.url_safety import is_safe_url
            if not is_safe_url(image_url):
                logger.warning(
                    "[%s] Refusing unsafe image URL, sending as text", self.name,
                )
                return await super().send_image(
                    chat_id, image_url, caption, reply_to, metadata=metadata,
                )
            local_path = await cache_image_from_url(image_url)
        except Exception as e:
            logger.warning(
                "[%s] Image download failed, sending URL as text: %s",
                self.name, self._redact(e),
            )
            return await super().send_image(
                chat_id, image_url, caption, reply_to, metadata=metadata,
            )
        blob = await self._read_local_file(local_path)
        if blob is None:
            return await super().send_image(
                chat_id, image_url, caption, reply_to, metadata=metadata,
            )
        filename = Path(local_path).name or "image.jpg"
        return await self._send_file_multipart(
            chat_id, blob, filename, caption=caption, reply_to=reply_to,
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        blob = await self._read_local_file(image_path)
        if blob is None:
            return SendResult(success=False, error="cannot read image file")
        return await self._send_file_multipart(
            chat_id, blob, Path(image_path).name, caption=caption, reply_to=reply_to,
        )

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
        blob = await self._read_local_file(file_path)
        if blob is None:
            return SendResult(success=False, error="cannot read file")
        return await self._send_file_multipart(
            chat_id, blob, file_name or Path(file_path).name,
            caption=caption, reply_to=reply_to,
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        blob = await self._read_local_file(video_path)
        if blob is None:
            return SendResult(success=False, error="cannot read video file")
        return await self._send_file_multipart(
            chat_id, blob, Path(video_path).name, caption=caption, reply_to=reply_to,
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send audio as a playable voice bubble when the container allows
        it (aac/ogg/m4a), otherwise as a regular file attachment."""
        blob = await self._read_local_file(audio_path)
        if blob is None:
            return SendResult(success=False, error="cannot read audio file")
        path = Path(audio_path)
        method = (
            "messages/sendVoice" if path.suffix.lower() in VOICE_EXTS
            else "messages/sendFile"
        )
        return await self._send_file_multipart(
            chat_id, blob, path.name, method=method, caption=caption, reply_to=reply_to,
        )

    # -- Interactive buttons (shared ea:/sc:/cl: callback convention) ---------

    def _is_callback_user_authorized(
        self,
        user_id: str,
        *,
        chat_id: Optional[str] = None,
        chat_type: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> bool:
        """Gate inline-button taps. Fails CLOSED — never admits everyone.

        Prompts appear in authorized chats, but any member of a group the
        bot is in can tap a button (including a dangerous-command Approve),
        so this MUST enforce the same allowlist the gateway enforces on
        inbound messages. It first delegates to the runner's
        ``_is_user_authorized`` (which also consults the pairing store and
        ``GATEWAY_ALLOWED_USERS`` / ``GATEWAY_ALLOW_ALL_USERS``), then falls
        back to env-only auth that defaults to DENY — mirroring the Telegram
        plugin's fix for the fail-open class (#24457).
        """
        normalized = str(user_id or "").strip()
        if not normalized:
            return False

        # Preferred path: the gateway's own authorization decision.
        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        auth_fn = getattr(runner, "_is_user_authorized", None)
        if callable(auth_fn):
            try:
                normalized_type = str(chat_type or "dm").strip().lower() or "dm"
                if normalized_type == "private":
                    normalized_type = "dm"
                source = self.build_source(
                    chat_id=chat_id or normalized,
                    chat_type=normalized_type,
                    user_id=normalized,
                    user_name=user_name,
                )
                return bool(auth_fn(source))
            except Exception:
                logger.debug(
                    "[%s] Falling back to env-only callback auth for %s",
                    self.name, normalized, exc_info=True,
                )

        # Env-only fallback (no runner wired, e.g. unit tests). Fail closed.
        if self._allow_all_users:
            return True
        if self._allowed_users:
            return "*" in self._allowed_users or normalized.lower() in self._allowed_users
        # No allowlist configured: defer to the gateway-wide allow-all flag,
        # defaulting to DENY. Never return True unconditionally.
        return os.getenv("GATEWAY_ALLOW_ALL_USERS", "").strip().lower() in ("1", "true", "yes")

    async def _answer_callback(
        self, query_id: str, text: Optional[str] = None, show_alert: bool = False,
    ) -> None:
        """Ack a callbackQuery — mandatory, or clients spin forever."""
        params: Dict[str, Any] = {"queryId": query_id}
        if text:
            params["text"] = text[:200]
        if show_alert:
            params["showAlert"] = "true"
        _, error, _ = await self._api_get("messages/answerCallbackQuery", params)
        if error:
            logger.debug("[%s] answerCallbackQuery failed: %s", self.name, error)

    async def _replace_prompt_message(
        self, chat_id: Optional[str], message_id: Optional[str], text: str,
    ) -> None:
        """Rewrite a button prompt with its resolution (also drops the keyboard)."""
        if not chat_id or not message_id:
            return
        try:
            await self._edit_text_raw(chat_id, message_id, text, formatted=False)
        except Exception as e:
            logger.debug("[%s] prompt edit failed: %s", self.name, self._redact(e))

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Dangerous-command approval prompt with inline buttons.

        Button taps route through ``resolve_gateway_approval()`` — the
        same mechanism as the text ``/approve`` flow.
        """
        if not self._http_client:
            return SendResult(success=False, error="Not connected")
        try:
            cmd_preview = command[:3000] + "..." if len(command) > 3000 else command
            source = (
                f"⚠️ **Command Approval Required**\n\n"
                f"```\n{cmd_preview}\n```\n"
                f"Reason: {description}"
            )
            approval_id = next(self._approval_counter)
            keyboard = [
                [
                    {"text": "✅ Allow Once", "callbackData": f"ea:once:{approval_id}", "style": "primary"},
                    {"text": "✅ Session", "callbackData": f"ea:session:{approval_id}"},
                ],
                [
                    {"text": "✅ Always", "callbackData": f"ea:always:{approval_id}"},
                    {"text": "❌ Deny", "callbackData": f"ea:deny:{approval_id}", "style": "attention"},
                ],
            ]
            result = await self._send_text_raw(
                chat_id,
                self.format_message(source) if self._parse_mode else source,
                keyboard=keyboard,
            )
            if result.success:
                self._register_prompt_state(self._approval_state, approval_id, session_key)
            return result
        except Exception as e:
            logger.warning("[%s] send_exec_approval failed: %s", self.name, self._redact(e))
            return SendResult(success=False, error=self._redact(e))

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Three-button slash-command confirmation (Once / Always / Cancel)."""
        if not self._http_client:
            return SendResult(success=False, error="Not connected")
        try:
            preview = message if len(message) <= 3500 else message[:3500] + "..."
            keyboard = [
                [
                    {"text": "✅ Approve Once", "callbackData": f"sc:once:{confirm_id}", "style": "primary"},
                    {"text": "🔒 Always Approve", "callbackData": f"sc:always:{confirm_id}"},
                ],
                [
                    {"text": "❌ Cancel", "callbackData": f"sc:cancel:{confirm_id}", "style": "attention"},
                ],
            ]
            result = await self._send_text_raw(
                chat_id,
                self.format_message(preview) if self._parse_mode else preview,
                keyboard=keyboard,
            )
            if result.success:
                self._register_prompt_state(self._slash_confirm_state, confirm_id, session_key)
            return result
        except Exception as e:
            logger.warning("[%s] send_slash_confirm failed: %s", self.name, self._redact(e))
            return SendResult(success=False, error=self._redact(e))

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Clarify prompt: one numbered button per choice plus "Other".

        Open-ended questions (no choices) go out as plain text; the
        gateway's text-intercept captures the next message.
        """
        if not self._http_client:
            return SendResult(success=False, error="Not connected")
        try:
            if not choices:
                return await self.send(chat_id=chat_id, content=f"❓ {question}")

            option_lines = "\n".join(
                f"{i + 1}. {choice}" for i, choice in enumerate(choices)
            )
            text = f"❓ {question}\n\n{option_lines}"
            # Numeric button labels; full option text lives in the body so
            # long choices stay readable on mobile.
            rows: List[List[Dict[str, str]]] = [
                [{"text": str(idx + 1), "callbackData": f"cl:{clarify_id}:{idx}"}]
                for idx in range(len(choices))
            ]
            rows.append(
                [{"text": "✏️ Other (type answer)", "callbackData": f"cl:{clarify_id}:other"}]
            )
            result = await self._send_text_raw(chat_id, text, keyboard=rows, formatted=False)
            if result.success:
                self._register_prompt_state(self._clarify_state, clarify_id, session_key)
            return result
        except Exception as e:
            logger.warning("[%s] send_clarify failed: %s", self.name, self._redact(e))
            return SendResult(success=False, error=self._redact(e))

    async def _on_callback_query(self, payload: Dict[str, Any]) -> None:
        """Route inline-button taps to the gateway-side resolvers."""
        query_id = str(payload.get("queryId") or "")
        data = str(payload.get("callbackData") or "")
        sender = payload.get("from") or {}
        user_id = str(sender.get("userId") or "")
        user_display = self._display_name(sender) or "User"
        message = payload.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = str(chat.get("chatId") or "") or None
        chat_type = str(chat.get("type") or "") or None
        prompt_msg_id = str(message.get("msgId") or "") or None
        if not query_id or not data:
            return

        if not self._is_callback_user_authorized(
            user_id, chat_id=chat_id, chat_type=chat_type, user_name=user_display,
        ):
            await self._answer_callback(
                query_id, "⛔ You are not authorized to answer this prompt.",
            )
            return

        # --- Exec approval (ea:choice:id) ---
        if data.startswith("ea:"):
            parts = data.split(":", 2)
            if len(parts) != 3:
                await self._answer_callback(query_id, "Invalid approval data.")
                return
            choice = parts[1]
            try:
                approval_id = int(parts[2])
            except ValueError:
                await self._answer_callback(query_id, "Invalid approval data.")
                return
            entry = self._approval_state.pop(approval_id, None)
            if not entry:
                await self._answer_callback(query_id, "This approval has already been resolved.")
                return
            session_key = entry[0]
            label = {
                "once": "✅ Approved once",
                "session": "✅ Approved for session",
                "always": "✅ Approved permanently",
                "deny": "❌ Denied",
            }.get(choice, "Resolved")
            await self._answer_callback(query_id, label)
            await self._replace_prompt_message(
                chat_id, prompt_msg_id, f"{label} by {user_display}",
            )
            count = 0
            try:
                from tools.approval import resolve_gateway_approval
                count = resolve_gateway_approval(session_key, choice)
                logger.info(
                    "[%s] Button resolved %d approval(s) for session %s (choice=%s)",
                    self.name, count, session_key, choice,
                )
            except Exception as exc:
                logger.error("[%s] Failed to resolve approval: %s", self.name, exc)
            # Typing was paused while the approval waited; resume it so the
            # indicator doesn't stay dead for the rest of the turn.
            if count and chat_id:
                self.resume_typing_for_chat(chat_id)
            return

        # --- Slash confirm (sc:choice:confirm_id) ---
        if data.startswith("sc:"):
            parts = data.split(":", 2)
            if len(parts) != 3:
                await self._answer_callback(query_id, "Invalid prompt data.")
                return
            choice, confirm_id = parts[1], parts[2]
            entry = self._slash_confirm_state.pop(confirm_id, None)
            if not entry:
                await self._answer_callback(query_id, "This prompt has already been resolved.")
                return
            session_key = entry[0]
            label = {
                "once": "✅ Approved once",
                "always": "🔒 Always approve",
                "cancel": "❌ Cancelled",
            }.get(choice, "Resolved")
            await self._answer_callback(query_id, label)
            await self._replace_prompt_message(
                chat_id, prompt_msg_id, f"{label} by {user_display}",
            )
            try:
                from tools import slash_confirm as _slash_confirm_mod
                result_text = await _slash_confirm_mod.resolve(
                    session_key, confirm_id, choice,
                )
                if result_text and chat_id:
                    await self.send(chat_id=chat_id, content=result_text)
            except Exception as exc:
                logger.error(
                    "[%s] slash-confirm callback failed: %s", self.name, exc, exc_info=True,
                )
            return

        # --- Clarify (cl:clarify_id:idx | cl:clarify_id:other) ---
        if data.startswith("cl:"):
            parts = data.split(":", 2)
            if len(parts) != 3:
                await self._answer_callback(query_id, "Invalid choice.")
                return
            clarify_id, choice_token = parts[1], parts[2]
            entry = self._clarify_state.get(clarify_id)
            if not entry:
                await self._answer_callback(query_id, "This prompt has already been resolved.")
                return

            if choice_token == "other":
                # Flip into text-capture mode; keep _clarify_state until
                # the typed answer resolves it.
                flipped = False
                try:
                    from tools.clarify_gateway import mark_awaiting_text
                    flipped = mark_awaiting_text(clarify_id)
                except Exception as exc:
                    logger.warning("[%s] mark_awaiting_text failed: %s", self.name, exc)
                if not flipped:
                    self._clarify_state.pop(clarify_id, None)
                    await self._answer_callback(
                        query_id, "⌛ This question expired.", show_alert=True,
                    )
                    return
                await self._answer_callback(query_id, "✏️ Type your answer in the chat.")
                await self._replace_prompt_message(
                    chat_id, prompt_msg_id,
                    f"{message.get('text') or ''}\n\nAwaiting typed response from {user_display}…",
                )
                return

            try:
                idx = int(choice_token)
            except ValueError:
                await self._answer_callback(query_id, "Invalid choice.")
                return
            # Recover the human-readable choice text from the clarify
            # primitive's registry (same pattern as the Telegram plugin).
            resolved_text: Optional[str] = None
            try:
                from tools.clarify_gateway import _entries as _clarify_entries  # type: ignore
                entry = _clarify_entries.get(clarify_id)
                if entry and entry.choices and 0 <= idx < len(entry.choices):
                    resolved_text = entry.choices[idx]
            except Exception:
                resolved_text = None
            if resolved_text is None:
                resolved_text = f"choice {idx + 1}"

            self._clarify_state.pop(clarify_id, None)
            resolved = False
            try:
                from tools.clarify_gateway import resolve_gateway_clarify
                resolved = resolve_gateway_clarify(clarify_id, resolved_text)
            except Exception as exc:
                logger.error("[%s] resolve_gateway_clarify failed: %s", self.name, exc)
            if resolved:
                await self._answer_callback(query_id, f"✓ {resolved_text[:60]}")
                await self._replace_prompt_message(
                    chat_id, prompt_msg_id,
                    f"{message.get('text') or ''}\n\n{user_display}: {resolved_text}",
                )
            else:
                await self._answer_callback(
                    query_id, "⌛ This question expired.", show_alert=True,
                )
            return

        logger.debug("[%s] Unhandled callbackData: %s", self.name, data[:40])
        await self._answer_callback(query_id)

    # -- Chat info & formatting -----------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Resolve chat metadata via /chats/getInfo."""
        data, error, _ = await self._api_get("chats/getInfo", {"chatId": chat_id})
        if not data:
            logger.debug("[%s] chats/getInfo failed for %s: %s", self.name, chat_id, error)
            return {
                "name": chat_id,
                "type": "group" if self._is_group_chat(chat_id) else "dm",
            }
        chat_type = {"private": "dm", "group": "group", "channel": "channel"}.get(
            str(data.get("type") or ""), "dm"
        )
        name = data.get("title") or self._display_name(data) or chat_id
        return {"name": name, "type": chat_type, "chat_id": chat_id}

    def format_message(self, content: str) -> str:
        """Convert standard markdown to VK Teams MarkdownV2.

        The dialect follows Telegram's MarkdownV2 model (single ``*`` for
        bold, ``_`` italic, ``~`` strikethrough, backslash-escaped special
        characters) minus Telegram-only constructs (spoilers, expandable
        quotes).  Protected regions (code blocks, inline code) are
        extracted first so their contents are never modified.
        """
        if not content or not self._parse_mode:
            return content
        if self._parse_mode == "HTML":
            return self._format_html(content)
        if self._parse_mode != "MarkdownV2":
            return content

        placeholders: Dict[str, str] = {}
        counter = [0]

        def _ph(value: str) -> str:
            key = f"\x00PH{counter[0]}\x00"
            counter[0] += 1
            placeholders[key] = value
            return key

        text = _wrap_markdown_tables(content)

        # 1) Fenced code blocks — per the MarkdownV2 spec, \ and ` inside
        #    pre/code must themselves be escaped.
        def _protect_fenced(m):
            raw = m.group(0)
            open_end = raw.index('\n') + 1 if '\n' in raw[3:] else 3
            opening = raw[:open_end]
            body = raw[open_end:-3]
            body = body.replace('\\', '\\\\').replace('`', '\\`')
            return _ph(opening + body + '```')

        text = re.sub(r'(```(?:[^\n]*\n)?[\s\S]*?```)', _protect_fenced, text)

        # 2) Inline code
        text = re.sub(
            r'(`[^`]+`)',
            lambda m: _ph(m.group(0).replace('\\', '\\\\')),
            text,
        )

        # 3) Links — escape display text; in URLs only ')' and '\' need it.
        def _convert_link(m):
            display = _escape_mdv2(m.group(1))
            url = m.group(2).replace('\\', '\\\\').replace(')', '\\)')
            return _ph(f'[{display}]({url})')

        text = re.sub(
            r'\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)', _convert_link, text,
        )

        # 4) Headers (## Title) → bold
        def _convert_header(m):
            inner = re.sub(r'\*\*(.+?)\*\*', r'\1', m.group(1).strip())
            return _ph(f'*{_escape_mdv2(inner)}*')

        text = re.sub(r'^#{1,6}\s+(.+)$', _convert_header, text, flags=re.MULTILINE)

        # 5) Bold: **text** → *text*
        text = re.sub(
            r'\*\*(.+?)\*\*',
            lambda m: _ph(f'*{_escape_mdv2(m.group(1))}*'),
            text,
        )

        # 6) Italic: *text* → _text_ ([^*\n]+ keeps bullet lists intact)
        text = re.sub(
            r'\*([^*\n]+)\*',
            lambda m: _ph(f'_{_escape_mdv2(m.group(1))}_'),
            text,
        )

        # 7) Strikethrough: ~~text~~ → ~text~
        text = re.sub(
            r'~~(.+?)~~',
            lambda m: _ph(f'~{_escape_mdv2(m.group(1))}~'),
            text,
        )

        # 8) Blockquotes: keep the leading > unescaped
        text = re.sub(
            r'^(>{1,3}) (.+)$',
            lambda m: _ph(f'{m.group(1)} {_escape_mdv2(m.group(2))}'),
            text,
            flags=re.MULTILINE,
        )

        # 9) Escape whatever remains, then restore protected regions.
        text = _escape_mdv2(text)
        for key in reversed(list(placeholders.keys())):
            text = text.replace(key, placeholders[key])
        return text

    def _format_html(self, content: str) -> str:
        """Convert standard markdown to VK Teams HTML.

        VK Teams' HTML mode is the robust choice: the only characters that
        must be escaped are ``& < >``, and lone ``_ * ~`` or unbalanced
        backticks are plain literals (unlike MarkdownV2, whose parser fails
        the whole message with "Format error" on them).  Supported tags:
        ``<b> <i> <u> <s> <code> <pre> <a> <blockquote>`` — matching the
        VK Teams tutorial's HTML section.

        Markdown constructs are converted to placeholders holding finished
        HTML; the remaining plain text is HTML-escaped last, so tags emitted
        here are never double-escaped and stray ``<``/``>`` in prose are.
        """
        placeholders: Dict[str, str] = {}
        counter = [0]

        def _ph(value: str) -> str:
            key = f"\x00PH{counter[0]}\x00"
            counter[0] += 1
            placeholders[key] = value
            return key

        text = _wrap_markdown_tables(content)

        # 1) Fenced code blocks → <pre>. Both fences MUST be at line start
        #    (``re.MULTILINE`` ``^```) so a stray ``` inside prose or content
        #    can't be mistaken for a fence — that mis-pairing is what makes a
        #    single block swallow everything after it. Language hint dropped;
        #    group(1) is the raw body between the fences.
        def _protect_fenced(m):
            return _ph(f'<pre>{_escape_html(m.group(1))}</pre>')

        text = re.sub(
            r'^```[^\n]*\n([\s\S]*?)\n?^```[ \t]*$',
            _protect_fenced, text, flags=re.MULTILINE,
        )

        # 2) Inline code → <code>. Restricted to a single line ([^`\n]+) so a
        #    lone stray backtick in prose can only ever pair with another on
        #    the same line, never swallow whole paragraphs across newlines.
        text = re.sub(
            r'`([^`\n]+)`',
            lambda m: _ph(f'<code>{_escape_html(m.group(1))}</code>'),
            text,
        )

        # 3) Links → <a href>
        def _convert_link(m):
            display = _escape_html(m.group(1))
            url = _escape_html(m.group(2))
            return _ph(f'<a href="{url}">{display}</a>')

        text = re.sub(
            r'\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)', _convert_link, text,
        )

        # 4) Headers (## Title) → bold
        def _convert_header(m):
            inner = re.sub(r'\*\*(.+?)\*\*', r'\1', m.group(1).strip())
            return _ph(f'<b>{_escape_html(inner)}</b>')

        text = re.sub(r'^#{1,6}\s+(.+)$', _convert_header, text, flags=re.MULTILINE)

        # 5) Inline emphasis, applied recursively so one style may nest
        #    inside another (bold containing italic, strike containing bold,
        #    …) instead of leaving the inner markers literal. Each match
        #    re-processes its own inner text; leaf text is HTML-escaped.
        #    ``*text*`` uses ``[^*\n]+`` to keep bullet lists intact, and the
        #    ``_text_`` form is word-boundary-guarded so snake_case
        #    identifiers like ``latency_ms`` in prose are left alone.
        def _inline(s: str) -> str:
            s = re.sub(
                r'\*\*(.+?)\*\*',
                lambda m: _ph(f'<b>{_inline(m.group(1))}</b>'), s,
            )
            s = re.sub(
                r'__(.+?)__',
                lambda m: _ph(f'<b>{_inline(m.group(1))}</b>'), s,
            )
            s = re.sub(
                r'\*([^*\n]+)\*',
                lambda m: _ph(f'<i>{_inline(m.group(1))}</i>'), s,
            )
            s = re.sub(
                r'(?<!\w)_([^_\n]+)_(?!\w)',
                lambda m: _ph(f'<i>{_inline(m.group(1))}</i>'), s,
            )
            s = re.sub(
                r'~~(.+?)~~',
                lambda m: _ph(f'<s>{_inline(m.group(1))}</s>'), s,
            )
            return _escape_html(s)

        # 6) Blockquotes: > text → <blockquote> (marker dropped; inner text
        #    still gets emphasis).  Extracted before the body pass so the tag
        #    itself is never escaped.
        text = re.sub(
            r'^>{1,3} (.+)$',
            lambda m: _ph(f'<blockquote>{_inline(m.group(1))}</blockquote>'),
            text,
            flags=re.MULTILINE,
        )

        # 7) Emphasis + leaf escaping over the remaining body, then restore
        #    protected regions (reversed so nested placeholders resolve).
        text = _inline(text)
        for key in reversed(list(placeholders.keys())):
            text = text.replace(key, placeholders[key])
        return text

    def _strip_formatting(self, text: str) -> str:
        """Reduce formatted text to plain for the server-rejected fallback,
        dispatching on the active parse mode."""
        if self._parse_mode == "HTML":
            return _strip_html(text)
        return _strip_mdv2(text)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env vars during config load.

    Runs BEFORE adapter construction so env-only setups surface in
    ``hermes gateway status`` / ``get_connected_platforms()``.  The
    special ``home_channel`` key becomes a proper ``HomeChannel`` on the
    ``PlatformConfig`` (handled by the core hook) rather than being
    merged into ``extra``.
    """
    token = os.getenv("VKTEAMS_BOT_TOKEN", "").strip()
    if not token:
        return None
    seed: dict = {
        "token": token,
        "api_base": (
            os.getenv("VKTEAMS_API_BASE", "").strip() or DEFAULT_API_BASE
        ).rstrip("/"),
    }
    parse_mode = os.getenv("VKTEAMS_PARSE_MODE", "").strip()
    if parse_mode:
        seed["parse_mode"] = parse_mode
    poll_time = os.getenv("VKTEAMS_POLL_TIME", "").strip()
    if poll_time:
        seed["poll_time"] = poll_time
    home = os.getenv("VKTEAMS_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("VKTEAMS_HOME_CHANNEL_NAME", home),
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
    """Out-of-process send for cron / send_message_tool fallbacks.

    Used when the gateway runner is not in this process (e.g. ``hermes
    cron`` standalone) — without it, ``deliver=vkteams`` jobs fail with
    ``No live adapter``.  ``thread_id`` is accepted for signature parity
    only (the Bot API has no threads); ``force_document`` likewise —
    every upload is a file on VK Teams.  Text goes out as plain text
    (no parseMode) so delivery never depends on markdown validity.

    Note: the bot cannot initiate dialogs — sends reach only chats the
    bot is a member of or users who already messaged it.
    """
    if not HTTPX_AVAILABLE:
        return {"error": "vkteams standalone send: httpx not installed"}

    token = _resolve_token(pconfig)
    api_base = _resolve_api_base(pconfig)
    if not token:
        return {"error": "vkteams standalone send: VKTEAMS_BOT_TOKEN not configured"}
    if not chat_id:
        return {"error": "vkteams standalone send: no chat_id"}

    sent_ids: List[str] = []
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            if message and message.strip():
                chunks = BasePlatformAdapter.truncate_message(message, MAX_MESSAGE_LENGTH)
                for i, chunk in enumerate(chunks):
                    resp = await client.get(
                        f"{api_base}/messages/sendText",
                        params={"token": token, "chatId": chat_id, "text": chunk},
                    )
                    data = resp.json() if resp.status_code < 300 else {}
                    if not isinstance(data, dict) or data.get("ok") is False or resp.status_code >= 300:
                        description = _redact_token(
                            str((data or {}).get("description") or f"HTTP {resp.status_code}"),
                            token,
                        )
                        return {"error": f"vkteams sendText failed: {description}"}
                    if data.get("msgId"):
                        sent_ids.append(str(data["msgId"]))
                    if i < len(chunks) - 1 and str(chat_id).endswith(GROUP_CHAT_SUFFIX):
                        await asyncio.sleep(1.05)

            for media_path in media_files or []:
                path = Path(media_path)
                try:
                    blob = await asyncio.to_thread(path.read_bytes)
                except OSError as e:
                    return {"error": f"vkteams sendFile failed: cannot read {path.name}: {e}"}
                if len(blob) > MAX_UPLOAD_BYTES:
                    return {"error": f"vkteams sendFile failed: {path.name} exceeds 50MB cap"}
                resp = await client.post(
                    f"{api_base}/messages/sendFile",
                    params={"token": token, "chatId": chat_id},
                    files={"file": (path.name, blob, _guess_mime(path.name))},
                    timeout=120.0,
                )
                data = resp.json() if resp.status_code < 300 else {}
                if not isinstance(data, dict) or data.get("ok") is False or resp.status_code >= 300:
                    description = _redact_token(
                        str((data or {}).get("description") or f"HTTP {resp.status_code}"),
                        token,
                    )
                    return {"error": f"vkteams sendFile failed: {description}"}
                if data.get("msgId"):
                    sent_ids.append(str(data["msgId"]))
    except Exception as e:
        return {"error": f"vkteams standalone send failed: {_redact_token(str(e), token)}"}

    return {
        "success": True,
        "platform": "vkteams",
        "chat_id": chat_id,
        "message_id": sent_ids[-1] if sent_ids else None,
    }


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="vkteams",
        label="VK Teams",
        adapter_factory=lambda cfg: VKTeamsAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["VKTEAMS_BOT_TOKEN"],
        install_hint="pip install httpx   # already a Hermes dependency",
        # Env-driven auto-configuration: seeds PlatformConfig.extra so
        # env-only setups show up in `hermes gateway status` without
        # instantiating the HTTP client.
        env_enablement_fn=_env_enablement,
        # Cron home-channel delivery — `deliver=vkteams` jobs route to
        # VKTEAMS_HOME_CHANNEL when set.
        cron_deliver_env_var="VKTEAMS_HOME_CHANNEL",
        # Out-of-process cron delivery (see _standalone_send docstring).
        standalone_sender_fn=_standalone_send,
        # Auth env vars for _is_user_authorized() integration.
        allowed_users_env="VKTEAMS_ALLOWED_USERS",
        allow_all_env="VKTEAMS_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="💠",
        allow_update_command=True,
        platform_hint=(
            "You are communicating via VK Teams (corporate messenger). "
            "Messages render MarkdownV2: *bold*, _italic_, `inline code`, "
            "``` code blocks ```, [links](url), > quotes, and lists. "
            "There are NO threads, reactions, or media albums — reply in "
            "the main chat flow. Group chats are rate-limited to 1 message "
            "per second, so prefer fewer, consolidated messages. Long "
            "responses are split into 4096-character chunks."
        ),
    )
