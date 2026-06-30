"""
Slack platform adapter.

Uses slack-bolt (Python) with Socket Mode for:
- Receiving messages from channels and DMs
- Sending responses back
- Handling slash commands
- Thread support
"""

import asyncio
import contextvars
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple, List

try:
    from slack_bolt.async_app import AsyncApp
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    from slack_sdk.web.async_client import AsyncWebClient
    import aiohttp

    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    AsyncApp = Any
    AsyncSocketModeHandler = Any
    AsyncWebClient = Any

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    SUPPORTED_DOCUMENT_TYPES,
    SUPPORTED_VIDEO_TYPES,
    _TEXT_INJECT_EXTENSIONS,
    is_host_excluded_by_no_proxy,
    resolve_proxy_url,
    safe_url_for_log,
    cache_document_from_bytes,
    cache_video_from_bytes,
)


logger = logging.getLogger(__name__)

# ContextVar carrying the user_id of the slash-command invoker.
# Set in _handle_slash_command, read in send() to match the correct
# stashed response_url when multiple users issue commands on the same
# channel concurrently.  ContextVars propagate to child asyncio.Tasks
# (Python 3.7+), so the value set in _handle_slash_command's task is
# visible in _process_message_background's child task.
_slash_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_slash_user_id",
    default=None,
)


@dataclass
class _ThreadContextCache:
    """Cache entry for fetched thread context."""

    content: str
    fetched_at: float = field(default_factory=time.monotonic)
    message_count: int = 0
    parent_text: str = ""  # Raw text of the thread parent (for reply_to_text injection)


def check_slack_requirements() -> bool:
    """Check if Slack dependencies are available.

    Lazy-installs slack-bolt/slack-sdk via ``tools.lazy_deps.ensure("platform.slack")``
    on first call if not present. Rebinds all module-level globals on success.
    """
    if SLACK_AVAILABLE:
        return True

    def _import():
        from slack_bolt.async_app import AsyncApp
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        from slack_sdk.web.async_client import AsyncWebClient
        import aiohttp

        return {
            "AsyncApp": AsyncApp,
            "AsyncSocketModeHandler": AsyncSocketModeHandler,
            "AsyncWebClient": AsyncWebClient,
            "aiohttp": aiohttp,
            "SLACK_AVAILABLE": True,
        }

    from tools.lazy_deps import ensure_and_bind

    return ensure_and_bind("platform.slack", _import, globals(), prompt=False)


def _extract_text_from_slack_blocks(blocks: list) -> str:
    """Extract readable text from Slack Block Kit blocks, including quoted/forwarded content.

    Slack's modern WYSIWYG composer sends messages with a ``blocks`` array
    containing ``rich_text`` elements. When a user forwards or quotes another
    message, the quoted content appears as nested ``rich_text_quote`` elements
    that are *not* included in the plain ``text`` field of the event.

    This helper walks the rich-text tree recursively and returns readable lines,
    preserving quotes, list items, and preformatted blocks so the agent can see
    forwarded/quoted content instead of only the lossy plain-text field.
    """
    if not blocks:
        return ""

    parts: list[str] = []

    def _render_inline_elements(elements: list) -> str:
        """Render inline elements (text, link, channel, user, emoji, etc.)."""
        pieces: list[str] = []
        for el in elements:
            el_type = el.get("type", "")
            if el_type == "text":
                pieces.append(el.get("text", ""))
            elif el_type == "link":
                url = el.get("url", "")
                text = el.get("text", "") or url
                pieces.append(f"{text} ({url})")
            elif el_type == "channel":
                pieces.append(f"<#{el.get('channel_id', '')}>")
            elif el_type == "user":
                pieces.append(f"<@{el.get('user_id', '')}>")
            elif el_type == "usergroup":
                pieces.append(f"<!subteam^{el.get('usergroup_id', '')}>")
            elif el_type == "emoji":
                pieces.append(f":{el.get('name', '')}:")
            elif el_type == "broadcast":
                pieces.append(f"<!{el.get('range', 'here')}>")
            elif el_type == "date":
                pieces.append(el.get("fallback", ""))
        return "".join(pieces)

    def _append_line(text: str, quote_depth: int = 0, bullet: str = "") -> None:
        if not text or not text.strip():
            return
        prefix = ((">" * quote_depth) + " ") if quote_depth else ""
        parts.append(f"{prefix}{bullet}{text}".rstrip())

    def _walk_elements(elements: list, quote_depth: int = 0, bullet: str = "") -> None:
        for elem in elements:
            elem_type = elem.get("type", "")

            if elem_type == "rich_text_section":
                _append_line(
                    _render_inline_elements(elem.get("elements", [])),
                    quote_depth=quote_depth,
                    bullet=bullet,
                )
            elif elem_type == "rich_text_quote":
                _walk_elements(elem.get("elements", []), quote_depth=quote_depth + 1)
            elif elem_type == "rich_text_list":
                list_style = elem.get("style")
                for idx, item in enumerate(elem.get("elements", [])):
                    item_bullet = "• " if list_style == "bullet" else f"{idx + 1}. "
                    _walk_elements([item], quote_depth=quote_depth, bullet=item_bullet)
            elif elem_type == "rich_text_preformatted":
                code_lines: list[str] = []
                for child in elem.get("elements", []):
                    child_type = child.get("type", "")
                    if child_type == "rich_text_section":
                        rendered = _render_inline_elements(child.get("elements", []))
                    else:
                        rendered = _render_inline_elements([child])
                    if rendered:
                        code_lines.append(rendered)
                code_text = "\n".join(code_lines)
                if code_text:
                    lang = elem.get("language", "")
                    _append_line(
                        f"```{lang}\n{code_text}\n```",
                        quote_depth=quote_depth,
                        bullet=bullet,
                    )
            else:
                rendered = _render_inline_elements([elem])
                if rendered:
                    _append_line(rendered, quote_depth=quote_depth, bullet=bullet)

    for block in blocks:
        if (block or {}).get("type") == "rich_text":
            _walk_elements(block.get("elements", []))

    return "\n".join(parts)


def _serialize_slack_blocks_for_agent(blocks: list, max_chars: int = 6000) -> str:
    """Return a compact, redacted JSON view of the current message's Block Kit payload."""
    if not blocks:
        return ""

    if all((block or {}).get("type") == "rich_text" for block in blocks):
        return ""

    scalar_allowlist = {
        "type",
        "block_id",
        "action_id",
        "style",
        "dispatch_action",
        "optional",
        "multiple",
        "emoji",
    }
    recursive_allowlist = {
        "text",
        "title",
        "description",
        "label",
        "placeholder",
        "accessory",
        "fields",
        "elements",
        "options",
        "option_groups",
        "confirm",
        "submit",
        "close",
        "hint",
    }

    def _sanitize(value):
        if isinstance(value, list):
            return [
                item
                for item in (_sanitize(v) for v in value)
                if item not in (None, {}, [], "")
            ]
        if isinstance(value, dict):
            sanitized = {}
            for key, item in value.items():
                if key in scalar_allowlist:
                    sanitized[key] = item
                elif key in recursive_allowlist:
                    cleaned = _sanitize(item)
                    if cleaned not in (None, {}, [], ""):
                        sanitized[key] = cleaned
            return sanitized
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    try:
        payload = json.dumps(_sanitize(blocks), ensure_ascii=False, indent=2)
    except Exception:
        payload = repr(blocks)

    if len(payload) > max_chars:
        payload = payload[: max_chars - 18].rstrip() + "\n... [truncated]"

    return f"[Slack Block Kit payload for this message]\n```json\n{payload}\n```"


def _apply_slack_proxy(client: Any, proxy_url: Optional[str]) -> None:
    """Apply a resolved proxy to a Slack SDK client or clear it explicitly."""
    if hasattr(client, "proxy"):
        client.proxy = proxy_url


_SLACK_PROXY_HOSTS = (
    "slack.com",
    "files.slack.com",
    "wss-primary.slack.com",
)


def _resolve_slack_proxy_url() -> Optional[str]:
    """Resolve a proxy URL that Slack SDK clients can safely use."""
    proxy_url = resolve_proxy_url()
    if not proxy_url:
        return None

    normalized = proxy_url.lower()
    if not normalized.startswith(("http://", "https://")):
        logger.info(
            "[Slack] Ignoring unsupported proxy scheme for Slack transport: %s",
            safe_url_for_log(proxy_url),
        )
        return None

    if any(is_host_excluded_by_no_proxy(host) for host in _SLACK_PROXY_HOSTS):
        logger.info("[Slack] NO_PROXY bypasses Slack proxy configuration")
        return None

    return proxy_url


# Map Slack audio mimetypes to the file extension that matches the actual
# container bytes.  Critically, Slack's in-app "record a clip" voice messages
# arrive as MP4/AAC containers (``audio/mp4``, filename ``audio_message*.mp4``),
# NOT Ogg — so the extension we cache them under must be one a downstream STT
# backend (OpenAI Whisper / gpt-4o-transcribe) will accept for that container.
# OpenAI sniffs the container from the FILENAME extension, so a wrong extension
# (e.g. caching MP4 bytes as ``.ogg``) makes transcription fail outright.
# Mirrors the proven map in gateway/platforms/bluebubbles.py.
_SLACK_AUDIO_MIME_TO_EXT = {
    "audio/ogg": ".ogg",
    "audio/opus": ".ogg",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/aac": ".m4a",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
}

# Extensions OpenAI/Whisper-family STT backends accept (kept in sync with
# tools/transcription_tools.SUPPORTED_FORMATS).
_SLACK_STT_SUPPORTED_EXTS = frozenset(
    {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".aac", ".flac"}
)

# Cached-extension → reported ``audio/*`` mimetype. Used when re-routing a
# ``video/mp4``-mislabeled voice clip onto the audio path so the reported
# media_type stays coherent with the bytes we actually cached (the gateway's
# STT gate keys on the ``audio/`` prefix + the cached filename extension, but a
# matching mimetype avoids surprising any consumer that inspects it). Anything
# unmapped falls back to ``audio/mp4`` — Slack voice clips are MP4/AAC.
_SLACK_EXT_TO_AUDIO_MIME = {
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
}


def _resolve_slack_audio_ext(file_obj: Dict[str, Any], mimetype: str) -> str:
    """Pick the cache extension that matches an inbound Slack audio file's bytes.

    Resolution order (mirrors the video branch + bluebubbles.py):

    1. The real extension from the uploaded filename, when it's a format a
       Whisper-family STT backend accepts (so ``audio_message.mp4`` →
       ``.mp4``, ``clip.m4a`` → ``.m4a``).
    2. A mimetype → extension lookup (so ``audio/mp4`` → ``.m4a``).
    3. ``.m4a`` as a last resort — never ``.ogg``, which was the original bug:
       MP4/AAC voice messages cached as ``.ogg`` are rejected by OpenAI because
       the bytes don't match the container the extension claims.
    """
    name = (file_obj.get("name") or "").strip()
    _, name_ext = os.path.splitext(name)
    name_ext = name_ext.lower()
    if name_ext in _SLACK_STT_SUPPORTED_EXTS:
        return name_ext

    mime_key = (mimetype or "").split(";", 1)[0].strip().lower()
    if mime_key in _SLACK_AUDIO_MIME_TO_EXT:
        return _SLACK_AUDIO_MIME_TO_EXT[mime_key]

    return ".m4a"


def _is_slack_voice_clip(file_obj: Dict[str, Any]) -> bool:
    """Return True when a Slack file is an audio-only voice clip.

    Slack's in-app voice recordings are audio-only MP4 containers, but Slack
    sometimes reports them with a ``video/mp4`` mimetype, which would otherwise
    route them to video understanding instead of speech-to-text. Detect them by
    Slack's stable markers — the ``slack_audio`` subtype and the
    ``audio_message*`` filename pattern — so genuine videos are left untouched.
    """
    subtype = (file_obj.get("subtype") or "").strip().lower()
    if subtype == "slack_audio":
        # slack_audio is always audio-only. (slack_video clips carry a real
        # video track, so they are deliberately NOT matched here.)
        return True
    name = (file_obj.get("name") or "").strip().lower()
    return name.startswith("audio_message")


class SlackAdapter(BasePlatformAdapter):
    """
    Slack bot adapter using Socket Mode.

    Requires two tokens:
      - SLACK_BOT_TOKEN (xoxb-...) for API calls
      - SLACK_APP_TOKEN (xapp-...) for Socket Mode connection

    Features:
      - DMs and channel messages (mention-gated in channels)
      - Thread support
      - File/image/audio attachments
      - Slash commands (/hermes)
      - Typing indicators (not natively supported by Slack bots)
    """

    MAX_MESSAGE_LENGTH = 39000  # Slack API allows 40,000 chars; leave margin
    supports_code_blocks = True  # Slack mrkdwn renders fenced code blocks
    splits_long_messages = True  # send() chunks via truncate_message(MAX_MESSAGE_LENGTH)
    # Slack blocks typed native slash commands inside threads ("/approve is
    # not supported in threads. Sorry!").  The adapter rewrites a leading
    # "!" to "/" for known commands (see _handle_slack_message), so "!" is
    # the prefix that works everywhere — instruction text must show it.
    typed_command_prefix = "!"

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SLACK)
        self._app: Optional[Any] = None
        self._handler: Optional[Any] = None
        self._bot_user_id: Optional[str] = None
        self._user_name_cache: Dict[str, str] = {}  # user_id → display name
        self._socket_mode_task: Optional[asyncio.Task] = None
        # Multi-workspace support
        self._team_clients: Dict[str, Any] = {}  # team_id → WebClient
        self._team_bot_user_ids: Dict[str, str] = {}  # team_id → bot_user_id
        self._channel_team: Dict[str, str] = {}  # channel_id → team_id
        # Dedup cache: prevents duplicate bot responses when Socket Mode
        # reconnects redeliver events.
        self._dedup = MessageDeduplicator()
        # Track pending approval message_ts → resolved flag to prevent
        # double-clicks on approval buttons.
        self._approval_resolved: Dict[str, bool] = {}
        # Track timestamps of messages sent by the bot so we can respond
        # to thread replies even without an explicit @mention.
        self._bot_message_ts: set = set()
        self._BOT_TS_MAX = 5000  # cap to avoid unbounded growth
        # Track threads where the bot has been @mentioned — once mentioned,
        # respond to ALL subsequent messages in that thread automatically.
        self._mentioned_threads: set = set()
        self._MENTIONED_THREADS_MAX = 5000
        # Assistant thread metadata keyed by (channel_id, thread_ts). Slack's
        # AI Assistant lifecycle events can arrive before/alongside message
        # events, and they carry the user/thread identity needed for stable
        # session + memory scoping.
        self._assistant_threads: Dict[Tuple[str, str], Dict[str, str]] = {}
        self._ASSISTANT_THREADS_MAX = 5000
        # Cache for _fetch_thread_context results: cache_key → _ThreadContextCache
        self._thread_context_cache: Dict[str, _ThreadContextCache] = {}
        self._THREAD_CACHE_TTL = 60.0
        # Track message IDs that should get reaction lifecycle (DMs / @mentions).
        self._reacting_message_ids: set = set()
        # Track active assistant thread status indicators so stop_typing can
        # clear them (chat_id → thread_ts).
        self._active_status_threads: Dict[str, str] = {}
        # Slash-command contexts: stash response_url + user_id so send()
        # can route the first reply ephemerally.  Keyed by
        # (channel_id, user_id) to avoid cross-user collisions.
        # Each value: {"response_url": str, "ts": float}
        self._slash_command_contexts: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Socket Mode resilience: track runtime connection state so we can
        # self-heal when Slack silently drops the websocket.
        self._app_token: Optional[str] = None
        self._proxy_url: Optional[str] = None
        self._socket_watchdog_task: Optional[asyncio.Task] = None
        self._socket_reconnect_lock = asyncio.Lock()
        self._socket_watchdog_interval_s = 15.0

    def _start_socket_mode_handler(self) -> None:
        """Start the Slack Socket Mode background task."""
        if not self._app or not self._app_token:
            raise RuntimeError("Socket Mode requires an initialized app and app token")

        self._handler = AsyncSocketModeHandler(
            self._app, self._app_token, proxy=self._proxy_url
        )
        _apply_slack_proxy(self._handler.client, self._proxy_url)

        task = asyncio.create_task(self._handler.start_async())
        self._socket_mode_task = task
        task.add_done_callback(self._on_socket_mode_task_done)

    async def _stop_socket_mode_handler(self) -> None:
        """Stop Socket Mode handler and task."""
        handler = self._handler
        task = self._socket_mode_task
        self._handler = None
        self._socket_mode_task = None

        if handler is not None:
            try:
                await handler.close_async()
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning(
                    "[Slack] Error while closing Socket Mode handler: %s",
                    e,
                    exc_info=True,
                )

        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive logging
                logger.debug(
                    "[Slack] Socket Mode task failed while stopping", exc_info=True
                )

    async def _socket_transport_connected(self) -> Optional[bool]:
        """Best-effort check of current Socket Mode transport state."""
        client = getattr(self._handler, "client", None)
        if client is None:
            return None

        state = getattr(client, "is_connected", None)
        if state is None:
            return None

        try:
            value = state() if callable(state) else state
            if asyncio.iscoroutine(value):
                value = await value
            return bool(value)
        except Exception:  # pragma: no cover - optional client API
            logger.debug(
                "[Slack] Could not inspect Socket Mode transport state", exc_info=True
            )
            return None

    async def _restart_socket_mode(self, reason: str) -> None:
        """Reconnect Socket Mode without rebuilding adapter state."""
        if not self._running:
            return

        async with self._socket_reconnect_lock:
            if not self._running or not self._app or not self._app_token:
                return

            logger.warning("[Slack] Socket Mode unhealthy (%s); reconnecting", reason)
            await self._stop_socket_mode_handler()

            try:
                self._start_socket_mode_handler()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(
                    "[Slack] Socket Mode reconnect failed: %s", exc, exc_info=True
                )

    async def _socket_watchdog_loop(self) -> None:
        """Monitor Socket Mode and reconnect if the task/transport dies.

        The body is wrapped in a broad except so a transient bug in
        ``_restart_socket_mode`` or the transport probe cannot permanently
        disable self-healing — the loop logs and keeps polling.
        """
        while self._running:
            try:
                await asyncio.sleep(self._socket_watchdog_interval_s)
                if not self._running:
                    break

                task = self._socket_mode_task
                if task is None:
                    await self._restart_socket_mode("socket task missing")
                    continue

                if task.done():
                    await self._restart_socket_mode("socket task stopped")
                    continue

                connected = await self._socket_transport_connected()
                if connected is False:
                    await self._restart_socket_mode("transport disconnected")
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive logging
                logger.warning(
                    "[Slack] Socket Mode watchdog iteration failed; continuing",
                    exc_info=True,
                )

    def _on_socket_watchdog_done(self, task: asyncio.Task) -> None:
        if task is not self._socket_watchdog_task:
            return
        if task.cancelled() or not self._running:
            return
        try:
            exc = task.exception()
        except (asyncio.CancelledError, Exception):  # pragma: no cover
            exc = None
        if exc is not None:
            logger.warning(
                "[Slack] Socket Mode watchdog exited with error; restarting: %s",
                exc,
                exc_info=True,
            )
        else:
            logger.warning("[Slack] Socket Mode watchdog exited; restarting")
        self._socket_watchdog_task = None
        self._ensure_socket_watchdog()

    def _ensure_socket_watchdog(self) -> None:
        if self._socket_watchdog_task is None or self._socket_watchdog_task.done():
            task = asyncio.create_task(self._socket_watchdog_loop())
            self._socket_watchdog_task = task
            task.add_done_callback(self._on_socket_watchdog_done)

    def _on_socket_mode_task_done(self, task: asyncio.Task) -> None:
        # Ignore stale tasks from intentional reconnect/shutdown.
        if task is not self._socket_mode_task:
            return
        if task.cancelled():
            return
        if not self._running:
            return

        exc = None
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "[Slack] Could not inspect Socket Mode task exception", exc_info=True
            )

        if exc is not None:
            logger.warning(
                "[Slack] Socket Mode task exited with error: %s", exc, exc_info=True
            )
        else:
            logger.warning("[Slack] Socket Mode task exited unexpectedly")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._restart_socket_mode("socket task exited"))

    def _describe_slack_api_error(
        self, response: Any, *, file_obj: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Convert Slack API auth/permission failures into actionable user-facing text."""
        if response is None or not hasattr(response, "get"):
            return None

        error = str(response.get("error", "") or "").strip()
        if not error:
            return None

        file_label = str(
            (file_obj or {}).get("name")
            or (file_obj or {}).get("id")
            or "this attachment"
        )
        needed = str(response.get("needed", "") or "").strip()
        provided = str(response.get("provided", "") or "").strip()
        reinstall_hint = " Update the Slack app scopes/settings and reinstall the app to the workspace."
        provided_hint = f" Current bot scopes: {provided}." if provided else ""

        if error == "missing_scope":
            needed_hint = (
                f"Missing scope: {needed}."
                if needed
                else "Missing required Slack scope."
            )
            return f"Slack attachment access failed for {file_label}. {needed_hint}{provided_hint}{reinstall_hint}"
        if error in {"not_authed", "invalid_auth", "account_inactive", "token_revoked"}:
            return f"Slack attachment access failed for {file_label} because the bot token is not authorized ({error}). Refresh the token/reinstall the app."
        if error in {"file_not_found", "file_deleted"}:
            return f"Slack attachment {file_label} is no longer available ({error})."
        if error in {
            "access_denied",
            "file_access_denied",
            "no_permission",
            "not_allowed_token_type",
            "restricted_action",
        }:
            return f"Slack attachment access failed for {file_label} because the bot does not have permission ({error}). Check workspace permissions/scopes and reinstall if needed."
        return None

    def _describe_slack_download_failure(
        self, exc: Exception, *, file_obj: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Translate Slack download exceptions into user-facing attachment diagnostics."""
        file_label = str(
            (file_obj or {}).get("name")
            or (file_obj or {}).get("id")
            or "this attachment"
        )

        response = getattr(exc, "response", None)
        api_detail = self._describe_slack_api_error(response, file_obj=file_obj)
        if api_detail:
            return api_detail

        try:
            import httpx
        except Exception:  # pragma: no cover
            httpx = None

        if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 401:
                return f"Slack attachment access failed for {file_label} with HTTP 401. The bot token is not authorized for this file."
            if status == 403:
                return f"Slack attachment access failed for {file_label} with HTTP 403. The bot likely lacks permission or scope to read this file."
            if status == 404:
                return f"Slack attachment {file_label} returned HTTP 404 and is no longer reachable."

        message = str(exc)
        if (
            "Slack returned HTML instead of media" in message
            or "non-image data" in message
        ):
            return (
                f"Slack attachment access failed for {file_label}: Slack returned an HTML/login or non-media response. "
                "This usually means a scope, auth, or file-permission problem."
            )
        return None

    # ------------------------------------------------------------------
    # Slash-command ephemeral helpers
    # ------------------------------------------------------------------

    _SLASH_CTX_TTL = 120.0  # seconds — response_url is valid for 30 min;
    # we use a much shorter TTL to avoid routing unrelated messages
    # as ephemeral if the command handler was slow or dropped.

    def _pop_slash_context(
        self,
        chat_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return and remove the slash-command context for *chat_id*, if fresh.

        Contexts older than ``_SLASH_CTX_TTL`` seconds are silently discarded.

        Uses the ``_slash_user_id`` ContextVar (set in ``_handle_slash_command``)
        to match the exact ``(channel_id, user_id)`` key.  This prevents a
        concurrent slash command from a different user on the same channel from
        stealing another user's ephemeral context.  Falls back to a
        channel-only scan when the ContextVar is unset (e.g. send() called
        from a non-slash code path — should not match anything).
        """
        now = time.monotonic()
        # Clean up stale entries on every lookup — dict is small.
        stale_keys = [
            k
            for k, v in self._slash_command_contexts.items()
            if now - v["ts"] > self._SLASH_CTX_TTL
        ]
        for k in stale_keys:
            self._slash_command_contexts.pop(k, None)

        # Precise match: (channel_id, user_id) from ContextVar.
        uid = _slash_user_id.get()
        if uid:
            return self._slash_command_contexts.pop((chat_id, uid), None)

        # Fallback: channel-only scan (only reachable when ContextVar is
        # unset, i.e. send() called outside a slash-command async context).
        match_key = None
        for key in list(self._slash_command_contexts):
            if key[0] == chat_id:
                match_key = key
                break
        if match_key is None:
            return None
        return self._slash_command_contexts.pop(match_key)

    async def _send_slash_ephemeral(
        self,
        ctx: Dict[str, Any],
        content: str,
    ) -> "SendResult":
        """Replace the initial ephemeral ack via ``response_url``.

        Slack's ``response_url`` accepts a POST with ``replace_original``
        for up to 30 minutes after the slash command was invoked.  This
        lets us swap the "Running /cmd…" placeholder with the real reply,
        and the message stays ephemeral ("Only visible to you").

        Falls back to a simple ``True`` SendResult if the POST fails —
        the user already saw the initial ack, so a delivery failure here
        is non-critical.
        """
        formatted = self.format_message(content)
        # Slack's response_url has the same ~40k char limit as chat_postMessage.
        # Truncate to MAX_MESSAGE_LENGTH and use only the first chunk — the
        # response_url replaces a single ephemeral ack, so multi-chunk isn't
        # possible.  Long responses are rare for command replies.
        chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
        text = chunks[0] if chunks else formatted
        payload = {
            "response_type": "ephemeral",
            "replace_original": True,
            "text": text,
        }
        try:
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.post(
                    ctx["response_url"],
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return SendResult(success=True, message_id=None)
                    body = await resp.text()
                    logger.warning(
                        "[Slack] response_url POST returned %s: %s",
                        resp.status,
                        body[:200],
                    )
        except Exception as e:
            logger.warning(
                "[Slack] response_url POST failed: %s",
                e,
            )
        # Non-fatal — the user saw the initial ack already.
        return SendResult(success=True, message_id=None)

    def _warn_if_missing_group_dm_scopes(self, auth_response, team_name: str) -> None:
        """Nudge existing installs to reinstall when group-DM scopes are absent.

        Group DMs only reach the bot when the app is subscribed to
        ``message.mpim`` and granted ``mpim:history`` (see slack_cli.py
        manifest). A missing event delivers *nothing* — there is no runtime
        API error to catch — so the only place we can detect a stale install
        is at connect time, by inspecting the ``x-oauth-scopes`` header the
        Slack ``auth.test`` response carries. If the app clearly handles 1:1
        DMs (``im:history`` present) but lacks ``mpim:history``, it predates
        this fix; log exactly what to add and that a reinstall is required.
        """
        try:
            # Track warned workspaces so the nudge fires once per process per
            # team, not on every reconnect. getattr-default keeps bare
            # object.__new__ test instances (no __init__) from crashing.
            warned = getattr(self, "_group_dm_scope_warned", None)
            if warned is None:
                warned = set()
                self._group_dm_scope_warned = warned
            headers = getattr(auth_response, "headers", None) or {}
            raw = headers.get("x-oauth-scopes") or headers.get("X-OAuth-Scopes") or ""
            if not raw:
                return  # Header absent (e.g. some proxies) — don't guess.
            granted = {s.strip() for s in raw.split(",") if s.strip()}
            team_key = team_name or ""
            if team_key in warned:
                return
            # Only nudge real DM-capable installs; "im:history" present but
            # "mpim:history" missing == stale manifest from before the fix.
            if "im:history" in granted and "mpim:history" not in granted:
                warned.add(team_key)
                logger.warning(
                    "[Slack] Group DMs (multi-person DMs) will not work in "
                    "workspace %s: the app is missing the 'mpim:history' scope "
                    "and 'message.mpim' event. Add 'mpim:history' (and "
                    "'mpim:read') to bot scopes, add 'message.mpim' to event "
                    "subscriptions, then REINSTALL the app to the workspace. "
                    "Regenerating the app from `hermes slack` produces a "
                    "manifest with these already included.",
                    team_key or "this workspace",
                )
        except Exception:  # pragma: no cover - diagnostics must never break connect
            pass

    def _warn_if_not_bot_token(self, auth_response, team_name: str) -> None:
        """Warn when the configured token authenticates as a human, not a bot.

        ``auth.test`` returns the ``user_id`` of *whatever principal owns the
        token*. For a real bot token (``xoxb-…``) that is the app's bot user
        and the response carries a ``bot_id``. For a **user** token
        (``xoxp-…`` / a legacy/personal OAuth token) it is the *installing
        human's* member ID and there is **no** ``bot_id``.

        When that happens, ``self._bot_user_id`` becomes a human's member ID,
        and every "is this the bot?" check downstream misfires: that one
        person's ``<@…>`` mentions wake the bot (``is_mentioned`` in
        ``_handle_slack_message``) and get stripped as if they were the bot's
        own mention — so the agent is genuinely told it was @mentioned and
        replies to messages merely *addressed to that human*. There is no
        runtime API error to catch; the only detectable moment is here at
        connect time, by noticing ``bot_id`` is absent from ``auth.test``.

        Warning-only: a user token can still send/receive, and we don't want
        to hard-fail a working-but-misconfigured install on connect. We log
        exactly what is wrong and how to fix it, once per workspace per
        process.
        """
        try:
            warned = getattr(self, "_user_token_warned", None)
            if warned is None:
                warned = set()
                self._user_token_warned = warned
            team_key = team_name or ""
            if team_key in warned:
                return
            # ``auth.test`` includes ``bot_id`` only for bot tokens. Its
            # absence (with a resolved user_id) means a user/legacy token.
            bot_id = ""
            user_id = ""
            try:
                bot_id = auth_response.get("bot_id", "") or ""
                user_id = auth_response.get("user_id", "") or ""
            except Exception:
                # Some response shapes are attribute-only; fall back to .data.
                data = getattr(auth_response, "data", None) or {}
                bot_id = data.get("bot_id", "") or ""
                user_id = data.get("user_id", "") or ""
            if not user_id:
                return  # Nothing resolved — don't guess.
            if not bot_id:
                warned.add(team_key)
                logger.warning(
                    "[Slack] The configured Slack token for workspace %s "
                    "authenticated as a USER (member %s), not a bot — the "
                    "auth.test response has no 'bot_id'. This is almost "
                    "certainly a user token (xoxp-...) instead of a Bot User "
                    "OAuth Token (xoxb-...). The bot's identity is now bound "
                    "to that member's ID, so mentions OF THAT PERSON will be "
                    "misrouted as mentions of the bot (the bot replies to "
                    "messages merely addressed to them). Use the 'Bot User "
                    "OAuth Token' (xoxb-...) from your Slack app's 'OAuth & "
                    "Permissions' page in SLACK_BOT_TOKEN.",
                    team_key or "this workspace",
                    user_id,
                )
        except Exception:  # pragma: no cover - diagnostics must never break connect
            pass

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to Slack via Socket Mode."""
        if not SLACK_AVAILABLE:
            logger.error(
                "[Slack] slack-bolt not installed. Run: pip install slack-bolt",
            )
            return False

        raw_token = self.config.token
        app_token = os.getenv("SLACK_APP_TOKEN")

        if not raw_token:
            logger.error("[Slack] SLACK_BOT_TOKEN not set")
            return False
        if not app_token:
            logger.error("[Slack] SLACK_APP_TOKEN not set")
            return False

        proxy_url = _resolve_slack_proxy_url()
        if proxy_url:
            logger.info(
                "[Slack] Using proxy for Slack transport: %s",
                safe_url_for_log(proxy_url),
            )

        # Support comma-separated bot tokens for multi-workspace
        bot_tokens = [t.strip() for t in raw_token.split(",") if t.strip()]

        # Also load tokens from OAuth token file
        from hermes_constants import get_hermes_home

        tokens_file = get_hermes_home() / "slack_tokens.json"
        if tokens_file.exists():
            try:
                saved = json.loads(tokens_file.read_text(encoding="utf-8"))
                for team_id, entry in saved.items():
                    tok = entry.get("token", "") if isinstance(entry, dict) else ""
                    if tok and tok not in bot_tokens:
                        bot_tokens.append(tok)
                        team_label = (
                            entry.get("team_name", team_id)
                            if isinstance(entry, dict)
                            else team_id
                        )
                        logger.info(
                            "[Slack] Loaded saved token for workspace %s", team_label
                        )
            except Exception as e:
                logger.warning("[Slack] Failed to read %s: %s", tokens_file, e)

        lock_acquired = False
        try:
            if not self._acquire_platform_lock(
                "slack-app-token", app_token, "Slack app token"
            ):
                return False
            lock_acquired = True
            self._running = False

            # Tear down any prior reconnect state before flipping ``_running``
            # back on. We must cancel + await the existing watchdog (not just
            # check ``task.done()`` later) so an old watchdog can't observe
            # ``_running=False``, exit, and then leave us with no monitor when
            # ``_ensure_socket_watchdog`` runs before the new task is visible.
            watchdog_task = self._socket_watchdog_task
            self._socket_watchdog_task = None
            if watchdog_task is not None and not watchdog_task.done():
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except asyncio.CancelledError:
                    pass
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug(
                        "[Slack] Prior watchdog task failed while stopping",
                        exc_info=True,
                    )

            # Close any previous handler before creating a new one so that
            # calling connect() a second time (e.g. during a gateway restart or
            # in-process reconnect attempt) does not leave a zombie Socket Mode
            # connection alive.  Both the old and new connections would otherwise
            # receive every Slack event and dispatch it twice, producing double
            # responses — the same bug that affected DiscordAdapter (#18187).
            await self._stop_socket_mode_handler()
            self._app = None
            self._app_token = app_token
            self._proxy_url = proxy_url

            # Reset multi-workspace state before re-populating it so a
            # reconnect that drops a workspace (or rotates the primary bot
            # token) doesn't carry stale ``_bot_user_id`` / ``_team_clients``
            # / ``_team_bot_user_ids`` entries from the prior session.
            self._bot_user_id = None
            self._team_clients = {}
            self._team_bot_user_ids = {}

            # First token is the primary — used for AsyncApp / Socket Mode
            primary_token = bot_tokens[0]
            self._app = AsyncApp(token=primary_token)
            _apply_slack_proxy(self._app.client, proxy_url)

            # Register each bot token and map team_id → client
            for token in bot_tokens:
                client = AsyncWebClient(token=token)
                _apply_slack_proxy(client, proxy_url)
                auth_response = await client.auth_test()
                team_id = auth_response.get("team_id", "")
                bot_user_id = auth_response.get("user_id", "")
                bot_name = auth_response.get("user", "unknown")
                team_name = auth_response.get("team", "unknown")

                self._team_clients[team_id] = client
                self._team_bot_user_ids[team_id] = bot_user_id

                # First token always wins as the primary bot user id; we
                # cleared ``_bot_user_id`` above so this picks up the current
                # token's identity even on reconnect.
                if self._bot_user_id is None:
                    self._bot_user_id = bot_user_id

                logger.info(
                    "[Slack] Authenticated as @%s in workspace %s (team: %s)",
                    bot_name,
                    team_name,
                    team_id,
                )

                self._warn_if_missing_group_dm_scopes(auth_response, team_name)
                self._warn_if_not_bot_token(auth_response, team_name)

            # Register message event handler
            @self._app.event("message")
            async def handle_message_event(event, say):
                await self._handle_slack_message(event)

            # Handle app_mention explicitly. In some Slack app configurations,
            # channel mentions arrive only as app_mention events rather than the
            # generic message event. Forward them into the normal message
            # pipeline so @mentions reliably produce replies.
            # NOTE: when Slack fires BOTH message and app_mention for the same
            # @mention, they share the same event ts — the dedup in
            # _handle_slack_message (MessageDeduplicator) suppresses the second.
            @self._app.event("app_mention")
            async def handle_app_mention(event, say):
                await self._handle_slack_message(event)

            # File lifecycle events can arrive around snippet uploads even when
            # the actual user message is what we care about. Ack them so Slack
            # doesn't log noisy 404 "unhandled request" warnings.
            @self._app.event("file_shared")
            async def handle_file_shared(event, say):
                await self._handle_slack_file_shared(event)

            @self._app.event("file_created")
            async def handle_file_created(event, say):
                pass

            @self._app.event("file_change")
            async def handle_file_change(event, say):
                pass

            # Reactions are useful lightweight acknowledgements in Slack, but
            # Hermes does not currently need to route them into the agent loop.
            # Ack the events explicitly so high-traffic channels do not fill
            # gateway.error.log with Slack Bolt "Unhandled request" warnings.
            @self._app.event("reaction_added")
            async def handle_reaction_added(event, say):
                pass

            @self._app.event("reaction_removed")
            async def handle_reaction_removed(event, say):
                pass

            @self._app.event("assistant_thread_started")
            async def handle_assistant_thread_started(event, say):
                await self._handle_assistant_thread_lifecycle_event(event)

            @self._app.event("assistant_thread_context_changed")
            async def handle_assistant_thread_context_changed(event, say):
                await self._handle_assistant_thread_lifecycle_event(event)

            # Register slash command handler(s)
            #
            # Every gateway command from COMMAND_REGISTRY is a native Slack
            # slash, matching Discord and Telegram's model (e.g. /btw, /stop,
            # /model work directly without /hermes prefix). A single regex
            # matcher dispatches all of them to one handler so we don't need
            # N identical @app.command() decorators.
            #
            # The slash commands must ALSO be declared in the Slack app
            # manifest (see `hermes slack manifest`). In Socket Mode, Slack
            # routes the command event through the socket regardless of the
            # manifest's request URL, but it will not deliver an event for
            # a slash command the manifest doesn't declare.
            from hermes_cli.commands import slack_native_slashes
            import re as _re

            _slash_names = [name for name, _d, _h in slack_native_slashes()]
            if _slash_names:
                _slash_pattern = _re.compile(
                    r"^/(?:" + "|".join(_re.escape(n) for n in _slash_names) + r")$"
                )
            else:  # pragma: no cover - registry always non-empty
                _slash_pattern = _re.compile(r"^/hermes$")

            @self._app.command(_slash_pattern)
            async def handle_hermes_command(ack, command):
                slash = (command.get("command") or "").lstrip("/")
                await ack(
                    response_type="ephemeral",
                    text=f"Running `/{slash}`…",
                )
                await self._handle_slash_command(command)

            # Register Block Kit action handlers for approval buttons
            for _action_id in (
                "hermes_approve_once",
                "hermes_approve_session",
                "hermes_approve_always",
                "hermes_deny",
            ):
                self._app.action(_action_id)(self._handle_approval_action)

            # Register Block Kit action handlers for slash-confirm buttons
            # (generic three-option prompts; see tools/slash_confirm.py).
            for _action_id in (
                "hermes_confirm_once",
                "hermes_confirm_always",
                "hermes_confirm_cancel",
            ):
                self._app.action(_action_id)(self._handle_slash_confirm_action)

            # Register plugin-provided Block Kit action handlers.
            #
            # Plugins call ``ctx.register_slack_action_handler(action_id, cb)``
            # at register() time; the manager queues them and the adapter
            # wires them into AsyncApp here so slack_bolt's matcher knows
            # about them before Socket Mode starts dispatching events.
            #
            # Each callback is wrapped so a misbehaving plugin can't take
            # down the gateway: any exception inside the plugin handler is
            # caught and logged, and slack_bolt still sees a clean ack.
            try:
                from hermes_cli.plugins import get_plugin_manager
                _plugin_handlers = get_plugin_manager().get_slack_action_handlers()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "[Slack] Could not load plugin action handlers: %s", e,
                )
                _plugin_handlers = []

            # Closure factory — keeps the wrapper's signature limited to
            # ``(ack, body, action)``. slack_bolt inspects listener
            # signatures via ``inspect.signature`` and passes ``None`` for
            # any parameter name it doesn't recognise, so capturing loop
            # vars as default args (``_cb=_cb`` etc.) silently clobbers
            # them at dispatch time.
            def _make_wrapper(cb, plugin_name):
                async def _wrapped(ack, body, action):
                    try:
                        await cb(ack, body, action)
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.error(
                            "[Slack] Plugin '%s' action handler raised: %s",
                            plugin_name, exc, exc_info=True,
                        )
                        # Best-effort ack so Slack doesn't retry the click.
                        try:
                            await ack()
                        except Exception:
                            pass
                return _wrapped

            for _action_id, _cb, _plugin_name in _plugin_handlers:
                self._app.action(_action_id)(_make_wrapper(_cb, _plugin_name))
                logger.debug(
                    "[Slack] Registered plugin action handler %s (from %s)",
                    _action_id, _plugin_name,
                )
            if _plugin_handlers:
                logger.info(
                    "[Slack] Wired %d plugin action handler(s)",
                    len(_plugin_handlers),
                )

            # Bring up the handler and watchdog atomically. ``_running`` only
            # flips to True after the handler is alive so the watchdog loop
            # observes the live task immediately; on any failure here we tear
            # down whatever we managed to start, leave ``_running=False``, and
            # let the ``finally`` block release the platform lock cleanly.
            try:
                self._start_socket_mode_handler()
                self._running = True
                self._ensure_socket_watchdog()
            except Exception:
                self._running = False
                try:
                    await self._stop_socket_mode_handler()
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug(
                        "[Slack] Cleanup after failed start raised", exc_info=True
                    )
                raise

            logger.info(
                "[Slack] Socket Mode connected (%d workspace(s))",
                len(self._team_clients),
            )
            return True

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[Slack] Connection failed: %s", e, exc_info=True)
            return False
        finally:
            if lock_acquired and not self._running:
                self._release_platform_lock()

    async def create_handoff_thread(
        self,
        parent_chat_id: str,
        name: str,
    ) -> Optional[str]:
        """Create a Slack thread anchor for a session handoff.

        Slack threads are anchored to a parent message (``thread_ts``), not
        a channel-level construct. So we post a seed message into the home
        channel and return its ``ts`` — the watcher uses that as the
        ``thread_id`` for subsequent sends.

        Returns the seed message ts as a string, or ``None`` on failure.
        """
        if not self._app:
            return None
        try:
            client = self._get_client(parent_chat_id)
            if client is None:
                return None
            seed_text = (
                f":thread: Hermes handoff — *{(name or 'session').strip()[:80]}*"
            )
            result = await client.chat_postMessage(
                channel=parent_chat_id,
                text=seed_text,
            )
            ts = (
                result.get("ts")
                if isinstance(result, dict)
                else getattr(result, "get", lambda _k, _d=None: None)("ts")
            )
            if ts:
                return str(ts)
        except Exception as exc:
            logger.warning(
                "[%s] Handoff thread: seed-post failed for channel %s: %s",
                self.name,
                parent_chat_id,
                exc,
            )
        return None

    async def disconnect(self) -> None:
        """Disconnect from Slack."""
        self._running = False

        watchdog_task = self._socket_watchdog_task
        self._socket_watchdog_task = None
        if watchdog_task is not None and not watchdog_task.done():
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive logging
                # Watchdog may have lost the cancellation race and exited with
                # an unrelated exception. Log and continue so handler cleanup
                # and lock release still happen.
                logger.debug(
                    "[Slack] Watchdog task raised during disconnect", exc_info=True
                )

        await self._stop_socket_mode_handler()
        self._app = None
        self._app_token = None
        self._proxy_url = None

        self._release_platform_lock()

        logger.info("[Slack] Disconnected")

    def _get_client(self, chat_id: str) -> Any:
        """Return the workspace-specific WebClient for a channel."""
        team_id = self._channel_team.get(chat_id)
        if team_id and team_id in self._team_clients:
            return self._team_clients[team_id]
        return self._app.client  # fallback to primary

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message to a Slack channel or DM."""
        if not self._app:
            return SendResult(success=False, error="Not connected")

        try:
            # Check for a pending slash-command context.  When the user ran a
            # native slash command (e.g. /q, /stop, /model), the initial ack
            # already showed an ephemeral "Running /cmd…" message.  If we have
            # a stashed response_url for this channel, replace that ack with
            # the actual command reply ephemerally instead of posting publicly.
            slash_ctx = self._pop_slash_context(chat_id)
            if slash_ctx:
                return await self._send_slash_ephemeral(
                    slash_ctx,
                    content,
                )

            # Convert standard markdown → Slack mrkdwn
            formatted = self.format_message(content)

            # Split long messages, preserving code block boundaries
            chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)

            thread_ts = self._resolve_thread_ts(reply_to, metadata)
            last_result = None

            # reply_broadcast: also post thread replies to the main channel.
            # Controlled via platform config: gateway.slack.reply_broadcast
            broadcast = self.config.extra.get("reply_broadcast", False)

            for i, chunk in enumerate(chunks):
                kwargs = {
                    "channel": chat_id,
                    "text": chunk,
                    "mrkdwn": True,
                }
                if thread_ts:
                    kwargs["thread_ts"] = thread_ts
                    # Only broadcast the first chunk of the first reply
                    if broadcast and i == 0:
                        kwargs["reply_broadcast"] = True

                last_result = await self._get_client(chat_id).chat_postMessage(**kwargs)

            # Clear Slack Assistant status as soon as the final message is posted.
            if thread_ts:
                await self.stop_typing(chat_id)

            # Track the sent message ts so we can auto-respond to thread
            # replies without requiring @mention.
            sent_ts = last_result.get("ts") if last_result else None
            if sent_ts:
                self._bot_message_ts.add(sent_ts)
                # Also register the thread root so replies-to-my-replies work
                if thread_ts:
                    self._bot_message_ts.add(thread_ts)
                if len(self._bot_message_ts) > self._BOT_TS_MAX:
                    excess = len(self._bot_message_ts) - self._BOT_TS_MAX // 2
                    for old_ts in list(self._bot_message_ts)[:excess]:
                        self._bot_message_ts.discard(old_ts)

            return SendResult(
                success=True,
                message_id=sent_ts,
                raw_response=last_result,
            )

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[Slack] Send error: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def send_private_notice(
        self,
        chat_id: str,
        user_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a Slack ephemeral message visible only to one user."""
        if not self._app:
            return SendResult(success=False, error="Not connected")
        if not chat_id or not user_id:
            return SendResult(success=False, error="chat_id and user_id are required")

        try:
            formatted = self.format_message(content)
            thread_ts = self._resolve_thread_ts(reply_to, metadata)
            kwargs = {
                "channel": chat_id,
                "user": user_id,
                "text": formatted,
                "mrkdwn": True,
            }
            if thread_ts:
                kwargs["thread_ts"] = thread_ts

            result = await self._get_client(chat_id).chat_postEphemeral(**kwargs)
            return SendResult(
                success=True,
                message_id=result.get("message_ts") or result.get("ts"),
                raw_response=result,
            )
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[Slack] Ephemeral send error: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Edit a previously sent Slack message."""
        if not self._app:
            return SendResult(success=False, error="Not connected")
        try:
            formatted = self.format_message(content)
            await self._get_client(chat_id).chat_update(
                channel=chat_id,
                ts=message_id,
                text=formatted,
            )
            if finalize:
                await self.stop_typing(chat_id)
            return SendResult(success=True, message_id=message_id)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[Slack] Failed to edit message %s in channel %s: %s",
                message_id,
                chat_id,
                e,
                exc_info=True,
            )
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Show a typing/status indicator using assistant.threads.setStatus.

        Displays "is thinking..." next to the bot name in a thread.
        Requires the assistant:write or chat:write scope.
        Auto-clears when the bot sends a reply to the thread.
        """
        if not self._app:
            return

        thread_ts = None
        if metadata:
            thread_ts = metadata.get("thread_id") or metadata.get("thread_ts")

        if not thread_ts:
            return  # Can only set status in a thread context

        self._active_status_threads[chat_id] = thread_ts
        try:
            await self._get_client(chat_id).assistant_threads_setStatus(
                channel_id=chat_id,
                thread_ts=thread_ts,
                status="is thinking...",
            )
        except Exception as e:
            # Silently ignore — may lack assistant:write scope or not be
            # in an assistant-enabled context. Falls back to reactions.
            logger.debug("[Slack] assistant.threads.setStatus failed: %s", e)

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        """Clear the assistant thread status indicator."""
        if not self._app:
            return
        thread_ts = self._active_status_threads.pop(chat_id, None)
        if not thread_ts:
            return
        try:
            await self._get_client(chat_id).assistant_threads_setStatus(
                channel_id=chat_id,
                thread_ts=thread_ts,
                status="",
            )
        except Exception as e:
            logger.debug("[Slack] assistant.threads.setStatus clear failed: %s", e)

    def _dm_top_level_threads_as_sessions(self) -> bool:
        """Whether top-level Slack DMs get per-message session threads.

        Defaults to ``True`` so each visible DM reply thread is isolated as its
        own Hermes session — matching the per-thread behavior channels already
        have.  Set ``platforms.slack.extra.dm_top_level_threads_as_sessions``
        to ``false`` in config.yaml to revert to the legacy behavior where all
        top-level DMs share one continuous session.
        """
        raw = self.config.extra.get("dm_top_level_threads_as_sessions")
        if raw is None:
            return True  # default: each DM thread is its own session
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _resolve_thread_ts(
        self,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Resolve the correct thread_ts for a Slack API call.

        Prefers metadata thread_id (the thread parent's ts, set by the
        gateway) over reply_to (which may be a child message's ts).

        When ``reply_in_thread`` is ``false`` in the platform extra config,
        top-level channel messages receive direct channel replies instead of
        thread replies.  Messages that originate inside an existing thread are
        always replied to in-thread to preserve conversation context.
        """
        # When reply_in_thread is disabled (default: True for backward compat),
        # only thread messages that are already part of an existing thread.
        # For top-level channel messages, the inbound handler sets
        # metadata.thread_id to the message's own ts as a session-keying
        # fallback (see the `thread_ts = event.get("thread_ts") or ts` branch),
        # so metadata alone can't distinguish a real thread reply from a
        # top-level message. reply_to is the incoming message's own id, so
        # when thread_id == reply_to the "thread" is synthetic and we reply
        # directly in the channel instead.
        if not self.config.extra.get("reply_in_thread", True):
            md = metadata or {}
            existing_thread = md.get("thread_id") or md.get("thread_ts")
            if existing_thread and reply_to and existing_thread == reply_to:
                existing_thread = None
            return existing_thread or None

        if metadata:
            if metadata.get("thread_id"):
                return metadata["thread_id"]
            if metadata.get("thread_ts"):
                return metadata["thread_ts"]
        return reply_to

    async def _upload_file(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a local file to Slack."""
        if not self._app:
            return SendResult(success=False, error="Not connected")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        thread_ts = self._resolve_thread_ts(reply_to, metadata)
        last_exc = None
        for attempt in range(3):
            try:
                result = await self._get_client(chat_id).files_upload_v2(
                    channel=chat_id,
                    file=file_path,
                    filename=os.path.basename(file_path),
                    initial_comment=caption or "",
                    thread_ts=thread_ts,
                )
                self._record_uploaded_file_thread(chat_id, thread_ts)
                return SendResult(success=True, raw_response=result)
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable_upload_error(exc) or attempt >= 2:
                    raise
                logger.debug(
                    "[Slack] Upload retry %d/2 for %s: %s",
                    attempt + 1,
                    file_path,
                    exc,
                )
                await asyncio.sleep(1.5 * (attempt + 1))

        raise last_exc

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send a batch of images as a single Slack message with multiple file uploads.

        Uses ``files_upload_v2`` with its ``file_uploads`` parameter so all
        images show up attached to one ``initial_comment`` message instead
        of N separate messages. Falls back to the base per-image loop on
        any failure.

        The batch limit is 10 file uploads per call (Slack server-side cap).
        """
        if not self._app:
            return
        if not images:
            return

        try:
            import httpx as _httpx
            from urllib.parse import unquote as _unquote
            from tools.url_safety import is_safe_url as _is_safe_url
        except Exception:
            await super().send_multiple_images(chat_id, images, metadata, human_delay)
            return

        thread_ts = self._resolve_thread_ts(None, metadata)

        CHUNK = 10
        chunks = [images[i : i + CHUNK] for i in range(0, len(images), CHUNK)]

        for chunk_idx, chunk in enumerate(chunks):
            if human_delay > 0 and chunk_idx > 0:
                await asyncio.sleep(human_delay)

            file_uploads: List[Dict[str, Any]] = []
            initial_comment_parts: List[str] = []
            try:
                async with _httpx.AsyncClient(
                    timeout=30.0, follow_redirects=True
                ) as http_client:
                    for image_url, alt_text in chunk:
                        if alt_text:
                            initial_comment_parts.append(alt_text)

                        if image_url.startswith("file://"):
                            local_path = _unquote(image_url[7:])
                            if not os.path.exists(local_path):
                                logger.warning(
                                    "[Slack] Skipping missing image: %s", local_path
                                )
                                continue
                            file_uploads.append(
                                {
                                    "file": local_path,
                                    "filename": os.path.basename(local_path),
                                }
                            )
                        else:
                            if not _is_safe_url(image_url):
                                logger.warning(
                                    "[Slack] Blocked unsafe image URL in batch"
                                )
                                continue
                            try:
                                response = await http_client.get(image_url)
                                response.raise_for_status()
                                ext = "png"
                                ct = response.headers.get("content-type", "")
                                if "jpeg" in ct or "jpg" in ct:
                                    ext = "jpg"
                                elif "gif" in ct:
                                    ext = "gif"
                                elif "webp" in ct:
                                    ext = "webp"
                                file_uploads.append(
                                    {
                                        "content": response.content,
                                        "filename": f"image_{len(file_uploads)}.{ext}",
                                    }
                                )
                            except Exception as dl_err:
                                logger.warning(
                                    "[Slack] Download failed for %s: %s",
                                    safe_url_for_log(image_url),
                                    dl_err,
                                )
                                continue

                if not file_uploads:
                    continue

                initial_comment = (
                    "\n".join(initial_comment_parts) if initial_comment_parts else ""
                )
                logger.info(
                    "[Slack] Sending %d image(s) in single files_upload_v2 (chunk %d/%d)",
                    len(file_uploads),
                    chunk_idx + 1,
                    len(chunks),
                )
                result = await self._get_client(chat_id).files_upload_v2(
                    channel=chat_id,
                    file_uploads=file_uploads,
                    initial_comment=initial_comment,
                    thread_ts=thread_ts,
                )
                self._record_uploaded_file_thread(chat_id, thread_ts)
                _ = result
            except Exception as e:
                logger.warning(
                    "[Slack] Multi-image files_upload_v2 failed (chunk %d/%d), falling back to per-image: %s",
                    chunk_idx + 1,
                    len(chunks),
                    e,
                    exc_info=True,
                )
                await super().send_multiple_images(
                    chat_id, chunk, metadata, human_delay=human_delay
                )

    def _record_uploaded_file_thread(
        self, chat_id: str, thread_ts: Optional[str]
    ) -> None:
        """Treat successful file uploads as bot participation in a thread."""
        if not thread_ts:
            return
        self._bot_message_ts.add(thread_ts)
        if len(self._bot_message_ts) > self._BOT_TS_MAX:
            excess = len(self._bot_message_ts) - self._BOT_TS_MAX // 2
            for old_ts in list(self._bot_message_ts)[:excess]:
                self._bot_message_ts.discard(old_ts)

    def _is_retryable_upload_error(self, exc: Exception) -> bool:
        """Best-effort detection for transient Slack upload failures."""
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code is not None:
            return status_code == 429 or status_code >= 500

        body = " ".join(
            str(part)
            for part in (
                exc,
                getattr(exc, "message", ""),
                getattr(exc, "response", None),
            )
            if part
        ).lower()
        if "rate_limited" in body or "ratelimited" in body or "429" in body:
            return True
        if (
            "connection reset" in body
            or "service unavailable" in body
            or "temporarily unavailable" in body
        ):
            return True
        return self._is_retryable_error(body)

    # ----- Markdown → mrkdwn conversion -----

    def format_message(self, content: str) -> str:
        """Convert standard markdown to Slack mrkdwn format.

        Protected regions (code blocks, inline code) are extracted first so
        their contents are never modified.  Standard markdown constructs
        (headers, bold, italic, links) are translated to mrkdwn syntax.
        """
        if not content:
            return content

        placeholders: dict = {}
        counter = [0]

        def _ph(value: str) -> str:
            """Stash value behind a placeholder that survives later passes."""
            key = f"\x00SL{counter[0]}\x00"
            counter[0] += 1
            placeholders[key] = value
            return key

        text = content

        # 1) Protect fenced code blocks (``` ... ```)
        text = re.sub(
            r"(```(?:[^\n]*\n)?[\s\S]*?```)",
            lambda m: _ph(m.group(0)),
            text,
        )

        # 2) Protect inline code (`...`)
        text = re.sub(r"(`[^`]+`)", lambda m: _ph(m.group(0)), text)

        # 3) Convert markdown links [text](url) → <url|text>
        def _convert_markdown_link(m):
            label = m.group(1)
            url = m.group(2).strip()
            if url.startswith("<") and url.endswith(">"):
                url = url[1:-1].strip()
            return _ph(f"<{url}|{label}>")

        text = re.sub(
            r"(?<!!)\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)",
            _convert_markdown_link,
            text,
        )

        # 4) Protect existing Slack entities/manual links so escaping and later
        #    formatting passes don't break them.
        text = re.sub(
            r"(<(?:[@#!]|(?:https?|mailto|tel):)[^>\n]+>)",
            lambda m: _ph(m.group(1)),
            text,
        )

        # 5) Protect blockquote markers before escaping
        text = re.sub(r"^(>+\s)", lambda m: _ph(m.group(0)), text, flags=re.MULTILINE)

        # 6) Escape Slack control characters in remaining plain text.
        # Unescape first so already-escaped input doesn't get double-escaped.
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # 7) Convert headers (## Title) → *Title* (bold)
        def _convert_header(m):
            inner = m.group(1).strip()
            # Strip redundant bold markers inside a header
            inner = re.sub(r"\*\*(.+?)\*\*", r"\1", inner)
            return _ph(f"*{inner}*")

        text = re.sub(r"^#{1,6}\s+(.+)$", _convert_header, text, flags=re.MULTILINE)

        # 8) Convert bold+italic: ***text*** → *_text_* (Slack bold wrapping italic)
        text = re.sub(
            r"\*\*\*(.+?)\*\*\*",
            lambda m: _ph(f"*_{m.group(1)}_*"),
            text,
        )

        # 9) Convert bold: **text** → *text* (Slack bold)
        text = re.sub(
            r"\*\*(.+?)\*\*",
            lambda m: _ph(f"*{m.group(1)}*"),
            text,
        )

        # 10) Convert italic: _text_ stays as _text_ (already Slack italic)
        #     Single *text* → _text_ (Slack italic), but only when the
        #     emphasized text touches non-whitespace on both sides so literal
        #     delimiters like "a * b * c" are preserved.
        text = re.sub(
            r"(?<!\*)\*(\S(?:[^*\n]*?\S)?)\*(?!\*)",
            lambda m: _ph(f"_{m.group(1)}_"),
            text,
        )

        # 11) Convert strikethrough: ~~text~~ → ~text~
        text = re.sub(
            r"~~(.+?)~~",
            lambda m: _ph(f"~{m.group(1)}~"),
            text,
        )

        # 12) Blockquotes: > prefix is already protected by step 5 above.

        # 13) Restore placeholders in reverse order
        for key in reversed(placeholders):
            text = text.replace(key, placeholders[key])

        return text

    # ----- Reactions -----

    async def _add_reaction(self, channel: str, timestamp: str, emoji: str) -> bool:
        """Add an emoji reaction to a message. Returns True on success."""
        if not self._app:
            return False
        try:
            await self._get_client(channel).reactions_add(
                channel=channel, timestamp=timestamp, name=emoji
            )
            return True
        except Exception as e:
            # Don't log as error — may fail if already reacted or missing scope
            logger.debug("[Slack] reactions.add failed (%s): %s", emoji, e)
            return False

    async def _remove_reaction(self, channel: str, timestamp: str, emoji: str) -> bool:
        """Remove an emoji reaction from a message. Returns True on success."""
        if not self._app:
            return False
        try:
            await self._get_client(channel).reactions_remove(
                channel=channel, timestamp=timestamp, name=emoji
            )
            return True
        except Exception as e:
            logger.debug("[Slack] reactions.remove failed (%s): %s", emoji, e)
            return False

    def _reactions_enabled(self) -> bool:
        """Check if message reactions are enabled via config/env."""
        return os.getenv("SLACK_REACTIONS", "true").lower() not in {"false", "0", "no"}

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add an in-progress reaction when message processing begins."""
        if not self._reactions_enabled():
            return
        ts = getattr(event, "message_id", None)
        if not ts or ts not in self._reacting_message_ids:
            return
        channel_id = getattr(event.source, "chat_id", None)
        if channel_id:
            await self._add_reaction(channel_id, ts, "eyes")

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        """Swap the in-progress reaction for a final success/failure reaction."""
        if not self._reactions_enabled():
            return
        ts = getattr(event, "message_id", None)
        if not ts or ts not in self._reacting_message_ids:
            return
        self._reacting_message_ids.discard(ts)
        channel_id = getattr(event.source, "chat_id", None)
        if not channel_id:
            return
        await self._remove_reaction(channel_id, ts, "eyes")
        if outcome == ProcessingOutcome.SUCCESS:
            await self._add_reaction(channel_id, ts, "white_check_mark")
        elif outcome == ProcessingOutcome.FAILURE:
            await self._add_reaction(channel_id, ts, "x")

    # ----- User identity resolution -----

    async def _resolve_user_name(self, user_id: str, chat_id: str = "") -> str:
        """Resolve a Slack user ID to a display name, with caching."""
        if not user_id:
            return ""
        if user_id in self._user_name_cache:
            return self._user_name_cache[user_id]

        if not self._app:
            return user_id

        try:
            client = self._get_client(chat_id) if chat_id else self._app.client
            result = await client.users_info(user=user_id)
            user = result.get("user", {})
            # Prefer display_name → real_name → user_id
            profile = user.get("profile", {})
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or user.get("real_name")
                or user.get("name")
                or user_id
            )
            self._user_name_cache[user_id] = name
            return name
        except Exception as e:
            logger.debug("[Slack] users.info failed for %s: %s", user_id, e)
            self._user_name_cache[user_id] = user_id
            return user_id

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local image file to Slack by uploading it."""
        try:
            return await self._upload_file(
                chat_id, image_path, caption, reply_to, metadata
            )
        except FileNotFoundError:
            return SendResult(
                success=False, error=f"Image file not found: {image_path}"
            )
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[%s] Failed to send local Slack image %s: %s",
                self.name,
                image_path,
                e,
                exc_info=True,
            )
            text = f"🖼️ Image: {image_path}"
            if caption:
                text = f"{caption}\n{text}"
            return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image to Slack by uploading the URL as a file."""
        if not self._app:
            return SendResult(success=False, error="Not connected")

        from tools.url_safety import is_safe_url

        if not is_safe_url(image_url):
            logger.warning("[Slack] Blocked unsafe image URL (SSRF protection)")
            return await super().send_image(
                chat_id, image_url, caption, reply_to, metadata=metadata
            )

        try:
            import httpx

            async def _ssrf_redirect_guard(response):
                """Re-check redirect targets so public URLs cannot bounce into private IPs."""
                if response.is_redirect and response.next_request:
                    redirect_url = str(response.next_request.url)
                    if not is_safe_url(redirect_url):
                        raise ValueError("Blocked redirect to private/internal address")

            # Download the image first
            async with httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                event_hooks={"response": [_ssrf_redirect_guard]},
            ) as client:
                response = await client.get(image_url)
                response.raise_for_status()

            thread_ts = self._resolve_thread_ts(reply_to, metadata)
            result = await self._get_client(chat_id).files_upload_v2(
                channel=chat_id,
                content=response.content,
                filename="image.png",
                initial_comment=caption or "",
                thread_ts=thread_ts,
            )
            self._record_uploaded_file_thread(chat_id, thread_ts)

            return SendResult(success=True, raw_response=result)

        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning(
                "[Slack] Failed to upload image from URL %s, falling back to text: %s",
                safe_url_for_log(image_url),
                e,
                exc_info=True,
            )
            # Fall back to sending the URL as text
            text = f"{caption}\n{image_url}" if caption else image_url
            return await self.send(
                chat_id=chat_id,
                content=text,
                reply_to=reply_to,
                metadata=metadata,
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
        """Send an audio file to Slack."""
        try:
            return await self._upload_file(
                chat_id, audio_path, caption, reply_to, metadata
            )
        except FileNotFoundError:
            return SendResult(
                success=False, error=f"Audio file not found: {audio_path}"
            )
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[Slack] Failed to send audio file %s: %s",
                audio_path,
                e,
                exc_info=True,
            )
            return SendResult(success=False, error=str(e))

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a video file to Slack."""
        if not self._app:
            return SendResult(success=False, error="Not connected")

        if not os.path.exists(video_path):
            return SendResult(
                success=False, error=f"Video file not found: {video_path}"
            )

        try:
            thread_ts = self._resolve_thread_ts(reply_to, metadata)
            last_exc = None
            for attempt in range(3):
                try:
                    result = await self._get_client(chat_id).files_upload_v2(
                        channel=chat_id,
                        file=video_path,
                        filename=os.path.basename(video_path),
                        initial_comment=caption or "",
                        thread_ts=thread_ts,
                    )
                    self._record_uploaded_file_thread(chat_id, thread_ts)
                    return SendResult(success=True, raw_response=result)
                except Exception as exc:
                    last_exc = exc
                    if not self._is_retryable_upload_error(exc) or attempt >= 2:
                        raise
                    logger.debug(
                        "[Slack] Video upload retry %d/2 for %s: %s",
                        attempt + 1,
                        video_path,
                        exc,
                    )
                    await asyncio.sleep(1.5 * (attempt + 1))

            raise last_exc

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[%s] Failed to send video %s: %s",
                self.name,
                video_path,
                e,
                exc_info=True,
            )
            text = f"🎬 Video: {video_path}"
            if caption:
                text = f"{caption}\n{text}"
            return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a document/file attachment to Slack."""
        if not self._app:
            return SendResult(success=False, error="Not connected")

        if not os.path.exists(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")

        display_name = file_name or os.path.basename(file_path)
        thread_ts = self._resolve_thread_ts(reply_to, metadata)

        try:
            last_exc = None
            for attempt in range(3):
                try:
                    result = await self._get_client(chat_id).files_upload_v2(
                        channel=chat_id,
                        file=file_path,
                        filename=display_name,
                        initial_comment=caption or "",
                        thread_ts=thread_ts,
                    )
                    self._record_uploaded_file_thread(chat_id, thread_ts)
                    return SendResult(success=True, raw_response=result)
                except Exception as exc:
                    last_exc = exc
                    if not self._is_retryable_upload_error(exc) or attempt >= 2:
                        raise
                    logger.debug(
                        "[Slack] Document upload retry %d/2 for %s: %s",
                        attempt + 1,
                        file_path,
                        exc,
                    )
                    await asyncio.sleep(1.5 * (attempt + 1))

            raise last_exc

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[%s] Failed to send document %s: %s",
                self.name,
                file_path,
                e,
                exc_info=True,
            )
            text = f"📎 File: {file_path}"
            if caption:
                text = f"{caption}\n{text}"
            return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a Slack channel."""
        if not self._app:
            return {"name": chat_id, "type": "unknown"}

        try:
            result = await self._get_client(chat_id).conversations_info(channel=chat_id)
            channel = result.get("channel", {})
            is_dm = channel.get("is_im", False)
            return {
                "name": channel.get("name", chat_id),
                "type": "dm" if is_dm else "group",
            }
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[Slack] Failed to fetch chat info for %s: %s",
                chat_id,
                e,
                exc_info=True,
            )
            return {"name": chat_id, "type": "unknown"}

    # ----- Internal handlers -----

    def _assistant_thread_key(
        self, channel_id: str, thread_ts: str
    ) -> Optional[Tuple[str, str]]:
        """Return a stable cache key for Slack assistant thread metadata."""
        if not channel_id or not thread_ts:
            return None
        return (str(channel_id), str(thread_ts))

    def _extract_assistant_thread_metadata(self, event: dict) -> Dict[str, str]:
        """Extract Slack Assistant thread identity data from an event payload."""
        assistant_thread = event.get("assistant_thread") or {}
        context = assistant_thread.get("context") or event.get("context") or {}

        channel_id = (
            assistant_thread.get("channel_id")
            or event.get("channel")
            or context.get("channel_id")
            or ""
        )
        thread_ts = (
            assistant_thread.get("thread_ts")
            or event.get("thread_ts")
            or event.get("message_ts")
            or ""
        )
        user_id = (
            assistant_thread.get("user_id")
            or event.get("user")
            or context.get("user_id")
            or ""
        )
        team_id = (
            event.get("team")
            or event.get("team_id")
            or assistant_thread.get("team_id")
            or ""
        )
        context_channel_id = context.get("channel_id") or ""

        return {
            "channel_id": str(channel_id) if channel_id else "",
            "thread_ts": str(thread_ts) if thread_ts else "",
            "user_id": str(user_id) if user_id else "",
            "team_id": str(team_id) if team_id else "",
            "context_channel_id": str(context_channel_id) if context_channel_id else "",
        }

    def _cache_assistant_thread_metadata(self, metadata: Dict[str, str]) -> None:
        """Remember assistant thread identity data for later message events."""
        channel_id = metadata.get("channel_id", "")
        thread_ts = metadata.get("thread_ts", "")
        key = self._assistant_thread_key(channel_id, thread_ts)
        if not key:
            return

        existing = self._assistant_threads.get(key, {})
        merged = dict(existing)
        merged.update({k: v for k, v in metadata.items() if v})
        self._assistant_threads[key] = merged

        # Evict oldest entries when the cache exceeds the limit
        if len(self._assistant_threads) > self._ASSISTANT_THREADS_MAX:
            excess = len(self._assistant_threads) - self._ASSISTANT_THREADS_MAX // 2
            for old_key in list(self._assistant_threads)[:excess]:
                del self._assistant_threads[old_key]

        team_id = merged.get("team_id", "")
        if team_id and channel_id:
            self._channel_team[channel_id] = team_id

    def _lookup_assistant_thread_metadata(
        self,
        event: dict,
        channel_id: str = "",
        thread_ts: str = "",
    ) -> Dict[str, str]:
        """Load cached assistant-thread metadata that matches the current event."""
        metadata = self._extract_assistant_thread_metadata(event)
        if channel_id and not metadata.get("channel_id"):
            metadata["channel_id"] = channel_id
        if thread_ts and not metadata.get("thread_ts"):
            metadata["thread_ts"] = thread_ts

        key = self._assistant_thread_key(
            metadata.get("channel_id", ""),
            metadata.get("thread_ts", ""),
        )
        cached = self._assistant_threads.get(key, {}) if key else {}
        if cached:
            merged = dict(cached)
            merged.update({k: v for k, v in metadata.items() if v})
            return merged
        return metadata

    def _seed_assistant_thread_session(self, metadata: Dict[str, str]) -> None:
        """Prime the session store so assistant threads get stable user scoping."""
        session_store = getattr(self, "_session_store", None)
        if not session_store:
            return

        channel_id = metadata.get("channel_id", "")
        thread_ts = metadata.get("thread_ts", "")
        user_id = metadata.get("user_id", "")
        if not channel_id or not thread_ts or not user_id:
            return

        source = self.build_source(
            chat_id=channel_id,
            chat_name=channel_id,
            chat_type="dm",
            user_id=user_id,
            thread_id=thread_ts,
            chat_topic=metadata.get("context_channel_id") or None,
        )

        try:
            session_store.get_or_create_session(source)
        except Exception:
            logger.debug(
                "[Slack] Failed to seed assistant thread session for %s/%s",
                channel_id,
                thread_ts,
                exc_info=True,
            )

    async def _handle_assistant_thread_lifecycle_event(self, event: dict) -> None:
        """Handle Slack Assistant lifecycle events that carry user/thread identity."""
        metadata = self._extract_assistant_thread_metadata(event)
        self._cache_assistant_thread_metadata(metadata)
        self._seed_assistant_thread_session(metadata)

    async def _handle_slack_file_shared(self, event: dict) -> None:
        """Fallback for Slack file shares that do not arrive as message.files.

        Slack documents ``file_shared`` as a file-ID-only event; callers must
        fetch ``files.info`` to get the file object. Keep this intentionally
        narrow: normal image/audio/document uploads already arrive on the
        message event, but some video shares have only been observed through
        this lifecycle event.
        """
        channel_id = event.get("channel_id") or event.get("channel") or ""
        file_id = event.get("file_id") or (event.get("file") or {}).get("id") or ""
        if not channel_id or not file_id:
            return

        team_id = event.get("team_id") or event.get("team") or ""
        try:
            client = self._team_clients.get(team_id) if team_id else None
            info_resp = await (client or self._get_client(channel_id)).files_info(
                file=file_id
            )
        except Exception as exc:
            response = getattr(exc, "response", None)
            detail = self._describe_slack_api_error(response, file_obj={"id": file_id})
            logger.warning("[Slack] files.info error for file_shared %s: %s", file_id, detail or exc)
            return

        if not info_resp.get("ok"):
            detail = self._describe_slack_api_error(info_resp, file_obj={"id": file_id})
            logger.warning(
                "[Slack] files.info failed for file_shared %s: %s",
                file_id,
                detail or info_resp.get("error"),
            )
            return

        file_obj = info_resp.get("file") or {}
        if not str(file_obj.get("mimetype", "")).startswith("video/"):
            return

        share = None
        for bucket in (file_obj.get("shares") or {}).values():
            if not isinstance(bucket, dict):
                continue
            channel_shares = bucket.get(channel_id)
            if channel_shares:
                share = channel_shares[0]
                break
            if share is None:
                for shares in bucket.values():
                    if shares:
                        share = shares[0]
                        break
        share = share or {}
        ts = share.get("ts") or event.get("event_ts") or ""
        thread_ts = share.get("thread_ts") or ""

        # Give Slack's normal message.file_share event a chance to arrive first.
        # If it does, _handle_slack_message records the same share ts and this
        # fallback skips instead of duplicating the user turn.
        await asyncio.sleep(0.75)
        if ts and self._dedup.is_duplicate(ts):
            return

        fallback_event = {
            "type": "message",
            "subtype": "file_share",
            "text": "",
            "user": event.get("user_id") or file_obj.get("user", ""),
            "channel": channel_id,
            "channel_type": "im" if channel_id.startswith("D") else "channel",
            "team": team_id,
            "ts": "",  # already recorded above; avoid tripping our own dedup guard
            "files": [file_obj],
        }
        if thread_ts and thread_ts != ts:
            fallback_event["thread_ts"] = thread_ts
        await self._handle_slack_message(fallback_event)

    async def _handle_slack_message(self, event: dict) -> None:
        """Handle an incoming Slack message event."""
        # Dedup: Slack Socket Mode can redeliver events after reconnects (#4777)
        event_ts = event.get("ts", "")
        if event_ts and self._dedup.is_duplicate(event_ts):
            return

        # Bot message filtering (SLACK_ALLOW_BOTS / config allow_bots):
        #   "none"     — ignore all bot messages (default, backward-compatible)
        #   "mentions" — accept bot messages only when they @mention us
        #   "all"      — accept all bot messages (except our own)
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            allow_bots = self.config.extra.get("allow_bots", "")
            if not allow_bots:
                allow_bots = os.getenv("SLACK_ALLOW_BOTS", "none")
            allow_bots = str(allow_bots).lower().strip()
            if allow_bots == "none":
                return
            elif allow_bots == "mentions":
                text_check = event.get("text", "")
                if self._bot_user_id and f"<@{self._bot_user_id}>" not in text_check:
                    return
            # "all" falls through to process the message
            # Always ignore our own messages to prevent echo loops
            msg_user = event.get("user", "")
            if msg_user and self._bot_user_id and msg_user == self._bot_user_id:
                return

        # Ignore message edits and deletions
        subtype = event.get("subtype")
        if subtype in {"message_changed", "message_deleted"}:
            return

        original_text = event.get("text", "")

        # Slack blocks native slash commands inside threads ("/queue is not
        # supported in threads. Sorry!").  As a workaround, recognise a
        # leading ``!`` as an alternate command prefix and rewrite it to
        # ``/`` so the rest of the pipeline (MessageType.COMMAND tagging,
        # gateway dispatcher) handles it like a normal slash command.  Only
        # rewrite when the first token resolves to a known gateway command
        # so casual messages like "!nice work" pass through unchanged.
        if original_text.startswith("!"):
            try:
                from hermes_cli.commands import is_gateway_known_command

                first_token = original_text[1:].split(maxsplit=1)[0]
                # Strip "@suffix" the same way get_command() does, so
                # forms like ``!stop@hermes`` still resolve.
                cmd_name = first_token.split("@", 1)[0].lower()
                if (
                    cmd_name
                    and "/" not in cmd_name
                    and is_gateway_known_command(cmd_name)
                ):
                    original_text = "/" + original_text[1:]
            except Exception:  # pragma: no cover - defensive
                pass

        text = original_text

        # Extract quoted/forwarded content from Slack blocks.
        # Slack's modern composer embeds forwarded messages in the ``blocks``
        # array as ``rich_text_quote`` elements, which are NOT reflected in
        # the plain ``text`` field.  Merge block text so the agent sees the
        # full message content.
        blocks = event.get("blocks")
        if blocks:
            blocks_text = _extract_text_from_slack_blocks(blocks)
            if blocks_text:
                # Only append if the blocks contain text not already present
                # in the plain text field (avoids duplication).
                stripped_blocks = blocks_text.strip()
                if stripped_blocks and stripped_blocks not in text.strip():
                    logger.debug(
                        "Slack: extracted additional text from blocks "
                        "(likely quoted/forwarded content): %s",
                        stripped_blocks[:300],
                    )
                    text = (text.strip() + "\n" + stripped_blocks).strip()

            blocks_payload = _serialize_slack_blocks_for_agent(blocks)
            if blocks_payload:
                text = (text.strip() + "\n\n" + blocks_payload).strip()

        # Extract link unfurls / rich attachments (e.g. Notion previews).
        # Slack places unfurled link previews in the ``attachments`` array with
        # fields like title, title_link/from_url, text, footer, and fallback.
        # Without reading these, the agent never sees shared link previews.
        slack_attachments = event.get("attachments") or []
        if slack_attachments:
            att_parts: list[str] = []
            for att in slack_attachments:
                att_title = att.get("title", "")
                att_url = att.get("title_link", "") or att.get("from_url", "")
                att_text = att.get("text", "")
                att_footer = att.get("footer", "")
                att_fallback = att.get("fallback", "")

                # Skip message-type attachments (e.g. Slack bot messages with
                # is_msg_unfurl) to avoid echoing our own content.
                if att.get("is_msg_unfurl"):
                    continue

                # Build a readable representation.
                if att_title and att_url:
                    header = f"📎 [{att_title}]({att_url})"
                elif att_title:
                    header = f"📎 {att_title}"
                elif att_url:
                    header = f"📎 {att_url}"
                else:
                    header = None

                # Prefer preview text, fall back to fallback description.
                body = att_text or att_fallback or ""
                if body:
                    body = body.strip()
                    if len(body) > 500:
                        body = body[:497] + "..."

                if header and body:
                    section = f"{header}\n   {body}"
                elif header:
                    section = header
                elif body:
                    section = f"📎 {body}"
                else:
                    continue

                # Deduplicate only when the fully rendered section is already
                # present. The shared URL often already appears in the user's
                # message text, and skipping on URL/title alone would hide the
                # preview body we actually want the agent to see.
                if section in text:
                    continue

                if att_footer:
                    section = f"{section}\n   _{att_footer}_"

                att_parts.append(section)

            if att_parts:
                attachment_text = "\n\n".join(att_parts)
                text = (text.strip() + "\n\n" + attachment_text).strip()
                logger.debug(
                    "Slack: appended %d link unfurl(s) to message text",
                    len(att_parts),
                )

        channel_id = event.get("channel", "")
        ts = event.get("ts", "")
        assistant_meta = self._lookup_assistant_thread_metadata(
            event,
            channel_id=channel_id,
            thread_ts=event.get("thread_ts", ""),
        )
        user_id = event.get("user") or assistant_meta.get("user_id", "")
        if not channel_id:
            channel_id = assistant_meta.get("channel_id", "")
        team_id = (
            event.get("team")
            or event.get("team_id")
            or assistant_meta.get("team_id", "")
        )

        # Track which workspace owns this channel
        if team_id and channel_id:
            self._channel_team[channel_id] = team_id

        # Determine if this is a DM or channel message
        channel_type = event.get("channel_type", "")
        if not channel_type and channel_id.startswith("D"):
            channel_type = "im"
        is_dm = channel_type in {"im", "mpim"}  # Both 1:1 and group DMs

        # Build thread_ts for session keying.
        # In channels: fall back to ts so each top-level @mention starts a
        #   new thread/session (the bot always replies in a thread).
        # In DMs: fall back to ts so each top-level DM reply thread gets
        #   its own session key (matching channel behavior). Set
        #   dm_top_level_threads_as_sessions: false in config to revert to
        #   legacy single-session-per-DM-channel behavior.
        if is_dm:
            thread_ts = event.get("thread_ts") or assistant_meta.get("thread_ts")
            if not thread_ts and self._dm_top_level_threads_as_sessions():
                thread_ts = ts
        else:
            # Channel message session scoping.
            #
            # Three cases:
            #   (a) genuine thread reply   → scope session per thread
            #   (b) top-level, reply_in_thread=true (the default)  →
            #       legacy behaviour: each top-level message becomes its
            #       own thread, so the UX still "replies in a thread"
            #       and sessions are keyed per thread root
            #   (c) top-level, reply_in_thread=false → scope one session
            #       across the whole channel so context accumulates across
            #       messages (#15421 bug 1)
            event_thread_ts_raw = event.get("thread_ts")
            # Align with ``is_thread_reply`` below — a ``thread_ts ==
            # ts`` payload (some thread-root shapes) is not a real reply
            # and must not prevent the shared-session path from taking
            # effect.  Matching the same invariant here keeps the two
            # branches in sync even if Slack introduces new payload
            # variants (Copilot on #15464).
            if event_thread_ts_raw and event_thread_ts_raw != ts:
                thread_ts = event_thread_ts_raw
            elif self.config.extra.get("reply_in_thread", True):
                # Legacy default: treat ts as a synthetic thread root so
                # this top-level message gets its own session.
                thread_ts = ts
            else:
                # reply_in_thread=false: no thread key → session manager
                # groups by (platform, channel_id, None) and the channel
                # shares one conversation.  reply_to_message_id at the
                # outbound side is already gated on ``thread_ts != ts``
                # so None here produces a non-threaded reply without
                # further changes.
                thread_ts = None

        # In channels, respond if:
        #   0. Channel is in free_response_channels, OR require_mention is
        #      disabled — always process regardless of mention.
        #   1. The bot is @mentioned in this message, OR
        #   2. The message is a reply in a thread the bot started/participated in, OR
        #   3. The message is in a thread where the bot was previously @mentioned, OR
        #   4. There's an existing session for this thread (survives restarts)
        bot_uid = self._team_bot_user_ids.get(team_id, self._bot_user_id)
        routing_text = original_text or ""
        is_mentioned = bool(
            (bot_uid and f"<@{bot_uid}>" in routing_text)
            or self._slack_message_matches_mention_patterns(routing_text)
        )
        event_thread_ts = event.get("thread_ts")
        is_thread_reply = bool(event_thread_ts and event_thread_ts != ts)

        if not is_dm and bot_uid:
            # Check allowed channels — if set, only respond in these channels (whitelist)
            allowed_channels = self._slack_allowed_channels()
            if allowed_channels and channel_id not in allowed_channels:
                logger.debug(
                    "[Slack] Ignoring message in non-allowed channel: %s", channel_id
                )
                return

            if channel_id in self._slack_free_response_channels():
                pass  # Free-response channel — always process
            elif not self._slack_require_mention():
                pass  # Mention requirement disabled globally for Slack
            elif self._slack_strict_mention() and not is_mentioned:
                return  # Strict mode: ignore until @-mentioned again
            elif not is_mentioned:
                reply_to_bot_thread = (
                    is_thread_reply and event_thread_ts in self._bot_message_ts
                )
                in_mentioned_thread = (
                    event_thread_ts is not None
                    and event_thread_ts in self._mentioned_threads
                )
                has_session = is_thread_reply and self._has_active_session_for_thread(
                    channel_id=channel_id,
                    thread_ts=event_thread_ts,
                    user_id=user_id,
                )
                if (
                    not reply_to_bot_thread
                    and not in_mentioned_thread
                    and not has_session
                ):
                    return

        if is_mentioned:
            # Strip the bot mention from the text
            text = text.replace(f"<@{bot_uid}>", "").strip()
            # Register this thread so all future messages auto-trigger the bot.
            # Skipped in strict mode: strict_mention=true bots must be
            # re-mentioned every turn, so remembering the thread would
            # defeat the feature (and re-enable agent-to-agent ack loops).
            if event_thread_ts and not self._slack_strict_mention():
                self._mentioned_threads.add(event_thread_ts)
                if len(self._mentioned_threads) > self._MENTIONED_THREADS_MAX:
                    to_remove = list(self._mentioned_threads)[
                        : self._MENTIONED_THREADS_MAX // 2
                    ]
                    for t in to_remove:
                        self._mentioned_threads.discard(t)

        # When entering a thread for the first time (no existing session),
        # fetch thread context so the agent understands the conversation.
        if is_thread_reply and not self._has_active_session_for_thread(
            channel_id=channel_id,
            thread_ts=event_thread_ts,
            user_id=user_id,
        ):
            thread_context = await self._fetch_thread_context(
                channel_id=channel_id,
                thread_ts=event_thread_ts,
                current_ts=ts,
                team_id=team_id,
            )
            if thread_context:
                text = thread_context + text

        # Determine message type
        msg_type = MessageType.TEXT
        if (original_text or "").startswith("/"):
            msg_type = MessageType.COMMAND

        # Handle file attachments
        media_urls = []
        media_types = []
        attachment_notices: List[str] = []
        files = event.get("files", [])
        for f in files:
            # Slack Connect channels return stub file objects with
            # file_access="check_file_info" and no URL fields. We must
            # call files.info to retrieve the full object (including url_private_download)
            # before we can download it.
            # https://docs.slack.dev/reference/objects/file-object/#slack_connect_files
            if f.get("file_access") == "check_file_info":
                file_id = f.get("id")
                if not file_id:
                    continue
                try:
                    info_resp = await self._get_client(channel_id).files_info(
                        file=file_id
                    )
                    if info_resp.get("ok"):
                        f = info_resp["file"]
                    else:
                        detail = self._describe_slack_api_error(info_resp, file_obj=f)
                        if detail:
                            attachment_notices.append(detail)
                            logger.warning("[Slack] %s", detail)
                        else:
                            logger.warning(
                                "[Slack] files.info failed for %s: %s",
                                file_id,
                                info_resp.get("error"),
                            )
                        continue
                except Exception as e:
                    response = getattr(e, "response", None)
                    detail = self._describe_slack_api_error(response, file_obj=f)
                    if detail:
                        attachment_notices.append(detail)
                        logger.warning("[Slack] %s", detail)
                    else:
                        logger.warning(
                            "[Slack] files.info error for %s: %s",
                            file_id,
                            e,
                            exc_info=True,
                        )
                    continue

            mimetype = f.get("mimetype", "unknown")
            url = f.get("url_private_download") or f.get("url_private", "")
            if mimetype.startswith("image/") and url:
                try:
                    ext = "." + mimetype.split("/")[-1].split(";")[0]
                    if ext not in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                        ext = ".jpg"
                    # Slack private URLs require the bot token as auth header
                    cached = await self._download_slack_file(url, ext, team_id=team_id)
                    media_urls.append(cached)
                    media_types.append(mimetype)
                except Exception as e:  # pragma: no cover - defensive logging
                    detail = self._describe_slack_download_failure(e, file_obj=f)
                    if detail:
                        attachment_notices.append(detail)
                        logger.warning("[Slack] %s", detail)
                    else:
                        logger.warning(
                            "[Slack] Failed to cache image from %s: %s",
                            url,
                            e,
                            exc_info=True,
                        )
            elif mimetype.startswith("audio/") and url:
                try:
                    ext = _resolve_slack_audio_ext(f, mimetype)
                    cached = await self._download_slack_file(
                        url, ext, audio=True, team_id=team_id
                    )
                    media_urls.append(cached)
                    media_types.append(mimetype)
                except Exception as e:  # pragma: no cover - defensive logging
                    detail = self._describe_slack_download_failure(e, file_obj=f)
                    if detail:
                        attachment_notices.append(detail)
                        logger.warning("[Slack] %s", detail)
                    else:
                        logger.warning(
                            "[Slack] Failed to cache audio from %s: %s",
                            url,
                            e,
                            exc_info=True,
                        )
            elif mimetype.startswith("video/") and url and _is_slack_voice_clip(f):
                # Slack in-app voice clips are audio-only MP4 containers that
                # Slack sometimes mislabels with a ``video/mp4`` mimetype.
                # Cache them as audio and report an ``audio/*`` type so the
                # gateway routes them to speech-to-text instead of video
                # understanding. Without this, voice messages recorded in Slack
                # never get transcribed.
                try:
                    ext = _resolve_slack_audio_ext(f, mimetype)
                    cached = await self._download_slack_file(
                        url, ext, audio=True, team_id=team_id
                    )
                    media_urls.append(cached)
                    # Report a coherent audio mimetype matching the cached
                    # extension so downstream STT routing recognizes it.
                    media_types.append(
                        _SLACK_EXT_TO_AUDIO_MIME.get(ext, "audio/mp4")
                    )
                    logger.debug(
                        "[Slack] Cached voice clip (mislabeled %s) as audio: %s",
                        mimetype,
                        cached,
                    )
                except Exception as e:  # pragma: no cover - defensive logging
                    detail = self._describe_slack_download_failure(e, file_obj=f)
                    if detail:
                        attachment_notices.append(detail)
                        logger.warning("[Slack] %s", detail)
                    else:
                        logger.warning(
                            "[Slack] Failed to cache voice clip from %s: %s",
                            url,
                            e,
                            exc_info=True,
                        )
            elif mimetype.startswith("video/") and url:
                try:
                    original_filename = f.get("name", "")
                    _, ext = os.path.splitext(original_filename)
                    ext = ext.lower()
                    if ext not in SUPPORTED_VIDEO_TYPES:
                        mime_to_ext = {v: k for k, v in SUPPORTED_VIDEO_TYPES.items()}
                        ext = mime_to_ext.get(
                            mimetype.split(";", 1)[0].lower(), ".mp4"
                        )

                    raw_bytes = await self._download_slack_file_bytes(
                        url, team_id=team_id
                    )
                    cached_path = cache_video_from_bytes(raw_bytes, ext=ext)
                    media_urls.append(cached_path)
                    media_types.append(SUPPORTED_VIDEO_TYPES.get(ext, mimetype))
                    logger.debug("[Slack] Cached user video: %s", cached_path)
                except Exception as e:  # pragma: no cover - defensive logging
                    detail = self._describe_slack_download_failure(e, file_obj=f)
                    if detail:
                        attachment_notices.append(detail)
                        logger.warning("[Slack] %s", detail)
                    else:
                        logger.warning(
                            "[Slack] Failed to cache video from %s: %s",
                            url,
                            e,
                            exc_info=True,
                        )
            elif url:
                # Try to handle as a document attachment
                try:
                    original_filename = f.get("name", "")
                    ext = ""
                    if original_filename:
                        _, ext = os.path.splitext(original_filename)
                        ext = ext.lower()

                    # Fallback: reverse-lookup from MIME type
                    if not ext and mimetype:
                        mime_to_ext = {
                            v: k for k, v in SUPPORTED_DOCUMENT_TYPES.items()
                        }
                        ext = mime_to_ext.get(mimetype, "")

                    # Any file type is accepted — authorization to message the
                    # agent is the gate, not the file extension. Known types keep
                    # their precise MIME; unknown types fall back to the source
                    # mimetype or octet-stream so the agent reaches for terminal
                    # tools.
                    in_allowlist = ext in SUPPORTED_DOCUMENT_TYPES

                    # Check file size (Slack limit: 20 MB for bots)
                    file_size = f.get("size", 0)
                    MAX_DOC_BYTES = 20 * 1024 * 1024
                    if not file_size or file_size > MAX_DOC_BYTES:
                        logger.warning(
                            "[Slack] Document too large or unknown size: %s", file_size
                        )
                        continue

                    # Download and cache
                    raw_bytes = await self._download_slack_file_bytes(
                        url, team_id=team_id
                    )
                    cached_path = cache_document_from_bytes(
                        raw_bytes, original_filename or f"document{ext or '.bin'}"
                    )
                    if in_allowlist:
                        doc_mime = SUPPORTED_DOCUMENT_TYPES[ext]
                    else:
                        doc_mime = mimetype or "application/octet-stream"
                    media_urls.append(cached_path)
                    media_types.append(doc_mime)
                    logger.debug("[Slack] Cached user document: %s (%s)", cached_path, doc_mime)

                    # Inject small text-ish files directly into the prompt so
                    # snippets like JSON/YAML/configs are actually visible to the
                    # agent. Gate on a text-like extension/MIME — NOT a blind
                    # UTF-8 decode, since binary formats (PDF/zip/docx) can have
                    # decodable ASCII headers. Binary files are surfaced as a
                    # cached path only (run.py emits a path-pointing note).
                    MAX_TEXT_INJECT_BYTES = 100 * 1024
                    _is_text = ext in _TEXT_INJECT_EXTENSIONS or (mimetype or "").startswith("text/")
                    if _is_text and len(raw_bytes) <= MAX_TEXT_INJECT_BYTES:
                        try:
                            text_content = raw_bytes.decode("utf-8")
                            display_name = original_filename or f"document{ext or '.txt'}"
                            display_name = re.sub(r"[^\w.\- ]", "_", display_name)
                            injection = f"[Content of {display_name}]:\n{text_content}"
                            if text:
                                text = f"{injection}\n\n{text}"
                            else:
                                text = injection
                        except UnicodeDecodeError:
                            pass  # Binary content, skip injection

                except Exception as e:  # pragma: no cover - defensive logging
                    detail = self._describe_slack_download_failure(e, file_obj=f)
                    if detail:
                        attachment_notices.append(detail)
                        logger.warning("[Slack] %s", detail)
                    else:
                        logger.warning(
                            "[Slack] Failed to cache document from %s: %s",
                            url,
                            e,
                            exc_info=True,
                        )

        if attachment_notices:
            notice_block = "[Slack attachment notice]\n" + "\n".join(
                f"- {n}" for n in attachment_notices
            )
            text = f"{notice_block}\n\n{text}" if text else notice_block

        if msg_type != MessageType.COMMAND and media_types:
            if any(m.startswith("image/") for m in media_types):
                msg_type = MessageType.PHOTO
            elif any(m.startswith("video/") for m in media_types):
                msg_type = MessageType.VIDEO
            elif any(m.startswith("audio/") for m in media_types):
                msg_type = MessageType.VOICE
            else:
                msg_type = MessageType.DOCUMENT

        # Resolve user display name (cached after first lookup)
        user_name = await self._resolve_user_name(user_id, chat_id=channel_id)

        # Build source
        source = self.build_source(
            chat_id=channel_id,
            chat_name=channel_id,  # Will be resolved later if needed
            chat_type="dm" if is_dm else "group",
            user_id=user_id,
            user_name=user_name,
            thread_id=thread_ts,
        )

        # Per-channel ephemeral prompt
        from gateway.platforms.base import (
            resolve_channel_prompt,
            resolve_channel_skills,
        )

        _channel_prompt = resolve_channel_prompt(
            self.config.extra,
            channel_id,
            None,
        )
        _auto_skill = resolve_channel_skills(
            self.config.extra,
            channel_id,
            None,
        )

        # Extract reply context if this message is a thread reply.
        # Mirrors the Telegram/Discord implementations so that gateway.run
        # can inject a `[Replying to: "..."]` prefix when the parent is not
        # already in the session history. Uses the thread-context cache when
        # available to avoid redundant conversations.replies calls.
        reply_to_text = None
        if thread_ts and thread_ts != ts:
            try:
                reply_to_text = (
                    await self._fetch_thread_parent_text(
                        channel_id=channel_id,
                        thread_ts=thread_ts,
                        team_id=team_id,
                    )
                    or None
                )
            except Exception:  # pragma: no cover - defensive
                reply_to_text = None

        msg_event = MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=event,
            message_id=ts,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=thread_ts if thread_ts != ts else None,
            channel_prompt=_channel_prompt,
            reply_to_text=reply_to_text,
            auto_skill=_auto_skill,
        )

        # Only react when bot is directly addressed (DM or @mention).
        # In listen-all channels (require_mention=false), reacting to every
        # casual message would be noisy.
        _should_react = (is_dm or is_mentioned) and self._reactions_enabled()
        if _should_react:
            self._reacting_message_ids.add(ts)

        await self.handle_message(msg_event)

    # ----- Approval button support (Block Kit) -----

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a Block Kit approval prompt with interactive buttons.

        The buttons call ``resolve_gateway_approval()`` to unblock the waiting
        agent thread — same mechanism as the text ``/approve`` flow.
        """
        if not self._app:
            return SendResult(success=False, error="Not connected")

        try:
            thread_ts = self._resolve_thread_ts(None, metadata)

            # Slack hard-caps a section block's text at 3000 chars; an
            # oversized block fails the whole send with ``invalid_blocks``
            # and the gateway falls back to the plain-text prompt (no
            # buttons).  execute_code approvals embed the entire script in
            # ``command``, so budget the preview against the fixed parts
            # instead of a flat truncation that overflows once the header +
            # reason are added.
            header = ":warning: *Command Approval Required*\n"
            reason = f"Reason: {description[:500]}"
            budget = 3000 - len(header) - len(reason) - len("``````\n") - len("...")
            cmd_preview = command[:budget] + "..." if len(command) > budget else command

            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{header}```{cmd_preview}```\n{reason}",
                    },
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Allow Once"},
                            "style": "primary",
                            "action_id": "hermes_approve_once",
                            "value": session_key,
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Allow Session"},
                            "action_id": "hermes_approve_session",
                            "value": session_key,
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Always Allow"},
                            "action_id": "hermes_approve_always",
                            "value": session_key,
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Deny"},
                            "style": "danger",
                            "action_id": "hermes_deny",
                            "value": session_key,
                        },
                    ],
                },
            ]

            kwargs: Dict[str, Any] = {
                "channel": chat_id,
                "text": f"⚠️ Command approval required: {cmd_preview[:100]}",
                "blocks": blocks,
            }
            if thread_ts:
                kwargs["thread_ts"] = thread_ts

            result = await self._get_client(chat_id).chat_postMessage(**kwargs)
            msg_ts = result.get("ts", "")
            if msg_ts:
                self._approval_resolved[msg_ts] = False

            return SendResult(success=True, message_id=msg_ts, raw_response=result)
        except Exception as e:
            logger.error("[Slack] send_exec_approval failed: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a Block Kit three-option slash-command confirmation prompt."""
        if not self._app:
            return SendResult(success=False, error="Not connected")

        try:
            thread_ts = self._resolve_thread_ts(None, metadata)
            # Same 3000-char section-block cap as send_exec_approval: budget
            # the body against the rendered title so the wrapper never pushes
            # the block over the limit (overflow → invalid_blocks → no buttons).
            _title = (title or "Confirm")[:150]
            budget = 3000 - len(f"*{_title}*\n\n") - len("...")
            body = message[:budget] + "..." if len(message) > budget else message
            # Encode session_key and confirm_id into the button value so the
            # callback handler can resolve without extra bookkeeping.
            value = f"{session_key}|{confirm_id}"

            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{_title}*\n\n{body}",
                    },
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Approve Once"},
                            "style": "primary",
                            "action_id": "hermes_confirm_once",
                            "value": value,
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Always Approve"},
                            "action_id": "hermes_confirm_always",
                            "value": value,
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Cancel"},
                            "style": "danger",
                            "action_id": "hermes_confirm_cancel",
                            "value": value,
                        },
                    ],
                },
            ]

            kwargs: Dict[str, Any] = {
                "channel": chat_id,
                "text": f"{title or 'Confirm'}: {body[:100]}",
                "blocks": blocks,
            }
            if thread_ts:
                kwargs["thread_ts"] = thread_ts

            result = await self._get_client(chat_id).chat_postMessage(**kwargs)
            return SendResult(
                success=True, message_id=result.get("ts", ""), raw_response=result
            )
        except Exception as e:
            logger.error("[Slack] send_slash_confirm failed: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e))

    def _is_interactive_user_authorized(
        self,
        user_id: str,
        *,
        channel_id: str = "",
        user_name: Optional[str] = None,
    ) -> bool:
        """Return whether a Slack interactive caller may perform gated actions."""
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return False

        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        auth_fn = getattr(runner, "_is_user_authorized", None)
        if callable(auth_fn):
            try:
                from gateway.session import SessionSource

                source = SessionSource(
                    platform=Platform.SLACK,
                    chat_id=str(channel_id or normalized_user_id),
                    chat_type="dm" if str(channel_id or "").startswith("D") else "group",
                    user_id=normalized_user_id,
                    user_name=str(user_name).strip() if user_name else None,
                )
                return bool(auth_fn(source))
            except Exception:
                logger.debug(
                    "[Slack] Falling back to env-only interactive auth for user %s",
                    normalized_user_id,
                    exc_info=True,
                )

        if os.getenv("SLACK_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}:
            return True

        allowed_ids = set()
        platform_allowlist = os.getenv("SLACK_ALLOWED_USERS", "").strip()
        if platform_allowlist:
            allowed_ids.update(uid.strip() for uid in platform_allowlist.split(",") if uid.strip())
        global_allowlist = os.getenv("GATEWAY_ALLOWED_USERS", "").strip()
        if global_allowlist:
            allowed_ids.update(uid.strip() for uid in global_allowlist.split(",") if uid.strip())

        if allowed_ids:
            return "*" in allowed_ids or normalized_user_id in allowed_ids

        return os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}

    async def _handle_slash_confirm_action(self, ack, body, action) -> None:
        """Handle a slash-confirm button click from Block Kit."""
        await ack()

        action_id = action.get("action_id", "")
        value = action.get("value", "")
        message = body.get("message", {})
        msg_ts = message.get("ts", "")
        channel_id = body.get("channel", {}).get("id", "")
        user_name = body.get("user", {}).get("name", "unknown")
        user_id = body.get("user", {}).get("id", "")
        if not self._is_interactive_user_authorized(
            user_id,
            channel_id=channel_id,
            user_name=user_name,
        ):
            logger.warning(
                "[Slack] Unauthorized slash-confirm click by %s (%s) - ignoring",
                user_name, user_id,
            )
            return

        # Authorization — reuse the exec-approval allowlist.
        allowed_csv = ""  # Interactive auth already ran above.
        if allowed_csv:
            allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
            if "*" not in allowed_ids and user_id not in allowed_ids:
                logger.warning(
                    "[Slack] Unauthorized slash-confirm click by %s (%s) — ignoring",
                    user_name,
                    user_id,
                )
                return

        # Parse session_key|confirm_id back out
        if "|" not in value:
            logger.warning("[Slack] Malformed slash-confirm value: %s", value)
            return
        session_key, confirm_id = value.split("|", 1)

        choice_map = {
            "hermes_confirm_once": "once",
            "hermes_confirm_always": "always",
            "hermes_confirm_cancel": "cancel",
        }
        choice = choice_map.get(action_id, "cancel")

        label_map = {
            "once": f"✅ Approved once by {user_name}",
            "always": f"🔒 Always approved by {user_name}",
            "cancel": f"❌ Cancelled by {user_name}",
        }
        decision_text = label_map.get(choice, f"Resolved by {user_name}")

        # Pull original prompt body out of the section block so we can show
        # the decision inline without losing context.
        original_text = ""
        for block in message.get("blocks", []):
            if block.get("type") == "section":
                original_text = block.get("text", {}).get("text", "")
                break

        updated_blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": original_text or "Confirmation prompt",
                },
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": decision_text},
                ],
            },
        ]

        try:
            await self._get_client(channel_id).chat_update(
                channel=channel_id,
                ts=msg_ts,
                text=decision_text,
                blocks=updated_blocks,
            )
        except Exception as e:
            logger.warning("[Slack] Failed to update slash-confirm message: %s", e)

        # Resolve via the module-level primitive and post any follow-up.
        try:
            from tools import slash_confirm as _slash_confirm_mod

            result_text = await _slash_confirm_mod.resolve(
                session_key, confirm_id, choice
            )
            if result_text:
                post_kwargs: Dict[str, Any] = {
                    "channel": channel_id,
                    "text": result_text,
                }
                # Inherit the thread so the reply stays in the same place.
                thread_ts = message.get("thread_ts") or msg_ts
                if thread_ts:
                    post_kwargs["thread_ts"] = thread_ts
                await self._get_client(channel_id).chat_postMessage(**post_kwargs)
            logger.info(
                "Slack button resolved slash-confirm for session %s (choice=%s, user=%s)",
                session_key,
                choice,
                user_name,
            )
        except Exception as exc:
            logger.error(
                "Failed to resolve slash-confirm from Slack button: %s",
                exc,
                exc_info=True,
            )

    async def _handle_approval_action(self, ack, body, action) -> None:
        """Handle an approval button click from Block Kit."""
        await ack()

        action_id = action.get("action_id", "")
        session_key = action.get("value", "")
        message = body.get("message", {})
        msg_ts = message.get("ts", "")
        channel_id = body.get("channel", {}).get("id", "")
        user_name = body.get("user", {}).get("name", "unknown")
        user_id = body.get("user", {}).get("id", "")

        if not self._is_interactive_user_authorized(
            user_id,
            channel_id=channel_id,
            user_name=user_name,
        ):
            logger.warning(
                "[Slack] Unauthorized approval click by %s (%s) - ignoring",
                user_name, user_id,
            )
            return

        # Only authorized users may click approval buttons.  Button clicks
        # bypass the normal message auth flow in gateway/run.py, so we must
        # check here as well.
        allowed_csv = ""  # Interactive auth already ran above.
        if allowed_csv:
            allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
            if "*" not in allowed_ids and user_id not in allowed_ids:
                logger.warning(
                    "[Slack] Unauthorized approval click by %s (%s) — ignoring",
                    user_name,
                    user_id,
                )
                return

        # Map action_id to approval choice
        choice_map = {
            "hermes_approve_once": "once",
            "hermes_approve_session": "session",
            "hermes_approve_always": "always",
            "hermes_deny": "deny",
        }
        choice = choice_map.get(action_id, "deny")

        # Prevent double-clicks — atomic pop; first caller gets False, others get True (default)
        if self._approval_resolved.pop(msg_ts, True):
            return

        # Update the message to show the decision and remove buttons
        label_map = {
            "once": f"✅ Approved once by {user_name}",
            "session": f"✅ Approved for session by {user_name}",
            "always": f"✅ Approved permanently by {user_name}",
            "deny": f"❌ Denied by {user_name}",
        }
        decision_text = label_map.get(choice, f"Resolved by {user_name}")

        # Get original text from the section block
        original_text = ""
        for block in message.get("blocks", []):
            if block.get("type") == "section":
                original_text = block.get("text", {}).get("text", "")
                break

        updated_blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": original_text or "Command approval request",
                },
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": decision_text},
                ],
            },
        ]

        try:
            await self._get_client(channel_id).chat_update(
                channel=channel_id,
                ts=msg_ts,
                text=decision_text,
                blocks=updated_blocks,
            )
        except Exception as e:
            logger.warning("[Slack] Failed to update approval message: %s", e)

        # Resolve the approval — this unblocks the agent thread
        try:
            from tools.approval import resolve_gateway_approval

            count = resolve_gateway_approval(session_key, choice)
            logger.info(
                "Slack button resolved %d approval(s) for session %s (choice=%s, user=%s)",
                count,
                session_key,
                choice,
                user_name,
            )
        except Exception as exc:
            logger.error(
                "Failed to resolve gateway approval from Slack button: %s", exc
            )

        # (approval state already consumed by atomic pop above)

    # ----- Thread context fetching -----

    async def _fetch_thread_context(
        self,
        channel_id: str,
        thread_ts: str,
        current_ts: str,
        team_id: str = "",
        limit: int = 30,
    ) -> str:
        """Fetch recent thread messages to provide context when the bot is
        mentioned mid-thread for the first time.

        This method is only called when there is NO active session for the
        thread (guarded at the call site by _has_active_session_for_thread).
        That guard ensures thread messages are prepended only on the very
        first turn — after that the session history already holds them, so
        there is no duplication across subsequent turns.

        Results are cached for _THREAD_CACHE_TTL seconds per thread to avoid
        hammering conversations.replies (Tier 3, ~50 req/min).

        Returns a formatted string with prior thread history, or empty string
        on failure or if the thread has no prior messages.
        """
        cache_key = f"{channel_id}:{thread_ts}:{team_id}"
        now = time.monotonic()
        cached = self._thread_context_cache.get(cache_key)
        if cached and (now - cached.fetched_at) < self._THREAD_CACHE_TTL:
            return cached.content

        try:
            client = self._get_client(channel_id)

            # Retry with exponential backoff for Tier-3 rate limits (429).
            result = None
            for attempt in range(3):
                try:
                    result = await client.conversations_replies(
                        channel=channel_id,
                        ts=thread_ts,
                        limit=limit + 1,  # +1 because it includes the current message
                        inclusive=True,
                    )
                    break
                except Exception as exc:
                    # Check for rate-limit error from slack_sdk
                    err_str = str(exc).lower()
                    is_rate_limit = (
                        "ratelimited" in err_str
                        or "429" in err_str
                        or "rate_limited" in err_str
                    )
                    if is_rate_limit and attempt < 2:
                        retry_after = 1.0 * (2**attempt)  # 1s, 2s
                        logger.warning(
                            "[Slack] conversations.replies rate limited; retrying in %.1fs (attempt %d/3)",
                            retry_after,
                            attempt + 1,
                        )
                        await asyncio.sleep(retry_after)
                        continue
                    raise

            if result is None:
                return ""

            messages = result.get("messages", [])
            if not messages:
                return ""

            bot_uid = self._team_bot_user_ids.get(team_id, self._bot_user_id)
            context_parts = []
            parent_text = ""
            for msg in messages:
                msg_ts = msg.get("ts", "")
                # Exclude the current triggering message — it will be delivered
                # as the user message itself, so including it here would duplicate it.
                if msg_ts == current_ts:
                    continue

                is_parent = msg_ts == thread_ts
                is_bot = bool(msg.get("bot_id")) or msg.get("subtype") == "bot_message"
                msg_user = msg.get("user", "")

                # Identify "our own" bot for this workspace (multi-workspace safe).
                msg_team = msg.get("team") or team_id
                self_bot_uid = (
                    self._team_bot_user_ids.get(msg_team) if msg_team else None
                ) or self._bot_user_id

                # Exclude only our own prior bot replies (circular context).
                # Keep:
                #   - the thread parent even if it was posted by a bot
                #     (e.g. a cron job summary we are now replying to);
                #   - other bots' child messages (useful third-party context).
                if (
                    is_bot
                    and not is_parent
                    and self_bot_uid
                    and msg_user == self_bot_uid
                ):
                    continue

                msg_text = msg.get("text", "").strip()
                if not msg_text:
                    continue

                # Strip bot mentions from context messages
                if bot_uid:
                    msg_text = msg_text.replace(f"<@{bot_uid}>", "").strip()

                prefix = "[thread parent] " if is_parent else ""
                display_user = msg_user or "unknown"
                # Prefer the bot's own name when the message is a bot post.
                if is_bot and not display_user:
                    display_user = msg.get("username") or "bot"
                name = await self._resolve_user_name(display_user, chat_id=channel_id)
                context_parts.append(f"{prefix}{name}: {msg_text}")
                if is_parent:
                    parent_text = msg_text

            content = ""
            if context_parts:
                content = (
                    "[Thread context — prior messages in this thread (not yet in conversation history):]\n"
                    + "\n".join(context_parts)
                    + "\n[End of thread context]\n\n"
                )

            self._thread_context_cache[cache_key] = _ThreadContextCache(
                content=content,
                fetched_at=now,
                message_count=len(context_parts),
                parent_text=parent_text,
            )
            return content

        except Exception as e:
            logger.warning("[Slack] Failed to fetch thread context: %s", e)
            return ""

    async def _fetch_thread_parent_text(
        self,
        channel_id: str,
        thread_ts: str,
        team_id: str = "",
    ) -> str:
        """Return the raw text of the thread parent message (for reply_to_text).

        Uses the same per-thread cache as :meth:`_fetch_thread_context` to avoid
        hitting ``conversations.replies`` twice. Falls back to a cheap single-
        message fetch (``limit=1, inclusive=True``) when the cache is cold.

        Returns empty string on any failure — callers should treat an empty
        return as "no parent context to inject".
        """
        cache_key = f"{channel_id}:{thread_ts}:{team_id}"
        now = time.monotonic()
        cached = self._thread_context_cache.get(cache_key)
        if cached and (now - cached.fetched_at) < self._THREAD_CACHE_TTL:
            return cached.parent_text

        try:
            client = self._get_client(channel_id)
            result = await client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=1,
                inclusive=True,
            )
            messages = result.get("messages", []) if result else []
            if not messages:
                return ""
            parent = messages[0]
            if parent.get("ts", "") != thread_ts:
                return ""
            bot_uid = self._team_bot_user_ids.get(team_id, self._bot_user_id)
            text = (parent.get("text") or "").strip()
            if bot_uid:
                text = text.replace(f"<@{bot_uid}>", "").strip()
            return text
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[Slack] Failed to fetch thread parent text: %s", exc)
            return ""

    async def _handle_slash_command(self, command: dict) -> None:
        """Handle Slack slash commands.

        Every gateway command in COMMAND_REGISTRY is registered as a native
        Slack slash (``/btw``, ``/stop``, ``/model``, etc.), matching the
        Discord and Telegram model. The slash name itself is the command;
        any text after it is the argument list.

        The legacy ``/hermes <subcommand> [args]`` form is preserved for
        backward compatibility with older workspace manifests and for users
        who want a single entry point for free-form questions (``/hermes
        what's the weather`` — non-slash text is treated as a regular
        message).
        """
        slash_name = (command.get("command") or "").lstrip("/").strip()
        text = command.get("text", "").strip()
        user_id = command.get("user_id", "")
        channel_id = command.get("channel_id", "")
        team_id = command.get("team_id", "")

        # Track which workspace owns this channel
        if team_id and channel_id:
            self._channel_team[channel_id] = team_id

        if slash_name in {"hermes", ""}:
            # Legacy /hermes <subcommand> [args] routing + free-form questions.
            # Empty slash_name falls into this branch for backward compat
            # with any caller that didn't populate command["command"].
            from hermes_cli.commands import slack_subcommand_map

            subcommand_map = slack_subcommand_map()
            subcommand_map["compact"] = "/compress"
            # Guard against whitespace-only text where ``text`` is truthy but
            # ``text.split()`` returns ``[]`` (e.g. user sends ``/hermes   ``).
            parts = text.split() if text else []
            first_word = parts[0] if parts else ""
            if first_word in subcommand_map:
                rest = text[len(first_word) :].strip()
                text = (
                    f"{subcommand_map[first_word]} {rest}".strip()
                    if rest
                    else subcommand_map[first_word]
                )
            elif text:
                pass  # Treat as a regular question
            else:
                text = "/help"
        else:
            # Native slash — /<slash_name> [args].  Route directly through the
            # gateway command dispatcher by prepending the slash.
            text = f"/{slash_name} {text}".strip()

        # Slack slash commands can originate from DMs or shared channels.
        # Preserve DM semantics only for DM channel IDs; shared channels must
        # keep group semantics so different users do not collide into one
        # session key.
        is_dm = str(channel_id).startswith("D")
        source = self.build_source(
            chat_id=channel_id,
            chat_type="dm" if is_dm else "group",
            user_id=user_id,
        )

        event = MessageEvent(
            text=text,
            message_type=(
                MessageType.COMMAND if text.startswith("/") else MessageType.TEXT
            ),
            source=source,
            raw_message=command,
        )

        # Stash the Slack response_url so the first reply for this
        # channel+user can be routed ephemerally (replaces the initial
        # "Running /cmd…" ack shown by handle_hermes_command).
        # Only stash for COMMAND events (text starts with "/") — free-form
        # questions via "/hermes <question>" must produce public replies so
        # the whole channel can see the agent's answer.
        response_url = command.get("response_url", "")
        if response_url and user_id and channel_id and text.startswith("/"):
            self._slash_command_contexts[(channel_id, user_id)] = {
                "response_url": response_url,
                "ts": time.monotonic(),
            }

        # Set the ContextVar so send() can match the correct stashed
        # response_url even when multiple users slash concurrently.
        _slash_user_id_token = _slash_user_id.set(user_id or None)
        try:
            await self.handle_message(event)
        finally:
            _slash_user_id.reset(_slash_user_id_token)

    def _has_active_session_for_thread(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
    ) -> bool:
        """Check if there's an active session for a thread.

        Used to determine if thread replies without @mentions should be
        processed (they should if there's an active session).

        Uses ``build_session_key()`` as the single source of truth for key
        construction — avoids the bug where manual key building didn't
        respect ``thread_sessions_per_user`` and ``group_sessions_per_user``
        settings correctly.
        """
        session_store = getattr(self, "_session_store", None)
        if not session_store:
            return False

        try:
            from gateway.session import SessionSource, build_session_key

            source = SessionSource(
                platform=Platform.SLACK,
                chat_id=channel_id,
                chat_type="group",
                user_id=user_id,
                thread_id=thread_ts,
            )

            # Read session isolation settings from the store's config
            store_cfg = getattr(session_store, "config", None)
            gspu = (
                getattr(store_cfg, "group_sessions_per_user", True)
                if store_cfg
                else True
            )
            tspu = (
                getattr(store_cfg, "thread_sessions_per_user", False)
                if store_cfg
                else False
            )

            session_key = build_session_key(
                source,
                group_sessions_per_user=gspu,
                thread_sessions_per_user=tspu,
            )

            session_store._ensure_loaded()
            return session_key in session_store._entries
        except Exception:
            return False

    async def _download_slack_file(
        self, url: str, ext: str, audio: bool = False, team_id: str = ""
    ) -> str:
        """Download a Slack file using the bot token for auth, with retry."""
        import httpx

        bot_token = (
            self._team_clients[team_id].token
            if team_id and team_id in self._team_clients
            else self.config.token
        )

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for attempt in range(3):
                try:
                    response = await client.get(
                        url,
                        headers={"Authorization": f"Bearer {bot_token}"},
                    )
                    response.raise_for_status()

                    # Slack may return an HTML sign-in/redirect page
                    # instead of actual media bytes (e.g. expired token,
                    # restricted file access).  Detect this early so we
                    # don't cache bogus data and confuse downstream tools.
                    ct = response.headers.get("content-type", "")
                    if "text/html" in ct:
                        raise ValueError(
                            "Slack returned HTML instead of media "
                            f"(content-type: {ct}); "
                            "check bot token scopes and file permissions"
                        )

                    if audio:
                        from gateway.platforms.base import cache_audio_from_bytes

                        return cache_audio_from_bytes(response.content, ext)
                    else:
                        from gateway.platforms.base import cache_image_from_bytes

                        return cache_image_from_bytes(response.content, ext)
                except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                    if (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code < 429
                    ):
                        raise
                    if attempt < 2:
                        logger.debug(
                            "Slack file download retry %d/2 for %s: %s",
                            attempt + 1,
                            url[:80],
                            exc,
                        )
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise

    async def _download_slack_file_bytes(self, url: str, team_id: str = "") -> bytes:
        """Download a Slack file and return raw bytes, with retry."""
        import httpx

        bot_token = (
            self._team_clients[team_id].token
            if team_id and team_id in self._team_clients
            else self.config.token
        )

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            for attempt in range(3):
                try:
                    response = await client.get(
                        url,
                        headers={"Authorization": f"Bearer {bot_token}"},
                    )
                    response.raise_for_status()
                    ct = response.headers.get("content-type", "")
                    if "text/html" in ct:
                        raise ValueError(
                            "Slack returned HTML instead of file bytes "
                            f"(content-type: {ct}); "
                            "check bot token scopes and file permissions"
                        )
                    return response.content
                except (
                    httpx.TimeoutException,
                    httpx.HTTPStatusError,
                    ValueError,
                ) as exc:
                    if (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code < 429
                    ):
                        raise
                    if isinstance(exc, ValueError):
                        raise
                    if attempt < 2:
                        logger.debug(
                            "Slack file download retry %d/2 for %s: %s",
                            attempt + 1,
                            url[:80],
                            exc,
                        )
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise

    # ── Channel mention gating ─────────────────────────────────────────────

    def _slack_require_mention(self) -> bool:
        """Return whether channel messages require an explicit bot mention.

        Uses explicit-false parsing (like Discord/Matrix) rather than
        truthy parsing, since the safe default is True (gating on).
        Unrecognised or empty values keep gating enabled.
        """
        configured = self.config.extra.get("require_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() not in {"false", "0", "no", "off"}
            return bool(configured)
        return os.getenv("SLACK_REQUIRE_MENTION", "true").lower() not in {
            "false",
            "0",
            "no",
            "off",
        }

    def _slack_strict_mention(self) -> bool:
        """When true, channel threads require an explicit @-mention on every
        message. Disables all auto-triggers (mentioned-thread memory,
        bot-message follow-up, session-presence). Defaults to False.
        """
        configured = self.config.extra.get("strict_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return os.getenv("SLACK_STRICT_MENTION", "false").lower() in {
            "true",
            "1",
            "yes",
            "on",
        }

    def _slack_free_response_channels(self) -> set:
        """Return channel IDs where no @mention is required."""
        raw = self.config.extra.get("free_response_channels")
        if raw is None:
            raw = os.getenv("SLACK_FREE_RESPONSE_CHANNELS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        # Coerce non-list scalars (str/int/float) to str before splitting.
        # A bare numeric YAML value (`free_response_channels: 1234567890`) is
        # loaded as int and was previously falling through the isinstance(str)
        # branch to return an empty set.  str() here accepts whatever scalar
        # the YAML loader hands us without changing existing string/CSV
        # semantics.
        s = str(raw).strip() if raw is not None else ""
        if s:
            return {part.strip() for part in s.split(",") if part.strip()}
        return set()

    def _slack_allowed_channels(self) -> set:
        """Return the whitelist of channel IDs the bot will respond in.

        When non-empty, messages from channels NOT in this set are silently
        ignored — even if the bot is @mentioned.  DMs are never filtered.
        Empty set means no restriction (fully backward compatible).
        """
        raw = self.config.extra.get("allowed_channels")
        if raw is None:
            raw = os.getenv("SLACK_ALLOWED_CHANNELS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        if isinstance(raw, str) and raw.strip():
            return {part.strip() for part in raw.split(",") if part.strip()}
        return set()

    def _slack_mention_patterns(self) -> List["re.Pattern"]:
        """Compile optional regex wake-word patterns for channel triggers.

        Parity with the other adapters (Telegram, DingTalk, Mattermost,
        WhatsApp, BlueBubbles, Photon): when ``require_mention`` is on, a
        channel message matching one of these patterns triggers the bot even
        without a literal ``<@BOTUID>`` mention. Reads ``slack.mention_patterns``
        (a list or single string) or ``SLACK_MENTION_PATTERNS`` (a JSON list, or
        newline/comma-separated values). Compiled patterns are cached on the
        instance. Previously this documented field was silently dropped.
        """
        cached = getattr(self, "_compiled_mention_patterns", None)
        if cached is not None:
            return cached

        patterns = self.config.extra.get("mention_patterns") if self.config.extra else None
        if patterns is None:
            raw = os.getenv("SLACK_MENTION_PATTERNS", "").strip()
            if raw:
                try:
                    import json as _json
                    patterns = _json.loads(raw)
                except Exception:
                    patterns = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]

        if isinstance(patterns, str):
            patterns = [patterns]

        compiled: List["re.Pattern"] = []
        if isinstance(patterns, list):
            for pat in patterns:
                if not isinstance(pat, str) or not pat.strip():
                    continue
                try:
                    compiled.append(re.compile(pat, re.IGNORECASE))
                except re.error as exc:
                    logger.warning("[Slack] Invalid mention pattern %r: %s", pat, exc)
        elif patterns is not None:
            logger.warning(
                "[Slack] mention_patterns must be a list or string; got %s",
                type(patterns).__name__,
            )

        if compiled:
            logger.info("[Slack] Loaded %d mention pattern(s)", len(compiled))
        self._compiled_mention_patterns = compiled
        return compiled

    def _slack_message_matches_mention_patterns(self, text: str) -> bool:
        """Return True when ``text`` matches a configured wake-word pattern."""
        if not text:
            return False
        return any(pattern.search(text) for pattern in self._slack_mention_patterns())


# ──────────────────────────────────────────────────────────────────────────
# Plugin migration glue (#41112 / #3823)
#
# Everything below this line was added when the Slack adapter moved from
# ``gateway/platforms/slack.py`` into this bundled plugin. It mirrors the
# Discord migration (PR #24356) exactly: a ``register(ctx)`` entry point plus
# the hook implementations (``_standalone_send``, ``interactive_setup``,
# ``_apply_yaml_config``, ``_is_connected``, ``_build_adapter``) that replace
# the per-platform core touchpoints (the ``Platform.SLACK`` elif in
# ``gateway/run.py``, the ``slack_cfg`` YAML→env block in ``gateway/config.py``,
# the ``_setup_slack`` wizard + ``_PLATFORMS["slack"]`` static dict in
# ``hermes_cli/{setup,gateway}.py``, and the ``_send_slack`` dispatch in
# ``tools/send_message_tool.py``).
# ──────────────────────────────────────────────────────────────────────────


async def _standalone_send(
    pconfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """Out-of-process Slack delivery via the Web API ``chat.postMessage``.

    Implements the ``standalone_sender_fn`` contract so ``deliver=slack`` cron
    jobs succeed when the cron process is not co-located with the gateway (the
    in-process adapter weakref is ``None`` in that case). Replaces the legacy
    ``_send_slack`` helper that used to live in ``tools/send_message_tool.py``.

    mrkdwn formatting is applied exactly as the legacy core path did — via a
    throwaway ``SlackAdapter`` instance's ``format_message`` — so cron-delivered
    Slack messages render identically to gateway-delivered ones.
    """
    token = getattr(pconfig, "token", None) or os.getenv("SLACK_BOT_TOKEN", "")
    if not token:
        return {"error": "Slack send failed: SLACK_BOT_TOKEN not configured"}

    formatted = message
    if message:
        try:
            _fmt_adapter = SlackAdapter.__new__(SlackAdapter)
            formatted = _fmt_adapter.format_message(message)
        except Exception:
            logger.debug(
                "Failed to apply Slack mrkdwn formatting in _standalone_send",
                exc_info=True,
            )

    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}

    try:
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp

        _proxy = resolve_proxy_url()
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
        url = "https://slack.com/api/chat.postMessage"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30), **_sess_kw
        ) as session:
            payload = {"channel": chat_id, "text": formatted, "mrkdwn": True}
            if thread_id:
                payload["thread_ts"] = thread_id
            async with session.post(
                url, headers=headers, json=payload, **_req_kw
            ) as resp:
                data = await resp.json()
                if data.get("ok"):
                    return {
                        "success": True,
                        "platform": "slack",
                        "chat_id": chat_id,
                        "message_id": data.get("ts"),
                    }
                return {"error": f"Slack API error: {data.get('error', 'unknown')}"}
    except Exception as e:
        return {"error": f"Slack send failed: {e}"}


def interactive_setup() -> None:
    """Guide the user through Slack bot setup.

    Mirrors Discord's ``interactive_setup`` shape: lazy-imports CLI helpers so
    the plugin's import surface stays small, generates and writes the Slack app
    manifest, prompts for the bot + app tokens, captures an allowlist, and
    offers to set a home channel. Replaces ``hermes_cli/setup.py::_setup_slack``.
    """
    from pathlib import Path
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli.cli_output import (
        prompt,
        prompt_yes_no,
        print_header,
        print_info,
        print_success,
        print_warning,
    )

    def _write_slack_manifest_and_instruct() -> None:
        """Generate the Slack manifest, write it under HERMES_HOME, and print
        paste-into-Slack instructions. Failures are non-fatal."""
        try:
            from hermes_cli.slack_cli import _build_full_manifest
            from hermes_constants import get_hermes_home
            import json as _json

            manifest = _build_full_manifest(
                bot_name="Hermes",
                bot_description="Your Hermes agent on Slack",
            )
            target = Path(get_hermes_home()) / "slack-manifest.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                _json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print_success(f"Slack app manifest written to: {target}")
            print_info(
                "   Paste it into https://api.slack.com/apps → your app → Features "
                "→ App Manifest → Edit, then Save.  Slack will prompt to "
                "reinstall if scopes or slash commands changed."
            )
            print_info(
                "   Re-run `hermes slack manifest --write` anytime to refresh after "
                "Hermes adds new commands."
            )
        except Exception as e:
            print_warning(f"Could not write Slack manifest: {e}")

    print_header("Slack")
    existing = get_env_value("SLACK_BOT_TOKEN")
    if existing:
        print_info("Slack: already configured")
        if not prompt_yes_no("Reconfigure Slack?", False):
            # Even without reconfiguring, offer to refresh the manifest so
            # new commands (e.g. /btw, /stop, ...) get registered in Slack.
            if prompt_yes_no(
                "Regenerate the Slack app manifest with the latest command "
                "list? (recommended after `hermes update`)",
                True,
            ):
                _write_slack_manifest_and_instruct()
            return

    print_info("Steps to create a Slack app:")
    print_info("   1. Go to https://api.slack.com/apps → Create New App")
    print_info("      Pick 'From an app manifest' — we'll generate one for you below.")
    print_info("   2. Enable Socket Mode: Settings → Socket Mode → Enable")
    print_info("      • Create an App-Level Token with 'connections:write' scope")
    print_info("   3. Install to Workspace: Settings → Install App")
    print_info("   4. After installing, invite the bot to channels: /invite @YourBot")
    print()
    print_info("   Full guide: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/slack/")
    print()

    # Generate and write manifest up-front so the user can paste it into
    # the "Create from manifest" flow instead of clicking through scopes /
    # events / slash commands one at a time.
    _write_slack_manifest_and_instruct()

    print()
    bot_token = prompt("Slack Bot Token (xoxb-...)", password=True)
    if not bot_token:
        return
    save_env_value("SLACK_BOT_TOKEN", bot_token)
    app_token = prompt("Slack App Token (xapp-...)", password=True)
    if app_token:
        save_env_value("SLACK_APP_TOKEN", app_token)
    print_success("Slack tokens saved")

    print()
    print_info("🔒 Security: Restrict who can use your bot")
    print_info("   To find a Member ID: click a user's name → View full profile → ⋮ → Copy member ID")
    print()
    allowed_users = prompt(
        "Allowed user IDs (comma-separated, leave empty to deny everyone except paired users)"
    )
    if allowed_users:
        save_env_value("SLACK_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("Slack allowlist configured")
    else:
        print_warning("⚠️  No Slack allowlist set - unpaired users will be denied by default.")
        print_info("   Set SLACK_ALLOW_ALL_USERS=true or GATEWAY_ALLOW_ALL_USERS=true only if you intentionally want open workspace access.")

    print()
    print_info("📬 Home Channel: where Hermes delivers cron job results,")
    print_info("   cross-platform messages, and notifications.")
    print_info("   To get a channel ID: open the channel in Slack, then right-click")
    print_info("   the channel name → Copy link — the ID starts with C (e.g. C01ABC2DE3F).")
    print_info("   You can also set this later by typing /set-home in a Slack channel.")
    home_channel = prompt("Home channel ID (leave empty to set later with /set-home)")
    if home_channel:
        save_env_value("SLACK_HOME_CHANNEL", home_channel.strip())


def _apply_yaml_config(yaml_cfg: dict, slack_cfg: dict) -> dict | None:
    """Translate ``config.yaml`` ``slack:`` keys into ``SLACK_*`` env vars.

    Implements the ``apply_yaml_config_fn`` contract (#24849). Mirrors the
    legacy ``slack_cfg`` block that used to live in
    ``gateway/config.py::load_gateway_config()`` before this migration.

    The SlackAdapter reads its runtime configuration via ``os.getenv()``
    throughout the connect / handle code paths, so rather than rewrite those
    call sites to read from ``PlatformConfig.extra``, this hook keeps the
    existing env-driven model and owns the YAML→env translation here, next to
    the adapter that consumes it. Env vars take precedence over YAML — every
    assignment is guarded by ``not os.getenv(...)`` so explicit env vars
    survive a config.yaml update. Returns ``None`` because no extras are
    seeded into ``PlatformConfig.extra`` directly (everything flows through env).
    """
    if "require_mention" in slack_cfg and not os.getenv("SLACK_REQUIRE_MENTION"):
        os.environ["SLACK_REQUIRE_MENTION"] = str(slack_cfg["require_mention"]).lower()
    if "strict_mention" in slack_cfg and not os.getenv("SLACK_STRICT_MENTION"):
        os.environ["SLACK_STRICT_MENTION"] = str(slack_cfg["strict_mention"]).lower()
    if "allow_bots" in slack_cfg and not os.getenv("SLACK_ALLOW_BOTS"):
        os.environ["SLACK_ALLOW_BOTS"] = str(slack_cfg["allow_bots"]).lower()
    frc = slack_cfg.get("free_response_channels")
    if frc is not None and not os.getenv("SLACK_FREE_RESPONSE_CHANNELS"):
        if isinstance(frc, list):
            frc = ",".join(str(v) for v in frc)
        os.environ["SLACK_FREE_RESPONSE_CHANNELS"] = str(frc)
    if "reactions" in slack_cfg and not os.getenv("SLACK_REACTIONS"):
        os.environ["SLACK_REACTIONS"] = str(slack_cfg["reactions"]).lower()
    ac = slack_cfg.get("allowed_channels")
    if ac is not None and not os.getenv("SLACK_ALLOWED_CHANNELS"):
        if isinstance(ac, list):
            ac = ",".join(str(v) for v in ac)
        os.environ["SLACK_ALLOWED_CHANNELS"] = str(ac)
    return None  # all settings flow through env; nothing to merge into extras


def _is_connected(config) -> bool:
    """Slack is considered connected when SLACK_BOT_TOKEN is set.

    Looks up via ``hermes_cli.gateway.get_env_value`` at call time (not via the
    plugin's own bound import) so tests that patch ``gateway_mod.get_env_value``
    can suppress ambient ``SLACK_BOT_TOKEN`` env vars. Matches what the legacy
    ``Platform.SLACK`` connected-check did before this migration.
    """
    import hermes_cli.gateway as gateway_mod

    return bool((gateway_mod.get_env_value("SLACK_BOT_TOKEN") or "").strip())


def _build_adapter(config):
    """Factory wrapper that constructs SlackAdapter from a PlatformConfig."""
    return SlackAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="slack",
        label="Slack",
        adapter_factory=_build_adapter,
        check_fn=check_slack_requirements,
        is_connected=_is_connected,
        required_env=["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"],
        install_hint="pip install 'hermes-agent[slack]'",
        # Interactive setup wizard — replaces hermes_cli/setup.py::_setup_slack
        # and the static _PLATFORMS["slack"] dict in hermes_cli/gateway.py.
        setup_fn=interactive_setup,
        # YAML→env config bridge — owns the translation of config.yaml slack:
        # keys (require_mention, strict_mention, allow_bots,
        # free_response_channels, reactions, allowed_channels) into SLACK_*
        # env vars that the adapter reads via os.getenv(). Replaces the
        # hardcoded block in gateway/config.py. Hook contract: #24849.
        apply_yaml_config_fn=_apply_yaml_config,
        # Auth env vars for _is_user_authorized() integration
        allowed_users_env="SLACK_ALLOWED_USERS",
        allow_all_env="SLACK_ALLOW_ALL_USERS",
        # Cron home-channel delivery
        cron_deliver_env_var="SLACK_HOME_CHANNEL",
        # Out-of-process cron delivery via the Slack Web API. Without this hook,
        # deliver=slack cron jobs fail with "No live adapter" when cron runs
        # separately from the gateway. Replaces the _send_slack helper.
        standalone_sender_fn=_standalone_send,
        # Slack API allows 40,000 chars; leave margin (matches the legacy
        # SlackAdapter.MAX_MESSAGE_LENGTH).
        max_message_length=39000,
        # Display
        emoji="💼",
        allow_update_command=True,
    )
