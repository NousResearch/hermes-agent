"""
Slack platform adapter.

Uses slack-bolt (Python) with Socket Mode for:
- Receiving messages from channels and DMs
- Sending responses back
- Handling slash commands
- Thread support
"""

import asyncio
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
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    AsyncApp = Any
    AsyncSocketModeHandler = Any
    AsyncWebClient = Any

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    SUPPORTED_DOCUMENT_TYPES,
    is_host_excluded_by_no_proxy,
    resolve_proxy_url,
    safe_url_for_log,
    cache_document_from_bytes,
)


logger = logging.getLogger(__name__)


@dataclass
class _ThreadContextCache:
    """Cache entry for fetched thread context."""
    content: str
    fetched_at: float = field(default_factory=time.monotonic)
    message_count: int = 0
    parent_text: str = ""  # Raw text of the thread parent (for reply_to_text injection)


def check_slack_requirements() -> bool:
    """Check if Slack dependencies are available."""
    return SLACK_AVAILABLE


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
                    _append_line(f"```{lang}\n{code_text}\n```", quote_depth=quote_depth, bullet=bullet)
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
            return [item for item in (_sanitize(v) for v in value) if item not in (None, {}, [], "")]
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

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SLACK)
        self._app: Optional[Any] = None
        self._handler: Optional[Any] = None
        self._bot_user_id: Optional[str] = None
        self._user_name_cache: Dict[str, str] = {}  # user_id → display name
        self._socket_mode_task: Optional[asyncio.Task] = None
        # Multi-workspace support
        self._team_clients: Dict[str, Any] = {}   # team_id → WebClient
        self._team_bot_user_ids: Dict[str, str] = {}          # team_id → bot_user_id
        self._channel_team: Dict[str, str] = {}                # channel_id → team_id
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

    def _describe_slack_api_error(self, response: Any, *, file_obj: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Convert Slack API auth/permission failures into actionable user-facing text."""
        if response is None or not hasattr(response, "get"):
            return None

        error = str(response.get("error", "") or "").strip()
        if not error:
            return None

        file_label = str((file_obj or {}).get("name") or (file_obj or {}).get("id") or "this attachment")
        needed = str(response.get("needed", "") or "").strip()
        provided = str(response.get("provided", "") or "").strip()
        reinstall_hint = " Update the Slack app scopes/settings and reinstall the app to the workspace."
        provided_hint = f" Current bot scopes: {provided}." if provided else ""

        if error == "missing_scope":
            needed_hint = f"Missing scope: {needed}." if needed else "Missing required Slack scope."
            return f"Slack attachment access failed for {file_label}. {needed_hint}{provided_hint}{reinstall_hint}"
        if error in {"not_authed", "invalid_auth", "account_inactive", "token_revoked"}:
            return f"Slack attachment access failed for {file_label} because the bot token is not authorized ({error}). Refresh the token/reinstall the app."
        if error in {"file_not_found", "file_deleted"}:
            return f"Slack attachment {file_label} is no longer available ({error})."
        if error in {"access_denied", "file_access_denied", "no_permission", "not_allowed_token_type", "restricted_action"}:
            return f"Slack attachment access failed for {file_label} because the bot does not have permission ({error}). Check workspace permissions/scopes and reinstall if needed."
        return None

    def _describe_slack_download_failure(self, exc: Exception, *, file_obj: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Translate Slack download exceptions into user-facing attachment diagnostics."""
        file_label = str((file_obj or {}).get("name") or (file_obj or {}).get("id") or "this attachment")

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
        if "Slack returned HTML instead of media" in message or "non-image data" in message:
            return (
                f"Slack attachment access failed for {file_label}: Slack returned an HTML/login or non-media response. "
                "This usually means a scope, auth, or file-permission problem."
            )
        return None

    async def connect(self) -> bool:
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
            logger.info("[Slack] Using proxy for Slack transport: %s", safe_url_for_log(proxy_url))

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
                        team_label = entry.get("team_name", team_id) if isinstance(entry, dict) else team_id
                        logger.info("[Slack] Loaded saved token for workspace %s", team_label)
            except Exception as e:
                logger.warning("[Slack] Failed to read %s: %s", tokens_file, e)

        lock_acquired = False
        try:
            if not self._acquire_platform_lock('slack-app-token', app_token, 'Slack app token'):
                return False
            lock_acquired = True

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

                # First token sets the primary bot_user_id (backward compat)
                if self._bot_user_id is None:
                    self._bot_user_id = bot_user_id

                logger.info(
                    "[Slack] Authenticated as @%s in workspace %s (team: %s)",
                    bot_name, team_name, team_id,
                )

            # Register message event handler
            @self._app.event("message")
            async def handle_message_event(event, say):
                await self._handle_slack_message(event)

            # Acknowledge app_mention events to prevent Bolt 404 errors.
            # The "message" handler above already processes @mentions in
            # channels, so this is intentionally a no-op to avoid duplicates.
            @self._app.event("app_mention")
            async def handle_app_mention(event, say):
                pass

            # File lifecycle events can arrive around snippet uploads even when
            # the actual user message is what we care about. Ack them so Slack
            # doesn't log noisy 404 "unhandled request" warnings.
            @self._app.event("file_shared")
            async def handle_file_shared(event, say):
                pass

            @self._app.event("file_created")
            async def handle_file_created(event, say):
                pass

            @self._app.event("file_change")
            async def handle_file_change(event, say):
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
                await ack()
                await self._handle_slash_command(command)

            # Register Block Kit action handlers for approval buttons
            for _action_id in (
                "hermes_approve_once",
                "hermes_approve_session",
                "hermes_approve_always",
                "hermes_deny",
            ):
                self._app.action(_action_id)(self._handle_approval_action)

            # Start Socket Mode handler in background
            self._handler = AsyncSocketModeHandler(self._app, app_token, proxy=proxy_url)
            _apply_slack_proxy(self._handler.client, proxy_url)
            self._socket_mode_task = asyncio.create_task(self._handler.start_async())

            self._running = True
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

    async def disconnect(self) -> None:
        """Disconnect from Slack."""
        if self._handler:
            try:
                await self._handler.close_async()
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning("[Slack] Error while closing Socket Mode handler: %s", e, exc_info=True)
        self._running = False

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

    async def stop_typing(self, chat_id: str) -> None:
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
        return str(raw).strip().lower() in ("1", "true", "yes", "on")

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

    def _record_uploaded_file_thread(self, chat_id: str, thread_ts: Optional[str]) -> None:
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
            str(part) for part in (
                exc,
                getattr(exc, "message", ""),
                getattr(exc, "response", None),
            ) if part
        ).lower()
        if "rate_limited" in body or "ratelimited" in body or "429" in body:
            return True
        if "connection reset" in body or "service unavailable" in body or "temporarily unavailable" in body:
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
            r'(```(?:[^\n]*\n)?[\s\S]*?```)',
            lambda m: _ph(m.group(0)),
            text,
        )

        # 2) Protect inline code (`...`)
        text = re.sub(r'(`[^`]+`)', lambda m: _ph(m.group(0)), text)

        # 3) Convert markdown links [text](url) → <url|text>
        def _convert_markdown_link(m):
            label = m.group(1)
            url = m.group(2).strip()
            if url.startswith('<') and url.endswith('>'):
                url = url[1:-1].strip()
            return _ph(f'<{url}|{label}>')

        text = re.sub(
            r'\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)',
            _convert_markdown_link,
            text,
        )

        # 4) Protect existing Slack entities/manual links so escaping and later
        #    formatting passes don't break them.
        text = re.sub(
            r'(<(?:[@#!]|(?:https?|mailto|tel):)[^>\n]+>)',
            lambda m: _ph(m.group(1)),
            text,
        )

        # 5) Protect blockquote markers before escaping
        text = re.sub(r'^(>+\s)', lambda m: _ph(m.group(0)), text, flags=re.MULTILINE)

        # 6) Escape Slack control characters in remaining plain text.
        # Unescape first so already-escaped input doesn't get double-escaped.
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # 7) Convert headers (## Title) → *Title* (bold)
        def _convert_header(m):
            inner = m.group(1).strip()
            # Strip redundant bold markers inside a header
            inner = re.sub(r'\*\*(.+?)\*\*', r'\1', inner)
            return _ph(f'*{inner}*')

        text = re.sub(
            r'^#{1,6}\s+(.+)$', _convert_header, text, flags=re.MULTILINE
        )

        # 8) Convert bold+italic: ***text*** → *_text_* (Slack bold wrapping italic)
        text = re.sub(
            r'\*\*\*(.+?)\*\*\*',
            lambda m: _ph(f'*_{m.group(1)}_*'),
            text,
        )

        # 9) Convert bold: **text** → *text* (Slack bold)
        text = re.sub(
            r'\*\*(.+?)\*\*',
            lambda m: _ph(f'*{m.group(1)}*'),
            text,
        )

        # 10) Convert italic: _text_ stays as _text_ (already Slack italic)
        #     Single *text* → _text_ (Slack italic)
        text = re.sub(
            r'(?<!\*)\*([^*\n]+)\*(?!\*)',
            lambda m: _ph(f'_{m.group(1)}_'),
            text,
        )

        # 11) Convert strikethrough: ~~text~~ → ~text~
        text = re.sub(
            r'~~(.+?)~~',
            lambda m: _ph(f'~{m.group(1)}~'),
            text,
        )

        # 12) Blockquotes: > prefix is already protected by step 5 above.

        # 13) Restore placeholders in reverse order
        for key in reversed(placeholders):
            text = text.replace(key, placeholders[key])

        return text

    # ----- Reactions -----

    async def _add_reaction(
        self, channel: str, timestamp: str, emoji: str
    ) -> bool:
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

    async def _remove_reaction(
        self, channel: str, timestamp: str, emoji: str
    ) -> bool:
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
        return os.getenv("SLACK_REACTIONS", "true").lower() not in ("false", "0", "no")

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

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
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
            return await self._upload_file(chat_id, image_path, caption, reply_to, metadata)
        except FileNotFoundError:
            return SendResult(success=False, error=f"Image file not found: {image_path}")
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
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)

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
            return await self._upload_file(chat_id, audio_path, caption, reply_to, metadata)
        except FileNotFoundError:
            return SendResult(success=False, error=f"Audio file not found: {audio_path}")
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
            return SendResult(success=False, error=f"Video file not found: {video_path}")

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

    def _assistant_thread_key(self, channel_id: str, thread_ts: str) -> Optional[Tuple[str, str]]:
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
        if subtype in ("message_changed", "message_deleted"):
            return

        original_text = event.get("text", "")
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
        is_dm = channel_type in ("im", "mpim")  # Both 1:1 and group DMs

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
            thread_ts = event.get("thread_ts") or ts  # ts fallback for channels

        # In channels, respond if:
        #   0. Channel is in free_response_channels, OR require_mention is
        #      disabled — always process regardless of mention.
        #   1. The bot is @mentioned in this message, OR
        #   2. The message is a reply in a thread the bot started/participated in, OR
        #   3. The message is in a thread where the bot was previously @mentioned, OR
        #   4. There's an existing session for this thread (survives restarts)
        bot_uid = self._team_bot_user_ids.get(team_id, self._bot_user_id)
        routing_text = original_text or ""
        is_mentioned = bot_uid and f"<@{bot_uid}>" in routing_text
        event_thread_ts = event.get("thread_ts")
        is_thread_reply = bool(event_thread_ts and event_thread_ts != ts)

        if not is_dm and bot_uid:
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
                has_session = (
                    is_thread_reply
                    and self._has_active_session_for_thread(
                        channel_id=channel_id,
                        thread_ts=event_thread_ts,
                        user_id=user_id,
                    )
                )
                if not reply_to_bot_thread and not in_mentioned_thread and not has_session:
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
                    to_remove = list(self._mentioned_threads)[:self._MENTIONED_THREADS_MAX // 2]
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
                    info_resp = await self._get_client(channel_id).files_info(file=file_id)
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
                                file_id, info_resp.get("error"),
                            )
                        continue
                except Exception as e:
                    response = getattr(e, "response", None)
                    detail = self._describe_slack_api_error(response, file_obj=f)
                    if detail:
                        attachment_notices.append(detail)
                        logger.warning("[Slack] %s", detail)
                    else:
                        logger.warning("[Slack] files.info error for %s: %s", file_id, e, exc_info=True)
                    continue

            mimetype = f.get("mimetype", "unknown")
            url = f.get("url_private_download") or f.get("url_private", "")
            if mimetype.startswith("image/") and url:
                try:
                    ext = "." + mimetype.split("/")[-1].split(";")[0]
                    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
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
                        logger.warning("[Slack] Failed to cache image from %s: %s", url, e, exc_info=True)
            elif mimetype.startswith("audio/") and url:
                try:
                    ext = "." + mimetype.split("/")[-1].split(";")[0]
                    if ext not in (".ogg", ".mp3", ".wav", ".webm", ".m4a"):
                        ext = ".ogg"
                    cached = await self._download_slack_file(url, ext, audio=True, team_id=team_id)
                    media_urls.append(cached)
                    media_types.append(mimetype)
                except Exception as e:  # pragma: no cover - defensive logging
                    detail = self._describe_slack_download_failure(e, file_obj=f)
                    if detail:
                        attachment_notices.append(detail)
                        logger.warning("[Slack] %s", detail)
                    else:
                        logger.warning("[Slack] Failed to cache audio from %s: %s", url, e, exc_info=True)
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
                        mime_to_ext = {v: k for k, v in SUPPORTED_DOCUMENT_TYPES.items()}
                        ext = mime_to_ext.get(mimetype, "")

                    if ext not in SUPPORTED_DOCUMENT_TYPES:
                        continue  # Skip unsupported file types silently

                    # Check file size (Slack limit: 20 MB for bots)
                    file_size = f.get("size", 0)
                    MAX_DOC_BYTES = 20 * 1024 * 1024
                    if not file_size or file_size > MAX_DOC_BYTES:
                        logger.warning("[Slack] Document too large or unknown size: %s", file_size)
                        continue

                    # Download and cache
                    raw_bytes = await self._download_slack_file_bytes(url, team_id=team_id)
                    cached_path = cache_document_from_bytes(
                        raw_bytes, original_filename or f"document{ext}"
                    )
                    doc_mime = SUPPORTED_DOCUMENT_TYPES[ext]
                    media_urls.append(cached_path)
                    media_types.append(doc_mime)
                    logger.debug("[Slack] Cached user document: %s", cached_path)

                    # Inject small text-ish files directly into the prompt so
                    # snippets like JSON/YAML/configs are actually visible to the agent.
                    MAX_TEXT_INJECT_BYTES = 100 * 1024
                    TEXT_INJECT_EXTENSIONS = {
                        ".md", ".txt", ".csv", ".log", ".json", ".xml",
                        ".yaml", ".yml", ".toml", ".ini", ".cfg",
                    }
                    if ext in TEXT_INJECT_EXTENSIONS and len(raw_bytes) <= MAX_TEXT_INJECT_BYTES:
                        try:
                            text_content = raw_bytes.decode("utf-8")
                            display_name = original_filename or f"document{ext}"
                            display_name = re.sub(r'[^\w.\- ]', '_', display_name)
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
                        logger.warning("[Slack] Failed to cache document from %s: %s", url, e, exc_info=True)

        if attachment_notices:
            notice_block = "[Slack attachment notice]\n" + "\n".join(f"- {n}" for n in attachment_notices)
            text = f"{notice_block}\n\n{text}" if text else notice_block

        if msg_type != MessageType.COMMAND and media_types:
            if any(m.startswith("image/") for m in media_types):
                msg_type = MessageType.PHOTO
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
        from gateway.platforms.base import resolve_channel_prompt, resolve_channel_skills
        _channel_prompt = resolve_channel_prompt(
            self.config.extra, channel_id, None,
        )
        _auto_skill = resolve_channel_skills(
            self.config.extra, channel_id, None,
        )

        # Extract reply context if this message is a thread reply.
        # Mirrors the Telegram/Discord implementations so that gateway.run
        # can inject a `[Replying to: "..."]` prefix when the parent is not
        # already in the session history. Uses the thread-context cache when
        # available to avoid redundant conversations.replies calls.
        reply_to_text = None
        if thread_ts and thread_ts != ts:
            try:
                reply_to_text = await self._fetch_thread_parent_text(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    team_id=team_id,
                ) or None
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
        self, chat_id: str, command: str, session_key: str,
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
            cmd_preview = command[:2900] + "..." if len(command) > 2900 else command
            thread_ts = self._resolve_thread_ts(None, metadata)

            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f":warning: *Command Approval Required*\n"
                            f"```{cmd_preview}```\n"
                            f"Reason: {description}"
                        ),
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

        # Only authorized users may click approval buttons.  Button clicks
        # bypass the normal message auth flow in gateway/run.py, so we must
        # check here as well.
        allowed_csv = os.getenv("SLACK_ALLOWED_USERS", "").strip()
        if allowed_csv:
            allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
            if "*" not in allowed_ids and user_id not in allowed_ids:
                logger.warning(
                    "[Slack] Unauthorized approval click by %s (%s) — ignoring",
                    user_name, user_id,
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
                count, session_key, choice, user_name,
            )
        except Exception as exc:
            logger.error("Failed to resolve gateway approval from Slack button: %s", exc)

        # (approval state already consumed by atomic pop above)

    # ----- Thread context fetching -----

    async def _fetch_thread_context(
        self, channel_id: str, thread_ts: str, current_ts: str,
        team_id: str = "", limit: int = 30,
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
                        retry_after = 1.0 * (2 ** attempt)  # 1s, 2s
                        logger.warning(
                            "[Slack] conversations.replies rate limited; retrying in %.1fs (attempt %d/3)",
                            retry_after, attempt + 1,
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
                    self._team_bot_user_ids.get(msg_team)
                    if msg_team
                    else None
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
        self, channel_id: str, thread_ts: str, team_id: str = "",
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

        if slash_name in ("hermes", ""):
            # Legacy /hermes <subcommand> [args] routing + free-form questions.
            # Empty slash_name falls into this branch for backward compat
            # with any caller that didn't populate command["command"].
            from hermes_cli.commands import slack_subcommand_map
            subcommand_map = slack_subcommand_map()
            subcommand_map["compact"] = "/compress"
            first_word = text.split()[0] if text else ""
            if first_word in subcommand_map:
                rest = text[len(first_word):].strip()
                text = f"{subcommand_map[first_word]} {rest}".strip() if rest else subcommand_map[first_word]
            elif text:
                pass  # Treat as a regular question
            else:
                text = "/help"
        else:
            # Native slash — /<slash_name> [args].  Route directly through the
            # gateway command dispatcher by prepending the slash.
            text = f"/{slash_name} {text}".strip()

        source = self.build_source(
            chat_id=channel_id,
            chat_type="dm",  # Slash commands are always in DM-like context
            user_id=user_id,
        )

        event = MessageEvent(
            text=text,
            message_type=MessageType.COMMAND if text.startswith("/") else MessageType.TEXT,
            source=source,
            raw_message=command,
        )

        await self.handle_message(event)

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
            gspu = getattr(store_cfg, "group_sessions_per_user", True) if store_cfg else True
            tspu = getattr(store_cfg, "thread_sessions_per_user", False) if store_cfg else False

            session_key = build_session_key(
                source,
                group_sessions_per_user=gspu,
                thread_sessions_per_user=tspu,
            )

            session_store._ensure_loaded()
            return session_key in session_store._entries
        except Exception:
            return False

    async def _download_slack_file(self, url: str, ext: str, audio: bool = False, team_id: str = "") -> str:
        """Download a Slack file using the bot token for auth, with retry."""
        import httpx

        bot_token = self._team_clients[team_id].token if team_id and team_id in self._team_clients else self.config.token

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
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 429:
                        raise
                    if attempt < 2:
                        logger.debug("Slack file download retry %d/2 for %s: %s",
                                     attempt + 1, url[:80], exc)
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise

    async def _download_slack_file_bytes(self, url: str, team_id: str = "") -> bytes:
        """Download a Slack file and return raw bytes, with retry."""
        import httpx

        bot_token = self._team_clients[team_id].token if team_id and team_id in self._team_clients else self.config.token

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
                except (httpx.TimeoutException, httpx.HTTPStatusError, ValueError) as exc:
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 429:
                        raise
                    if isinstance(exc, ValueError):
                        raise
                    if attempt < 2:
                        logger.debug("Slack file download retry %d/2 for %s: %s",
                                     attempt + 1, url[:80], exc)
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
                return configured.lower() not in ("false", "0", "no", "off")
            return bool(configured)
        return os.getenv("SLACK_REQUIRE_MENTION", "true").lower() not in ("false", "0", "no", "off")

    def _slack_strict_mention(self) -> bool:
        """When true, channel threads require an explicit @-mention on every
        message. Disables all auto-triggers (mentioned-thread memory,
        bot-message follow-up, session-presence). Defaults to False.
        """
        configured = self.config.extra.get("strict_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in ("true", "1", "yes", "on")
            return bool(configured)
        return os.getenv("SLACK_STRICT_MENTION", "false").lower() in ("true", "1", "yes", "on")

    def _slack_free_response_channels(self) -> set:
        """Return channel IDs where no @mention is required."""
        raw = self.config.extra.get("free_response_channels")
        if raw is None:
            raw = os.getenv("SLACK_FREE_RESPONSE_CHANNELS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        if isinstance(raw, str) and raw.strip():
            return {part.strip() for part in raw.split(",") if part.strip()}
        return set()
