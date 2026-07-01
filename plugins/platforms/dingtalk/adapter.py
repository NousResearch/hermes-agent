"""
DingTalk platform adapter using Stream Mode.

Uses dingtalk-stream SDK (>=0.20) for real-time message reception without webhooks.
Responses are sent via DingTalk's session webhook (markdown format).
Supports: text, images, audio, video, rich text, files, and group @mentions.

Requires:
    pip install "dingtalk-stream>=0.20" httpx
    DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET env vars

Configuration in config.yaml:
    platforms:
      dingtalk:
        enabled: true
        # Optional group-chat gating (mirrors Slack/Telegram/Discord):
        require_mention: true            # or DINGTALK_REQUIRE_MENTION env var
        # free_response_chats:           # conversations that skip require_mention
        #   - cidABC==
        # mention_patterns:              # regex wake-words (e.g. Chinese bot names)
        #   - "^小马"
        # allowed_users:                 # staff_id or sender_id list; "*" = any
        #   - "manager1234"
        extra:
          client_id: "your-app-key"      # or DINGTALK_CLIENT_ID env var
          client_secret: "your-secret"   # or DINGTALK_CLIENT_SECRET env var
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import re
import struct
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Set, cast

try:
    import dingtalk_stream
    from dingtalk_stream import ChatbotMessage
    from dingtalk_stream.frames import CallbackMessage, AckMessage

    DINGTALK_STREAM_AVAILABLE = True
except Exception:  # noqa: BLE001 — broad: optional SDK's transitive deps (cryptography) may raise non-ImportError; degrade gracefully (#41112)
    DINGTALK_STREAM_AVAILABLE = False
    dingtalk_stream = None  # type: ignore[assignment]
    ChatbotMessage = None  # type: ignore[assignment]
    CallbackMessage = None  # type: ignore[assignment]
    AckMessage = type(
        "AckMessage",
        (),
        {
            "STATUS_OK": 200,
            "STATUS_SYSTEM_EXCEPTION": 500,
        },
    )  # type: ignore[assignment]

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

# Card SDK for AI Cards (following QwenPaw pattern).
# Catch broad Exception, not just ImportError: the alibabacloud_dingtalk SDK
# transitively imports cryptography and can raise AttributeError (not
# ImportError) when the installed cryptography version skews from what the SDK
# expects (e.g. `cryptography.utils.DeprecatedIn46` missing on older
# cryptography). An optional SDK with a broken dependency chain must degrade
# gracefully — same as a missing one — rather than crash the whole adapter
# (and therefore the whole plugin) import. #41112.
try:
    from alibabacloud_dingtalk.card_1_0 import (
        client as dingtalk_card_client,
        models as dingtalk_card_models,
    )
    from alibabacloud_dingtalk.robot_1_0 import (
        client as dingtalk_robot_client,
        models as dingtalk_robot_models,
    )
    dingtalk_conv_file_client = import_module(
        "alibabacloud_dingtalk.conv_file_1_0.client"
    )
    dingtalk_conv_file_models = import_module(
        "alibabacloud_dingtalk.conv_file_1_0.models"
    )
    dingtalk_storage_client = import_module("alibabacloud_dingtalk.storage_1_0.client")
    dingtalk_storage_models = import_module("alibabacloud_dingtalk.storage_1_0.models")
    from alibabacloud_tea_openapi import models as open_api_models
    from alibabacloud_tea_util import models as tea_util_models

    CARD_SDK_AVAILABLE = True
except Exception:
    CARD_SDK_AVAILABLE = False
    dingtalk_card_client = None
    dingtalk_card_models = None
    dingtalk_robot_client = None
    dingtalk_robot_models = None
    dingtalk_conv_file_client = None
    dingtalk_conv_file_models = None
    dingtalk_storage_client = None
    dingtalk_storage_models = None
    open_api_models = None
    tea_util_models = None

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SUPPORTED_DOCUMENT_TYPES,
    cache_document_from_bytes,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 20000
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]
_SESSION_WEBHOOKS_MAX = 500
_DINGTALK_WEBHOOK_RE = re.compile(r'^https://(?:api|oapi)\.dingtalk\.com/')


def _sdk_model(models: Any, class_name: str, **kwargs: Any) -> Any:
    cls = getattr(models, class_name, None) if models is not None else None
    if cls is None:
        return SimpleNamespace(**kwargs)
    return cls(**kwargs)


def _sdk_runtime_options() -> Any:
    cls = getattr(tea_util_models, "RuntimeOptions", None) if tea_util_models is not None else None
    return cls() if cls is not None else SimpleNamespace()


# DingTalk message type → runtime content type
DINGTALK_TYPE_MAPPING = {
    "picture": "image",
    "voice": "audio",
}


def check_dingtalk_requirements() -> bool:
    """Check if DingTalk dependencies are available and configured.

    Lazy-installs dingtalk-stream via ``tools.lazy_deps.ensure("platform.dingtalk")``
    on first call if not present.
    """
    global DINGTALK_STREAM_AVAILABLE, dingtalk_stream, ChatbotMessage, CallbackMessage, AckMessage
    global HTTPX_AVAILABLE, httpx
    if not DINGTALK_STREAM_AVAILABLE or not HTTPX_AVAILABLE:
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("platform.dingtalk", prompt=False)
        except Exception:
            return False
        try:
            import dingtalk_stream as _ds
            from dingtalk_stream import ChatbotMessage as _CM
            from dingtalk_stream.frames import CallbackMessage as _CBM, AckMessage as _AM
            import httpx as _httpx
        except Exception:
            return False
        dingtalk_stream = _ds
        ChatbotMessage = _CM
        CallbackMessage = _CBM
        AckMessage = _AM
        httpx = _httpx
        DINGTALK_STREAM_AVAILABLE = True
        HTTPX_AVAILABLE = True
    if not os.getenv("DINGTALK_CLIENT_ID") or not os.getenv("DINGTALK_CLIENT_SECRET"):
        return False
    return True


class DingTalkAdapter(BasePlatformAdapter):
    """DingTalk chatbot adapter using Stream Mode.

    The dingtalk-stream SDK maintains a long-lived WebSocket connection.
    Incoming messages arrive via a ChatbotHandler callback. Replies are
    sent via the incoming message's session_webhook URL using httpx.

    Features:
    - Text messages (plain + rich text)
    - Images, audio, video, files (via download codes)
    - Group chat @mention detection
    - Session webhook caching with expiry tracking
    - Markdown formatted replies
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    @property
    def SUPPORTS_MESSAGE_EDITING(self) -> bool:  # noqa: N802
        """Edits only meaningful when AI Cards are configured.

        The gateway gates streaming cursor + edit behaviour on this flag,
        so we must reflect the actual adapter capability at runtime.
        """
        return bool(self._card_template_id and self._card_sdk)

    @property
    def REQUIRES_EDIT_FINALIZE(self) -> bool:  # noqa: N802
        """AI Card lifecycle requires an explicit ``finalize=True`` edit
        to close the streaming indicator, even when the final content is
        identical to the last streamed update.  Enabled only when cards
        are configured — webhook-only DingTalk doesn't need it.
        """
        return bool(self._card_template_id and self._card_sdk)

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.DINGTALK)

        extra = config.extra or {}
        self._client_id: str = extra.get("client_id") or os.getenv(
            "DINGTALK_CLIENT_ID", ""
        )
        self._client_secret: str = extra.get("client_secret") or os.getenv(
            "DINGTALK_CLIENT_SECRET", ""
        )

        # Group-chat gating (mirrors Slack/Telegram/Discord/WhatsApp conventions).
        # Mention state is the structured ``is_in_at_list`` attribute from the
        # dingtalk-stream SDK (set from the callback's ``isInAtList`` flag),
        # not text parsing.
        self._mention_patterns: List[re.Pattern] = self._compile_mention_patterns()
        self._allowed_users: Set[str] = self._load_allowed_users()

        self._stream_client: Any = None
        self._stream_task: Optional[asyncio.Task] = None
        self._http_client: Optional["httpx.AsyncClient"] = None
        self._card_sdk: Optional[Any] = None
        self._robot_sdk: Optional[Any] = None
        self._conv_file_sdk: Optional[Any] = None
        self._storage_sdk: Optional[Any] = None
        self._robot_code: str = extra.get("robot_code") or self._client_id
        self._file_operator_union_id: str = (
            extra.get("file_operator_union_id")
            or extra.get("union_id")
            or os.getenv("DINGTALK_FILE_OPERATOR_UNION_ID", "")
            or os.getenv("DINGTALK_UNION_ID", "")
        )
        self._file_parent_id: str = (
            extra.get("file_parent_id")
            or os.getenv("DINGTALK_FILE_PARENT_ID", "")
            or "0"
        )

        # Message deduplication
        self._dedup = MessageDeduplicator(max_size=1000)
        # Map chat_id -> (session_webhook, expired_time_ms) for reply routing
        self._session_webhooks: Dict[str, tuple[str, int]] = {}
        # Map chat_id -> last inbound ChatbotMessage. Keyed by chat_id instead
        # of a single class attribute to avoid cross-message clobbering when
        # multiple conversations run concurrently.
        self._message_contexts: Dict[str, Any] = {}
        self._card_template_id: Optional[str] = extra.get("card_template_id")

        # Chats for which we've already fired the Done reaction — prevents
        # double-firing across segment boundaries or parallel flows
        # (tool-progress + stream-consumer both finalizing their cards).
        # Reset each inbound message.
        self._done_emoji_fired: Set[str] = set()
        # Cards in streaming state per chat: chat_id -> { out_track_id -> last_content }.
        # Every `send()` creates+finalizes a card (closed state).  A subsequent
        # `edit_message(finalize=False)` re-opens the card (DingTalk's API
        # allows streaming_update on a finalized card — it flips back to
        # streaming).  We track those reopened cards so the next `send()` can
        # auto-close them as siblings — otherwise tool-progress cards get
        # stuck in streaming state forever.
        self._streaming_cards: Dict[str, Dict[str, str]] = {}
        # Track fire-and-forget emoji/reaction coroutines so Python's GC
        # doesn't drop them mid-flight, and we can cancel them on disconnect.
        self._bg_tasks: Set[asyncio.Task] = set()

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to DingTalk via Stream Mode."""
        if not DINGTALK_STREAM_AVAILABLE:
            logger.warning(
                "[%s] dingtalk-stream not installed. Run: pip install 'dingtalk-stream>=0.20'",
                self.name,
            )
            return False
        if not HTTPX_AVAILABLE:
            logger.warning(
                "[%s] httpx not installed. Run: pip install httpx", self.name
            )
            return False
        if not self._client_id or not self._client_secret:
            logger.warning(
                "[%s] DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET required", self.name
            )
            return False

        try:
            # Tighter keepalive so idle CLOSE_WAIT drains promptly (#18451).
            from gateway.platforms._http_client_limits import platform_httpx_limits
            self._http_client = httpx.AsyncClient(
                timeout=30.0, limits=platform_httpx_limits(),
            )

            credential = dingtalk_stream.Credential(
                self._client_id, self._client_secret
            )
            self._stream_client = dingtalk_stream.DingTalkStreamClient(credential)

            # Initialize card SDK if available and configured
            if CARD_SDK_AVAILABLE and self._card_template_id:
                card_client = cast(Any, dingtalk_card_client)
                robot_client = cast(Any, dingtalk_robot_client)
                conv_file_client = cast(Any, dingtalk_conv_file_client)
                storage_client = cast(Any, dingtalk_storage_client)
                sdk_config = cast(Any, open_api_models).Config()
                sdk_config.protocol = "https"
                sdk_config.region_id = "central"
                self._card_sdk = card_client.Client(sdk_config)
                self._robot_sdk = robot_client.Client(sdk_config)
                self._conv_file_sdk = conv_file_client.Client(sdk_config)
                self._storage_sdk = storage_client.Client(sdk_config)
                logger.info(
                    "[%s] Card SDK initialized with template: %s",
                    self.name,
                    self._card_template_id,
                )
            elif CARD_SDK_AVAILABLE:
                # Initialize robot SDK even without card template (for media download)
                robot_client = cast(Any, dingtalk_robot_client)
                conv_file_client = cast(Any, dingtalk_conv_file_client)
                storage_client = cast(Any, dingtalk_storage_client)
                sdk_config = cast(Any, open_api_models).Config()
                sdk_config.protocol = "https"
                sdk_config.region_id = "central"
                self._robot_sdk = robot_client.Client(sdk_config)
                self._conv_file_sdk = conv_file_client.Client(sdk_config)
                self._storage_sdk = storage_client.Client(sdk_config)
                logger.info("[%s] Robot SDK initialized (media download/send)", self.name)

            # Capture the current event loop for cross-thread dispatch
            loop = asyncio.get_running_loop()
            handler = _IncomingHandler(self, loop)
            self._stream_client.register_callback_handler(
                dingtalk_stream.ChatbotMessage.TOPIC, handler
            )

            self._stream_task = asyncio.create_task(self._run_stream())
            self._mark_connected()
            logger.info("[%s] Connected via Stream Mode", self.name)
            return True
        except Exception as e:
            logger.error("[%s] Failed to connect: %s", self.name, e)
            return False

    async def _run_stream(self) -> None:
        """Run the async stream client with auto-reconnection."""
        backoff_idx = 0
        while self._running:
            try:
                logger.debug("[%s] Starting stream client...", self.name)
                await self._stream_client.start()
            except asyncio.CancelledError:
                return
            except Exception as e:
                if not self._running:
                    return
                logger.warning("[%s] Stream client error: %s", self.name, e)

            if not self._running:
                return

            delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
            logger.info("[%s] Reconnecting in %ds...", self.name, delay)
            await asyncio.sleep(delay)
            backoff_idx += 1

    async def disconnect(self) -> None:
        """Disconnect from DingTalk."""
        self._running = False
        self._mark_disconnected()

        # Close the active websocket first so the stream task sees the
        # disconnection and exits cleanly, rather than getting stuck
        # awaiting frames that will never arrive.
        websocket = getattr(self._stream_client, "websocket", None) if self._stream_client else None
        if websocket is not None:
            try:
                await websocket.close()
            except Exception as e:
                logger.debug("[%s] websocket close during disconnect failed: %s", self.name, e)

        if self._stream_task:
            # Try graceful close first if SDK supports it. The SDK's close()
            # is sync and may block on network I/O, so offload to a thread.
            if hasattr(self._stream_client, "close"):
                try:
                    await asyncio.to_thread(self._stream_client.close)
                except Exception:
                    pass

            self._stream_task.cancel()
            try:
                await asyncio.wait_for(self._stream_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.debug("[%s] stream task did not exit cleanly during disconnect", self.name)
            self._stream_task = None

        # Cancel any in-flight background tasks (emoji reactions, etc.)
        if self._bg_tasks:
            for task in list(self._bg_tasks):
                task.cancel()
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
            self._bg_tasks.clear()

        # Finalize any open streaming cards before the HTTP client closes so
        # they don't stay stuck in streaming state on DingTalk's UI after
        # a gateway restart.  _close_streaming_siblings handles its own
        # per-card exceptions; the outer try is a safety net for token fetch.
        for _chat_id in list(self._streaming_cards):
            try:
                await self._close_streaming_siblings(_chat_id)
            except Exception as _exc:
                logger.debug(
                    "[%s] Failed to finalize streaming card on disconnect for %s: %s",
                    self.name, _chat_id, _exc,
                )

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._stream_client = None
        self._session_webhooks.clear()
        self._message_contexts.clear()
        self._streaming_cards.clear()
        self._done_emoji_fired.clear()
        self._dedup.clear()
        logger.info("[%s] Disconnected", self.name)

    # -- Group gating --------------------------------------------------------

    def _dingtalk_require_mention(self) -> bool:
        """Return whether group chats should require an explicit bot trigger."""
        configured = self.config.extra.get("require_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return os.getenv("DINGTALK_REQUIRE_MENTION", "false").lower() in {"true", "1", "yes", "on"}

    def _dingtalk_free_response_chats(self) -> Set[str]:
        raw = self.config.extra.get("free_response_chats")
        if raw is None:
            raw = os.getenv("DINGTALK_FREE_RESPONSE_CHATS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _dingtalk_allowed_chats(self) -> Set[str]:
        """Return the whitelist of group chat IDs the bot will respond in.

        When non-empty, group messages from chats NOT in this set are silently
        ignored — even if the bot is @mentioned.  DMs are never filtered.
        Empty set means no restriction (fully backward compatible).
        """
        raw = self.config.extra.get("allowed_chats") if self.config.extra else None
        if raw is None:
            raw = os.getenv("DINGTALK_ALLOWED_CHATS", "")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _compile_mention_patterns(self) -> List[re.Pattern]:
        """Compile optional regex wake-word patterns for group triggers."""
        patterns = self.config.extra.get("mention_patterns") if self.config.extra else None
        if patterns is None:
            raw = os.getenv("DINGTALK_MENTION_PATTERNS", "").strip()
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
                "[%s] dingtalk mention_patterns must be a list or string; got %s",
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
                logger.warning("[%s] Invalid DingTalk mention pattern %r: %s", self.name, pattern, exc)
        if compiled:
            logger.info("[%s] Loaded %d DingTalk mention pattern(s)", self.name, len(compiled))
        return compiled

    def _load_allowed_users(self) -> Set[str]:
        """Load allowed-users list from config.extra or env var.

        IDs are matched case-insensitively against the sender's ``staff_id`` and
        ``sender_id``. A wildcard ``*`` disables the check.
        """
        raw = self.config.extra.get("allowed_users") if self.config.extra else None
        if raw is None:
            raw = os.getenv("DINGTALK_ALLOWED_USERS", "")
        if isinstance(raw, list):
            items = [str(part).strip() for part in raw if str(part).strip()]
        else:
            items = [part.strip() for part in str(raw).split(",") if part.strip()]
        return {item.lower() for item in items}

    def _is_user_allowed(self, sender_id: str, sender_staff_id: str) -> bool:
        if not self._allowed_users or "*" in self._allowed_users:
            return True
        candidates = {(sender_id or "").lower(), (sender_staff_id or "").lower()}
        candidates.discard("")
        return bool(candidates & self._allowed_users)

    def _message_mentions_bot(self, message: "ChatbotMessage") -> bool:
        """True if the bot was @-mentioned in a group message.

        dingtalk-stream sets ``is_in_at_list`` on the incoming ChatbotMessage
        when the bot is addressed via @-mention.
        """
        return bool(getattr(message, "is_in_at_list", False))

    def _message_matches_mention_patterns(self, text: str) -> bool:
        if not text or not self._mention_patterns:
            return False
        return any(pattern.search(text) for pattern in self._mention_patterns)

    def _should_process_message(self, message: "ChatbotMessage", text: str, is_group: bool, chat_id: str) -> bool:
        """Apply DingTalk group trigger rules.

        DMs remain unrestricted (subject to ``allowed_users`` which is enforced
        earlier). Group messages are accepted when:
        - the chat passes the ``allowed_chats`` whitelist (when set)
        - the chat is explicitly allowlisted in ``free_response_chats``
        - ``require_mention`` is disabled
        - the bot is @mentioned (``is_in_at_list``)
        - the text matches a configured regex wake-word pattern

        When ``allowed_chats`` is non-empty, it acts as a hard gate — messages
        from any group chat not in the list are ignored regardless of the
        other rules.
        """
        if not is_group:
            return True
        allowed = self._dingtalk_allowed_chats()
        if allowed and chat_id and chat_id not in allowed:
            return False
        if chat_id and chat_id in self._dingtalk_free_response_chats():
            return True
        if not self._dingtalk_require_mention():
            return True
        if self._message_mentions_bot(message):
            return True
        return self._message_matches_mention_patterns(text)

    def _spawn_bg(self, coro) -> None:
        """Start a fire-and-forget coroutine and track it for cleanup."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    # -- AI Card lifecycle helpers ------------------------------------------

    async def _close_streaming_siblings(self, chat_id: str) -> None:
        """Finalize any previously-open streaming cards for this chat.

        Called at the start of every ``send()`` so lingering tool-progress
        cards that were reopened by ``edit_message(finalize=False)`` get
        cleanly closed before the next card is created.  Without this,
        tool-progress cards stay stuck in streaming state after the agent
        moves on (there is no explicit "turn end" signal from the gateway).
        """
        cards = self._streaming_cards.pop(chat_id, None)
        if not cards:
            return
        token = await self._get_access_token()
        if not token:
            return
        for out_track_id, last_content in list(cards.items()):
            try:
                await self._stream_card_content(
                    out_track_id, token, last_content, finalize=True,
                )
                logger.debug(
                    "[%s] AI Card sibling closed: %s",
                    self.name, out_track_id,
                )
            except Exception as e:
                logger.debug(
                    "[%s] Sibling close failed for %s: %s",
                    self.name, out_track_id, e,
                )

    def _fire_done_reaction(self, chat_id: str) -> None:
        """Swap 🤔Thinking → 🥳Done on the original user message.

        Idempotent per chat_id — safe to call from segment-break flushes
        and final-done flushes without double-firing.
        """
        if chat_id in self._done_emoji_fired:
            return
        self._done_emoji_fired.add(chat_id)
        msg = self._message_contexts.get(chat_id)
        if not msg:
            return
        msg_id = getattr(msg, "message_id", "") or ""
        conversation_id = getattr(msg, "conversation_id", "") or ""
        if not (msg_id and conversation_id):
            return

        async def _swap() -> None:
            await self._send_emotion(
                msg_id, conversation_id, "🤔Thinking", recall=True,
            )
            await self._send_emotion(
                msg_id, conversation_id, "🥳Done", recall=False,
            )

        self._spawn_bg(_swap())

    # -- Inbound message processing -----------------------------------------

    async def _on_message(
        self,
        message: "ChatbotMessage",
    ) -> None:
        """Process an incoming DingTalk chatbot message."""
        msg_id = getattr(message, "message_id", None) or uuid.uuid4().hex
        if self._dedup.is_duplicate(msg_id):
            logger.debug("[%s] Duplicate message %s, skipping", self.name, msg_id)
            return

        # Chat context
        conversation_id = getattr(message, "conversation_id", "") or ""
        conversation_type = getattr(message, "conversation_type", "1")
        is_group = str(conversation_type) == "2"
        sender_id = getattr(message, "sender_id", "") or ""
        sender_nick = getattr(message, "sender_nick", "") or sender_id
        sender_staff_id = getattr(message, "sender_staff_id", "") or ""

        chat_id = conversation_id or sender_id
        chat_type = "group" if is_group else "dm"

        # Allowed-users gate (applies to both DM and group)
        if not self._is_user_allowed(sender_id, sender_staff_id):
            logger.debug(
                "[%s] Dropping message from non-allowlisted user staff_id=%s sender_id=%s",
                self.name, sender_staff_id, sender_id,
            )
            return

        # Group mention/pattern gate.  DMs pass through unconditionally.
        # We need the message text for regex wake-word matching; extract it
        # early but don't consume the rest of the pipeline until after the
        # gate decides whether to process.
        _early_text = self._extract_text(message) or ""
        if not self._should_process_message(message, _early_text, is_group, chat_id):
            logger.debug(
                "[%s] Dropping group message that failed mention gate message_id=%s chat_id=%s",
                self.name, msg_id, chat_id,
            )
            return

        # Stash the incoming message keyed by chat_id so concurrent
        # conversations don't clobber each other's context.  Also reset
        # the per-chat "Done emoji fired" marker so a new inbound message
        # gets its own Thinking→Done cycle.
        if chat_id:
            self._message_contexts[chat_id] = message
            self._done_emoji_fired.discard(chat_id)

        # Store session webhook
        session_webhook = getattr(message, "session_webhook", None) or ""
        session_webhook_expired_time = (
            getattr(message, "session_webhook_expired_time", 0) or 0
        )
        if session_webhook and chat_id and _DINGTALK_WEBHOOK_RE.match(session_webhook):
            if len(self._session_webhooks) >= _SESSION_WEBHOOKS_MAX:
                try:
                    self._session_webhooks.pop(next(iter(self._session_webhooks)))
                except StopIteration:
                    pass
            self._session_webhooks[chat_id] = (
                session_webhook,
                session_webhook_expired_time,
            )

        # Resolve media download codes to URLs so vision tools can use them
        await self._resolve_media_codes(message)

        # Extract text content
        text = self._extract_text(message)

        # Determine message type and build media list
        msg_type, media_urls, media_types = self._extract_media(message)

        if not text and not media_urls:
            logger.debug("[%s] Empty message, skipping", self.name)
            return

        source = self.build_source(
            chat_id=chat_id,
            chat_name=getattr(message, "conversation_title", None),
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_nick,
            user_id_alt=sender_staff_id if sender_staff_id else None,
        )

        # Parse timestamp
        create_at = getattr(message, "create_at", None)
        try:
            timestamp = (
                datetime.fromtimestamp(int(create_at) / 1000, tz=timezone.utc)
                if create_at
                else datetime.now(tz=timezone.utc)
            )
        except (ValueError, OSError, TypeError):
            timestamp = datetime.now(tz=timezone.utc)

        event = MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            message_id=msg_id,
            raw_message=message,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
        )

        logger.debug(
            "[%s] Message from %s in %s: %s",
            self.name,
            sender_nick,
            chat_id[:20] if chat_id else "?",
            text[:80] if text else "(media)",
        )
        await self.handle_message(event)

    @staticmethod
    def _extract_text(message: "ChatbotMessage") -> str:
        """Extract plain text from a DingTalk chatbot message.

        Handles both legacy and current dingtalk-stream SDK payload shapes:
          * legacy: ``message.text`` was a dict ``{"content": "..."}``
          * >= 0.20: ``message.text`` is a ``TextContent`` dataclass whose
            ``__str__`` returns ``"TextContent(content=...)"`` — never fall
            back to ``str(text)`` without extracting ``.content`` first.
          * rich text moved from ``message.rich_text`` (list) to
            ``message.rich_text_content.rich_text_list`` (list of dicts).
        """
        text = getattr(message, "text", None) or ""

        # Handle TextContent object (SDK style)
        if hasattr(text, "content"):
            content = (text.content or "").strip()
        elif isinstance(text, dict):
            content = text.get("content", "").strip()
        else:
            content = str(text).strip()

        if not content:
            rich_text = getattr(message, "rich_text_content", None) or getattr(
                message, "rich_text", None
            )
            if rich_text:
                rich_list = getattr(rich_text, "rich_text_list", None) or rich_text
                if isinstance(rich_list, list):
                    parts = []
                    for item in rich_list:
                        if isinstance(item, dict):
                            t = item.get("text") or item.get("content") or ""
                            if t:
                                parts.append(t)
                        elif hasattr(item, "text") and item.text:
                            parts.append(item.text)
                    content = " ".join(parts).strip()

        # Do NOT strip "@bot" from the text.  The mention is a routing
        # signal (delivered structurally via callback `isInAtList`), and
        # regex-stripping @handles would collateral-damage e-mails
        # (alice@example.com), SSH URLs (git@github.com), and literal
        # references the user wrote ("what does @openai think").  Let the
        # LLM see the raw text — it handles "@bot hello" cleanly.
        return content

    def _extract_media(self, message: "ChatbotMessage"):
        """Extract media info from message. Returns (MessageType, [urls], [mime_types])."""
        msg_type = MessageType.TEXT
        media_urls = []
        media_types = []

        # Check for image/picture
        image_content = getattr(message, "image_content", None)
        if image_content:
            download_code = getattr(image_content, "download_code", None)
            if download_code:
                media_urls.append(download_code)
                media_types.append("image")
                msg_type = MessageType.PHOTO

        # Check for raw file payload preserved from msgtype=file callbacks.
        file_content = getattr(message, "file_content", None)
        if file_content:
            dl_code = self._obj_value(file_content, "downloadCode", "download_code")
            if dl_code:
                media_urls.append(dl_code)
                media_types.append("application/octet-stream")
                msg_type = MessageType.DOCUMENT

        # Check for rich text with mixed content
        rich_text = getattr(message, "rich_text_content", None) or getattr(
            message, "rich_text", None
        )
        if rich_text:
            rich_list = getattr(rich_text, "rich_text_list", None) or rich_text
            if isinstance(rich_list, list):
                for item in rich_list:
                    if isinstance(item, dict):
                        dl_code = (
                            item.get("downloadCode") or item.get("download_code") or ""
                        )
                        item_type = item.get("type", "")
                        if dl_code:
                            mapped = DINGTALK_TYPE_MAPPING.get(item_type, "file")
                            media_urls.append(dl_code)
                            if mapped == "image":
                                media_types.append("image")
                                if msg_type == MessageType.TEXT:
                                    msg_type = MessageType.PHOTO
                            elif mapped == "audio":
                                media_types.append("audio")
                                if msg_type == MessageType.TEXT:
                                    # DingTalk's "voice" rich-text item is a
                                    # native voice note — route through STT.
                                    # "audio" comes from file uploads only;
                                    # keep those as AUDIO (no auto-STT).
                                    if item_type == "voice":
                                        msg_type = MessageType.VOICE
                                    else:
                                        msg_type = MessageType.AUDIO
                            elif mapped == "video":
                                media_types.append("video")
                                if msg_type == MessageType.TEXT:
                                    msg_type = MessageType.VIDEO
                            else:
                                media_types.append("application/octet-stream")
                                if msg_type == MessageType.TEXT:
                                    msg_type = MessageType.DOCUMENT

        msg_type_str = getattr(message, "message_type", "") or ""
        if msg_type_str == "picture" and not media_urls:
            msg_type = MessageType.PHOTO
        elif msg_type_str == "file":
            if media_urls:
                msg_type = MessageType.DOCUMENT
            else:
                msg_type = MessageType.TEXT
        elif msg_type_str == "richText":
            if any("image" in t for t in media_types):
                msg_type = MessageType.PHOTO
            elif media_urls:
                # Rich-text file messages arrive as richText with a downloadCode.
                # Preserve them as documents so the gateway document pipeline can
                # parse cached spreadsheets instead of reducing the event to text.
                msg_type = MessageType.DOCUMENT
            else:
                msg_type = MessageType.TEXT

        return msg_type, media_urls, media_types

    # -- Outbound messaging -------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a markdown reply via DingTalk session webhook."""
        metadata = metadata or {}
        logger.debug(
            "[%s] send() chat_id=%s card_enabled=%s",
            self.name,
            chat_id,
            bool(self._card_template_id and self._card_sdk),
        )

        # Check metadata first (for direct webhook sends)
        session_webhook = metadata.get("session_webhook")
        if not session_webhook:
            webhook_info = self._get_valid_webhook(chat_id)
            if not webhook_info:
                logger.warning(
                    "[%s] No valid session_webhook for chat_id=%s",
                    self.name, chat_id,
                )
                return SendResult(
                    success=False,
                    error="No valid session_webhook available. Reply must follow an incoming message.",
                )
            session_webhook, _ = webhook_info

        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")

        # Look up the inbound message for this chat (for AI Card routing)
        current_message = self._message_contexts.get(chat_id)

        # ``reply_to`` is the signal that this send is the FINAL response
        # to an inbound user message — only `base.py:_send_with_retry` sets
        # it.  Tool-progress, commentary, and stream-consumer first-sends
        # all leave it None.  We use it for two orthogonal decisions:
        #   1. finalize on create?  Yes if final reply, No if intermediate
        #      (intermediate cards stay in streaming state so edit_message
        #      updates don't flicker closed→streaming→closed repeatedly).
        #   2. fire Done reaction?  Only when this is the final reply.
        is_final_reply = reply_to is not None

        # Try AI Card first (using alibabacloud_dingtalk.card_1_0 SDK).
        if self._card_template_id and current_message and self._card_sdk:
            # Close any previously-open streaming cards for this chat
            # before creating a new one (handles tool-progress → final-
            # response handoff; also cleans up lingering commentary cards).
            await self._close_streaming_siblings(chat_id)

            result = await self._create_and_stream_card(
                chat_id, current_message, content,
                finalize=is_final_reply,
            )
            if result and result.success:
                if is_final_reply:
                    # Final reply: card closed, swap Thinking → Done.
                    self._fire_done_reaction(chat_id)
                else:
                    # Intermediate (tool progress / commentary / streaming
                    # first chunk): keep the card open and track it so the
                    # next send() auto-closes it as a sibling, or
                    # edit_message(finalize=True) closes it explicitly.
                    self._streaming_cards.setdefault(chat_id, {})[
                        result.message_id
                    ] = content
                return result

            logger.warning("[%s] AI Card send failed, falling back to webhook", self.name)

        logger.debug("[%s] Sending via webhook", self.name)
        # Normalize markdown for DingTalk
        normalized = self._normalize_markdown(content[: self.MAX_MESSAGE_LENGTH])

        payload = {
            "msgtype": "markdown",
            "markdown": {"title": "Hermes", "text": normalized},
        }

        try:
            resp = await self._http_client.post(
                session_webhook, json=payload, timeout=15.0
            )
            status_code = getattr(resp, "status_code", None)
            if isinstance(status_code, int) and status_code < 300:
                # Webhook path: fire Done only for final replies, same as
                # the card path.
                if is_final_reply:
                    self._fire_done_reaction(chat_id)
                return SendResult(success=True, message_id=uuid.uuid4().hex[:12])
            body = str(getattr(resp, "text", ""))
            logger.warning(
                "[%s] Send failed HTTP %s: %s", self.name, status_code, body[:200]
            )
            return SendResult(
                success=False, error=f"HTTP {status_code}: {body[:200]}"
            )
        except httpx.TimeoutException:
            return SendResult(
                success=False, error="Timeout sending message to DingTalk"
            )
        except Exception as e:
            logger.error("[%s] Send error: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    def _union_id_candidates_for_chat(self, chat_id: str) -> List[str]:
        candidates: List[str] = []
        configured = (self._file_operator_union_id or "").strip()
        if configured:
            candidates.append(configured)

        msg = self._message_contexts.get(chat_id)
        if msg:
            raw = getattr(msg, "raw_payload", None)
            if isinstance(raw, dict):
                for key in ("senderUnionId", "senderId", "senderStaffId", "chatbotUserId"):
                    value = raw.get(key)
                    if value:
                        candidates.append(str(value))
            for attr in ("sender_union_id", "sender_id", "sender_staff_id"):
                value = getattr(msg, attr, None)
                if value:
                    candidates.append(str(value))

        seen: Set[str] = set()
        unique: List[str] = []
        for value in candidates:
            value = value.strip()
            if value and value not in seen:
                seen.add(value)
                unique.append(value)
        return unique

    async def _get_conversation_space_id(self, chat_id: str, token: str, union_id: str) -> str:
        if not self._conv_file_sdk:
            raise RuntimeError("DingTalk conv_file SDK is unavailable")
        conv_file_models = cast(Any, dingtalk_conv_file_models)
        request = _sdk_model(
            conv_file_models,
            "GetSpaceRequest",
            open_conversation_id=chat_id,
            union_id=union_id,
        )
        headers = _sdk_model(
            conv_file_models,
            "GetSpaceHeaders",
            x_acs_dingtalk_access_token=token,
        )
        runtime = _sdk_runtime_options()
        response = await self._conv_file_sdk.get_space_with_options_async(
            request, headers, runtime
        )
        space = getattr(getattr(response, "body", None), "space", None)
        space_id = getattr(space, "space_id", None)
        if not space_id:
            raise RuntimeError("DingTalk returned no conversation spaceId")
        return space_id

    async def _upload_file_to_dingtalk_space(
        self,
        space_id: str,
        parent_id: str,
        token: str,
        union_id: str,
        file_path: str,
        file_name: str,
    ) -> str:
        if not self._storage_sdk or not self._http_client:
            raise RuntimeError("DingTalk storage SDK/http client is unavailable")
        storage_models = cast(Any, dingtalk_storage_models)

        path = Path(file_path).expanduser()
        data = path.read_bytes()
        size = len(data)
        md5 = hashlib.md5(data).hexdigest()
        precheck = _sdk_model(
            storage_models,
            "GetFileUploadInfoRequestOptionPreCheckParam",
            md_5=md5,
            name=file_name,
            parent_id=parent_id,
            size=size,
        )
        upload_request = _sdk_model(
            storage_models,
            "GetFileUploadInfoRequest",
            multipart=False,
            protocol="HEADER_SIGNATURE",
            union_id=union_id,
            option=_sdk_model(
                storage_models,
                "GetFileUploadInfoRequestOption",
                pre_check_param=precheck,
                prefer_intranet=False,
                storage_driver="DINGTALK",
            ),
        )
        upload_headers = _sdk_model(
            storage_models,
            "GetFileUploadInfoHeaders",
            x_acs_dingtalk_access_token=token,
        )
        runtime = _sdk_runtime_options()
        upload_response = await self._storage_sdk.get_file_upload_info_with_options_async(
            space_id,
            upload_request,
            upload_headers,
            runtime,
        )
        body = getattr(upload_response, "body", None)
        signature = getattr(body, "header_signature_info", None)
        upload_key = getattr(body, "upload_key", None)
        resource_urls = getattr(signature, "resource_urls", None) or []
        if not upload_key or not resource_urls:
            raise RuntimeError("DingTalk returned incomplete upload info")
        upload_url = resource_urls[0]
        signed_headers = dict(getattr(signature, "headers", None) or {})

        put_response = await self._http_client.put(
            upload_url,
            content=data,
            headers=signed_headers,
            timeout=60.0,
        )
        put_response.raise_for_status()

        commit_request = _sdk_model(
            storage_models,
            "CommitFileRequest",
            name=file_name,
            parent_id=parent_id,
            upload_key=upload_key,
            union_id=union_id,
        )
        commit_headers = _sdk_model(
            storage_models,
            "CommitFileHeaders",
            x_acs_dingtalk_access_token=token,
        )
        commit_response = await self._storage_sdk.commit_file_with_options_async(
            space_id,
            commit_request,
            commit_headers,
            runtime,
        )
        dentry = getattr(getattr(commit_response, "body", None), "dentry", None)
        dentry_id = getattr(dentry, "id", None) or getattr(dentry, "uuid", None)
        if not dentry_id:
            raise RuntimeError("DingTalk returned no committed dentryId")
        return dentry_id

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
        """Upload a local file to DingTalk conversation storage and send it."""
        path = Path(file_path).expanduser()
        if not path.is_file():
            return SendResult(success=False, error=f"File not found: {file_path}")
        if not self._conv_file_sdk or not self._storage_sdk:
            return SendResult(success=False, error="DingTalk file SDK is not initialized")
        conv_file_models = cast(Any, dingtalk_conv_file_models)

        token = await self._get_access_token()
        if not token:
            return SendResult(success=False, error="Unable to get DingTalk access token")

        name = Path(file_name or path.name).name
        errors: List[str] = []
        for union_id in self._union_id_candidates_for_chat(chat_id):
            try:
                space_id = await self._get_conversation_space_id(chat_id, token, union_id)
                dentry_id = await self._upload_file_to_dingtalk_space(
                    space_id=space_id,
                    parent_id=self._file_parent_id,
                    token=token,
                    union_id=union_id,
                    file_path=str(path),
                    file_name=name,
                )
                request = _sdk_model(
                    conv_file_models,
                    "SendRequest",
                    dentry_id=dentry_id,
                    open_conversation_id=chat_id,
                    space_id=space_id,
                    union_id=union_id,
                )
                headers = _sdk_model(
                    conv_file_models,
                    "SendHeaders",
                    x_acs_dingtalk_access_token=token,
                )
                runtime = _sdk_runtime_options()
                response = await self._conv_file_sdk.send_with_options_async(
                    request, headers, runtime
                )
                if caption:
                    await self.send(chat_id=chat_id, content=caption, reply_to=reply_to, metadata=metadata)
                logger.info("[%s] Sent DingTalk file %s to %s", self.name, name, chat_id)
                return SendResult(success=True, message_id=dentry_id, raw_response=response)
            except Exception as exc:
                errors.append(f"{union_id}: {type(exc).__name__}: {exc}")
                logger.debug("[%s] DingTalk file send failed with union_id=%s", self.name, union_id, exc_info=True)

        if not errors:
            errors.append(
                "missing unionId; configure dingtalk.extra.file_operator_union_id "
                "or DINGTALK_FILE_OPERATOR_UNION_ID"
            )
        return SendResult(success=False, error="; ".join(errors[-3:]))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """DingTalk does not support typing indicators."""
        pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image via DingTalk markdown.

        DingTalk's session webhook only supports text/markdown payloads, not
        native image/file attachments. For remote image URLs, render the image
        inline with markdown so the user still sees the image. Local files need
        OpenAPI media upload and are handled separately.
        """
        image_block = f"![image]({image_url})"
        content = f"{caption}\n\n{image_block}" if caption else image_block
        return await self.send(
            chat_id=chat_id,
            content=content,
            reply_to=reply_to,
            metadata=metadata,
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
        """DingTalk webhook replies cannot send local image files directly."""
        return SendResult(
            success=False,
            error=(
                "DingTalk session webhook replies do not support local image uploads. "
                "Only markdown/text replies are supported without OpenAPI media upload."
            ),
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about a DingTalk conversation."""
        return {
            "name": chat_id,
            "type": "group" if "group" in chat_id.lower() else "dm",
        }

    def _get_valid_webhook(self, chat_id: str) -> Optional[tuple[str, int]]:
        """Get a valid (non-expired) session webhook for the given chat_id."""
        info = self._session_webhooks.get(chat_id)
        if not info:
            return None
        webhook, expired_time_ms = info
        # Check expiry with 5-minute safety margin
        if expired_time_ms and expired_time_ms > 0:
            now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            safety_margin_ms = 5 * 60 * 1000
            if now_ms + safety_margin_ms >= expired_time_ms:
                # Expired, remove from cache
                self._session_webhooks.pop(chat_id, None)
                return None
        return info

    async def _create_and_stream_card(
        self,
        chat_id: str,
        message: Any,
        content: str,
        *,
        finalize: bool = True,
    ) -> Optional[SendResult]:
        """Create an AI Card, deliver it to the conversation, and stream initial content.

        Always called with ``finalize=True`` from ``send()`` (closed state).
        If the caller later issues ``edit_message(finalize=False)``, the
        DingTalk streaming_update API reopens the card into streaming
        state, and we track that in ``_streaming_cards`` for sibling
        cleanup on the next send.
        """
        try:
            token = await self._get_access_token()
            if not token:
                return None

            out_track_id = f"hermes_{uuid.uuid4().hex[:12]}"

            conversation_id = getattr(message, "conversation_id", "") or ""
            conversation_type = getattr(message, "conversation_type", "1")
            is_group = str(conversation_type) == "2"
            sender_staff_id = getattr(message, "sender_staff_id", "") or ""

            runtime = _sdk_runtime_options()

            # Step 1: Create card with STREAM callback type
            create_request = _sdk_model(
                dingtalk_card_models,
                "CreateCardRequest",
                card_template_id=self._card_template_id,
                out_track_id=out_track_id,
                card_data=_sdk_model(
                    dingtalk_card_models,
                    "CreateCardRequestCardData",
                    card_param_map={"content": ""},
                ),
                callback_type="STREAM",
                im_group_open_space_model=(
                    _sdk_model(
                        dingtalk_card_models,
                        "CreateCardRequestImGroupOpenSpaceModel",
                        support_forward=True,
                    )
                ),
                im_robot_open_space_model=(
                    _sdk_model(
                        dingtalk_card_models,
                        "CreateCardRequestImRobotOpenSpaceModel",
                        support_forward=True,
                    )
                ),
            )

            create_headers = _sdk_model(
                dingtalk_card_models,
                "CreateCardHeaders",
                x_acs_dingtalk_access_token=token,
            )

            await self._card_sdk.create_card_with_options_async(
                create_request, create_headers, runtime
            )

            # Step 2: Deliver card to the conversation
            if is_group:
                open_space_id = f"dtv1.card//IM_GROUP.{conversation_id}"
                deliver_request = _sdk_model(
                    dingtalk_card_models,
                    "DeliverCardRequest",
                    out_track_id=out_track_id,
                    user_id_type=1,
                    open_space_id=open_space_id,
                    im_group_open_deliver_model=(
                        _sdk_model(
                            dingtalk_card_models,
                            "DeliverCardRequestImGroupOpenDeliverModel",
                            robot_code=self._robot_code,
                        )
                    ),
                )
            else:
                if not sender_staff_id:
                    logger.warning(
                        "[%s] AI Card skipped: missing sender_staff_id for DM",
                        self.name,
                    )
                    return None
                open_space_id = f"dtv1.card//IM_ROBOT.{sender_staff_id}"
                deliver_request = _sdk_model(
                    dingtalk_card_models,
                    "DeliverCardRequest",
                    out_track_id=out_track_id,
                    user_id_type=1,
                    open_space_id=open_space_id,
                    im_robot_open_deliver_model=(
                        _sdk_model(
                            dingtalk_card_models,
                            "DeliverCardRequestImRobotOpenDeliverModel",
                            space_type="IM_ROBOT",
                        )
                    ),
                )

            deliver_headers = _sdk_model(
                dingtalk_card_models,
                "DeliverCardHeaders",
                x_acs_dingtalk_access_token=token,
            )

            await self._card_sdk.deliver_card_with_options_async(
                deliver_request, deliver_headers, runtime
            )

            # Step 3: Stream initial content.  finalize=True closes the
            # card immediately (one-shot); finalize=False keeps it open
            # for streaming edit_message updates by out_track_id.
            await self._stream_card_content(
                out_track_id, token, content, finalize=finalize,
            )

            logger.info(
                "[%s] AI Card %s: %s",
                self.name,
                "created+finalized" if finalize else "created (streaming)",
                out_track_id,
            )
            return SendResult(success=True, message_id=out_track_id)

        except Exception as e:
            logger.warning(
                "[%s] AI Card create failed: %s\n%s",
                self.name, e, traceback.format_exc(),
            )
            return None

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Edit an AI Card by streaming updated content.

        ``message_id`` is the out_track_id returned by the initial ``send()``
        call that created this card.  Callers (stream_consumer, tool
        progress) track their own ids independently so two parallel flows
        on the same chat_id don't interfere.
        """
        if not message_id:
            return SendResult(success=False, error="message_id required")
        token = await self._get_access_token()
        if not token:
            return SendResult(success=False, error="No access token")

        try:
            await self._stream_card_content(
                message_id, token, content, finalize=finalize,
            )
            if finalize:
                # Remove from streaming-cards tracking and fire Done.  This
                # is the canonical "response ended" signal from stream
                # consumer's final edit.
                self._streaming_cards.get(chat_id, {}).pop(message_id, None)
                if not self._streaming_cards.get(chat_id):
                    self._streaming_cards.pop(chat_id, None)
                logger.debug(
                    "[%s] AI Card finalized (edit): %s",
                    self.name, message_id,
                )
                self._fire_done_reaction(chat_id)
            else:
                # Non-final edit reopens the card into streaming state —
                # track it so the next send() can auto-close it as a
                # sibling.
                self._streaming_cards.setdefault(chat_id, {})[message_id] = content
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.warning("[%s] Card edit failed: %s", self.name, e)
            return SendResult(success=False, error=str(e))

    async def _stream_card_content(
        self,
        out_track_id: str,
        token: str,
        content: str,
        finalize: bool = False,
    ) -> None:
        """Stream content to an existing AI Card."""
        stream_request = _sdk_model(
            dingtalk_card_models,
            "StreamingUpdateRequest",
            out_track_id=out_track_id,
            guid=str(uuid.uuid4()),
            key="content",
            content=content[: self.MAX_MESSAGE_LENGTH],
            is_full=True,
            is_finalize=finalize,
            is_error=False,
        )

        stream_headers = _sdk_model(
            dingtalk_card_models,
            "StreamingUpdateHeaders",
            x_acs_dingtalk_access_token=token,
        )

        runtime = _sdk_runtime_options()
        await self._card_sdk.streaming_update_with_options_async(
            stream_request, stream_headers, runtime
        )

    async def _get_access_token(self) -> Optional[str]:
        """Get access token using SDK's cached token."""
        if not self._stream_client:
            return None
        try:
            # SDK's get_access_token is sync and uses requests
            token = await asyncio.to_thread(self._stream_client.get_access_token)
            return token
        except Exception as e:
            logger.error("[%s] Failed to get access token: %s", self.name, e)
            return None

    async def _send_emotion(
        self,
        open_msg_id: str,
        open_conversation_id: str,
        emoji_name: str,
        *,
        recall: bool = False,
    ) -> None:
        """Add or recall an emoji reaction on a message."""
        if not self._robot_sdk or not open_msg_id or not open_conversation_id:
            return
        action = "recall" if recall else "reply"
        try:
            token = await self._get_access_token()
            if not token:
                return

            emotion_kwargs = {
                "robot_code": self._robot_code,
                "open_msg_id": open_msg_id,
                "open_conversation_id": open_conversation_id,
                "emotion_type": 2,
                "emotion_name": emoji_name,
            }
            runtime = tea_util_models.RuntimeOptions()

            if recall:
                emotion_kwargs["text_emotion"] = (
                    dingtalk_robot_models.RobotRecallEmotionRequestTextEmotion(
                        emotion_id="2659900",
                        emotion_name=emoji_name,
                        text=emoji_name,
                        background_id="im_bg_1",
                    )
                )
                request = dingtalk_robot_models.RobotRecallEmotionRequest(
                    **emotion_kwargs,
                )
                sdk_headers = dingtalk_robot_models.RobotRecallEmotionHeaders(
                    x_acs_dingtalk_access_token=token,
                )
                await self._robot_sdk.robot_recall_emotion_with_options_async(
                    request, sdk_headers, runtime
                )
            else:
                emotion_kwargs["text_emotion"] = (
                    dingtalk_robot_models.RobotReplyEmotionRequestTextEmotion(
                        emotion_id="2659900",
                        emotion_name=emoji_name,
                        text=emoji_name,
                        background_id="im_bg_1",
                    )
                )
                request = dingtalk_robot_models.RobotReplyEmotionRequest(
                    **emotion_kwargs,
                )
                sdk_headers = dingtalk_robot_models.RobotReplyEmotionHeaders(
                    x_acs_dingtalk_access_token=token,
                )
                await self._robot_sdk.robot_reply_emotion_with_options_async(
                    request, sdk_headers, runtime
                )
            logger.info(
                "[%s] _send_emotion: %s %s on msg=%s",
                self.name, action, emoji_name, open_msg_id[:24],
            )
        except Exception:
            logger.debug(
                "[%s] _send_emotion %s failed", self.name, action, exc_info=True
            )

    async def _resolve_media_codes(self, message: "ChatbotMessage") -> None:
        """Resolve download codes in message to actual URLs."""
        token = await self._get_access_token()
        if not token:
            return

        robot_code = getattr(message, "robot_code", None) or self._client_id
        codes_to_resolve = []

        # Collect codes and references to update
        # 1. Single image content
        img_content = getattr(message, "image_content", None)
        if img_content and getattr(img_content, "download_code", None):
            codes_to_resolve.append((img_content, "download_code"))

        # 2. File payload from msgtype=file callbacks
        file_content = getattr(message, "file_content", None)
        if file_content:
            for key in ("downloadCode", "download_code"):
                if self._obj_value(file_content, key):
                    codes_to_resolve.append((file_content, key))
                    break

        # 3. Rich text list
        rich_text = getattr(message, "rich_text_content", None)
        if rich_text:
            rich_list = getattr(rich_text, "rich_text_list", []) or []
            for item in rich_list:
                if isinstance(item, dict):
                    for key in ("downloadCode", "pictureDownloadCode", "download_code"):
                        if item.get(key):
                            codes_to_resolve.append((item, key))

        # picture_url fallback for image downloads (when download API returns 500)
        picture_url = getattr(message, "picture_url", None)

        if not codes_to_resolve:
            return

        # Resolve all codes in parallel
        tasks = []
        for obj, key in codes_to_resolve:
            code = getattr(obj, key, None) if hasattr(obj, key) else obj.get(key)
            if code:
                tasks.append(
                    self._fetch_download_url(code, robot_code, token, obj, key, picture_url=picture_url)
                )

        await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _obj_value(obj, *keys: str) -> str:
        for key in keys:
            value = getattr(obj, key, None) if hasattr(obj, key) else None
            if value is None and isinstance(obj, dict):
                value = obj.get(key)
            if value is not None and value.__class__.__module__.startswith("unittest.mock"):
                continue
            if value:
                return str(value)
        return ""

    @staticmethod
    def _extension_from_content_type(content_type: str) -> str:
        mime = (content_type or "").split(";", 1)[0].strip().lower()
        if not mime:
            return ""
        reverse = {v: k for k, v in SUPPORTED_DOCUMENT_TYPES.items()}
        return reverse.get(mime) or mimetypes.guess_extension(mime) or ""

    def _filename_for_media(self, obj, url: str, content_type: str = "") -> str:
        filename = self._obj_value(
            obj,
            "fileName",
            "file_name",
            "filename",
            "name",
            "title",
        ).strip()
        if not filename:
            parsed_name = Path(urlparse(url).path).name
            filename = parsed_name or "dingtalk_file"

        suffix = Path(filename).suffix
        if not suffix:
            suffix = self._extension_from_content_type(content_type) or ".bin"
            filename = f"{filename}{suffix}"
        return filename

    @staticmethod
    def _filename_from_content_disposition(header_value: str) -> str:
        if not header_value:
            return ""
        match = re.search(r"filename\*?=(?:UTF-8''|\")?([^\";]+)", header_value, re.I)
        if not match:
            return ""
        return match.group(1).strip()

    async def _download_media_url_to_cache(self, url: str, obj) -> Optional[str]:
        if not self._http_client:
            return None
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return None
        try:
            response = await self._http_client.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            filename = self._filename_from_content_disposition(
                response.headers.get("content-disposition", "")
            ) or self._filename_for_media(obj, url, content_type)
            local_path = cache_document_from_bytes(response.content, filename)
            logger.info("[%s] Cached DingTalk media file at %s", self.name, local_path)
            return local_path
        except Exception as e:
            logger.warning("[%s] Failed to cache DingTalk media URL: %s", self.name, e)
            return None

    async def _fetch_download_url(
        self, code: str, robot_code: str, token: str, obj, key: str,
        picture_url: Optional[str] = None,
    ) -> Optional[str]:
        """Fetch and cache a DingTalk media file for a single download code.

        The gateway downstream expects local file paths for document parsing and
        vision/STT enrichment. Resolve the temporary DingTalk URL, download it,
        and store the local cache path back into the SDK object.

        If the API returns a 500 error and picture_url is provided, falls back
        to using the picture_url directly (temporary OSS URL from the message).

        Returns the local file path or download URL on success, None on failure.
        """
        if not HTTPX_AVAILABLE or not self._http_client:
            logger.warning(
                "[%s] httpx not available, cannot resolve media code", self.name
            )
            return None

        def store_path(path: str) -> None:
            if hasattr(obj, key):
                setattr(obj, key, path)
            elif isinstance(obj, dict):
                obj[key] = path

        async def use_download_url(url: str) -> Optional[str]:
            cached = await self._download_media_url_to_cache(url, obj)
            path = cached or url
            store_path(path)
            return path

        if self._robot_sdk:
            try:
                request = dingtalk_robot_models.RobotMessageFileDownloadRequest(
                    download_code=code,
                    robot_code=robot_code,
                )
                headers = dingtalk_robot_models.RobotMessageFileDownloadHeaders(
                    x_acs_dingtalk_access_token=token,
                )
                runtime = tea_util_models.RuntimeOptions()
                sdk_response = await self._robot_sdk.robot_message_file_download_with_options_async(
                    request, headers, runtime
                )
                body = sdk_response.body if sdk_response else None
                url = getattr(body, "download_url", None) if body else None
                if url:
                    return await use_download_url(url)
                logger.warning(
                    "[%s] Robot SDK returned no download URL for code %s",
                    self.name, code,
                )
            except Exception as e:
                logger.warning("[%s] Robot SDK media download failed for code %s: %s", self.name, code, e)

        try:
            endpoint = "https://api.dingtalk.com/v1.0/robot/messageFiles/download"
            headers = {
                "Content-Type": "application/json",
                "x-acs-dingtalk-access-token": token,
            }
            payload = {
                "downloadCode": code,
                "robotCode": robot_code,
            }
            response = await self._http_client.post(endpoint, json=payload, headers=headers, timeout=15.0)
            response.raise_for_status()
            result = response.json()
            url = result.get("data", {}).get("downloadUrl") or result.get("downloadUrl")
            if url:
                return await use_download_url(url)
            logger.warning(
                "[%s] Failed to get download URL: empty response for code %s, body=%s",
                self.name, code, str(result)[:200],
            )
        except Exception as e:
            logger.error("[%s] Error resolving media code %s: %s", self.name, code, e)

        # Fallback: if API failed and we have a pictureUrl, try downloading directly
        # This handles the case where DingTalk's download API returns 500 but the
        # temporary OSS URL from the message payload is still valid.
        if picture_url:
            try:
                download_response = await self._http_client.get(picture_url, timeout=30.0)
                if download_response.status_code == 200:
                    ext = self._guess_extension_from_url(picture_url) or ".jpg"
                    filename = self._filename_for_media(
                        obj, picture_url, download_response.headers.get("content-type", "")
                    )
                    local_path = cache_document_from_bytes(
                        download_response.content, filename or f"dingtalk_picture{ext}"
                    )
                    logger.info("[%s] Downloaded picture via pictureUrl fallback to: %s", self.name, local_path)
                    store_path(local_path)
                    return local_path
                else:
                    logger.warning(
                        "[%s] pictureUrl fallback also failed (HTTP %d) for code %s",
                        self.name, download_response.status_code, code,
                    )
            except Exception as fallback_err:
                logger.warning(
                    "[%s] pictureUrl fallback failed with exception: %s",
                    self.name, fallback_err,
                )

        return None

    @staticmethod
    def _get_video_duration_ms(video_path: str) -> Optional[int]:
        """Parse video duration from MP4 moov/mvhd atom.

        Returns duration in milliseconds, or None if it cannot be determined.
        Does not require external libraries — reads the MP4 container directly.
        """
        try:
            with open(video_path, "rb") as f:
                data = f.read()
            # Find moov atom
            moov_pos = data.find(b"moov")
            if moov_pos == -1:
                return None
            moov_data = data[moov_pos:]
            # Find mvhd inside moov
            mvhd_pos = moov_data.find(b"mvhd")
            if mvhd_pos == -1:
                return None
            mvhd_payload = moov_data[mvhd_pos + 4:]  # skip size(4) + 'mvhd'(4)
            version = mvhd_payload[0]
            if version == 0:
                # 32-bit: timescale at offset 12, duration at offset 16
                timescale = struct.unpack(">I", mvhd_payload[12:16])[0]
                duration = struct.unpack(">I", mvhd_payload[16:20])[0]
            else:
                # 64-bit: timescale at offset 20, duration at offset 24
                timescale = struct.unpack(">I", mvhd_payload[20:24])[0]
                duration = struct.unpack(">Q", mvhd_payload[24:32])[0]
            if timescale == 0:
                return None
            return int(duration / timescale * 1000)
        except Exception:
            return None

    def _guess_extension_from_url(self, url: str) -> Optional[str]:
        """Guess file extension from URL path."""
        path = url.split("?")[0]
        _, ext = os.path.splitext(path)
        ext_map = {
            ".jpg": ".jpg",
            ".jpeg": ".jpg",
            ".png": ".png",
            ".gif": ".gif",
            ".mp4": ".mp4",
            ".mp3": ".mp3",
            ".wav": ".wav",
            ".pdf": ".pdf",
            ".doc": ".doc",
            ".docx": ".docx",
            ".xls": ".xls",
            ".xlsx": ".xlsx",
            ".ppt": ".ppt",
            ".pptx": ".pptx",
        }
        normalized = ext_map.get(ext.lower())
        if normalized:
            return normalized
        if ext:
            return ext.lower()
        return None

    @staticmethod
    def _normalize_markdown(text: str) -> str:
        """Normalize markdown for DingTalk's parser.

        DingTalk's markdown renderer has quirks:
        - Numbered lists need blank line before them
        - Indented code blocks may render incorrectly
        """
        lines = text.split("\n")
        out = []
        for i, line in enumerate(lines):
            # Ensure blank line before numbered list items
            is_numbered = re.match(r"^\d+\.\s", line.strip())
            if is_numbered and i > 0:
                prev = lines[i - 1]
                if prev.strip() and not re.match(r"^\d+\.\s", prev.strip()):
                    out.append("")
            # Dedent fenced code blocks
            if line.strip().startswith("```") and line != line.lstrip():
                indent = len(line) - len(line.lstrip())
                line = line[indent:]
            out.append(line)
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Internal stream handler
# ---------------------------------------------------------------------------


class _IncomingHandler(
    dingtalk_stream.ChatbotHandler if DINGTALK_STREAM_AVAILABLE else object
):
    """dingtalk-stream ChatbotHandler that forwards messages to the adapter.

    SDK >= 0.20 changed process() from sync to async, and the message
    parameter from ChatbotMessage to CallbackMessage. We parse the
    CallbackMessage.data dict into a ChatbotMessage before forwarding.
    """

    def __init__(self, adapter: DingTalkAdapter, loop: Optional[asyncio.AbstractEventLoop] = None):
        if DINGTALK_STREAM_AVAILABLE:
            super().__init__()
        self._adapter = adapter
        self._loop = loop

    @staticmethod
    def _chatbot_message_from_raw(data: Any) -> Any:
        if not isinstance(data, dict):
            return SimpleNamespace(raw_payload=data)

        msg_type = data.get("msgtype") or data.get("messageType") or ""
        content = data.get("content")
        text = data.get("text")
        if text is None and msg_type == "text" and isinstance(content, dict):
            text = content

        rich_list = None
        if isinstance(content, dict):
            rich_list = content.get("richText") or content.get("rich_text")
        rich_text_content = (
            SimpleNamespace(rich_text_list=rich_list)
            if isinstance(rich_list, list)
            else None
        )

        msg = SimpleNamespace(
            message_type=msg_type,
            text=text,
            rich_text=rich_list,
            rich_text_content=rich_text_content,
            sender_id=data.get("senderId") or data.get("sender_id") or "",
            sender_staff_id=data.get("senderStaffId") or data.get("sender_staff_id") or "",
            sender_nick=data.get("senderNick") or data.get("sender_nick") or "",
            conversation_id=data.get("conversationId") or data.get("conversation_id") or "",
            conversation_type=data.get("conversationType") or data.get("conversation_type") or "1",
            message_id=data.get("msgId") or data.get("messageId") or data.get("message_id") or "",
            session_webhook=data.get("sessionWebhook") or data.get("session_webhook") or "",
            session_webhook_expired_time=(
                data.get("sessionWebhookExpiredTime")
                or data.get("session_webhook_expired_time")
                or 0
            ),
            create_at=data.get("createAt") or data.get("create_at") or 0,
            at_users=data.get("atUsers") or data.get("at_users") or [],
            is_in_at_list=bool(data.get("isInAtList") or data.get("is_in_at_list")),
            raw_payload=data,
        )
        if msg_type == "file" and isinstance(content, dict):
            msg.file_content = content
        if msg_type == "picture" and isinstance(content, dict):
            msg.picture_url = content.get("pictureUrl")
        return msg

    @staticmethod
    def _diag_value(value: Any) -> Any:
        """Return a redacted, compact value for DingTalk payload diagnostics."""
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            compact = value.replace("\n", " ")
            if len(compact) <= 96:
                return compact
            return f"{compact[:48]}...{compact[-16:]}(len={len(compact)})"
        if isinstance(value, list):
            return f"list[{len(value)}]"
        if isinstance(value, dict):
            return f"dict[{len(value)}]"
        return type(value).__name__

    @classmethod
    def _summarize_raw_payload(cls, data: Any) -> Dict[str, Any]:
        """Summarize attachment-related DingTalk callback fields without logging secrets."""
        if not isinstance(data, dict):
            return {"data_type": type(data).__name__}

        content = data.get("content")
        summary: Dict[str, Any] = {
            "msgtype": data.get("msgtype"),
            "messageId": cls._diag_value(data.get("messageId") or data.get("msgId")),
            "conversationId": cls._diag_value(data.get("conversationId")),
            "top_keys": sorted(str(k) for k in data.keys()),
            "content_type": type(content).__name__,
        }
        if isinstance(content, dict):
            summary["content_keys"] = sorted(str(k) for k in content.keys())
            for key in (
                "downloadCode",
                "pictureDownloadCode",
                "fileName",
                "filename",
                "name",
                "title",
                "extension",
                "fileType",
                "pictureUrl",
                "mediaId",
            ):
                if key in content:
                    summary[f"content.{key}"] = cls._diag_value(content.get(key))
            rich_list = content.get("richText") or content.get("rich_text")
            if isinstance(rich_list, list):
                summary["richText_count"] = len(rich_list)
                items = []
                interesting = (
                    "type",
                    "text",
                    "content",
                    "downloadCode",
                    "download_code",
                    "pictureDownloadCode",
                    "fileName",
                    "filename",
                    "name",
                    "title",
                    "extension",
                    "fileType",
                    "pictureUrl",
                    "mediaId",
                )
                for item in rich_list[:8]:
                    if isinstance(item, dict):
                        items.append(
                            {
                                "keys": sorted(str(k) for k in item.keys()),
                                **{k: cls._diag_value(item.get(k)) for k in interesting if k in item},
                            }
                        )
                    else:
                        items.append({"item_type": type(item).__name__})
                summary["richText_items"] = items
        else:
            summary["content"] = cls._diag_value(content)
        return summary

    @classmethod
    def _summarize_chatbot_media(cls, chatbot_msg: "ChatbotMessage") -> Dict[str, Any]:
        rich_text = getattr(chatbot_msg, "rich_text_content", None) or getattr(
            chatbot_msg, "rich_text", None
        )
        rich_list = getattr(rich_text, "rich_text_list", None) if rich_text else None
        image_content = getattr(chatbot_msg, "image_content", None)
        return {
            "message_type": getattr(chatbot_msg, "message_type", None),
            "has_image_content": bool(image_content),
            "image_download_code": bool(getattr(image_content, "download_code", None))
            if image_content
            else False,
            "has_rich_text": bool(rich_text),
            "rich_text_type": type(rich_text).__name__ if rich_text else None,
            "rich_text_count": len(rich_list) if isinstance(rich_list, list) else None,
            "has_picture_url": bool(getattr(chatbot_msg, "picture_url", None)),
        }

    def pre_start(self) -> None:
        """No-op pre-start hook required by dingtalk-stream SDK.

        The SDK calls ``pre_start()`` on every registered handler before
        opening the WebSocket connection.  Without this method, the SDK
        raises ``AttributeError: '_IncomingHandler' object has no
        attribute 'pre_start'`` and kills the stream connection.
        """
        return

    async def process(self, message: "CallbackMessage"):
        """Called by dingtalk-stream (>=0.20) when a message arrives.

        dingtalk-stream >= 0.24 passes a CallbackMessage whose ``.data`` contains
        the chatbot payload. Convert it to ChatbotMessage via
        ``ChatbotMessage.from_dict()``.

        Message processing is dispatched as a background task so that this
        method returns the ACK immediately — blocking here would prevent the
        SDK from sending heartbeats, eventually causing a disconnect.
        """
        try:
            # CallbackMessage.data is a dict containing the raw DingTalk payload
            data = message.data
            if isinstance(data, str):
                data = json.loads(data)

            if isinstance(data, dict):
                try:
                    logger.debug(
                        "[%s] DingTalk raw callback summary: %s",
                        self._adapter.name,
                        json.dumps(
                            self._summarize_raw_payload(data),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                except Exception:
                    logger.exception(
                        "[%s] Failed to summarize DingTalk raw callback",
                        self._adapter.name,
                    )

            # Parse dict into ChatbotMessage using SDK's from_dict when the
            # optional dingtalk-stream package is installed; otherwise build a
            # small message object with the fields the adapter uses.
            if ChatbotMessage is not None and hasattr(ChatbotMessage, "from_dict"):
                chatbot_msg = ChatbotMessage.from_dict(data)
            else:
                chatbot_msg = self._chatbot_message_from_raw(data)
            if isinstance(data, dict):
                chatbot_msg.raw_payload = data

            # dingtalk-stream does not currently map msgtype=file content into
            # a first-class SDK attribute, so preserve the raw file payload for
            # the existing media download/cache pipeline.
            if isinstance(data, dict) and data.get("msgtype") == "file":
                raw_content = data.get("content")
                if isinstance(raw_content, dict):
                    chatbot_msg.file_content = raw_content

            try:
                logger.debug(
                    "[%s] DingTalk SDK media summary: %s",
                    self._adapter.name,
                    json.dumps(
                        self._summarize_chatbot_media(chatbot_msg),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            except Exception:
                logger.exception(
                    "[%s] Failed to summarize DingTalk SDK message",
                    self._adapter.name,
                )

            # Ensure session_webhook is populated even if the SDK's
            # from_dict() did not map it (field name mismatch across
            # SDK versions).
            if not getattr(chatbot_msg, "session_webhook", None):
                webhook = (
                    data.get("sessionWebhook")
                    or data.get("session_webhook")
                    or ""
                ) if isinstance(data, dict) else ""
                if webhook:
                    chatbot_msg.session_webhook = webhook

            # Ensure is_in_at_list is populated from the structured callback
            # flag even if from_dict() did not map it.  DingTalk sends
            # ``isInAtList`` in the raw payload; the adapter's mention check
            # reads the ChatbotMessage attribute ``is_in_at_list``.
            if not getattr(chatbot_msg, "is_in_at_list", False):
                raw_flag = (
                    data.get("isInAtList") if isinstance(data, dict) else False
                )
                if raw_flag:
                    chatbot_msg.is_in_at_list = True

            # Preserve pictureUrl from raw content for image fallback download.
            # The DingTalk callback includes a temporary OSS URL (pictureUrl) for
            # images which can be used when the download API returns 500.
            # pictureUrl appears in both msgtype=picture AND msgtype=richText (image in rich text).
            picture_url = None
            if isinstance(data, dict):
                msg_type = data.get("msgtype")
                if msg_type == "picture":
                    raw_content = data.get("content")
                    if isinstance(raw_content, dict):
                        picture_url = raw_content.get("pictureUrl")
                elif msg_type == "richText":
                    # Images embedded in rich text are in content.richText[].pictureUrl
                    raw_content = data.get("content")
                    if isinstance(raw_content, dict):
                        rich_list = raw_content.get("richText") or []
                        if isinstance(rich_list, list):
                            for item in rich_list:
                                if isinstance(item, dict) and item.get("type") == "image":
                                    picture_url = item.get("pictureUrl")
                                    if picture_url:
                                        break
            if picture_url:
                logger.info("[%s] Extracted pictureUrl for fallback: %.50s...", self._adapter.name, picture_url)
                chatbot_msg.picture_url = picture_url

            msg_id = getattr(chatbot_msg, "message_id", None) or ""
            conversation_id = getattr(chatbot_msg, "conversation_id", None) or ""

            # Thinking reaction — fire-and-forget, tracked
            if msg_id and conversation_id:
                self._adapter._spawn_bg(
                    self._adapter._send_emotion(
                        msg_id, conversation_id, "🤔Thinking", recall=False,
                    )
                )

            # Fire-and-forget: return ACK immediately, process in background.
            # Blocking here would prevent the SDK from sending heartbeats,
            # eventually causing a disconnect.  _on_message is wrapped so
            # exceptions inside the task surface in logs instead of
            # disappearing into the event loop.
            asyncio.create_task(self._safe_on_message(chatbot_msg))
        except Exception:
            logger.exception(
                "[%s] Error preparing incoming message", self._adapter.name
            )
            return AckMessage.STATUS_SYSTEM_EXCEPTION, "error"

        return AckMessage.STATUS_OK, "OK"

    async def _safe_on_message(self, chatbot_msg: "ChatbotMessage") -> None:
        """Wrapper that catches exceptions from _on_message."""
        try:
            await self._adapter._on_message(chatbot_msg)
        except Exception:
            logger.exception(
                "[%s] Error processing incoming message", self._adapter.name
            )


# ──────────────────────────────────────────────────────────────────────────
# Plugin migration glue (#41112 / #3823)
#
# Added when the DingTalk adapter moved from gateway/platforms/dingtalk.py into
# this bundled plugin. Mirrors the Discord (#24356) / Slack migrations: a
# register(ctx) entry point plus hook implementations that replace the
# per-platform core touchpoints (the Platform.DINGTALK elif in gateway/run.py,
# the dingtalk_cfg YAML→env block + _PLATFORM_CONNECTED_CHECKERS entry in
# gateway/config.py, the _setup_dingtalk wizard + _PLATFORMS["dingtalk"] static
# dict in hermes_cli/gateway.py, and the _send_dingtalk dispatch in
# tools/send_message_tool.py).
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
    """Out-of-process DingTalk delivery via a static robot webhook URL.

    Implements the standalone_sender_fn contract so deliver=dingtalk cron jobs
    succeed when cron runs separately from the gateway. The live adapter uses
    per-session webhook URLs from incoming messages, which aren't available
    out-of-process; this path uses the static DINGTALK_WEBHOOK_URL / extra
    webhook_url instead. Replaces the legacy _send_dingtalk helper.
    """
    extra = getattr(pconfig, "extra", {}) or {}
    try:
        import httpx
    except ImportError:
        return {"error": "httpx not installed"}
    try:
        webhook_url = extra.get("webhook_url") or os.getenv("DINGTALK_WEBHOOK_URL", "")
        if not webhook_url:
            return {"error": "DingTalk not configured. Set DINGTALK_WEBHOOK_URL env var or webhook_url in dingtalk platform extra config."}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                webhook_url,
                json={"msgtype": "text", "text": {"content": message}},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("errcode", 0) != 0:
                return {"error": f"DingTalk API error: {data.get('errmsg', 'unknown')}"}
        return {"success": True, "platform": "dingtalk", "chat_id": chat_id}
    except Exception as e:
        # Redact the access_token from webhook URLs that may appear in the
        # exception text. Reuse send_message_tool._error's redaction so the
        # logic stays single-sourced (lazy import avoids a circular at module
        # load). Falls back to a plain message if that helper is unavailable.
        try:
            from tools.send_message_tool import _error as _redact_error
            return _redact_error(f"DingTalk send failed: {e}")
        except Exception:
            return {"error": f"DingTalk send failed: {e}"}


def interactive_setup() -> None:
    """Configure DingTalk — QR scan (recommended) or manual credential entry.

    Replaces hermes_cli/setup.py-era _setup_dingtalk + the static
    _PLATFORMS["dingtalk"] dict in hermes_cli/gateway.py. CLI helpers are
    lazy-imported so the plugin's module-load surface stays minimal.
    """
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli.setup import prompt_choice
    from hermes_cli.cli_output import (
        prompt,
        prompt_yes_no,
        print_header,
        print_success,
        print_warning,
    )

    print_header("DingTalk")
    existing = get_env_value("DINGTALK_CLIENT_ID")
    if existing:
        print_success(f"DingTalk is already configured (Client ID: {existing}).")
        if not prompt_yes_no("Reconfigure DingTalk?", False):
            return

    method = prompt_choice(
        "Choose setup method",
        [
            "QR Code Scan (Recommended, auto-obtain Client ID and Client Secret)",
            "Manual Input (Client ID and Client Secret)",
        ],
        default=0,
    )

    if method == 0:
        try:
            from hermes_cli.dingtalk_auth import dingtalk_qr_auth
        except ImportError as exc:
            print_warning(f"QR auth module failed to load ({exc}), falling back to manual input.")
            _manual_credential_entry(prompt, save_env_value, print_success)
            return
        result = dingtalk_qr_auth()
        if result is None:
            print_warning("QR auth incomplete, falling back to manual input.")
            _manual_credential_entry(prompt, save_env_value, print_success)
            return
        client_id, client_secret = result
        save_env_value("DINGTALK_CLIENT_ID", client_id)
        save_env_value("DINGTALK_CLIENT_SECRET", client_secret)
        print_success("DingTalk configured via QR scan!")
    else:
        _manual_credential_entry(prompt, save_env_value, print_success)


def _manual_credential_entry(prompt, save_env_value, print_success) -> None:
    client_id = prompt("DingTalk Client ID (app key)")
    if not client_id:
        return
    save_env_value("DINGTALK_CLIENT_ID", client_id)
    client_secret = prompt("DingTalk Client Secret", password=True)
    if client_secret:
        save_env_value("DINGTALK_CLIENT_SECRET", client_secret)
    print_success("DingTalk credentials saved")


def _apply_yaml_config(yaml_cfg: dict, dingtalk_cfg: dict) -> dict | None:
    """Translate config.yaml dingtalk: keys into DINGTALK_* env vars.

    Implements the apply_yaml_config_fn contract (#24849). Mirrors the legacy
    dingtalk_cfg block from gateway/config.py::load_gateway_config(). Env vars
    take precedence over YAML (each assignment guarded by not os.getenv(...)).
    Returns None — everything flows through env.
    """
    import json as _json
    if "require_mention" in dingtalk_cfg and not os.getenv("DINGTALK_REQUIRE_MENTION"):
        os.environ["DINGTALK_REQUIRE_MENTION"] = str(dingtalk_cfg["require_mention"]).lower()
    if "mention_patterns" in dingtalk_cfg and not os.getenv("DINGTALK_MENTION_PATTERNS"):
        os.environ["DINGTALK_MENTION_PATTERNS"] = _json.dumps(dingtalk_cfg["mention_patterns"])
    frc = dingtalk_cfg.get("free_response_chats")
    if frc is not None and not os.getenv("DINGTALK_FREE_RESPONSE_CHATS"):
        if isinstance(frc, list):
            frc = ",".join(str(v) for v in frc)
        os.environ["DINGTALK_FREE_RESPONSE_CHATS"] = str(frc)
    ac = dingtalk_cfg.get("allowed_chats")
    if ac is not None and not os.getenv("DINGTALK_ALLOWED_CHATS"):
        if isinstance(ac, list):
            ac = ",".join(str(v) for v in ac)
        os.environ["DINGTALK_ALLOWED_CHATS"] = str(ac)
    allowed = dingtalk_cfg.get("allowed_users")
    if allowed is not None and not os.getenv("DINGTALK_ALLOWED_USERS"):
        if isinstance(allowed, list):
            allowed = ",".join(str(v) for v in allowed)
        os.environ["DINGTALK_ALLOWED_USERS"] = str(allowed)
    return None


def _is_connected(config) -> bool:
    """DingTalk is connected when client_id + client_secret are present.

    Mirrors the legacy _PLATFORM_CONNECTED_CHECKERS[Platform.DINGTALK] entry.
    Reads from PlatformConfig.extra first, then env vars.
    """
    extra = getattr(config, "extra", {}) or {}
    return bool(
        (extra.get("client_id") or os.getenv("DINGTALK_CLIENT_ID"))
        and (extra.get("client_secret") or os.getenv("DINGTALK_CLIENT_SECRET"))
    )


def _build_adapter(config):
    """Factory wrapper that constructs DingTalkAdapter from a PlatformConfig."""
    return DingTalkAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="dingtalk",
        label="DingTalk",
        adapter_factory=_build_adapter,
        check_fn=check_dingtalk_requirements,
        is_connected=_is_connected,
        validate_config=_is_connected,
        required_env=["DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET"],
        install_hint="pip install 'dingtalk-stream>=0.20' httpx",
        setup_fn=interactive_setup,
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="DINGTALK_ALLOWED_USERS",
        allow_all_env="DINGTALK_ALLOW_ALL_USERS",
        cron_deliver_env_var="DINGTALK_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        emoji="🐳",
        allow_update_command=True,
    )
