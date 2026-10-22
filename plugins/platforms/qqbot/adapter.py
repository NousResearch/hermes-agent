"""QQ Bot platform adapter (Official QQ Bot API v2).

``qqbot-agent-sdk`` owns the gateway socket, access token, and REST calls.
This adapter keeps session routing, access policy, inbound dispatch, and delivery.

config.yaml ``platforms.qq.extra``: app_id / client_secret (or QQ_APP_ID /
QQ_CLIENT_SECRET), markdown_support, dm_policy + allow_from, group_policy +
group_allow_from (open | allowlist | disabled | pairing), and optional ``stt``
{provider, baseUrl, apiKey, model} (or QQ_STT_* env vars). Voice transcription
tries QQ's free ``asr_refer_text`` first, then the configured STT provider.
"""

from __future__ import annotations

from pm import install_hint
import asyncio
import contextlib
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter, ExecApprovalPrompt, SendResult,
    cache_document_from_bytes_async, cache_image_from_bytes_async,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.helpers import strip_markdown
from gateway.platforms.helpers import MessageDeduplicator, platform_http_client
from gateway.platforms.access_policy_mixin import OwnAccessPolicyMixin
from gateway.platforms.media_cache import ext_for_mime

logger = logging.getLogger(__name__)


from .constants import (
    DEDUP_WINDOW_SECONDS, DEDUP_MAX_SIZE, MAX_MESSAGE_LENGTH,
    MEDIA_TYPE_IMAGE, MEDIA_TYPE_VIDEO, MEDIA_TYPE_VOICE, MEDIA_TYPE_FILE)
from .utils import coerce_list as _coerce_list
from .keyboards import (
    ApprovalRequest, InlineKeyboard, InteractionEvent, build_approval_keyboard,
    build_update_prompt_keyboard, parse_approval_button_data, parse_interaction_event,
    parse_update_prompt_button_data)
from gateway.platforms._shared import get_scoped_secret as _resolve_qq_secret


def check_qq_requirements() -> bool:
    """httpx plus qqbot-agent-sdk. The SDK ships with the ``all`` extra."""
    if not HTTPX_AVAILABLE:
        return False
    try:
        import qqbot_agent_sdk  # presence check; the name is unused
    except ImportError:
        return False
    return True


_VOICE_EXTENSIONS = (".silk", ".amr", ".mp3", ".wav", ".ogg", ".m4a", ".aac", ".speex", ".flac")
_STT_PROVIDER_BASE_URLS = {
    "zai": "https://open.bigmodel.cn/api/coding/paas/v4",
    # Aliases that target direct REST APIs not modeled as first-class providers in PROVIDER_REGISTRY. Used
    # for ``auxiliary.<task>.provider`` so users can write the obvious name and have it resolve to a working
    # ``custom`` endpoint without needing to know our internal provider IDs. Why these specifically:
    # PROVIDER_REGISTRY has ``openai-codex`` (OAuth) and ``custom`` (manual base_url + OPENAI_API_KEY) but
    # no plain ``openai`` for direct API-key access. Users predictably type ``provider: openai`` and expect
    # it to use OPENAI_API_KEY against api.openai.com. Previously this silently fell back to the user's main
    # provider, sending OpenAI model names to e.g. DeepSeek and producing cryptic ``unknown variant
    # 'image_url'`` errors (issue #31179).
    "openai": "https://api.openai.com/v1",
    "glm": "https://open.bigmodel.cn/api/coding/paas/v4"}
_AUDIO_URL_EXTENSIONS = {".silk", ".amr", ".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}


class QQAdapter(OwnAccessPolicyMixin, BasePlatformAdapter):
    """QQ Bot adapter backed by the official QQ Bot WebSocket Gateway + REST API."""

    # QQ Bot API does not support editing sent messages.
    SUPPORTS_MESSAGE_EDITING = False
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    ALLOW_ALL_ENV_PREFIX = "QQ"
    _TYPING_INPUT_SECONDS = 60  # input_notify duration reported to QQ
    _TYPING_DEBOUNCE_SECONDS = 50  # refresh before it expires

    @property
    def _log_tag(self) -> str:
        """Log prefix including app_id for multi-instance disambiguation."""
        app_id = getattr(self, "_app_id", None)
        return f"QQBot:{app_id}" if app_id else "QQBot"

    def _fail_pending(self, reason: str) -> None:
        for fut in self._pending_responses.values():
            if not fut.done():
                fut.set_exception(RuntimeError(reason))
        self._pending_responses.clear()

    def _mark_transport_disconnected(self) -> None:
        """Mark QQ WS down without stopping the reconnect loop (base's _running
        doubles as lifecycle flag; the listener must survive transient drops)."""
        if self.has_fatal_error:
            return
        self._write_runtime_status_safe(
            "disconnected", platform_state="disconnected", error_code=None, error_message=None)

    @property
    def is_connected(self) -> bool:
        """Return True only when the QQ WebSocket transport is usable."""
        return bool(self._running and self._qq_ws is not None and self._sdk_connected)

    @property
    def send_path_degraded(self) -> bool:
        """True until the SDK reports READY. The gateway thread starts before the socket is up."""
        return not self._sdk_connected

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.QQBOT)

        extra = config.extra or {}
        self._app_id = str(extra.get("app_id") or _resolve_qq_secret("QQ_APP_ID", "")).strip()
        self._client_secret = str(extra.get("client_secret") or _resolve_qq_secret("QQ_CLIENT_SECRET", "")).strip()
        self._markdown_support = bool(extra.get("markdown_support", True))
        self._dm_policy = str(extra.get("dm_policy", "pairing")).strip().lower()
        self._allow_from = _coerce_list(extra.get("allow_from") or extra.get("allowFrom"))
        self._group_policy = str(extra.get("group_policy", "pairing")).strip().lower()
        self._group_allow_from = _coerce_list(extra.get("group_allow_from") or extra.get("groupAllowFrom"))

        self._http_client: Optional[httpx.AsyncClient] = None
        self._qq_api: Any = None
        self._qq_ws: Any = None
        self._sdk_connected = False
        self._session_id: Optional[str] = None
        self._last_seq: Optional[int] = None
        self._chat_type_map: dict[str, str] = {}  # chat_id → "c2c"|"group"|"guild"|"dm"
        self._pending_responses: dict[str, asyncio.Future] = {}  # request/response correlation
        self._dedup = MessageDeduplicator(max_size=DEDUP_MAX_SIZE, ttl_seconds=DEDUP_WINDOW_SECONDS)
        self._last_msg_id: dict[str, str] = {}  # last inbound message ID per chat (send_typing)
        self._typing_sent_at: dict[str, float] = {}  # typing debounce: chat_id → last send_typing ts

        # Inline-keyboard interaction routing: invoked for every INTERACTION_CREATE
        # after the adapter ACKed it. Defaults to the approval/update-prompt
        # dispatcher; override via set_interaction_callback() (None drops clicks).
        self._interaction_callback: Optional[Callable[[InteractionEvent], Awaitable[None]]] = (
            self._default_interaction_dispatch)

    # ── Properties ──

    @property
    def name(self) -> str:
        return "QQBot"

    # ── Connection lifecycle ──

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Authenticate, obtain gateway URL, and open the WebSocket. ``is_reconnect``
        is accepted for interface conformance only (QQBot has no server-side update queue)."""
        for ok, code, what, hint in (
            (HTTPX_AVAILABLE, "qq_missing_dependency", "httpx not installed", ". Run: hermes pm repair"),
            (self._app_id and self._client_secret, "qq_missing_credentials",
             "QQ_APP_ID and QQ_CLIENT_SECRET are required", "")):
            if not ok:
                message = f"QQ startup failed: {what}"
                self._set_fatal_error(code, message, retryable=True)
                logger.warning("[%s] %s%s", self._log_tag, message, hint)
                return False

        if not self._acquire_platform_lock("qqbot-appid", self._app_id, "QQBot app ID"):
            return False

        try:
            # Tighter keepalive pool so idle CLOSE_WAIT sockets drain faster behind proxies.
            # See #18451. Injected into the SDK so attachment downloads keep the SSRF guard.
            from qqbot_agent_sdk import QQApiClient

            from .sdk_bridge import apply_sdk_identity, scoped_websocket_class

            apply_sdk_identity()
            self._http_client = platform_http_client()
            self._qq_api = QQApiClient(self._app_id, self._client_secret, log_tag=self._log_tag)
            self._qq_api.setup(self._http_client)
            self._qq_ws = scoped_websocket_class()(callbacks=self._sdk_callbacks(), log_tag=self._log_tag)
            gateway_url = await self._qq_api.get_gateway_url()
            logger.info("[%s] Connecting to QQ gateway", self._log_tag)
            self._qq_ws.start(gateway_url, asyncio.get_running_loop())
            # READY (on_connected) is the first proof the socket is up. Until then the
            # gateway shows retrying via send_path_degraded.
            self._mark_connected()
            logger.info("[%s] QQ gateway thread started", self._log_tag)
            self._wire_plugin_handlers(None)
            return True
        except Exception as exc:  # health: allow BLE001 -- startup failure is the QQ response text; a traceback repeats it in the gateway log
            message = f"QQ startup failed: {exc}"
            self._set_fatal_error("qq_connect_error", message, retryable=True)
            logger.error("[%s] %s", self._log_tag, message)
            await self._cleanup()
            self._release_platform_lock()
            return False

    def _sdk_callbacks(self):
        """Sync token/session hooks run on the SDK thread; inbound events hop back here."""
        from qqbot_agent_sdk import WSCallbacks

        def _set_session(session_id, seq):
            self._session_id = session_id
            self._last_seq = seq

        def _on_connected():
            self._sdk_connected = True
            if self._running and not self.has_fatal_error:
                self._mark_connected()
            logger.info("[%s] Connected", self._log_tag)

        def _on_disconnected():
            if self._qq_ws is None:
                return
            self._sdk_connected = False
            self._mark_transport_disconnected()

        def _on_fatal(code, message):
            if self._qq_ws is None:
                return
            self._sdk_connected = False
            self._set_fatal_error(str(code), str(message), retryable=True)
            # The SDK does not retry an open() that dies before start_listeners().
            # Ask the gateway to restart this adapter; its own reconnect loop never started.
            if str(code) != "qq_gateway_open_failed":
                return
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            # The loop only keeps a weak ref; hold the task on the adapter until it finishes.
            task = loop.create_task(self._notify_fatal_error())
            self._qq_open_failed_task = task
            task.add_done_callback(lambda _done: setattr(self, "_qq_open_failed_task", None))

        async def _on_interaction_event(_event_type, data):
            await self._on_interaction(data)

        return WSCallbacks(
            on_message_event=self._on_message,
            on_connected=_on_connected,
            on_disconnected=_on_disconnected,
            on_fatal_error=_on_fatal,
            get_token=self._qq_api.ensure_token_sync,
            get_session=lambda: (self._session_id, self._last_seq),
            set_session=_set_session,
            set_heartbeat_interval=lambda _seconds: None,
            clear_token=self._qq_api.clear_token,
            fail_pending=self._fail_pending,
            get_gateway_url=self._qq_api.get_gateway_url_sync,
            on_interaction_event=_on_interaction_event,
        )

    async def disconnect(self) -> None:
        self._sdk_connected = False
        self._running = False
        self._mark_disconnected()
        ws = self._qq_ws
        self._qq_ws = None
        if ws is not None and getattr(ws, "_ws_thread", None) is not None:
            await ws.async_stop()
        self._qq_api = None
        await self._cleanup()
        self._release_platform_lock()
        logger.info("[%s] Disconnected", self._log_tag)

    async def _cleanup(self) -> None:
        """Close the injected HTTP client and fail pending futures."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._fail_pending("Disconnected")

    # ── Inbound message handling ──

    async def handle_message(self, event: MessageEvent) -> None:
        """Cache the last message ID per chat, then delegate to base."""
        if event.message_id and event.source.chat_id:
            self._last_msg_id[event.source.chat_id] = event.message_id
        await super().handle_message(event)

    async def _on_message(self, event_type: str, d: Any) -> None:
        if not isinstance(d, dict):
            return
        msg_id = str(d.get("id", ""))
        if not msg_id or self._dedup.is_duplicate(msg_id):
            logger.debug("[%s] Duplicate or missing message id: %s", self._log_tag, msg_id)
            return
        handler = self._INBOUND_HANDLERS.get(event_type)
        if handler:
            author = d.get("author") if isinstance(d.get("author"), dict) else {}
            await getattr(self, handler)(
                d, msg_id, str(d.get("content", "")).strip(), author, str(d.get("timestamp", "")))

    # ── Inline-keyboard interactions (INTERACTION_CREATE) ──

    def set_interaction_callback(self, callback: Optional[Callable[[InteractionEvent], Awaitable[None]]]) -> None:
        """Register (or clear) the callback invoked per ACKed INTERACTION_CREATE."""
        self._interaction_callback = callback

    async def _on_interaction(self, d: Any) -> None:
        """Parse INTERACTION_CREATE, ACK it promptly (else the client shows an error
        icon on the button), then dispatch to the registered callback."""
        if not isinstance(d, dict):
            return
        try:
            event = parse_interaction_event(d)
        except Exception as exc:
            logger.warning("[%s] Failed to parse INTERACTION_CREATE: %s", self._log_tag, exc)
            return
        if not event.id:
            logger.warning("[%s] INTERACTION_CREATE missing id, skipping ACK", self._log_tag)
            return

        try:
            await self._acknowledge_interaction(event.id)
        except Exception as exc:
            logger.warning("[%s] Failed to ACK interaction %s: %s", self._log_tag, event.id, exc)

        logger.info(
            "[%s] Interaction: scene=%s button_data=%r operator=%s",
            self._log_tag, event.scene, event.button_data, event.operator_openid)
        callback = self._interaction_callback
        if callback is None:
            logger.debug(
                "[%s] No interaction callback registered; dropping button click %r", self._log_tag, event.button_data
            )
            return
        try:
            await callback(event)
        except Exception as exc:
            logger.error("[%s] Interaction callback raised: %s", self._log_tag, exc, exc_info=True)

    async def _acknowledge_interaction(self, interaction_id: str, code: int = 0) -> None:
        """ACK a button interaction through the SDK (code 0 = success)."""
        await self._require_api().acknowledge_interaction(interaction_id, code=code)

    # Button decision → ``choice`` for tools.approval.resolve_gateway_approval. The
    # 3-button layout folds "session" into "always"; ``/approve session`` still works.
    _APPROVAL_BUTTON_TO_CHOICE = {"allow-once": "once", "allow-always": "always", "deny": "deny"}

    @staticmethod
    def _parse_gateway_session_key(session_key: str) -> Optional[dict[str, str]]:
        """Parse ``agent:<namespace>:<platform>:<chat_type>:<chat_id>[:<user_id>]``.

        The namespace slot carries the multiplex profile ("main" for the
        default profile — see ``gateway.session._session_key_namespace``);
        it takes no part in any authorization decision, so any non-empty
        value is accepted and every later slot keeps its position.
        """
        parts = str(session_key or "").split(":")
        if len(parts) < 5 or parts[0] != "agent" or not parts[1]:
            return None
        parsed = {"platform": parts[2], "chat_type": parts[3], "chat_id": parts[4]}
        if len(parts) > 5:
            parsed["user_id"] = parts[5]
        return parsed

    def _is_authorized_interaction_for_session(self, event: InteractionEvent, session_key: str) -> bool:
        """Authorize approval/update interactions against session + operator."""
        parsed = self._parse_gateway_session_key(session_key)
        operator = str(event.operator_openid or "").strip()
        if not parsed or parsed.get("platform") != "qqbot" or not operator:
            return False

        chat_type = parsed.get("chat_type", "")
        chat_id = parsed.get("chat_id", "")
        # Approval keys come from build_source (chat_type="dm"); update-prompt keys from event.scene ("c2c").
        if chat_type in {"c2c", "dm"}:
            return bool(chat_id) and operator == chat_id
        if chat_type in {"group", "guild"}:
            event_chat = str(event.group_openid or event.guild_id or "").strip()
            if not event_chat or event_chat != chat_id:
                return False
            session_user = str(parsed.get("user_id", "")).strip()
            return bool(session_user) and operator == session_user
        return False

    def _update_prompt_session_key(self, event: InteractionEvent, chat: str) -> str:
        """Session key an update-prompt click is authorized against, built by the ONE key builder so
        it carries the profile namespace (a hard-coded ``agent:main:`` prefix never matched a
        multiplexed secondary bot's lane). No participant: ``c2c`` authorizes on chat == operator."""
        return self._source_session_key(self.build_source(chat_id=chat, chat_type=event.scene))

    async def _default_interaction_dispatch(self, event: InteractionEvent) -> None:
        """Default interaction callback: ``approve:<session_key>:<decision>`` →
        tools.approval.resolve_gateway_approval; ``update_prompt:<answer>`` →
        ``~/.hermes/.update_response``; anything else is ignored at DEBUG."""
        button_data = event.button_data
        if not button_data:
            return

        approval = parse_approval_button_data(button_data)
        if approval is not None:
            session_key, decision = approval
            choice = self._APPROVAL_BUTTON_TO_CHOICE.get(decision)
            if choice is None:
                logger.warning("[%s] Unknown approval decision %r (session=%s)", self._log_tag, decision, session_key)
                return
            if not self._is_authorized_interaction_for_session(event, session_key):
                logger.warning(
                    "[%s] Rejected unauthorized approval click for session %s (operator=%s)",
                    self._log_tag, session_key, event.operator_openid)
                return
            try:
                from tools.approval import resolve_gateway_approval  # lazy: keep adapter light
                count = resolve_gateway_approval(session_key, choice)
                logger.info(
                    "[%s] Button resolved %d approval(s) for session %s (choice=%s, operator=%s)",
                    self._log_tag, count, session_key, choice, event.operator_openid)
            except Exception as exc:
                logger.error("[%s] resolve_gateway_approval failed for session %s: %s", self._log_tag, session_key, exc)
            return

        update_answer = parse_update_prompt_button_data(button_data)
        if update_answer is not None:
            chat = event.group_openid or event.guild_id or event.user_openid
            if not self._is_authorized_interaction_for_session(event, self._update_prompt_session_key(event, chat)):
                logger.warning(
                    "[%s] Rejected unauthorized update prompt click (operator=%s)", self._log_tag, event.operator_openid
                )
                return
            self._write_update_response(update_answer, event.operator_openid)
            return

        logger.debug("[%s] Unrecognised button_data %r from interaction %s", self._log_tag, button_data, event.id)

    @staticmethod
    def _write_update_response(answer: str, operator: str = "") -> None:
        """Atomically (tmp + rename) write the update-prompt answer to
        ``.update_response``, polled by the detached ``hermes update --gateway`` watcher."""
        try:
            from hermes_constants import get_hermes_home
            response_path = get_hermes_home() / ".update_response"
            tmp = response_path.with_suffix(".tmp")
            tmp.write_text(answer, encoding="utf-8")
            tmp.replace(response_path)
            logger.info("QQ update prompt answered %r by %s", answer, operator or "(unknown)")
        except Exception as exc:
            logger.error("Failed to write update response: %s", exc)

    async def _handle_c2c_message(self, d, msg_id, content, author, timestamp) -> None:
        user_openid = str(author.get("user_openid", ""))
        if not user_openid or not self._is_dm_intake_allowed(user_openid):
            return

        attachments_raw = d.get("attachments")
        logger.info(
            "[%s] C2C message: id=%s content=%r attachments=%s",
            self._log_tag, msg_id, content[:50] if content else "",
            f"{len(attachments_raw) if isinstance(attachments_raw, list) else 0} items" if attachments_raw else "None",
        )
        if attachments_raw and isinstance(attachments_raw, list):
            for _i, _att in enumerate(attachments_raw):
                if isinstance(_att, dict):
                    logger.info(
                        "[%s] attachment[%d]: content_type=%s url=%s filename=%s",
                        self._log_tag, _i, _att.get("content_type", ""),
                        str(_att.get("url", ""))[:80], _att.get("filename", ""))

        await self._ingest(
            d, msg_id, content, attachments_raw, timestamp, verbose=True,
            chat_id=user_openid, qq_chat_type="c2c", user_id=user_openid, chat_type="dm")

    async def _handle_group_message(self, d, msg_id, content, author, timestamp) -> None:
        group_openid = str(d.get("group_openid", ""))
        member = str(author.get("member_openid", ""))
        if not group_openid or not self._is_group_allowed(group_openid, member):
            return
        await self._ingest(
            d, msg_id, self._strip_at_mention(content), d.get("attachments"), timestamp,
            chat_id=group_openid, qq_chat_type="group", user_id=member, chat_type="group")

    async def _handle_guild_message(self, d, msg_id, content, author, timestamp) -> None:
        channel_id = str(d.get("channel_id", ""))
        if not channel_id:
            return
        # group_policy ACL — guild channels are group-like; without it any guild
        # member could bypass the allowlist.
        guild_id = str(d.get("guild_id", ""))
        author_id = str(author.get("id", ""))
        if not self._is_group_allowed(guild_id or channel_id, author_id):
            logger.debug("[%s] Guild message blocked by ACL: channel=%s user=%s", self._log_tag, channel_id, author_id)
            return

        member = d.get("member") if isinstance(d.get("member"), dict) else {}
        nick = str(member.get("nick", "")) or str(author.get("username", ""))
        await self._ingest(
            d, msg_id, content, d.get("attachments"), timestamp,
            chat_id=channel_id, qq_chat_type="guild", user_id=author_id, user_name=nick or None, chat_type="group")

    async def _handle_dm_message(self, d, msg_id, content, author, timestamp) -> None:
        guild_id = str(d.get("guild_id", ""))
        if not guild_id:
            return
        # dm_policy ACL — without it any guild member could bypass the allowlist via DM.
        author_id = str(author.get("id", ""))
        if not self._is_dm_intake_allowed(author_id):
            logger.debug("[%s] Guild DM blocked by ACL: guild=%s user=%s", self._log_tag, guild_id, author_id)
            return
        await self._ingest(
            d, msg_id, content, d.get("attachments"), timestamp,
            chat_id=guild_id, qq_chat_type="dm", user_id=author_id, chat_type="dm")

    _INBOUND_HANDLERS = {
        "C2C_MESSAGE_CREATE": "_handle_c2c_message",
        "GROUP_AT_MESSAGE_CREATE": "_handle_group_message",
        "GUILD_MESSAGE_CREATE": "_handle_guild_message",
        "GUILD_AT_MESSAGE_CREATE": "_handle_guild_message",
        "DIRECT_MESSAGE_CREATE": "_handle_dm_message"}

    # ── Shared inbound pipeline (all four message kinds) ──

    @staticmethod
    def _append_block(text: str, block: str) -> str:
        """Append *block* to *text* after a blank line (or return block alone if text is blank)."""
        return (text + "\n\n" + block).strip() if text.strip() else block

    async def _ingest(
        self, d: dict[str, Any], msg_id: str, content: str, attachments: Any, timestamp: str, *,
        chat_id: str, qq_chat_type: str, verbose: bool = False, **source_kwargs: Any) -> None:
        """Shared inbound tail: fold attachment transcripts/file info and quoted context
        into the text, drop empty events, remember the QQ chat kind and dispatch."""
        att = await self._process_attachments(attachments)
        text = content
        voice_transcripts = att["voice_transcripts"]
        if voice_transcripts:
            text = self._append_block(text, "\n".join(voice_transcripts))
        if att["attachment_info"]:
            text = self._append_block(text, att["attachment_info"])
        image_urls, image_media_types = att["image_urls"], att["image_media_types"]
        if verbose:
            logger.info("[%s] After processing: images=%d, voice=%d", self._log_tag, len(image_urls), len(voice_transcripts))

        quoted = await self._process_quoted_context(d)
        text = self._merge_quote_into(text, quoted["quote_block"])
        if quoted["image_urls"]:
            image_urls = image_urls + quoted["image_urls"]
            image_media_types = image_media_types + quoted["image_media_types"]
        if not text.strip() and not image_urls:
            return

        self._chat_type_map[chat_id] = qq_chat_type
        event = MessageEvent(
            source=self.build_source(chat_id=chat_id,** source_kwargs), text=text,
            message_type=self._detect_message_type(image_urls, image_media_types), raw_message=d,
            message_id=msg_id, media_urls=image_urls, media_types=image_media_types,
            timestamp=self._parse_qq_timestamp(timestamp),
        )
        await self.handle_message(event)

    # ── Quoted-message handling ──

    async def _process_quoted_context(self, d: dict[str, Any]) -> dict[str, Any]:
        """Process the quoted message a user is replying to (``message_type == 103``;
        referenced content + attachments live in ``msg_elements``). Quoted attachments
        go through _process_attachments so quoted voice gets STT and quoted images are
        cached identically. Returns ``{"quote_block", "image_urls", "image_media_types"}``;
        quote_block is "" when nothing is quoted."""
        empty = {"quote_block": "", "image_urls": [], "image_media_types": []}
        try:
            is_quote = int(d.get("message_type", 0) or 0) == 103
        except (TypeError, ValueError):
            is_quote = False
        elements = d.get("msg_elements")
        if not is_quote or not isinstance(elements, list) or not elements:
            return empty

        elements = [e for e in elements if isinstance(e, dict)]
        quoted_text_parts = [t for t in (str(e.get("content", "")).strip() for e in elements) if t]
        all_attachments = [
            a for e in elements if isinstance(e.get("attachments"), list) for a in e["attachments"] if isinstance(a, dict)]
        att_result = await self._process_attachments(all_attachments)
        quoted_images = att_result.get("image_urls") or []

        lines: list[str] = [" ".join(quoted_text_parts)] if quoted_text_parts else []
        lines.extend(att_result.get("voice_transcripts") or [])
        if att_result.get("attachment_info"):
            lines.append(att_result["attachment_info"])
        if not lines and not quoted_images:
            return empty
        # Images-only quote still gets a marker so the LLM knows context was referenced.
        return {
            "quote_block": "[Quoted message]:\n" + "\n".join(lines) if lines else "[Quoted message]: (image)",
            "image_urls": quoted_images,
            "image_media_types": att_result.get("image_media_types") or []}

    @staticmethod
    def _merge_quote_into(text: str, quote_block: str) -> str:
        """Prepend ``quote_block`` to *text*, separated by a blank line."""
        if not quote_block:
            return text
        return f"{quote_block}\n\n{text}".strip() if text.strip() else quote_block

    # ── Attachment processing ──

    @staticmethod
    def _detect_message_type(media_urls: list, media_types: list):
        if not media_urls:
            return MessageType.TEXT
        if not media_types:
            return MessageType.PHOTO
        first_type = media_types[0].lower()
        if "audio" in first_type or "voice" in first_type or "silk" in first_type:
            return MessageType.VOICE
        if "video" in first_type:
            return MessageType.VIDEO
        if "image" in first_type or "photo" in first_type:
            return MessageType.PHOTO
        logger.debug("Unknown media content_type '%s', defaulting to TEXT", first_type)
        return MessageType.TEXT

    async def _process_attachments(self, attachments: Any) -> dict[str, Any]:
        """Process inbound attachments uniformly. Returns ``{"image_urls",
        "image_media_types", "voice_transcripts", "attachment_info"}`` (cached image
        paths + MIME types, "[Voice] ..." transcripts, text description of other files)."""
        image_urls: list[str] = []
        image_media_types: list[str] = []
        voice_transcripts: list[str] = []
        other_attachments: list[str] = []

        for att in attachments if isinstance(attachments, list) else ():
            if not isinstance(att, dict):
                continue
            ct = str(att.get("content_type", "")).strip().lower()
            url = str(att.get("url", "")).strip()
            filename = str(att.get("filename", ""))
            if not url:
                continue
            if url.startswith("//"):
                url = f"https:{url}"
            logger.debug(
                "[%s] Processing attachment: content_type=%s, url=%s, filename=%s",
                self._log_tag, ct, url[:80], filename)

            if self._is_voice_content_type(ct, filename):
                asr_refer, wav_url = (self._opt_str(att.get(k)) for k in ("asr_refer_text", "voice_wav_url"))
                transcript = await self._stt_voice_attachment(
                    url, ct, filename, asr_refer_text=asr_refer, voice_wav_url=wav_url)
                if transcript:
                    voice_transcripts.append(f"[Voice] {transcript}")
                    logger.debug("[%s] Voice transcript: %s", self._log_tag, transcript)
                else:
                    logger.warning("[%s] Voice STT failed for %s", self._log_tag, url[:60])
                    voice_transcripts.append("[Voice] [语音识别失败]")
                continue

            is_image = ct.startswith("image/")
            try:
                cached_path = await self._download_and_cache(url, ct, filename)
            except Exception as exc:
                logger.debug("[%s] Failed to cache %s: %s", self._log_tag, "image" if is_image else "attachment", exc)
                continue
            if not cached_path:
                continue
            if not is_image:
                label = "video" if ct.startswith("video/") else "file"
                other_attachments.append(f"[{label}: {filename or ct} ({cached_path})]")
            elif os.path.isfile(cached_path):
                image_urls.append(cached_path)
                image_media_types.append(ct or "image/jpeg")
            else:
                logger.warning("[%s] Cached image path does not exist: %s", self._log_tag, cached_path)

        return {
            "image_urls": image_urls,
            "image_media_types": image_media_types,
            "voice_transcripts": voice_transcripts,
            "attachment_info": "\n".join(other_attachments)}

    @staticmethod
    def _opt_str(value: Any) -> Optional[str]:
        return (value.strip() if isinstance(value, str) else "") or None

    async def _download_and_cache(self, url: str, content_type: str, original_name: str = "") -> Optional[str]:
        """Download a URL and cache it locally (``original_name`` falls back to the URL basename)."""
        from tools.url_safety import is_safe_url

        if not is_safe_url(url):
            raise ValueError(f"Blocked unsafe URL: {url[:80]}")
        if not self._http_client:
            return None
        try:
            resp = await self._http_client.get(url, timeout=30.0, headers=self._qq_media_headers())
            resp.raise_for_status()
            data = resp.content
        except Exception as exc:
            logger.debug("[%s] Download failed for %s: %s", self._log_tag, url[:80], exc)
            return None

        if content_type.startswith("image/"):
            # Historical qqbot mapping: trust mimetypes' guess (never the shared table), fall back to .jpg.
            ext = ext_for_mime(content_type, use_defaults=False, use_mimetypes=True, fallback=".jpg") or ".jpg"
            return await cache_image_from_bytes_async(data, ext)
        if content_type == "voice" or content_type.startswith("audio/"):
            # QQ voice is usually .amr/.silk — convert to .wav for STT engines.
            return await self._convert_audio_to_wav(data, url)
        filename = original_name or Path(urlparse(url).path).name or "qq_attachment"
        return await cache_document_from_bytes_async(data, filename)

    @staticmethod
    def _is_voice_content_type(content_type: str, filename: str) -> bool:
        ct = content_type.strip().lower()
        if ct == "voice" or ct.startswith("audio/"):
            return True
        # content_type="file" is an explicit upload: never route .wav/.mp3 files into STT.
        if ct == "file":
            return False
        return filename.strip().lower().endswith(_VOICE_EXTENSIONS)

    def _qq_media_headers(self) -> dict[str, str]:
        """Authorization and Hermes user agent for QQ multimedia CDN downloads."""
        from qqbot_agent_sdk import build_user_agent

        from .sdk_bridge import apply_sdk_identity

        token = getattr(self._qq_api, "access_token", None) if self._qq_api is not None else None
        if not token:
            return {}
        apply_sdk_identity()
        return {"Authorization": f"QQBot {token}", "User-Agent": build_user_agent()}

    async def _stt_voice_attachment(
        self, url: str, content_type: str, filename: str, *, asr_refer_text: Optional[str] = None,
        voice_wav_url: Optional[str] = None) -> Optional[str]:
        """Transcribe a voice attachment. Priority: QQ's free ``asr_refer_text`` →
        STT on ``voice_wav_url`` (pre-converted WAV, no SILK decode) → STT on the
        original URL (SILK→WAV). Returns the transcript or None."""
        if asr_refer_text:
            logger.debug("[%s] STT: using QQ asr_refer_text: %r", self._log_tag, asr_refer_text[:100])
            return asr_refer_text

        is_pre_wav = bool(voice_wav_url)
        download_url = url
        if is_pre_wav:
            download_url = f"https:{voice_wav_url}" if voice_wav_url.startswith("//") else voice_wav_url
            logger.debug("[%s] STT: using voice_wav_url (pre-converted WAV)", self._log_tag)

        from tools.url_safety import is_safe_url
        if not is_safe_url(download_url):
            logger.warning("[QQ] STT blocked unsafe URL: %s", download_url[:80])
            return None

        try:
            if not self._http_client:
                logger.warning("[%s] STT: no HTTP client", self._log_tag)
                return None
            download_headers = self._qq_media_headers()  # QQ CDN requires Authorization
            logger.debug(
                "[%s] STT: downloading voice from %s (pre_wav=%s, headers=%s)",
                self._log_tag, download_url[:80], is_pre_wav, bool(download_headers))
            resp = await self._http_client.get(
                download_url, timeout=30.0, headers=download_headers, follow_redirects=True)
            resp.raise_for_status()
            audio_data = resp.content
            logger.debug(
                "[%s] STT: downloaded %d bytes, content_type=%s",
                self._log_tag, len(audio_data), resp.headers.get("content-type", "unknown"))
            if len(audio_data) < 10:
                logger.warning("[%s] STT: downloaded data too small (%d bytes), skipping", self._log_tag, len(audio_data))
                return None

            if is_pre_wav:
                wav_path = self._write_temp(audio_data, ".wav")
                logger.debug("[%s] STT: using pre-converted WAV directly (%d bytes)", self._log_tag, len(audio_data))
            else:
                logger.debug("[%s] STT: converting to wav, filename=%r", self._log_tag, filename)
                wav_path = await self._convert_audio_to_wav_file(audio_data, filename)
                if not wav_path or not Path(wav_path).exists():
                    logger.warning("[%s] STT: ffmpeg conversion produced no output", self._log_tag)
                    return None

            logger.debug("[%s] STT: calling ASR on %s", self._log_tag, wav_path)
            try:
                transcript = await self._call_stt(wav_path)
            finally:
                self._unlink_quiet(wav_path)
            if transcript:
                logger.debug("[%s] STT success: %r", self._log_tag, transcript[:100])
            else:
                logger.warning("[%s] STT: ASR returned empty transcript", self._log_tag)
            return transcript
        except (OSError, httpx.HTTPStatusError, httpx.TransportError) as exc:
            logger.warning("[%s] STT failed for voice attachment: %s: %s", self._log_tag, type(exc).__name__, exc)
            return None

    @staticmethod
    def _write_temp(data: bytes, suffix: str) -> str:
        """Write *data* to a persistent NamedTemporaryFile and return its path."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            return tmp.name

    @staticmethod
    def _unlink_quiet(path: str) -> None:
        with contextlib.suppress(OSError):
            os.unlink(path)

    @staticmethod
    def _wav_ok(wav_path: str) -> bool:
        """True when *wav_path* exists and holds more than a bare 44-byte header."""
        return Path(wav_path).exists() and Path(wav_path).stat().st_size > 44

    @classmethod
    def _temp_pair(cls, audio_data: bytes, ext: str) -> tuple[str, str]:
        """Write *audio_data* to a temp ``<x>{ext}`` and return ``(src_path, sibling .wav path)``."""
        src_path = cls._write_temp(audio_data, ext)
        return src_path, src_path.rsplit(".", 1)[0] + ".wav"

    async def _convert_audio_to_wav_file(self, audio_data: bytes, filename: str) -> Optional[str]:
        """Convert audio bytes to a temp .wav: pilk (SILK, which ffmpeg can't decode)
        → ffmpeg → raw-PCM last resort. Returns the wav path or None."""
        ext = Path(filename).suffix.lower() or self._guess_ext_from_data(audio_data)
        logger.info(
            "[%s] STT: audio_data size=%d, ext=%r, first_20_bytes=%r",
            self._log_tag, len(audio_data), ext, audio_data[:20])
        src_path, wav_path = self._temp_pair(audio_data, ext)
        result = (
            await self._convert_silk_to_wav(src_path, wav_path)
            or await self._convert_ffmpeg_to_wav(src_path, wav_path)
            or await self._convert_raw_to_wav(audio_data, wav_path))
        self._unlink_quiet(src_path)
        return result

    _MAGIC_EXTS = (
        (b"#!SILK", ".silk"), (b"\x02!", ".silk"), (b"RIFF", ".wav"), (b"fLaC", ".flac"),
        (b"\xff\xfb", ".mp3"), (b"\xff\xf3", ".mp3"), (b"\xff\xf2", ".mp3"),
        (b"\x30\x26\xb2\x75", ".ogg"), (b"\x4f\x67\x67\x53", ".ogg"))

    @classmethod
    def _guess_ext_from_data(cls, data: bytes) -> str:
        """Guess file extension from magic bytes (unknown → .amr, QQ's most common)."""
        return next((ext for magic, ext in cls._MAGIC_EXTS if data.startswith(magic)), ".amr")

    @classmethod
    def _looks_like_silk(cls, data: bytes) -> bool:
        return cls._guess_ext_from_data(data) == ".silk"

    async def _convert_silk_to_wav(self, src_path: str, wav_path: str) -> Optional[str]:
        """Convert to WAV with pilk: as-is first, then copied to .silk (pilk checks the extension)."""
        try:
            import pilk
        except ImportError:
            logger.warning("[%s] pilk not installed — cannot decode SILK audio. Run: "
                           f"{install_hint('silk')}", self._log_tag)
            return None

        silk_path = src_path.rsplit(".", 1)[0] + ".silk"
        try:
            for path, label, how in ((src_path, "", "direct"), (silk_path, " (as .silk)", ".silk")):
                try:
                    if path == silk_path:
                        import shutil

                        shutil.copy2(src_path, silk_path)
                    pilk.silk_to_wav(path, wav_path, rate=16000)
                    if self._wav_ok(wav_path):
                        logger.debug(
                            "[%s] pilk converted %s%s to wav (%d bytes)",
                            self._log_tag, Path(src_path).name, label, Path(wav_path).stat().st_size)
                        return wav_path
                except Exception as exc:
                    logger.debug("[%s] pilk %s conversion failed: %s", self._log_tag, how, exc)
        finally:
            self._unlink_quiet(silk_path)
        return None

    async def _convert_raw_to_wav(self, audio_data: bytes, wav_path: str) -> Optional[str]:
        """Last resort: wrap bytes as raw PCM 16-bit mono 16kHz WAV (garbage if not
        PCM, but the ASR engine returns empty instead of crashing)."""
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
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path,
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            await asyncio.wait_for(proc.wait(), timeout=30)
            if proc.returncode != 0:
                stderr = await proc.stderr.read() if proc.stderr else b""
                logger.warning(
                    "[%s] ffmpeg failed for %s: %s",
                    self._log_tag, Path(src_path).name, stderr[:200].decode(errors="replace"))
                return None
        except (TimeoutError, FileNotFoundError) as exc:
            logger.warning("[%s] ffmpeg conversion error: %s", self._log_tag, exc)
            return None

        if not self._wav_ok(wav_path):
            logger.warning("[%s] ffmpeg produced no/small output for %s", self._log_tag, Path(src_path).name)
            return None
        logger.debug(
            "[%s] ffmpeg converted %s to wav (%d bytes)",
            self._log_tag, Path(src_path).name, Path(wav_path).stat().st_size)
        return wav_path

    def _resolve_stt_config(self) -> Optional[dict[str, Any]]:
        """Resolve STT backend: ``extra["stt"]`` config first, then ``QQ_STT_*`` env
        vars; None when unconfigured (QQ's built-in ASR still works). ``timeout`` (seconds,
        default 60) follows the shared STT client default so a self-hosted model's cold start
        is not cut off at 30s (#112939)."""
        from tools.transcription_common import DEFAULT_STT_TIMEOUT, _config_number  # lazy: keep adapter light
        stt_cfg = (self.config.extra or {}).get("stt")
        if isinstance(stt_cfg, dict) and stt_cfg.get("enabled") is not False:
            base_url = stt_cfg.get("baseUrl") or stt_cfg.get("base_url", "")
            api_key = stt_cfg.get("apiKey") or stt_cfg.get("api_key", "")
            model = stt_cfg.get("model", "")
            timeout = _config_number(stt_cfg, "timeout", DEFAULT_STT_TIMEOUT)
            if base_url and api_key:
                return {"base_url": base_url.rstrip("/"), "api_key": api_key, "model": model or "whisper-1",
                        "timeout": timeout}
            if api_key:  # provider-only config
                provider = stt_cfg.get("provider", "zai")
                base_url = _STT_PROVIDER_BASE_URLS.get(provider, "")
                if base_url:
                    default_model = "glm-asr" if provider in {"zai", "glm"} else "whisper-1"
                    return {"base_url": base_url, "api_key": api_key, "model": model or default_model,
                            "timeout": timeout}

        qq_stt_key = _resolve_qq_secret("QQ_STT_API_KEY", "")
        if qq_stt_key:
            base_url = _resolve_qq_secret("QQ_STT_BASE_URL", _STT_PROVIDER_BASE_URLS["zai"])
            model = _resolve_qq_secret("QQ_STT_MODEL", "glm-asr")
            return {"base_url": base_url.rstrip("/"), "api_key": qq_stt_key, "model": model,
                    "timeout": DEFAULT_STT_TIMEOUT}
        return None

    async def _call_stt(self, wav_path: str) -> Optional[str]:
        """Transcribe a wav via an OpenAI-compatible STT API; None if unconfigured/failed."""
        stt_cfg = self._resolve_stt_config()
        if not stt_cfg:
            logger.warning("[%s] STT not configured (no stt config or QQ_STT_API_KEY)", self._log_tag)
            return None

        base_url, api_key, model = stt_cfg["base_url"], stt_cfg["api_key"], stt_cfg["model"]
        try:
            with open(wav_path, "rb") as f:  # noqa: ASYNC230 -- file handle is streamed to the upload; a local open() is non-blocking in practice
                resp = await self._http_client.post(
                    f"{base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (Path(wav_path).name, f, "audio/wav")},
                    data={"model": model},
                    timeout=stt_cfg["timeout"])
            resp.raise_for_status()
            result = resp.json()
            # Zhipu/GLM: {"choices": [{"message": {"content": ...}}]}; OpenAI/Whisper: {"text": ...}
            choices = result.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            return content.strip() or result.get("text", "").strip() or None
        except (OSError, httpx.HTTPStatusError) as exc:
            logger.warning("[%s] STT API call failed (model=%s, base=%s): %s", self._log_tag, model, base_url[:50], exc)
            return None

    async def _convert_audio_to_wav(self, audio_data: bytes, source_url: str) -> Optional[str]:
        """Convert audio bytes to .wav (pilk for SILK, else ffmpeg) and cache the result;
        on conversion failure the original bytes are cached as ``qq_voice<ext>``."""
        ext = Path(urlparse(source_url).path).suffix.lower()
        if ext not in _AUDIO_URL_EXTENSIONS:
            ext = self._guess_ext_from_data(audio_data)
        src_path, wav_path = self._temp_pair(audio_data, ext)
        is_silk = ext == ".silk" or self._looks_like_silk(audio_data)
        convert = self._convert_silk_to_wav if is_silk else self._convert_ffmpeg_to_wav
        try:
            if not await convert(src_path, wav_path):
                logger.warning("[%s] audio conversion failed for %s (format=%s)", self._log_tag, source_url[:60], ext)
                return await cache_document_from_bytes_async(audio_data, f"qq_voice{ext}")
        except Exception:
            return await cache_document_from_bytes_async(audio_data, f"qq_voice{ext}")
        finally:
            self._unlink_quiet(src_path)

        try:
            wav_data = Path(wav_path).read_bytes()
            os.unlink(wav_path)
            return await cache_document_from_bytes_async(wav_data, "qq_voice.wav")
        except Exception as exc:
            logger.debug("[%s] Failed to read converted wav: %s", self._log_tag, exc)
            return None

    # ── Outbound messaging — REST API ──

    def _require_api(self):
        """The connected ``QQApiClient``. Every QQ REST call goes through it."""
        if self._qq_api is None:
            raise RuntimeError("QQ API client not initialized — not connected?")
        return self._qq_api

    def _sent(self, data: Any) -> SendResult:
        payload = data if isinstance(data, dict) else {}
        message_id = str(payload.get("id") or uuid.uuid4().hex[:12])
        return SendResult(success=True, message_id=message_id, raw_response=payload)

    _RECONNECT_WAIT_SECONDS = 15.0  # max wait for reconnection before giving up on send
    _RECONNECT_POLL_INTERVAL = 0.5  # is_connected poll interval while waiting

    async def _wait_for_reconnection(self) -> bool:
        """Poll is_connected for up to _RECONNECT_WAIT_SECONDS — covers the race where
        send() lands between a disconnect and the SDK socket reconnect."""
        logger.info(
            "[%s] Not connected — waiting for reconnection (up to %.0fs)", self._log_tag, self._RECONNECT_WAIT_SECONDS
        )
        waited = 0.0
        while waited < self._RECONNECT_WAIT_SECONDS:
            await asyncio.sleep(self._RECONNECT_POLL_INTERVAL)
            waited += self._RECONNECT_POLL_INTERVAL
            if self.is_connected:
                logger.info("[%s] Reconnected after %.1fs", self._log_tag, waited)
                return True
        logger.warning("[%s] Still not connected after %.0fs", self._log_tag, self._RECONNECT_WAIT_SECONDS)
        return False

    @property
    def _NOT_CONNECTED(self) -> SendResult:
        return SendResult(success=False, error="Not connected", retryable=True)

    async def _ensure_connected(self) -> bool:
        """True when connected now or after waiting for the listener to reconnect."""
        return self.is_connected or await self._wait_for_reconnection()

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send text/markdown: format, split via truncate_message(), retry transient failures."""
        del metadata
        if not await self._ensure_connected():
            return self._NOT_CONNECTED
        if not content or not content.strip():
            return SendResult(success=True)

        chunks = self.truncate_message(self.format_message(content), self.MAX_MESSAGE_LENGTH)
        last_result = SendResult(success=False, error="No chunks")
        for chunk in chunks:
            last_result = await self._send_chunk(chat_id, chunk, reply_to)
            if not last_result.success:
                return last_result
            reply_to = None  # only reply_to the first chunk
        return last_result

    _PERMANENT_SEND_ERRORS = ("invalid", "forbidden", "not found")

    async def _send_chunk(self, chat_id: str, content: str, reply_to: Optional[str] = None) -> SendResult:
        last_exc: Optional[Exception] = None
        sender = self._text_sender(self._guess_chat_type(chat_id))
        if sender is None:
            return SendResult(success=False, error=f"Unknown chat type for {chat_id}")
        for attempt in range(3):
            try:
                return await sender(chat_id, content, reply_to)
            except Exception as exc:
                last_exc = exc
                if any(k in str(exc).lower() for k in self._PERMANENT_SEND_ERRORS + ("bad request",)):
                    break  # permanent — don't retry
                if attempt < 2:
                    delay = 1.0 * (2 ** attempt)
                    logger.warning("[%s] send retry %d/3 after %.1fs: %s", self._log_tag, attempt + 1, delay, exc)
                    await asyncio.sleep(delay)

        error_msg = (str(last_exc) or type(last_exc).__name__) if last_exc else "Unknown error"
        logger.error("[%s] Send failed: %s", self._log_tag, error_msg)
        retryable = not any(k in error_msg.lower() for k in self._PERMANENT_SEND_ERRORS)
        return SendResult(success=False, error=error_msg, retryable=retryable)

    async def _send_text_to(
        self, chat_type: str, target_id: str, content: str, reply_to: Optional[str] = None,
        keyboard: Optional[InlineKeyboard] = None) -> SendResult:
        """Send text (optionally with an inline keyboard) through ``QQApiClient``."""
        api = self._require_api()
        if keyboard is None:
            data = await api.send_text(
                chat_type, target_id, content, reply_to=reply_to,
                markdown=self._markdown_support, retries=1, max_length=self.MAX_MESSAGE_LENGTH)
            return self._sent(data)
        msg = api.build_text_body(
            content, reply_to=reply_to, markdown=self._markdown_support,
            max_length=self.MAX_MESSAGE_LENGTH)
        if chat_type == "group":
            data = await api.post_group_message(target_id, msg, keyboard=keyboard)
        else:
            data = await api.post_c2c_message(target_id, msg, keyboard=keyboard)
        return self._sent(data)

    async def _send_c2c_text(self, openid, content, reply_to=None, keyboard=None) -> SendResult:
        return await self._send_text_to("c2c", openid, content, reply_to, keyboard)

    async def _send_group_text(self, group_openid, content, reply_to=None, keyboard=None) -> SendResult:
        return await self._send_text_to("group", group_openid, content, reply_to, keyboard)

    _TEXT_SENDERS = {"c2c": "_send_c2c_text", "group": "_send_group_text", "guild": "_send_guild_text"}

    def _text_sender(self, chat_type: str, *, keyboard_ok: bool = False):
        """Bound text sender for *chat_type* (None if unsupported); guild lacks keyboards."""
        if keyboard_ok and chat_type == "guild":
            return None
        name = self._TEXT_SENDERS.get(chat_type)
        return getattr(self, name) if name else None

    async def _send_guild_text(self, channel_id: str, content: str, reply_to: Optional[str] = None) -> SendResult:
        return await self._send_text_to("guild", channel_id, content, reply_to)

    # ── Inline-keyboard outbound helpers (approval / update-prompt flows) ──

    async def send_with_keyboard(
        self, chat_id: str, content: str, keyboard: InlineKeyboard, reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send ONE text message with an inline keyboard (no chunking — splitting
        would orphan the buttons; keep bodies short). Guild chats are unsupported."""
        if not await self._ensure_connected():
            return self._NOT_CONNECTED
        chat_type = self._guess_chat_type(chat_id)
        sender = self._text_sender(chat_type, keyboard_ok=True)
        if sender is None:
            return SendResult(
                success=False, error=f"Inline keyboards not supported for chat_type {chat_type!r}", retryable=False)
        truncated = self.format_message(content)[: self.MAX_MESSAGE_LENGTH]
        try:
            return await sender(chat_id, truncated, reply_to, keyboard=keyboard)
        except Exception as exc:
            logger.error("[%s] send_with_keyboard failed: %s", self._log_tag, exc)
            return SendResult(success=False, error=str(exc) or type(exc).__name__)

    async def send_approval_request(
        self, chat_id: str, req: ApprovalRequest, reply_to: Optional[str] = None) -> SendResult:
        """Send a 3-button approval request (allow-once / allow-always / deny);
        clicks come back as INTERACTION_CREATE decoded by parse_approval_button_data."""
        from .keyboards import build_approval_text
        keyboard = build_approval_keyboard(req.session_key, allow_permanent=getattr(req, "allow_permanent", True))
        return await self.send_with_keyboard(chat_id, build_approval_text(req), keyboard, reply_to=reply_to)

    # Cross-adapter gateway contract: gateway/run.py detects send_exec_approval /
    # send_update_prompt on the adapter class for button-based approval/update UX.

    _APPROVAL_TIMEOUT_SECONDS = 300  # matches gateway's default gateway_timeout

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        """Keyboard-card approval (called while the agent blocks on approval); clicks resolve via
        _default_interaction_dispatch. QQ's 3-button keyboard has no session tier and no thread /
        DM targeting, so only the ``always`` choice and the raw command/reason are used."""
        description = prompt.description
        if prompt.smart_denied:
            description += " Owner override applies to this one operation only."
        req = ApprovalRequest(
            session_key=prompt.session_key, title="Execute this command?", description=description,
            command_preview=prompt.command, timeout_sec=self._APPROVAL_TIMEOUT_SECONDS,
            allow_permanent="always" in prompt.choices)
        # QQ requires a msg_id for passive replies; the last inbound id is the natural one.
        return await self.send_approval_request(
            prompt.chat_id, req, reply_to=self._last_msg_id.get(prompt.chat_id))

    async def send_update_prompt(
        self, chat_id: str, prompt: str, default: str = "", session_key: str = "",
        metadata: Optional[dict[str, Any]] = None) -> SendResult:
        """Yes/No update-confirmation prompt; button clicks (``update_prompt:y|n``)
        are written to ``~/.hermes/.update_response`` by the interaction callback."""
        del session_key, metadata  # present for contract parity only.
        default_hint = f" (default: {default})" if default else ""
        content = f"☤ **Update Needs Your Input**\n\n{prompt}{default_hint}"
        return await self.send_with_keyboard(
            chat_id, content, build_update_prompt_keyboard(), reply_to=self._last_msg_id.get(chat_id)
        )

    # ── Native media sending ──

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None) -> SendResult:
        """Send an image natively via QQ Bot API upload; URL sources fall back to text."""
        del metadata
        result = await self._send_media(chat_id, image_url, MEDIA_TYPE_IMAGE, "image", caption, reply_to)
        if result.success or not self._is_url(image_url):
            return result
        logger.warning("[%s] Image send failed, falling back to text: %s", self._log_tag, result.error)
        fallback = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id=chat_id, content=fallback, reply_to=reply_to)

    async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None, **kwargs) -> SendResult:
        return await self._send_media(chat_id, image_path, MEDIA_TYPE_IMAGE, "image", caption, reply_to)

    async def send_voice(self, chat_id, audio_path, caption=None, reply_to=None, **kwargs) -> SendResult:
        return await self._send_media(chat_id, audio_path, MEDIA_TYPE_VOICE, "voice", caption, reply_to)

    async def send_video(self, chat_id, video_path, caption=None, reply_to=None, **kwargs) -> SendResult:
        return await self._send_media(chat_id, video_path, MEDIA_TYPE_VIDEO, "video", caption, reply_to)

    async def send_document(
        self, chat_id, file_path, caption=None, file_name=None, reply_to=None, **kwargs
    ) -> SendResult:
        return await self._send_media(
            chat_id, file_path, MEDIA_TYPE_FILE, "file", caption, reply_to, file_name=file_name)

    async def _send_media(
        self, chat_id: str, media_source: str, file_type: int, kind: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, file_name: Optional[str] = None) -> SendResult:
        """Upload via ``MediaUploader`` and send the rich-media message through the SDK."""
        from qqbot_agent_sdk import (
            MediaInfo, MediaUploader, MessageToCreate, QQApiClient, QQMessageType,
            UploadDailyLimitExceededError, UploadFileTooLargeError)

        if not await self._ensure_connected():
            return self._NOT_CONNECTED
        chat_type = self._guess_chat_type(chat_id)
        if chat_type == "guild":
            return SendResult(success=False, error="Guild media send not supported via this path")

        try:
            api = self._require_api()
            resolved_name = file_name or Path(urlparse(str(media_source)).path).name or "media"
            file_info = await MediaUploader(
                api, self._http_client, log_tag=self._log_tag,
            ).upload(chat_type, chat_id, media_source, file_type, resolved_name)
            msg = MessageToCreate(
                msg_type=int(QQMessageType.RICH_MEDIA),
                msg_seq=QQApiClient.next_msg_seq(chat_id),
                media=MediaInfo(file_info=file_info),
                content=(caption or "")[: self.MAX_MESSAGE_LENGTH],
                msg_id=reply_to or "",
            )
            poster = api.post_group_message if chat_type == "group" else api.post_c2c_message
            return self._sent(await poster(chat_id, msg))
        except UploadDailyLimitExceededError as exc:
            # Non-retryable quota hit; give the model actionable text.
            logger.warning("[%s] Daily upload limit exceeded for %s (%s)", self._log_tag, exc.file_name, exc.file_size_human)
            return SendResult(
                success=False, retryable=False,
                error=f"QQ daily upload limit exceeded for {exc.file_name!r} ({exc.file_size_human}). Retry tomorrow.")
        except UploadFileTooLargeError as exc:
            logger.warning(
                "[%s] File too large: %s (%s, platform limit %s)",
                self._log_tag, exc.file_name, exc.file_size_human, exc.limit_human)
            return SendResult(
                success=False, retryable=False,
                error=f"{exc.file_name!r} ({exc.file_size_human}) exceeds the QQ per-file upload limit ({exc.limit_human}).")
        except Exception as exc:
            logger.error("[%s] Media send (%s) failed: %s", self._log_tag, kind, exc)
            return SendResult(success=False, error=str(exc) or type(exc).__name__)

    # ── Typing indicator ──

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """C2C-only input notify, debounced to ~50s (API shows a 60s indicator);
        needs the last inbound msg_id from ``_last_msg_id``."""
        msg_id = self._last_msg_id.get(chat_id)
        now = time.time()
        if (
            not self.is_connected or self._guess_chat_type(chat_id) != "c2c" or not msg_id
            or now - self._typing_sent_at.get(chat_id, 0.0) < self._TYPING_DEBOUNCE_SECONDS
        ):
            return
        try:
            await self._require_api().send_typing(
                chat_id, msg_id, input_seconds=self._TYPING_INPUT_SECONDS)
            self._typing_sent_at[chat_id] = now
        except Exception as exc:
            logger.debug("[%s] send_typing failed: %s", self._log_tag, exc)

    # ── Format / chat info / helpers ──

    def format_message(self, content: str) -> str:
        """Pass markdown through when supported, else strip it (as BlueBubbles/SMS do)."""
        return content if self._markdown_support else strip_markdown(content)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        chat_type = self._guess_chat_type(chat_id)
        return {"name": chat_id, "type": "group" if chat_type in {"group", "guild"} else "dm"}

    @staticmethod
    def _is_url(source: str) -> bool:
        return urlparse(str(source)).scheme in {"http", "https"}

    def _guess_chat_type(self, chat_id: str) -> str:
        """Determine chat type from stored inbound metadata, fallback to 'c2c'."""
        return self._chat_type_map.get(chat_id, "c2c")

    @staticmethod
    def _strip_at_mention(content: str) -> str:
        return re.sub(r"^@\S+\s*", "", content.strip())

    def _entry_matches(self, entries: list[str], target: str) -> bool:
        normalized_target = str(target).strip().lower()
        return any(str(e).strip().lower() in ("*", normalized_target) for e in entries)

    def _parse_qq_timestamp(self, raw: str) -> datetime:
        """Parse a QQ timestamp — ISO 8601 string (current) or integer ms (legacy)."""
        if raw:
            with contextlib.suppress(ValueError, TypeError):
                return datetime.fromisoformat(raw)
            with contextlib.suppress(ValueError, TypeError):
                return datetime.fromtimestamp(int(raw) / 1000, tz=UTC)
        return datetime.now(tz=UTC)
