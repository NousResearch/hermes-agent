# -*- coding: utf-8 -*-
"""QQ Bot platform adapter for hermes-agent.

Thin hermes-specific layer on top of the ``core/`` SDK.  All QQ Bot protocol
logic lives in ``core/``; this file is responsible only for:

1. Reading hermes config and constructing core components with injected deps.
2. Implementing the :class:`~gateway.platforms.base.BasePlatformAdapter`
   interface (``connect``, ``disconnect``, ``send``, ``send_image``, …).
3. Converting :class:`~core.event_parser.InboundEvent` →
   :class:`~gateway.platforms.base.MessageEvent` (≈ 30 lines).
4. Wrapping core results in :class:`~gateway.platforms.base.SendResult`.

Configuration in config.yaml::

    platforms:
      qq:
        enabled: true
        extra:
          app_id: "your-app-id"            # or QQ_APP_ID env var
          client_secret: "your-secret"     # or QQ_CLIENT_SECRET env var
          markdown_support: true
          dm_policy: "open"                # open | allowlist | disabled
          allow_from: ["openid_1"]
          group_policy: "open"
          group_allow_from: ["group_openid_1"]
          stt:
            provider: "zai"
            baseUrl: "https://open.bigmodel.cn/api/coding/paas/v4"
            apiKey: "your-stt-api-key"
            model: "glm-asr"

Reference: https://bot.q.qq.com/wiki/develop/api-v2/
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]

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
    _ssrf_redirect_guard,
    cache_image_from_url,
)
from gateway.platforms.helpers import strip_markdown

from .core.api_client import QQApiClient
from .core.approval import (
    ApprovalRequest,
    ApprovalSender,
    build_update_prompt_keyboard,
    parse_approval_button_data,
    parse_update_prompt_button_data,
)
from .core.attachment_processor import (
    AttachmentDownloader,
    AttachmentProcessor,
    ProcessedAttachment,
    STTPipeline,
)
from .core.audio import resolve_stt_config
from .core.constants import (
    MAX_MESSAGE_LENGTH,
    MEDIA_TYPE_FILE,
    MEDIA_TYPE_IMAGE,
    MEDIA_TYPE_VIDEO,
    MEDIA_TYPE_VOICE,
)
from .core.dto import (
    GuildMessageToCreate,
    InputNotify,
    MessageToCreate,
    QQMessageType,
    parse_interaction_event,
)
from .core.event_parser import EventParser, InboundEvent
from .core.media_loader import (
    MediaLoader,
    MediaUploader,
    UploadDailyLimitExceededError,
    UploadFileTooLargeError,
)
from .core.utils import coerce_list
from .core.websocket import QQCloseError, QQWebSocket, WSCallbacks

logger = logging.getLogger(__name__)


# ── Dependency checks ─────────────────────────────────────────────────

def check_qq_requirements() -> bool:
    """Return True if all required runtime packages are available."""
    return AIOHTTP_AVAILABLE and HTTPX_AVAILABLE


# ── QQAdapter ─────────────────────────────────────────────────────────

class QQAdapter(BasePlatformAdapter):
    """hermes QQ Bot adapter backed by the official QQ Bot WebSocket Gateway.

    Delegates all QQ protocol logic to ``core/``.  This class is the
    sole file that may import from ``gateway.platforms.base`` and other
    hermes modules.
    """

    SUPPORTS_MESSAGE_EDITING = False
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    _TYPING_INPUT_SECONDS = 60
    _TYPING_DEBOUNCE_SECONDS = 50
    _RECONNECT_WAIT_SECONDS = 15.0
    _RECONNECT_POLL_INTERVAL = 0.5

    @property
    def _log_tag(self) -> str:
        app_id = getattr(self, "_app_id", None)
        return f"QQBot:{app_id}" if app_id else "QQBot"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config, Platform.QQBOT)

        extra = config.extra or {}
        self._app_id = str(extra.get("app_id") or os.getenv("QQ_APP_ID", "")).strip()
        self._client_secret = str(
            extra.get("client_secret") or os.getenv("QQ_CLIENT_SECRET", "")
        ).strip()
        self._markdown_support = bool(extra.get("markdown_support", True))

        # ACL policies
        self._dm_policy = str(extra.get("dm_policy", "open")).strip().lower()
        self._allow_from: List[str] = coerce_list(
            extra.get("allow_from") or extra.get("allowFrom")
        )
        self._group_policy = str(extra.get("group_policy", "open")).strip().lower()
        self._group_allow_from: List[str] = coerce_list(
            extra.get("group_allow_from") or extra.get("groupAllowFrom")
        )

        # Connection state
        self._http_client: Optional[Any] = None
        self._session_id: Optional[str] = None
        self._last_seq: Optional[int] = None
        self._heartbeat_interval: float = 30.0
        self._session_dirty: bool = False
        self._chat_type_map: Dict[str, str] = {}
        self._last_msg_id: Dict[str, str] = {}
        self._typing_sent_at: Dict[str, float] = {}
        self._bot_username: str = ""

        # Core SDK components (HTTP client injected after connect())
        self._api = QQApiClient(
            app_id=self._app_id,
            client_secret=self._client_secret,
            log_tag=self._log_tag,
        )

        # Downloader + STT (http_client injected in connect())
        self._downloader = AttachmentDownloader(
            http_client=None,
            cache_dir=self._get_cache_dir(),
            media_headers_fn=self._media_headers,
            log_tag=self._log_tag,
        )
        self._stt = STTPipeline(
            http_client=None,
            stt_config_fn=lambda: resolve_stt_config(extra),
            downloader=self._downloader,
            log_tag=self._log_tag,
        )
        self._att_processor = AttachmentProcessor(
            downloader=self._downloader,
            stt_pipeline=self._stt,
        )

        self._event_parser = EventParser()
        self._media_uploader = MediaUploader(
            self._api, http_client=None, log_tag=self._log_tag
        )
        self._approval_sender = ApprovalSender(self._api, log_tag=self._log_tag)

        # Session persistence — load previous session for Resume
        from .core.session_store import WSSessionStore
        from .core.dto import DEFAULT_INTENTS
        self._ws_session_store = WSSessionStore(self._get_cache_dir())
        self._intents = int(DEFAULT_INTENTS)
        persisted = self._ws_session_store.get(self._app_id)
        if persisted.is_resumable and persisted.is_fresh() and persisted.intents == self._intents:
            self._session_id = persisted.session_id
            self._last_seq = persisted.seq
            self._bot_username = persisted.bot_username
            logger.info(
                "[%s] Loaded persisted session: session_id=%s seq=%s (age=%.0fs)",
                self._log_tag, persisted.session_id, persisted.seq, persisted.age_seconds,
            )
        else:
            if persisted.session_id:
                reason = (
                    "stale" if not persisted.is_fresh()
                    else "intents changed" if persisted.intents != self._intents
                    else "incomplete"
                )
                logger.info(
                    "[%s] Discarding persisted session (%s), will identify fresh",
                    self._log_tag, reason,
                )

        # WebSocket manager (injected callbacks — no adapter reference in core)
        self._ws_manager = QQWebSocket(
            callbacks=self._build_ws_callbacks(),
            log_tag=self._log_tag,
        )

    def _build_ws_callbacks(self) -> WSCallbacks:
        """Build the WSCallbacks dataclass wiring core ↔ adapter."""
        return WSCallbacks(
            on_message_event=self._on_core_message,
            on_connected=self._mark_connected,
            on_disconnected=self._mark_disconnected,
            on_fatal_error=self._set_fatal_error,
            get_token=self._api.ensure_token_sync,
            get_gateway_url=self._api.get_gateway_url_sync,
            get_session=lambda: (self._session_id, self._last_seq),
            set_session=self._set_session,
            set_heartbeat_interval=lambda v: setattr(self, "_heartbeat_interval", v),
            clear_token=self._api.clear_token,
            fail_pending=self._fail_pending,
            on_interaction_event=self._on_interaction,
            on_ready=self._on_ready,
            on_heartbeat_ack=self._flush_session,
        )

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "QQBot"

    def _on_ready(self, ready: Any) -> None:
        """Called by WSCallbacks after READY — capture bot identity."""
        if ready and ready.user and ready.user.username:
            self._bot_username = ready.user.username
            # Flush immediately so bot_username is persisted.
            self._session_dirty = True
            self._flush_session()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Authenticate, obtain gateway URL, and open the WebSocket."""
        if not AIOHTTP_AVAILABLE:
            return self._fail_startup("qq_missing_dependency", "aiohttp not installed")
        if not HTTPX_AVAILABLE:
            return self._fail_startup("qq_missing_dependency", "httpx not installed")
        if not self._app_id or not self._client_secret:
            return self._fail_startup(
                "qq_missing_credentials",
                "QQ_APP_ID and QQ_CLIENT_SECRET are required",
            )
        if not self._acquire_platform_lock("qqbot-appid", self._app_id, "QQBot app ID"):
            return False

        try:
            self.setup_http_client(httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                event_hooks={"response": [_ssrf_redirect_guard]},
            ))

            await self._api.ensure_token()
            gateway_url = await self._api.get_gateway_url()
            logger.info("[%s] Gateway URL: %s", self._log_tag, gateway_url)

            main_loop = asyncio.get_running_loop()
            self._ws_manager.start(gateway_url, main_loop)
            self._mark_connected()
            logger.info("[%s] Connected", self._log_tag)
            return True

        except Exception as exc:
            message = f"QQ startup failed: {exc}"
            self._set_fatal_error("qq_connect_error", message, retryable=True)
            logger.error("[%s] %s", self._log_tag, message, exc_info=True)
            await self._cleanup()
            self._release_platform_lock()
            return False

    async def disconnect(self) -> None:
        """Close all connections and stop listeners."""
        self._running = False
        self._mark_disconnected()
        self._flush_session()
        await self._ws_manager.async_stop()
        await self._cleanup()
        self._release_platform_lock()
        logger.info("[%s] Disconnected", self._log_tag)

    def setup_http_client(self, http_client: Any) -> None:
        """Inject an HTTP client into all components that need one.

        Called by ``connect()`` after creating the httpx client, and by
        one-shot callers (e.g. send_message_tool) that skip the WebSocket
        ``connect()`` flow.

        :param http_client: An ``httpx.AsyncClient`` (or compatible object).
        """
        self._http_client = http_client
        self._api.setup(http_client)
        self._downloader.update_http_client(http_client)
        self._stt.update_http_client(http_client)
        self._media_uploader.update_http_client(http_client)

    async def _cleanup(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._fail_pending("Disconnected")

    def _fail_startup(self, code: str, message: str) -> bool:
        full = f"QQ startup failed: {message}"
        self._set_fatal_error(code, full, retryable=True)
        logger.warning("[%s] %s", self._log_tag, full)
        return False

    # ------------------------------------------------------------------
    # Inbound: button interaction → approval resolve
    # ------------------------------------------------------------------

    async def _on_interaction(self, event_type: str, raw: Dict[str, Any]) -> None:
        """Handle INTERACTION_CREATE (button click).

        1. ACK the interaction immediately so the button stops spinning.
        2. If the button_data matches an approval payload, resolve the
           waiting agent thread via ``tools.approval.resolve_gateway_approval``.
        """
        del event_type

        event = parse_interaction_event(raw)
        interaction_id = event.id

        # Always ACK first — QQ requires a prompt response.
        try:
            await self._api.acknowledge_interaction(interaction_id)
        except Exception as exc:
            logger.warning("[%s] Failed to ACK interaction %s: %s", self._log_tag, interaction_id, exc)

        button_data = event.data.resolved.button_data if event.data and event.data.resolved else ""
        if not button_data:
            logger.debug("[%s] Interaction %s has no button_data", self._log_tag, interaction_id)
            return

        parsed = parse_approval_button_data(button_data)
        if parsed is not None:
            session_key, decision = parsed
            choice_map = {
                "allow-once": "once",
                "allow-session": "session",
                "allow-always": "always",
                "deny": "deny",
            }
            choice = choice_map.get(decision, "deny")
            try:
                from tools.approval import resolve_gateway_approval
                count = resolve_gateway_approval(session_key, choice)
                logger.info(
                    "[%s] Approval resolved: session=%s choice=%s count=%d",
                    self._log_tag, session_key[:30], choice, count,
                )
            except Exception as exc:
                logger.error(
                    "[%s] resolve_gateway_approval failed for session %s: %s",
                    self._log_tag, session_key[:30], exc,
                )
            return

        answer = parse_update_prompt_button_data(button_data)
        if answer is not None:
            try:
                from hermes_constants import get_hermes_home
                home = get_hermes_home()
                response_path = home / ".update_response"
                tmp = response_path.with_suffix(".tmp")
                tmp.write_text(answer)
                tmp.replace(response_path)
                logger.info("[%s] Update prompt answered '%s'", self._log_tag, answer)
            except Exception as exc:
                logger.error("[%s] Failed to write update response: %s", self._log_tag, exc)
            return

        logger.debug("[%s] Interaction %s: unrecognised button_data %r", self._log_tag, interaction_id, button_data)

    # ------------------------------------------------------------------
    # Outbound: send approval prompt with inline keyboard
    # ------------------------------------------------------------------

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an inline-keyboard approval prompt.

        Called by ``gateway/run.py`` when the agent requests approval for a
        dangerous command.  Delegates to :class:`~core.approval.ApprovalSender`.

        :param chat_id: Target chat (user openid or group openid).
        :param command: The command string to display.
        :param session_key: Hermes session key embedded in button payloads.
        :param description: Short reason string.
        :param metadata: Ignored (kept for API parity with other adapters).
        :returns: :class:`~gateway.platforms.base.SendResult`.
        """
        del metadata

        chat_type = self._guess_chat_type(chat_id)
        msg_id = self._last_msg_id.get(chat_id)
        req = ApprovalRequest(
            session_key=session_key,
            title=description,
            command_preview=command,
        )
        ok = await self._approval_sender.send(chat_type, chat_id, req, msg_id=msg_id)
        if ok:
            return SendResult(success=True)
        return SendResult(success=False, error="ApprovalSender.send() returned False")

    async def send_update_prompt(
        self,
        chat_id: str,
        prompt: str,
        default: str = "",
        session_key: str = "",
    ) -> SendResult:
        """Send a Yes/No inline-keyboard prompt for hermes update confirmations.

        Called by ``gateway/run.py`` when ``hermes update --gateway`` needs user
        input (e.g. stash restore, config migration).  The user's choice is written
        to ``~/.hermes/.update_response`` so the detached update process can pick it up.

        :param chat_id: Target chat (user openid or group openid).
        :param prompt: Question text to display.
        :param default: Default answer hint (``'y'`` or ``'n'``).
        :param session_key: Unused for QQBot (kept for API parity).
        :returns: :class:`~gateway.platforms.base.SendResult`.
        """
        chat_type = self._guess_chat_type(chat_id)
        msg_id = self._last_msg_id.get(chat_id)
        default_hint = f"（默认：{'是' if default == 'y' else '否'}）" if default else ""
        text = f"⚕ **更新需要确认**\n\n{prompt}{default_hint}"
        keyboard = build_update_prompt_keyboard()
        body = self._api.build_text_body(text, reply_to=msg_id, markdown=True)
        try:
            if chat_type == "c2c":
                await self._api.post_c2c_message(chat_id, body, keyboard=keyboard)
            else:
                await self._api.post_group_message(chat_id, body, keyboard=keyboard)
            logger.info("[%s] send_update_prompt sent to %s:%s", self._log_tag, chat_type, chat_id)
        except Exception as exc:
            logger.error("[%s] send_update_prompt failed: %s", self._log_tag, exc)
            raise

    # ------------------------------------------------------------------
    # Inbound: core event → hermes MessageEvent
    # ------------------------------------------------------------------

    async def _on_core_message(self, event_type: str, raw: Dict[str, Any]) -> None:
        """Called by QQWebSocket for each inbound user message."""
        if not isinstance(raw, dict):
            return

        logger.debug("[%s] Raw inbound: %s", self._log_tag, raw)

        event = self._event_parser.parse(event_type, raw)
        if event is None:
            return

        if not self._check_acl(event):
            return

        msg_event = await self._build_message_event(event)
        if msg_event is None:
            return

        self._chat_type_map[event.chat_id] = event.chat_scope
        self._last_msg_id[event.chat_id] = event.message_id
        await super().handle_message(msg_event)

    async def _build_message_event(
        self,
        event: InboundEvent,
    ) -> Optional[MessageEvent]:
        """Convert InboundEvent → hermes MessageEvent."""
        # Process attachments
        processed = await self._att_processor.process(event.attachments)

        # Resolve quote
        reply_to_id, reply_to_text = await self._resolve_quote(event)

        # Build text
        text = event.content
        for att in processed:
            if att.kind == "voice" and att.transcript:
                text = _append_block(text, f"[Voice] {att.transcript}")
            elif att.description:
                text = _append_block(text, att.description)

        # Collect image paths
        image_urls = [a.local_path for a in processed if a.kind == "image" and a.local_path]
        image_types = [a.content_type for a in processed if a.kind == "image" and a.local_path]

        if not text.strip() and not image_urls:
            return None

        return MessageEvent(
            source=self.build_source(
                chat_id=event.chat_id,
                user_id=event.user_id,
                user_name=event.user_name,
                chat_type=(
                    "dm" if event.chat_scope in ("c2c", "dm")
                    else "channel" if event.chat_scope == "guild"
                    else "group"
                ),
            ),
            text=text,
            message_type=_detect_message_type(image_urls, image_types),
            raw_message=event.raw,
            message_id=event.message_id,
            media_urls=image_urls,
            media_types=image_types,
            timestamp=self._parse_qq_timestamp(event.timestamp),
            reply_to_message_id=reply_to_id,
            reply_to_text=reply_to_text,
        )

    # ------------------------------------------------------------------
    # Quote resolution (引用消息)
    # ------------------------------------------------------------------

    async def _resolve_quote(
        self,
        event: InboundEvent,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve quoted message elements into (ref_id, text) pair."""
        from .core.dto import MSG_TYPE_QUOTE

        if event.message_type != MSG_TYPE_QUOTE or not event.msg_elements:
            return None, None

        elem = event.msg_elements[0]
        ref_id = elem.msg_idx or None
        parts: List[str] = []

        if elem.content.strip():
            parts.append(elem.content.strip())

        for att in elem.attachments:
            ct = att.content_type.lower()
            fname = att.filename
            url = att.resolved_url
            cached = None
            if url:
                try:
                    cached = await self._downloader.download(url, ct, fname)
                except Exception as exc:
                    logger.debug("[%s] Failed to cache quoted attachment: %s", self._log_tag, exc)
            parts.append(_describe_attachment(ct, fname, cached))

        body = " ".join(parts) if parts else "[empty message]"
        # Wrap in brackets to bypass found_in_history exact match constraint.
        return ref_id, f"[{body}]"

    # ------------------------------------------------------------------
    # Outbound: send text
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text or markdown message."""
        del metadata

        if not content or not content.strip():
            return SendResult(success=True)

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)

        last_result = SendResult(success=False, error="No chunks")
        for chunk in chunks:
            last_result = await self._send_chunk(chat_id, chunk, reply_to)
            if not last_result.success:
                return last_result
            reply_to = None
        return last_result

    async def _send_chunk(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Send one chunk with up to 3 retries and exponential backoff."""
        chat_type = self._guess_chat_type(chat_id)
        last_exc: Optional[Exception] = None

        for attempt in range(3):
            try:
                return await self._send_by_type(chat_type, chat_id, content, reply_to)
            except Exception as exc:
                last_exc = exc
                if self._is_fatal_send_error(str(exc)):
                    break
                if attempt < 2:
                    delay = 1.0 * (2 ** attempt)
                    logger.warning(
                        "[%s] send retry %d/3 after %.1fs: %s",
                        self._log_tag, attempt + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)

        error_msg = str(last_exc) if last_exc else "Unknown error"
        logger.error("[%s] Send failed: %s", self._log_tag, error_msg)
        return SendResult(
            success=False,
            error=error_msg,
            retryable=not self._is_fatal_send_error(error_msg),
        )

    async def _send_by_type(
        self,
        chat_type: str,
        chat_id: str,
        content: str,
        reply_to: Optional[str],
    ) -> SendResult:
        """Dispatch send to the correct endpoint based on chat type."""
        if chat_type == "c2c":
            return await self._send_c2c_text(chat_id, content, reply_to)
        if chat_type == "group":
            return await self._send_group_text(chat_id, content, reply_to)
        if chat_type == "guild":
            return await self._send_guild_text(chat_id, content, reply_to)
        return SendResult(success=False, error=f"Unknown chat type for {chat_id!r}")

    async def _send_c2c_text(
        self,
        openid: str,
        content: str,
        reply_to: Optional[str],
    ) -> SendResult:
        msg = self._api.build_text_body(
            content,
            reply_to=reply_to,
            markdown=self._markdown_support,
            max_length=self.MAX_MESSAGE_LENGTH,
        )
        data = await self._api.post_c2c_message(openid, msg)
        return SendResult(
            success=True,
            message_id=str(data.get("id", uuid.uuid4().hex[:12])),
            raw_response=data,
        )

    async def _send_group_text(
        self,
        group_openid: str,
        content: str,
        reply_to: Optional[str],
    ) -> SendResult:
        msg = self._api.build_text_body(
            content,
            reply_to=reply_to,
            markdown=self._markdown_support,
            max_length=self.MAX_MESSAGE_LENGTH,
        )
        data = await self._api.post_group_message(group_openid, msg)
        return SendResult(
            success=True,
            message_id=str(data.get("id", uuid.uuid4().hex[:12])),
            raw_response=data,
        )

    async def _send_guild_text(
        self,
        channel_id: str,
        content: str,
        reply_to: Optional[str],
    ) -> SendResult:
        msg = GuildMessageToCreate(
            content=content[: self.MAX_MESSAGE_LENGTH],
            msg_id=reply_to or "",
        )
        data = await self._api.post_guild_message(channel_id, msg)
        return SendResult(
            success=True,
            message_id=str(data.get("id", uuid.uuid4().hex[:12])),
            raw_response=data,
        )

    # ------------------------------------------------------------------
    # Outbound: send media
    # ------------------------------------------------------------------

    def _media_upload_error_text(self, exc: Exception) -> str:
        """Build a user-facing Chinese error message for upload failures."""
        if isinstance(exc, UploadDailyLimitExceededError):
            return (
                f"今天的文件上传额度用完啦 😅 {exc.file_name}（{exc.file_size_human}）暂时发不过去，"
                f"明天额度重置后再试试吧~"
            )
        if isinstance(exc, UploadFileTooLargeError):
            return (
                f"{exc.file_name}（{exc.file_size_human}）有点大，超过平台 {exc.limit_human} 的限制了 😓"
                f"压缩一下再发吧~"
            )
        return str(exc)

    async def _send_media_to_user(
        self,
        chat_id: str,
        source: str,
        file_type: int,
        caption: Optional[str],
        reply_to: Optional[str],
        file_name: Optional[str] = None,
    ) -> SendResult:
        """``_send_media`` wrapper for gateway send paths (user-facing).

        Catches upload quota/size exceptions and sends a Chinese notification
        to the user instead of propagating the exception.
        """
        try:
            return await self._send_media(chat_id, source, file_type, caption, reply_to, file_name)
        except (UploadDailyLimitExceededError, UploadFileTooLargeError) as exc:
            logger.warning("[%s] Media upload error: %s", self._log_tag, exc)
            user_msg = self._media_upload_error_text(exc)
            await self.send(chat_id, user_msg)
            return SendResult(success=False, retryable=False, error=str(exc))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        del metadata
        result = await self._send_media_to_user(chat_id, image_url, MEDIA_TYPE_IMAGE, caption, reply_to)
        if result.success or not MediaLoader.is_url(image_url):
            return result
        # Fallback to text URL
        logger.warning("[%s] Image send failed, falling back to text: %s", self._log_tag, result.error)
        fallback = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id, fallback, reply_to=reply_to)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del kwargs
        return await self._send_media_to_user(chat_id, image_path, MEDIA_TYPE_IMAGE, caption, reply_to)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del kwargs
        return await self._send_media_to_user(chat_id, audio_path, MEDIA_TYPE_VOICE, caption, reply_to)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del kwargs
        return await self._send_media_to_user(chat_id, video_path, MEDIA_TYPE_VIDEO, caption, reply_to)

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
        return await self._send_media_to_user(
            chat_id, file_path, MEDIA_TYPE_FILE, caption, reply_to, file_name=file_name
        )

    async def _send_media(
        self,
        chat_id: str,
        source: str,
        file_type: int,
        caption: Optional[str],
        reply_to: Optional[str],
        file_name: Optional[str] = None,
    ) -> SendResult:
        """Core media send pipeline: upload → send rich-media message."""
        chat_type = self._guess_chat_type(chat_id)
        if chat_type == "guild":
            return SendResult(success=False, error="Guild media send not supported via this path")

        try:
            file_info = await self._media_uploader.upload(
                chat_type, chat_id, source, file_type, file_name
            )
        except (UploadDailyLimitExceededError, UploadFileTooLargeError):
            raise
        except Exception as exc:
            logger.error("[%s] Media upload failed: %s", self._log_tag, exc)
            return SendResult(success=False, error=str(exc))

        send_msg = MessageToCreate(
            msg_type=QQMessageType.RICH_MEDIA,
            msg_seq=self._api.next_msg_seq(chat_id),
            msg_id=reply_to or "",
            content=caption[: self.MAX_MESSAGE_LENGTH] if caption else "",
        )
        from .core.dto import MediaInfo
        send_msg.media = MediaInfo(file_info=file_info)

        send_path = (
            f"/v2/users/{chat_id}/messages"
            if chat_type == "c2c"
            else f"/v2/groups/{chat_id}/messages"
        )
        try:
            data = await self._api.request("POST", send_path, send_msg.to_dict())
            return SendResult(
                success=True,
                message_id=str(data.get("id", uuid.uuid4().hex[:12])),
                raw_response=data,
            )
        except Exception as exc:
            logger.error("[%s] Media send failed: %s", self._log_tag, exc)
            return SendResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Typing indicator
    # ------------------------------------------------------------------

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        """Send an input notify to a C2C user."""
        del metadata
        if not self.is_connected:
            return
        if self._guess_chat_type(chat_id) != "c2c":
            return

        msg_id = self._last_msg_id.get(chat_id)
        if not msg_id:
            return

        now = time.time()
        if now - self._typing_sent_at.get(chat_id, 0.0) < self._TYPING_DEBOUNCE_SECONDS:
            return

        try:
            msg = MessageToCreate(
                msg_type=QQMessageType.INPUT_NOTIFY,
                msg_id=msg_id,
                msg_seq=self._api.next_msg_seq(chat_id),
                input_notify=InputNotify(
                    input_type=1,
                    input_second=self._TYPING_INPUT_SECONDS,
                ),
            )
            await self._api.post_c2c_message(chat_id, msg)
            self._typing_sent_at[chat_id] = now
        except Exception as exc:
            logger.debug("[%s] send_typing failed: %s", self._log_tag, exc)

    # ------------------------------------------------------------------
    # Format / info
    # ------------------------------------------------------------------

    def format_message(self, content: str) -> str:
        if self._markdown_support:
            return content
        return strip_markdown(content)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_type = self._guess_chat_type(chat_id)
        return {
            "name": chat_id,
            "type": "group" if chat_type in ("group", "guild") else "dm",
        }

    # ------------------------------------------------------------------
    # Reconnection wait
    # ------------------------------------------------------------------

    async def _wait_for_reconnection(self) -> bool:
        logger.info(
            "[%s] Not connected — waiting for reconnection (up to %.0fs)",
            self._log_tag,
            self._RECONNECT_WAIT_SECONDS,
        )
        waited = 0.0
        while waited < self._RECONNECT_WAIT_SECONDS:
            await asyncio.sleep(self._RECONNECT_POLL_INTERVAL)
            waited += self._RECONNECT_POLL_INTERVAL
            if self.is_connected:
                logger.info("[%s] Reconnected after %.1fs", self._log_tag, waited)
                return True
        logger.warning(
            "[%s] Still not connected after %.0fs",
            self._log_tag,
            self._RECONNECT_WAIT_SECONDS,
        )
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_session(
        self,
        session_id: Optional[str],
        last_seq: Optional[int],
    ) -> None:
        self._session_id = session_id
        if last_seq is not None:
            self._last_seq = last_seq
        if session_id and self._app_id:
            # Mark dirty — flushed on next heartbeat ACK or disconnect.
            self._session_dirty = True
        elif not session_id and self._app_id:
            # Session cleared (re-identify) — remove stale persisted data.
            self._session_dirty = False
            app_id = self._app_id
            store = self._ws_session_store
            try:
                loop = asyncio.get_running_loop()
                loop.run_in_executor(None, store.clear, app_id)
            except RuntimeError:
                store.clear(app_id)

    def _flush_session(self) -> None:
        """Write dirty session state to disk via thread pool to avoid blocking the event loop."""
        if not self._session_dirty:
            return
        self._session_dirty = False
        if not (self._session_id and self._app_id):
            return
        # Capture values now (in the event-loop thread) before handing off.
        app_id = self._app_id
        session_id = self._session_id
        seq = self._last_seq
        intents = self._intents
        bot_username = self._bot_username
        store = self._ws_session_store

        def _write() -> None:
            store.save(
                app_id=app_id,
                session_id=session_id,
                seq=seq,
                intents=intents,
                bot_username=bot_username,
            )

        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _write)
        except RuntimeError:
            # No running loop (e.g. called from disconnect before loop teardown)
            _write()

    def _fail_pending(self, reason: str) -> None:
        """No-op: QQAdapter doesn't use pending-response futures."""

    def _guess_chat_type(self, chat_id: str) -> str:
        return self._chat_type_map.get(chat_id, "c2c")

    def _media_headers(self) -> Dict[str, str]:
        return self._api.media_headers()

    def _is_dm_allowed(self, user_id: str) -> bool:
        if self._dm_policy == "disabled":
            return False
        if self._dm_policy == "allowlist":
            return _entry_matches(self._allow_from, user_id)
        return True

    def _is_group_allowed(self, group_id: str, user_id: str) -> bool:
        del user_id
        if self._group_policy == "disabled":
            return False
        if self._group_policy == "allowlist":
            return _entry_matches(self._group_allow_from, group_id)
        return True

    def _check_acl(self, event: InboundEvent) -> bool:
        from .core.dto import EventType

        if event.event_type == EventType.C2C_MESSAGE_CREATE:
            return self._is_dm_allowed(event.user_id)
        if event.event_type == EventType.GROUP_AT_MESSAGE_CREATE:
            return self._is_group_allowed(event.chat_id, event.user_id)
        return True

    def _parse_qq_timestamp(self, raw: str) -> datetime:
        if not raw:
            return datetime.now(tz=timezone.utc)
        try:
            return datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            pass
        try:
            return datetime.fromtimestamp(int(raw) / 1000, tz=timezone.utc)
        except (ValueError, TypeError):
            pass
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _get_cache_dir() -> str:
        """Return the hermes cache directory for QQ Bot media files."""
        try:
            from hermes_constants import get_hermes_home

            return str(get_hermes_home() / "cache" / "qqbot")
        except Exception:
            import tempfile

            return str(tempfile.gettempdir())

    @staticmethod
    def _is_fatal_send_error(error_msg: str) -> bool:
        lower = error_msg.lower()
        return any(k in lower for k in ("invalid", "forbidden", "not found", "bad request"))


# ── Module-level helpers ──────────────────────────────────────────────

def _append_block(base: str, block: str) -> str:
    """Append *block* to *base* with double-newline separator."""
    if base.strip():
        return (base + "\n\n" + block).strip()
    return block


def _detect_message_type(
    media_urls: List[str],
    media_types: List[str],
) -> MessageType:
    """Infer MessageType from attachment MIME types."""
    if not media_urls:
        return MessageType.TEXT
    if not media_types:
        return MessageType.PHOTO
    first = media_types[0].lower()
    if "audio" in first or "voice" in first or "silk" in first:
        return MessageType.VOICE
    if "video" in first:
        return MessageType.VIDEO
    if "image" in first or "photo" in first:
        return MessageType.PHOTO
    return MessageType.TEXT


def _describe_attachment(ct: str, fname: str, cached: Optional[str]) -> str:
    """Build a text description for a quoted attachment."""
    if ct.startswith("image/"):
        return f"[image: {fname} ({cached})]" if cached else (f"[image: {fname}]" if fname else "[image]")
    if "audio" in ct or "voice" in ct or "silk" in ct:
        return f"[voice message ({cached})]" if cached else "[voice message]"
    if ct.startswith("video/"):
        return f"[video: {fname} ({cached})]" if cached else (f"[video: {fname}]" if fname else "[video]")
    return f"[file: {fname} ({cached})]" if cached else (f"[file: {fname}]" if fname else "[attachment]")


def _entry_matches(entries: List[str], target: str) -> bool:
    """Return True if *target* matches any entry in the allowlist."""
    normalized = str(target).strip().lower()
    for entry in entries:
        e = str(entry).strip().lower()
        if e in ("*", normalized):
            return True
    return False
