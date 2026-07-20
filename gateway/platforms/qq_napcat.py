"""QQ platform adapter via NapCat / OneBot 11 WebSocket API."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, unquote, urlsplit, urlunsplit

try:
    import aiohttp

    QQ_NAPCAT_AVAILABLE = True
except ImportError:
    aiohttp = None
    QQ_NAPCAT_AVAILABLE = False

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.group_runtime import (
    GroupBatchItem,
    GroupDispatchThresholds,
    GroupTriggerState,
    build_group_message_metadata,
    decide_project_group_dispatch,
    resolve_group_trigger_reason,
    text_looks_like_request,
)
from gateway.group_runtime_service import (
    qq_group_message_allowed,
    qq_policy_has_runtime_override,
    resolve_qq_effective_group_policy,
)
from gateway.qq_group_archive import QqGroupArchiveStore
from gateway.qq_intel_assignments import get_group_monitoring_overlay, list_intel_workers
from gateway.qq_intents import (
    _QQ_BUSY_SHORTCUT_MARKERS,
    _QQ_DEFAULT_TRIGGER_ALIASES,
    _QQ_LOW_VALUE_IMAGE_HINTS,
)
from gateway.qq_napcat_runtime import diagnose_local_qq_napcat_endpoint
from gateway.qq_group_policies import default_group_policy, get_group_policy, has_group_policy
from gateway.qq_social_policy import get_social_policy
from gateway.qq_social_requests import record_social_request_event, update_social_request_status
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_url,
)

logger = logging.getLogger(__name__)


@dataclass
class _BufferedGroupMessage:
    event: MessageEvent
    payload: Dict[str, Any]
    observed_at: float


def check_qq_napcat_requirements() -> bool:
    """Return True when the optional transport dependencies are installed."""
    return QQ_NAPCAT_AVAILABLE


def _guess_ext_from_name(name: str, default: str) -> str:
    suffix = Path(_decoded_path_from_ref(name)).suffix.lower()
    return suffix or default


def _decoded_path_from_ref(value: str) -> str:
    """Return the path portion of a URL/URI with escapes decoded."""
    text = str(value or "").strip()
    parsed = urlsplit(text)
    return unquote(parsed.path or text)


def _normalized_media_ref(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = urlsplit(text)
    if parsed.scheme in {"http", "https"}:
        return f"{parsed.netloc}{unquote(parsed.path)}".lower()
    return _decoded_path_from_ref(text).lower()


def _looks_like_low_value_image_ref(value: Any) -> bool:
    ref = _normalized_media_ref(value)
    if not ref:
        return False
    if ref.endswith(".gif"):
        return True
    if ref.endswith(".webp") and any(hint in ref for hint in _QQ_LOW_VALUE_IMAGE_HINTS):
        return True
    return False


def _is_qq_signed_image_ref(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    parsed = urlsplit(text)
    host = (parsed.netloc or "").strip().lower()
    return parsed.scheme in {"http", "https"} and host == "multimedia.nt.qq.com.cn"


def _with_access_token(ws_url: str, access_token: str) -> str:
    """Append/replace access_token without corrupting an existing path or query."""
    ws_url = str(ws_url or "").strip()
    token = str(access_token or "").strip()
    if not token:
        return ws_url

    parsed = urlsplit(ws_url)
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "access_token"
    ]
    query_items.append(("access_token", token))
    return urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        urlencode(query_items),
        parsed.fragment,
    ))


def resolve_qq_napcat_group_id(value: Any) -> int:
    """Normalize a QQ group target into its numeric group id."""
    text = str(value or "").strip()
    if text.startswith("qq_napcat:"):
        text = text.split(":", 1)[1]
    if text.startswith("group:"):
        return int(text.split(":", 1)[1])
    if text.startswith("dm:"):
        raise ValueError("QQ NapCat group file actions require a group target, not dm:<id>")
    if text.lstrip("-").isdigit():
        return int(text)
    raise ValueError("QQ NapCat group target must use 'group:<id>' or a numeric group id")


def normalize_qq_napcat_local_path(path: str) -> str:
    """Expand and absolutize a local file path for NapCat file APIs."""
    return str(Path(os.path.abspath(os.path.expanduser(path))))


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def _unique_nonempty_text(values: list[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


class QqNapCatAdapter(BasePlatformAdapter):
    """Hermes platform adapter for QQ via NapCat's OneBot websocket."""

    platform = Platform.QQ_NAPCAT
    MAX_MESSAGE_LENGTH = 4000

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.QQ_NAPCAT)
        extra = config.extra or {}
        self.ws_url = str(extra.get("ws_url") or "").strip()
        self.access_token = str(extra.get("access_token") or "").strip()
        self.reconnect_interval = int(extra.get("reconnect_interval") or 5)
        self.allowed_groups = {str(group) for group in (extra.get("allowed_groups") or []) if str(group).strip()}
        self.allow_all_groups = bool(extra.get("allow_all_groups", False))
        self._admin_users = {str(user).strip() for user in (extra.get("admin_users") or []) if str(user).strip()}
        self._mention_patterns = self._compile_mention_patterns()
        self._default_trigger_aliases = _QQ_DEFAULT_TRIGGER_ALIASES
        self._bot_user_id = ""
        self._followup_window_seconds = int(extra.get("followup_window_seconds") or 900)
        self._project_group_mode = _coerce_bool(extra.get("project_group_mode"), False)
        self._group_batch_debounce_seconds = max(
            0.0,
            float(extra.get("group_batch_debounce_seconds") or 1.0),
        )
        self._group_min_model_interval_seconds = max(
            0.0,
            float(extra.get("group_min_model_interval_seconds") or 8.0),
        )
        self._group_batch_retry_seconds = max(
            0.05,
            float(extra.get("group_batch_retry_seconds") or 0.5),
        )
        self._group_observed_max_messages = max(
            1,
            int(extra.get("group_observed_max_messages") or 80),
        )
        self._group_batch_max_messages = max(
            1,
            int(extra.get("group_batch_max_messages") or 30),
        )
        self._group_trigger_min_messages = 4
        self._group_trigger_min_speakers = 3
        self._group_trigger_min_chars = 160
        self._group_followup_windows: Dict[Tuple[str, str], float] = {}
        self._group_shared_followup_windows: Dict[str, float] = {}
        self._recent_group_bot_messages: Dict[str, Tuple[str, float]] = {}
        self._max_recent_group_messages = 500
        self._group_observed_messages: Dict[str, list[_BufferedGroupMessage]] = {}
        self._group_pending_batches: Dict[str, list[_BufferedGroupMessage]] = {}
        self._group_batch_tasks: Dict[str, asyncio.Task] = {}
        self._group_last_dispatch_at: Dict[str, float] = {}
        self._group_last_included_at: Dict[str, float] = {}
        self._recent_inbound_message_ids: Dict[str, float] = {}
        self._inbound_message_dedupe_ttl_seconds = max(
            5.0,
            float(extra.get("inbound_message_dedupe_ttl_seconds") or 120.0),
        )
        self._group_archive_store = QqGroupArchiveStore()

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._send_lock = asyncio.Lock()
        self._pending_calls: Dict[str, asyncio.Future] = {}
        self._echo_counter = itertools.count(1)
        self._chat_types: Dict[str, str] = {}

    def _qq_require_mention(self) -> bool:
        configured = self.config.extra.get("require_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in ("true", "1", "yes", "on")
            return bool(configured)
        return os.getenv("QQ_NAPCAT_REQUIRE_MENTION", "false").lower() in ("true", "1", "yes", "on")

    def _compile_mention_patterns(self):
        patterns = self.config.extra.get("mention_patterns")
        if patterns is None:
            raw = os.getenv("QQ_NAPCAT_MENTION_PATTERNS", "").strip()
            if raw:
                try:
                    patterns = json.loads(raw)
                except Exception:
                    patterns = [part.strip() for part in raw.splitlines() if part.strip()]
                    if not patterns:
                        patterns = [part.strip() for part in raw.split(",") if part.strip()]
        if patterns is None:
            return []
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            logger.warning("[%s] qq_napcat mention_patterns must be a list or string; got %s", self.name, type(patterns).__name__)
            return []

        compiled = []
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                logger.warning("[%s] Invalid QQ mention pattern %r: %s", self.name, pattern, exc)
        if compiled:
            logger.info("[%s] Loaded %d QQ mention pattern(s)", self.name, len(compiled))
        return compiled

    @staticmethod
    def _extract_reply_message_id(payload: Dict[str, Any]) -> Optional[str]:
        segments = payload.get("message")
        if isinstance(segments, list):
            for segment in segments:
                if str(segment.get("type") or "").lower() != "reply":
                    continue
                reply_id = str((segment.get("data") or {}).get("id") or "").strip()
                if reply_id:
                    return reply_id
        reply_id = str(payload.get("reply_to_message_id") or "").strip()
        return reply_id or None

    def _message_mentions_bot(self, payload: Dict[str, Any]) -> bool:
        bot_id = str(payload.get("self_id") or self._bot_user_id or "").strip()
        if bot_id:
            self._bot_user_id = bot_id

        segments = payload.get("message")
        if isinstance(segments, list):
            for segment in segments:
                if str(segment.get("type") or "").lower() != "at":
                    continue
                qq = str((segment.get("data") or {}).get("qq") or "").strip()
                if qq and bot_id and qq == bot_id:
                    return True
                if qq and self._bot_user_id and qq == self._bot_user_id:
                    return True

        raw_text = str(payload.get("raw_message") or "")
        for candidate in filter(None, {bot_id, self._bot_user_id}):
            if f"@{candidate}" in raw_text:
                return True
        return False

    def _is_explicit_busy_followup(self, event: MessageEvent) -> bool:
        if super()._is_explicit_busy_followup(event):
            return True
        if getattr(event.source, "chat_type", "") != "group":
            return False

        user_id = str(getattr(event.source, "user_id", "") or "").strip()
        if user_id and user_id in self._admin_users:
            return True

        for body in self._message_text_candidates(event):
            if not body:
                continue
            if any(alias in body for alias in self._default_trigger_aliases):
                return True
            for pattern in self._mention_patterns:
                try:
                    if pattern.search(body):
                        return True
                except Exception:
                    continue
        return False

    @staticmethod
    def _looks_like_busy_shortcut_phrase(body: str) -> bool:
        text = str(body or "").strip()
        if not text:
            return False
        return any(marker in text for marker in _QQ_BUSY_SHORTCUT_MARKERS)

    def _should_inline_active_session_message(self, event: MessageEvent) -> bool:
        source = getattr(event, "source", None)
        if getattr(source, "platform", None) != Platform.QQ_NAPCAT:
            return False
        if event.get_command():
            return False
        if getattr(event, "message_type", None) != MessageType.TEXT:
            return False
        if getattr(event, "media_urls", None):
            return False

        user_id = str(getattr(source, "user_id", "") or "").strip()
        is_admin = bool(user_id and user_id in self._admin_users)
        explicit_followup = self._is_explicit_busy_followup(event)

        try:
            from gateway.run import _looks_like_qq_active_session_inline_candidate
        except Exception:
            return False

        for body in self._message_text_candidates(event):
            if not body:
                continue
            if explicit_followup and self._looks_like_busy_shortcut_phrase(body):
                return True
            if _looks_like_qq_active_session_inline_candidate(
                body,
                is_admin=is_admin,
                explicit_followup=explicit_followup,
            ):
                return True
        return False

    def _busy_followup_ack(self, event: MessageEvent, *, interrupting: bool = False) -> str:
        source = getattr(event, "source", None)
        if getattr(source, "chat_type", "") == "dm":
            if interrupting:
                return "收到，上一轮有点久，我先切到你这条，马上接着回你。"
            return "收到，这条我先排队，上一轮忙完马上接着回你。"

        if self._is_explicit_busy_followup(event):
            if interrupting:
                return "收到，上一轮有点久，我先切到这条，马上接着回。"
            return "收到，这条我先排队，上一轮忙完接着回你。"
        return ""

    def _message_matches_mention_patterns(self, payload: Dict[str, Any]) -> bool:
        body = str(payload.get("raw_message") or "")
        if not body:
            segments = payload.get("message")
            if isinstance(segments, list):
                body = "".join(
                    str((segment.get("data") or {}).get("text") or "")
                    for segment in segments
                    if str(segment.get("type") or "").lower() == "text"
                )
        if any(alias in body for alias in self._default_trigger_aliases):
            return True
        if not self._mention_patterns:
            return False
        return any(pattern.search(body) for pattern in self._mention_patterns)

    def _group_message_trigger_reason(self, payload: Dict[str, Any]) -> Optional[str]:
        if not self._group_message_allowed(payload):
            return None

        group_id = str(payload.get("group_id") or "").strip()
        state = GroupTriggerState(
            require_explicit_trigger=self._qq_require_mention(),
            slash_command=str(payload.get("raw_message") or "").strip().startswith("/"),
            mentioned_bot=self._message_mentions_bot(payload),
            replied_to_bot=self._message_is_reply_to_bot(payload),
            shared_followup=self._group_runs_project_mode(group_id) and self._has_group_followup_window(payload),
            user_followup=self._has_followup_window(payload),
            recent_session_followup=self._has_recent_session_followup(payload),
            name_trigger=self._message_matches_mention_patterns(payload),
        )
        return resolve_group_trigger_reason(state)

    def _clean_bot_mention_text(self, text: str, payload: Dict[str, Any]) -> str:
        bot_id = str(payload.get("self_id") or self._bot_user_id or "").strip()
        if bot_id:
            self._bot_user_id = bot_id

        segments = payload.get("message")
        if isinstance(segments, list):
            parts = []
            for segment in segments:
                seg_type = str(segment.get("type") or "").lower()
                data = segment.get("data") or {}
                if seg_type == "at":
                    qq = str(data.get("qq") or "").strip()
                    if qq and bot_id and qq == bot_id:
                        continue
                    if qq and self._bot_user_id and qq == self._bot_user_id:
                        continue
                    label = str(data.get("name") or qq or "").strip()
                    if label:
                        parts.append(label if label.startswith("@") else f"@{label}")
                    continue
                if seg_type == "text":
                    parts.append(str(data.get("text") or ""))
            return "".join(parts).strip()

        cleaned = text or ""
        for candidate in filter(None, {bot_id, self._bot_user_id}):
            cleaned = re.sub(rf"@{re.escape(candidate)}\b[,:\-]*\s*", "", cleaned)
        return cleaned.strip()

    def _cleanup_group_tracking_state(self) -> None:
        now = time.time()

        expired_followups = [
            key for key, expires_at in self._group_followup_windows.items()
            if expires_at <= now
        ]
        for key in expired_followups:
            self._group_followup_windows.pop(key, None)

        expired_shared_followups = [
            group_id
            for group_id, expires_at in self._group_shared_followup_windows.items()
            if expires_at <= now
        ]
        for group_id in expired_shared_followups:
            self._group_shared_followup_windows.pop(group_id, None)

        expired_message_ids = [
            message_id
            for message_id, (_, expires_at) in self._recent_group_bot_messages.items()
            if expires_at <= now
        ]
        for message_id in expired_message_ids:
            self._recent_group_bot_messages.pop(message_id, None)

        expired_inbound_ids = [
            message_id
            for message_id, expires_at in self._recent_inbound_message_ids.items()
            if expires_at <= now
        ]
        for message_id in expired_inbound_ids:
            self._recent_inbound_message_ids.pop(message_id, None)

        overflow = len(self._recent_group_bot_messages) - self._max_recent_group_messages
        if overflow > 0:
            stale_items = sorted(
                self._recent_group_bot_messages.items(),
                key=lambda item: item[1][1],
            )[:overflow]
            for message_id, _ in stale_items:
                self._recent_group_bot_messages.pop(message_id, None)

        for group_id, items in list(self._group_observed_messages.items()):
            if len(items) > self._group_observed_max_messages:
                self._group_observed_messages[group_id] = items[-self._group_observed_max_messages:]

    def _inbound_message_dedupe_key(self, payload: Dict[str, Any]) -> Optional[str]:
        message_id = str(
            payload.get("message_id")
            or payload.get("raw_message_id")
            or payload.get("real_id")
            or ""
        ).strip()
        if not message_id:
            return None

        message_type = str(payload.get("message_type") or "private").strip().lower()
        if message_type == "group":
            chat_id = str(payload.get("group_id") or "").strip()
        else:
            chat_id = str(payload.get("user_id") or "").strip()
        if not chat_id:
            return None
        return f"{message_type}:{chat_id}:{message_id}"

    def _is_duplicate_inbound_message(self, payload: Dict[str, Any]) -> bool:
        dedupe_key = self._inbound_message_dedupe_key(payload)
        if not dedupe_key:
            return False

        self._cleanup_group_tracking_state()
        now = time.time()
        expires_at = self._recent_inbound_message_ids.get(dedupe_key)
        if expires_at and expires_at > now:
            return True

        self._recent_inbound_message_ids[dedupe_key] = (
            now + self._inbound_message_dedupe_ttl_seconds
        )
        return False

    def _message_is_reply_to_bot(self, payload: Dict[str, Any]) -> bool:
        group_id = str(payload.get("group_id") or "").strip()
        reply_id = self._extract_reply_message_id(payload)
        if not group_id or not reply_id:
            return False

        self._cleanup_group_tracking_state()
        tracked = self._recent_group_bot_messages.get(reply_id)
        if not tracked:
            return False
        tracked_group_id, expires_at = tracked
        if expires_at <= time.time():
            self._recent_group_bot_messages.pop(reply_id, None)
            return False
        return tracked_group_id == group_id

    @staticmethod
    def _segment_text_for_message(segments: Any) -> str:
        """Rebuild human-visible text from structured QQ message segments.

        NapCat ``raw_message`` may contain CQ markup such as
        ``[CQ:image,file=...,url=...]``. That markup is not useful to the agent
        and can leak opaque file hashes into the prompt. For inbound message
        text, only keep the user-visible text and explicit ``@`` mentions.
        """
        if not isinstance(segments, list):
            return ""

        parts: list[str] = []
        for segment in segments:
            seg_type = str(segment.get("type") or "").strip().lower()
            data = segment.get("data") or {}
            if seg_type == "text":
                parts.append(str(data.get("text") or ""))
            elif seg_type == "at":
                qq = str(data.get("qq") or "").strip()
                if qq:
                    parts.append(f"@{qq}")
        return "".join(parts).strip()

    def _has_followup_window(self, payload: Dict[str, Any]) -> bool:
        group_id = str(payload.get("group_id") or "").strip()
        user_id = str(payload.get("user_id") or "").strip()
        if not group_id or not user_id:
            return False

        self._cleanup_group_tracking_state()
        expires_at = self._group_followup_windows.get((group_id, user_id))
        if not expires_at:
            return False
        if expires_at <= time.time():
            self._group_followup_windows.pop((group_id, user_id), None)
            return False
        return True

    def _session_key_for_source(self, source) -> str:
        """Build the effective session key for a source routed by this adapter."""
        from gateway.session import build_session_key

        extra = getattr(self.config, "extra", None) or {}
        group_sessions_per_user = bool(extra.get("group_sessions_per_user", True))
        thread_sessions_per_user = bool(extra.get("thread_sessions_per_user", False))
        return build_session_key(
            source,
            group_sessions_per_user=group_sessions_per_user,
            thread_sessions_per_user=thread_sessions_per_user,
        )

    def _has_group_followup_window(self, payload: Dict[str, Any]) -> bool:
        group_id = str(payload.get("group_id") or "").strip()
        if not group_id:
            return False

        self._cleanup_group_tracking_state()
        expires_at = self._group_shared_followup_windows.get(group_id)
        if not expires_at:
            return False
        if expires_at <= time.time():
            self._group_shared_followup_windows.pop(group_id, None)
            return False
        return True

    def _has_recent_session_followup(self, payload: Dict[str, Any]) -> bool:
        if self._followup_window_seconds <= 0:
            return False

        session_store = getattr(self, "_session_store", None)
        if session_store is None or not hasattr(session_store, "list_sessions"):
            return False

        try:
            source = self._build_message_event(payload).source
            session_key = self._session_key_for_source(source)
        except Exception:
            return False

        active_minutes = max(1, (self._followup_window_seconds + 59) // 60)
        cutoff = time.time() - self._followup_window_seconds

        try:
            entries = session_store.list_sessions(active_minutes=active_minutes)
        except Exception:
            return False

        for entry in entries:
            if getattr(entry, "session_key", None) != session_key:
                continue
            last_visible_reply_at = getattr(entry, "last_visible_reply_at", None)
            if last_visible_reply_at is None:
                continue
            try:
                if last_visible_reply_at.timestamp() >= cutoff:
                    return True
            except Exception:
                continue
        return False

    def _group_message_allowed(self, payload: Dict[str, Any]) -> bool:
        group_id = str(payload.get("group_id") or "").strip()
        return qq_group_message_allowed(
            group_id,
            allow_all_groups=bool(self.allow_all_groups),
            allowed_groups=set(self.allowed_groups or set()),
            has_policy=has_group_policy(group_id),
            overlay_active=bool(self._intel_group_overlay(group_id).get("active")),
        )

    def _intel_group_overlay(self, group_id: str) -> dict[str, Any]:
        normalized_group_id = str(group_id or "").strip()
        if not normalized_group_id:
            return {"active": False, "mode": "default", "archive_enabled": False, "daily_report_enabled": False, "workers": []}
        try:
            return get_group_monitoring_overlay(normalized_group_id)
        except Exception:
            logger.exception(
                "[%s] Failed to load QQ intel overlay for %s",
                self.name,
                normalized_group_id,
            )
            return {"active": False, "mode": "default", "archive_enabled": False, "daily_report_enabled": False, "workers": []}

    @staticmethod
    def _policy_has_runtime_override(policy: dict[str, Any]) -> bool:
        return qq_policy_has_runtime_override(policy)

    def _effective_group_policy(self, group_id: str) -> dict[str, Any]:
        return resolve_qq_effective_group_policy(
            group_id,
            policy_loader=get_group_policy,
            default_policy_loader=default_group_policy,
            overlay_loader=self._intel_group_overlay,
        )

    def _group_runs_project_mode(self, group_id: str) -> bool:
        if self._project_group_mode:
            return True
        policy = self._effective_group_policy(group_id)
        return str(policy.get("mode") or "").strip().lower() == "project_mode"

    async def _archive_group_payload(self, payload: Dict[str, Any]) -> None:
        try:
            await asyncio.to_thread(self._group_archive_store.archive_payload, dict(payload))
        except Exception:
            logger.exception(
                "[%s] Failed to archive QQ group message %s",
                self.name,
                payload.get("message_id"),
            )

    def _group_message_triggers_ai(self, payload: Dict[str, Any]) -> bool:
        return self._group_message_trigger_reason(payload) is not None

    def _message_has_nontext_media(self, payload: Dict[str, Any]) -> bool:
        segments = payload.get("message")
        if not isinstance(segments, list):
            return False
        for segment in segments:
            if str(segment.get("type") or "").lower() in {"image", "record", "video", "file"}:
                return True
        return False

    @staticmethod
    def _text_looks_like_request(text: str) -> bool:
        return text_looks_like_request(text)

    def _project_group_batch_should_dispatch(
        self,
        group_id: str,
        batch: list[_BufferedGroupMessage],
    ) -> tuple[bool, str]:
        normalized = [
            GroupBatchItem(
                speaker_id=str(item.event.source.user_id or "").strip(),
                text=str(item.event.text or "").strip(),
                direct_trigger_reason=self._group_message_trigger_reason(item.payload),
                is_admin=str(item.payload.get("user_id") or "").strip() in self._admin_users,
                has_nontext_media=self._message_has_nontext_media(item.payload),
            )
            for item in batch
        ]
        return decide_project_group_dispatch(
            normalized,
            thresholds=GroupDispatchThresholds(
                min_messages=self._group_trigger_min_messages,
                min_speakers=self._group_trigger_min_speakers,
                min_chars=self._group_trigger_min_chars,
            ),
        )

    def _should_process_group_message(self, payload: Dict[str, Any]) -> bool:
        return self._group_message_triggers_ai(payload)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not QQ_NAPCAT_AVAILABLE:
            logger.error("QQ NapCat: aiohttp is not installed")
            return False
        if not self.ws_url:
            logger.error("QQ NapCat: missing ws_url in platform config")
            return False

        await self.disconnect()

        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None)
            )
            await self._connect_websocket()
        except Exception as exc:
            await self.disconnect()
            self._record_connect_failure(exc)
            return False

        self._running = True
        self._mark_connected()
        self._reader_task = asyncio.create_task(self._reader_loop())
        logger.info("QQ NapCat connected to %s", self.ws_url)
        return True

    def _record_connect_failure(self, exc: Exception) -> None:
        diagnostic = diagnose_local_qq_napcat_endpoint(self.ws_url)
        if diagnostic:
            logger.error("QQ NapCat: %s", diagnostic["message"])
        else:
            logger.error("QQ NapCat: websocket connect failed: %s", exc)

        try:
            from gateway.status import write_runtime_status

            write_runtime_status(
                platform=self.platform.value,
                platform_state="unavailable" if diagnostic else "disconnected",
                error_code=(diagnostic or {}).get("code") or "qq_napcat_connect_failed",
                error_message=(diagnostic or {}).get("message") or str(exc),
            )
        except Exception:
            pass

    async def disconnect(self) -> None:
        self._running = False

        pending_batch_tasks = list(self._group_batch_tasks.values())
        for task in pending_batch_tasks:
            task.cancel()
        if pending_batch_tasks:
            await asyncio.gather(*pending_batch_tasks, return_exceptions=True)
        self._group_batch_tasks.clear()

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

        for future in self._pending_calls.values():
            if not future.done():
                future.cancel()
        self._pending_calls.clear()
        self._mark_disconnected()

    async def _connect_websocket(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None)
            )

        url = _with_access_token(self.ws_url, self.access_token)
        self._ws = await self._session.ws_connect(url, heartbeat=30)

    async def _reader_loop(self) -> None:
        while self._running:
            ws = self._ws
            if ws is None:
                try:
                    await self._connect_websocket()
                    self._mark_connected()
                    ws = self._ws
                except Exception as exc:
                    logger.warning(
                        "QQ NapCat reconnect failed, retrying in %ss: %s",
                        self.reconnect_interval,
                        exc,
                    )
                    await asyncio.sleep(self.reconnect_interval)
                    continue

            try:
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        if msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
                        continue

                    data = msg.json(loads=json.loads)
                    echo = data.get("echo")
                    if echo:
                        future = self._pending_calls.pop(str(echo), None)
                        if future and not future.done():
                            future.set_result(data)
                        continue

                    if data.get("post_type") in {"message", "request"}:
                        asyncio.create_task(self._handle_payload(data))

                if not self._running:
                    return
                self._ws = None
                await asyncio.sleep(self.reconnect_interval)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(
                    "QQ NapCat reader error, retrying in %ss: %s",
                    self.reconnect_interval,
                    exc,
                )
                self._ws = None
                await asyncio.sleep(self.reconnect_interval)

    async def _call_api(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("QQ NapCat websocket is not connected")

        echo = f"hermes-qq-napcat-{next(self._echo_counter)}"
        payload = {"action": action, "params": params, "echo": echo}

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_calls[echo] = future

        async with self._send_lock:
            await self._ws.send_json(payload)

        try:
            response = await asyncio.wait_for(future, timeout=30)
        finally:
            self._pending_calls.pop(echo, None)

        if response.get("status") != "ok" or response.get("retcode") not in (0, None):
            message = response.get("message") or "unknown error"
            raise RuntimeError(f"NapCat API error ({response.get('retcode')}): {message}")
        return response.get("data") or {}

    @staticmethod
    def _normalize_social_notify_target(target: str | None) -> str | None:
        text = str(target or "").strip()
        if not text:
            return None
        if text.startswith("qq_napcat:"):
            text = text.split(":", 1)[1]
        if text.startswith(("group:", "dm:")):
            return text
        return None

    @staticmethod
    def _worker_matches_request_group(worker: dict[str, Any], group_id: str) -> bool:
        normalized_group_id = str(group_id or "").strip()
        if not normalized_group_id:
            return False
        if str(worker.get("target_group_id") or "").strip() == normalized_group_id:
            return True
        target_ref = str(worker.get("target_group_ref") or "").strip()
        if target_ref.startswith("qq_napcat:group:"):
            target_ref = target_ref.split(":", 2)[2]
        elif target_ref.startswith("group:"):
            target_ref = target_ref.split(":", 1)[1]
        return target_ref == normalized_group_id

    async def _send_social_auto_notice(self, target: str | None, message: str) -> None:
        normalized_target = self._normalize_social_notify_target(target)
        if not normalized_target or not str(message or "").strip():
            return
        result = await self.send(normalized_target, message)
        if not result.success:
            logger.warning("[%s] Failed to send QQ social auto-handling notice: %s", self.name, result.error)

    async def _auto_handle_social_request(self, request: dict[str, Any]) -> None:
        request_key = str(request.get("request_key") or "").strip()
        if not request_key:
            return
        if str(request.get("status") or "pending").strip().lower() != "pending":
            return

        try:
            policy = await asyncio.to_thread(get_social_policy)
        except Exception:
            logger.exception("[%s] Failed to load QQ social auto-handling policy", self.name)
            return

        request_type = str(request.get("request_type") or "").strip().lower()
        sub_type = str(request.get("sub_type") or "add").strip().lower() or "add"
        group_id = str(request.get("group_id") or "").strip()
        note: str | None = None
        handled_by: str | None = None
        handled_via: str | None = None

        if request_type == "friend":
            if bool(policy.get("auto_approve_friend_requests")):
                note = "按社交自动处理策略自动通过好友请求。"
                handled_by = "qq_napcat:auto_social_policy"
                handled_via = "auto_social_policy"
        elif request_type == "group":
            if sub_type == "invite" and bool(policy.get("auto_approve_group_invites")):
                note = "按社交自动处理策略自动通过加群邀请。"
                handled_by = "qq_napcat:auto_social_policy"
                handled_via = "auto_social_policy"
            elif sub_type != "invite" and bool(policy.get("auto_approve_group_add_requests")):
                note = "按社交自动处理策略自动通过加群请求。"
                handled_by = "qq_napcat:auto_social_policy"
                handled_via = "auto_social_policy"
            elif group_id:
                try:
                    pending_workers = await asyncio.to_thread(list_intel_workers)
                except Exception:
                    logger.exception("[%s] Failed to inspect QQ intel workers for request auto-approval", self.name)
                    pending_workers = []
                for worker in pending_workers:
                    status = str(worker.get("status") or "").strip().lower()
                    if status not in {"awaiting_group_approval", "failed"}:
                        continue
                    if self._worker_matches_request_group(worker, group_id):
                        worker_name = str(worker.get("worker_name") or "情报员").strip() or "情报员"
                        note = f"匹配到{worker_name}的潜伏任务，自动通过加群请求。"
                        handled_by = "qq_napcat:auto_intel_worker"
                        handled_via = "auto_intel_worker"
                        break

        if not note:
            return

        try:
            if request_type == "group":
                await self._call_api(
                    "set_group_add_request",
                    {
                        "flag": str(request.get("flag") or "").strip(),
                        "sub_type": sub_type,
                        "approve": True,
                    },
                )
            elif request_type == "friend":
                await self._call_api(
                    "set_friend_add_request",
                    {
                        "flag": str(request.get("flag") or "").strip(),
                        "approve": True,
                    },
                )
            else:
                return
        except Exception as exc:
            logger.warning("[%s] QQ social auto-handling failed for %s: %s", self.name, request_key, exc)
            await self._send_social_auto_notice(
                policy.get("notify_target"),
                f"QQ 社交请求自动处理失败：{request_key}\n原因：{exc}",
            )
            return

        try:
            updated = await asyncio.to_thread(
                update_social_request_status,
                request_key,
                status="approved",
                handled_by=handled_by,
                handled_via=handled_via,
                note=note,
            )
        except Exception:
            logger.exception("[%s] Failed to persist QQ social auto-handling result for %s", self.name, request_key)
            updated = request

        request_label = "好友请求" if request_type == "friend" else "加群请求"
        summary = f"{request_label}已自动通过：{request_key}"
        requester = str(updated.get("user_id") or "").strip()
        if requester:
            summary += f"\n发起人：{requester}"
        if group_id:
            summary += f"\n群号：{group_id}"
        summary += f"\n当前状态：{updated.get('status') or 'approved'}"
        if handled_via:
            summary += f"\n处理来源：{handled_via}"
        summary += f"\n说明：{note}"
        await self._send_social_auto_notice(policy.get("notify_target"), summary)

    def _build_message_event(self, payload: Dict[str, Any]) -> MessageEvent:
        message_type = str(payload.get("message_type") or "private").lower()
        is_group = message_type == "group"
        chat_id = str(payload.get("group_id") if is_group else payload.get("user_id") or "")
        user_id = str(payload.get("user_id") or "")
        sender = payload.get("sender") or {}
        nickname = sender.get("nickname") or user_id
        user_name = sender.get("card") or nickname if is_group else nickname

        self._chat_types[chat_id] = "group" if is_group else "private"

        segments = payload.get("message")
        text = str(payload.get("raw_message") or "").strip()
        structured_text = self._segment_text_for_message(segments)
        if structured_text or "[CQ:" in text:
            text = structured_text

        normalized_type = MessageType.TEXT
        if isinstance(segments, list):
            seg_types = {str(segment.get("type") or "").lower() for segment in segments}
            if "record" in seg_types:
                normalized_type = MessageType.VOICE
            elif "video" in seg_types:
                normalized_type = MessageType.VIDEO
            elif "image" in seg_types:
                normalized_type = MessageType.PHOTO
            elif "file" in seg_types:
                normalized_type = MessageType.DOCUMENT

        source = self.build_source(
            chat_id=chat_id,
            chat_name=str(payload.get("group_id") or chat_id) if is_group else user_name,
            chat_type="group" if is_group else "dm",
            user_id=user_id or None,
            user_name=user_name or None,
        )
        metadata = None
        if is_group:
            trigger_reason = self._group_message_trigger_reason(payload)
            explicit_trigger_reason = ""
            if self._message_mentions_bot(payload):
                explicit_trigger_reason = "bot_mention"
            elif self._message_is_reply_to_bot(payload):
                explicit_trigger_reason = "reply_to_bot"
            elif self._message_matches_mention_patterns(payload):
                explicit_trigger_reason = "name_trigger"
            if trigger_reason:
                metadata = build_group_message_metadata(
                    trigger_reason=trigger_reason,
                    explicit_reason=explicit_trigger_reason,
                )
        return MessageEvent(
            text=text,
            message_type=normalized_type,
            source=source,
            raw_message=payload,
            message_id=str(payload.get("message_id")) if payload.get("message_id") is not None else None,
            metadata=metadata,
            reply_to_message_id=self._extract_reply_message_id(payload),
            timestamp=datetime.fromtimestamp(payload.get("time", time.time())),
        )

    def _record_successful_response_context(
        self,
        event: MessageEvent,
        sent_message_ids: list[str],
    ) -> None:
        metadata = getattr(event, "metadata", None) or {}
        if bool(metadata.get("skip_successful_response_context")):
            return
        source = getattr(event, "source", None)
        if not source or str(getattr(source, "chat_type", "")).lower() != "group":
            return

        group_id = str(getattr(source, "chat_id", "") or "").strip()
        user_id = str(getattr(source, "user_id", "") or "").strip()
        if not group_id or self._followup_window_seconds <= 0:
            return

        self._cleanup_group_tracking_state()
        expires_at = time.time() + self._followup_window_seconds
        if user_id:
            self._group_followup_windows[(group_id, user_id)] = expires_at
        if self._group_runs_project_mode(group_id):
            self._group_shared_followup_windows[group_id] = expires_at

        for message_id in sent_message_ids:
            normalized_message_id = str(message_id or "").strip()
            if normalized_message_id:
                self._recent_group_bot_messages[normalized_message_id] = (
                    group_id,
                    expires_at,
                )

        self._cleanup_group_tracking_state()

    def _remember_group_message(self, group_id: str, item: _BufferedGroupMessage) -> None:
        messages = self._group_observed_messages.setdefault(group_id, [])
        messages.append(item)
        if len(messages) > self._group_observed_max_messages:
            del messages[:-self._group_observed_max_messages]

    def _seed_group_batch(self, group_id: str) -> list[_BufferedGroupMessage]:
        since = self._group_last_included_at.get(group_id, 0.0)
        messages = self._group_observed_messages.get(group_id, [])
        seeded = [item for item in messages if item.observed_at > since]
        if not seeded and messages:
            seeded = [messages[-1]]
        return list(seeded)

    def _schedule_group_batch_flush(self, group_id: str) -> None:
        prior_task = self._group_batch_tasks.get(group_id)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._group_batch_tasks[group_id] = asyncio.create_task(
            self._flush_group_batch(group_id)
        )

    def _describe_group_message(self, item: _BufferedGroupMessage) -> str:
        text = str(item.event.text or "").strip()
        if text:
            return text

        segments = item.payload.get("message")
        if not isinstance(segments, list):
            return "[空消息]"

        labels = []
        for segment in segments:
            seg_type = str(segment.get("type") or "").lower()
            if seg_type == "image":
                labels.append("[图片]")
            elif seg_type == "record":
                labels.append("[语音]")
            elif seg_type == "video":
                labels.append("[视频]")
            elif seg_type == "file":
                labels.append("[文件]")
        return " ".join(labels) or "[空消息]"

    async def _build_group_batch_event(
        self,
        group_id: str,
        batch: list[_BufferedGroupMessage],
    ) -> MessageEvent:
        selected = batch[-self._group_batch_max_messages:]
        omitted_count = len(batch) - len(selected)
        latest = selected[-1]
        latest_user_id = str(latest.event.source.user_id or "").strip()
        latest_is_admin = bool(latest_user_id and latest_user_id in self._admin_users)
        batch_trigger_reason = ""
        batch_explicit_reason = ""
        for item in batch:
            item_metadata = getattr(item.event, "metadata", None) or {}
            item_trigger_reason = str(item_metadata.get("group_trigger_reason") or "").strip()
            item_explicit_reason = str(
                item_metadata.get("address_reason")
                or item_metadata.get("explicit_group_trigger_reason")
                or ""
            ).strip()
            if item_explicit_reason:
                batch_trigger_reason = item_trigger_reason or item_explicit_reason
                batch_explicit_reason = item_explicit_reason
                break
            if item_trigger_reason and not batch_trigger_reason:
                batch_trigger_reason = item_trigger_reason

        merged_lines = [f"[QQ项目群合并消息，共 {len(batch)} 条]"]
        if omitted_count > 0:
            merged_lines.append(f"[已截取最近 {len(selected)} 条，省略 {omitted_count} 条更早消息]")

        merged_media_urls: list[str] = []
        merged_media_types: list[str] = []
        merged_media_sources: list[str] = []
        for item in selected:
            if item.event.media_urls or item.event.message_type in {
                MessageType.PHOTO,
                MessageType.VIDEO,
                MessageType.VOICE,
                MessageType.DOCUMENT,
            }:
                await self._populate_media(item.event, item.payload)
                merged_media_urls.extend(list(item.event.media_urls or []))
                merged_media_types.extend(list(item.event.media_types or []))
                merged_media_sources.extend(
                    list(getattr(item.event, "media_sources", None) or [])
                )

            speaker = str(item.event.source.user_name or item.event.source.user_id or "unknown").strip()
            merged_lines.append(f"{speaker}: {self._describe_group_message(item)}")

        merged_metadata = dict(getattr(latest.event, "metadata", None) or {})
        trigger_metadata = build_group_message_metadata(
            trigger_reason=batch_trigger_reason or None,
            explicit_reason=batch_explicit_reason or None,
        )
        if trigger_metadata:
            merged_metadata.update(trigger_metadata)

        event = MessageEvent(
            text="\n".join(line for line in merged_lines if line),
            message_type=MessageType.TEXT,
            source=latest.event.source,
            raw_message={
                "qq_group_batch": True,
                "group_id": group_id,
                "message_ids": [item.event.message_id for item in selected if item.event.message_id],
                "latest_user_id": latest_user_id,
                "latest_is_admin": latest_is_admin,
            },
            message_id=latest.event.message_id,
            metadata=merged_metadata or None,
            media_urls=merged_media_urls,
            media_types=merged_media_types,
            reply_to_message_id=latest.event.reply_to_message_id,
            timestamp=latest.event.timestamp,
        )
        event.media_sources = merged_media_sources
        return event

    async def _flush_group_batch(self, group_id: str) -> None:
        current_task = asyncio.current_task()
        try:
            while True:
                batch = self._group_pending_batches.get(group_id)
                if not batch:
                    return

                last_observed_at = batch[-1].observed_at
                next_ready_at = max(
                    last_observed_at + self._group_batch_debounce_seconds,
                    self._group_last_dispatch_at.get(group_id, 0.0) + self._group_min_model_interval_seconds,
                )
                wait_seconds = max(0.0, next_ready_at - time.time())
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
                    continue

                merged_event = await self._build_group_batch_event(group_id, batch)
                session_key = self._session_key_for_source(merged_event.source)
                if session_key in self._active_sessions:
                    await asyncio.sleep(self._group_batch_retry_seconds)
                    continue

                policy = self._effective_group_policy(group_id)
                policy_mode = str(policy.get("mode") or "").strip().lower()
                if policy_mode in {"collect_only", "disabled"}:
                    logger.debug(
                        "[%s] Dropping buffered QQ group batch for %s (%d messages, mode=%s)",
                        self.name,
                        group_id,
                        len(batch),
                        policy_mode,
                    )
                    self._group_pending_batches.pop(group_id, None)
                    return

                should_dispatch, reason = self._project_group_batch_should_dispatch(
                    group_id, batch
                )
                if not should_dispatch:
                    logger.debug(
                        "[%s] Skipping low-signal QQ group batch for %s (%d messages, %s)",
                        self.name,
                        group_id,
                        len(batch),
                        reason,
                    )
                    self._group_pending_batches.pop(group_id, None)
                    return

                self._group_pending_batches.pop(group_id, None)
                self._group_last_dispatch_at[group_id] = time.time()
                self._group_last_included_at[group_id] = max(
                    item.observed_at for item in batch
                )
                logger.debug(
                    "[%s] Dispatching QQ group batch for %s (%d messages, %s)",
                    self.name,
                    group_id,
                    len(batch),
                    reason,
                )
                await self.handle_message(merged_event)
                return
        finally:
            if self._group_batch_tasks.get(group_id) is current_task:
                self._group_batch_tasks.pop(group_id, None)

    async def _handle_project_group_payload(self, payload: Dict[str, Any]) -> None:
        event = self._build_message_event(payload)
        event.text = self._clean_bot_mention_text(event.text, payload)

        raw_text = str(payload.get("raw_message") or "").strip()
        if raw_text.startswith("/"):
            await self._populate_media(event, payload)
            await self.handle_message(event)
            return

        group_id = str(payload.get("group_id") or "").strip()
        item = _BufferedGroupMessage(
            event=event,
            payload=payload,
            observed_at=time.time(),
        )
        self._remember_group_message(group_id, item)

        if group_id in self._group_pending_batches:
            self._group_pending_batches[group_id].append(item)
            self._schedule_group_batch_flush(group_id)
            return

        seed = self._seed_group_batch(group_id)
        if not seed:
            seed = [item]
        self._group_pending_batches[group_id] = seed
        self._schedule_group_batch_flush(group_id)

    async def _handle_payload(self, payload: Dict[str, Any]) -> None:
        post_type = str(payload.get("post_type") or "").strip().lower()
        if post_type == "request":
            try:
                request = await asyncio.to_thread(record_social_request_event, dict(payload))
            except Exception:
                logger.exception("QQ NapCat: failed to persist request payload")
            else:
                try:
                    await self._auto_handle_social_request(request)
                except Exception:
                    logger.exception("QQ NapCat: failed to auto-handle request payload")
            return

        if post_type != "message":
            return

        message_type = str(payload.get("message_type") or "").lower()
        if self._is_duplicate_inbound_message(payload):
            logger.debug(
                "[%s] Ignoring duplicate QQ %s message %s",
                self.name,
                message_type or "unknown",
                payload.get("message_id"),
            )
            return

        try:
            if message_type == "group":
                if not self._group_message_allowed(payload):
                    return
                group_id = str(payload.get("group_id") or "").strip()
                policy = self._effective_group_policy(group_id)
                policy_mode = str(policy.get("mode") or "").strip().lower()
                if policy_mode == "disabled":
                    logger.debug(
                        "[%s] Ignoring QQ group message for %s (reason=policy_disabled)",
                        self.name,
                        group_id,
                    )
                    return
                if bool(policy.get("archive_enabled")):
                    await self._archive_group_payload(payload)
                if policy_mode == "collect_only":
                    logger.debug(
                        "[%s] Archived QQ group message for %s without dispatch (reason=collect_only)",
                        self.name,
                        group_id,
                    )
                    return
                if self._group_runs_project_mode(group_id):
                    await self._handle_project_group_payload(payload)
                    return
                if not self._should_process_group_message(payload):
                    logger.debug(
                        "[%s] Ignoring QQ group message for %s (reason=no_trigger)",
                        self.name,
                        payload.get("group_id"),
                    )
                    return
            event = self._build_message_event(payload)
            if message_type == "group":
                event.text = self._clean_bot_mention_text(event.text, payload)
            await self._populate_media(event, payload)
            await self.handle_message(event)
        except Exception:
            logger.exception("QQ NapCat: failed to handle payload")

    async def _populate_media(self, event: MessageEvent, payload: Dict[str, Any]) -> None:
        segments = payload.get("message")
        if not isinstance(segments, list):
            return
        if not hasattr(event, "media_sources") or event.media_sources is None:
            event.media_sources = []

        for segment in segments:
            seg_type = str(segment.get("type") or "").lower()
            if seg_type not in {"image", "record", "video", "file"}:
                continue
            data = segment.get("data") or {}
            if seg_type == "image" and self._should_skip_media_segment(seg_type, data):
                logger.debug(
                    "QQ NapCat: skipping low-value %s segment %s",
                    seg_type,
                    str(data.get("url") or data.get("file") or "")[:160],
                )
                continue
            try:
                cached_path, mime_type = await self._cache_segment_media(seg_type, data)
            except Exception as exc:
                logger.debug("QQ NapCat: failed to cache %s segment: %s", seg_type, exc)
                continue
            if cached_path:
                if not hasattr(event, "media_sources") or event.media_sources is None:
                    event.media_sources = []
                preferred = self._preferred_media_source(seg_type, data, cached_path)
                event.media_urls.append(cached_path)
                event.media_types.append(mime_type)
                event.media_sources.append(preferred)

    @staticmethod
    def _should_skip_media_segment(seg_type: str, data: Dict[str, Any]) -> bool:
        if seg_type != "image":
            return False
        source_ref = str(data.get("url") or data.get("file") or "").strip()
        if not source_ref:
            return False
        return _looks_like_low_value_image_ref(source_ref)

    @staticmethod
    def _preferred_media_source(seg_type: str, data: Dict[str, Any], cached_path: str) -> str:
        """Return the best source string for later media analysis.

        For images, preserve the original remote URL whenever NapCat provides
        one so providers that reject ``data:image/...`` payloads can analyze
        the image directly. Local files and non-image media keep using the
        cached/local path.
        """
        if seg_type != "image":
            return cached_path
        source_ref = str(data.get("url") or data.get("file") or "").strip()
        if source_ref.startswith(("http://", "https://")):
            return source_ref
        return cached_path

    async def _cache_segment_media(self, seg_type: str, data: Dict[str, Any]) -> tuple[Optional[str], str]:
        url = str(data.get("url") or data.get("file") or "").strip()
        if not url:
            return None, "application/octet-stream"

        if url.startswith("file://"):
            local_path = unquote(urlsplit(url).path)
            if os.path.exists(local_path):
                return local_path, self._mime_for_segment(seg_type, local_path)
            return None, "application/octet-stream"

        if os.path.isabs(url) and os.path.exists(url):
            return url, self._mime_for_segment(seg_type, url)

        from tools.url_safety import is_safe_url

        if not is_safe_url(url):
            raise ValueError("unsafe media URL rejected")

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            raw = response.content

        if seg_type == "image":
            from gateway.platforms.base import cache_image_from_bytes

            ext = _guess_ext_from_name(url, ".jpg")
            image_path = cache_image_from_bytes(raw, ext=ext)
            return image_path, self._mime_for_segment(seg_type, image_path)

        if seg_type == "record":
            ext = _guess_ext_from_name(url, ".ogg")
            return cache_audio_from_bytes(raw, ext=ext), self._mime_for_segment(seg_type, ext)

        ext = _guess_ext_from_name(url, ".bin")
        filename = Path(_decoded_path_from_ref(url)).name or f"napcat{ext}"
        return cache_document_from_bytes(raw, filename), self._mime_for_segment(seg_type, filename)

    @staticmethod
    def _mime_for_segment(seg_type: str, value: str) -> str:
        suffix = Path(str(value)).suffix.lower()
        if seg_type == "image":
            return {
                ".gif": "image/gif",
                ".png": "image/png",
                ".webp": "image/webp",
            }.get(suffix, "image/jpeg")
        if seg_type == "record":
            return {
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".m4a": "audio/mp4",
            }.get(suffix, "audio/ogg")
        if seg_type == "video":
            return "video/mp4"
        return "application/octet-stream"

    @staticmethod
    def _local_file_uri(path: str) -> str:
        return Path(os.path.abspath(os.path.expanduser(path))).as_uri()

    def _resolve_target(self, chat_id: str) -> tuple[str, int]:
        if chat_id.startswith("group:"):
            return "group", int(chat_id.split(":", 1)[1])
        if chat_id.startswith("dm:"):
            return "private", int(chat_id.split(":", 1)[1])
        remembered = self._chat_types.get(str(chat_id), "private")
        return remembered, int(str(chat_id))

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ) -> SendResult:
        try:
            chat_type, numeric_id = self._resolve_target(chat_id)
            action = "send_group_msg" if chat_type == "group" else "send_private_msg"
            id_key = "group_id" if chat_type == "group" else "user_id"
            message = []
            if reply_to:
                message.append({"type": "reply", "data": {"id": str(reply_to)}})
            message.append({"type": "text", "data": {"text": self.format_message(content)}})
            data = await self._call_api(action, {id_key: numeric_id, "message": message})
            return SendResult(
                success=True,
                message_id=str(data.get("message_id")) if data.get("message_id") is not None else None,
                raw_response=data,
            )
        except Exception as exc:
            logger.warning(
                "[%s] QQ send failed (action=%s chat_type=%s reply=%s chars=%d): %s",
                self.name,
                action if "action" in locals() else "unknown",
                chat_type if "chat_type" in locals() else "unknown",
                bool(reply_to),
                len(str(content or "")),
                exc,
            )
            return SendResult(success=False, error=str(exc))

    async def _send_media(
        self,
        chat_id: str,
        segment_type: str,
        path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        try:
            if not os.path.exists(path):
                return SendResult(success=False, error=f"Media file not found: {path}")

            chat_type, numeric_id = self._resolve_target(chat_id)
            action = "send_group_msg" if chat_type == "group" else "send_private_msg"
            id_key = "group_id" if chat_type == "group" else "user_id"
            message = []
            if reply_to:
                message.append({"type": "reply", "data": {"id": str(reply_to)}})
            if caption:
                message.append({"type": "text", "data": {"text": caption}})
            message.append(
                {
                    "type": segment_type,
                    "data": {"file": self._local_file_uri(path)},
                }
            )
            data = await self._call_api(action, {id_key: numeric_id, "message": message})
            return SendResult(
                success=True,
                message_id=str(data.get("message_id")) if data.get("message_id") is not None else None,
                raw_response=data,
            )
        except Exception as exc:
            logger.warning(
                "[%s] QQ media send failed (segment=%s chat_id=%s reply=%s file=%s): %s",
                self.name,
                segment_type,
                chat_id,
                bool(reply_to),
                Path(str(path or "")).name or str(path or ""),
                exc,
            )
            return SendResult(success=False, error=str(exc))

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_media(chat_id, "image", image_path, caption=caption, reply_to=reply_to)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_media(chat_id, "record", audio_path, caption=caption, reply_to=reply_to)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_media(chat_id, "video", video_path, caption=caption, reply_to=reply_to)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del file_name
        return await self._send_media(chat_id, "file", file_path, caption=caption, reply_to=reply_to)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        del chat_id, metadata
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_type, _ = self._resolve_target(chat_id)
        return {"name": str(chat_id), "type": "group" if chat_type == "group" else "dm"}

    async def upload_group_file(
        self,
        group_id: str,
        file_path: str,
        folder_id: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        local_path = normalize_qq_napcat_local_path(file_path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Group file not found: {local_path}")

        params: Dict[str, Any] = {
            "group_id": resolve_qq_napcat_group_id(group_id),
            "file": local_path,
            "name": str(file_name or Path(local_path).name),
        }
        normalized_folder_id = str(folder_id or "").strip() or None
        if normalized_folder_id and normalized_folder_id != "/":
            params["folder"] = normalized_folder_id
        return await self._call_api("upload_group_file", params)

    async def get_group_root_files(self, group_id: str) -> Dict[str, Any]:
        return await self._call_api(
            "get_group_root_files",
            {"group_id": resolve_qq_napcat_group_id(group_id)},
        )

    async def get_group_files_by_folder(self, group_id: str, folder_id: str) -> Dict[str, Any]:
        if not str(folder_id or "").strip():
            raise ValueError("folder_id is required when listing a QQ group folder")
        return await self._call_api(
            "get_group_files_by_folder",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "folder_id": str(folder_id),
            },
        )

    async def delete_group_file(self, group_id: str, file_id: str, busid: int) -> Dict[str, Any]:
        if not str(file_id or "").strip():
            raise ValueError("file_id is required to delete a QQ group file")
        return await self._call_api(
            "delete_group_file",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "file_id": str(file_id),
                "busid": int(busid),
            },
        )

    async def create_group_file_folder(
        self,
        group_id: str,
        name: str,
        parent_id: str = "/",
    ) -> Dict[str, Any]:
        if not str(name or "").strip():
            raise ValueError("name is required to create a QQ group folder")
        parent = str(parent_id or "").strip() or "/"
        if parent != "/":
            raise ValueError("NapCat group folder creation currently supports only the root parent_id '/'")
        return await self._call_api(
            "create_group_file_folder",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "name": str(name),
                "parent_id": parent,
            },
        )

    async def delete_group_folder(self, group_id: str, folder_id: str) -> Dict[str, Any]:
        if not str(folder_id or "").strip():
            raise ValueError("folder_id is required to delete a QQ group folder")
        return await self._call_api(
            "delete_group_folder",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "folder_id": str(folder_id),
            },
        )

    async def get_group_file_system_info(self, group_id: str) -> Dict[str, Any]:
        return await self._call_api(
            "get_group_file_system_info",
            {"group_id": resolve_qq_napcat_group_id(group_id)},
        )

    async def get_group_file_url(self, group_id: str, file_id: str, busid: int) -> Dict[str, Any]:
        if not str(file_id or "").strip():
            raise ValueError("file_id is required to fetch a QQ group file URL")
        return await self._call_api(
            "get_group_file_url",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "file_id": str(file_id),
                "busid": int(busid),
            },
        )

    async def move_group_file(self, group_id: str, file_id: str, target_dir: str) -> Dict[str, Any]:
        if not str(file_id or "").strip():
            raise ValueError("file_id is required to move a QQ group file")
        if not str(target_dir or "").strip():
            raise ValueError("target_dir is required to move a QQ group file")
        return await self._call_api(
            "move_group_file",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "file_id": str(file_id),
                "target_dir": str(target_dir),
            },
        )

    async def rename_group_file(
        self,
        group_id: str,
        file_id: str,
        current_parent_directory: str,
        new_name: str,
    ) -> Dict[str, Any]:
        if not str(file_id or "").strip():
            raise ValueError("file_id is required to rename a QQ group file")
        if not str(current_parent_directory or "").strip():
            raise ValueError("current_parent_directory is required to rename a QQ group file")
        if not str(new_name or "").strip():
            raise ValueError("new_name is required to rename a QQ group file")
        return await self._call_api(
            "rename_group_file",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "file_id": str(file_id),
                "current_parent_directory": str(current_parent_directory),
                "new_name": str(new_name),
            },
        )

    async def trans_group_file(self, group_id: str, file_id: str, target_group_id: str) -> Dict[str, Any]:
        if not str(file_id or "").strip():
            raise ValueError("file_id is required to forward a QQ group file")
        return await self._call_api(
            "trans_group_file",
            {
                "group_id": resolve_qq_napcat_group_id(group_id),
                "file_id": str(file_id),
                "target_group_id": resolve_qq_napcat_group_id(target_group_id),
            },
        )
