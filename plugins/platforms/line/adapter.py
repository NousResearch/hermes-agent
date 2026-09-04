"""LINE Messaging API adapter: aiohttp webhook (signature-verified) → BasePlatformAdapter.

* Reply token preferred (free, single-use, ~60s TTL), metered Push as fallback.
* Slow-LLM postback button past ``slow_response_threshold`` (45s; 0 disables): the reply
  token is burned on a Template Buttons bubble; the tap yields a fresh free token that
  delivers the cached answer (PENDING → READY → DELIVERED, ERROR on cancel).
* Three allowlists (users U…, groups C…, rooms R…); ``LINE_ALLOW_ALL_USERS`` is dev-only.
* Media via public HTTPS only: local files served under ``/line/media/<token>/<name>``
  (allowed-roots guard); ``LINE_PUBLIC_URL`` overrides host:port behind tunnels/wildcards.
* ≤5 message objects per call; text chunked at 4500 chars (bubble hard limit 5000).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import enum
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import re
import secrets
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import quote as _urlquote

from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret
from gateway.platforms.base import (
    gateway_trust_env, BasePlatformAdapter, MessageEvent, MessageType, SendResult,
    cache_audio_from_bytes_async, cache_document_from_bytes_async, cache_image_from_bytes_async,
    cache_video_from_bytes_async)
from gateway.config import Platform

logger = logging.getLogger(__name__)

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
LINE_LOADING_URL = "https://api.line.me/v2/bot/chat/loading/start"
LINE_CONTENT_URL_FMT = "https://api-data.line.me/v2/bot/message/{message_id}/content"
LINE_BOT_INFO_URL = "https://api.line.me/v2/bot/info"
LINE_PER_BUBBLE_CHARS = 5000  # LINE hard limit
LINE_SAFE_BUBBLE_CHARS = 4500  # conservative chunking limit
LINE_MAX_MESSAGES_PER_CALL = 5
LINE_REPLY_TOKEN_TTL_SECONDS = 50  # below LINE's ~60s
WEBHOOK_BODY_MAX_BYTES = 1_048_576  # 1 MiB — webhooks are tiny JSON
DEFAULT_WEBHOOK_PORT = 8646
DEFAULT_WEBHOOK_PATH = "/line/webhook"
DEFAULT_MEDIA_PATH_PREFIX = "/line/media"
# ``None`` → asyncio binds BOTH address families (mirrors gateway/platforms/webhook.py).
# "0.0.0.0" is unreachable on IPv6-only networks (Fly.io 6PN → 502s); "::" breaks IPv4
# loopback probes on IPV6_V6ONLY=1 hosts. Pin via ``LINE_HOST`` / ``extra.host``.
DEFAULT_HOST = None
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})  # LINE can't fetch media from these → public URL required
DEFAULT_SLOW_RESPONSE_THRESHOLD = 45.0  # seconds; 0 disables the postback button
DEFAULT_PENDING_REPLY_TEXT = "🤔 Still thinking. Tap below to fetch the answer when it's ready."
DEFAULT_BUTTON_LABEL = "Get answer"
DEFAULT_DELIVERED_TEXT = "Already replied ✅"
DEFAULT_INTERRUPTED_TEXT = "Run was interrupted before completion."
MEDIA_TOKEN_TTL_SECONDS = 1800  # 30 minutes; LINE caches the URL aggressively
LINE_IMAGE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per LINE docs
LINE_AV_MAX_BYTES = 200 * 1024 * 1024  # 200 MB for voice/video
# LINE message type → normalized MessageType. LINE audio is recorded voice clips →
# VOICE (STT path), like Telegram/WhatsApp. Unknown types fall back to TEXT.
_LINE_MESSAGE_TYPES = {
    "text": MessageType.TEXT, "image": MessageType.PHOTO, "video": MessageType.VIDEO, "audio": MessageType.VOICE,
    "file": MessageType.DOCUMENT, "location": MessageType.LOCATION, "sticker": MessageType.STICKER}
# 1×1 transparent PNG: fallback video preview (LINE requires ``previewImageUrl``).
_FALLBACK_PNG_PREVIEW = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c63000100000005000100377a7ff20000000049454e"
    "44ae426082")
# Markdown LINE can't render, applied in order (code blocks first so their content survives).
_MD_STRIP_RULES: Tuple[Tuple[re.Pattern, Any], ...] = (
    (re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL), lambda m: m.group(1).rstrip("\n")),
    (re.compile(r"`([^`]+)`"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)"), lambda m: f"{m.group(1)} ({m.group(2)})"),
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)"), r"\1"),
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    (re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE), "• "))


def strip_markdown_preserving_urls(text: str) -> str:
    """Strip Markdown LINE can't render; ``[label](url)`` → ``label (url)`` keeps URLs
    tappable (LINE auto-links bare URLs only). Code-block content is kept.

    Source: PR #18153 (leepoweii) — adapted to keep code-block content visible (LINE users frequently want
    command snippets to land as plain text, not be eaten by the fence).
    """
    if not text:
        return text
    for pattern, repl in _MD_STRIP_RULES:
        text = pattern.sub(repl, text)
    return text


def split_for_line(text: str, max_chars: int = LINE_SAFE_BUBBLE_CHARS) -> List[str]:
    """Split into ≤5 LINE bubbles at paragraph/line/word breaks; overflow is ellipsised."""
    if not text or len(text) <= max_chars:
        return [text] if text else []
    chunks: List[str] = []
    remaining = text
    while remaining and len(chunks) < LINE_MAX_MESSAGES_PER_CALL and len(remaining) > max_chars:
        # Prefer paragraph, then line, then word breaks past the half-way mark; else a hard cut.
        cuts = [remaining.rfind(sep, 0, max_chars) for sep in ("\n\n", "\n", " ")]
        cut = next((c for c in cuts if c >= int(max_chars * 0.5)), cuts[-1])
        if cut <= 0:
            cut = max_chars
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining and len(chunks) < LINE_MAX_MESSAGES_PER_CALL:
        chunks.append(remaining)
    elif remaining:  # budget exhausted → ellipsis on the last bubble
        chunks[-1] = chunks[-1][: max_chars - 1].rstrip() + "…"
    return chunks


def verify_line_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    """Verify ``X-Line-Signature``: base64(HMAC-SHA256(secret, raw body)), constant-time."""
    if not signature or not channel_secret or body is None:
        return False
    try:
        digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
        expected = base64.b64encode(digest).decode("utf-8")
    except Exception:
        return False
    # Bytes: compare_digest raises TypeError on non-ASCII str, and the header is raw.
    return hmac.compare_digest(expected.encode(), signature.encode())


class State(enum.Enum):
    """Slow-LLM postback cache states."""

    PENDING = "pending"  # button sent, LLM still running
    READY = "ready"  # response cached, waiting for postback tap
    DELIVERED = "delivered"
    ERROR = "error"  # LLM raised / interrupted; error text cached


@dataclass
class _CacheEntry:
    state: State
    payload: Any = None


class RequestCache:
    """In-memory cache for slow-LLM postback retrieval (PENDING → READY|ERROR → DELIVERED).

    We keep the same model here. See #18153.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, _CacheEntry] = {}

    def register_pending(self, chat_id: str) -> str:
        rid = str(uuid.uuid4())
        self._entries[rid] = _CacheEntry(state=State.PENDING)
        return rid

    def get(self, request_id: str) -> Optional[_CacheEntry]:
        return self._entries.get(request_id)

    def _transition(self, request_id: str, allowed: Set[State], state: State, payload: Any = None) -> None:
        entry = self._entries.get(request_id)
        if entry is not None and entry.state in allowed:
            entry.state = state
            entry.payload = entry.payload if state is State.DELIVERED else payload

    def set_ready(self, request_id: str, payload: Any) -> None:
        self._transition(request_id, {State.PENDING}, State.READY, payload)

    def set_error(self, request_id: str, message: str) -> None:
        self._transition(request_id, {State.PENDING}, State.ERROR, message)

    def mark_delivered(self, request_id: str) -> None:
        self._transition(request_id, {State.READY, State.ERROR}, State.DELIVERED)


class _MessageDeduplicator:
    """Bounded LRU of LINE webhook event IDs to ignore at-least-once retries."""

    def __init__(self, max_size: int = 1000) -> None:
        self._seen: Dict[str, float] = {}
        self._max = max_size

    def is_duplicate(self, event_id: str) -> bool:
        if not event_id:
            return False
        if event_id in self._seen:
            return True
        if len(self._seen) >= self._max:  # drop the oldest 10% so we don't trim every insert
            cutoff = sorted(self._seen.values())[len(self._seen) // 10 or 1]
            self._seen = {k: v for k, v in self._seen.items() if v > cutoff}
        self._seen[event_id] = time.time()
        return False


# LINE source type → (id key, normalized chat_type)
_SOURCE_KINDS = {"group": ("groupId", "group"), "room": ("roomId", "room"), "user": ("userId", "dm")}


def _resolve_chat(source: Dict[str, Any]) -> Tuple[str, str]:
    """Return ``(chat_id, chat_type)`` from a LINE event ``source`` block (user/group/room).

    Source: PR #21023 (perng), unchanged.
    """
    kind = _SOURCE_KINDS.get((source or {}).get("type", ""))
    return ("", "dm") if kind is None else (source.get(kind[0], ""), kind[1])


def _allowed_for_source(
    source: Dict[str, Any], *, allow_all: bool, user_ids: Set[str], group_ids: Set[str], room_ids: Set[str]) -> bool:
    """Three-list gate: users, groups, rooms.

    See #18153.
    """
    if allow_all:
        return True
    sid, chat_type = _resolve_chat(source)
    return bool(sid) and sid in {"dm": user_ids, "group": group_ids, "room": room_ids}[chat_type]


class _LineClient:
    """Thin aiohttp wrapper around the LINE Messaging API (no ``line-bot-sdk`` dependency)."""

    def __init__(self, channel_access_token: str, *, timeout: float = 15.0) -> None:
        self._token = channel_access_token
        self._timeout = timeout
        self._headers = {"Authorization": f"Bearer {channel_access_token}", "Content-Type": "application/json"}

    @staticmethod
    def _session(timeout: float):
        import aiohttp
        return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout), trust_env=gateway_trust_env())

    async def _post_messages(self, url: str, label: str, payload: Dict[str, Any]) -> None:
        async with self._session(self._timeout) as session:
            async with session.post(url, headers=self._headers, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(f"LINE {label} {resp.status}: {body[:200]}")

    async def reply(self, reply_token: str, messages: List[Dict[str, Any]]) -> None:
        await self._post_messages(LINE_REPLY_URL, "reply", {"replyToken": reply_token, "messages": messages})

    async def push(self, chat_id: str, messages: List[Dict[str, Any]]) -> None:
        await self._post_messages(LINE_PUSH_URL, "push", {"to": chat_id, "messages": messages})

    async def loading(self, chat_id: str, seconds: int = 60) -> None:
        """Loading indicator (DM only). LINE rejects this for groups/rooms."""
        if not chat_id or not chat_id.startswith("U"):
            return
        import aiohttp  # noqa: F401 — ImportError must escape the swallow-all below
        clamped = max(5, min(60, (seconds // 5) * 5 or 5))  # LINE: 5-step increments, max 60
        try:
            async with self._session(5.0) as session:
                await session.post(LINE_LOADING_URL, headers=self._headers, json={"chatId": chat_id, "loadingSeconds": clamped})
        except Exception as exc:  # best-effort; never raise
            logger.debug("LINE loading indicator failed: %s", exc)

    async def fetch_content(self, message_id: str) -> bytes:
        async with self._session(30.0) as session:
            url = LINE_CONTENT_URL_FMT.format(message_id=message_id)
            async with session.get(url, headers={"Authorization": f"Bearer {self._token}"}) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"LINE content {resp.status}")
                return await resp.read()

    async def get_bot_user_id(self) -> Optional[str]:
        """Fetch this channel's own userId so we can filter self-messages."""
        import aiohttp  # noqa: F401 — ImportError must escape the swallow-all below
        try:
            async with self._session(10.0) as session:
                async with session.get(LINE_BOT_INFO_URL, headers=self._headers) as resp:
                    return None if resp.status >= 400 else (await resp.json()).get("userId")
        except Exception:
            return None


def _text_message(text: str) -> Dict[str, Any]:
    """Build a LINE text message object, capped to per-bubble max."""
    return {"type": "text", "text": text if len(text) <= LINE_PER_BUBBLE_CHARS else text[: LINE_PER_BUBBLE_CHARS - 1] + "…"}


def _text_messages(content: str) -> List[Dict[str, Any]]:
    """Markdown-strip, chunk and cap ``content`` into ≤5 LINE text messages."""
    chunks = split_for_line(strip_markdown_preserving_urls(content))
    return [_text_message(c) for c in chunks][:LINE_MAX_MESSAGES_PER_CALL]


def build_postback_button_message(text: str, button_label: str, request_id: str) -> Dict[str, Any]:
    """Slow-LLM postback bubble. Template Buttons stay tappable from history (Quick
    Reply chips vanish on the next message). LINE limits: text ≤160, altText ≤400.

    See #18153.
    """
    truncated = text if len(text) <= 160 else text[:157] + "..."
    alt = text if len(text) <= 400 else text[:397] + "..."
    action = {
        "type": "postback",
        "label": button_label[:20] or "Get answer",
        "data": json.dumps({"action": "show_response", "request_id": request_id}),
        "displayText": button_label[:300] or "Get answer"}
    return {"type": "template", "altText": alt, "template": {"type": "buttons", "text": truncated, "actions": [action]}}


# Gateway busy-ack prefixes (interrupting / queued / steered / background review);
# these bypass a PENDING postback cache so they land as visible bubbles.
_SYSTEM_BYPASS_PREFIXES: Tuple[str, ...] = ("⚡ Interrupting", "⏳ Queued", "⏩ Steered", "💾")


def _is_system_bypass(content: str) -> bool:
    return bool(content) and any(content.startswith(p) for p in _SYSTEM_BYPASS_PREFIXES)


def _csv_set(value: str) -> Set[str]:
    return {x.strip() for x in (value or "").split(",") if x.strip()}


def _truthy_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _credentials(config) -> Tuple[str, str]:
    """Return ``(channel_access_token, channel_secret)`` from scoped secrets, then ``extra``."""
    extra = getattr(config, "extra", {}) or {}
    return (
        _get_scoped_secret("LINE_CHANNEL_ACCESS_TOKEN") or extra.get("channel_access_token", ""),
        _get_scoped_secret("LINE_CHANNEL_SECRET") or extra.get("channel_secret", ""))


def _coerce(cast: Callable[[Any], Any], value: Any, default: Any) -> Any:
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


# Outbound media kinds → (size cap, size error, missing-public-URL error).
_OUTBOUND_MEDIA = {
    "image": (
        LINE_IMAGE_MAX_BYTES, "image exceeds 10 MB LINE limit",
        "LINE_PUBLIC_URL must be set to send images (LINE only accepts publicly reachable HTTPS URLs)"),
    "audio": (LINE_AV_MAX_BYTES, "audio exceeds 200 MB LINE limit", "LINE_PUBLIC_URL must be set to send audio"),
    "video": (LINE_AV_MAX_BYTES, "video exceeds 200 MB LINE limit", "LINE_PUBLIC_URL must be set to send video"),
}

# Inbound media kinds → cached file extension.
_INBOUND_MEDIA_EXT = {"image": ".jpg", "audio": ".m4a", "video": ".mp4", "file": ".bin"}
_INBOUND_AV_CACHERS = {"audio": cache_audio_from_bytes_async, "video": cache_video_from_bytes_async}
_LIFECYCLE_EVENTS = frozenset({"follow", "unfollow", "join", "leave"})
_ENV_SEED_KEYS = (("LINE_HOST", "host"), ("LINE_PUBLIC_URL", "public_url"), ("LINE_HOME_CHANNEL", "home_channel"))


class LineAdapter(BasePlatformAdapter):
    """LINE Messaging API gateway adapter (no message editing → REQUIRES_EDIT_FINALIZE stays False)."""

    def __init__(self, config, **kwargs):
        super().__init__(config=config, platform=Platform("line"))
        extra = getattr(config, "extra", {}) or {}

        def env_or(env: str, key: str, default: Any = "") -> Any:
            return os.getenv(env) or extra.get(key, default)

        def allowlist(env: str, key: str) -> Set[str]:
            return _csv_set(os.getenv(env, "")) | set(extra.get(key, []))

        self.channel_access_token, self.channel_secret = _credentials(config)
        # Host ``None`` → dual-stack bind (see DEFAULT_HOST); empty string collapses to None.
        self.webhook_host = env_or("LINE_HOST", "host", DEFAULT_HOST) or DEFAULT_HOST
        self.webhook_port = _coerce(int, env_or("LINE_PORT", "port", DEFAULT_WEBHOOK_PORT), DEFAULT_WEBHOOK_PORT)
        self.webhook_path = extra.get("webhook_path", DEFAULT_WEBHOOK_PATH)
        # Required for media when the bind isn't publicly reachable.
        self.public_base_url = (env_or("LINE_PUBLIC_URL", "public_url") or "").rstrip("/")
        self.allow_all = _truthy_env("LINE_ALLOW_ALL_USERS", bool(extra.get("allow_all_users", False)))
        self.allowed_users = allowlist("LINE_ALLOWED_USERS", "allowed_users")
        self.allowed_groups = allowlist("LINE_ALLOWED_GROUPS", "allowed_groups")
        self.allowed_rooms = allowlist("LINE_ALLOWED_ROOMS", "allowed_rooms")
        # Slow-LLM postback button threshold + user-overridable copy
        threshold = env_or("LINE_SLOW_RESPONSE_THRESHOLD", "slow_response_threshold", DEFAULT_SLOW_RESPONSE_THRESHOLD)
        self.slow_response_threshold = _coerce(float, threshold, DEFAULT_SLOW_RESPONSE_THRESHOLD)
        for attr, env, default in (
            ("pending_text", "LINE_PENDING_TEXT", DEFAULT_PENDING_REPLY_TEXT),
            ("button_label", "LINE_BUTTON_LABEL", DEFAULT_BUTTON_LABEL),
            ("delivered_text", "LINE_DELIVERED_TEXT", DEFAULT_DELIVERED_TEXT),
            ("interrupted_text", "LINE_INTERRUPTED_TEXT", DEFAULT_INTERRUPTED_TEXT)):
            setattr(self, attr, env_or(env, attr, default))
        # Runtime state
        self._client: Optional[_LineClient] = None
        self._app = self._runner = self._site = None  # aiohttp web.Application / AppRunner / TCPSite
        self._reply_tokens: Dict[str, Tuple[str, float]] = {}  # chat_id → (token, expiry)
        self._cache = RequestCache()
        self._dedup = _MessageDeduplicator()
        self._bot_user_id: Optional[str] = None
        self._lock_key: Optional[str] = None
        self._media_tokens: Dict[str, Tuple[str, float]] = {}  # token → (path, expiry)
        self._media_temp_paths: Set[str] = set()
        self._media_ttl = MEDIA_TOKEN_TTL_SECONDS
        self._pending_buttons: Dict[str, str] = {}  # one outstanding button per chat: chat_id → request_id

    def _fail(self, code: str, detail: str, *, retryable: bool = False) -> bool:  # fatal connect error → False
        self._set_fatal_error(code, detail, retryable=retryable)
        return False

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.channel_access_token or not self.channel_secret:
            return self._fail("config_missing", "LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET must be set")
        # One profile per channel token; lock on a hash so the secret never hits disk.
        try:
            from gateway.status import acquire_scoped_lock
            tok_hash = hashlib.sha256(self.channel_access_token.encode()).hexdigest()[:16]
            if not acquire_scoped_lock("line", tok_hash):
                return self._fail("lock_conflict", "LINE channel already in use by another profile")
            self._lock_key = tok_hash
        except ImportError:
            self._lock_key = None
        self._client = _LineClient(self.channel_access_token)
        try:  # best-effort self-userId for self-echo filtering (LINE rarely echoes anyway)
            self._bot_user_id = await self._client.get_bot_user_id()
        except Exception as exc:
            logger.debug("LINE: get_bot_user_id failed: %s", exc)
            self._bot_user_id = None
        try:
            from aiohttp import web
        except ImportError:
            return self._fail("missing_dep", "aiohttp is required for the LINE adapter — install with `pip install aiohttp`")
        self._app = web.Application(client_max_size=WEBHOOK_BODY_MAX_BYTES)
        self._app.router.add_post(self.webhook_path, self._handle_webhook)
        self._app.router.add_get(f"{self.webhook_path}/health", self._handle_health)  # tunnel/proxy probe
        self._app.router.add_get(f"{DEFAULT_MEDIA_PATH_PREFIX}/{{token}}/{{filename}}", self._handle_media)
        # Plugin-registered routes must be wired before AppRunner.setup() freezes the router.
        self._wire_plugin_handlers(self._app)
        self._runner = web.AppRunner(self._app)
        try:
            await self._runner.setup()
            # SO_REUSEADDR: on macOS/BSD two sockets with it can silently split traffic →
            # disable; on Linux it only allows rebinding past TIME_WAIT → keep default.
            self._site = web.TCPSite(
                self._runner, self.webhook_host, self.webhook_port,
                reuse_address=False if sys.platform == "darwin" else None)
            await self._site.start()
        except OSError as exc:
            return self._fail(
                "bind_failed",
                f"Could not bind LINE webhook on {self.webhook_host or 'all IPv4+IPv6 interfaces'}:"
                f"{self.webhook_port}: {exc}",
                retryable=True)
        self._mark_connected()
        logger.info(
            "LINE: webhook listening on %s:%s%s%s",
            self.webhook_host or "* (all interfaces, IPv4+IPv6)",
            self.webhook_port,
            self.webhook_path,
            f" (public: {self.public_base_url})" if self.public_base_url else "")
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        for attr, method in (("_site", "stop"), ("_runner", "cleanup")):
            obj = getattr(self, attr)
            if obj is not None:
                with contextlib.suppress(Exception):
                    await getattr(obj, method)()
                setattr(self, attr, None)
        self._app = None
        for path in list(self._media_temp_paths):
            _unlink_quietly(path)
        self._media_temp_paths.clear()
        self._media_tokens.clear()
        if self._lock_key:
            with contextlib.suppress(Exception):
                from gateway.status import release_scoped_lock
                release_scoped_lock("line", self._lock_key)
            self._lock_key = None

    async def _handle_health(self, request) -> Any:
        from aiohttp import web
        return web.json_response({"status": "ok", "platform": "line"})

    async def _handle_webhook(self, request) -> Any:
        from aiohttp import web
        try:  # explicit body cap: aiohttp's client_max_size only covers some body modes
            body = await request.read()
        except Exception as exc:
            logger.debug("LINE: read failed: %s", exc)
            return web.Response(status=400, text="bad request")
        if len(body) > WEBHOOK_BODY_MAX_BYTES:
            return web.Response(status=413, text="payload too large")
        if not verify_line_signature(body, request.headers.get("X-Line-Signature", ""), self.channel_secret):
            return web.Response(status=401, text="invalid signature")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return web.Response(status=400, text="bad json")
        for event in payload.get("events", []) or []:
            try:
                await self._dispatch_event(event)
            except Exception:
                logger.exception("LINE: dispatch_event failed")
        return web.Response(status=200, text="ok")

    async def _dispatch_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("type")
        source = event.get("source") or {}
        webhook_event_id = event.get("webhookEventId", "") or ""
        if webhook_event_id and self._dedup.is_duplicate(webhook_event_id):  # at-least-once redelivery
            logger.debug("LINE: ignoring duplicate webhook event %s", webhook_event_id)
            return
        if self._bot_user_id and source.get("userId", "") == self._bot_user_id:
            return
        if not _allowed_for_source(source, allow_all=self.allow_all, user_ids=self.allowed_users,
                                   group_ids=self.allowed_groups, room_ids=self.allowed_rooms):
            logger.info("LINE: rejecting unauthorized source %s", source)
            return
        if event_type == "message":
            await self._handle_message_event(event)
        elif event_type == "postback":
            await self._handle_postback_event(event)
        elif event_type in _LIFECYCLE_EVENTS:
            logger.info("LINE: lifecycle event %s from %s", event_type, source)
        else:
            logger.debug("LINE: ignoring event type %r", event_type)

    async def _handle_message_event(self, event: Dict[str, Any]) -> None:
        msg = event.get("message") or {}
        msg_type, message_id = msg.get("type", ""), msg.get("id", "")
        reply_token = event.get("replyToken", "")
        source = event.get("source") or {}
        chat_id, chat_type = _resolve_chat(source)
        user_id = source.get("userId", "") or chat_id
        if chat_id and reply_token:  # stash the reply token for outbound use
            self._reply_tokens[chat_id] = (reply_token, time.time() + LINE_REPLY_TOKEN_TTL_SECONDS)
        media_urls: List[str] = []
        media_types: List[str] = []
        if msg_type == "text":
            text = msg.get("text", "") or ""
        elif msg_type in _INBOUND_MEDIA_EXT:  # fetch, cache, surface a vision-friendly local path
            local_path, media_type = await self._download_media(
                message_id, msg_type, filename=msg.get("fileName") or msg.get("file_name"))
            if local_path:
                media_urls, media_types = [local_path], [media_type]
            text = f"[{msg_type}]"
        elif msg_type == "sticker":
            text = f"[sticker: {', '.join(msg['keywords'])}]" if msg.get("keywords") else "[sticker]"
        elif msg_type == "location":
            text = f"[location: {msg.get('title', '')} {msg.get('address', '')}]".strip()
        else:
            text = f"[unsupported message type: {msg_type}]"
        if chat_type == "dm" and self._client:  # best-effort typing indicator (DM only)
            asyncio.create_task(self._client.loading(chat_id))
        source_obj = self.build_source(
            chat_id=chat_id, chat_type=chat_type, user_id=user_id, user_name=user_id, chat_name=chat_id)
        await self.handle_message(MessageEvent(
            text=text, message_type=_LINE_MESSAGE_TYPES.get(msg_type, MessageType.TEXT), source=source_obj,
            raw_message=event, message_id=message_id, media_urls=media_urls, media_types=media_types))

    async def _handle_postback_event(self, event: Dict[str, Any]) -> None:
        """User tapped the slow-LLM postback button — deliver the cached payload. READY replies (push
        fallback) and ERROR replies settle the entry; DELIVERED / PENDING just re-issue their notice."""
        reply_token = event.get("replyToken", "")
        chat_id, _ = _resolve_chat(event.get("source") or {})
        try:
            parsed = json.loads((event.get("postback") or {}).get("data", "") or "")
        except (TypeError, json.JSONDecodeError):
            return
        request_id = parsed.get("request_id", "") if parsed.get("action") == "show_response" else ""
        entry = self._cache.get(request_id) if request_id else None
        if not self._client or not reply_token or not entry:
            return
        state = entry.state
        if state is State.READY:
            messages = _text_messages(str(entry.payload or ""))
        elif state is State.ERROR:
            messages = [_text_message(str(entry.payload or self.interrupted_text))]
        else:
            messages = [_text_message(self.delivered_text if state is State.DELIVERED else self.pending_text)]
        try:
            await self._client.reply(reply_token, messages)
        except Exception as exc:
            if state is not State.READY:
                if state is State.ERROR:
                    logger.warning("LINE: postback ERROR reply failed: %s", exc)
                return
            logger.warning("LINE: postback reply failed (%s); falling back to push", exc)
            try:
                await self._client.push(chat_id, messages)
            except Exception as exc2:
                logger.error("LINE: postback push fallback failed: %s", exc2)
                return
        if state in (State.READY, State.ERROR):
            self._cache.mark_delivered(request_id)
            self._pending_buttons.pop(chat_id, None)

    async def _download_media(
        self, message_id: str, msg_type: str, *, filename: Optional[str] = None) -> Tuple[Optional[str], str]:
        if not self._client or not message_id:
            return None, ""
        try:
            data = await self._client.fetch_content(message_id)
        except Exception as exc:
            logger.warning("LINE: failed to fetch %s content for %s: %s", msg_type, message_id, exc)
            return None, ""
        ext = _INBOUND_MEDIA_EXT.get(msg_type, ".bin")
        try:
            if msg_type == "image":
                return await cache_image_from_bytes_async(data, ext=ext), "image/jpeg"
            if msg_type in _INBOUND_AV_CACHERS:
                return await _INBOUND_AV_CACHERS[msg_type](data, ext=ext), mimetypes.guess_type(f"{msg_type}{ext}")[0] or f"{msg_type}/mp4"
            document_name = filename or f"line_file{ext}"
            mime = mimetypes.guess_type(document_name)[0] or "application/octet-stream"
            return await cache_document_from_bytes_async(data, document_name), mime
        except Exception as exc:
            logger.warning("LINE: failed to cache %s payload: %s", msg_type, exc)
            return None, ""

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="LINE adapter not connected")
        # A PENDING postback button caches the response for the tap — except system
        # busy-acks, which must land as visible bubbles.
        pending_rid = self._pending_buttons.get(chat_id)
        if pending_rid and not _is_system_bypass(content):
            self._cache.set_ready(pending_rid, content)
            return SendResult(success=True, message_id=pending_rid)
        # System busy-acks (interrupting / queued / steered) bypass the postback cache and route directly to
        # LINE so they reach the user as visible bubbles. Source: PR #18153.
        return await self._send_text_chunks(chat_id, content, force_push=False)

    async def _send_text_chunks(self, chat_id: str, content: str, *, force_push: bool) -> SendResult:
        return await self._send_messages(chat_id, _text_messages(content), force_push=force_push, text=True)

    def _consume_reply_token(self, chat_id: str) -> Tuple[str, bool]:
        """Consume a stashed reply token if present and unexpired → ``(token, used_reply)``."""
        token, expires_at = self._reply_tokens.pop(chat_id, None) or ("", 0.0)
        return (token, True) if token and time.time() < expires_at else ("", False)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Trigger LINE's loading-animation indicator (DM only)."""
        if self._client and chat_id:
            await self._client.loading(chat_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Best-effort chat info inferred from the ID prefix (U=user, C=group, R=room)."""
        chat_type = {"U": "dm", "C": "group", "R": "channel"}.get((chat_id or "")[:1], "dm")
        return {"name": chat_id or "", "type": chat_type}

    def format_message(self, content: str) -> str:
        """Strip Markdown that LINE can't render. URLs are preserved."""
        return strip_markdown_preserving_urls(content)

    async def _keep_typing(self, chat_id: str, *args, **kwargs) -> None:
        """Wrap the base typing heartbeat; fire the slow-LLM postback button at threshold."""
        if self.slow_response_threshold <= 0 or not self._client or not chat_id:
            await super()._keep_typing(chat_id, *args, **kwargs)
            return

        async def _fire_postback() -> None:
            await asyncio.sleep(self.slow_response_threshold)
            # Only fire while a usable reply token remains (the agent responding
            # consumes it) and no button is already outstanding.
            if chat_id not in self._reply_tokens or chat_id in self._pending_buttons:
                return
            rid = self._cache.register_pending(chat_id)
            self._pending_buttons[chat_id] = rid
            token, used = self._consume_reply_token(chat_id)
            if not used:
                self._pending_buttons.pop(chat_id, None)
                return
            msg = build_postback_button_message(self.pending_text, self.button_label, rid)
            try:
                await self._client.reply(token, [msg])
                logger.info("LINE: sent slow-LLM postback button for chat %s (rid=%s)", chat_id, rid)
            except Exception as exc:
                logger.warning("LINE: postback button send failed: %s", exc)
                self._pending_buttons.pop(chat_id, None)

        post_task = asyncio.create_task(_fire_postback())
        try:
            await super()._keep_typing(chat_id, *args, **kwargs)
        finally:
            if not post_task.done():
                post_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await post_task

    async def interrupt_session_activity(self, session_key: str, chat_id: str) -> None:
        """Resolve any orphan PENDING postback so the button doesn't loop."""
        await super().interrupt_session_activity(session_key, chat_id)
        rid = self._pending_buttons.pop(chat_id, None)
        if rid:
            self._cache.set_error(rid, self.interrupted_text)

    def _register_media(self, file_path: str, *, cleanup: bool = False) -> str:
        """Register a local file for HTTPS serving (evicting expired tokens); return the URL token."""
        now = time.time()
        for token, (path, exp) in list(self._media_tokens.items()):
            if now > exp:
                self._media_tokens.pop(token, None)
                if path in self._media_temp_paths:
                    self._media_temp_paths.discard(path)
                    _unlink_quietly(path)
        resolved = str(Path(file_path).resolve())
        token = secrets.token_urlsafe(32)
        self._media_tokens[token] = (resolved, now + self._media_ttl)
        if cleanup:
            self._media_temp_paths.add(resolved)
        return token

    def _media_url(self, token: str, filename: str) -> str:
        if self.public_base_url:
            base = self.public_base_url
        else:
            # Wildcard/dual-stack binds have no fetchable hostname (the _missing_public_url
            # guard should have fired); fall back to localhost so the URL is well-formed.
            host = "127.0.0.1" if self._missing_public_url() else self.webhook_host
            base = f"https://{host}" if self.webhook_port == 443 else f"https://{host}:{self.webhook_port}"
        return f"{base}{DEFAULT_MEDIA_PATH_PREFIX}/{token}/{_urlquote(filename, safe='')}"

    def _serve_file(self, path: Path) -> str:
        return self._media_url(self._register_media(str(path.resolve())), path.name)

    def _missing_public_url(self) -> bool:
        """True when no LINE_PUBLIC_URL is set and the bind host is wildcard/dual-stack ``None``."""
        return not self.public_base_url and (self.webhook_host is None or self.webhook_host in _WILDCARD_HOSTS)

    def _check_media_file(self, kind: str, file_path: str) -> Tuple[Optional[Path], Optional[SendResult]]:
        """Shared preflight for send_image_file/send_voice/send_video → ``(path, error)``."""
        max_bytes, size_error, url_error = _OUTBOUND_MEDIA[kind]
        path = Path(file_path)
        if not path.is_file():
            return None, SendResult(success=False, error=f"{kind} file not found: {file_path}")
        for failed, error in (
            (path.stat().st_size > max_bytes, size_error),
            (not self._client, "LINE adapter not connected"),
            (self._missing_public_url(), url_error)):
            if failed:
                return None, SendResult(success=False, error=error)
        return path, None

    async def _handle_media(self, request) -> Any:
        """Serve a registered local file for LINE's media URLs. Defence-in-depth: the resolved
        path is rechecked against allowed roots (tempdir, ``/tmp``→``/private/tmp`` on macOS, HERMES_HOME).

        Defence-in-depth: even though ``_register_media`` is only called from trusted internal code, we
        recheck the resolved path against an allowed-roots set before serving. Sources allowed:
        ``tempfile.gettempdir()``, ``/tmp`` (which resolves to ``/private/tmp`` on macOS), and
        ``HERMES_HOME``. PR #8398.
        """
        from aiohttp import web
        token = request.match_info["token"]
        file_path, expires_at = self._media_tokens.get(token) or ("", 0.0)
        if not file_path:
            return web.Response(status=404, text="not found")
        if time.time() > expires_at:
            self._media_tokens.pop(token, None)
            return web.Response(status=410, text="gone")
        path = Path(file_path)
        if not path.is_file():
            return web.Response(status=404, text="not found")
        try:
            from hermes_constants import get_hermes_home
            hermes_home = Path(get_hermes_home()).resolve()
        except Exception:
            hermes_home = Path.home().joinpath(".hermes").resolve()
        resolved = path.resolve()
        if not any(resolved.is_relative_to(r) for r in (Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve(), hermes_home)):
            logger.warning("LINE: refusing to serve outside allowed roots: %s", resolved)
            return web.Response(status=403, text="forbidden")
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return web.FileResponse(path, headers={"Content-Type": content_type})

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        path, err = self._check_media_file("image", image_path)
        if err:
            return err
        url = self._serve_file(path)
        if not url.lower().startswith("https://"):
            return SendResult(success=False, error=f"LINE image URL must be HTTPS: {url}")
        msgs: List[Dict[str, Any]] = [{"type": "image", "originalContentUrl": url, "previewImageUrl": url}]
        return await self._send_messages(chat_id, msgs + ([_text_message(caption)] if caption else []))

    async def send_voice(
        self, chat_id: str, audio_path: str, duration_ms: int = 1000, metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        path, err = self._check_media_file("audio", audio_path)
        if err:
            return err
        msg = {"type": "audio", "originalContentUrl": self._serve_file(path), "duration": int(duration_ms)}
        return await self._send_messages(chat_id, [msg])

    async def send_video(
        self, chat_id: str, video_path: str, preview_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        path, err = self._check_media_file("video", video_path)
        if err:
            return err
        # LINE requires previewImageUrl: use the supplied preview, else a stdlib 1×1 PNG.
        # Use one if supplied, otherwise write a stdlib 1×1 PNG to /tmp and serve it. PR #8398.
        if preview_path and Path(preview_path).is_file():
            preview_url = self._serve_file(Path(preview_path))
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            try:
                tmp.write(_FALLBACK_PNG_PREVIEW)
                tmp.flush()
                tmp.close()
                preview_url = self._media_url(self._register_media(tmp.name, cleanup=True), "preview.png")
            except Exception:
                _unlink_quietly(tmp.name)
                raise
        msg = {"type": "video", "originalContentUrl": self._serve_file(path), "previewImageUrl": preview_url}
        return await self._send_messages(chat_id, [msg])

    async def _send_messages(
        self, chat_id: str, messages: List[Dict[str, Any]], *, force_push: bool = False, text: bool = False
    ) -> SendResult:
        """Send built message objects, batched at 5/call: reply token first, then push. ``text``
        selects the text contract: reply success reports the token as message_id; push failure logs at error."""
        if not self._client:
            return SendResult(success=False, error="LINE adapter not connected")
        if not messages:
            return SendResult(success=True, message_id=None)
        n = LINE_MAX_MESSAGES_PER_CALL
        batches = [messages[i:i + n] for i in range(0, len(messages), n)]
        token, used_reply = self._consume_reply_token(chat_id)
        start = 0
        if used_reply and not force_push:
            try:
                await self._client.reply(token, batches[0])
                if text:
                    return SendResult(success=True, message_id=token)
                start = 1
            except Exception as exc:
                logger.info("LINE: reply token rejected (%s); falling back to push", exc)
        for i in range(start, len(batches)):  # push the rest (reply token is single-use)
            try:
                await self._client.push(chat_id, batches[i])
            except Exception as exc:
                if i > 0:
                    logger.warning("LINE: push for follow-up batch failed: %s", exc)
                elif text:
                    logger.error("LINE: push send failed: %s", exc)
                return SendResult(success=False, error=str(exc))
        return SendResult(success=True, message_id=None)


def _unlink_quietly(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def _env_credentials_present() -> bool:
    return bool(_get_scoped_secret("LINE_CHANNEL_ACCESS_TOKEN") and _get_scoped_secret("LINE_CHANNEL_SECRET"))


def check_requirements() -> bool:
    """Plugin gate: require credentials AND aiohttp at runtime."""
    if not _env_credentials_present():
        return False
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        return False


def validate_config(config) -> bool:
    return all(_credentials(config))


def is_connected(config) -> bool:
    """Surface in ``hermes status`` even before the adapter is instantiated."""
    return validate_config(config)


def _env_enablement() -> Optional[Dict[str, Any]]:
    """Seed PlatformConfig.extra from env-only setups so ``hermes status`` sees them."""
    if not _env_credentials_present():
        return None
    seeded: Dict[str, Any] = {}
    if os.getenv("LINE_PORT"):
        with contextlib.suppress(ValueError):
            seeded["port"] = int(os.environ["LINE_PORT"])
    seeded.update({key: os.environ[env] for env, key in _ENV_SEED_KEYS if os.getenv(env)})
    return seeded


async def _standalone_send(
    pconfig, chat_id: str, message: str, *,
    thread_id: Optional[str] = None, media_files: Optional[List[str]] = None, force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process Push delivery for cron jobs detached from the gateway (no inbound event → no
    reply token). ``thread_id`` is ignored (no threads); ``media_files`` need the webhook server."""
    extra = getattr(pconfig, "extra", {}) or {}
    token = _get_scoped_secret("LINE_CHANNEL_ACCESS_TOKEN") or extra.get("channel_access_token", "")
    if not token or not chat_id:
        return {"error": "LINE standalone send: missing token or chat_id"}
    messages = _text_messages(message or "") or [_text_message("")]
    if media_files:  # tell the recipient media was generated but not delivered
        messages.append(_text_message(f"[{len(media_files)} attachment(s) generated; not deliverable from cron]"))
        messages = messages[:LINE_MAX_MESSAGES_PER_CALL]
    try:
        await _LineClient(token).push(chat_id, messages)
        return {"success": True, "message_id": None}
    except Exception as exc:
        return {"error": str(exc)}


_SETUP_PROMPTS = (  # (env var, prompt, masked)
    ("LINE_CHANNEL_ACCESS_TOKEN", "Channel access token", True),
    ("LINE_CHANNEL_SECRET", "Channel secret", True),
    ("LINE_PUBLIC_URL", "Public HTTPS base URL (optional, e.g. https://my-tunnel.example.com)", False),
    ("LINE_ALLOWED_USERS", "Allowed user IDs (comma-separated; blank=skip)", False))


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes setup line`` (writes ``~/.hermes/.env``)."""
    print("\nLINE Messaging API setup\n------------------------\n"
          "Create a Messaging API channel at https://developers.line.biz/console/\nthen copy the values below.\n")
    try:
        from hermes_cli.config import get_env_value as _get_env, save_env_value as _set_env
    except ImportError:
        print("hermes_cli.config not available; set LINE_* vars manually in ~/.hermes/.env")
        return

    for var, prompt, secret in _SETUP_PROMPTS:
        existing = _get_env(var) if callable(_get_env) else None
        suffix = " [keep current]" if existing else ""
        try:
            if secret:
                from hermes_cli.secret_prompt import masked_secret_prompt
                value = masked_secret_prompt(f"{prompt}{suffix}: ")
            else:
                value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            continue
        if value:
            _set_env(var, value)
    print("Done. Set the webhook URL in the LINE console to <your-public-url>/line/webhook and enable 'Use webhook'.")


def register(ctx) -> None:
    ctx.register_platform(
        name="line", label="LINE", adapter_factory=lambda cfg: LineAdapter(cfg), check_fn=check_requirements,
        validate_config=validate_config, is_connected=is_connected,
        required_env=["LINE_CHANNEL_ACCESS_TOKEN", "LINE_CHANNEL_SECRET"], install_hint="pip install aiohttp",
        setup_fn=interactive_setup, env_enablement_fn=_env_enablement, cron_deliver_env_var="LINE_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send, allowed_users_env="LINE_ALLOWED_USERS",
        allow_all_env="LINE_ALLOW_ALL_USERS",
        max_message_length=LINE_SAFE_BUBBLE_CHARS,  # per-bubble cap is 5000; smart-chunker uses 4500
        emoji="💚", pii_safe=False, allow_update_command=True,
        platform_hint=(
            "You are chatting via LINE Messaging API. LINE does NOT render "
            "Markdown — text bubbles show ** and # literally. Bare URLs are "
            "auto-linked, but \\[label\\](url) syntax is not. Each text bubble "
            "is capped at 5000 characters and at most 5 bubbles are sent per "
            "reply, so keep responses concise. Image/audio/video sending "
            "requires LINE_PUBLIC_URL configured to a publicly reachable HTTPS "
            "host. Slow responses surface a 'Get answer' button the user taps "
            "to fetch the reply via a fresh free token."))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from dataclasses import field  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
