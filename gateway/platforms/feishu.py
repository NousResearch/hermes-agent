"""
Feishu/Lark platform adapter.

Supports:
- WebSocket long connection and Webhook transport
- Direct-message and group @mention-gated text receive/send
- Inbound image/file/audio/media caching
- Gateway allowlist integration via FEISHU_ALLOWED_USERS
- Persistent dedup state across restarts
- Per-chat serial message processing (matches openclaw createChatQueue)
- Persistent ACK emoji reaction on inbound messages
- Reaction events routed as synthetic text events (matches openclaw)
- Interactive card button-click events routed as synthetic COMMAND events
- Webhook anomaly tracking (matches openclaw createWebhookAnomalyTracker)
- Verification token validation as second auth layer (matches openclaw)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import itertools
import json
import logging
import mimetypes
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

# aiohttp/websockets are independent optional deps — import outside lark_oapi
# so they remain available for tests and webhook mode even if lark_oapi is missing.
try:
    import aiohttp
    from aiohttp import web
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

try:
    import lark_oapi as lark
    from lark_oapi.api.application.v6 import GetApplicationRequest
    from lark_oapi.api.im.v1 import (
        CreateFileRequest,
        CreateFileRequestBody,
        CreateImageRequest,
        CreateImageRequestBody,
        CreateMessageRequest,
        CreateMessageRequestBody,
        GetChatRequest,
        GetMessageRequest,
        GetMessageResourceRequest,
        P2ImMessageMessageReadV1,
        ReplyMessageRequest,
        ReplyMessageRequestBody,
        UpdateMessageRequest,
        UpdateMessageRequestBody,
    )
    from lark_oapi.core.const import FEISHU_DOMAIN, LARK_DOMAIN
    from lark_oapi.event.callback.model.p2_card_action_trigger import P2CardActionTriggerResponse
    from lark_oapi.event.dispatcher_handler import EventDispatcherHandler
    from lark_oapi.ws import Client as FeishuWSClient

    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None  # type: ignore[assignment]
    P2CardActionTriggerResponse = None  # type: ignore[assignment]
    EventDispatcherHandler = None  # type: ignore[assignment]
    FeishuWSClient = None  # type: ignore[assignment]
    FEISHU_DOMAIN = None  # type: ignore[assignment]
    LARK_DOMAIN = None  # type: ignore[assignment]

FEISHU_WEBSOCKET_AVAILABLE = websockets is not None
FEISHU_WEBHOOK_AVAILABLE = aiohttp is not None

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SUPPORTED_DOCUMENT_TYPES,
    cache_document_from_bytes,
    cache_image_from_url,
    cache_audio_from_bytes,
    cache_image_from_bytes,
)
from gateway.status import acquire_scoped_lock, release_scoped_lock
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_MARKDOWN_HINT_RE = re.compile(
    r"(^#{1,6}\s)|(^\s*[-*]\s)|(^\s*\d+\.\s)|(^\s*---+\s*$)|(```)|(`[^`\n]+`)|(\*\*[^*\n].+?\*\*)|(~~[^~\n].+?~~)|(<u>.+?</u>)|(\*[^*\n]+\*)|(\[[^\]]+\]\([^)]+\))|(^>\s)",
    re.MULTILINE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MENTION_RE = re.compile(r"@_user_\d+")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_POST_CONTENT_INVALID_RE = re.compile(r"content format of the post type is incorrect", re.IGNORECASE)
# ---------------------------------------------------------------------------
# Media type sets and upload constants
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
_AUDIO_EXTENSIONS = {".ogg", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".opus", ".webm"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}
_DOCUMENT_MIME_TO_EXT = {mime: ext for ext, mime in SUPPORTED_DOCUMENT_TYPES.items()}
_FEISHU_IMAGE_UPLOAD_TYPE = "message"
_FEISHU_FILE_UPLOAD_TYPE = "stream"
_FEISHU_OPUS_UPLOAD_EXTENSIONS = {".ogg", ".opus"}
_FEISHU_MEDIA_UPLOAD_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v"}
_FEISHU_DOC_UPLOAD_TYPES = {
    ".pdf": "pdf",
    ".doc": "doc",
    ".docx": "doc",
    ".xls": "xls",
    ".xlsx": "xls",
    ".ppt": "ppt",
    ".pptx": "ppt",
}
# ---------------------------------------------------------------------------
# Connection, retry and batching tuning
# ---------------------------------------------------------------------------

_MAX_TEXT_INJECT_BYTES = 100 * 1024
_FEISHU_CONNECT_ATTEMPTS = 3
_FEISHU_SEND_ATTEMPTS = 3
_FEISHU_APP_LOCK_SCOPE = "feishu-app-id"
_DEFAULT_TEXT_BATCH_DELAY_SECONDS = 0.6
_DEFAULT_TEXT_BATCH_MAX_MESSAGES = 8
_DEFAULT_TEXT_BATCH_MAX_CHARS = 4000
_DEFAULT_MEDIA_BATCH_DELAY_SECONDS = 0.8
_DEFAULT_DEDUP_CACHE_SIZE = 2048
_DEFAULT_WEBHOOK_HOST = "127.0.0.1"
_DEFAULT_WEBHOOK_PORT = 8765
_DEFAULT_WEBHOOK_PATH = "/feishu/webhook"
# ---------------------------------------------------------------------------
# TTL, rate-limit and webhook security constants
# ---------------------------------------------------------------------------

_FEISHU_DEDUP_TTL_SECONDS = 24 * 60 * 60          # 24 hours — matches openclaw
_FEISHU_SENDER_NAME_TTL_SECONDS = 10 * 60          # 10 minutes sender-name cache
_FEISHU_WEBHOOK_MAX_BODY_BYTES = 1 * 1024 * 1024   # 1 MB body limit
_FEISHU_WEBHOOK_RATE_WINDOW_SECONDS = 60            # sliding window for rate limiter
_FEISHU_WEBHOOK_RATE_LIMIT_MAX = 120               # max requests per window per IP — matches openclaw
_FEISHU_WEBHOOK_RATE_MAX_KEYS = 4096               # max tracked keys (prevents unbounded growth)
_FEISHU_WEBHOOK_BODY_TIMEOUT_SECONDS = 30          # max seconds to read request body
_FEISHU_WEBHOOK_ANOMALY_THRESHOLD = 25             # consecutive error responses before WARNING log
_FEISHU_WEBHOOK_ANOMALY_TTL_SECONDS = 6 * 60 * 60  # anomaly tracker TTL (6 hours) — matches openclaw
_FEISHU_CARD_ACTION_DEDUP_TTL_SECONDS = 15 * 60    # card action token dedup window (15 min)
_FEISHU_BOT_MSG_TRACK_SIZE = 512                   # LRU size for tracking sent message IDs
_FEISHU_REPLY_FALLBACK_CODES = frozenset({230011, 231003})  # reply target withdrawn/missing → create fallback
_FEISHU_ACK_EMOJI = "OK"
# ---------------------------------------------------------------------------
# Fallback display strings
# ---------------------------------------------------------------------------

FALLBACK_POST_TEXT = "[Rich text message]"
FALLBACK_FORWARD_TEXT = "[Merged forward message]"
FALLBACK_SHARE_CHAT_TEXT = "[Shared chat]"
FALLBACK_INTERACTIVE_TEXT = "[Interactive message]"
FALLBACK_IMAGE_TEXT = "[Image]"
FALLBACK_ATTACHMENT_TEXT = "[Attachment]"
# ---------------------------------------------------------------------------
# Post/card parsing helpers
# ---------------------------------------------------------------------------

_PREFERRED_LOCALES = ("zh_cn", "en_us")
_MARKDOWN_SPECIAL_CHARS_RE = re.compile(r"([\\`*_{}\[\]()#+\-!|>~])")
_MENTION_PLACEHOLDER_RE = re.compile(r"@_user_\d+")
_WHITESPACE_RE = re.compile(r"\s+")
_SUPPORTED_CARD_TEXT_KEYS = (
    "title",
    "text",
    "content",
    "label",
    "value",
    "name",
    "summary",
    "subtitle",
    "description",
    "placeholder",
    "hint",
)
_SKIP_TEXT_KEYS = {
    "tag",
    "type",
    "msg_type",
    "message_type",
    "chat_id",
    "open_chat_id",
    "share_chat_id",
    "file_key",
    "image_key",
    "user_id",
    "open_id",
    "union_id",
    "url",
    "href",
    "link",
    "token",
    "template",
    "locale",
}


@dataclass(frozen=True)
class FeishuPostMediaRef:
    file_key: str
    file_name: str = ""
    resource_type: str = "file"


@dataclass(frozen=True)
class FeishuPostParseResult:
    text_content: str
    image_keys: List[str] = field(default_factory=list)
    media_refs: List[FeishuPostMediaRef] = field(default_factory=list)
    mentioned_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeishuNormalizedMessage:
    raw_type: str
    text_content: str
    preferred_message_type: str = "text"
    image_keys: List[str] = field(default_factory=list)
    media_refs: List[FeishuPostMediaRef] = field(default_factory=list)
    mentioned_ids: List[str] = field(default_factory=list)
    relation_kind: str = "plain"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeishuAdapterSettings:
    app_id: str
    app_secret: str
    domain_name: str
    connection_mode: str
    encrypt_key: str
    verification_token: str
    group_policy: str
    allowed_group_users: frozenset[str]
    bot_open_id: str
    bot_user_id: str
    bot_name: str
    dedup_cache_size: int
    text_batch_delay_seconds: float
    text_batch_split_delay_seconds: float
    text_batch_max_messages: int
    text_batch_max_chars: int
    media_batch_delay_seconds: float
    webhook_host: str
    webhook_port: int
    webhook_path: str
    ws_reconnect_nonce: int = 30
    ws_reconnect_interval: int = 120
    ws_ping_interval: Optional[int] = None
    ws_ping_timeout: Optional[int] = None
    admins: frozenset[str] = frozenset()
    default_group_policy: str = ""
    group_rules: Dict[str, FeishuGroupRule] = field(default_factory=dict)


@dataclass
class FeishuGroupRule:
    """Per-group policy rule for controlling which users may interact with the bot."""

    policy: str  # "open" | "allowlist" | "blacklist" | "admin_only" | "disabled"
    allowlist: set[str] = field(default_factory=set)
    blacklist: set[str] = field(default_factory=set)


@dataclass
class FeishuBatchState:
    events: Dict[str, MessageEvent] = field(default_factory=dict)
    tasks: Dict[str, asyncio.Task] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Markdown rendering helpers
# ---------------------------------------------------------------------------


def _escape_markdown_text(text: str) -> str:
    return _MARKDOWN_SPECIAL_CHARS_RE.sub(r"\\\1", text)


def _to_boolean(value: Any) -> bool:
    return value is True or value == 1 or value == "true"


def _is_style_enabled(style: Dict[str, Any] | None, key: str) -> bool:
    if not style:
        return False
    return _to_boolean(style.get(key))


def _wrap_inline_code(text: str) -> str:
    max_run = max([0, *[len(run) for run in re.findall(r"`+", text)]])
    fence = "`" * (max_run + 1)
    body = f" {text} " if text.startswith("`") or text.endswith("`") else text
    return f"{fence}{body}{fence}"


def _sanitize_fence_language(language: str) -> str:
    return language.strip().replace("\n", " ").replace("\r", " ")


def _render_text_element(element: Dict[str, Any]) -> str:
    text = str(element.get("text", "") or "")
    style = element.get("style")
    style_dict = style if isinstance(style, dict) else None

    if _is_style_enabled(style_dict, "code"):
        return _wrap_inline_code(text)

    rendered = _escape_markdown_text(text)
    if not rendered:
        return ""
    if _is_style_enabled(style_dict, "bold"):
        rendered = f"**{rendered}**"
    if _is_style_enabled(style_dict, "italic"):
        rendered = f"*{rendered}*"
    if _is_style_enabled(style_dict, "underline"):
        rendered = f"<u>{rendered}</u>"
    if _is_style_enabled(style_dict, "strikethrough"):
        rendered = f"~~{rendered}~~"
    return rendered


def _render_code_block_element(element: Dict[str, Any]) -> str:
    language = _sanitize_fence_language(
        str(element.get("language", "") or "") or str(element.get("lang", "") or "")
    )
    code = (
        str(element.get("text", "") or "") or str(element.get("content", "") or "")
    ).replace("\r\n", "\n")
    trailing_newline = "" if code.endswith("\n") else "\n"
    return f"```{language}\n{code}{trailing_newline}```"


def _strip_markdown_to_plain_text(text: str) -> str:
    plain = text.replace("\r\n", "\n")
    plain = _MARKDOWN_LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2).strip()})", plain)
    plain = re.sub(r"^#{1,6}\s+", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^>\s?", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s*---+\s*$", "---", plain, flags=re.MULTILINE)
    plain = re.sub(r"```(?:[^\n]*\n)?([\s\S]*?)```", lambda m: m.group(1).strip("\n"), plain)
    plain = re.sub(r"`([^`\n]+)`", r"\1", plain)
    plain = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", plain)
    plain = re.sub(r"\*([^*\n]+)\*", r"\1", plain)
    plain = re.sub(r"~~([^~\n]+)~~", r"\1", plain)
    plain = re.sub(r"<u>([\s\S]*?)</u>", r"\1", plain)
    plain = re.sub(r"\n{3,}", "\n\n", plain)
    return plain.strip()


def _coerce_int(value: Any, default: Optional[int] = None, min_value: int = 0) -> Optional[int]:
    """Coerce value to int with optional default and minimum constraint."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= min_value else default


def _coerce_required_int(value: Any, default: int, min_value: int = 0) -> int:
    parsed = _coerce_int(value, default=default, min_value=min_value)
    return default if parsed is None else parsed


# ---------------------------------------------------------------------------
# Post payload builders and parsers
# ---------------------------------------------------------------------------


def _build_markdown_post_payload(content: str) -> str:
    return json.dumps(
        {
            "zh_cn": {
                "content": [
                    [
                        {
                            "tag": "md",
                            "text": content,
                        }
                    ]
                ],
            }
        },
        ensure_ascii=False,
    )


def parse_feishu_post_content(raw_content: str) -> FeishuPostParseResult:
    try:
        parsed = json.loads(raw_content) if raw_content else {}
    except json.JSONDecodeError:
        return FeishuPostParseResult(text_content=FALLBACK_POST_TEXT)
    return parse_feishu_post_payload(parsed)


def parse_feishu_post_payload(payload: Any) -> FeishuPostParseResult:
    resolved = _resolve_post_payload(payload)
    if not resolved:
        return FeishuPostParseResult(text_content=FALLBACK_POST_TEXT)

    image_keys: List[str] = []
    media_refs: List[FeishuPostMediaRef] = []
    mentioned_ids: List[str] = []
    parts: List[str] = []

    title = _normalize_feishu_text(str(resolved.get("title", "")).strip())
    if title:
        parts.append(title)

    for row in resolved.get("content", []) or []:
        if not isinstance(row, list):
            continue
        row_text = _normalize_feishu_text(
            "".join(_render_post_element(item, image_keys, media_refs, mentioned_ids) for item in row)
        )
        if row_text:
            parts.append(row_text)

    return FeishuPostParseResult(
        text_content="\n".join(parts).strip() or FALLBACK_POST_TEXT,
        image_keys=image_keys,
        media_refs=media_refs,
        mentioned_ids=mentioned_ids,
    )


def _resolve_post_payload(payload: Any) -> Dict[str, Any]:
    direct = _to_post_payload(payload)
    if direct:
        return direct
    if not isinstance(payload, dict):
        return {}

    wrapped = payload.get("post")
    wrapped_direct = _resolve_locale_payload(wrapped)
    if wrapped_direct:
        return wrapped_direct
    return _resolve_locale_payload(payload)


def _resolve_locale_payload(payload: Any) -> Dict[str, Any]:
    direct = _to_post_payload(payload)
    if direct:
        return direct
    if not isinstance(payload, dict):
        return {}

    for key in _PREFERRED_LOCALES:
        candidate = _to_post_payload(payload.get(key))
        if candidate:
            return candidate
    for value in payload.values():
        candidate = _to_post_payload(value)
        if candidate:
            return candidate
    return {}


def _to_post_payload(candidate: Any) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    content = candidate.get("content")
    if not isinstance(content, list):
        return {}
    return {
        "title": str(candidate.get("title", "") or ""),
        "content": content,
    }


def _render_post_element(
    element: Any,
    image_keys: List[str],
    media_refs: List[FeishuPostMediaRef],
    mentioned_ids: List[str],
) -> str:
    if isinstance(element, str):
        return element
    if not isinstance(element, dict):
        return ""

    tag = str(element.get("tag", "")).strip().lower()
    if tag == "text":
        return _render_text_element(element)
    if tag == "a":
        href = str(element.get("href", "")).strip()
        label = str(element.get("text", href) or "").strip()
        if not label:
            return ""
        escaped_label = _escape_markdown_text(label)
        return f"[{escaped_label}]({href})" if href else escaped_label
    if tag == "at":
        mentioned_id = (
            str(element.get("open_id", "")).strip()
            or str(element.get("user_id", "")).strip()
        )
        if mentioned_id and mentioned_id not in mentioned_ids:
            mentioned_ids.append(mentioned_id)
        display_name = (
            str(element.get("user_name", "")).strip()
            or str(element.get("name", "")).strip()
            or str(element.get("text", "")).strip()
            or mentioned_id
        )
        return f"@{_escape_markdown_text(display_name)}" if display_name else "@"
    if tag in {"img", "image"}:
        image_key = str(element.get("image_key", "")).strip()
        if image_key and image_key not in image_keys:
            image_keys.append(image_key)
        alt = str(element.get("text", "")).strip() or str(element.get("alt", "")).strip()
        return f"[Image: {alt}]" if alt else "[Image]"
    if tag in {"media", "file", "audio", "video"}:
        file_key = str(element.get("file_key", "")).strip()
        file_name = (
            str(element.get("file_name", "")).strip()
            or str(element.get("title", "")).strip()
            or str(element.get("text", "")).strip()
        )
        if file_key:
            media_refs.append(
                FeishuPostMediaRef(
                    file_key=file_key,
                    file_name=file_name,
                    resource_type=tag if tag in {"audio", "video"} else "file",
                )
            )
        return f"[Attachment: {file_name}]" if file_name else "[Attachment]"
    if tag in {"emotion", "emoji"}:
        label = str(element.get("text", "")).strip() or str(element.get("emoji_type", "")).strip()
        return f":{_escape_markdown_text(label)}:" if label else "[Emoji]"
    if tag == "br":
        return "\n"
    if tag in {"hr", "divider"}:
        return "\n\n---\n\n"
    if tag == "code":
        code = str(element.get("text", "") or "") or str(element.get("content", "") or "")
        return _wrap_inline_code(code) if code else ""
    if tag in {"code_block", "pre"}:
        return _render_code_block_element(element)

    nested_parts: List[str] = []
    for key in ("text", "title", "content", "children", "elements"):
        value = element.get(key)
        extracted = _render_nested_post(value, image_keys, media_refs, mentioned_ids)
        if extracted:
            nested_parts.append(extracted)
    return " ".join(part for part in nested_parts if part)


def _render_nested_post(
    value: Any,
    image_keys: List[str],
    media_refs: List[FeishuPostMediaRef],
    mentioned_ids: List[str],
) -> str:
    if isinstance(value, str):
        return _escape_markdown_text(value)
    if isinstance(value, list):
        return " ".join(
            part
            for item in value
            for part in [_render_nested_post(item, image_keys, media_refs, mentioned_ids)]
            if part
        )
    if isinstance(value, dict):
        direct = _render_post_element(value, image_keys, media_refs, mentioned_ids)
        if direct:
            return direct
        return " ".join(
            part
            for item in value.values()
            for part in [_render_nested_post(item, image_keys, media_refs, mentioned_ids)]
            if part
        )
    return ""


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------


def normalize_feishu_message(*, message_type: str, raw_content: str) -> FeishuNormalizedMessage:
    normalized_type = str(message_type or "").strip().lower()
    payload = _load_feishu_payload(raw_content)

    if normalized_type == "text":
        return FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content=_normalize_feishu_text(str(payload.get("text", "") or "")),
        )
    if normalized_type == "post":
        parsed_post = parse_feishu_post_payload(payload)
        return FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content=parsed_post.text_content,
            image_keys=list(parsed_post.image_keys),
            media_refs=list(parsed_post.media_refs),
            mentioned_ids=list(parsed_post.mentioned_ids),
            relation_kind="post",
        )
    if normalized_type == "image":
        image_key = str(payload.get("image_key", "") or "").strip()
        alt_text = _normalize_feishu_text(
            str(payload.get("text", "") or "")
            or str(payload.get("alt", "") or "")
            or FALLBACK_IMAGE_TEXT
        )
        return FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content=alt_text if alt_text != FALLBACK_IMAGE_TEXT else "",
            preferred_message_type="photo",
            image_keys=[image_key] if image_key else [],
            relation_kind="image",
        )
    if normalized_type in {"file", "audio", "media"}:
        media_ref = _build_media_ref_from_payload(payload, resource_type=normalized_type)
        placeholder = _attachment_placeholder(media_ref.file_name)
        return FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content="",
            preferred_message_type="audio" if normalized_type == "audio" else "document",
            media_refs=[media_ref] if media_ref.file_key else [],
            relation_kind=normalized_type,
            metadata={"placeholder_text": placeholder},
        )
    if normalized_type == "merge_forward":
        return _normalize_merge_forward_message(payload)
    if normalized_type == "share_chat":
        return _normalize_share_chat_message(payload)
    if normalized_type in {"interactive", "card"}:
        return _normalize_interactive_message(normalized_type, payload)

    return FeishuNormalizedMessage(raw_type=normalized_type, text_content="")


def _load_feishu_payload(raw_content: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw_content) if raw_content else {}
    except json.JSONDecodeError:
        return {"text": raw_content}
    return parsed if isinstance(parsed, dict) else {"content": parsed}


def _normalize_merge_forward_message(payload: Dict[str, Any]) -> FeishuNormalizedMessage:
    title = _first_non_empty_text(
        payload.get("title"),
        payload.get("summary"),
        payload.get("preview"),
        _find_first_text(payload, keys=("title", "summary", "preview", "description")),
    )
    entries = _collect_forward_entries(payload)
    lines: List[str] = []
    if title:
        lines.append(title)
    lines.extend(entries[:8])
    text_content = "\n".join(lines).strip() or FALLBACK_FORWARD_TEXT
    return FeishuNormalizedMessage(
        raw_type="merge_forward",
        text_content=text_content,
        relation_kind="merge_forward",
        metadata={"entry_count": len(entries), "title": title},
    )


def _normalize_share_chat_message(payload: Dict[str, Any]) -> FeishuNormalizedMessage:
    chat_name = _first_non_empty_text(
        payload.get("chat_name"),
        payload.get("name"),
        payload.get("title"),
        _find_first_text(payload, keys=("chat_name", "name", "title")),
    )
    share_id = _first_non_empty_text(
        payload.get("chat_id"),
        payload.get("open_chat_id"),
        payload.get("share_chat_id"),
    )
    lines = []
    if chat_name:
        lines.append(f"Shared chat: {chat_name}")
    else:
        lines.append(FALLBACK_SHARE_CHAT_TEXT)
    if share_id:
        lines.append(f"Chat ID: {share_id}")
    text_content = "\n".join(lines)
    return FeishuNormalizedMessage(
        raw_type="share_chat",
        text_content=text_content,
        relation_kind="share_chat",
        metadata={"chat_id": share_id, "chat_name": chat_name},
    )


def _normalize_interactive_message(message_type: str, payload: Dict[str, Any]) -> FeishuNormalizedMessage:
    card_payload = payload.get("card") if isinstance(payload.get("card"), dict) else payload
    title = _first_non_empty_text(
        _find_header_title(card_payload),
        payload.get("title"),
        _find_first_text(card_payload, keys=("title", "summary", "subtitle")),
    )
    body_lines = _collect_card_lines(card_payload)
    actions = _collect_action_labels(card_payload)

    lines: List[str] = []
    if title:
        lines.append(title)
    for line in body_lines:
        if line != title:
            lines.append(line)
    if actions:
        lines.append(f"Actions: {', '.join(actions)}")

    text_content = "\n".join(lines[:12]).strip() or FALLBACK_INTERACTIVE_TEXT
    return FeishuNormalizedMessage(
        raw_type=message_type,
        text_content=text_content,
        relation_kind="interactive",
        metadata={"title": title, "actions": actions},
    )


# ---------------------------------------------------------------------------
# Content extraction utilities (card / forward / text walking)
# ---------------------------------------------------------------------------


def _collect_forward_entries(payload: Dict[str, Any]) -> List[str]:
    candidates: List[Any] = []
    for key in ("messages", "items", "message_list", "records", "content"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    entries: List[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            text = _normalize_feishu_text(str(item or ""))
            if text:
                entries.append(f"- {text}")
            continue
        sender = _first_non_empty_text(
            item.get("sender_name"),
            item.get("user_name"),
            item.get("sender"),
            item.get("name"),
        )
        nested_type = str(item.get("message_type", "") or item.get("msg_type", "")).strip().lower()
        if nested_type == "post":
            body = parse_feishu_post_payload(item.get("content") or item).text_content
        else:
            body = _first_non_empty_text(
                item.get("text"),
                item.get("summary"),
                item.get("preview"),
                item.get("content"),
                _find_first_text(item, keys=("text", "content", "summary", "preview", "title")),
            )
        body = _normalize_feishu_text(body)
        if sender and body:
            entries.append(f"- {sender}: {body}")
        elif body:
            entries.append(f"- {body}")
    return _unique_lines(entries)


def _collect_card_lines(payload: Any) -> List[str]:
    lines = _collect_text_segments(payload, in_rich_block=False)
    normalized = [_normalize_feishu_text(line) for line in lines]
    return _unique_lines([line for line in normalized if line])


def _collect_action_labels(payload: Any) -> List[str]:
    labels: List[str] = []
    for item in _walk_nodes(payload):
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag", "") or item.get("type", "")).strip().lower()
        if tag not in {"button", "select_static", "overflow", "date_picker", "picker"}:
            continue
        label = _first_non_empty_text(
            item.get("text"),
            item.get("name"),
            item.get("value"),
            _find_first_text(item, keys=("text", "content", "name", "value")),
        )
        if label:
            labels.append(label)
    return _unique_lines(labels)


def _collect_text_segments(value: Any, *, in_rich_block: bool) -> List[str]:
    if isinstance(value, str):
        return [_normalize_feishu_text(value)] if in_rich_block else []
    if isinstance(value, list):
        segments: List[str] = []
        for item in value:
            segments.extend(_collect_text_segments(item, in_rich_block=in_rich_block))
        return segments
    if not isinstance(value, dict):
        return []

    tag = str(value.get("tag", "") or value.get("type", "")).strip().lower()
    next_in_rich_block = in_rich_block or tag in {
        "plain_text",
        "lark_md",
        "markdown",
        "note",
        "div",
        "column_set",
        "column",
        "action",
        "button",
        "select_static",
        "date_picker",
    }

    segments: List[str] = []
    for key in _SUPPORTED_CARD_TEXT_KEYS:
        item = value.get(key)
        if isinstance(item, str) and next_in_rich_block:
            normalized = _normalize_feishu_text(item)
            if normalized:
                segments.append(normalized)

    for key, item in value.items():
        if key in _SKIP_TEXT_KEYS:
            continue
        segments.extend(_collect_text_segments(item, in_rich_block=next_in_rich_block))
    return segments


def _build_media_ref_from_payload(payload: Dict[str, Any], *, resource_type: str) -> FeishuPostMediaRef:
    file_key = str(payload.get("file_key", "") or "").strip()
    file_name = _first_non_empty_text(
        payload.get("file_name"),
        payload.get("title"),
        payload.get("text"),
    )
    effective_type = resource_type if resource_type in {"audio", "video"} else "file"
    return FeishuPostMediaRef(file_key=file_key, file_name=file_name, resource_type=effective_type)


def _attachment_placeholder(file_name: str) -> str:
    normalized_name = _normalize_feishu_text(file_name)
    return f"[Attachment: {normalized_name}]" if normalized_name else FALLBACK_ATTACHMENT_TEXT


def _find_header_title(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    header = payload.get("header")
    if not isinstance(header, dict):
        return ""
    title = header.get("title")
    if isinstance(title, dict):
        return _first_non_empty_text(title.get("content"), title.get("text"), title.get("name"))
    return _normalize_feishu_text(str(title or ""))


def _find_first_text(payload: Any, *, keys: tuple[str, ...]) -> str:
    for node in _walk_nodes(payload):
        if not isinstance(node, dict):
            continue
        for key in keys:
            value = node.get(key)
            if isinstance(value, str):
                normalized = _normalize_feishu_text(value)
                if normalized:
                    return normalized
    return ""


def _walk_nodes(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_nodes(item)


def _first_non_empty_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str):
            normalized = _normalize_feishu_text(value)
            if normalized:
                return normalized
        elif value is not None and not isinstance(value, (dict, list)):
            normalized = _normalize_feishu_text(str(value))
            if normalized:
                return normalized
    return ""


# ---------------------------------------------------------------------------
# General text utilities
# ---------------------------------------------------------------------------


def _normalize_feishu_text(text: str) -> str:
    cleaned = _MENTION_PLACEHOLDER_RE.sub(" ", text or "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(_WHITESPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n"))
    cleaned = "\n".join(line for line in cleaned.split("\n") if line)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _unique_lines(lines: List[str]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for line in lines:
        if not line or line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique


def _run_official_feishu_ws_client(ws_client: Any, adapter: Any) -> None:
    """Run the official Lark WS client in its own thread-local event loop."""
    import lark_oapi.ws.client as ws_client_module

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ws_client_module.loop = loop
    adapter._ws_thread_loop = loop

    original_connect = ws_client_module.websockets.connect
    original_configure = getattr(ws_client, "_configure", None)

    def _apply_runtime_ws_overrides() -> None:
        try:
            setattr(ws_client, "_reconnect_nonce", adapter._ws_reconnect_nonce)
            setattr(ws_client, "_reconnect_interval", adapter._ws_reconnect_interval)
            if adapter._ws_ping_interval is not None:
                setattr(ws_client, "_ping_interval", adapter._ws_ping_interval)
        except Exception:
            logger.debug("[Feishu] Failed to apply websocket runtime overrides", exc_info=True)

    async def _connect_with_overrides(*args: Any, **kwargs: Any) -> Any:
        if adapter._ws_ping_interval is not None and "ping_interval" not in kwargs:
            kwargs["ping_interval"] = adapter._ws_ping_interval
        if adapter._ws_ping_timeout is not None and "ping_timeout" not in kwargs:
            kwargs["ping_timeout"] = adapter._ws_ping_timeout
        return await original_connect(*args, **kwargs)

    def _configure_with_overrides(conf: Any) -> Any:
        if original_configure is None:
            raise RuntimeError("Feishu _configure_with_overrides called but original_configure is None")
        result = original_configure(conf)
        _apply_runtime_ws_overrides()
        return result

    ws_client_module.websockets.connect = _connect_with_overrides
    if original_configure is not None:
        setattr(ws_client, "_configure", _configure_with_overrides)
    _apply_runtime_ws_overrides()
    try:
        ws_client.start()
    except Exception:
        pass
    finally:
        ws_client_module.websockets.connect = original_connect
        if original_configure is not None:
            setattr(ws_client, "_configure", original_configure)
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        try:
            loop.stop()
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass
        adapter._ws_thread_loop = None


def check_feishu_requirements() -> bool:
    """Check if Feishu/Lark dependencies are available."""
    return FEISHU_AVAILABLE


class FeishuAdapter(BasePlatformAdapter):
    """Feishu/Lark bot adapter."""

    MAX_MESSAGE_LENGTH = 8000
    # Threshold for detecting Feishu client-side message splits.
    # When a chunk is near the ~4096-char practical limit, a continuation
    # is almost certain.
    _SPLIT_THRESHOLD = 4000

    # =========================================================================
    # Lifecycle — init / settings / connect / disconnect
    # =========================================================================

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.FEISHU)

        self._settings = self._load_settings(config.extra or {})
        self._apply_settings(self._settings)
        self._client: Optional[Any] = None
        self._ws_client: Optional[Any] = None
        self._ws_future: Optional[asyncio.Future] = None
        self._ws_thread_loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._webhook_runner: Optional[Any] = None
        self._webhook_site: Optional[Any] = None
        self._event_handler: Optional[Any] = None
        self._seen_message_ids: Dict[str, float] = {}  # message_id → seen_at (time.time())
        self._seen_message_order: List[str] = []
        self._dedup_state_path = get_hermes_home() / "feishu_seen_message_ids.json"
        self._dedup_lock = threading.Lock()
        self._sender_name_cache: Dict[str, tuple[str, float]] = {}  # sender_id → (name, expire_at)
        self._webhook_rate_counts: Dict[str, tuple[int, float]] = {}  # rate_key → (count, window_start)
        self._webhook_anomaly_counts: Dict[str, tuple[int, str, float]] = {}  # ip → (count, last_status, first_seen)
        self._card_action_tokens: Dict[str, float] = {}  # token → first_seen_time
        self._chat_locks: Dict[str, asyncio.Lock] = {}  # chat_id → lock (per-chat serial processing)
        self._sent_message_ids_to_chat: Dict[str, str] = {}  # message_id → chat_id (for reaction routing)
        self._sent_message_id_order: List[str] = []  # LRU order for _sent_message_ids_to_chat
        self._chat_info_cache: Dict[str, Dict[str, Any]] = {}
        self._message_text_cache: Dict[str, Optional[str]] = {}
        self._app_lock_identity: Optional[str] = None
        self._text_batch_state = FeishuBatchState()
        self._pending_text_batches = self._text_batch_state.events
        self._pending_text_batch_tasks = self._text_batch_state.tasks
        self._pending_text_batch_counts = self._text_batch_state.counts
        self._media_batch_state = FeishuBatchState()
        self._pending_media_batches = self._media_batch_state.events
        self._pending_media_batch_tasks = self._media_batch_state.tasks
        # Exec approval button state (approval_id → {session_key, message_id, chat_id})
        self._approval_state: Dict[int, Dict[str, str]] = {}
        self._approval_counter = itertools.count(1)
        self._load_seen_message_ids()

    @staticmethod
    def _load_settings(extra: Dict[str, Any]) -> FeishuAdapterSettings:
        # Parse per-group rules from config
        raw_group_rules = extra.get("group_rules", {})
        group_rules: Dict[str, FeishuGroupRule] = {}
        if isinstance(raw_group_rules, dict):
            for chat_id, rule_cfg in raw_group_rules.items():
                if not isinstance(rule_cfg, dict):
                    continue
                group_rules[str(chat_id)] = FeishuGroupRule(
                    policy=str(rule_cfg.get("policy", "open")).strip().lower(),
                    allowlist=set(str(u).strip() for u in rule_cfg.get("allowlist", []) if str(u).strip()),
                    blacklist=set(str(u).strip() for u in rule_cfg.get("blacklist", []) if str(u).strip()),
                )

        # Bot-level admins
        raw_admins = extra.get("admins", [])
        admins = frozenset(str(u).strip() for u in raw_admins if str(u).strip())

        # Default group policy (for groups not in group_rules)
        default_group_policy = str(extra.get("default_group_policy", "")).strip().lower()

        return FeishuAdapterSettings(
            app_id=str(extra.get("app_id") or os.getenv("FEISHU_APP_ID", "")).strip(),
            app_secret=str(extra.get("app_secret") or os.getenv("FEISHU_APP_SECRET", "")).strip(),
            domain_name=str(extra.get("domain") or os.getenv("FEISHU_DOMAIN", "feishu")).strip().lower(),
            connection_mode=str(
                extra.get("connection_mode") or os.getenv("FEISHU_CONNECTION_MODE", "websocket")
            ).strip().lower(),
            encrypt_key=os.getenv("FEISHU_ENCRYPT_KEY", "").strip(),
            verification_token=os.getenv("FEISHU_VERIFICATION_TOKEN", "").strip(),
            group_policy=os.getenv("FEISHU_GROUP_POLICY", "allowlist").strip().lower(),
            allowed_group_users=frozenset(
                item.strip()
                for item in os.getenv("FEISHU_ALLOWED_USERS", "").split(",")
                if item.strip()
            ),
            bot_open_id=os.getenv("FEISHU_BOT_OPEN_ID", "").strip(),
            bot_user_id=os.getenv("FEISHU_BOT_USER_ID", "").strip(),
            bot_name=os.getenv("FEISHU_BOT_NAME", "").strip(),
            dedup_cache_size=max(
                32,
                int(os.getenv("HERMES_FEISHU_DEDUP_CACHE_SIZE", str(_DEFAULT_DEDUP_CACHE_SIZE))),
            ),
            text_batch_delay_seconds=float(
                os.getenv("HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS", str(_DEFAULT_TEXT_BATCH_DELAY_SECONDS))
            ),
            text_batch_split_delay_seconds=float(
                os.getenv("HERMES_FEISHU_TEXT_BATCH_SPLIT_DELAY_SECONDS", "2.0")
            ),
            text_batch_max_messages=max(
                1,
                int(os.getenv("HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES", str(_DEFAULT_TEXT_BATCH_MAX_MESSAGES))),
            ),
            text_batch_max_chars=max(
                1,
                int(os.getenv("HERMES_FEISHU_TEXT_BATCH_MAX_CHARS", str(_DEFAULT_TEXT_BATCH_MAX_CHARS))),
            ),
            media_batch_delay_seconds=float(
                os.getenv("HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS", str(_DEFAULT_MEDIA_BATCH_DELAY_SECONDS))
            ),
            webhook_host=str(
                extra.get("webhook_host") or os.getenv("FEISHU_WEBHOOK_HOST", _DEFAULT_WEBHOOK_HOST)
            ).strip(),
            webhook_port=int(
                extra.get("webhook_port") or os.getenv("FEISHU_WEBHOOK_PORT", str(_DEFAULT_WEBHOOK_PORT))
            ),
            webhook_path=(
                str(extra.get("webhook_path") or os.getenv("FEISHU_WEBHOOK_PATH", _DEFAULT_WEBHOOK_PATH)).strip()
                or _DEFAULT_WEBHOOK_PATH
            ),
            ws_reconnect_nonce=_coerce_required_int(extra.get("ws_reconnect_nonce"), default=30, min_value=0),
            ws_reconnect_interval=_coerce_required_int(extra.get("ws_reconnect_interval"), default=120, min_value=1),
            ws_ping_interval=_coerce_int(extra.get("ws_ping_interval"), default=None, min_value=1),
            ws_ping_timeout=_coerce_int(extra.get("ws_ping_timeout"), default=None, min_value=1),
            admins=admins,
            default_group_policy=default_group_policy,
            group_rules=group_rules,
        )

    def _apply_settings(self, settings: FeishuAdapterSettings) -> None:
        self._app_id = settings.app_id
        self._app_secret = settings.app_secret
        self._domain_name = settings.domain_name
        self._connection_mode = settings.connection_mode
        self._encrypt_key = settings.encrypt_key
        self._verification_token = settings.verification_token
        self._group_policy = settings.group_policy
        self._allowed_group_users = set(settings.allowed_group_users)
        self._admins = set(settings.admins)
        self._default_group_policy = settings.default_group_policy or settings.group_policy
        self._group_rules = settings.group_rules
        self._bot_open_id = settings.bot_open_id
        self._bot_user_id = settings.bot_user_id
        self._bot_name = settings.bot_name
        self._dedup_cache_size = settings.dedup_cache_size
        self._text_batch_delay_seconds = settings.text_batch_delay_seconds
        self._text_batch_split_delay_seconds = settings.text_batch_split_delay_seconds
        self._text_batch_max_messages = settings.text_batch_max_messages
        self._text_batch_max_chars = settings.text_batch_max_chars
        self._media_batch_delay_seconds = settings.media_batch_delay_seconds
        self._webhook_host = settings.webhook_host
        self._webhook_port = settings.webhook_port
        self._webhook_path = settings.webhook_path
        self._ws_reconnect_nonce = settings.ws_reconnect_nonce
        self._ws_reconnect_interval = settings.ws_reconnect_interval
        self._ws_ping_interval = settings.ws_ping_interval
        self._ws_ping_timeout = settings.ws_ping_timeout

    def _build_event_handler(self) -> Any:
        if EventDispatcherHandler is None:
            return None
        return (
            EventDispatcherHandler.builder(
                self._encrypt_key,
                self._verification_token,
            )
            .register_p2_im_message_message_read_v1(self._on_message_read_event)
            .register_p2_im_message_receive_v1(self._on_message_event)
            .register_p2_im_message_reaction_created_v1(
                lambda data: self._on_reaction_event("im.message.reaction.created_v1", data)
            )
            .register_p2_im_message_reaction_deleted_v1(
                lambda data: self._on_reaction_event("im.message.reaction.deleted_v1", data)
            )
            .register_p2_card_action_trigger(self._on_card_action_trigger)
            .build()
        )

    async def connect(self) -> bool:
        """Connect to Feishu/Lark."""
        if not FEISHU_AVAILABLE:
            logger.error("[Feishu] lark-oapi not installed")
            return False
        if not self._app_id or not self._app_secret:
            logger.error("[Feishu] FEISHU_APP_ID or FEISHU_APP_SECRET not set")
            return False
        if self._connection_mode not in {"websocket", "webhook"}:
            logger.error(
                "[Feishu] Unsupported FEISHU_CONNECTION_MODE=%s. Supported modes: websocket, webhook.",
                self._connection_mode,
            )
            return False

        try:
            self._app_lock_identity = self._app_id
            acquired, existing = acquire_scoped_lock(
                _FEISHU_APP_LOCK_SCOPE,
                self._app_lock_identity,
                metadata={"platform": self.platform.value},
            )
            if not acquired:
                owner_pid = existing.get("pid") if isinstance(existing, dict) else None
                message = (
                    "Another local Hermes gateway is already using this Feishu app_id"
                    + (f" (PID {owner_pid})." if owner_pid else ".")
                    + " Stop the other gateway before starting a second Feishu websocket client."
                )
                logger.error("[Feishu] %s", message)
                self._set_fatal_error("feishu_app_lock", message, retryable=False)
                return False

            self._loop = asyncio.get_running_loop()
            await self._connect_with_retry()
            self._mark_connected()
            logger.info("[Feishu] Connected in %s mode (%s)", self._connection_mode, self._domain_name)
            return True
        except Exception as exc:
            await self._release_app_lock()
            message = f"Feishu startup failed: {exc}"
            self._set_fatal_error("feishu_connect_error", message, retryable=True)
            logger.error("[Feishu] Failed to connect: %s", exc, exc_info=True)
            return False

    async def disconnect(self) -> None:
        """Disconnect from Feishu/Lark."""
        self._running = False
        await self._cancel_pending_tasks(self._pending_text_batch_tasks)
        await self._cancel_pending_tasks(self._pending_media_batch_tasks)
        self._reset_batch_buffers()
        self._disable_websocket_auto_reconnect()
        await self._stop_webhook_server()

        ws_thread_loop = self._ws_thread_loop
        if ws_thread_loop is not None and not ws_thread_loop.is_closed():
            logger.debug("[Feishu] Cancelling websocket thread tasks and stopping loop")

            def cancel_all_tasks() -> None:
                tasks = [t for t in asyncio.all_tasks(ws_thread_loop) if not t.done()]
                logger.debug("[Feishu] Found %d pending tasks in websocket thread", len(tasks))
                for task in tasks:
                    task.cancel()
                ws_thread_loop.call_later(0.1, ws_thread_loop.stop)

            ws_thread_loop.call_soon_threadsafe(cancel_all_tasks)

        ws_future = self._ws_future
        if ws_future is not None:
            try:
                logger.debug("[Feishu] Waiting for websocket thread to exit (timeout=10s)")
                await asyncio.wait_for(asyncio.shield(ws_future), timeout=10.0)
                logger.debug("[Feishu] Websocket thread exited cleanly")
            except asyncio.TimeoutError:
                logger.warning("[Feishu] Websocket thread did not exit within 10s - may be stuck")
            except asyncio.CancelledError:
                logger.debug("[Feishu] Websocket thread cancelled during disconnect")
            except Exception as exc:
                logger.debug("[Feishu] Websocket thread exited with error: %s", exc, exc_info=True)

        self._ws_future = None
        self._ws_thread_loop = None
        self._loop = None
        self._event_handler = None
        self._persist_seen_message_ids()
        await self._release_app_lock()

        self._mark_disconnected()
        logger.info("[Feishu] Disconnected")

    async def _cancel_pending_tasks(self, tasks: Dict[str, asyncio.Task]) -> None:
        pending = [task for task in tasks.values() if task and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        tasks.clear()

    def _reset_batch_buffers(self) -> None:
        self._pending_text_batches.clear()
        self._pending_text_batch_counts.clear()
        self._pending_media_batches.clear()

    def _disable_websocket_auto_reconnect(self) -> None:
        if self._ws_client is None:
            return
        try:
            setattr(self._ws_client, "_auto_reconnect", False)
        except Exception:
            pass
        finally:
            self._ws_client = None

    async def _stop_webhook_server(self) -> None:
        if self._webhook_runner is None:
            return
        try:
            await self._webhook_runner.cleanup()
        finally:
            self._webhook_runner = None
            self._webhook_site = None

    # =========================================================================
    # Outbound — send / edit / send_image / send_voice / …
    # =========================================================================

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a Feishu message."""
        if not self._client:
            return SendResult(success=False, error="Not connected")

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
        last_response = None

        try:
            for chunk in chunks:
                msg_type, payload = self._build_outbound_payload(chunk)
                try:
                    response = await self._feishu_send_with_retry(
                        chat_id=chat_id,
                        msg_type=msg_type,
                        payload=payload,
                        reply_to=reply_to,
                        metadata=metadata,
                    )
                except Exception as exc:
                    if msg_type != "post" or not _POST_CONTENT_INVALID_RE.search(str(exc)):
                        raise
                    logger.warning("[Feishu] Invalid post payload rejected by API; falling back to plain text")
                    response = await self._feishu_send_with_retry(
                        chat_id=chat_id,
                        msg_type="text",
                        payload=json.dumps({"text": _strip_markdown_to_plain_text(chunk)}, ensure_ascii=False),
                        reply_to=reply_to,
                        metadata=metadata,
                    )
                if (
                    msg_type == "post"
                    and not self._response_succeeded(response)
                    and _POST_CONTENT_INVALID_RE.search(str(getattr(response, "msg", "") or ""))
                ):
                    logger.warning("[Feishu] Post payload rejected by API response; falling back to plain text")
                    response = await self._feishu_send_with_retry(
                        chat_id=chat_id,
                        msg_type="text",
                        payload=json.dumps({"text": _strip_markdown_to_plain_text(chunk)}, ensure_ascii=False),
                        reply_to=reply_to,
                        metadata=metadata,
                    )
                last_response = response

            return self._finalize_send_result(last_response, "send failed")
        except Exception as exc:
            logger.error("[Feishu] Send error: %s", exc, exc_info=True)
            return SendResult(success=False, error=str(exc))

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
    ) -> SendResult:
        """Edit a previously sent Feishu text/post message."""
        if not self._client:
            return SendResult(success=False, error="Not connected")

        try:
            msg_type, payload = self._build_outbound_payload(content)
            body = self._build_update_message_body(msg_type=msg_type, content=payload)
            request = self._build_update_message_request(message_id=message_id, request_body=body)
            response = await asyncio.to_thread(self._client.im.v1.message.update, request)
            result = self._finalize_send_result(response, "update failed")
            if not result.success and msg_type == "post" and _POST_CONTENT_INVALID_RE.search(result.error or ""):
                logger.warning("[Feishu] Invalid post update payload rejected by API; falling back to plain text")
                fallback_body = self._build_update_message_body(
                    msg_type="text",
                    content=json.dumps({"text": _strip_markdown_to_plain_text(content)}, ensure_ascii=False),
                )
                fallback_request = self._build_update_message_request(message_id=message_id, request_body=fallback_body)
                fallback_response = await asyncio.to_thread(self._client.im.v1.message.update, fallback_request)
                result = self._finalize_send_result(fallback_response, "update failed")
            if result.success:
                result.message_id = message_id
            return result
        except Exception as exc:
            logger.error("[Feishu] Failed to edit message %s: %s", message_id, exc, exc_info=True)
            return SendResult(success=False, error=str(exc))

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an interactive card with approval buttons.

        The buttons carry ``hermes_action`` in their value dict so that
        ``_handle_card_action_event`` can intercept them and call
        ``resolve_gateway_approval()`` to unblock the waiting agent thread.
        """
        if not self._client:
            return SendResult(success=False, error="Not connected")

        try:
            approval_id = next(self._approval_counter)
            cmd_preview = command[:3000] + "..." if len(command) > 3000 else command

            def _btn(label: str, action_name: str, btn_type: str = "default") -> dict:
                return {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": label},
                    "type": btn_type,
                    "value": {"hermes_action": action_name, "approval_id": approval_id},
                }

            card = {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"content": "⚠️ Command Approval Required", "tag": "plain_text"},
                    "template": "orange",
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": f"```\n{cmd_preview}\n```\n**Reason:** {description}",
                    },
                    {
                        "tag": "action",
                        "actions": [
                            _btn("✅ Allow Once", "approve_once", "primary"),
                            _btn("✅ Session", "approve_session"),
                            _btn("✅ Always", "approve_always"),
                            _btn("❌ Deny", "deny", "danger"),
                        ],
                    },
                ],
            }

            payload = json.dumps(card, ensure_ascii=False)
            response = await self._feishu_send_with_retry(
                chat_id=chat_id,
                msg_type="interactive",
                payload=payload,
                reply_to=None,
                metadata=metadata,
            )

            result = self._finalize_send_result(response, "send_exec_approval failed")
            if result.success:
                self._approval_state[approval_id] = {
                    "session_key": session_key,
                    "message_id": result.message_id or "",
                    "chat_id": chat_id,
                }
            return result
        except Exception as exc:
            logger.warning("[Feishu] send_exec_approval failed: %s", exc)
            return SendResult(success=False, error=str(exc))

    async def _update_approval_card(
        self, message_id: str, label: str, user_name: str, choice: str,
    ) -> None:
        """Replace the approval card with a resolved status card."""
        if not self._client or not message_id:
            return
        icon = "❌" if choice == "deny" else "✅"
        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"content": f"{icon} {label}", "tag": "plain_text"},
                "template": "red" if choice == "deny" else "green",
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": f"{icon} **{label}** by {user_name}",
                },
            ],
        }
        try:
            payload = json.dumps(card, ensure_ascii=False)
            body = self._build_update_message_body(msg_type="interactive", content=payload)
            request = self._build_update_message_request(message_id=message_id, request_body=body)
            await asyncio.to_thread(self._client.im.v1.message.update, request)
        except Exception as exc:
            logger.warning("[Feishu] Failed to update approval card %s: %s", message_id, exc)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send audio to Feishu as a file attachment plus optional caption."""
        return await self._send_uploaded_file_message(
            chat_id=chat_id,
            file_path=audio_path,
            reply_to=reply_to,
            metadata=metadata,
            caption=caption,
            outbound_message_type="audio",
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
        """Send a document/file attachment to Feishu."""
        return await self._send_uploaded_file_message(
            chat_id=chat_id,
            file_path=file_path,
            reply_to=reply_to,
            metadata=metadata,
            caption=caption,
            file_name=file_name,
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
        """Send a video file to Feishu."""
        return await self._send_uploaded_file_message(
            chat_id=chat_id,
            file_path=video_path,
            reply_to=reply_to,
            metadata=metadata,
            caption=caption,
            outbound_message_type="media",
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
        """Send a local image file to Feishu."""
        if not self._client:
            return SendResult(success=False, error="Not connected")
        if not os.path.exists(image_path):
            return SendResult(success=False, error=f"Image file not found: {image_path}")

        try:
            with open(image_path, "rb") as image_file:
                body = self._build_image_upload_body(
                    image_type=_FEISHU_IMAGE_UPLOAD_TYPE,
                    image=image_file,
                )
                request = self._build_image_upload_request(body)
                upload_response = await asyncio.to_thread(self._client.im.v1.image.create, request)
            image_key = self._extract_response_field(upload_response, "image_key")
            if not image_key:
                return self._response_error_result(
                    upload_response,
                    default_message="image upload failed",
                    override_error="Feishu image upload missing image_key",
                )

            if caption:
                post_payload = self._build_media_post_payload(
                    caption=caption,
                    media_tag={"tag": "img", "image_key": image_key},
                )
                message_response = await self._feishu_send_with_retry(
                    chat_id=chat_id,
                    msg_type="post",
                    payload=post_payload,
                    reply_to=reply_to,
                    metadata=metadata,
                )
            else:
                message_response = await self._feishu_send_with_retry(
                    chat_id=chat_id,
                    msg_type="image",
                    payload=json.dumps({"image_key": image_key}, ensure_ascii=False),
                    reply_to=reply_to,
                    metadata=metadata,
                )
            return self._finalize_send_result(message_response, "image send failed")
        except Exception as exc:
            logger.error("[Feishu] Failed to send image %s: %s", image_path, exc, exc_info=True)
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Feishu bot API does not expose a typing indicator."""
        return None

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Download a remote image then send it through the native Feishu image flow."""
        try:
            image_path = await self._download_remote_image(image_url)
        except Exception as exc:
            logger.error("[Feishu] Failed to download image %s: %s", image_url, exc, exc_info=True)
            return await super().send_image(
                chat_id=chat_id,
                image_url=image_url,
                caption=caption,
                reply_to=reply_to,
                metadata=metadata,
            )
        return await self.send_image_file(
            chat_id=chat_id,
            image_path=image_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Feishu has no native GIF bubble; degrade to a downloadable file."""
        try:
            file_path, file_name = await self._download_remote_document(
                animation_url,
                default_ext=".gif",
                preferred_name="animation.gif",
            )
        except Exception as exc:
            logger.error("[Feishu] Failed to download animation %s: %s", animation_url, exc, exc_info=True)
            return await super().send_animation(
                chat_id=chat_id,
                animation_url=animation_url,
                caption=caption,
                reply_to=reply_to,
                metadata=metadata,
            )
        degraded_caption = f"[GIF downgraded to file]\n{caption}" if caption else "[GIF downgraded to file]"
        return await self.send_document(
            chat_id=chat_id,
            file_path=file_path,
            file_name=file_name,
            caption=degraded_caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return real chat metadata from Feishu when available."""
        fallback = {
            "chat_id": chat_id,
            "name": chat_id,
            "type": "dm",
        }
        if not self._client:
            return fallback

        cached = self._chat_info_cache.get(chat_id)
        if cached is not None:
            return dict(cached)

        try:
            request = self._build_get_chat_request(chat_id)
            response = await asyncio.to_thread(self._client.im.v1.chat.get, request)
            if not response or getattr(response, "success", lambda: False)() is False:
                code = getattr(response, "code", "unknown")
                msg = getattr(response, "msg", "chat lookup failed")
                logger.warning("[Feishu] Failed to get chat info for %s: [%s] %s", chat_id, code, msg)
                return fallback

            data = getattr(response, "data", None)
            raw_chat_type = str(getattr(data, "chat_type", "") or "").strip().lower()
            info = {
                "chat_id": chat_id,
                "name": str(getattr(data, "name", None) or chat_id),
                "type": self._map_chat_type(raw_chat_type),
                "raw_type": raw_chat_type or None,
            }
            self._chat_info_cache[chat_id] = info
            return dict(info)
        except Exception:
            logger.warning("[Feishu] Failed to get chat info for %s", chat_id, exc_info=True)
            return fallback

    def format_message(self, content: str) -> str:
        """Feishu text messages are plain text by default."""
        return content.strip()

    # =========================================================================
    # Inbound event handlers
    # =========================================================================

    def _on_message_event(self, data: Any) -> None:
        """Normalize Feishu inbound events into MessageEvent."""
        loop = self._loop
        if loop is None or bool(getattr(loop, "is_closed", lambda: False)()):
            logger.warning("[Feishu] Dropping inbound message before adapter loop is ready")
            return
        future = asyncio.run_coroutine_threadsafe(
            self._handle_message_event_data(data),
            loop,
        )
        future.add_done_callback(self._log_background_failure)

    async def _handle_message_event_data(self, data: Any) -> None:
        """Shared inbound message handling for websocket and webhook transports."""
        event = getattr(data, "event", None)
        message = getattr(event, "message", None)
        sender = getattr(event, "sender", None)
        sender_id = getattr(sender, "sender_id", None)
        if not message or not sender_id:
            logger.debug("[Feishu] Dropping malformed inbound event: missing message or sender_id")
            return

        message_id = getattr(message, "message_id", None)
        if not message_id or self._is_duplicate(message_id):
            logger.debug("[Feishu] Dropping duplicate/missing message_id: %s", message_id)
            return
        if getattr(sender, "sender_type", "") == "bot":
            logger.debug("[Feishu] Dropping bot-originated event: %s", message_id)
            return

        chat_type = getattr(message, "chat_type", "p2p")
        chat_id = getattr(message, "chat_id", "") or ""
        if chat_type != "p2p" and not self._should_accept_group_message(message, sender_id, chat_id):
            logger.debug("[Feishu] Dropping group message that failed mention/policy gate: %s", message_id)
            return
        await self._process_inbound_message(
            data=data,
            message=message,
            sender_id=sender_id,
            chat_type=chat_type,
            message_id=message_id,
        )

    def _on_message_read_event(self, data: P2ImMessageMessageReadV1) -> None:
        """Ignore read-receipt events that Hermes does not act on."""
        event = getattr(data, "event", None)
        message = getattr(event, "message", None)
        message_id = getattr(message, "message_id", None) or ""
        logger.debug("[Feishu] Ignoring message_read event: %s", message_id)

    def _on_bot_added_to_chat(self, data: Any) -> None:
        """Handle bot being added to a group chat."""
        event = getattr(data, "event", None)
        chat_id = str(getattr(event, "chat_id", "") or "")
        logger.info("[Feishu] Bot added to chat: %s", chat_id)
        self._chat_info_cache.pop(chat_id, None)

    def _on_bot_removed_from_chat(self, data: Any) -> None:
        """Handle bot being removed from a group chat."""
        event = getattr(data, "event", None)
        chat_id = str(getattr(event, "chat_id", "") or "")
        logger.info("[Feishu] Bot removed from chat: %s", chat_id)
        self._chat_info_cache.pop(chat_id, None)

    def _on_reaction_event(self, event_type: str, data: Any) -> None:
        """Route user reactions on bot messages as synthetic text events."""
        event = getattr(data, "event", None)
        message_id = str(getattr(event, "message_id", "") or "")
        operator_type = str(getattr(event, "operator_type", "") or "")
        reaction_type_obj = getattr(event, "reaction_type", None)
        emoji_type = str(getattr(reaction_type_obj, "emoji_type", "") or "")
        action = "added" if "created" in event_type else "removed"
        logger.debug(
            "[Feishu] Reaction %s on message %s (operator_type=%s, emoji=%s)",
            action,
            message_id,
            operator_type,
            emoji_type,
        )
        # Only process reactions from real users. Ignore app/bot-generated reactions
        # and Hermes' own ACK emoji to avoid feedback loops.
        loop = self._loop
        if (
            operator_type in {"bot", "app"}
            or emoji_type == _FEISHU_ACK_EMOJI
            or not message_id
            or loop is None
            or bool(getattr(loop, "is_closed", lambda: False)())
        ):
            return
        future = asyncio.run_coroutine_threadsafe(
            self._handle_reaction_event(event_type, data),
            loop,
        )
        future.add_done_callback(self._log_background_failure)

    def _on_card_action_trigger(self, data: Any) -> Any:
        """Schedule Feishu card actions on the adapter loop and acknowledge immediately."""
        loop = self._loop
        if loop is None or bool(getattr(loop, "is_closed", lambda: False)()):
            logger.warning("[Feishu] Dropping card action before adapter loop is ready")
        else:
            future = asyncio.run_coroutine_threadsafe(
                self._handle_card_action_event(data),
                loop,
            )
            future.add_done_callback(self._log_background_failure)
        if P2CardActionTriggerResponse is None:
            return None
        return P2CardActionTriggerResponse()

    async def _handle_reaction_event(self, event_type: str, data: Any) -> None:
        """Fetch the reacted-to message; if it was sent by this bot, emit a synthetic text event."""
        if not self._client:
            return
        event = getattr(data, "event", None)
        message_id = str(getattr(event, "message_id", "") or "")
        if not message_id:
            return

        # Fetch the target message to verify it was sent by us and to obtain chat context.
        try:
            request = self._build_get_message_request(message_id)
            response = await asyncio.to_thread(self._client.im.v1.message.get, request)
            if not response or not getattr(response, "success", lambda: False)():
                return
            items = getattr(getattr(response, "data", None), "items", None) or []
            msg = items[0] if items else None
            if not msg:
                return
            sender = getattr(msg, "sender", None)
            sender_type = str(getattr(sender, "sender_type", "") or "").lower()
            if sender_type != "app":
                return  # only route reactions on our own bot messages
            chat_id = str(getattr(msg, "chat_id", "") or "")
            chat_type_raw = str(getattr(msg, "chat_type", "p2p") or "p2p")
            if not chat_id:
                return
        except Exception:
            logger.debug("[Feishu] Failed to fetch message for reaction routing", exc_info=True)
            return

        user_id_obj = getattr(event, "user_id", None)
        reaction_type_obj = getattr(event, "reaction_type", None)
        emoji_type = str(getattr(reaction_type_obj, "emoji_type", "") or "UNKNOWN")
        action = "added" if "created" in event_type else "removed"
        synthetic_text = f"reaction:{action}:{emoji_type}"

        sender_profile = await self._resolve_sender_profile(user_id_obj)
        chat_info = await self.get_chat_info(chat_id)
        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_info.get("name") or chat_id or "Feishu Chat",
            chat_type=self._resolve_source_chat_type(chat_info=chat_info, event_chat_type=chat_type_raw),
            user_id=sender_profile["user_id"],
            user_name=sender_profile["user_name"],
            thread_id=None,
            user_id_alt=sender_profile["user_id_alt"],
        )
        synthetic_event = MessageEvent(
            text=synthetic_text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=data,
            message_id=message_id,
            timestamp=datetime.now(),
        )
        logger.info("[Feishu] Routing reaction %s:%s on bot message %s as synthetic event", action, emoji_type, message_id)
        await self._handle_message_with_guards(synthetic_event)

    def _is_card_action_duplicate(self, token: str) -> bool:
        """Return True if this card action token was already processed within the dedup window."""
        now = time.time()
        # Prune expired tokens lazily each call.
        expired = [t for t, ts in self._card_action_tokens.items() if now - ts > _FEISHU_CARD_ACTION_DEDUP_TTL_SECONDS]
        for t in expired:
            del self._card_action_tokens[t]
        if token in self._card_action_tokens:
            return True
        self._card_action_tokens[token] = now
        return False

    async def _handle_card_action_event(self, data: Any) -> None:
        """Route Feishu interactive card button clicks as synthetic COMMAND events."""
        event = getattr(data, "event", None)
        token = str(getattr(event, "token", "") or "")
        if token and self._is_card_action_duplicate(token):
            logger.debug("[Feishu] Dropping duplicate card action token: %s", token)
            return

        context = getattr(event, "context", None)
        chat_id = str(getattr(context, "open_chat_id", "") or "")
        operator = getattr(event, "operator", None)
        open_id = str(getattr(operator, "open_id", "") or "")
        if not chat_id or not open_id:
            logger.debug("[Feishu] Card action missing chat_id or operator open_id, dropping")
            return

        action = getattr(event, "action", None)
        action_tag = str(getattr(action, "tag", "") or "button")
        action_value = getattr(action, "value", {}) or {}

        # --- Exec approval button intercept ---
        hermes_action = action_value.get("hermes_action") if isinstance(action_value, dict) else None
        if hermes_action:
            approval_id = action_value.get("approval_id")
            state = self._approval_state.pop(approval_id, None)
            if not state:
                logger.debug("[Feishu] Approval %s already resolved or unknown", approval_id)
                return

            choice_map = {
                "approve_once": "once",
                "approve_session": "session",
                "approve_always": "always",
                "deny": "deny",
            }
            choice = choice_map.get(hermes_action, "deny")

            label_map = {
                "once": "Approved once",
                "session": "Approved for session",
                "always": "Approved permanently",
                "deny": "Denied",
            }
            label = label_map.get(choice, "Resolved")

            # Resolve sender name for the status card
            sender_id = SimpleNamespace(open_id=open_id, user_id=None, union_id=None)
            sender_profile = await self._resolve_sender_profile(sender_id)
            user_name = sender_profile.get("user_name") or open_id

            # Resolve the approval — unblocks the agent thread
            try:
                from tools.approval import resolve_gateway_approval
                count = resolve_gateway_approval(state["session_key"], choice)
                logger.info(
                    "Feishu button resolved %d approval(s) for session %s (choice=%s, user=%s)",
                    count, state["session_key"], choice, user_name,
                )
            except Exception as exc:
                logger.error("Failed to resolve gateway approval from Feishu button: %s", exc)

            # Update the card to show the decision
            await self._update_approval_card(state.get("message_id", ""), label, user_name, choice)
            return

        synthetic_text = f"/card {action_tag}"
        if action_value:
            try:
                synthetic_text += f" {json.dumps(action_value, ensure_ascii=False)}"
            except Exception:
                pass

        sender_id = SimpleNamespace(open_id=open_id, user_id=None, union_id=None)
        sender_profile = await self._resolve_sender_profile(sender_id)
        chat_info = await self.get_chat_info(chat_id)
        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_info.get("name") or chat_id or "Feishu Chat",
            chat_type=self._resolve_source_chat_type(chat_info=chat_info, event_chat_type="group"),
            user_id=sender_profile["user_id"],
            user_name=sender_profile["user_name"],
            thread_id=None,
            user_id_alt=sender_profile["user_id_alt"],
        )
        synthetic_event = MessageEvent(
            text=synthetic_text,
            message_type=MessageType.COMMAND,
            source=source,
            raw_message=data,
            message_id=token or str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        logger.info("[Feishu] Routing card action %r from %s in %s as synthetic command", action_tag, open_id, chat_id)
        await self._handle_message_with_guards(synthetic_event)

    # =========================================================================
    # Per-chat serialization and typing indicator
    # =========================================================================

    def _get_chat_lock(self, chat_id: str) -> asyncio.Lock:
        """Return (creating if needed) the per-chat asyncio.Lock for serial message processing."""
        lock = self._chat_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._chat_locks[chat_id] = lock
        return lock

    async def _handle_message_with_guards(self, event: MessageEvent) -> None:
        """Dispatch a single event through the agent pipeline with per-chat serialization
        and a persistent ACK emoji reaction before processing starts.

        - Per-chat lock: ensures messages in the same chat are processed one at a time
          (matches openclaw's createChatQueue serial queue behaviour).
        - ACK indicator: adds a CHECK reaction to the triggering message before handing
          off to the agent and leaves it in place as a receipt marker.
        """
        chat_id = getattr(event.source, "chat_id", "") or "" if event.source else ""
        chat_lock = self._get_chat_lock(chat_id)
        async with chat_lock:
            message_id = event.message_id
            if message_id:
                await self._add_ack_reaction(message_id)
            await self.handle_message(event)

    async def _add_ack_reaction(self, message_id: str) -> Optional[str]:
        """Add a persistent ACK emoji reaction to signal the message was received."""
        if not self._client or not message_id:
            return None
        try:
            from lark_oapi.api.im.v1 import (  # lazy import — keeps optional dep optional
                CreateMessageReactionRequest,
                CreateMessageReactionRequestBody,
            )
            body = (
                CreateMessageReactionRequestBody.builder()
                .reaction_type({"emoji_type": _FEISHU_ACK_EMOJI})
                .build()
            )
            request = (
                CreateMessageReactionRequest.builder()
                .message_id(message_id)
                .request_body(body)
                .build()
            )
            response = await asyncio.to_thread(self._client.im.v1.message_reaction.create, request)
            if response and getattr(response, "success", lambda: False)():
                data = getattr(response, "data", None)
                return getattr(data, "reaction_id", None)
            logger.warning(
                "[Feishu] Failed to add ack reaction to %s: code=%s msg=%s",
                message_id,
                getattr(response, "code", None),
                getattr(response, "msg", None),
            )
        except Exception:
            logger.warning("[Feishu] Failed to add ack reaction to %s", message_id, exc_info=True)
        return None

    # =========================================================================
    # Webhook server and security
    # =========================================================================

    def _record_webhook_anomaly(self, remote_ip: str, status: str) -> None:
        """Increment the anomaly counter for remote_ip and emit a WARNING every threshold hits.

        Mirrors openclaw's createWebhookAnomalyTracker: TTL 6 hours, log every 25 consecutive
        error responses from the same IP.
        """
        now = time.time()
        entry = self._webhook_anomaly_counts.get(remote_ip)
        if entry is not None:
            count, _last_status, first_seen = entry
            if now - first_seen < _FEISHU_WEBHOOK_ANOMALY_TTL_SECONDS:
                count += 1
                if count % _FEISHU_WEBHOOK_ANOMALY_THRESHOLD == 0:
                    logger.warning(
                        "[Feishu] Webhook anomaly: %d consecutive error responses (%s) from %s "
                        "over the last %.0fs",
                        count,
                        status,
                        remote_ip,
                        now - first_seen,
                    )
                self._webhook_anomaly_counts[remote_ip] = (count, status, first_seen)
                return
        # Either first occurrence or TTL expired — start fresh.
        self._webhook_anomaly_counts[remote_ip] = (1, status, now)

    def _clear_webhook_anomaly(self, remote_ip: str) -> None:
        """Reset the anomaly counter for remote_ip after a successful request."""
        self._webhook_anomaly_counts.pop(remote_ip, None)

    # =========================================================================
    # Inbound processing pipeline
    # =========================================================================

    async def _process_inbound_message(
        self,
        *,
        data: Any,
        message: Any,
        sender_id: Any,
        chat_type: str,
        message_id: str,
    ) -> None:
        text, inbound_type, media_urls, media_types = await self._extract_message_content(message)
        if inbound_type == MessageType.TEXT and not text and not media_urls:
            logger.debug("[Feishu] Ignoring unsupported or empty message type: %s", getattr(message, "message_type", ""))
            return

        if inbound_type == MessageType.TEXT and text.startswith("/"):
            inbound_type = MessageType.COMMAND

        reply_to_message_id = (
            getattr(message, "parent_id", None)
            or getattr(message, "upper_message_id", None)
            or None
        )
        reply_to_text = await self._fetch_message_text(reply_to_message_id) if reply_to_message_id else None

        logger.info(
            "[Feishu] Inbound %s message received: id=%s type=%s chat_id=%s text=%r media=%d",
            "dm" if chat_type == "p2p" else "group",
            message_id,
            inbound_type.value,
            getattr(message, "chat_id", "") or "",
            text[:120],
            len(media_urls),
        )

        chat_id = getattr(message, "chat_id", "") or ""
        chat_info = await self.get_chat_info(chat_id)
        sender_profile = await self._resolve_sender_profile(sender_id)
        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_info.get("name") or chat_id or "Feishu Chat",
            chat_type=self._resolve_source_chat_type(chat_info=chat_info, event_chat_type=chat_type),
            user_id=sender_profile["user_id"],
            user_name=sender_profile["user_name"],
            thread_id=getattr(message, "thread_id", None) or None,
            user_id_alt=sender_profile["user_id_alt"],
        )
        normalized = MessageEvent(
            text=text,
            message_type=inbound_type,
            source=source,
            raw_message=data,
            message_id=message_id,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=reply_to_message_id,
            reply_to_text=reply_to_text,
            timestamp=datetime.now(),
        )
        await self._dispatch_inbound_event(normalized)

    async def _dispatch_inbound_event(self, event: MessageEvent) -> None:
        """Apply Feishu-specific burst protection before entering the base adapter."""
        if event.message_type == MessageType.TEXT and not event.is_command():
            await self._enqueue_text_event(event)
            return
        if self._should_batch_media_event(event):
            await self._enqueue_media_event(event)
            return
        await self._handle_message_with_guards(event)

    # =========================================================================
    # Media batching
    # =========================================================================

    def _should_batch_media_event(self, event: MessageEvent) -> bool:
        return bool(
            event.media_urls
            and event.message_type in {MessageType.PHOTO, MessageType.VIDEO, MessageType.DOCUMENT, MessageType.AUDIO}
        )

    def _media_batch_key(self, event: MessageEvent) -> str:
        from gateway.session import build_session_key

        session_key = build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )
        return f"{session_key}:media:{event.message_type.value}"

    @staticmethod
    def _media_batch_is_compatible(existing: MessageEvent, incoming: MessageEvent) -> bool:
        return (
            existing.message_type == incoming.message_type
            and existing.reply_to_message_id == incoming.reply_to_message_id
            and existing.reply_to_text == incoming.reply_to_text
            and existing.source.thread_id == incoming.source.thread_id
        )

    async def _enqueue_media_event(self, event: MessageEvent) -> None:
        key = self._media_batch_key(event)
        existing = self._pending_media_batches.get(key)
        if existing is None:
            self._pending_media_batches[key] = event
            self._schedule_media_batch_flush(key)
            return
        if not self._media_batch_is_compatible(existing, event):
            await self._flush_media_batch_now(key)
            self._pending_media_batches[key] = event
            self._schedule_media_batch_flush(key)
            return
        existing.media_urls.extend(event.media_urls)
        existing.media_types.extend(event.media_types)
        if event.text:
            existing.text = self._merge_caption(existing.text, event.text)
        existing.timestamp = event.timestamp
        if event.message_id:
            existing.message_id = event.message_id
        self._schedule_media_batch_flush(key)

    def _schedule_media_batch_flush(self, key: str) -> None:
        self._reschedule_batch_task(
            self._pending_media_batch_tasks,
            key,
            self._flush_media_batch,
        )

    async def _flush_media_batch(self, key: str) -> None:
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(self._media_batch_delay_seconds)
            await self._flush_media_batch_now(key)
        finally:
            if self._pending_media_batch_tasks.get(key) is current_task:
                self._pending_media_batch_tasks.pop(key, None)

    async def _flush_media_batch_now(self, key: str) -> None:
        event = self._pending_media_batches.pop(key, None)
        if not event:
            return
        logger.info(
            "[Feishu] Flushing media batch %s with %d attachment(s)",
            key,
            len(event.media_urls),
        )
        await self._handle_message_with_guards(event)

    async def _download_remote_image(self, image_url: str) -> str:
        ext = self._guess_remote_extension(image_url, default=".jpg")
        return await cache_image_from_url(image_url, ext=ext)

    async def _download_remote_document(
        self,
        file_url: str,
        *,
        default_ext: str,
        preferred_name: str,
    ) -> tuple[str, str]:
        from tools.url_safety import is_safe_url
        if not is_safe_url(file_url):
            raise ValueError(f"Blocked unsafe URL (SSRF protection): {file_url[:80]}")

        import httpx

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(
                file_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)",
                    "Accept": "*/*",
                },
            )
            response.raise_for_status()
        filename = self._derive_remote_filename(
            file_url,
            content_type=str(response.headers.get("Content-Type", "")),
            default_name=preferred_name,
            default_ext=default_ext,
        )
        cached_path = cache_document_from_bytes(response.content, filename)
        return cached_path, filename

    @staticmethod
    def _guess_remote_extension(url: str, *, default: str) -> str:
        ext = Path((url or "").split("?", 1)[0]).suffix.lower()
        return ext if ext in (_IMAGE_EXTENSIONS | _AUDIO_EXTENSIONS | _VIDEO_EXTENSIONS | set(SUPPORTED_DOCUMENT_TYPES)) else default

    @staticmethod
    def _derive_remote_filename(file_url: str, *, content_type: str, default_name: str, default_ext: str) -> str:
        candidate = Path((file_url or "").split("?", 1)[0]).name or default_name
        ext = Path(candidate).suffix.lower()
        if not ext:
            guessed = mimetypes.guess_extension((content_type or "").split(";", 1)[0].strip().lower() or "") or default_ext
            candidate = f"{candidate}{guessed}"
        return candidate

    @staticmethod
    def _namespace_from_mapping(value: Any) -> Any:
        if isinstance(value, dict):
            return SimpleNamespace(**{key: FeishuAdapter._namespace_from_mapping(item) for key, item in value.items()})
        if isinstance(value, list):
            return [FeishuAdapter._namespace_from_mapping(item) for item in value]
        return value

    async def _handle_webhook_request(self, request: Any) -> Any:
        remote_ip = (getattr(request, "remote", None) or "unknown")

        # Rate limiting — composite key: app_id:path:remote_ip (matches openclaw key structure).
        rate_key = f"{self._app_id}:{self._webhook_path}:{remote_ip}"
        if not self._check_webhook_rate_limit(rate_key):
            logger.warning("[Feishu] Webhook rate limit exceeded for %s", remote_ip)
            self._record_webhook_anomaly(remote_ip, "429")
            return web.Response(status=429, text="Too Many Requests")

        # Content-Type guard — Feishu always sends application/json.
        headers = getattr(request, "headers", {}) or {}
        content_type = str(headers.get("Content-Type", "") or "").split(";")[0].strip().lower()
        if content_type and content_type != "application/json":
            logger.warning("[Feishu] Webhook rejected: unexpected Content-Type %r from %s", content_type, remote_ip)
            self._record_webhook_anomaly(remote_ip, "415")
            return web.Response(status=415, text="Unsupported Media Type")

        # Body size guard — reject early via Content-Length when present.
        content_length = getattr(request, "content_length", None)
        if content_length is not None and content_length > _FEISHU_WEBHOOK_MAX_BODY_BYTES:
            logger.warning("[Feishu] Webhook body too large (%d bytes) from %s", content_length, remote_ip)
            self._record_webhook_anomaly(remote_ip, "413")
            return web.Response(status=413, text="Request body too large")

        try:
            body_bytes: bytes = await asyncio.wait_for(
                request.read(),
                timeout=_FEISHU_WEBHOOK_BODY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("[Feishu] Webhook body read timed out after %ds from %s", _FEISHU_WEBHOOK_BODY_TIMEOUT_SECONDS, remote_ip)
            self._record_webhook_anomaly(remote_ip, "408")
            return web.Response(status=408, text="Request Timeout")
        except Exception:
            self._record_webhook_anomaly(remote_ip, "400")
            return web.json_response({"code": 400, "msg": "failed to read body"}, status=400)

        if len(body_bytes) > _FEISHU_WEBHOOK_MAX_BODY_BYTES:
            logger.warning("[Feishu] Webhook body exceeds limit (%d bytes) from %s", len(body_bytes), remote_ip)
            self._record_webhook_anomaly(remote_ip, "413")
            return web.Response(status=413, text="Request body too large")

        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._record_webhook_anomaly(remote_ip, "400")
            return web.json_response({"code": 400, "msg": "invalid json"}, status=400)

        # URL verification challenge — respond before other checks so that Feishu's
        # subscription setup works even before encrypt_key is wired.
        if payload.get("type") == "url_verification":
            return web.json_response({"challenge": payload.get("challenge", "")})

        # Verification token check — second layer of defence beyond signature (matches openclaw).
        if self._verification_token:
            header = payload.get("header") or {}
            incoming_token = str(header.get("token") or payload.get("token") or "")
            if not incoming_token or not hmac.compare_digest(incoming_token, self._verification_token):
                logger.warning("[Feishu] Webhook rejected: invalid verification token from %s", remote_ip)
                self._record_webhook_anomaly(remote_ip, "401-token")
                return web.Response(status=401, text="Invalid verification token")

        # Timing-safe signature verification (only enforced when encrypt_key is set).
        if self._encrypt_key and not self._is_webhook_signature_valid(request.headers, body_bytes):
            logger.warning("[Feishu] Webhook rejected: invalid signature from %s", remote_ip)
            self._record_webhook_anomaly(remote_ip, "401-sig")
            return web.Response(status=401, text="Invalid signature")

        if payload.get("encrypt"):
            logger.error("[Feishu] Encrypted webhook payloads are not supported by Hermes webhook mode")
            self._record_webhook_anomaly(remote_ip, "400-encrypted")
            return web.json_response({"code": 400, "msg": "encrypted webhook payloads are not supported"}, status=400)

        self._clear_webhook_anomaly(remote_ip)

        event_type = str((payload.get("header") or {}).get("event_type") or "")
        data = self._namespace_from_mapping(payload)
        if event_type == "im.message.receive_v1":
            self._on_message_event(data)
        elif event_type == "im.message.message_read_v1":
            self._on_message_read_event(data)
        elif event_type == "im.chat.member.bot.added_v1":
            self._on_bot_added_to_chat(data)
        elif event_type == "im.chat.member.bot.deleted_v1":
            self._on_bot_removed_from_chat(data)
        elif event_type in ("im.message.reaction.created_v1", "im.message.reaction.deleted_v1"):
            self._on_reaction_event(event_type, data)
        elif event_type == "card.action.trigger":
            self._on_card_action_trigger(data)
        else:
            logger.debug("[Feishu] Ignoring webhook event type: %s", event_type or "unknown")
        return web.json_response({"code": 0, "msg": "ok"})

    def _is_webhook_signature_valid(self, headers: Any, body_bytes: bytes) -> bool:
        """Verify Feishu webhook signature using timing-safe comparison.

        Feishu signature algorithm:
            SHA256(timestamp + nonce + encrypt_key + body_string)
        Headers checked: x-lark-request-timestamp, x-lark-request-nonce, x-lark-signature.
        """
        timestamp = str(headers.get("x-lark-request-timestamp", "") or "")
        nonce = str(headers.get("x-lark-request-nonce", "") or "")
        signature = str(headers.get("x-lark-signature", "") or "")
        if not timestamp or not nonce or not signature:
            return False
        try:
            body_str = body_bytes.decode("utf-8", errors="replace")
            content = f"{timestamp}{nonce}{self._encrypt_key}{body_str}"
            computed = hashlib.sha256(content.encode("utf-8")).hexdigest()
            return hmac.compare_digest(computed, signature)
        except Exception:
            logger.debug("[Feishu] Signature verification raised an exception", exc_info=True)
            return False

    def _check_webhook_rate_limit(self, rate_key: str) -> bool:
        """Return False when the composite rate_key has exceeded _FEISHU_WEBHOOK_RATE_LIMIT_MAX.

        The rate_key is composed as "{app_id}:{path}:{remote_ip}" — matching openclaw's key
        structure so the limit is scoped to a specific (account, endpoint, IP) triple rather
        than a bare IP, which causes fewer false-positive denials in multi-tenant setups.

        The tracking dict is capped at _FEISHU_WEBHOOK_RATE_MAX_KEYS entries to prevent unbounded
        memory growth. Stale (expired) entries are pruned when the cap is reached.
        """
        now = time.time()
        # Fast path: existing entry within the current window.
        entry = self._webhook_rate_counts.get(rate_key)
        if entry is not None:
            count, window_start = entry
            if now - window_start < _FEISHU_WEBHOOK_RATE_WINDOW_SECONDS:
                if count >= _FEISHU_WEBHOOK_RATE_LIMIT_MAX:
                    return False
                self._webhook_rate_counts[rate_key] = (count + 1, window_start)
                return True
        # New window for an existing key, or a brand-new key — prune stale entries first.
        if len(self._webhook_rate_counts) >= _FEISHU_WEBHOOK_RATE_MAX_KEYS:
            stale_keys = [
                k for k, (_, ws) in self._webhook_rate_counts.items()
                if now - ws >= _FEISHU_WEBHOOK_RATE_WINDOW_SECONDS
            ]
            for k in stale_keys:
                del self._webhook_rate_counts[k]
            # If still at capacity after pruning, allow through without tracking.
            if rate_key not in self._webhook_rate_counts and len(self._webhook_rate_counts) >= _FEISHU_WEBHOOK_RATE_MAX_KEYS:
                return True
        self._webhook_rate_counts[rate_key] = (1, now)
        return True

    # =========================================================================
    # Text batching
    # =========================================================================

    def _text_batch_key(self, event: MessageEvent) -> str:
        """Return the session-scoped key used for Feishu text aggregation."""
        from gateway.session import build_session_key

        return build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )

    @staticmethod
    def _text_batch_is_compatible(existing: MessageEvent, incoming: MessageEvent) -> bool:
        """Only merge text events when reply/thread context is identical."""
        return (
            existing.reply_to_message_id == incoming.reply_to_message_id
            and existing.reply_to_text == incoming.reply_to_text
            and existing.source.thread_id == incoming.source.thread_id
        )

    async def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Debounce rapid Feishu text bursts into a single MessageEvent."""
        key = self._text_batch_key(event)
        chunk_len = len(event.text or "")
        existing = self._pending_text_batches.get(key)
        if existing is None:
            event._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            self._pending_text_batches[key] = event
            self._pending_text_batch_counts[key] = 1
            self._schedule_text_batch_flush(key)
            return

        if not self._text_batch_is_compatible(existing, event):
            await self._flush_text_batch_now(key)
            self._pending_text_batches[key] = event
            self._pending_text_batch_counts[key] = 1
            self._schedule_text_batch_flush(key)
            return

        existing_count = self._pending_text_batch_counts.get(key, 1)
        next_count = existing_count + 1
        appended_text = event.text or ""
        next_text = f"{existing.text}\n{appended_text}" if existing.text and appended_text else (existing.text or appended_text)
        if next_count > self._text_batch_max_messages or len(next_text) > self._text_batch_max_chars:
            await self._flush_text_batch_now(key)
            self._pending_text_batches[key] = event
            self._pending_text_batch_counts[key] = 1
            self._schedule_text_batch_flush(key)
            return

        existing.text = next_text
        existing._last_chunk_len = chunk_len  # type: ignore[attr-defined]
        existing.timestamp = event.timestamp
        if event.message_id:
            existing.message_id = event.message_id
        self._pending_text_batch_counts[key] = next_count
        self._schedule_text_batch_flush(key)

    def _schedule_text_batch_flush(self, key: str) -> None:
        """Reset the debounce timer for a pending Feishu text batch."""
        self._reschedule_batch_task(
            self._pending_text_batch_tasks,
            key,
            self._flush_text_batch,
        )

    @staticmethod
    def _reschedule_batch_task(
        task_map: Dict[str, asyncio.Task],
        key: str,
        flush_fn: Any,
    ) -> None:
        prior_task = task_map.get(key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        task_map[key] = asyncio.create_task(flush_fn(key))

    async def _flush_text_batch(self, key: str) -> None:
        """Flush a pending text batch after the quiet period.

        Uses a longer delay when the latest chunk is near Feishu's ~4096-char
        split point, since a continuation chunk is almost certain.
        """
        current_task = asyncio.current_task()
        try:
            # Adaptive delay: if the latest chunk is near the split threshold,
            # a continuation is almost certain — wait longer.
            pending = self._pending_text_batches.get(key)
            last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
            if last_len >= self._SPLIT_THRESHOLD:
                delay = self._text_batch_split_delay_seconds
            else:
                delay = self._text_batch_delay_seconds
            await asyncio.sleep(delay)
            await self._flush_text_batch_now(key)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    async def _flush_text_batch_now(self, key: str) -> None:
        """Dispatch the current text batch immediately."""
        event = self._pending_text_batches.pop(key, None)
        self._pending_text_batch_counts.pop(key, None)
        if not event:
            return
        logger.info(
            "[Feishu] Flushing text batch %s (%d chars)",
            key,
            len(event.text or ""),
        )
        await self._handle_message_with_guards(event)

    # =========================================================================
    # Message content extraction and resource download
    # =========================================================================

    async def _extract_message_content(self, message: Any) -> tuple[str, MessageType, List[str], List[str]]:
        """Extract text and cached media from a normalized Feishu message."""
        raw_content = getattr(message, "content", "") or ""
        raw_type = getattr(message, "message_type", "") or ""
        message_id = str(getattr(message, "message_id", "") or "")
        logger.info("[Feishu] Received raw message type=%s message_id=%s", raw_type, message_id)

        normalized = normalize_feishu_message(message_type=raw_type, raw_content=raw_content)
        media_urls, media_types = await self._download_feishu_message_resources(
            message_id=message_id,
            normalized=normalized,
        )
        inbound_type = self._resolve_normalized_message_type(normalized, media_types)
        text = normalized.text_content

        if (
            inbound_type in {MessageType.DOCUMENT, MessageType.AUDIO, MessageType.VIDEO, MessageType.PHOTO}
            and len(media_urls) == 1
            and normalized.preferred_message_type in {"document", "audio"}
        ):
            injected = await self._maybe_extract_text_document(media_urls[0], media_types[0])
            if injected:
                text = injected

        return text, inbound_type, media_urls, media_types

    async def _download_feishu_message_resources(
        self,
        *,
        message_id: str,
        normalized: FeishuNormalizedMessage,
    ) -> tuple[List[str], List[str]]:
        media_urls: List[str] = []
        media_types: List[str] = []

        for image_key in normalized.image_keys:
            cached_path, media_type = await self._download_feishu_image(
                message_id=message_id,
                image_key=image_key,
            )
            if cached_path:
                media_urls.append(cached_path)
                media_types.append(media_type)

        for media_ref in normalized.media_refs:
            cached_path, media_type = await self._download_feishu_message_resource(
                message_id=message_id,
                file_key=media_ref.file_key,
                resource_type=media_ref.resource_type,
                fallback_filename=media_ref.file_name,
            )
            if cached_path:
                media_urls.append(cached_path)
                media_types.append(media_type)

        return media_urls, media_types

    @staticmethod
    def _resolve_media_message_type(media_type: str, *, default: MessageType) -> MessageType:
        normalized = (media_type or "").lower()
        if normalized.startswith("image/"):
            return MessageType.PHOTO
        if normalized.startswith("audio/"):
            return MessageType.AUDIO
        if normalized.startswith("video/"):
            return MessageType.VIDEO
        return default

    def _resolve_normalized_message_type(
        self,
        normalized: FeishuNormalizedMessage,
        media_types: List[str],
    ) -> MessageType:
        preferred = normalized.preferred_message_type
        if preferred == "photo":
            return self._resolve_media_message_type(media_types[0] if media_types else "", default=MessageType.PHOTO)
        if preferred == "audio":
            return self._resolve_media_message_type(media_types[0] if media_types else "", default=MessageType.AUDIO)
        if preferred == "document":
            return self._resolve_media_message_type(media_types[0] if media_types else "", default=MessageType.DOCUMENT)
        return MessageType.TEXT

    def _normalize_inbound_text(self, text: str) -> str:
        """Strip Feishu mention placeholders from inbound text."""
        text = _MENTION_RE.sub(" ", text or "")
        text = _MULTISPACE_RE.sub(" ", text)
        return text.strip()

    async def _maybe_extract_text_document(self, cached_path: str, media_type: str) -> str:
        if not cached_path or not media_type.startswith("text/"):
            return ""
        try:
            if os.path.getsize(cached_path) > _MAX_TEXT_INJECT_BYTES:
                return ""
            ext = Path(cached_path).suffix.lower()
            if ext not in {".txt", ".md"} and media_type not in {"text/plain", "text/markdown"}:
                return ""
            content = Path(cached_path).read_text(encoding="utf-8")
            display_name = self._display_name_from_cached_path(cached_path)
            return f"[Content of {display_name}]:\n{content}"
        except (OSError, UnicodeDecodeError):
            logger.warning("[Feishu] Failed to inject text document content from %s", cached_path, exc_info=True)
            return ""

    async def _download_feishu_image(self, *, message_id: str, image_key: str) -> tuple[str, str]:
        if not self._client or not message_id:
            return "", ""
        try:
            request = self._build_message_resource_request(
                message_id=message_id,
                file_key=image_key,
                resource_type="image",
            )
            response = await asyncio.to_thread(self._client.im.v1.message_resource.get, request)
            if not response or not response.success():
                logger.warning(
                    "[Feishu] Failed to download image %s: %s %s",
                    image_key,
                    getattr(response, "code", "unknown"),
                    getattr(response, "msg", "request failed"),
                )
                return "", ""
            raw_bytes = self._read_binary_response(response)
            if not raw_bytes:
                return "", ""
            content_type = self._get_response_header(response, "Content-Type")
            filename = getattr(response, "file_name", None) or f"{image_key}.jpg"
            ext = self._guess_extension(filename, content_type, ".jpg", allowed=_IMAGE_EXTENSIONS)
            cached_path = cache_image_from_bytes(raw_bytes, ext=ext)
            media_type = self._normalize_media_type(content_type, default=self._default_image_media_type(ext))
            return cached_path, media_type
        except Exception:
            logger.warning("[Feishu] Failed to cache image resource %s", image_key, exc_info=True)
            return "", ""

    async def _download_feishu_message_resource(
        self,
        *,
        message_id: str,
        file_key: str,
        resource_type: str,
        fallback_filename: str,
    ) -> tuple[str, str]:
        if not self._client or not message_id:
            return "", ""

        request_types = [resource_type]
        if resource_type in {"audio", "media"}:
            request_types.append("file")

        for request_type in request_types:
            try:
                request = self._build_message_resource_request(
                    message_id=message_id,
                    file_key=file_key,
                    resource_type=request_type,
                )
                response = await asyncio.to_thread(self._client.im.v1.message_resource.get, request)
                if not response or not response.success():
                    logger.debug(
                        "[Feishu] Resource download failed for %s/%s via type=%s: %s %s",
                        message_id,
                        file_key,
                        request_type,
                        getattr(response, "code", "unknown"),
                        getattr(response, "msg", "request failed"),
                    )
                    continue

                raw_bytes = self._read_binary_response(response)
                if not raw_bytes:
                    continue
                content_type = self._get_response_header(response, "Content-Type")
                response_filename = getattr(response, "file_name", None) or ""
                filename = response_filename or fallback_filename or f"{request_type}_{file_key}"
                media_type = self._normalize_media_type(
                    content_type,
                    default=self._guess_media_type_from_filename(filename),
                )

                if media_type.startswith("image/"):
                    ext = self._guess_extension(filename, content_type, ".jpg", allowed=_IMAGE_EXTENSIONS)
                    cached_path = cache_image_from_bytes(raw_bytes, ext=ext)
                    logger.info("[Feishu] Cached message image resource at %s", cached_path)
                    return cached_path, media_type or self._default_image_media_type(ext)

                if request_type == "audio" or media_type.startswith("audio/"):
                    ext = self._guess_extension(filename, content_type, ".ogg", allowed=_AUDIO_EXTENSIONS)
                    cached_path = cache_audio_from_bytes(raw_bytes, ext=ext)
                    logger.info("[Feishu] Cached message audio resource at %s", cached_path)
                    return cached_path, (media_type or f"audio/{ext.lstrip('.') or 'ogg'}")

                if media_type.startswith("video/"):
                    if not Path(filename).suffix:
                        filename = f"{filename}.mp4"
                    cached_path = cache_document_from_bytes(raw_bytes, filename)
                    logger.info("[Feishu] Cached message video resource at %s", cached_path)
                    return cached_path, media_type

                if not Path(filename).suffix and media_type in _DOCUMENT_MIME_TO_EXT:
                    filename = f"{filename}{_DOCUMENT_MIME_TO_EXT[media_type]}"
                cached_path = cache_document_from_bytes(raw_bytes, filename)
                logger.info("[Feishu] Cached message document resource at %s", cached_path)
                return cached_path, (media_type or self._guess_document_media_type(filename))
            except Exception:
                logger.warning(
                    "[Feishu] Failed to cache message resource %s/%s",
                    message_id,
                    file_key,
                    exc_info=True,
                )
        return "", ""

    # =========================================================================
    # Static helpers — extension / media-type guessing
    # =========================================================================

    @staticmethod
    def _read_binary_response(response: Any) -> bytes:
        file_obj = getattr(response, "file", None)
        if file_obj is None:
            return b""
        if hasattr(file_obj, "getvalue"):
            return bytes(file_obj.getvalue())
        return bytes(file_obj.read())

    @staticmethod
    def _get_response_header(response: Any, name: str) -> str:
        raw = getattr(response, "raw", None)
        headers = getattr(raw, "headers", {}) or {}
        return str(headers.get(name, headers.get(name.lower(), "")) or "").split(";", 1)[0].strip().lower()

    @staticmethod
    def _guess_extension(filename: str, content_type: str, default: str, *, allowed: set[str]) -> str:
        ext = Path(filename or "").suffix.lower()
        if ext in allowed:
            return ext
        guessed = mimetypes.guess_extension((content_type or "").split(";", 1)[0].strip().lower() or "")
        if guessed in allowed:
            return guessed
        return default

    @staticmethod
    def _normalize_media_type(content_type: str, *, default: str) -> str:
        normalized = (content_type or "").split(";", 1)[0].strip().lower()
        return normalized or default

    @staticmethod
    def _guess_document_media_type(filename: str) -> str:
        ext = Path(filename or "").suffix.lower()
        return SUPPORTED_DOCUMENT_TYPES.get(ext, mimetypes.guess_type(filename or "")[0] or "application/octet-stream")

    @staticmethod
    def _display_name_from_cached_path(path: str) -> str:
        basename = os.path.basename(path)
        parts = basename.split("_", 2)
        display_name = parts[2] if len(parts) >= 3 else basename
        return re.sub(r"[^\w.\- ]", "_", display_name)

    @staticmethod
    def _guess_media_type_from_filename(filename: str) -> str:
        guessed = (mimetypes.guess_type(filename or "")[0] or "").lower()
        if guessed:
            return guessed
        ext = Path(filename or "").suffix.lower()
        if ext in _VIDEO_EXTENSIONS:
            return f"video/{ext.lstrip('.')}"
        if ext in _AUDIO_EXTENSIONS:
            return f"audio/{ext.lstrip('.')}"
        if ext in _IMAGE_EXTENSIONS:
            return FeishuAdapter._default_image_media_type(ext)
        return ""

    @staticmethod
    def _map_chat_type(raw_chat_type: str) -> str:
        normalized = (raw_chat_type or "").strip().lower()
        if normalized == "p2p":
            return "dm"
        if "topic" in normalized or "thread" in normalized or "forum" in normalized:
            return "forum"
        if normalized == "group":
            return "group"
        return "dm"

    @staticmethod
    def _resolve_source_chat_type(*, chat_info: Dict[str, Any], event_chat_type: str) -> str:
        resolved = str(chat_info.get("type") or "").strip().lower()
        if resolved in {"group", "forum"}:
            return resolved
        if event_chat_type == "p2p":
            return "dm"
        return "group"

    async def _resolve_sender_profile(self, sender_id: Any) -> Dict[str, Optional[str]]:
        open_id = getattr(sender_id, "open_id", None) or None
        user_id = getattr(sender_id, "user_id", None) or None
        union_id = getattr(sender_id, "union_id", None) or None
        primary_id = open_id or user_id
        display_name = await self._resolve_sender_name_from_api(primary_id or union_id)
        return {
            "user_id": primary_id,
            "user_name": display_name,
            "user_id_alt": union_id,
        }

    async def _resolve_sender_name_from_api(self, sender_id: Optional[str]) -> Optional[str]:
        """Fetch the sender's display name from the Feishu contact API with a 10-minute cache.

        ID-type detection mirrors openclaw: ou_ → open_id, on_ → union_id, else user_id.
        Failures are silently suppressed; the message pipeline must not block on name resolution.
        """
        if not sender_id or not self._client:
            return None
        trimmed = sender_id.strip()
        if not trimmed:
            return None
        now = time.time()
        cached = self._sender_name_cache.get(trimmed)
        if cached is not None:
            name, expire_at = cached
            if now < expire_at:
                return name
        try:
            from lark_oapi.api.contact.v3 import GetUserRequest  # lazy import
            if trimmed.startswith("ou_"):
                id_type = "open_id"
            elif trimmed.startswith("on_"):
                id_type = "union_id"
            else:
                id_type = "user_id"
            request = GetUserRequest.builder().user_id(trimmed).user_id_type(id_type).build()
            response = await asyncio.to_thread(self._client.contact.v3.user.get, request)
            if not response or not response.success():
                return None
            user = getattr(getattr(response, "data", None), "user", None)
            name = (
                getattr(user, "name", None)
                or getattr(user, "display_name", None)
                or getattr(user, "nickname", None)
                or getattr(user, "en_name", None)
            )
            if name and isinstance(name, str):
                name = name.strip()
                if name:
                    self._sender_name_cache[trimmed] = (name, now + _FEISHU_SENDER_NAME_TTL_SECONDS)
                    return name
        except Exception:
            logger.debug("[Feishu] Failed to resolve sender name for %s", sender_id, exc_info=True)
        return None

    async def _fetch_message_text(self, message_id: str) -> Optional[str]:
        if not self._client or not message_id:
            return None
        if message_id in self._message_text_cache:
            return self._message_text_cache[message_id]
        try:
            request = self._build_get_message_request(message_id)
            response = await asyncio.to_thread(self._client.im.v1.message.get, request)
            if not response or getattr(response, "success", lambda: False)() is False:
                code = getattr(response, "code", "unknown")
                msg = getattr(response, "msg", "message lookup failed")
                logger.warning("[Feishu] Failed to fetch parent message %s: [%s] %s", message_id, code, msg)
                return None
            items = getattr(getattr(response, "data", None), "items", None) or []
            parent = items[0] if items else None
            body = getattr(parent, "body", None)
            msg_type = getattr(parent, "msg_type", "") or ""
            raw_content = getattr(body, "content", "") or ""
            text = self._extract_text_from_raw_content(msg_type=msg_type, raw_content=raw_content)
            self._message_text_cache[message_id] = text
            return text
        except Exception:
            logger.warning("[Feishu] Failed to fetch parent message %s", message_id, exc_info=True)
            return None

    def _extract_text_from_raw_content(self, *, msg_type: str, raw_content: str) -> Optional[str]:
        normalized = normalize_feishu_message(message_type=msg_type, raw_content=raw_content)
        if normalized.text_content:
            return normalized.text_content
        placeholder = normalized.metadata.get("placeholder_text") if isinstance(normalized.metadata, dict) else None
        return str(placeholder).strip() or None

    @staticmethod
    def _default_image_media_type(ext: str) -> str:
        normalized_ext = (ext or "").lower()
        if normalized_ext in {".jpg", ".jpeg"}:
            return "image/jpeg"
        return f"image/{normalized_ext.lstrip('.') or 'jpeg'}"

    @staticmethod
    def _log_background_failure(future: Any) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("[Feishu] Background inbound processing failed")

    # =========================================================================
    # Group policy and mention gating
    # =========================================================================

    def _allow_group_message(self, sender_id: Any, chat_id: str = "") -> bool:
        """Per-group policy gate for non-DM traffic."""
        sender_open_id = getattr(sender_id, "open_id", None)
        sender_user_id = getattr(sender_id, "user_id", None)
        sender_ids = {sender_open_id, sender_user_id} - {None}

        if sender_ids and self._admins and (sender_ids & self._admins):
            return True

        rule = self._group_rules.get(chat_id) if chat_id else None
        if rule:
            policy = rule.policy
            allowlist = rule.allowlist
            blacklist = rule.blacklist
        else:
            policy = self._default_group_policy or self._group_policy
            allowlist = self._allowed_group_users
            blacklist = set()

        if policy == "disabled":
            return False
        if policy == "open":
            return True
        if policy == "admin_only":
            return False
        if policy == "allowlist":
            return bool(sender_ids and (sender_ids & allowlist))
        if policy == "blacklist":
            return bool(sender_ids and not (sender_ids & blacklist))

        return bool(sender_ids and (sender_ids & self._allowed_group_users))

    def _should_accept_group_message(self, message: Any, sender_id: Any, chat_id: str = "") -> bool:
        """Require an explicit @mention before group messages enter the agent."""
        if not self._allow_group_message(sender_id, chat_id):
            return False
        # @_all is Feishu's @everyone placeholder — always route to the bot.
        raw_content = getattr(message, "content", "") or ""
        if "@_all" in raw_content:
            return True
        mentions = getattr(message, "mentions", None) or []
        if mentions:
            return self._message_mentions_bot(mentions)
        normalized = normalize_feishu_message(
            message_type=getattr(message, "message_type", "") or "",
            raw_content=raw_content,
        )
        if normalized.mentioned_ids:
            return self._post_mentions_bot(normalized.mentioned_ids)
        return False

    def _message_mentions_bot(self, mentions: List[Any]) -> bool:
        """Check whether any mention targets the configured or inferred bot identity."""
        for mention in mentions:
            mention_id = getattr(mention, "id", None)
            mention_open_id = getattr(mention_id, "open_id", None)
            mention_user_id = getattr(mention_id, "user_id", None)
            mention_name = (getattr(mention, "name", None) or "").strip()

            if self._bot_open_id and mention_open_id == self._bot_open_id:
                return True
            if self._bot_user_id and mention_user_id == self._bot_user_id:
                return True
            if self._bot_name and mention_name == self._bot_name:
                return True

        return False

    def _post_mentions_bot(self, mentioned_ids: List[str]) -> bool:
        if not mentioned_ids:
            return False
        if self._bot_open_id and self._bot_open_id in mentioned_ids:
            return True
        if self._bot_user_id and self._bot_user_id in mentioned_ids:
            return True
        return False

    async def _hydrate_bot_identity(self) -> None:
        """Best-effort discovery of bot identity for precise group mention gating."""
        if not self._client:
            return
        if any((self._bot_open_id, self._bot_user_id, self._bot_name)):
            return
        try:
            request = self._build_get_application_request(app_id=self._app_id, lang="en_us")
            response = await asyncio.to_thread(self._client.application.v6.application.get, request)
            if not response or not response.success():
                code = getattr(response, "code", None)
                if code == 99991672:
                    logger.warning(
                        "[Feishu] Unable to hydrate bot identity from application info. "
                        "Grant admin:app.info:readonly or application:application:self_manage "
                        "so group @mention gating can resolve the bot name precisely."
                    )
                return
            app = getattr(getattr(response, "data", None), "app", None)
            app_name = (getattr(app, "app_name", None) or "").strip()
            if app_name:
                self._bot_name = app_name
        except Exception:
            logger.debug("[Feishu] Failed to hydrate bot identity", exc_info=True)

    # =========================================================================
    # Deduplication — seen message ID cache (persistent)
    # =========================================================================

    def _load_seen_message_ids(self) -> None:
        try:
            payload = json.loads(self._dedup_state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            logger.warning("[Feishu] Failed to load persisted dedup state from %s", self._dedup_state_path, exc_info=True)
            return
        seen_data = payload.get("message_ids", {}) if isinstance(payload, dict) else {}
        now = time.time()
        ttl = _FEISHU_DEDUP_TTL_SECONDS
        # Backward-compat: old format stored a plain list of IDs (no timestamps).
        if isinstance(seen_data, list):
            entries: Dict[str, float] = {str(item).strip(): 0.0 for item in seen_data if str(item).strip()}
        elif isinstance(seen_data, dict):
            entries = {k: float(v) for k, v in seen_data.items() if isinstance(k, str) and k.strip()}
        else:
            return
        # Filter out TTL-expired entries (entries saved with ts=0.0 are treated as immortal
        # for one migration cycle to avoid nuking old data on first upgrade).
        valid: Dict[str, float] = {
            msg_id: ts for msg_id, ts in entries.items()
            if ts == 0.0 or ttl <= 0 or now - ts < ttl
        }
        # Apply size cap; keep the most recently seen IDs.
        sorted_ids = sorted(valid, key=lambda k: valid[k], reverse=True)[:self._dedup_cache_size]
        self._seen_message_order = list(reversed(sorted_ids))
        self._seen_message_ids = {k: valid[k] for k in sorted_ids}

    def _persist_seen_message_ids(self) -> None:
        try:
            self._dedup_state_path.parent.mkdir(parents=True, exist_ok=True)
            recent = self._seen_message_order[-self._dedup_cache_size:]
            # Save as {msg_id: timestamp} so TTL filtering works across restarts.
            payload = {"message_ids": {k: self._seen_message_ids[k] for k in recent if k in self._seen_message_ids}}
            self._dedup_state_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            logger.warning("[Feishu] Failed to persist dedup state to %s", self._dedup_state_path, exc_info=True)

    def _is_duplicate(self, message_id: str) -> bool:
        now = time.time()
        ttl = _FEISHU_DEDUP_TTL_SECONDS
        with self._dedup_lock:
            seen_at = self._seen_message_ids.get(message_id)
            if seen_at is not None and (ttl <= 0 or now - seen_at < ttl):
                return True
            # Record with current wall-clock timestamp so TTL works across restarts.
            self._seen_message_ids[message_id] = now
            self._seen_message_order.append(message_id)
            while len(self._seen_message_order) > self._dedup_cache_size:
                stale = self._seen_message_order.pop(0)
                self._seen_message_ids.pop(stale, None)
            self._persist_seen_message_ids()
            return False

    # =========================================================================
    # Outbound payload construction and send pipeline
    # =========================================================================

    def _build_outbound_payload(self, content: str) -> tuple[str, str]:
        if _MARKDOWN_HINT_RE.search(content):
            return "post", _build_markdown_post_payload(content)
        text_payload = {"text": content}
        return "text", json.dumps(text_payload, ensure_ascii=False)

    async def _send_uploaded_file_message(
        self,
        *,
        chat_id: str,
        file_path: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        outbound_message_type: str = "file",
    ) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Not connected")
        if not os.path.exists(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")

        display_name = file_name or os.path.basename(file_path)
        upload_file_type, resolved_message_type = self._resolve_outbound_file_routing(
            file_path=display_name,
            requested_message_type=outbound_message_type,
        )
        try:
            with open(file_path, "rb") as file_obj:
                body = self._build_file_upload_body(
                    file_type=upload_file_type,
                    file_name=display_name,
                    file=file_obj,
                )
                request = self._build_file_upload_request(body)
                upload_response = await asyncio.to_thread(self._client.im.v1.file.create, request)
            file_key = self._extract_response_field(upload_response, "file_key")
            if not file_key:
                return self._response_error_result(
                    upload_response,
                    default_message="file upload failed",
                    override_error="Feishu file upload missing file_key",
                )

            if caption:
                media_tag = {
                    "tag": "media",
                    "file_key": file_key,
                    "file_name": display_name,
                }
                message_response = await self._feishu_send_with_retry(
                    chat_id=chat_id,
                    msg_type="post",
                    payload=self._build_media_post_payload(caption=caption, media_tag=media_tag),
                    reply_to=reply_to,
                    metadata=metadata,
                )
            else:
                message_response = await self._feishu_send_with_retry(
                    chat_id=chat_id,
                    msg_type=resolved_message_type,
                    payload=json.dumps({"file_key": file_key}, ensure_ascii=False),
                    reply_to=reply_to,
                    metadata=metadata,
                )
            return self._finalize_send_result(message_response, "file send failed")
        except Exception as exc:
            logger.error("[Feishu] Failed to send file %s: %s", file_path, exc, exc_info=True)
            return SendResult(success=False, error=str(exc))

    async def _send_raw_message(
        self,
        *,
        chat_id: str,
        msg_type: str,
        payload: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Any:
        reply_in_thread = bool((metadata or {}).get("thread_id"))
        if reply_to:
            body = self._build_reply_message_body(
                content=payload,
                msg_type=msg_type,
                reply_in_thread=reply_in_thread,
                uuid_value=str(uuid.uuid4()),
            )
            request = self._build_reply_message_request(reply_to, body)
            return await asyncio.to_thread(self._client.im.v1.message.reply, request)

        body = self._build_create_message_body(
            receive_id=chat_id,
            msg_type=msg_type,
            content=payload,
            uuid_value=str(uuid.uuid4()),
        )
        request = self._build_create_message_request("chat_id", body)
        return await asyncio.to_thread(self._client.im.v1.message.create, request)

    @staticmethod
    def _response_succeeded(response: Any) -> bool:
        return bool(response and getattr(response, "success", lambda: False)())

    @staticmethod
    def _extract_response_field(response: Any, field_name: str) -> Any:
        if not FeishuAdapter._response_succeeded(response):
            return None
        data = getattr(response, "data", None)
        return getattr(data, field_name, None) if data else None

    def _response_error_result(
        self,
        response: Any,
        *,
        default_message: str,
        override_error: Optional[str] = None,
    ) -> SendResult:
        if override_error:
            return SendResult(success=False, error=override_error, raw_response=response)
        code = getattr(response, "code", "unknown")
        msg = getattr(response, "msg", default_message)
        return SendResult(success=False, error=f"[{code}] {msg}", raw_response=response)

    def _finalize_send_result(self, response: Any, default_message: str) -> SendResult:
        if not self._response_succeeded(response):
            return self._response_error_result(response, default_message=default_message)
        return SendResult(
            success=True,
            message_id=self._extract_response_field(response, "message_id"),
            raw_response=response,
        )

    # =========================================================================
    # Connection internals — websocket / webhook setup
    # =========================================================================

    async def _connect_with_retry(self) -> None:
        for attempt in range(_FEISHU_CONNECT_ATTEMPTS):
            try:
                if self._connection_mode == "websocket":
                    await self._connect_websocket()
                else:
                    await self._connect_webhook()
                return
            except Exception as exc:
                self._running = False
                self._disable_websocket_auto_reconnect()
                self._ws_future = None
                await self._stop_webhook_server()
                if attempt >= _FEISHU_CONNECT_ATTEMPTS - 1:
                    raise
                wait_seconds = 2 ** attempt
                logger.warning(
                    "[Feishu] Connect attempt %d/%d failed; retrying in %ds: %s",
                    attempt + 1,
                    _FEISHU_CONNECT_ATTEMPTS,
                    wait_seconds,
                    exc,
                )
                await asyncio.sleep(wait_seconds)

    async def _connect_websocket(self) -> None:
        if not FEISHU_WEBSOCKET_AVAILABLE:
            raise RuntimeError("websockets not installed; websocket mode unavailable")
        domain = FEISHU_DOMAIN if self._domain_name != "lark" else LARK_DOMAIN
        self._client = self._build_lark_client(domain)
        self._event_handler = self._build_event_handler()
        if self._event_handler is None:
            raise RuntimeError("failed to build Feishu event handler")
        loop = self._loop
        if loop is None or loop.is_closed():
            raise RuntimeError("adapter loop is not ready")
        await self._hydrate_bot_identity()
        self._ws_client = FeishuWSClient(
            app_id=self._app_id,
            app_secret=self._app_secret,
            log_level=lark.LogLevel.INFO,
            event_handler=self._event_handler,
            domain=domain,
        )
        self._ws_future = loop.run_in_executor(
            None,
            _run_official_feishu_ws_client,
            self._ws_client,
            self,
        )

    async def _connect_webhook(self) -> None:
        if not FEISHU_WEBHOOK_AVAILABLE:
            raise RuntimeError("aiohttp not installed; webhook mode unavailable")
        domain = FEISHU_DOMAIN if self._domain_name != "lark" else LARK_DOMAIN
        self._client = self._build_lark_client(domain)
        self._event_handler = self._build_event_handler()
        if self._event_handler is None:
            raise RuntimeError("failed to build Feishu event handler")
        await self._hydrate_bot_identity()
        app = web.Application()
        app.router.add_post(self._webhook_path, self._handle_webhook_request)
        self._webhook_runner = web.AppRunner(app)
        await self._webhook_runner.setup()
        self._webhook_site = web.TCPSite(self._webhook_runner, self._webhook_host, self._webhook_port)
        await self._webhook_site.start()

    def _build_lark_client(self, domain: Any) -> Any:
        return (
            lark.Client.builder()
            .app_id(self._app_id)
            .app_secret(self._app_secret)
            .domain(domain)
            .log_level(lark.LogLevel.WARNING)
            .build()
        )

    async def _feishu_send_with_retry(
        self,
        *,
        chat_id: str,
        msg_type: str,
        payload: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Any:
        last_error: Optional[Exception] = None
        active_reply_to = reply_to
        for attempt in range(_FEISHU_SEND_ATTEMPTS):
            try:
                response = await self._send_raw_message(
                    chat_id=chat_id,
                    msg_type=msg_type,
                    payload=payload,
                    reply_to=active_reply_to,
                    metadata=metadata,
                )
                # If replying to a message failed because it was withdrawn or not found,
                # fall back to posting a new message directly to the chat.
                if active_reply_to and not self._response_succeeded(response):
                    code = getattr(response, "code", None)
                    if code in _FEISHU_REPLY_FALLBACK_CODES:
                        logger.warning(
                            "[Feishu] Reply to %s failed (code %s — message withdrawn/missing); "
                            "falling back to new message in chat %s",
                            active_reply_to,
                            code,
                            chat_id,
                        )
                        active_reply_to = None
                        response = await self._send_raw_message(
                            chat_id=chat_id,
                            msg_type=msg_type,
                            payload=payload,
                            reply_to=None,
                            metadata=metadata,
                        )
                return response
            except Exception as exc:
                last_error = exc
                if msg_type == "post" and _POST_CONTENT_INVALID_RE.search(str(exc)):
                    raise
                if attempt >= _FEISHU_SEND_ATTEMPTS - 1:
                    raise
                wait_seconds = 2 ** attempt
                logger.warning(
                    "[Feishu] Send attempt %d/%d failed for chat %s; retrying in %ds: %s",
                    attempt + 1,
                    _FEISHU_SEND_ATTEMPTS,
                    chat_id,
                    wait_seconds,
                    exc,
                )
                await asyncio.sleep(wait_seconds)
        raise last_error or RuntimeError("Feishu send failed")

    async def _release_app_lock(self) -> None:
        if not self._app_lock_identity:
            return
        try:
            release_scoped_lock(_FEISHU_APP_LOCK_SCOPE, self._app_lock_identity)
        except Exception as exc:
            logger.warning("[Feishu] Failed to release app lock: %s", exc, exc_info=True)
        finally:
            self._app_lock_identity = None

    # =========================================================================
    # Lark API request builders
    # =========================================================================

    @staticmethod
    def _build_get_chat_request(chat_id: str) -> Any:
        if "GetChatRequest" in globals():
            return GetChatRequest.builder().chat_id(chat_id).build()
        return SimpleNamespace(chat_id=chat_id)

    @staticmethod
    def _build_get_message_request(message_id: str) -> Any:
        if "GetMessageRequest" in globals():
            return GetMessageRequest.builder().message_id(message_id).build()
        return SimpleNamespace(message_id=message_id)

    @staticmethod
    def _build_message_resource_request(*, message_id: str, file_key: str, resource_type: str) -> Any:
        if "GetMessageResourceRequest" in globals():
            return (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(file_key)
                .type(resource_type)
                .build()
            )
        return SimpleNamespace(message_id=message_id, file_key=file_key, type=resource_type)

    @staticmethod
    def _build_get_application_request(*, app_id: str, lang: str) -> Any:
        if "GetApplicationRequest" in globals():
            return (
                GetApplicationRequest.builder()
                .app_id(app_id)
                .lang(lang)
                .build()
            )
        return SimpleNamespace(app_id=app_id, lang=lang)

    @staticmethod
    def _build_reply_message_body(*, content: str, msg_type: str, reply_in_thread: bool, uuid_value: str) -> Any:
        if "ReplyMessageRequestBody" in globals():
            return (
                ReplyMessageRequestBody.builder()
                .content(content)
                .msg_type(msg_type)
                .reply_in_thread(reply_in_thread)
                .uuid(uuid_value)
                .build()
            )
        return SimpleNamespace(
            content=content,
            msg_type=msg_type,
            reply_in_thread=reply_in_thread,
            uuid=uuid_value,
        )

    @staticmethod
    def _build_reply_message_request(message_id: str, request_body: Any) -> Any:
        if "ReplyMessageRequest" in globals():
            return (
                ReplyMessageRequest.builder()
                .message_id(message_id)
                .request_body(request_body)
                .build()
            )
        return SimpleNamespace(message_id=message_id, request_body=request_body)

    @staticmethod
    def _build_update_message_body(*, msg_type: str, content: str) -> Any:
        if "UpdateMessageRequestBody" in globals():
            return (
                UpdateMessageRequestBody.builder()
                .msg_type(msg_type)
                .content(content)
                .build()
            )
        return SimpleNamespace(msg_type=msg_type, content=content)

    @staticmethod
    def _build_update_message_request(message_id: str, request_body: Any) -> Any:
        if "UpdateMessageRequest" in globals():
            return (
                UpdateMessageRequest.builder()
                .message_id(message_id)
                .request_body(request_body)
                .build()
            )
        return SimpleNamespace(message_id=message_id, request_body=request_body)

    @staticmethod
    def _build_create_message_body(*, receive_id: str, msg_type: str, content: str, uuid_value: str) -> Any:
        if "CreateMessageRequestBody" in globals():
            return (
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type(msg_type)
                .content(content)
                .uuid(uuid_value)
                .build()
            )
        return SimpleNamespace(
            receive_id=receive_id,
            msg_type=msg_type,
            content=content,
            uuid=uuid_value,
        )

    @staticmethod
    def _build_create_message_request(receive_id_type: str, request_body: Any) -> Any:
        if "CreateMessageRequest" in globals():
            return (
                CreateMessageRequest.builder()
                .receive_id_type(receive_id_type)
                .request_body(request_body)
                .build()
            )
        return SimpleNamespace(receive_id_type=receive_id_type, request_body=request_body)

    @staticmethod
    def _build_image_upload_body(*, image_type: str, image: Any) -> Any:
        if "CreateImageRequestBody" in globals():
            return (
                CreateImageRequestBody.builder()
                .image_type(image_type)
                .image(image)
                .build()
            )
        return SimpleNamespace(image_type=image_type, image=image)

    @staticmethod
    def _build_image_upload_request(request_body: Any) -> Any:
        if "CreateImageRequest" in globals():
            return CreateImageRequest.builder().request_body(request_body).build()
        return SimpleNamespace(request_body=request_body)

    @staticmethod
    def _build_file_upload_body(*, file_type: str, file_name: str, file: Any) -> Any:
        if "CreateFileRequestBody" in globals():
            return (
                CreateFileRequestBody.builder()
                .file_type(file_type)
                .file_name(file_name)
                .file(file)
                .build()
            )
        return SimpleNamespace(file_type=file_type, file_name=file_name, file=file)

    @staticmethod
    def _build_file_upload_request(request_body: Any) -> Any:
        if "CreateFileRequest" in globals():
            return CreateFileRequest.builder().request_body(request_body).build()
        return SimpleNamespace(request_body=request_body)

    def _build_post_payload(self, content: str) -> str:
        return _build_markdown_post_payload(content)

    def _build_media_post_payload(self, *, caption: str, media_tag: Dict[str, str]) -> str:
        payload = json.loads(self._build_post_payload(caption))
        content = payload.setdefault("zh_cn", {}).setdefault("content", [])
        content.append([media_tag])
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _resolve_outbound_file_routing(
        *,
        file_path: str,
        requested_message_type: str,
    ) -> tuple[str, str]:
        ext = Path(file_path).suffix.lower()

        if ext in _FEISHU_OPUS_UPLOAD_EXTENSIONS:
            return "opus", "audio"

        if ext in _FEISHU_MEDIA_UPLOAD_EXTENSIONS:
            return "mp4", "media"

        if ext in _FEISHU_DOC_UPLOAD_TYPES:
            return _FEISHU_DOC_UPLOAD_TYPES[ext], "file"

        if requested_message_type == "file":
            return _FEISHU_FILE_UPLOAD_TYPE, "file"

        return _FEISHU_FILE_UPLOAD_TYPE, "file"
