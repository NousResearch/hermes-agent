Warning: truncated output (original token count: 56773)
Total output lines: 4533

"""
Feishu/Lark platform adapter.

Supports WebSocket + Webhook transports, DM/@mention-gated group text, inbound
media caching, FEISHU_ALLOWED_USERS allowlisting, persistent dedup, per-chat
serial processing, processing-status reactions (Typing while working, CrossMark
on failure), reaction/card-button events as synthetic events, webhook anomaly
tracking and verification-token validation (all mirroring openclaw).

Feishu identity model (https://open.feishu.cn/document/home/user-identity-introduction/introduction):
  open_id  (ou_xxx) — app-scoped; differs per Feishu app; always in event payloads.
  user_id  (u_xxx)  — tenant-scoped; needs ``contact:user.employee_id:readonly``; may be absent.
  union_id (on_xxx) — developer-scoped; stable across one developer's apps.
Bot: ``app_id`` is the credential; the bot's own open_id (from ``/bot/v3/info``) is
what Feishu puts in ``mentions[].id.open_id`` — used for mention gating only.
Session keys prefer union_id (user_id_alt) over open_id (user_id) for stability.
"""

from __future__ import annotations

from pm import install_hint
import asyncio
import collections
import concurrent.futures
import contextvars
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
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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

# lark_oapi is slow to import; the SDK names below stay None until Feishu actually connects
# (_load_lark_oapi binds every name in _LARK_SDK_IMPORTS plus ``lark`` and ``FeishuWSClient``).
_LARK_SDK_IMPORTS = (
    ("lark_oapi.api.application.v6", ("GetApplicationRequest",)),
    ("lark_oapi.api.im.v1", (
        "CreateFileRequest", "CreateFileRequestBody", "CreateImageRequest", "CreateImageRequestBody",
        "CreateMessageRequest", "CreateMessageRequestBody", "GetChatRequest", "GetMessageRequest",
        "DeleteMessageRequest", "GetMessageResourceRequest", "P2ImMessageMessageReadV1",
        "ReplyMessageRequest", "ReplyMessageRequestBody",
        "UpdateMessageRequest", "UpdateMessageRequestBody",
    )),
    ("lark_oapi.core", ("AccessTokenType", "HttpMethod")),
    ("lark_oapi.core.const", ("FEISHU_DOMAIN", "LARK_DOMAIN")),
    ("lark_oapi.core.model", ("BaseRequest",)),
    ("lark_oapi.event.callback.model.p2_card_action_trigger", ("CallBackCard", "P2CardActionTriggerResponse")),
    ("lark_oapi.event.dispatcher_handler", ("EventDispatcherHandler",)),
)
lark = FeishuWSClient = None  # type: ignore[assignment]
globals().update({name: None for _, names in _LARK_SDK_IMPORTS for name in names})
FEISHU_AVAILABLE = False
_lark_import_lock = threading.Lock()

FEISHU_WEBSOCKET_AVAILABLE = websockets is not None
FEISHU_WEBHOOK_AVAILABLE = aiohttp is not None

from gateway.config import Platform, PlatformConfig
from agent.i18n import t
from gateway.platforms.base import (
    BasePlatformAdapter, ExecApprovalPrompt, SendResult,
    SUPPORTED_DOCUMENT_TYPES, cache_document_from_bytes_async, cache_image_from_url,
    cache_audio_from_bytes_async, cache_image_from_bytes_async,
)
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.status import acquire_scoped_lock, release_scoped_lock
from hermes_constants import get_hermes_home
from utils import atomic_json_write, env_float, env_int

from gateway.platforms._shared import (
    apply_yaml_bridge as _apply_yaml_bridge, extra_or_secret as _shared_extra_or_secret,
    get_scoped_secret as _get_scoped_secret, send_error
)


logger = logging.getLogger(__name__)

# --- Regex patterns ---
_MARKDOWN_HINT_RE = re.compile(
    # Pipe table: any header line + separator line both starting with '|'.
    r"(^\|.*\|\s*\n\|[-:|\s]+\|)"
    # Headings, lists, code, bold/italic/strike/underline, links, blockquotes.
    r"|(^#{1,6}\s)"
    r"|(^\s*[-*]\s)"
    r"|(^\s*\d+\.\s)"
    r"|(^\s*---+\s*$)"
    r"|(```)"
    r"|(`[^`\n]+`)"
    r"|(\*\*[^*\n].+?\*\*)"
    r"|(~~[^~\n].+?~~)"
    r"|(<u>.+?</u>)"
    r"|(\*[^*\n]+\*)"
    r"|(\[[^\]]+\]\([^)]+\))"
    r"|(^>\s)",
    re.MULTILINE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MARKDOWN_FENCE_OPEN_RE = re.compile(r"^```([^\n`]*)\s*$")
_MARKDOWN_FENCE_CLOSE_RE = re.compile(r"^```\s*$")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_POST_CONTENT_INVALID_RE = re.compile(r"content format of the post type is incorrect", re.IGNORECASE)
# --- Media type sets and upload constants ---
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
_AUDIO_EXTENSIONS = {".ogg", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".opus", ".webm"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}
_DOCUMENT_MIME_TO_EXT = {mime: ext for ext, mime in SUPPORTED_DOCUMENT_TYPES.items()}
_FEISHU_IMAGE_UPLOAD_TYPE = "message"
_FEISHU_FILE_UPLOAD_TYPE = "stream"
_FEISHU_OPUS_UPLOAD_EXTENSIONS = {".ogg", ".opus"}
_FEISHU_MEDIA_UPLOAD_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v"}
_FEISHU_DOC_UPLOAD_TYPES = {
    ".pdf": "pdf", ".doc": "doc", ".docx": "doc", ".xls": "xls", ".xlsx": "xls", ".ppt": "ppt", ".pptx": "ppt",
}
# --- Connection, retry and batching tuning ---
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
# --- TTL, rate-limit and webhook security constants ---
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

_APPROVAL_CHOICE_MAP: Dict[str, str] = {
    "approve_once": "once", "approve_session": "session", "approve_always": "always", "deny": "deny",
}
# choice → catalog key of the resolved-card title; looked up through ``t()`` at resolve time.
_APPROVAL_LABEL_KEYS: Dict[str, str] = {
    "once": "platform.feishu.approval.resolved_once", "session": "platform.feishu.approval.resolved_session",
    "always": "platform.feishu.approval.resolved_always", "deny": "platform.feishu.approval.resolved_deny",
}


async def _read_limited_feishu_webhook_body(request: Any, max_bytes: int) -> bytes:
    """Read at most ``max_bytes`` from an aiohttp request body."""
    try:
        body = await request.content.readexactly(max_bytes + 1)
    except asyncio.IncompleteReadError as exc:
        body = exc.partial
    if len(body) > max_bytes:
        raise ValueError("payload too large")
    return body


_FEISHU_REPLY_FALLBACK_CODES = frozenset({230011, 231003})  # reply target withdrawn/missing → create fallback

# Feishu reactions render as prominent badges, unlike Discord/Telegram's
# small footer emoji — a success badge on every message would add noise, so
# we only mark start (Typing) and failure (CrossMark); the reply itself is
# the success signal.
_FEISHU_REACTION_IN_PROGRESS = "Typing"
_FEISHU_REACTION_FAILURE = "CrossMark"
# Bound on the (message_id → reaction_id) handle cache. Happy-path entries
# drain on completion; the cap is a safeguard against unbounded growth from
# delete-failures, not a capacity plan.
_FEISHU_PROCESSING_REACTION_CACHE_SIZE = 1024
_FEISHU_MESSAGE_TEXT_CACHE_SIZE = 512       # LRU cap for reply-context message text lookups

# QR onboarding constants
_ONBOARD_ACCOUNTS_URLS = {
    "feishu": "https://accounts.feishu.cn",
    "lark": "https://accounts.larksuite.com",
}
_ONBOARD_OPEN_URLS = {"feishu": "https://open.feishu.cn", "lark": "https://open.larksuite.com"}
_REGISTRATION_PATH = "/oauth/v1/app/registration"
_ONBOARD_REQUEST_TIMEOUT_S = 10

# --- Fallback display strings ---
FALLBACK_POST_TEXT = "[Rich text message]"
FALLBACK_FORWARD_TEXT = "[Merged forward message]"
FALLBACK_SHARE_CHAT_TEXT = "[Shared chat]"
FALLBACK_INTERACTIVE_TEXT = "[Interactive message]"
FALLBACK_IMAGE_TEXT = "[Image]"
FALLBACK_ATTACHMENT_TEXT = "[Attachment]"
# --- Post/card parsing helpers ---
_PREFERRED_LOCALES = ("zh_cn", "en_us")
_MARKDOWN_SPECIAL_CHARS_RE = re.compile(r"([\\`*_{}\[\]()#+\-!|>~])")
_MENTION_PLACEHOLDER_RE = re.compile(r"@_user_\d+")
_MENTION_BOUNDARY_CHARS = frozenset(" \t\n\r.,;:!?、，。；：！？()[]{}<>\"'`")
_TRAILING_TERMINAL_PUNCT = frozenset(" \t\n\r.!?。！？")
_WHITESPACE_RE = re.compile(r"\s+")
_SUPPORTED_CARD_TEXT_KEYS = (
    "title", "text", "content", "label", "value", "name", "summary", "subtitle", "description", "placeholder", "hint",
)
_RICH_BLOCK_TAGS = {
    "plain_text", "lark_md", "markdown", "note", "div", "column_set", "column", "action", "button", "select_static",
    "date_picker",
}
_SKIP_TEXT_KEYS = {
    "tag", "type", "msg_type", "message_type", "chat_id", "open_chat_id", "share_chat_id", "file_key", "image_key",
    "user_id", "open_id", "union_id", "url", "href", "link", "token", "template", "locale",
}


@dataclass(frozen=True)
class FeishuPostMediaRef:
    file_key: str
    file_name: str = ""
    resource_type: str = "file"


@dataclass(frozen=True)
class FeishuMentionRef:
    name: str = ""
    open_id: str = ""
    is_all: bool = False
    is_self: bool = False


@dataclass(frozen=True)
class _FeishuBotIdentity:
    open_id: str = ""
    user_id: str = ""
    name: str = ""

    def matches(self, *, open_id: str, user_id: str, name: str) -> bool:
        # Precedence: open_id > user_id > name. IDs are authoritative when both
        # sides have them; the next tier is only considered when either side
        # lacks the current one.
        if open_id and self.open_id:
            return open_id == self.open_id
        if user_id and self.user_id:
            return user_id == self.user_id
        return bool(self.name) and name == self.name


@dataclass(frozen=True)
class FeishuPostParseResult:
    text_content: str
    image_keys: List[str] = field(default_factory=list)
    media_refs: List[FeishuPostMediaRef] = field(default_factory=list)


@dataclass(frozen=True)
class FeishuNormalizedMessage:
    raw_type: str
    text_content: str
    preferred_message_type: str = "text"
    image_keys: List[str] = field(default_factory=list)
    media_refs: List[FeishuPostMediaRef] = field(default_factory=list)
    mentions: List[FeishuMentionRef] = field(default_factory=list)
    relation_kind: str = "plain"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeishuAdapterSettings:
    """Every field is mirrored onto the adapter as ``self._<field>`` by ``_apply_settings``."""

    app_id: str  # credential identifier, never taken from event payloads
    app_secret: str
    domain_name: str
    connection_mode: str
    encrypt_key: str
    verification_token: str
    group_policy: str
    allowed_group_users: frozenset[str]
    bot_open_id: str  # app-scoped id from /bot/v3/info — what mentions[].id.open_id carries
    bot_user_id: str  # tenant-scoped fallback for mention matching
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
    allow_bots: str = "none"  # "none" | "mentions" | "all"
    require_mention: bool = True
    allow_all_dm: bool = False  # resolved per-profile so multiplexed adapters honor their own .env


@dataclass
class FeishuGroupRule:
    """Per-group policy rule for controlling which users may interact with the bot."""

    policy: str  # "open" | "allowlist" | "blacklist" | "admin_only" | "disabled"
    allowlist: set[str] = field(default_factory=set)
    blacklist: set[str] = field(default_factory=set)
    require_mention: Optional[bool] = None  # None = inherit global


@dataclass
class FeishuBatchState:
    events: Dict[str, MessageEvent] = field(default_factory=dict)
    tasks: Dict[str, asyncio.Task] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)


# --- Admission: policy types ---

RejectReason = Literal["self_echo", "self_ids_unknown", "bots_disabled", "bot_not_mentioned", "group_policy_rejected"]


def _is_bot_sender(sender: Any) -> bool:
    # receive_v1 docs say {user, bot}; accept "app" defensively.
    return getattr(sender, "sender_type", "") in {"bot", "app"}


def _sender_identity(sender: Any) -> frozenset:
    # Take any non-empty id variant — tenant sender_id_type decides which are populated.
    sid = getattr(sender, "sender_id", None)
    if sid is None:
        return frozenset()
    return frozenset(v for v in (getattr(sid, k, None) for k in ("open_id", "user_id", "union_id")) if v)


# --- Markdown rendering helpers ---

def _escape_markdown_text(text: str) -> str:
    return _MARKDOWN_SPECIAL_CHARS_RE.sub(r"\\\1", text)


def _to_boolean(value: Any) -> bool:
    return value is True or value == 1 or value == "true"


def _rule_id_set(values: Any) -> set[str]:
    return {str(u).strip() for u in values if str(u).strip()}


def _parse_feishu_group_rule(rule_cfg: dict) -> FeishuGroupRule:
    return FeishuGroupRule(
        policy=str(rule_cfg.get("policy", "open")).strip().lower(),
        allowlist=_rule_id_set(rule_cfg.get("allowlist", [])),
        blacklist=_rule_id_set(rule_cfg.get("blacklist", [])),
        # Only override when explicitly set — missing vs false must not collapse.
        require_mention=_to_boolean(rule_cfg["require_mention"]) if "require_mention" in rule_cfg else None,
    )


def _merge_hot_group_rule(base: Optional[FeishuGroupRule], rule_cfg: dict) -> FeishuGroupRule:
    """Field-level overlay: hot keys win; omitted keys keep the boot rule (or parse defaults)."""
    if base is None:
        return _parse_feishu_group_rule(rule_cfg)
    return FeishuGroupRule(
        policy=str(rule_cfg["policy"]).strip().lower() if "policy" in rule_cfg else base.policy,
        allowlist=_rule_id_set(rule_cfg["allowlist"]) if "allowlist" in rule_cfg else set(base.allowlist),
        blacklist=_rule_id_set(rule_cfg["blacklist"]) if "blacklist" in rule_cfg else set(base.blacklist),
        require_mention=_to_boolean(rule_cfg["require_mention"]) if "require_mention" in rule_cfg else base.require_mention,
    )


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


_TEXT_STYLE_WRAPPERS = (("bold", "**", "**"), ("italic", "*", "*"), ("underline", "<u>", "</u>"), ("strikethrough", "~~", "~~"))


def _render_text_element(element: Dict[str, Any]) -> str:
    text = str(element.get("text", "") or "")
    style = element.get("style")
    style_dict = style if isinstance(style, dict) else None
    if _is_style_enabled(style_dict, "code"):
        return _wrap_inline_code(text)
    # Post text elements carry raw text plus separate style flags; the style wrappers below
    # re-create the markdown. Escaping the text here put `\*\*bold\*\*` / `\`code\`` into the
    # model's context and those backslashes came straight back out in replies (#9816).
    rendered = text
    if not rendered:
        return ""
    for key, prefix, suffix in _TEXT_STYLE_WRAPPERS:  # order matters for nesting
        if _is_style_enabled(style_dict, key):
            rendered = f"{prefix}{rendered}{suffix}"
    return rendered


def _render_code_block_element(element: Dict[str, Any]) -> str:
    language = _sanitize_fence_language(str(element.get("language", "") or "") or str(element.get("lang", "") or ""))
    code = (str(element.get("text", "") or "") or str(element.get("content", "") or "")).replace("\r\n", "\n")
    trailing_newline = "" if code.endswith("\n") else "\n"
    return f"```{language}\n{code}{trailing_newline}```"


def _strip_markdown_to_plain_text(text: str) -> str:
    """Plain-text fallback: shared strip_markdown plus Feishu extras (blockquote, ~~, <u>, hr, CRLF)."""
    from gateway.platforms.helpers import strip_markdown
    plain = text.replace("\r\n", "\n")
    plain = _MARKDOWN_LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2).strip()})", plain)
    plain = re.sub(r"^>\s?", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s*---+\s*$", "---", plain, flags=re.MULTILINE)
    plain = re.sub(r"~~([^~\n]+)~~", r"\1", plain)
    plain = re.sub(r"<u>([\s\S]*?)</u>", r"\1", plain)
    return strip_markdown(plain)


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


# --- Post payload builders and parsers ---

def _build_markdown_post_payload(content: str) -> str:
    rows = _build_markdown_post_rows(content)
    return json.dumps({"zh_cn": {"content": rows}}, ensure_ascii=False)


def _build_markdown_post_rows(content: str) -> List[List[Dict[str, str]]]:
    """Build Feishu post rows, giving each fenced code block its own row.

    Feishu's `md` renderer can swallow trailing content when a fence sits inside one
    large element; splitting at real fence lines keeps surrounding prose visible.
    """
    if not content:
        return [[{"tag": "md", "text": ""}]]
    if "```" not in content:
        return [[{"tag": "md", "text": content}]]

    rows: List[List[Dict[str, str]]] = []
    current: List[str] = []
    in_code_block = False

    def _flush_current() -> None:
        nonlocal current
        segment = "\n".join(current)
        if segment.strip():
            rows.append([{"tag": "md", "text": segment}])
        current = []

    for raw_line in content.splitlines():
        fence_re = _MARKDOWN_FENCE_CLOSE_RE if in_code_block else _MARKDOWN_FENCE_OPEN_RE
        is_fence = bool(fence_re.match(raw_line.strip()))
        if is_fence and not in_code_block:  # opening fence: prose before it becomes its own row
            _flush_current()
        current.append(raw_line)
        if is_fence:
            in_code_block = not in_code_block
            if not in_code_block:  # closing fence: the code block becomes its own row
                _flush_current()
    _flush_current()
    return rows or [[{"tag": "md", "text": content}]]


def parse_feishu_post_payload(
    payload: Any, *, mentions_map: Optional[Dict[str, FeishuMentionRef]] = None,
) -> FeishuPostParseResult:
    resolved = _resolve_post_payload(payload)
    if not resolved:
        return FeishuPostParseResult(text_content=FALLBACK_POST_TEXT)
    image_keys: List[str] = []
    media_refs: List[FeishuPostMediaRef] = []
    parts: List[str] = []
    title = _normalize_feishu_text(str(resolved.get("title", "")).strip())
    if title:
        parts.append(title)
    for row in resolved.get("content", []) or []:
        if not isinstance(row, list):
            continue
        row_text = _normalize_feishu_text(
            "".join(_render_post_element(item, image_keys, media_refs, mentions_map) for item in row)
        )
        if row_text:
            parts.append(row_text)
    for entry in resolved.get("files", []) or []:
        if (
            not isinstance(entry, dict)
            or _to_boolean(entry.get("is_folder"))
            or not str(entry.get("file_key", "")).strip()
        ):
            continue
        placeholder = _render_post_element(
            {**entry, "tag": "file"}, image_keys, media_refs, mentions_map,
        )
        if placeholder:
            parts.append(placeholder)
    return FeishuPostParseResult(
        text_content="\n".join(parts).strip() or FALLBACK_POST_TEXT, image_keys=image_keys, media_refs=media_refs,
    )


def _resolve_post_payload(payload: Any) -> Dict[str, Any]:
    direct = _to_post_payload(payload)
    if direct:
        return direct
    if not isinstance(payload, dict):
        return {}
    return _resolve_locale_payload(payload.get("post")) or _resolve_locale_payload(payload)


def _resolve_locale_payload(payload: Any) -> Dict[str, Any]:
    direct = _to_post_payload(payload)
    if direct:
        return direct
    if not isinstance(payload, dict):
        return {}
    # Preferred locales first, then any locale that carries a content list.
    preferred = (payload.get(key) for key in _PREFERRED_LOCALES)
    for candidate in map(_to_post_payload, itertools.chain(preferred, payload.values())):
        if candidate:
            return candidate
    return {}


def _to_post_payload(candidate: Any) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    content = candidate.get("content")
    if not isinstance(content, list):
        return {}
    files = candidate.get("files")
    return {
        "title": str(candidate.get("title", "") or ""),
        "content": content,
        "files": files if isinstance(files, list) else [],
    }


_STATIC_POST_TAGS = {"br": "\n", "hr": "\n\n---\n\n", "divider": "\n\n---\n\n"}


def _render_post_element(
    element: Any, image_keys: List[str], media_refs: List[FeishuPostMediaRef],
    mentions_map: Optional[Dict[str, FeishuMentionRef]] = None,
) -> str:
    if isinstance(element, str):
        return element
    if not isinstance(element, dict):
        return ""

    tag = str(element.get("tag", "")).strip().lower()
    if tag in _STATIC_POST_TAGS:
        return _STATIC_POST_TAGS[tag]
    if tag == "text":
        return _render_text_element(element)
    if tag in {"code_block", "pre"}:
        return _render_code_block_element(element)
    if tag == "a":
        href = str(element.get("href", "")).strip()
        label = str(element.get("text", href) or "").strip()
        if not label:
            return ""
        escaped_label = _escape_markdown_text(label)
        return f"[{escaped_label}]({href})" if href else escaped_label
    if tag == "at":
        # <at>.user_id is a placeholder ("@_user_N" / "@_all"); mentions_map has the real ref.
        placeholder = str(element.get("user_id", "")).strip()
        if placeholder == "@_all":
            # The SDK sometimes omits @_all from top-level mentions; record it so callers see it.
            if mentions_map is not None and "@_all" not in mentions_map:
                mentions_map["@_all"] = FeishuMentionRef(is_all=True)
            return "@all"
        ref = (mentions_map or {}).get(placeholder)
        display_name = (ref.name or ref.open_id or "user") if ref is not None else (
            str(element.get("user_name", "")).strip() or "user"
        )
        return f"@{_escape_markdown_text(display_name)}"
    if tag in {"img", "image"}:
        image_key = str(element.get("image_key", "")).strip()
        if image_key and image_key not in image_keys:
            image_keys.append(image_key)
        alt = str(element.get("text", "")).strip() or str(element.get("alt", "")).strip()
        return f"[Image: {alt}]" if alt else "[Image]"
    if tag in {"media", "file", "audio", "video"}:
        file_key = str(element.get("file_key", "")).strip()
        names = (str(element.get(k, "")).strip() for k in ("file_name", "title", "text"))
        file_name = next((n for n in names if n), "")
        placeholder = f"[Attachment: {file_name}]" if file_name else "[Attachment]"
        if not file_key:
            return placeholder
        if not any(ref.file_key == file_key for ref in media_refs):
            media_refs.append(FeishuPostMediaRef(
                file_key=file_key, file_name=file_name, resource_type=tag if tag in {"audio", "video"} else "file",
            ))
            return placeholder
        return ""
    if tag in {"emotion", "emoji"}:
        label = str(element.get("text", "")).strip() or str(element.get("emoji_type", "")).strip()
        return f":{_escape_markdown_text(label)}:" if label else "[Emoji]"
    if tag == "code":
        code = str(element.get("text", "") or "") or str(element.get("content", "") or "")
        return _wrap_inline_code(code) if code else ""
    nested = (element.get(key) for key in ("text", "title", "content", "children", "elements"))
    return _join_nested_posts(nested, image_keys, media_refs, mentions_map)


def _join_nested_posts(values: Any, image_keys: Any, media_refs: Any, mentions_map: Any) -> str:
    parts = (_render_nested_post(item, image_keys, media_refs, mentions_map) for item in values)
    return " ".join(part for part in parts if part)


def _render_nested_post(
    value: Any, image_keys: List[str], media_refs: List[FeishuPostMediaRef],
    mentions_map: Optional[Dict[str, FeishuMentionRef]] = None,
) -> str:
    if isinstance(value, str):
        return _escape_markdown_text(value)
    if isinstance(value, list):
        return _join_nested_posts(value, image_keys, media_refs, mentions_map)
    if isinstance(value, dict):
        direct = _render_post_element(value, image_keys, media_refs, mentions_map)
        return direct or _join_nested_posts(value.values(), image_keys, media_refs, mentions_map)
    return ""


# --- Message normalization ---

def normalize_feishu_message(
    *, message_type: str, raw_content: str, mentions: Optional[Sequence[Any]] = None,
    bot: _FeishuBotIdentity = _FeishuBotIdentity(),
) -> FeishuNormalizedMessage:
    normalized_type = str(message_type or "").strip().lower()
    payload = _load_feishu_payload(raw_content)
    mentions_map = _build_mentions_map(mentions, bot)

    if normalized_type == "text":
        text = str(payload.get("text", "") or "")
        # Feishu SDK sometimes omits @_all from the mentions payload even when
        # the text literal contains it (confirmed via im.v1.message.get).
        if "@_all" in text and "@_all" not in mentions_map:
            mentions_map["@_all"] = FeishuMentionRef(is_all=True)
        return FeishuNormalizedMessage(
            raw_type=normalized_type, text_content=_normalize_feishu_text(text, mentions_map),
            mentions=list(mentions_map.values()),
        )
    if normalized_type == "post":
        # The walker writes back to mentions_map if it encounters
        # <at user_id="@_all">, so reading .values() after parsing is enough.
        parsed_post = parse_feishu_post_payload(payload, mentions_map=mentions_map)
        return FeishuNormalizedMessage(
            raw_type=normalized_type, text_content=parsed_post.text_content,
            image_keys=list(parsed_post.image_keys), media_refs=list(parsed_post.media_refs),
            mentions=list(mentions_map.values()), relation_kind="post",
        )
    mention_refs = list(mentions_map.values())
    if normalized_type == "image":
        image_key = str(payload.get("image_key", "") or "").strip()
        alt_text = _normalize_feishu_text(
            str(payload.get("text", "") or "")
            or str(payload.get("alt", "") or "")
            or FALLBACK_IMAGE_TEXT,
            mentions_map,
        )
        return FeishuNormalizedMessage(
            raw_type=normalized_type,
            text_content=alt_text if alt_text != FALLBACK_IMAGE_TEXT else "",
            preferred_message_type="photo", image_keys=[image_key] if image_key else [],
            relation_kind="image", mentions=mention_refs,
        )
    if normalized_type in {"file", "audio", "media"}:
        media_ref = _build_media_ref_from_payload(payload, resource_type=normalized_type)
        return FeishuNormalizedMessage(
            raw_type=normalized_type, text_content="",
            preferred_message_type="audio" if normalized_type == "audio" else "document",
            media_refs=[media_ref] if media_ref.file_key else [], relation_kind=normalized_type,
            metadata={"placeholder_text": _attachment_placeholder(media_ref.file_name)},
            mentions=mention_refs,
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
    title = _first_text_field(payload, "title", "summary", "preview", deep=("title", "summary", "preview", "description"))
    entries = _collect_forward_entries(payload)
    lines = ([title] if title else []) + entries[:8]
    return FeishuNormalizedMessage(
        raw_type="merge_forward", text_content="\n".join(lines).strip() or FALLBACK_FORWARD_TEXT,
        relation_kind="merge_forward", metadata={"entry_count": len(entries), "title": title},
    )


def _normalize_share_chat_message(payload: Dict[str, Any]) -> FeishuNormalizedMessage:
    chat_name = _first_text_field(payload, "chat_name", "name", "title", deep=("chat_name", "name", "title"))
    share_id = _first_text_field(payload, "chat_id", "open_chat_id", "share_chat_id")
    lines = [f"Shared chat: {chat_name}" if chat_name else FALLBACK_SHARE_CHAT_TEXT]
    if share_id:
        lines.append(f"Chat ID: {share_id}")
    return FeishuNormalizedMessage(
        raw_type="share_chat", text_content="\n".join(lines), relation_kind="share_chat",
        metadata={"chat_id": share_id, "chat_name": chat_name},
    )


def _normalize_interactive_message(message_type: str, payload: Dict[str, Any]) -> FeishuNormalizedMessage:
    card_payload = payload.get("card") if isinstance(payload.get("card"), dict) else payload
    title = _first_non_empty_text(
        _find_header_title(card_payload), payload.get("title"),
        _find_first_text(card_payload, keys=("title", "summary", "subtitle")),
    )
    actions = _collect_action_labels(card_payload)
    lines = ([title] if title else []) + [line for line in _collect_card_lines(card_payload) if line != title]
    if actions:
        lines.append(f"Actions: {', '.join(actions)}")
    return FeishuNormalizedMessage(
        raw_type=message_type,
        text_content="\n".join(lines[:12]).strip() or FALLBACK_INTERACTIVE_TEXT,
        relation_kind="interactive", metadata={"title": title, "actions": actions},
    )


# --- Content extraction utilities (card / forward / text walking) ---

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
        sender = _first_text_field(item, "sender_name", "user_name", "sender", "name")
        nested_type = str(item.get("message_type", "") or item.get("msg_type", "")).strip().lower()
        if nested_type == "post":
            body = parse_feishu_post_payload(item.get("content") or item).text_content
        else:
            body = _first_text_field(
                item, "text", "summary", "preview", "content", deep=("text", "content", "summary", "preview", "title"),
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
        label = _first_text_field(item, "text", "name", "value", deep=("text", "content", "name", "value"))
        if label:
            labels.append(label)
    return _unique_lines(labels)


def _collect_text_segments(value: Any, *, in_rich_block: bool) -> List[str]:
    if isinstance(value, str):
        return [_normalize_feishu_text(value)] if in_rich_block else []
    if isinstance(value, list):
        return [seg for item in value for seg in _collect_text_segments(item, in_rich_block=in_rich_block)]
    if not isinstance(value, dict):
        return []
    tag = str(value.get("tag", "") or value.get("type", "")).strip().lower()
    next_in_rich_block = in_rich_block or tag in _RICH_BLOCK_TAGS
    segments: List[str] = []
    if next_in_rich_block:
        for key in _SUPPORTED_CARD_TEXT_KEYS:
            item = value.get(key)
            if isinstance(item, str) and _normalize_feishu_text(item):
                segments.append(_normalize_feishu_text(item))
    for key, item in value.items():
        if key not in _SKIP_TEXT_KEYS:
            segments.extend(_collect_text_segments(item, in_rich_block=next_in_rich_block))
    return segments


def _build_media_ref_from_payload(payload: Dict[str, Any], *, resource_type: str) -> FeishuPostMediaRef:
    file_key = str(payload.get("file_key", "") or "").strip()
    file_name = _first_text_field(payload, "file_name", "title", "text")
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
    """First scalar (non-dict/list, non-None) value that normalizes to non-empty text."""
    for value in values:
        if value is None or isinstance(value, (dict, list)):
            continue
        normalized = _normalize_feishu_text(value if isinstance(value, str) else str(value))
        if normalized:
            return normalized
    return ""


def _first_text_field(payload: Dict[str, Any], *keys: str, deep: tuple[str, ...] = ()) -> str:
    """``_first_non_empty_text`` over ``payload[key]`` for each key, then a deep ``_find_first_text``."""
    values = [payload.get(key) for key in keys]
    if deep:
        values.append(_find_first_text(payload, keys=deep))
    return _first_non_empty_text(*values)


# --- General text utilities ---

def _normalize_feishu_text(text: str, mentions_map: Optional[Dict[str, FeishuMentionRef]] = None) -> str:
    def _sub(match: "re.Match[str]") -> str:
        ref = (mentions_map or {}).get(match.group(0))
        return " " if ref is None else f"@{ref.name or ref.open_id or 'user'}"

    cleaned = _MENTION_PLACEHOLDER_RE.sub(_sub, text or "")
    cleaned = cleaned.replace("@_all", "@all")
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


# --- Mention helpers ---

def _extract_mention_ids(mention: Any) -> tuple[str, str]:
    """(open_id, user_id): message.get gives a string id + id_type; events give a nested UserId object."""
    mention_id = getattr(mention, "id", None)
    if isinstance(mention_id, str):
        id_type = str(getattr(mention, "id_type", "") or "").lower()
        return (mention_id, "") if id_type == "open_id" else ("", mention_id) if id_type == "user_id" else ("", "")
    if mention_id is None:
        return "", ""
    return str(getattr(mention_id, "open_id", "") or ""), str(getattr(mention_id, "user_id", "") or "")


def _build_mentions_map(mentions: Optional[Sequence[Any]], bot: _FeishuBotIdentity) -> Dict[str, FeishuMentionRef]:
    result: Dict[str, FeishuMentionRef] = {}
    for mention in mentions or []:
        key = str(getattr(mention, "key", "") or "")
        if not key:
            continue
        if key == "@_all":
            result[key] = FeishuMentionRef(is_all=True)
            continue
        open_id, user_id = _extract_mention_ids(mention)
        name = str(getattr(mention, "name", "") or "").strip()
        is_self = bot.matches(open_id=open_id, user_id=user_id, name=name)
        result[key] = FeishuMentionRef(name=name, open_id=open_id, is_self=is_self)
    return result


def _build_mention_hint(mentions: Sequence[FeishuMentionRef]) -> str:
    parts: List[str] = []
    seen: set = set()
    for ref in mentions:
        if ref.is_self:
            continue
        signature = (ref.is_all, ref.open_id, ref.name)
        if signature in seen:
            continue
        seen.add(signature)
        if ref.is_all:
            parts.append("@all")
        elif ref.open_id:
            parts.append(f"{ref.name or 'unknown'} (open_id={ref.open_id})")
        else:
            parts.append(ref.name or "unknown")
    return f"[Mentioned: {', '.join(parts)}]" if parts else ""


def _strip_edge_self_mentions(text: str, mentions: Sequence[FeishuMentionRef]) -> str:
    # Leading self-mentions are stripped unconditionally (word-boundary so @Al can't eat @Alice);
    # trailing ones only when followed by whitespace/terminal punct so "don't @Bot again" survives.
    if not text:
        return text
    self_names = [f"@{ref.name or ref.open_id or 'user'}" for ref in mentions if ref.is_self]
    if not self_names:
        return text
    remaining = text.lstrip()
    while True:
        for nm in self_names:
            if not remaining.startswith(nm):
                continue
            after = remaining[len(nm):]
            if after and after[0] not in _MENTION_BOUNDARY_CHARS:
                continue
            remaining = after.lstrip()
            break
        else:
            break
    while True:
        i = len(remaining)
        while i > 0 and remaining[i - 1] in _TRAILING_TERMINAL_PUNCT:
            i -= 1
        body = remaining[:i]
        tail = remaining[i:]
        for nm in self_names:
            if body.endswith(nm):
                remaining = body[: -len(nm)].rstrip() + tail
                break
        else:
            return remaining


# --- Multiplex isolation for the lark_oapi WebSocket client ---
#
# ``lark_oapi.ws.client`` keeps the asyncio loop in a *module-level global* (``loop``), and
# Hermes monkey-patches ``websockets.connect`` on the shared module to inject ping settings.
# In multiplex mode N profiles each run a WS client on their own thread, so they overwrite
# each other's globals (last-write-wins): tasks land on a sibling's loop ("Future attached
# to a different loop") or a client binds the wrong loop and goes deaf. Fix: install
# process-wide, thread-dispatching shims exactly once —
#   * ``ws_client_module.loop`` becomes a proxy forwarding to the loop registered by the
#     *current thread* (all SDK reads happen on the loop-owning thread); unregistered
#     threads fall back to the SDK's original loop.
#   * ``websockets.connect`` becomes one dispatcher that merges the calling thread's
#     registered ping overrides, so profiles stop racing over the global patch.

# --------------------------------------------------------------------------- Multiplex isolation for the
# lark_oapi WebSocket client (#73779)
# --------------------------------------------------------------------------- ``lark_oapi.ws.client`` keeps
# the asyncio loop used by ``Client.start()`` and every coroutine it spawns in a *module-level global*
# (``loop``), and Hermes also monkey-patches ``websockets.connect`` on the shared ``websockets`` module to
# inject per-adapter ping settings. In multiplex mode every profile runs its own WS client on a dedicated
# thread, so the N threads overwrite each other's module globals (last-write-wins): a client ends up
# scheduling tasks on a sibling profile's loop ("Future attached to a different loop" crashes) or binds to
# the wrong loop at construction time and goes deaf from the start. The fix installs process-wide,
# thread-dispatching shims exactly once: * ``ws_client_module.loop`` becomes a proxy that forwards every
# attribute access to the loop registered by the *current thread*. All SDK reads of the global happen on the
# thread that owns the loop (``start()`` blocks in ``run_until_complete`` and every ``create_task`` callback
# runs on the loop's own thread), so each profile transparently sees its own loop. Threads that never
# registered one (single-profile installs, CLI) fall back to the SDK's original module loop. *
# ``websockets.connect`` becomes a single dispatcher that merges the per-thread ping overrides registered by
# the calling profile, so profiles no longer race over the global patch or restore each other's hooks while
# a sibling is still connected.
_WS_ISOLATION_LOCK = threading.Lock()
_WS_ISOLATION_INSTALLED = False
_ws_isolation_state = threading.local()  # per WS thread: .loop and .connect_kwargs


class _ThreadLocalLoopProxy:
    """Forwards attribute access to the current thread's registered loop."""

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback

    def _target(self) -> Any:
        return getattr(_ws_isolation_state, "loop", None) or self._fallback

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ThreadLocalLoopProxy target={self._target()!r}>"


def _install_lark_ws_isolation(ws_client_module: Any) -> None:
    """Install the thread-dispatching shims once per process (idempotent)."""
    global _WS_ISOLATION_INSTALLED
    with _WS_ISOLATION_LOCK:
        if _WS_ISOLATION_INSTALLED:
            return
        ws_client_module.loop = _ThreadLocalLoopProxy(ws_client_module.loop)
        real_connect = ws_client_module.websockets.connect

        def _dispatch_connect(*args: Any, **kwargs: Any) -> Any:
            overrides = getattr(_ws_isolation_state, "connect_kwargs", None) or {}
            for key, value in overrides.items():
                kwargs.setdefault(key, value)
            return real_connect(*args, **kwargs)

        # Keep inspect.signature(websockets.connect) honest — the SDK probes it for ``proxy`` support.
        _dispatch_connect.__wrapped__ = real_connect
        _dispatch_connect.__name__ = getattr(real_connect, "__name__", "connect")
        ws_client_module.websockets.connect = _dispatch_connect

        original_receive_loop = ws_client_module.Client._receive_message_loop

        async def _receive_message_loop_exit_notify(self: Any) -> None:
            # The SDK schedules this coroutine right after the websocket handshake succeeded, so its
            # entry is the only in-thread proof that a (re)built link is actually up.
            on_link_up = getattr(_ws_isolation_state, "on_link_up", None)
            if on_link_up is not None:
                on_link_up()
            try:
                await original_receive_loop(self)
            except Exception:
                # ``Client.start()`` parks in ``run_until_complete(_select())``, which only returns
                # when this worker loop stops, and the receive loop runs as a bare ``create_task``
                # whose exception nobody retrieves — so every unrecoverable exit (reconnect ladder
                # disabled, or its ``ClientException``/``ServerUnreachableException`` re-raise)
                # left a deaf-but-ESTABLISHED socket whose executor future never completed and the
                # supervisor never rebuilt (#113662). Log the root cause here and stop the loop so
                # ``start()`` raises, the future completes and ``_supervise_websocket_thread`` fires.
                # A *normal* return means the SDK's own ladder already reconnected (it scheduled a
                # fresh receive loop) and must NOT stop the loop. Deliberate disconnects nil
                # ``_ws_client`` first, so the supervisor exits without restarting.
                adapter = getattr(_ws_isolation_state, "adapter", None)
                if adapter is None or getattr(adapter, "_running", True):
                    logger.exception(
                        "[Feishu] lark WS receive loop died; stopping the worker "
                        "loop so the supervisor can rebuild"
                    )
                else:
                    # ``disconnect()`` sent the CLOSE frame itself: the loop ending here is expected.
                    logger.debug("[Feishu] lark WS receive loop ended during disconnect", exc_info=True)
                asyncio.get_running_loop().stop()

        ws_client_module.Client._receive_message_loop = _receive_message_loop_exit_notify
        _WS_ISOLATION_INSTALLED = True


def _run_official_feishu_ws_client(ws_client: Any, adapter: Any) -> None:
    """Run the official Lark WS client in its own thread-local loop (see isolation notes above)."""
    import lark_oapi.ws.client as ws_client_module

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    adapter._ws_thread_loop = loop
    original_configure = getattr(ws_client, "_configure", None)

    def _apply_runtime_ws_overrides() -> None:
        try:
            setattr(ws_client, "_reconnect_nonce", adapter._ws_reconnect_nonce)
            setattr(ws_client, "_reconnect_interval", adapter._ws_reconnect_interval)
            if adapter._ws_ping_interval is not None:
                setattr(ws_client, "_ping_interval", adapter._ws_ping_interval)
            # SDK observer (lark-oapi ``Client.on_reconnecting``, fired first thing in ``_reconnect()``):
            # on the live link ``_auto_reconnect`` is on, so the ladder runs *inside* the receive loop
            # and the thread never dies — without this the supervisor's ``retrying`` is never published.
            setattr(ws_client, "on_reconnecting", _on_reconnecting)
        except Exception:
            logger.debug("[Feishu] Failed to apply websocket runtime overrides", exc_info=True)

    connect_overrides = {
        key: value
        for key, value in (("ping_interval", adapter._ws_ping_interval), ("ping_timeout", adapter._ws_ping_timeout))
        if value is not None
    }
    adapter_loop = adapter._loop

    def _on_reconnecting() -> None:
        if adapter_loop is not None and not adapter_loop.is_closed():
            adapter_loop.call_soon_threadsafe(adapter._ws_link_retrying, ws_client)

    def _on_link_up() -> None:
        # Fired on the WS thread when the SDK scheduled a receive loop (handshake done); hop to the
        # adapter loop so the ``connected`` re-stamp after a supervisor rebuild runs where the adapter's
        # state lives.
        if adapter_loop is not None and not adapter_loop.is_closed():
            adapter_loop.call_soon_threadsafe(adapter._ws_link_up, ws_client)

    _install_lark_ws_isolation(ws_client_module)
    _ws_isolation_state.loop = loop
    _ws_isolation_state.connect_kwargs = connect_overrides
    _ws_isolation_state.on_link_up = _on_link_up
    _ws_isolation_state.adapter = adapter

    def _configure_with_overrides(conf: Any) -> Any:
        if original_configure is None:
            raise RuntimeError("Feishu _configure_with_overrides called but original_configure is None")
        result = original_configure(conf)
        _apply_runtime_ws_overrides()
        return result

    if original_configure is not None:
        setattr(ws_client, "_configure", _configure_with_overrides)
    _apply_runtime_ws_overrides()
    try:
        ws_client.start()
    except Exception:
        pass
    finally:
        _ws_isolation_state.loop = None
        _ws_isolation_state.connect_kwargs = None
        _ws_isolation_state.on_link_up = None
        _ws_isolation_state.adapter = None
        if original_configure is not None:
            setattr(ws_client, "_configure", original_configure)
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        for closer in (loop.stop, loop.close):
            try:
                closer()
            except Exception:
                pass
        adapter._ws_thread_loop = None


def _load_lark_oapi() -> bool:
    """Import and bind the Feishu SDK after an explicit connection request."""
    if FEISHU_AVAILABLE:
        return True
    with _lark_import_lock:
        if FEISHU_AVAILABLE:
            return True
        import importlib
        try:
            bound: Dict[str, Any] = {"lark": importlib.import_module("lark_oapi")}
            for module_name, names in _LARK_SDK_IMPORTS:
                module = importlib.import_module(module_name)
                bound.update({name: getattr(module, name) for name in names})
            bound["FeishuWSClient"] = importlib.import_module("lark_oapi.ws").Client
        except (ImportError, AttributeError):
            return False
        bound["FEISHU_AVAILABLE"] = True
        globals().update(bound)
        return True


def feishu_deps_present() -> bool:
    """PASSIVE registry ``check_fn``: is lark-oapi installed? Must never install or import the SDK.

    Uses cheap importlib.metadata lookups; the real import is deferred to ``_load_lark_oapi``
    and the ACTIVE installer is ``check_feishu_requirements`` (``ensure_deps_fn``).

    Registry ``check_fn`` — called from status displays and config loading, so it must never install
    anything. The ACTIVE lazy-installer (``check_feishu_requirements``) is registered as ``ensure_deps_fn``
    and runs from ``create_adapter()`` when this returns False (#79812).
    """
    if FEISHU_AVAILABLE:
        return True
    try:
        from pm.extras import available
        return available("feishu")
    except Exception:  # pragma: no cover — defensive
        return False


def check_feishu_requirements() -> bool:
    """Ensure Feishu dependencies are installed without importing the SDK."""
    if FEISHU_AVAILABLE:
        return True

    from pm import ensure_import

    try:
        ensure_import("feishu")
        return True
    except Exception:
        return False


def _tenant_get_request(uri: str, *, queries: Optional[List[tuple[str, str]]] = None) -> Any:
    """Raw ``BaseRequest`` GET with the tenant access token (bot/v3 endpoints have no typed SDK request)."""
    builder = BaseRequest.builder().http_method(HttpMethod.GET).uri(uri)
    if queries is not None:
        builder = builder.queries(queries)
    return builder.token_types({AccessTokenType.TENANT}).build()


def _sdk_domain(domain_name: str) -> Any:
    return LARK_DOMAIN if domain_name == "lark" else FEISHU_DOMAIN


def _build_lark_client(app_id: str, app_secret: str, sdk_domain: Any) -> Any:
    return lark.Client.builder().app_id(app_id).app_secret(app_secret).domain(sdk_domain).log_level(lark.LogLevel.WARNING).build()


def _card_button(label: str, btn_type: str, value: Dict[str, Any]) -> Dict[str, Any]:
    return {"tag": "button", "text": {"tag": "plain_text", "content": label}, "type": btn_type, "value": value}


def _card(title: str, template: str, markdown: str, *, actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Wide interactive card: colored header + one markdown block (+ an optional button row)."""
    elements: List[Dict[str, Any]] = [{"tag": "markdown", "content": markdown}]
    if actions is not None:
        elements.append({"tag": "action", "actions": actions})
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"content": title, "tag": "plain_text"}, "template": template},
        "elements": elements,
    }


def _sdk_build(request_cls: Any, **fields: Any) -> Any:
    """``request_cls.builder().<field>(value)...build()``; SimpleNamespace when the SDK is unbound."""
    if request_cls is None:
        return SimpleNamespace(**fields)
    builder = request_cls.builder()
    for name, value in fields.items():
        builder = getattr(builder, name)(value)
    return builder.build()


class FeishuAdapter(BasePlatformAdapter):
    """Feishu/Lark bot adapter."""
    # Answers /p/<profile>/... on the default listener for a served secondary (shared_ingress).
    serves_profile_prefix: bool = True

    supports_code_blocks = True  # Feishu renders fenced code blocks
    splits_long_messages = True  # send() chunks via truncate_message(MAX_MESSAGE_LENGTH)

    MAX_MESSAGE_LENGTH = 8000
    CHAT_LOCK_MAX_SIZE: int = 1000  # distinct chat IDs kept in _chat_locks before LRU eviction
    _SPLIT_THRESHOLD = 4000  # chunk near Feishu's ~4096-char client split → continuation almost certain

    # --- Lifecycle — init / settings / connect / disconnect ---
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.FEISHU)
        self._settings = self._load_settings(config.extra or {})
        self._apply_settings(self._settings)
        self._client: Optional[Any] = None
        # Adapter-owned pool for blocking SDK calls, recreated on demand: a torn-down default
        # executor can no longer wedge sends with "Executor shutdown has been called".
        # See issue #10849.
        self._sdk_executor_lock = threading.Lock()
        self._sdk_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._sdk_executor_closing = False  # set on disconnect so a real teardown isn't resurrected
        self._ws_client = self._ws_future = self._ws_supervisor = self._ws_thread_loop = None
        self._ws_restart_backoff = 5.0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._webhook_runner = self._webhook_site = self._event_handler = None
        self._seen_message_ids: Dict[str, float] = {}  # message_id → seen_at (time.time())
        self._seen_message_order: List[str] = []
        self._dedup_state_path = get_hermes_home() / "feishu_seen_message_ids.json"
        self._dedup_lock = threading.Lock()
        # Serializes the offloaded dedup-state flushes so two concurrent
        # inbound messages cannot land their writes out of order.
        self._dedup_persist_lock = asyncio.Lock()
        self._sender_name_cache: Dict[str, tuple[str, float]] = {}  # sender_id → (name, expire_at)
        self._webhook_rate_counts: Dict[str, tuple[int, float]] = {}  # rate_key → (count, window_start)
        self._webhook_anomaly_counts: Dict[str, tuple[int, str, float]] = {}  # ip → (count, last_status, first_seen)
        self._card_action_tokens: Dict[str, float] = {}  # token → first_seen_time
        # Inbound events that arrived before the loop was ready; one drainer thread replays them.
        self._pending_inbound_events: List[Any] = []
        self._pending_inbound_lock = threading.Lock()
        self._pending_drain_scheduled = False
        self._pending_inbound_max_depth = 1000  # cap queue; drop oldest beyond
        self._chat_locks: "collections.OrderedDict[str, asyncio.Lock]" = collections.OrderedDict()  # chat_id → lock (per-chat serial processing, LRU-bounded)
        self._chat_info_cache: Dict[str, Dict[str, Any]] = {}
        self._message_text_cache: "OrderedDict[str, Optional[str]]" = OrderedDict()
        self._app_lock_identity: Optional[str] = None
        self._text_batch_state = FeishuBatchState()
        self._pending_text_batches = self._text_batch_state.events
        self._pending_text_batch_tasks = self.…26773 tokens truncated…id)
            return self._message_text_cache[message_id]
        try:
            request = self._build_get_message_request(message_id)
            response = await self._run_blocking(self._client.im.v1.message.get, request)
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
            parent_mentions = getattr(parent, "mentions", None) if parent else None
            text = self._extract_text_from_raw_content(
                msg_type=msg_type, raw_content=raw_content, mentions=parent_mentions,
            )
            self._message_text_cache[message_id] = text
            while len(self._message_text_cache) > _FEISHU_MESSAGE_TEXT_CACHE_SIZE:
                self._message_text_cache.popitem(last=False)
            return text
        except Exception:
            logger.warning("[Feishu] Failed to fetch parent message %s", message_id, exc_info=True)
            return None

    def _extract_text_from_raw_content(
        self, *, msg_type: str, raw_content: str, mentions: Optional[Sequence[Any]] = None,
    ) -> Optional[str]:
        normalized = self._normalize(msg_type, raw_content, mentions)
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

    def _refresh_group_rules_from_hot_file(self) -> None:
        """Overlay ``~/.hermes/feishu_group_rules.json`` onto boot ``group_rules`` (mtime-cached)."""
        settings = getattr(self, "_settings", None)
        if settings is None:
            return
        try:
            from plugins.platforms.feishu.feishu_group_rules import load_hot_group_rules

            hot_raw = load_hot_group_rules()
        except Exception:
            logger.warning("[Feishu] Failed to refresh group rules from hot file", exc_info=True)
            return
        boot = getattr(settings, "group_rules", None) or {}
        if not hot_raw:
            self._group_rules = dict(boot)
            return
        merged = dict(boot)
        for chat_id, rule_cfg in hot_raw.items():
            merged[chat_id] = _merge_hot_group_rule(merged.get(chat_id), rule_cfg)
        self._group_rules = merged

    # --- Inbound admission ---
    def _admit(self, sender: Any, message: Any) -> Optional[RejectReason]:
        self._refresh_group_rules_from_hot_file()
        sender_ids = _sender_identity(sender)
        self_ids = frozenset(v for v in (self._bot_open_id, self._bot_user_id) if v)
        is_bot = _is_bot_sender(sender)
        is_group = getattr(message, "chat_type", "p2p") != "p2p"
        chat_id = getattr(message, "chat_id", "") or ""
        require_mention = is_group and self._require_mention_for(chat_id)
        # Defensive only — Feishu doesn't echo our outbound back as inbound,
        # and open_id is always populated on both sides.
        if self_ids and sender_ids & self_ids:
            return "self_echo"
        if is_bot:
            mode = self._allow_bots
            if mode not in ("mentions", "all"):
                return "bots_disabled"
            if not self_ids or not sender_ids:  # pre-hydration or malformed payloads
                return "self_ids_unknown"
            # The group step below enforces mentions when require_mention is on; cover the rest here.
            if mode == "mentions" and not require_mention and not self._mentions_self(message):
                return "bot_not_mentioned"
        if not is_group:
            # _allow_all_dm is snapshotted per-profile in _load_settings: _admit runs on the
            # lark_oapi WS thread with no secret scope, so a bare os.getenv would read the
            # default profile's value.
            # Empty FEISHU_ALLOWED_USERS is setup's pairing-mode default: forward DMs so the
            # pairing handshake can run (gateway auth fail-closes until approval).
            if self._allow_all_dm or not self._allowed_group_users:
                return None
            return None if sender_ids & self._allowed_group_users else "dm_policy_rejected"
        if not self._allow_group_message(getattr(sender, "sender_id", None), chat_id, is_bot=is_bot):
            return "group_policy_rejected"
        if require_mention and not self._mentions_self(message):
            return "group_policy_rejected"
        return None

    # Class default so the first drop after construction is the one that warns.
    _warned_empty_allowlist_deny = False

    def _warn_once_empty_allowlist_deny(self, chat_id: str) -> None:
        """One WARNING when group traffic dies on the untouched allowlist default (#111420).

        The policy read is scoped per profile on purpose — never inherit another profile's
        FEISHU_GROUP_POLICY here; only make the resulting deny visible above DEBUG.
        """
        if self._warned_empty_allowlist_deny:
            return
        from plugins.platforms.feishu.feishu_admission_diagnostics import empty_allowlist_drop_warning
        text = empty_allowlist_drop_warning(
            chat_id=chat_id, group_rules=self._group_rules,
            default_group_policy=self._default_group_policy, allowed_group_users=self._allowed_group_users,
        )
        if text:
            self._warned_empty_allowlist_deny = True
            logger.warning(text)

    def _require_mention_for(self, chat_id: str) -> bool:
        rule = self._group_rules.get(chat_id) if chat_id else None
        if rule and rule.require_mention is not None:
            return rule.require_mention
        return self._require_mention

    def _allow_group_message(self, sender_id: Any, chat_id: str = "", *, is_bot: bool = False) -> bool:
        """Per-group policy gate for non-DM traffic."""
        sender_ids = {getattr(sender_id, "open_id", None), getattr(sender_id, "user_id", None)} - {None}
        if sender_ids and self._admins and (sender_ids & self._admins):
            return True
        rule = self._group_rules.get(chat_id) if chat_id else None
        if rule:
            policy, allowlist, blacklist = rule.policy, rule.allowlist, rule.blacklist
        else:
            policy, allowlist, blacklist = self._default_group_policy or self._group_policy, self._allowed_group_users, set()
        # Channel locks apply to everyone; allowlist/blacklist only gate humans (bots were
        # already cleared upstream by FEISHU_ALLOW_BOTS).
        if policy in ("disabled", "admin_only"):
            return False
        if policy == "open" or is_bot:
            return True
        if policy == "allowlist":
            return bool(sender_ids and (sender_ids & allowlist))
        if policy == "blacklist":
            return bool(sender_ids and not (sender_ids & blacklist))
        return bool(sender_ids and (sender_ids & self._allowed_group_users))

    def _mentions_self(self, message: Any) -> bool:
        # @_all is Feishu's @everyone placeholder.
        raw_content = getattr(message, "content", "") or ""
        if "@_all" in raw_content:
            return True
        mentions = getattr(message, "mentions", None) or []
        if mentions and self._message_mentions_bot(mentions):
            return True
        normalized = self._normalize(getattr(message, "message_type", "") or "", raw_content, getattr(message, "mentions", None))
        return self._post_mentions_bot(normalized.mentions)

    def _message_mentions_bot(self, mentions: List[Any]) -> bool:
        # Same precedence as _FeishuBotIdentity.matches (open_id > user_id > name); a non-empty
        # bot_name here only matches an exact stripped name, while an empty bot_name never matches.
        bot = self._bot_identity()
        for mention in mentions:
            mention_id = getattr(mention, "id", None)
            if bot.matches(
                open_id=(getattr(mention_id, "open_id", None) or "").strip(),
                user_id=(getattr(mention_id, "user_id", None) or "").strip(),
                name=(getattr(mention, "name", None) or "").strip(),
            ):
                return True
        return False

    def _post_mentions_bot(self, mentions: List[FeishuMentionRef]) -> bool:
        return any(m.is_self for m in mentions)

    def _bot_identity(self) -> _FeishuBotIdentity:
        return _FeishuBotIdentity(open_id=self._bot_open_id, user_id=self._bot_user_id, name=self._bot_name)

    async def _hydrate_bot_identity(self) -> None:
        """Best-effort bot identity discovery for mention gating and self-event filtering.

        /bot/v3/info (tenant token, no extra scopes) always wins over env values so stale
        FEISHU_BOT_* from app migrations can't break gating; the application-info endpoint is
        a name-only fallback. On failure env-provided values are kept.
        """
        if not self._client:
            return
        try:
            content = await self._tenant_get_raw("/open-apis/bot/v3/info")
            if content:
                payload = json.loads(content)
                parsed = _parse_bot_response(payload) or {}
                open_id = (parsed.get("bot_open_id") or "").strip()
                bot_name = (parsed.get("bot_name") or "").strip()
                if open_id:
                    if self._bot_open_id and self._bot_open_id != open_id:
                        logger.warning(
                            "[Feishu] FEISHU_BOT_OPEN_ID is stale; using /bot/v3/info open_id for group @mention gating."
                        )
                    self._bot_open_id = open_id
                if bot_name:
                    if self._bot_name and self._bot_name != bot_name:
                        logger.info(
                            "[Feishu] FEISHU_BOT_NAME differs from /bot/v3/info; using hydrated bot name for group @mention gating."
                        )
                    self._bot_name = bot_name
        except Exception:
            logger.debug("[Feishu] /bot/v3/info probe failed during hydration", exc_info=True)

        if self._bot_name:
            return
        # Name-only fallback; needs admin:app.info:readonly or application:application:self_manage.
        try:
            request = self._build_get_application_request(app_id=self._app_id, lang="en_us")
            response = await self._run_blocking(self._client.application.v6.application.get, request)
            if not response or not response.success():
                code = getattr(response, "code", None)
                if code == 99991672:
                    logger.warning(
                        "[Feishu] Unable to hydrate bot name from application info. "
                        "Grant admin:app.info:readonly or application:application:self_manage "
                        "so group @mention gating can resolve the bot name precisely."
                    )
                return
            app = getattr(getattr(response, "data", None), "app", None)
            app_name = (getattr(app, "app_name", None) or "").strip()
            if app_name and not self._bot_name:
                self._bot_name = app_name
        except Exception:
            logger.debug("[Feishu] Failed to hydrate bot name from application info", exc_info=True)

    # --- Deduplication — seen message ID cache (persistent) ---
    def _load_seen_message_ids(self) -> None:
        try:
            payload = json.loads(self._dedup_state_path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            logger.warning("[Feishu] Failed to load persisted dedup state from %s", self._dedup_state_path, exc_info=True)
            return
        seen_data = payload.get("message_ids", {}) if isinstance(payload, dict) else {}
        now = time.time()
        ttl = _FEISHU_DEDUP_TTL_SECONDS
        if isinstance(seen_data, list):  # legacy format: plain list of IDs, no timestamps
            entries: Dict[str, float] = {str(item).strip(): 0.0 for item in seen_data if str(item).strip()}
        elif isinstance(seen_data, dict):
            entries = {}
            for key, value in seen_data.items():
                if isinstance(key, str) and key.strip() and isinstance(value, (int, float, str)):
                    try:
                        entries[key] = float(value)
                    except ValueError:
                        pass
        else:
            return
        # Drop TTL-expired entries; ts=0.0 (legacy) is immortal for one migration cycle.
        valid: Dict[str, float] = {m: ts for m, ts in entries.items() if ts == 0.0 or ttl <= 0 or now - ts < ttl}
        # Size cap keeps the most recently seen IDs.
        sorted_ids = sorted(valid, key=lambda k: valid[k], reverse=True)[:self._dedup_cache_size]
        self._seen_message_order = list(reversed(sorted_ids))
        self._seen_message_ids = {k: valid[k] for k in sorted_ids}

    def _persist_seen_message_ids(self) -> None:
        try:
            from hermes_constants import mkdir_under_hermes_home
            mkdir_under_hermes_home(self._dedup_state_path.parent)
            with self._dedup_lock:
                recent = self._seen_message_order[-self._dedup_cache_size:]
                # Save as {msg_id: timestamp} so TTL filtering works across restarts.
                payload = {"message_ids": {k: self._seen_message_ids[k] for k in recent if k in self._seen_message_ids}}
            atomic_json_write(self._dedup_state_path, payload, indent=None)
        except OSError:
            logger.warning("[Feishu] Failed to persist dedup state to %s", self._dedup_state_path, exc_info=True)

    async def _is_duplicate(self, message_id: str) -> bool:
        now, ttl = time.time(), _FEISHU_DEDUP_TTL_SECONDS
        with self._dedup_lock:
            seen_at = self._seen_message_ids.get(message_id)
            if seen_at is not None and (ttl <= 0 or now - seen_at < ttl):
                return True
            self._seen_message_ids[message_id] = now  # wall-clock so TTL survives restarts
            self._seen_message_order.append(message_id)
            while len(self._seen_message_order) > self._dedup_cache_size:
                self._seen_message_ids.pop(self._seen_message_order.pop(0), None)
        # atomic_json_write() fsyncs; this runs on the event loop for every inbound message, so
        # offload the flush onto the adapter-owned pool: the loop's default executor may already
        # be torn down by a dead background loop, which used to wedge every inbound message in
        # the dedup gate (#111020). The lock keeps flushes in mutation order (the snapshot
        # inside the worker is taken under _dedup_lock, but the write itself is not).
        async with self._dedup_persist_lock_or_create():
            await self._run_blocking(self._persist_seen_message_ids)
        return False

    def _dedup_persist_lock_or_create(self) -> asyncio.Lock:
        # Tests build bare adapters via object.__new__ and install dedup state
        # by hand; create the lock lazily so those fixtures keep working.
        lock = getattr(self, "_dedup_persist_lock", None)
        if lock is None:
            lock = self._dedup_persist_lock = asyncio.Lock()
        return lock

    # --- Outbound payload construction and send pipeline ---
    def _build_outbound_payload(self, content: str, *, prefer_post: bool = False) -> tuple[str, str]:
        # Feishu clients render markdown tables inside ``post`` ``md`` elements natively, so tables
        # take the common markdown path (no text downgrade). ``prefer_post`` lets ``send`` keep every
        # chunk of a split markdown reply as ``post`` even when a chunk alone looks like prose.
        # The previous table-downgrade branch forced any table-containing message to ``text``, which left
        # Feishu readers seeing the raw pipe-and-dash source instead of a rendered table. ``prefer_post``
        # lets ``send`` treat the chunk as part of a larger markdown document: when a long markdown reply is
        # split at MAX_MESSAGE_LENGTH, the per-chunk regex would otherwise mis-classify a plain-prose chunk
        # as ``text``. See #26841.
        if prefer_post or _MARKDOWN_HINT_RE.search(content):
            return "post", _build_markdown_post_payload(content)
        return "text", json.dumps({"text": content}, ensure_ascii=False)

    @staticmethod
    def _get_audio_duration_ms(file_path: str) -> int:
        """OGG/Opus duration in ms (pure Python): last granule position / 48000 Hz; 0 on non-OGG or error."""
        import struct
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            pos = last_granule = 0
            while pos < len(data) - 27:
                pos = data.find(b"OggS", pos)
                if pos == -1 or pos + 27 > len(data):
                    break
                granule = struct.unpack_from("<q", data, pos + 6)[0]
                num_segments = data[pos + 26]
                last_granule = granule if granule > 0 else last_granule
                if pos + 27 + num_segments > len(data):
                    break
                pos += num_segments + sum(data[pos + 27 : pos + 27 + num_segments])
            return int(last_granule / 48000 * 1000) if last_granule > 0 else 0
        except Exception:
            return 0

    async def _send_uploaded_file_message(
        self, *, chat_id: str, file_path: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]],
        caption: Optional[str] = None, file_name: Optional[str] = None, outbound_message_type: str = "file",
    ) -> SendResult:
        if not self._client:
            return SendResult(success=False, error="Not connected")
        if not os.path.exists(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")

        display_name = file_name or os.path.basename(file_path)
        upload_file_type, resolved_message_type = self._resolve_outbound_file_routing(
            file_path=display_name, requested_message_type=outbound_message_type,
        )
        try:
            duration_ms = self._get_audio_duration_ms(file_path) if upload_file_type == "opus" else 0
            with open(file_path, "rb") as file_obj:
                body = self._build_file_upload_body(
                    file_type=upload_file_type, file_name=display_name, file=file_obj, duration=duration_ms,
                )
                request = self._build_file_upload_request(body)
                upload_response = await self._run_blocking(self._client.im.v1.file.create, request)
            file_key = self._extract_response_field(upload_response, "file_key")
            if not file_key:
                return self._response_error_result(
                    upload_response, default_message="file upload failed",
                    override_error="Feishu file upload missing file_key",
                )

            key_payload = {"file_key": file_key}
            message_response = await self._send_uploaded_key(
                chat_id=chat_id, reply_to=reply_to, metadata=metadata, caption=caption,
                key_msg_type=resolved_message_type, key_payload=key_payload,
                media_tag={"tag": "media", "file_key": file_key, "file_name": display_name},
            )
            # Audio may fail with 99992402 under thread_id routing: retry as a reply to the
            # thread's last message, then fall back to a plain chat_id send.
            if (not caption
                    and not self._response_succeeded(message_response)
                    and getattr(message_response, "code", None) == 99992402
                    and resolved_message_type == "audio"
                    and (metadata or {}).get("thread_id")):
                payload = json.dumps(key_payload, ensure_ascii=False)
                thread_msg_id = (metadata or {}).get("reply_to_message_id")
                if not thread_msg_id:
                    thread_msg_id = await self._fetch_last_message_in_thread((metadata or {}).get("thread_id"))
                if thread_msg_id:
                    logger.info("[Feishu] Audio: retrying via reply API in thread")
                    message_response = await self._feishu_send_with_retry(
                        chat_id=chat_id, msg_type="audio", payload=payload, reply_to=thread_msg_id, metadata=metadata,
                    )
                if not self._response_succeeded(message_response):
                    logger.warning("[Feishu] Audio send failed in thread, retrying with chat_id")
                    message_response = await self._feishu_send_with_retry(
                        chat_id=chat_id, msg_type="audio", payload=payload, reply_to=None, metadata=None,
                    )
            return self._finalize_send_result(message_response, "file send failed")
        except Exception as exc:
            logger.error("[Feishu] Failed to send file %s: %s", file_path, exc, exc_info=True)
            return SendResult(success=False, error=str(exc))

    async def _send_uploaded_key(
        self, *, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]], caption: Optional[str],
        key_msg_type: str, key_payload: Dict[str, str], media_tag: Dict[str, str],
    ) -> Any:
        """Send an uploaded image/file key: as a captioned ``post`` or as a bare key message."""
        if caption:
            msg_type = "post"
            payload = self._build_media_post_payload(caption=caption, media_tag=media_tag)
        else:
            msg_type = key_msg_type
            payload = json.dumps(key_payload, ensure_ascii=False)
        return await self._feishu_send_with_retry(
            chat_id=chat_id, msg_type=msg_type, payload=payload, reply_to=reply_to, metadata=metadata,
        )

    async def _fetch_last_message_in_thread(self, thread_id: str) -> Optional[str]:
        """Fetch the last message_id in a thread for reply-based routing."""
        if not self._client or not thread_id:
            return None
        try:
            from lark_oapi.api.im.v1 import ListMessageRequest
            request = ListMessageRequest.builder().container_id_type("thread").container_id(thread_id).page_size(1).build()
            response = await self._run_blocking(self._client.im.v1.message.list, request)
            if self._response_succeeded(response):
                items = getattr(getattr(response, "data", None), "items", None)
                if items and len(items) > 0:
                    return getattr(items[0], "message_id", None)
        except Exception as exc:
            logger.debug("[Feishu] Failed to fetch last message in thread %s: %s", thread_id, exc)
        return None

    async def _send_raw_message(
        self, *, chat_id: str, msg_type: str, payload: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]],
    ) -> Any:
        thread_id = (metadata or {}).get("thread_id")
        effective_reply_to = reply_to or ((metadata or {}).get("reply_to_message_id") if thread_id else None)
        if effective_reply_to:
            body = self._build_reply_message_body(
                content=payload, msg_type=msg_type, reply_in_thread=bool(thread_id), uuid_value=str(uuid.uuid4()),
            )
            request = self._build_reply_message_request(effective_reply_to, body)
            return await self._run_blocking(self._client.im.v1.message.reply, request)
        if thread_id:
            # reply→create fallback inside a topic: thread_id as receive_id keeps it in the topic.
            receive_id, receive_id_type = thread_id, "thread_id"
        elif chat_id.startswith("feishu_user_id:"):
            receive_id, receive_id_type = chat_id.split(":", 1)[1], "user_id"
        else:
            receive_id, receive_id_type = chat_id, "open_id" if chat_id.startswith("ou_") else "chat_id"
        body = self._build_create_message_body(
            receive_id=receive_id, msg_type=msg_type, content=payload, uuid_value=str(uuid.uuid4()),
        )
        request = self._build_create_message_request(receive_id_type, body)
        return await self._run_blocking(self._client.im.v1.message.create, request)

    @staticmethod
    def _response_succeeded(response: Any) -> bool:
        return bool(response and getattr(response, "success", lambda: False)())

    @staticmethod
    def _extract_response_field(response: Any, field_name: str) -> Any:
        data = getattr(response, "data", None) if FeishuAdapter._response_succeeded(response) else None
        return getattr(data, field_name, None) if data else None

    def _response_error_result(
        self, response: Any, *, default_message: str, override_error: Optional[str] = None,
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
            success=True, message_id=self._extract_response_field(response, "message_id"),
            raw_response=response,
        )

    # --- Connection internals — websocket / webhook setup ---
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
                    attempt + 1, _FEISHU_CONNECT_ATTEMPTS, wait_seconds, exc,
                )
                await asyncio.sleep(wait_seconds)

    async def _supervise_websocket_thread(self) -> None:
        """Restart the WS client thread if it dies while the adapter is up.

        ``lark_oapi.start()`` only returns on fatal errors; without this watcher a dead thread
        left the profile silently deaf until a gateway restart. Rebuild with capped backoff.

        See #73779.
        """
        backoff = initial_backoff = float(self._ws_restart_backoff)
        last_dead: Optional[asyncio.Future] = None
        while self._running:
            ws_future = self._ws_future
            if ws_future is None:
                return
            try:
                await asyncio.shield(ws_future)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            # Deliberate disconnects nil ``_ws_client``/``_running`` first; only restart a live link.
            if not self._running or self._ws_client is None:
                return
            if ws_future is not last_dead:
                logger.error("[Feishu] WebSocket client thread exited unexpectedly; restarting in %.0fs", backoff)
                last_dead = ws_future
                # Still running, link unproven: ``connected`` stays wrong until ``_ws_link_up`` re-stamps it.
                self._write_runtime_status_safe(
                    "ws_link_lost", platform_state="retrying", error_code=None,
                    error_message="Feishu websocket link lost; rebuilding",
                )
            await asyncio.sleep(backoff)
            if not self._running:
                return
            try:
                await self._connect_websocket()
                backoff = initial_backoff
            except Exception as exc:
                logger.warning("[Feishu] WebSocket restart failed (retrying): %s", exc)
                backoff = min(backoff * 2, 60.0)

    def _ws_link_up(self, ws_client: Any) -> None:
        """WS thread reports its link is up (SDK receive loop scheduled); re-stamp ``connected`` after a rebuild."""
        if self._running and self._ws_client is ws_client:
            self._mark_connected()

    def _ws_link_retrying(self, ws_client: Any) -> None:
        """WS thread reports the SDK's own reconnect ladder started; ``_ws_link_up`` re-stamps ``connected``."""
        if self._running and self._ws_client is ws_client:
            self._write_runtime_status_safe(
                "ws_link_lost", platform_state="retrying", error_code=None,
                error_message="Feishu websocket link lost; reconnecting",
            )

    async def _connect_websocket(self) -> None:
        if not FEISHU_WEBSOCKET_AVAILABLE:
            raise RuntimeError("websockets not installed; websocket mode unavailable")
        domain = self._prepare_client()
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
            # Without the "channel" UA tag Feishu won't push group @mention events over WS.
            extra_ua_tags=["channel"],
        )
        # The lark SDK owns this thread and fires every event/card callback on it; those hop back
        # to the adapter loop via run_coroutine_threadsafe, which copies the CALLER's context — so
        # whatever scope the WS thread carries is what pre-handler work (inbound media caching,
        # .update_response marker, reactions env, drive comments) runs under. A bare executor
        # thread has an empty context = launch profile. connect() runs inside the profile scope
        # under multiplex (and the supervisor task inherits it), so snapshot it here.
        self._ws_future = loop.run_in_executor(
            self._get_sdk_executor(), contextvars.copy_context().run, _run_official_feishu_ws_client, self._ws_client, self)

    async def _connect_webhook(self) -> None:
        if not FEISHU_WEBHOOK_AVAILABLE:
            raise RuntimeError("aiohttp not installed; webhook mode unavailable")
        self._prepare_client()
        await self._hydrate_bot_identity()
        # client_max_size backstops the bounded reader in _handle_webhook_request on every read path.
        # See #58536, #58902, #59180.
        app = web.Application(client_max_size=_FEISHU_WEBHOOK_MAX_BODY_BYTES)
        app.router.add_post(self._webhook_path, self._handle_webhook_request)
        # Shared-listener mode (multiplex secondary): no bind; served at /p/<profile>/<webhook_path>.
        from gateway.platforms.shared_ingress import bind_listener
        self._webhook_runner = await bind_listener(self, app, self._webhook_host, self._webhook_port, self._webhook_path)

    def _prepare_client(self) -> Any:
        """Build the lark client + event dispatcher for this adapter's domain; returns the SDK domain."""
        domain = _sdk_domain(self._domain_name)
        self._client = self._build_lark_client(domain)
        self._event_handler = self._build_event_handler()
        if self._event_handler is None:
            raise RuntimeError("failed to build Feishu event handler")
        return domain

    def _build_lark_client(self, domain: Any) -> Any:
        return _build_lark_client(self._app_id, self._app_secret, domain)

    async def _feishu_send_with_retry(
        self, *, chat_id: str, msg_type: str, payload: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]],
    ) -> Any:
        last_error: Optional[Exception] = None
        active_reply_to = reply_to

        async def _raw(reply_target: Optional[str]) -> Any:
            return await self._send_raw_message(
                chat_id=chat_id, msg_type=msg_type, payload=payload, reply_to=reply_target, metadata=metadata,
            )

        for attempt in range(_FEISHU_SEND_ATTEMPTS):
            try:
                response = await _raw(active_reply_to)
                # Reply target withdrawn/missing → post a new message to the chat instead.
                if active_reply_to and not self._response_succeeded(response):
                    code = getattr(response, "code", None)
                    if code in _FEISHU_REPLY_FALLBACK_CODES:
                        if (metadata or {}).get("thread_id"):
                            logger.warning(
                                "[Feishu] Reply to %s failed in thread %s (code %s — message withdrawn/missing); "
                                "skipping top-level fallback to avoid creating a new topic",
                                active_reply_to, (metadata or {}).get("thread_id"), code,
                            )
                            return response
                        logger.warning(
                            "[Feishu] Reply to %s failed (code %s — message withdrawn/missing); "
                            "falling back to new message in chat %s",
                            active_reply_to, code, chat_id,
                        )
                        active_reply_to = None
                        response = await _raw(None)
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
                    attempt + 1, _FEISHU_SEND_ATTEMPTS, chat_id, wait_seconds, exc,
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

    # --- Lark API request builders (SimpleNamespace fallback when the SDK is unbound) ---
    @staticmethod
    def _build_get_chat_request(chat_id: str) -> Any:
        return _sdk_build(GetChatRequest, chat_id=chat_id)

    @staticmethod
    def _build_get_message_request(message_id: str) -> Any:
        return _sdk_build(GetMessageRequest, message_id=message_id)

    @staticmethod
    def _build_message_resource_request(*, message_id: str, file_key: str, resource_type: str) -> Any:
        return _sdk_build(GetMessageResourceRequest, message_id=message_id, file_key=file_key, type=resource_type)

    @staticmethod
    def _build_get_application_request(*, app_id: str, lang: str) -> Any:
        return _sdk_build(GetApplicationRequest, app_id=app_id, lang=lang)

    @staticmethod
    def _build_reply_message_body(*, content: str, msg_type: str, reply_in_thread: bool, uuid_value: str) -> Any:
        return _sdk_build(
            ReplyMessageRequestBody,
            content=content, msg_type=msg_type, reply_in_thread=reply_in_thread, uuid=uuid_value,
        )

    @staticmethod
    def _build_reply_message_request(message_id: str, request_body: Any) -> Any:
        return _sdk_build(ReplyMessageRequest, message_id=message_id, request_body=request_body)

    @staticmethod
    def _build_update_message_body(*, msg_type: str, content: str) -> Any:
        return _sdk_build(UpdateMessageRequestBody, msg_type=msg_type, content=content)

    @staticmethod
    def _build_update_message_request(message_id: str, request_body: Any) -> Any:
        return _sdk_build(UpdateMessageRequest, message_id=message_id, request_body=request_body)

    @staticmethod
    def _build_delete_message_request(message_id: str) -> Any:
        return _sdk_build(DeleteMessageRequest, message_id=message_id)

    @staticmethod
    def _build_create_message_body(*, receive_id: str, msg_type: str, content: str, uuid_value: str) -> Any:
        return _sdk_build(
            CreateMessageRequestBody, receive_id=receive_id, msg_type=msg_type, content=content, uuid=uuid_value,
        )

    @staticmethod
    def _build_create_message_request(receive_id_type: str, request_body: Any) -> Any:
        return _sdk_build(CreateMessageRequest, receive_id_type=receive_id_type, request_body=request_body)

    @staticmethod
    def _build_image_upload_body(*, image_type: str, image: Any) -> Any:
        return _sdk_build(CreateImageRequestBody, image_type=image_type, image=image)

    @staticmethod
    def _build_image_upload_request(request_body: Any) -> Any:
        return _sdk_build(CreateImageRequest, request_body=request_body)

    @staticmethod
    def _build_file_upload_body(*, file_type: str, file_name: str, file: Any, duration: int = 0) -> Any:
        if CreateFileRequestBody is None:
            return SimpleNamespace(file_type=file_type, file_name=file_name, file=file, duration=duration)
        fields: Dict[str, Any] = {"file_type": file_type, "file_name": file_name, "file": file}
        if duration > 0:
            fields["duration"] = duration
        return _sdk_build(CreateFileRequestBody, **fields)

    @staticmethod
    def _build_file_upload_request(request_body: Any) -> Any:
        return _sdk_build(CreateFileRequest, request_body=request_body)

    def _build_media_post_payload(self, *, caption: str, media_tag: Dict[str, str]) -> str:
        payload = json.loads(_build_markdown_post_payload(caption))
        content = payload.setdefault("zh_cn", {}).setdefault("content", [])
        content.append([media_tag])
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _resolve_outbound_file_routing(*, file_path: str, requested_message_type: str) -> tuple[str, str]:
        # requested_message_type is accepted for call-site symmetry; routing is by extension only.
        ext = Path(file_path).suffix.lower()
        if ext in _FEISHU_OPUS_UPLOAD_EXTENSIONS:
            return "opus", "audio"
        if ext in _FEISHU_MEDIA_UPLOAD_EXTENSIONS:
            return "mp4", "media"
        if ext in _FEISHU_DOC_UPLOAD_TYPES:
            return _FEISHU_DOC_UPLOAD_TYPES[ext], "file"
        return _FEISHU_FILE_UPLOAD_TYPE, "file"


# --- QR scan-to-create onboarding (device-code flow; Feishu creates a configured bot app) ---


def _accounts_base_url(domain: str) -> str:
    return _ONBOARD_ACCOUNTS_URLS.get(domain, _ONBOARD_ACCOUNTS_URLS["feishu"])


def _onboard_open_base_url(domain: str) -> str:
    return _ONBOARD_OPEN_URLS.get(domain, _ONBOARD_OPEN_URLS["feishu"])


def _post_registration(base_url: str, body: Dict[str, str]) -> dict:
    """POST form data to the registration endpoint; parse JSON even on 4xx (poll's pending is a 400)."""
    req = Request(
        f"{base_url}{_REGISTRATION_PATH}", data=urlencode(body).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urlopen(req, timeout=_ONBOARD_REQUEST_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body_bytes = exc.read()
        if body_bytes:
            try:
                return json.loads(body_bytes.decode("utf-8"))
            except (ValueError, json.JSONDecodeError):
                raise exc from None
        raise


def _init_registration(domain: str = "feishu") -> None:
    """Raise RuntimeError unless the registration environment supports client_secret auth."""
    res = _post_registration(_accounts_base_url(domain), {"action": "init"})
    methods = res.get("supported_auth_methods") or []
    if "client_secret" not in methods:
        raise RuntimeError(
            f"Feishu / Lark registration environment does not support client_secret auth. "
            f"Supported: {methods}"
        )


def _begin_registration(domain: str = "feishu") -> dict:
    """Start the device-code flow. Returns device_code, qr_url, user_code, interval, expire_in."""
    res = _post_registration(_accounts_base_url(domain), {
        "action": "begin", "archetype": "PersonalAgent", "auth_method": "client_secret", "request_user_info": "open_id",
    })
    device_code = res.get("device_code")
    if not device_code:
        raise RuntimeError("Feishu / Lark registration did not return a device_code")
    qr_url = res.get("verification_uri_complete", "")
    qr_url += ("&" if "?" in qr_url else "?") + "from=hermes&tp=hermes"
    return {
        "device_code": device_code, "qr_url": qr_url, "user_code": res.get("user_code", ""),
        "interval": res.get("interval") or 5, "expire_in": res.get("expire_in") or 600,
    }


def _poll_registration(*, device_code: str, interval: int, expire_in: int, domain: str = "feishu") -> Optional[dict]:
    """Poll until scan (→ {app_id, app_secret, domain, open_id}), or None on denial/timeout."""
    deadline = time.monotonic() + expire_in
    current_domain = domain
    domain_switched = False
    poll_count = 0

    while time.monotonic() < deadline:
        try:
            body = {"action": "poll", "device_code": device_code, "tp": "ob_app"}
            res = _post_registration(_accounts_base_url(current_domain), body)
        except (URLError, OSError, json.JSONDecodeError):
            time.sleep(interval)
            continue

        poll_count += 1
        if poll_count == 1:
            print("  Fetching configuration results...", end="", flush=True)
        elif poll_count % 6 == 0:
            print(".", end="", flush=True)

        # Domain auto-detection; fall through — this same response may carry credentials.
        user_info = res.get("user_info") or {}
        if user_info.get("tenant_brand") == "lark" and not domain_switched:
            current_domain = "lark"
            domain_switched = True
        if res.get("client_id") and res.get("client_secret"):
            if poll_count > 0:
                print()  # newline after "Fetching configuration results..." dots
            return {
                "app_id": res["client_id"], "app_secret": res["client_secret"], "domain": current_domain,
                "open_id": user_info.get("open_id"),
            }

        error = res.get("error", "")
        if error in {"access_denied", "expired_token"}:
            if poll_count > 0:
                print()
            logger.warning("[Feishu onboard] Registration %s", error)
            return None
        time.sleep(interval)  # authorization_pending or unknown — keep polling

    if poll_count > 0:
        print()
    logger.warning("[Feishu onboard] Poll timed out after %ds", expire_in)
    return None


try:
    import qrcode as _qrcode_mod
except (ImportError, TypeError):
    _qrcode_mod = None  # type: ignore[assignment]


def _render_qr(url: str) -> bool:
    """Try to render a QR code in the terminal. Returns True if successful."""
    if _qrcode_mod is None:
        return False
    try:
        qr = _qrcode_mod.QRCode()
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        return True
    except Exception:
        return False


def probe_bot(app_id: str, app_secret: str, domain: str) -> Optional[dict]:
    """Probe /open-apis/bot/v3/info → {"bot_name", "bot_open_id"} (app-scoped open_id, NOT app_id) or None.

    Onboarding runs before connect(), so load the SDK here instead of always falling back to HTTP.
    """
    if _load_lark_oapi():
        return _probe_bot_sdk(app_id, app_secret, domain)
    return _probe_bot_http(app_id, app_secret, domain)


def _build_onboard_client(app_id: str, app_secret: str, domain: str) -> Any:
    """Build a lark Client for the given credentials and domain name ("feishu"/"lark")."""
    return _build_lark_client(app_id, app_secret, _sdk_domain(domain))


def _parse_bot_response(data: dict) -> Optional[dict]:
    # /bot/v3/info returns bot.app_name; legacy paths used bot_name — accept both.
    if data.get("code") != 0:
        return None
    bot = data.get("bot") or data.get("data", {}).get("bot") or {}
    return {
        "bot_name": bot.get("app_name") or bot.get("bot_name"),
        "bot_open_id": bot.get("open_id"),
    }


def _probe_bot_sdk(app_id: str, app_secret: str, domain: str) -> Optional[dict]:
    """Probe bot info using lark_oapi SDK."""
    try:
        resp = _build_onboard_client(app_id, app_secret, domain).request(_tenant_get_request("/open-apis/bot/v3/info"))
        content = getattr(getattr(resp, "raw", None), "content", None)
        return None if content is None else _parse_bot_response(json.loads(content))
    except Exception as exc:
        logger.debug("[Feishu onboard] SDK probe failed: %s", exc)
        return None


def _probe_bot_http(app_id: str, app_secret: str, domain: str) -> Optional[dict]:
    """Fallback probe using raw HTTP (when lark_oapi is not installed)."""
    base_url = _onboard_open_base_url(domain)

    def _get_json(path: str, *, data: Optional[bytes] = None, extra_headers: Optional[Dict[str, str]] = None) -> dict:
        headers = {**(extra_headers or {}), "Content-Type": "application/json"}
        with urlopen(Request(f"{base_url}{path}", data=data, headers=headers), timeout=_ONBOARD_REQUEST_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        token_res = _get_json(
            "/open-apis/auth/v3/tenant_access_token/internal",
            data=json.dumps({"app_id": app_id, "app_secret": app_secret}).encode("utf-8"),
        )
        access_token = token_res.get("tenant_access_token")
        if not access_token:
            return None
        bot_res = _get_json("/open-apis/bot/v3/info", extra_headers={"Authorization": f"Bearer {access_token}"})
        return _parse_bot_response(bot_res)
    except (URLError, OSError, KeyError, json.JSONDecodeError) as exc:
        logger.debug("[Feishu onboard] HTTP probe failed: %s", exc)
        return None


def qr_register(*, initial_domain: str = "feishu", timeout_seconds: int = 600) -> Optional[dict]:
    """Scan-to-create flow → {app_id, app_secret, domain, open_id, bot_name, bot_open_id}.

    None on expected failures (network, denied, timeout); unexpected errors propagate.
    """
    try:
        return _qr_register_inner(initial_domain=initial_domain, timeout_seconds=timeout_seconds)
    except (RuntimeError, URLError, OSError, json.JSONDecodeError) as exc:
        logger.warning("[Feishu onboard] Registration failed: %s", exc)
        return None


def _qr_register_inner(*, initial_domain: str, timeout_seconds: int) -> Optional[dict]:
    """Run init → begin → poll → probe. Raises on network/protocol errors."""
    print("  Connecting to Feishu / Lark...", end="", flush=True)
    _init_registration(initial_domain)
    begin = _begin_registration(initial_domain)
    print(" done.")
    print()
    qr_url = begin["qr_url"]
    if _render_qr(qr_url):
        print(f"\n  Scan the QR code above, or open this URL directly:\n  {qr_url}")
    else:
        print(f"  Open this URL in Feishu / Lark on your phone:\n\n  {qr_url}\n")
        print("  Tip: from the Hermes environment, run: "
              f"{install_hint('messaging')} "
              "to display a scannable QR code here next time")
    print()
    result = _poll_registration(
        device_code=begin["device_code"], interval=begin["interval"],
        expire_in=min(begin["expire_in"], timeout_seconds), domain=initial_domain,
    )
    if not result:
        return None
    bot_info = probe_bot(result["app_id"], result["app_secret"], result["domain"]) or {}  # best-effort
    result["bot_name"] = bot_info.get("bot_name")
    result["bot_open_id"] = bot_info.get("bot_open_id")
    return result


# --- Plugin glue: register(ctx) + the hook fns that replaced the per-platform core touchpoints ---

# ────────────────────────────────────────────────────────────────────────── Plugin migration glue (#41112 /
# #3823) Added when the Feishu adapter (+ its feishu_comment / feishu_comment_rules / feishu_meeting_invite
# satellites) moved from gateway/platforms/ into this bundled plugin. Mirrors the Discord (#24356) / Slack
# migrations: a register(ctx) entry point plus hook implementations that replace the per-platform core
# touchpoints (the Platform.FEISHU elif in gateway/run.py, the feishu_cfg YAML→env block +
# _PLATFORM_CONNECTED_CHECKERS entry in gateway/config.py, the _setup_feishu wizard + _PLATFORMS["feishu"]
# static dict in hermes_cli/gateway.py, and the _send_feishu dispatch in tools/send_message_tool.py).
# ──────────────────────────────────────────────────────────────────────────
_MIGRATION_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_MIGRATION_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_MIGRATION_AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac"}


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """standalone_sender_fn: out-of-process delivery (cron without gateway) via a transient adapter."""
    if not await asyncio.to_thread(_load_lark_oapi):
        return send_error("Feishu dependencies not installed. Run `hermes setup` to install Feishu support.")
    try:
        adapter = FeishuAdapter(pconfig)
        adapter._client = adapter._build_lark_client(_sdk_domain(getattr(adapter, "_domain_name", "feishu")))
        metadata = {"thread_id": thread_id} if thread_id else None
        last_result = None
        if message.strip():
            last_result = await adapter.send(chat_id, message, metadata=metadata)
            if not last_result.success:
                return send_error(f"Feishu send failed: {last_result.error}")
        for media_path, _is_voice in media_files or []:
            if not os.path.exists(media_path):
                return send_error(f"Media file not found: {media_path}")
            ext = os.path.splitext(media_path)[1].lower()
            if ext in _MIGRATION_IMAGE_EXTS:
                sender = adapter.send_image_file
            elif ext in _MIGRATION_VIDEO_EXTS:
                sender = adapter.send_video
            elif ext in _MIGRATION_AUDIO_EXTS:  # voice + non-voice audio both go out as voice
                sender = adapter.send_voice
            else:
                sender = adapter.send_document
            last_result = await sender(chat_id, media_path, metadata=metadata)
            if not last_result.success:
                return send_error(f"Feishu media send failed: {last_result.error}")
        if last_result is None:
            return send_error("No deliverable text or media remained after processing MEDIA tags")
        return {"success": True, "platform": "feishu", "chat_id": chat_id, "message_id": last_result.message_id}
    except Exception as e:
        return send_error(f"Feishu send failed: {e}")


def interactive_setup() -> None:
    """Interactive setup for Feishu / Lark — scan-to-create or manual creds (CLI helpers lazy-imported)."""
    from hermes_cli.config import remove_env_value, save_env_value
    from hermes_cli.setup import prompt_choice
    from hermes_cli.cli_output import prompt, print_header, print_info, print_success, print_warning
    from hermes_cli.setup_platforms import declines_reconfigure

    print_header("Feishu / Lark")
    if declines_reconfigure("Feishu / Lark", "Reconfigure Feishu / Lark?", "FEISHU_APP_ID"):
        return

    method_idx = prompt_choice(
        "How would you like to set up Feishu / Lark?",
        ["Scan QR code to create a new bot automatically (recommended)", "Enter existing App ID and App Secret manually"],
        0,
    )
    credentials = None
    if method_idx == 0:
        try:
            credentials = qr_register()
        except KeyboardInterrupt:
            print_warning("Feishu / Lark setup cancelled.")
            return
        except Exception as exc:
            print_warning(f"QR registration failed: {exc}")
        if not credentials:
            print_info("QR setup did not complete. Continuing with manual input.")
    used_qr = bool(credentials)

    if not credentials:
        print_info("Go to https://open.feishu.cn/ (or https://open.larksuite.com/ for Lark)")
        print_info("Create an app, enable the Bot capability, and copy the credentials.")
        app_id = prompt("App ID", password=False)
        if not app_id:
            print_warning("Skipped — Feishu / Lark won't work without an App ID.")
            return
        app_secret = prompt("App Secret", password=True)
        if not app_secret:
            print_warning("Skipped — Feishu / Lark won't work without an App Secret.")
            return
        domain = "lark" if prompt_choice("Domain", ["feishu (China)", "lark (International)"], 0) == 1 else "feishu"
        bot_name = None
        try:
            bot_info = probe_bot(app_id, app_secret, domain)
            if bot_info:
                bot_name = bot_info.get("bot_name")
                print_success(f"Credentials verified — bot: {bot_name or 'unnamed'}")
            else:
                print_warning("Could not verify bot connection. Credentials saved anyway.")
        except Exception as exc:
            print_warning(f"Credential verification skipped: {exc}")

        credentials = {"app_id": app_id, "app_secret": app_secret, "domain": domain, "open_id": None, "bot_name": bot_name}

    app_id, app_secret = credentials["app_id"], credentials["app_secret"]
    domain = credentials.get("domain", "feishu")
    open_id, bot_name = credentials.get("open_id"), credentials.get("bot_name")
    save_env_value("FEISHU_APP_ID", app_id)
    save_env_value("FEISHU_APP_SECRET", app_secret)
    save_env_value("FEISHU_DOMAIN", domain)

    if used_qr:
        connection_mode = "websocket"
    else:
        mode_idx = prompt_choice(
            "Connection mode",
            ["WebSocket (recommended — no public URL needed)", "Webhook (requires a reachable HTTP endpoint)"],
            0,
        )
        connection_mode = "webhook" if mode_idx == 1 else "websocket"
        if connection_mode == "webhook":
            print_info("Webhook defaults: 127.0.0.1:8765/feishu/webhook")
            print_info("Override with FEISHU_WEBHOOK_HOST / FEISHU_WEBHOOK_PORT / FEISHU_WEBHOOK_PATH")
            print_info("For signature verification, set FEISHU_ENCRYPT_KEY and FEISHU_VERIFICATION_TOKEN")
    save_env_value("FEISHU_CONNECTION_MODE", connection_mode)

    if bot_name:
        print_success(f"Bot created: {bot_name}")

    access_idx = prompt_choice(
        "How should direct messages be authorized?",
        ["Use DM pairing approval (recommended)", "Allow all direct messages", "Only allow listed user IDs"],
        0,
    )
    save_env_value("FEISHU_ALLOW_ALL_USERS", "true" if access_idx == 1 else "false")
    if access_idx == 2:
        allowlist = prompt("Allowed user IDs (comma-separated)", open_id or "", password=False).replace(" ", "")
        save_env_value("FEISHU_ALLOWED_USERS", allowlist)
        print_success("Allowlist saved.")
    else:
        save_env_value("FEISHU_ALLOWED_USERS", "")
        if access_idx == 0:
            print_success("DM pairing enabled.")
            print_info("Unknown users can request access; approve with `hermes pairing approve`.")
        else:
            print_warning("Open DM access enabled for Feishu / Lark.")

    group_idx = prompt_choice(
        "How should group chats be handled?",
        ["Respond only when @mentioned in groups (recommended)", "Disable group chats"], 0,
    )
    save_env_value("FEISHU_GROUP_POLICY", "open" if group_idx == 0 else "disabled")
    print_info("Group chats enabled (bot must be @mentioned)." if group_idx == 0 else "Group chats disabled.")

    print_info("Leave blank to clear a previously saved home channel (cron / notifications).")
    home_channel = prompt("Home chat ID (optional, for cron/notifications)", password=False).strip()
    if home_channel:
        save_env_value("FEISHU_HOME_CHANNEL", home_channel)
        print_success(f"Home channel set to {home_channel}")
    elif remove_env_value("FEISHU_HOME_CHANNEL"):
        print_info("Home channel cleared.")

    print_success("🪽 Feishu / Lark configured!")
    print_info(f"App ID: {app_id}")
    print_info(f"Domain: {domain}")
    if bot_name:
        print_info(f"Bot: {bot_name}")


def _apply_yaml_config(yaml_cfg: dict, feishu_cfg: dict) -> dict | None:
    """``apply_yaml_config_fn`` (#24849): bridge config.yaml feishu.allow_bots to FEISHU_ALLOW_BOTS (env wins) and
    seed ``extra.allow_bots`` so a multiplexed secondary profile's adapter reads its own value."""
    seeded = _apply_yaml_bridge(feishu_cfg, (("allow_bots", "FEISHU_ALLOW_BOTS", "lower"),))
    return {"allow_bots": str(seeded["allow_bots"]).lower()} if seeded else None



def _is_connected(config) -> bool:
    """Feishu counts as connected once app_id is configured."""
    extra = getattr(config, "extra", {}) or {}
    return bool(extra.get("app_id"))



def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="feishu", label="Feishu / Lark", adapter_factory=FeishuAdapter,
        check_fn=feishu_deps_present, ensure_deps_fn=check_feishu_requirements,
        is_connected=_is_connected, validate_config=_is_connected,
        required_env=["FEISHU_APP_ID", "FEISHU_APP_SECRET"],
        install_hint="Run `hermes setup` to install Feishu support.", setup_fn=interactive_setup,
        apply_yaml_config_fn=_apply_yaml_config, allowed_users_env="FEISHU_ALLOWED_USERS",
        allow_all_env="FEISHU_ALLOW_ALL_USERS", cron_deliver_env_var="FEISHU_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send, max_message_length=8000, emoji="🪽",
        allow_update_command=True,
    )
