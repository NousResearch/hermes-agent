# -*- coding: utf-8 -*-
"""QQ Bot API data transfer objects — mirrors botgo/dto.

Provides strongly-typed dataclass models for all QQ Bot API payloads so the
rest of the SDK can use attribute access (``msg.author.user_openid``) instead
of fragile ``dict.get()`` chains.

Reference: https://bot.q.qq.com/wiki/develop/api-v2/
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── OPCode ────────────────────────────────────────────────────────────

class OPCode(enum.IntEnum):
    """WebSocket op codes — mirrors botgo ``dto.OPCode``."""

    DISPATCH = 0
    HEARTBEAT = 1
    IDENTIFY = 2
    RESUME = 6
    RECONNECT = 7
    INVALID_SESSION = 9
    HELLO = 10
    HEARTBEAT_ACK = 11


# ── EventType ─────────────────────────────────────────────────────────

class EventType(str, enum.Enum):
    """Dispatch event type strings — mirrors botgo ``dto.EventType``."""

    READY = "READY"
    RESUMED = "RESUMED"
    C2C_MESSAGE_CREATE = "C2C_MESSAGE_CREATE"
    GROUP_AT_MESSAGE_CREATE = "GROUP_AT_MESSAGE_CREATE"
    DIRECT_MESSAGE_CREATE = "DIRECT_MESSAGE_CREATE"
    GUILD_MESSAGE_CREATE = "GUILD_MESSAGE_CREATE"
    GUILD_AT_MESSAGE_CREATE = "GUILD_AT_MESSAGE_CREATE"
    INTERACTION_CREATE = "INTERACTION_CREATE"


# Event types that carry user-originated messages.
MESSAGE_EVENT_TYPES = frozenset({
    EventType.C2C_MESSAGE_CREATE,
    EventType.GROUP_AT_MESSAGE_CREATE,
    EventType.DIRECT_MESSAGE_CREATE,
    EventType.GUILD_MESSAGE_CREATE,
    EventType.GUILD_AT_MESSAGE_CREATE,
})

# Event types that carry button interaction callbacks.
INTERACTION_EVENT_TYPES = frozenset({
    EventType.INTERACTION_CREATE,
})


# ── Intent ────────────────────────────────────────────────────────────

class Intent(enum.IntFlag):
    """WebSocket intent bitmask — mirrors botgo ``dto.Intent``."""

    GUILDS = 1 << 0
    GUILD_MEMBERS = 1 << 1
    GUILD_MESSAGES = 1 << 9
    DIRECT_MESSAGES = 1 << 12
    INTERACTION = 1 << 26
    GROUP_MESSAGES = 1 << 25
    GUILD_AT_MESSAGE = 1 << 30


DEFAULT_INTENTS = (
    Intent.GROUP_MESSAGES
    | Intent.GUILD_AT_MESSAGE
    | Intent.DIRECT_MESSAGES
    | Intent.INTERACTION
)


# ── QQMessageType ─────────────────────────────────────────────────────

class QQMessageType(enum.IntEnum):
    """Message type codes for the QQ Bot send API."""

    TEXT = 0
    MARKDOWN = 2
    ARK = 3
    EMBED = 4
    INPUT_NOTIFY = 6
    RICH_MEDIA = 7


# ── Inbound DTOs ──────────────────────────────────────────────────────

@dataclass
class User:
    """QQ Bot user — mirrors botgo ``dto.User``."""

    id: str = ""
    username: str = ""
    avatar: str = ""
    bot: bool = False
    union_openid: str = ""
    user_openid: str = ""
    member_openid: str = ""


@dataclass
class Member:
    """Guild member — mirrors botgo ``dto.Member``."""

    guild_id: str = ""
    nick: str = ""
    user: Optional[User] = None
    roles: List[str] = field(default_factory=list)
    joined_at: str = ""


@dataclass
class MessageAttachment:
    """Inbound message attachment — mirrors botgo ``dto.MessageAttachment``."""

    url: str = ""
    filename: str = ""
    content_type: str = ""
    height: int = 0
    width: int = 0
    size: int = 0
    voice_wav_url: str = ""
    asr_refer_text: str = ""

    @property
    def resolved_url(self) -> str:
        """Return URL with protocol, normalising ``//`` prefix."""
        raw = self.url.strip()
        if raw.startswith("//"):
            return f"https:{raw}"
        return raw


@dataclass
class MsgElement:
    """Element from ``msg_elements`` in inbound push data.

    When ``message_type=103`` (引用消息), the platform pushes the referenced
    message content and attachments in ``msg_elements[0]``.
    """

    msg_idx: str = ""
    content: str = ""
    attachments: List[MessageAttachment] = field(default_factory=list)


@dataclass
class MessageScene:
    """``message_scene`` from inbound push data."""

    ext: List[str] = field(default_factory=list)


# QQ 引用（回复）消息类型常量
MSG_TYPE_QUOTE = 103


@dataclass
class Message:
    """Inbound message — mirrors botgo ``dto.Message``."""

    id: str = ""
    channel_id: str = ""
    guild_id: str = ""
    group_id: str = ""
    group_openid: str = ""
    content: str = ""
    timestamp: str = ""
    author: User = field(default_factory=User)
    member: Optional[Member] = None
    attachments: List[MessageAttachment] = field(default_factory=list)
    direct_message: bool = False
    src_guild_id: str = ""
    message_type: int = 0
    message_scene: Optional[MessageScene] = None
    msg_elements: List[MsgElement] = field(default_factory=list)


# ── WebSocket Payload DTOs ────────────────────────────────────────────

@dataclass
class WSPayload:
    """Raw WebSocket frame — mirrors botgo ``dto.WSPayload``."""

    op: int = 0
    s: Optional[int] = None
    t: str = ""
    d: Optional[Dict[str, Any]] = None


@dataclass
class WSHelloData:
    """op 10 Hello payload."""

    heartbeat_interval: int = 30000


@dataclass
class WSReadyData:
    """READY event payload."""

    version: int = 0
    session_id: str = ""
    user: Optional[User] = None
    shard: List[int] = field(default_factory=list)


# ── Close-code classification ─────────────────────────────────────────

class CloseAction(str, enum.Enum):
    """Strategy for handling a WebSocket close code.

    Mirrors the official QQ Bot close code table::

        Code    Meaning                     Resume?  Identify?
        4001    Invalid opcode              No       No
        4002    Invalid payload             No       No
        4006    Invalid session id          No       Yes
        4007    Seq error                   No       Yes
        4008    Send payload too fast        Yes      Yes
        4009    Connection expired           Yes      Yes
        4010    Invalid shard               No       No
        4011    Too many guilds             No       No
        4012    Invalid version             No       No
        4013    Invalid intent              No       No
        4014    Intent not authorized       No       No
        4900-4913  Internal error           No       Yes
        4914    Bot offline (sandbox only)  No       No
        4915    Bot banned                  No       No
    """

    STOP = "stop"
    """Fatal — do not reconnect at all."""

    RESUME_OK = "resume_ok"
    """Can resume (4008, 4009): sleep then reconnect, try Resume first."""

    IDENTIFY_ONLY = "identify_only"
    """Cannot resume but can re-Identify (4006, 4007, 4900-4913):
    clear session, reconnect with fresh Identify."""
    
    RATE_LIMIT = "rate_limit"
    """Rate limited (4008): sleep extra then reconnect."""

    RECONNECT = "reconnect"
    """Unknown code — try resume, fall back to identify."""


# Codes that allow Resume (session_id stays valid).
_RESUME_OK_CODES = frozenset({4008, 4009})

# Codes that require a fresh Identify (session is invalid).
_IDENTIFY_ONLY_CODES = frozenset(
    {4006, 4007}
    | set(range(4900, 4914))
)

# Fatal close codes (stop reconnecting entirely).
_FATAL_CODES = frozenset({4001, 4002, 4010, 4011, 4012, 4013, 4014, 4914, 4915})


def classify_close_code(code: Optional[int]) -> CloseAction:
    """Map a QQ WebSocket close code to a reconnect strategy.

    :param code: WebSocket close code, or ``None`` for unknown.
    :returns: :class:`CloseAction` indicating what to do next.
    """
    if code is None:
        return CloseAction.RECONNECT
    if code in _FATAL_CODES:
        return CloseAction.STOP
    if code == 4008:
        return CloseAction.RATE_LIMIT
    if code in _RESUME_OK_CODES:
        return CloseAction.RESUME_OK
    if code in _IDENTIFY_ONLY_CODES:
        return CloseAction.IDENTIFY_ONLY
    return CloseAction.RECONNECT


# ── Outbound message DTOs ─────────────────────────────────────────────

@dataclass
class MarkdownContent:
    """Markdown body for msg_type=2."""

    content: str = ""


@dataclass
class MediaInfo:
    """Rich media file reference (from upload response)."""

    file_info: str = ""


@dataclass
class MessageReference:
    """Reply reference — mirrors botgo ``dto.MessageReference``."""

    message_id: str = ""
    ignore_get_message_error: bool = False


@dataclass
class InputNotify:
    """Typing indicator body — mirrors botgo ``dto.InputNotify``."""

    input_type: int = 1
    input_second: int = 60


@dataclass
class MessageToCreate:
    """Outbound message body — mirrors botgo ``dto.MessageToCreate``.

    Covers text, markdown, media, and input_notify message types.
    """

    content: str = ""
    msg_type: int = 0
    msg_id: str = ""
    msg_seq: int = 0
    markdown: Optional[MarkdownContent] = None
    media: Optional[MediaInfo] = None
    message_reference: Optional[MessageReference] = None
    input_notify: Optional[InputNotify] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a QQ API request body, omitting empty/None fields."""
        d: Dict[str, Any] = {"msg_type": self.msg_type}
        if self.msg_seq:
            d["msg_seq"] = self.msg_seq
        if self.msg_id:
            d["msg_id"] = self.msg_id
        if self.content:
            d["content"] = self.content
        if self.markdown is not None:
            d["markdown"] = {"content": self.markdown.content}
        if self.media is not None:
            d["media"] = {"file_info": self.media.file_info}
        if self.message_reference is not None:
            d["message_reference"] = {
                "message_id": self.message_reference.message_id,
            }
        if self.input_notify is not None:
            d["input_notify"] = {
                "input_type": self.input_notify.input_type,
                "input_second": self.input_notify.input_second,
            }
        return d


@dataclass
class RichMediaMessage:
    """Upload payload for rich media — mirrors botgo ``dto.RichMediaMessage``."""

    file_type: int = 0
    url: str = ""
    srv_send_msg: bool = False
    file_data: str = ""
    file_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a QQ API upload body, omitting empty fields."""
        d: Dict[str, Any] = {
            "file_type": self.file_type,
            "srv_send_msg": self.srv_send_msg,
        }
        if self.url:
            d["url"] = self.url
        if self.file_data:
            d["file_data"] = self.file_data
        if self.file_name:
            d["file_name"] = self.file_name
        return d


@dataclass
class GuildMessageToCreate:
    """Outbound guild/channel message body (simpler than C2C/group)."""

    content: str = ""
    msg_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a QQ API request body."""
        d: Dict[str, Any] = {"content": self.content}
        if self.msg_id:
            d["msg_id"] = self.msg_id
        return d


# ── Parsing factories ─────────────────────────────────────────────────

def _parse_user(raw: Any) -> User:
    """Parse a :class:`User` from a raw dict."""
    if not isinstance(raw, dict):
        return User()
    return User(
        id=str(raw.get("id", "")),
        username=str(raw.get("username", "")),
        avatar=str(raw.get("avatar", "")),
        bot=bool(raw.get("bot", False)),
        union_openid=str(raw.get("union_openid", "")),
        user_openid=str(raw.get("user_openid", "")),
        member_openid=str(raw.get("member_openid", "")),
    )


def _parse_member(raw: Any) -> Optional[Member]:
    """Parse a :class:`Member` from a raw dict."""
    if not isinstance(raw, dict):
        return None
    return Member(
        guild_id=str(raw.get("guild_id", "")),
        nick=str(raw.get("nick", "")),
        user=_parse_user(raw.get("user")),
        roles=list(raw.get("roles", [])),
        joined_at=str(raw.get("joined_at", "")),
    )


def _parse_attachment(raw: Any) -> Optional[MessageAttachment]:
    """Parse a :class:`MessageAttachment` from a raw dict."""
    if not isinstance(raw, dict):
        return None
    return MessageAttachment(
        url=str(raw.get("url", "")),
        filename=str(raw.get("filename", "")),
        content_type=str(raw.get("content_type", "")),
        height=int(raw.get("height", 0) or 0),
        width=int(raw.get("width", 0) or 0),
        size=int(raw.get("size", 0) or 0),
        voice_wav_url=str(raw.get("voice_wav_url", "")),
        asr_refer_text=str(raw.get("asr_refer_text", "")),
    )


def _parse_msg_elements(raw_list: Any) -> List[MsgElement]:
    """Parse ``msg_elements`` from a raw list."""
    elements: List[MsgElement] = []
    for elem_raw in raw_list or []:
        if not isinstance(elem_raw, dict):
            continue
        elem_attachments: List[MessageAttachment] = []
        for att_raw in elem_raw.get("attachments") or []:
            att = _parse_attachment(att_raw)
            if att is not None:
                elem_attachments.append(att)
        elements.append(MsgElement(
            msg_idx=str(elem_raw.get("msg_idx", "")),
            content=str(elem_raw.get("content", "")),
            attachments=elem_attachments,
        ))
    return elements


def parse_message(raw: Dict[str, Any]) -> Message:
    """Parse a raw JSON dict into a strongly-typed :class:`Message`.

    This is the **single entry point** for converting QQ API JSON to DTO.
    All ``dict.get()`` access is concentrated here.

    :param raw: Raw event dict from the WebSocket dispatch payload.
    :returns: Parsed :class:`Message` instance.
    """
    attachments: List[MessageAttachment] = []
    for att_raw in raw.get("attachments") or []:
        att = _parse_attachment(att_raw)
        if att is not None:
            attachments.append(att)

    scene = None
    scene_raw = raw.get("message_scene")
    if isinstance(scene_raw, dict):
        ext = scene_raw.get("ext")
        if isinstance(ext, list):
            scene = MessageScene(ext=[str(e) for e in ext if isinstance(e, str)])

    return Message(
        id=str(raw.get("id", "")),
        channel_id=str(raw.get("channel_id", "")),
        guild_id=str(raw.get("guild_id", "")),
        group_id=str(raw.get("group_id", "")),
        group_openid=str(raw.get("group_openid", "")),
        content=str(raw.get("content", "")),
        timestamp=str(raw.get("timestamp", "")),
        author=_parse_user(raw.get("author")),
        member=_parse_member(raw.get("member")),
        attachments=attachments,
        direct_message=bool(raw.get("direct_message", False)),
        src_guild_id=str(raw.get("src_guild_id", "")),
        message_type=int(raw.get("message_type", 0) or 0),
        message_scene=scene,
        msg_elements=_parse_msg_elements(raw.get("msg_elements")),
    )


def parse_ws_payload(raw: Dict[str, Any]) -> WSPayload:
    """Parse a raw WebSocket JSON frame into :class:`WSPayload`."""
    t_val = raw.get("t", "")
    # Store the raw string, not an enum repr.
    if hasattr(t_val, "value"):
        t_val = t_val.value
    return WSPayload(
        op=int(raw.get("op", 0)),
        s=raw.get("s"),
        t=str(t_val) if t_val else "",
        d=raw.get("d"),
    )


def parse_hello(raw: Any) -> WSHelloData:
    """Parse the ``d`` field of an op 10 Hello."""
    if not isinstance(raw, dict):
        return WSHelloData()
    return WSHelloData(
        heartbeat_interval=int(raw.get("heartbeat_interval", 30000)),
    )


def parse_ready(raw: Any) -> WSReadyData:
    """Parse the ``d`` field of a READY dispatch."""
    if not isinstance(raw, dict):
        return WSReadyData()
    return WSReadyData(
        version=int(raw.get("version", 0)),
        session_id=str(raw.get("session_id", "")),
        user=_parse_user(raw.get("user")),
        shard=list(raw.get("shard", [])),
    )


# ── Chunked upload DTOs ───────────────────────────────────────────────
#
# Mirrors the proto definitions in:
#   trpc.group_openapi.group_relation / qqntv2/richmedia/richmedia.proto
#
# Request DTOs  (Req suffix)  — serialised to JSON and sent to the API.
# Response DTOs (Response suffix) — parsed from the API JSON response.

# Default upload concurrency when server does not return ``upload_config``.
_DEFAULT_UPLOAD_CONCURRENCY = 1


# ── Request DTOs ──────────────────────────────────────────────────────

@dataclass
class UploadPrepareRequest:
    """Request body for ``POST /v2/{type}/{id}/upload_prepare``.

    Mirrors proto ``UploadPrepareReq``.
    """

    file_type: int
    """Media type: 1=image, 2=video, 3=voice, 4=file."""

    file_size: int
    """Total file size in bytes (proto: uint64)."""

    file_name: str
    """Original filename including extension."""

    md5: str
    """Full-file MD5 hex string."""

    sha1: str
    """Full-file SHA1 hex string."""

    md5_10m: str
    """MD5 of the first 10,002,432 bytes (equals full MD5 for smaller files)."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to API request body."""
        return {
            "file_type": self.file_type,
            "file_size": self.file_size,
            "file_name": self.file_name,
            "md5": self.md5,
            "sha1": self.sha1,
            "md5_10m": self.md5_10m,
        }


@dataclass
class UploadPartFinishRequest:
    """Request body for ``POST /v2/{type}/{id}/upload_part_finish``.

    Mirrors proto ``UploadPartFinishReq``.
    """

    upload_id: str
    """Upload task ID from ``upload_prepare``."""

    part_index: int
    """1-based part index (proto: uint32)."""

    block_size: int
    """Actual byte count of this part (proto: uint64)."""

    md5: str
    """MD5 hex of this part's data."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to API request body."""
        return {
            "upload_id": self.upload_id,
            "part_index": self.part_index,
            "block_size": self.block_size,
            "md5": self.md5,
        }


@dataclass
class CompleteUploadRequest:
    """Request body for completing a chunked upload.

    Sent to ``POST /v2/users/{id}/files`` (C2C) or
    ``POST /v2/groups/{id}/files`` (Group) with ``upload_id`` to signal
    the chunked completion path (same endpoint as simple upload).
    """

    upload_id: str
    """Upload task ID from ``upload_prepare``."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to API request body."""
        return {"upload_id": self.upload_id}


# ── Response DTOs ─────────────────────────────────────────────────────

@dataclass
class UploadConfig:
    """Server-controlled upload configuration.

    Mirrors proto ``UploadConfig`` (nested inside ``UploadPrepareRsp``).
    """

    concurrency: int = _DEFAULT_UPLOAD_CONCURRENCY
    """Maximum parts to upload in parallel."""

    retry_timeout: int = 0
    """Seconds to retry upload_part_finish on 40093001 (0 = use client default)."""

    retry_delay: int = 0
    """Retry delay in seconds (0 = use client default)."""


@dataclass
class UploadPart:
    """A single presigned part returned by ``upload_prepare``.

    Mirrors proto ``UploadPart``.
    """

    index: int
    """1-based part index (proto: uint32)."""

    presigned_url: str
    """COS presigned URL for PUT upload."""

    block_size: int = 0
    """Part-specific block size in bytes (proto: uint64); 0 = use Rsp.block_size."""


@dataclass
class UploadPrepareResponse:
    """Response from ``POST /v2/{type}/{id}/upload_prepare``.

    Mirrors proto ``UploadPrepareRsp``.
    """

    upload_id: str
    """Opaque upload task ID used in subsequent calls."""

    block_size: int
    """Default block size in bytes for all parts (proto: uint64)."""

    parts: List[UploadPart]
    """Presigned part list."""

    upload_config: UploadConfig = None  # type: ignore[assignment]
    """Server-controlled upload configuration (concurrency, retry settings)."""

    def __post_init__(self) -> None:
        if self.upload_config is None:
            self.upload_config = UploadConfig()

    @property
    def concurrency(self) -> int:
        """Shortcut for ``upload_config.concurrency``."""
        return self.upload_config.concurrency

    @property
    def retry_timeout(self) -> float:
        """Shortcut for ``upload_config.retry_timeout`` as float seconds."""
        return float(self.upload_config.retry_timeout)


@dataclass
class CompleteUploadResponse:
    """Response from ``POST /v2/{type}/{id}/upload_complete``.

    Mirrors proto ``MediaUploadRsp`` (the complete endpoint reuses this shape).
    """

    file_info: str = ""
    """Opaque bytes token used in ``MessageToCreate.media.file_info``."""

    file_uuid: str = ""
    """File UUID (alternative token returned by some API versions)."""

    ttl: int = 0
    """Time-to-live in seconds (0 = not returned)."""

    @property
    def token(self) -> str:
        """Return ``file_info`` if present, otherwise ``file_uuid``."""
        return self.file_info or self.file_uuid


# ── Parsing factories ──────────────────────────────────────────────────

def parse_upload_prepare(raw: Dict[str, Any]) -> UploadPrepareResponse:
    """Parse a raw ``upload_prepare`` response dict.

    Reads ``upload_config`` from the nested ``UploadConfig`` sub-object
    (mirrors proto ``UploadPrepareRsp.upload_config``).

    :param raw: API response dict.
    :returns: Typed :class:`UploadPrepareResponse`.
    :raises ValueError: If required fields are missing.
    """
    upload_id = str(raw.get("upload_id", ""))
    if not upload_id:
        raise ValueError(f"upload_prepare response missing upload_id: {raw}")

    block_size = int(raw.get("block_size", 0))
    if not block_size:
        raise ValueError(f"upload_prepare response missing block_size: {raw}")

    parts: List[UploadPart] = []
    for p in raw.get("parts") or []:
        if not isinstance(p, dict):
            continue
        parts.append(UploadPart(
            index=int(p.get("index", 0)),
            presigned_url=str(p.get("presigned_url", "")),
            block_size=int(p.get("block_size", 0)),
        ))
    if not parts:
        raise ValueError(f"upload_prepare response has no parts: {raw}")

    # upload_config is a nested object in the proto; fall back gracefully
    # if the HTTP layer flattens it (some older gateway versions do).
    cfg_raw = raw.get("upload_config")
    if isinstance(cfg_raw, dict):
        upload_config = UploadConfig(
            concurrency=int(cfg_raw.get("concurrency") or _DEFAULT_UPLOAD_CONCURRENCY),
            retry_timeout=int(cfg_raw.get("retry_timeout") or 0),
            retry_delay=int(cfg_raw.get("retry_delay") or 0),
        )
    else:
        # Fallback: some gateway versions flatten concurrency/retry_timeout to Rsp root.
        upload_config = UploadConfig(
            concurrency=int(raw.get("concurrency") or _DEFAULT_UPLOAD_CONCURRENCY),
            retry_timeout=int(raw.get("retry_timeout") or 0),
            retry_delay=int(raw.get("retry_delay") or 0),
        )

    return UploadPrepareResponse(
        upload_id=upload_id,
        block_size=block_size,
        parts=parts,
        upload_config=upload_config,
    )


def parse_complete_upload(raw: Dict[str, Any]) -> CompleteUploadResponse:
    """Parse a raw ``complete_upload`` response dict."""
    return CompleteUploadResponse(
        file_info=str(raw.get("file_info", "")),
        file_uuid=str(raw.get("file_uuid", "")),
        ttl=int(raw.get("ttl", 0) or 0),
    )


# ── Inline Keyboard DTOs ──────────────────────────────────────────────
#
# Used for sending approval messages with interactive buttons.
# Mirrors the QQ Bot API inline keyboard spec and TypeScript types.ts.

@dataclass
class KeyboardButtonPermission:
    """Button permission — who can click.

    type=2 → all users.
    """

    type: int = 2


@dataclass
class KeyboardButtonAction:
    """Button action definition.

    type=1 (Callback) → clicking triggers INTERACTION_CREATE with
    ``data.resolved.button_data = data``.
    """

    type: int
    """Action type: 1=Callback, 2=Link, etc."""

    data: str
    """Payload delivered in INTERACTION_CREATE (for type=1)."""

    permission: KeyboardButtonPermission = None  # type: ignore[assignment]
    click_limit: int = 1
    """Maximum clicks per user (1 = single-use)."""

    def __post_init__(self) -> None:
        if self.permission is None:
            self.permission = KeyboardButtonPermission()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "data": self.data,
            "permission": {"type": self.permission.type},
            "click_limit": self.click_limit,
        }


@dataclass
class KeyboardButtonRenderData:
    """Visual rendering data for a keyboard button."""

    label: str
    """Label shown before the user clicks."""

    visited_label: str
    """Label shown after the user clicks."""

    style: int = 1
    """Button style: 0=grey, 1=blue."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "visited_label": self.visited_label,
            "style": self.style,
        }


@dataclass
class KeyboardButton:
    """A single keyboard button.

    group_id: buttons with the same group_id are mutually exclusive
    (clicking one greys out the rest).
    """

    id: str
    render_data: KeyboardButtonRenderData
    action: KeyboardButtonAction
    group_id: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "render_data": self.render_data.to_dict(),
            "action": self.action.to_dict(),
            "group_id": self.group_id,
        }


@dataclass
class KeyboardRow:
    """A row of keyboard buttons."""

    buttons: List[KeyboardButton] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"buttons": [b.to_dict() for b in self.buttons]}


@dataclass
class KeyboardContent:
    """Keyboard content — list of rows."""

    rows: List[KeyboardRow] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"rows": [r.to_dict() for r in self.rows]}


@dataclass
class InlineKeyboard:
    """Top-level inline keyboard payload sent in ``MessageToCreate.keyboard``."""

    content: KeyboardContent = field(default_factory=KeyboardContent)

    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content.to_dict()}


# ── InteractionEvent DTO ──────────────────────────────────────────────

@dataclass
class InteractionResolved:
    """``data.resolved`` from an INTERACTION_CREATE event."""

    button_data: str = ""
    """Payload set in ``KeyboardButtonAction.data`` (approval: ``approve:<id>:<decision>``)."""

    button_id: str = ""
    """Button element id."""

    user_id: str = ""
    """Operator user id (guild only)."""


@dataclass
class InteractionData:
    """``data`` field of an INTERACTION_CREATE event."""

    type: int = 0
    """Interaction type: 11=message button, 2001=config query, 2002=config update."""

    resolved: InteractionResolved = field(default_factory=InteractionResolved)


@dataclass
class InteractionEvent:
    """INTERACTION_CREATE dispatch event payload.

    Sent when a user clicks an inline keyboard button.
    Mirrors TypeScript ``InteractionEvent`` in types.ts.
    """

    id: str = ""
    """Interaction event ID — required for ``PUT /interactions/{id}`` ACK."""

    type: int = 0
    """Event type code (11=message button)."""

    chat_type: int = 0
    """0=guild, 1=group, 2=c2c."""

    scene: str = ""
    """Human-readable scene: 'guild' | 'group' | 'c2c'."""

    group_openid: str = ""
    """Group openid (group scene)."""

    group_member_openid: str = ""
    """Operator member openid (group scene)."""

    user_openid: str = ""
    """Operator user openid (c2c scene)."""

    channel_id: str = ""
    """Channel id (guild scene)."""

    guild_id: str = ""
    """Guild id (guild scene)."""

    data: InteractionData = field(default_factory=InteractionData)

    @property
    def operator_openid(self) -> str:
        """Return the best available operator openid."""
        return (
            self.group_member_openid
            or self.user_openid
            or self.data.resolved.user_id
            or ""
        )

    @property
    def chat_id(self) -> str:
        """Return the chat id for routing (group_openid or user_openid)."""
        return self.group_openid or self.user_openid or self.channel_id or ""

    @property
    def is_c2c(self) -> bool:
        return bool(self.user_openid and not self.group_openid)

    @property
    def is_group(self) -> bool:
        return bool(self.group_openid)


def parse_interaction_event(raw: Dict[str, Any]) -> InteractionEvent:
    """Parse a raw INTERACTION_CREATE dispatch payload."""
    data_raw = raw.get("data") or {}
    resolved_raw = data_raw.get("resolved") or {}
    resolved = InteractionResolved(
        button_data=str(resolved_raw.get("button_data", "")),
        button_id=str(resolved_raw.get("button_id", "")),
        user_id=str(resolved_raw.get("user_id", "")),
    )
    data = InteractionData(
        type=int(data_raw.get("type", 0)),
        resolved=resolved,
    )
    return InteractionEvent(
        id=str(raw.get("id", "")),
        type=int(raw.get("type", 0)),
        chat_type=int(raw.get("chat_type", 0)),
        scene=str(raw.get("scene", "")),
        group_openid=str(raw.get("group_openid", "")),
        group_member_openid=str(raw.get("group_member_openid", "")),
        user_openid=str(raw.get("user_openid", "")),
        channel_id=str(raw.get("channel_id", "")),
        guild_id=str(raw.get("guild_id", "")),
        data=data,
    )



