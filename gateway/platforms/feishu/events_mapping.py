"""SDK InboundMessage → Hermes MessageEvent conversion + synthetic events.

Public API:

    async def to_message_event(msg, *, channel) -> MessageEvent: ...
    async def _to_text_event_from_reaction(evt, *, channel, ...) -> Optional[MessageEvent]: ...
    def _to_command_event_from_card_action(action, *, channel) -> MessageEvent: ...

Module-internal helpers:
    - _strip_edge_self_mentions
    - _build_mention_hint
    - _sdk_content_to_message_type
    - self_get_chat_info_safe
    - _resolve_source_chat_type_for_event
    - _sdk_comment_to_legacy_dict

This module **does not import** FeishuAdapter at module-load time. The two
staticmethods it needs (``_map_chat_type`` / ``_resolve_source_chat_type``)
are accessed via a lazy local import inside the call sites so the
sub-module dependency direction stays strictly ``adapter → events_mapping``.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.feishu.types import (
    FeishuMentionRef,
    map_chat_type,
    resolve_source_chat_type,
)
from gateway.session import SessionSource
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


# Module-internal text helpers (mention boundary + trailing terminal punctuation).
_MENTION_BOUNDARY_CHARS = frozenset(" \t\n\r.,;:!?、，。；：！？()[]{}<>\"'`")
_TRAILING_TERMINAL_PUNCT = frozenset(" \t\n\r.!?。！？")
def _build_mention_hint(mentions: Sequence["FeishuMentionRef"]) -> str:
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


def _strip_edge_self_mentions(
    text: str,
    mentions: Sequence["FeishuMentionRef"],
) -> str:
    # Leading: strip consecutive self-mentions unconditionally.
    # Trailing: strip only when followed by whitespace/terminal punct, so
    # mid-sentence references ("don't @Bot again") stay intact.
    # Leading word-boundary prevents @Al from eating @Alice.
    if not text:
        return text
    self_names = [
        f"@{ref.name or ref.open_id or 'user'}"
        for ref in mentions
        if ref.is_self
    ]
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


def _sdk_content_to_message_type(content: Any) -> MessageType:
    """Map SDK MessageContent.kind → Hermes MessageType.

    SDK exposes a 19-variant discriminated union; Hermes' MessageType has
    only 9 values, so several SDK kinds collapse onto TEXT/DOCUMENT.
    """
    kind = getattr(content, "kind", "unknown")
    return {
        "text":                  MessageType.TEXT,
        "post":                  MessageType.TEXT,        # post → flat markdown text
        "image":                 MessageType.PHOTO,
        "file":                  MessageType.DOCUMENT,
        "audio":                 MessageType.AUDIO,
        "media":                 MessageType.VIDEO,       # SDK names video msgs "media"
        "sticker":               MessageType.STICKER,
        "interactive":           MessageType.TEXT,
        "share_chat":            MessageType.TEXT,
        "share_user":            MessageType.TEXT,
        "system":                MessageType.TEXT,
        "location":              MessageType.LOCATION,
        "folder":                MessageType.DOCUMENT,
        "hongbao":               MessageType.TEXT,        # SDK uses "hongbao" wire-name
        "general_calendar":      MessageType.TEXT,
        "share_calendar_event":  MessageType.TEXT,
        "video_chat":            MessageType.TEXT,
        "calendar":              MessageType.TEXT,
        "vote":                  MessageType.TEXT,
        "todo":                  MessageType.TEXT,
        "merge_forward":         MessageType.TEXT,
        "unknown":               MessageType.TEXT,
    }.get(kind, MessageType.TEXT)


def _resource_media_type(resource: Any, path: Any) -> str:
    """Return a gateway-facing MIME type for an SDK resource descriptor."""
    file_name = str(getattr(resource, "file_name", "") or "")
    guessed, _ = mimetypes.guess_type(file_name or str(path))
    if guessed:
        return guessed
    kind = str(getattr(resource, "type", "") or "").lower()
    if kind == "image":
        return "image/unknown"
    if kind == "audio":
        return "audio/unknown"
    if kind == "video":
        return "video/unknown"
    return "application/octet-stream"


def _download_file_name_for_resource(resource: Any, content: Any = None) -> Optional[str]:
    """Return a stable download name for resources whose SDK descriptor lacks one."""
    file_name = str(getattr(resource, "file_name", "") or "").strip()
    if file_name:
        return file_name
    kind = str(getattr(resource, "type", "") or "").lower()
    if kind == "file" and content is not None:
        content_file_key = str(getattr(content, "file_key", "") or "").strip()
        resource_file_key = str(getattr(resource, "file_key", "") or "").strip()
        if content_file_key and resource_file_key == content_file_key:
            content_file_name = str(getattr(content, "file_name", "") or "").strip()
            if content_file_name:
                return content_file_name
    if kind != "audio":
        return None
    file_key = str(getattr(resource, "file_key", "") or "").strip()
    if not file_key:
        return None
    safe_key = file_key.replace("/", "_").replace("\\", "_")
    return f"{safe_key}.ogg"


async def self_get_chat_info_safe(
    channel: "FeishuChannel", chat_id: str
) -> Dict[str, Any]:
    """Defensive wrapper around ``channel.get_chat_info``.

    Returns a dict shaped for ``resolve_source_chat_type``
    (key ``"type"`` in {"dm","group","forum"}), with a safe fallback when
    the SDK returns ``None`` or raises.
    """
    fallback = {"chat_id": chat_id, "name": chat_id, "type": "dm"}
    try:
        info = await channel.get_chat_info(chat_id)
        if info is None:
            return fallback
        raw_chat_type = str(getattr(info, "chat_type", "") or "").strip().lower()
        return {
            "chat_id": getattr(info, "chat_id", None) or chat_id,
            "name": str(getattr(info, "name", None) or chat_id),
            "type": map_chat_type(raw_chat_type),
            "raw_type": raw_chat_type or None,
        }
    except Exception:
        return fallback


def _resolve_source_chat_type_for_event(
    *, chat_info: Dict[str, Any], event_chat_type: str
) -> str:
    """Module-level shim around ``resolve_source_chat_type``.

    Lets module-level helpers reuse the helper without needing an
    instance. The module-level function stays the source of truth.
    """
    return resolve_source_chat_type(
        chat_info=chat_info, event_chat_type=event_chat_type
    )


async def to_message_event(
    msg: "InboundMessage",
    *,
    channel: "FeishuChannel",
) -> MessageEvent:
    """SDK InboundMessage → Hermes MessageEvent.

    SDK has already done:
        - dedup / stale / policy / mention / lock / batch / queue filtering
        - merge_forward expansion (per InboundConfig)
        - identity resolution (Identity.display_name + Mention resolved)
        - content_text flat-render (markdown text + xml-like media placeholders)
        - resources list of ResourceDescriptor (file_key + type)

    Hermes layers on top:
        1) message_type from content.kind → Hermes MessageType
        2) media_urls via channel.download_resource_to_file → cached path
        3) reply_to_text from msg.reply (prefer pre-populated text, else fetch)
        4) mention hint injection via _build_mention_hint
        5) leading self-mention strip via _strip_edge_self_mentions
        6) source resolution via resolve_source_chat_type
    """
    text = msg.content_text or ""

    # 1) message_type mapping
    inbound_type = _sdk_content_to_message_type(msg.content)

    # 2) Hermes mentions reconstruction
    bot_open_id = ""
    bot_user_id = ""
    bot_name = ""
    bot_identity = getattr(channel, "bot_identity", None)
    if bot_identity is not None:
        bot_open_id = str(getattr(bot_identity, "open_id", "") or "")
        bot_user_id = str(getattr(bot_identity, "user_id", "") or "")
        bot_name = (
            str(getattr(bot_identity, "name", "") or "")
            or str(getattr(bot_identity, "display_name", "") or "")
        )

    mentions: List[FeishuMentionRef] = []
    for m in (msg.mentions or []):
        m_open_id = str(getattr(m, "open_id", "") or "")
        m_user_id = str(getattr(m, "user_id", "") or "")
        m_name = str(getattr(m, "name", "") or "")
        mention_has_id = bool(m_open_id or m_user_id)
        is_self = bool(
            (bot_open_id and m_open_id == bot_open_id)
            or (bot_user_id and m_user_id == bot_user_id)
            or (not mention_has_id and bot_name and m_name == bot_name)
        )
        mentions.append(
            FeishuMentionRef(
                name=getattr(m, "name", "") or "",
                open_id=m_open_id,
                is_self=is_self,
                is_all=False,
            )
        )
    if getattr(msg, "mentioned_all", False):
        mentions.append(FeishuMentionRef(name="@all", open_id="", is_all=True))

    # 3) Leading self-mention strip + command detection
    if inbound_type == MessageType.TEXT:
        text = _strip_edge_self_mentions(text, mentions)
        if text.startswith("/"):
            inbound_type = MessageType.COMMAND

    # 4) Mention hint injection (skip for COMMAND so /cmd parsing stays clean)
    if inbound_type != MessageType.COMMAND:
        hint = _build_mention_hint(mentions)
        if hint:
            text = f"{hint}\n\n{text}" if text else hint

    # 5) media_urls / media_types: download each SDK resource to local cache.
    media_urls: List[str] = []
    media_types: List[str] = []
    cache_dir = get_hermes_home() / "feishu_media_cache"
    resource_sources: List[Any] = list(getattr(msg, "batched_sources", None) or [])
    if not resource_sources:
        resource_sources = [msg]
    elif list(getattr(msg, "resources", None) or []) and all(
        id(source) != id(msg) for source in resource_sources
    ):
        resource_sources.append(msg)

    for source_msg in resource_sources:
        source_message_id = str(getattr(source_msg, "id", None) or msg.id)
        for res in list(getattr(source_msg, "resources", None) or []):
            try:
                download_file_name = _download_file_name_for_resource(
                    res, getattr(source_msg, "content", None)
                )
                try:
                    path = await channel.download_resource_to_file(
                        res.file_key,
                        resource_type=res.type,
                        message_id=source_message_id,
                        dest_dir=cache_dir,
                        file_name=download_file_name,
                    )
                except Exception:
                    if res.type not in {"audio", "video"}:
                        raise
                    # Feishu's message-resource endpoint serves native voice
                    # and media file_keys through type=file in some tenants.
                    path = await channel.download_resource_to_file(
                        res.file_key,
                        resource_type="file",
                        message_id=source_message_id,
                        dest_dir=cache_dir,
                        file_name=download_file_name,
                    )
                media_urls.append(str(path))
                media_types.append(_resource_media_type(res, path))
                logger.info(
                    "[Feishu] cached inbound resource: message_id=%s type=%s path=%s",
                    source_message_id,
                    res.type,
                    path,
                )
            except Exception as e:
                logger.warning(
                    "[Feishu] download_resource_to_file failed file_key=%s type=%s: %s",
                    res.file_key, res.type, e,
                )
                # single-resource failure must not block whole-message dispatch
    if media_urls and inbound_type == MessageType.TEXT:
        if any(media_type.startswith("image/") for media_type in media_types):
            inbound_type = MessageType.PHOTO
        elif any(
            media_type.startswith(("application/", "text/", "audio/", "video/"))
            for media_type in media_types
        ):
            inbound_type = MessageType.DOCUMENT

    # 6) reply_to_text — prefer SDK-pre-populated, fall back to fetch_message
    reply_to_message_id = msg.reply.message_id if msg.reply else None
    reply_to_text: Optional[str] = None
    if msg.reply is not None:
        pre = getattr(msg.reply, "text", None)
        if pre:
            reply_to_text = str(pre)
        elif reply_to_message_id:
            try:
                wire = await channel.fetch_message(reply_to_message_id)
                items = (wire or {}).get("data", {}).get("items") or []
                if items:
                    body = items[0].get("body") or {}
                    msg_type = items[0].get("msg_type") or ""
                    raw_content = body.get("content") or ""
                    if msg_type == "text":
                        try:
                            reply_to_text = json.loads(raw_content).get("text", "")
                        except Exception:
                            reply_to_text = raw_content
                    else:
                        # post / other — richer extraction left for later
                        reply_to_text = ""
            except Exception as e:
                logger.warning(
                    "[Feishu] fetch reply_to_message %s failed: %s",
                    reply_to_message_id, e,
                )

    # 7) source resolution
    chat_id = msg.conversation.chat_id
    chat_info_dict = await self_get_chat_info_safe(channel, chat_id)
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id=str(chat_id),
        chat_name=chat_info_dict.get("name") or chat_id or "Feishu Chat",
        chat_type=_resolve_source_chat_type_for_event(
            chat_info=chat_info_dict,
            event_chat_type=msg.conversation.chat_type,
        ),
        user_id=msg.sender.user_id or msg.sender.open_id,
        user_name=msg.sender.display_name or "",
        # SDK exposes only conversation.thread_id (set on topic-aware
        # chats). When absent, surface root_id / upper_message_id from
        # raw so in-thread replies preserve the topic anchor — without
        # this, outbound replies on threaded chats would create new
        # topics.
        thread_id=(
            msg.conversation.thread_id
            or (msg.raw or {}).get("root_id")
            or (msg.raw or {}).get("upper_message_id")
            or None
        ),
        user_id_alt=msg.sender.union_id,
        is_bot=bool(getattr(msg.sender, "is_bot", False)),
    )

    return MessageEvent(
        text=text,
        message_type=inbound_type,
        source=source,
        raw_message=msg.raw,
        message_id=msg.id,
        media_urls=media_urls,
        media_types=media_types,
        reply_to_message_id=reply_to_message_id,
        reply_to_text=reply_to_text,
        timestamp=datetime.now(),
    )


def _to_command_event_from_card_action(
    action: "CardActionEvent",
    *,
    channel: "FeishuChannel",
) -> MessageEvent:
    """SDK CardActionEvent (no hermes_action) → Hermes MessageEvent.COMMAND.

    Synthesis shape:
      text = "/card {tag} {json.dumps(value)}"
      message_type = COMMAND
      source.user_id = action.operator.open_id
      message_id = action.message_id (carries the original card msg id; SDK
                  push_action dedup already de-duped redeliveries before us)
    """
    payload = action.action if action.action else None
    value = payload.value if payload is not None else None
    if not isinstance(value, dict):
        value = {"value": value} if value is not None else {}
    tag = (payload.tag if payload is not None else "") or "button"

    synthetic_text = f"/card {tag}"
    if value:
        try:
            synthetic_text += f" {json.dumps(value, ensure_ascii=False)}"
        except Exception:
            pass

    operator = action.operator
    operator_open_id = operator.open_id if operator else ""
    operator_name = (operator.name if operator else "") or operator_open_id

    # Source derivation: chat_id + operator info. chat_type defaults to
    # "group"; the caller (_on_sdk_card_action) overrides it after a
    # channel.get_chat_info lookup so P2P card actions are classified correctly.
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id=action.chat_id or "",
        chat_name=action.chat_id or "Feishu Chat",
        chat_type="group",
        user_id=operator_open_id,
        user_id_alt=(operator.user_id if operator else None),
        user_name=operator_name,
        thread_id=None,
    )
    return MessageEvent(
        text=synthetic_text,
        message_type=MessageType.COMMAND,
        source=source,
        raw_message=action.raw,
        message_id=action.message_id or str(uuid.uuid4()),
        timestamp=datetime.now(),
    )


async def _to_text_event_from_reaction(
    evt: "ReactionEvent",
    *,
    channel: "FeishuChannel",
    bot_open_id_fallback: str = "",
    chat_id_fallback: str = "",
    chat_type_fallback: str = "",
    operator_is_bot: bool = False,
) -> Optional[MessageEvent]:
    """SDK ReactionEvent → Hermes synthetic TEXT MessageEvent.

    Returns None when:
      - evt.action is not an add/remove variant
      - evt.operator.open_id == bot's own open_id (avoid self-reaction loop)

    Otherwise:
      text = f"reaction:<added|removed>:{emoji_type}"
      message_type = TEXT
      source built from chat info + operator profile
    """
    action = str(evt.action or "").strip().lower()
    if action in {"created", "create", "added", "add"}:
        action = "added"
    elif action in {"deleted", "delete", "removed", "remove"}:
        action = "removed"
    else:
        return None

    operator = evt.operator
    operator_open_id = operator.open_id if operator else ""
    if not operator_open_id:
        return None

    # Filter bot-origin reactions (mirrors legacy _on_reaction_event drop).
    bot_identity = getattr(channel, "bot_identity", None)
    bot_open_id = (getattr(bot_identity, "open_id", "") if bot_identity else "") or bot_open_id_fallback or ""
    if bot_open_id and operator_open_id == bot_open_id:
        return None

    emoji_type = evt.emoji_type or "UNKNOWN"
    synthetic_text = f"reaction:{action}:{emoji_type}"

    chat_id = evt.chat_id or chat_id_fallback or ""
    if not chat_id:
        return None
    # Resolve chat_info; identical handling to _on_sdk_card_action.
    chat_info_dict: Dict[str, Any] = {}
    try:
        if chat_id:
            info = await channel.get_chat_info(chat_id)
            if info is not None:
                raw_chat_type = str(getattr(info, "chat_type", "") or "").strip().lower()
                chat_info_dict = {
                    "chat_id": getattr(info, "chat_id", chat_id) or chat_id,
                    "name": str(getattr(info, "name", "") or chat_id),
                    "type": map_chat_type(raw_chat_type),
                    "raw_type": raw_chat_type or None,
                }
    except Exception:
        chat_info_dict = {}

    chat_type_event = (evt.chat_type or chat_type_fallback or "")
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id=chat_id,
        chat_name=chat_info_dict.get("name") or chat_id or "Feishu Chat",
        chat_type=resolve_source_chat_type(
            chat_info=chat_info_dict, event_chat_type=chat_type_event,
        ),
        user_id=operator_open_id,
        user_id_alt=(operator.user_id if operator else None),
        user_name=(operator.name if operator else None) or operator_open_id,
        thread_id=None,
        is_bot=operator_is_bot,
    )
    return MessageEvent(
        text=synthetic_text,
        message_type=MessageType.TEXT,
        source=source,
        raw_message=evt.raw,
        message_id=evt.message_id,
        timestamp=datetime.now(),
    )


def _sdk_comment_to_legacy_dict(evt: "CommentEvent") -> SimpleNamespace:
    """SDK CommentEvent → legacy envelope-shaped namespace for
    comments.handle_drive_comment_event(client, data, *, self_open_id).

    Returns a SimpleNamespace mimicking the original webhook envelope shape:
      ns.event = SimpleNamespace(...)  # inner event payload from evt.raw
      ns.header = SimpleNamespace(create_time=str(evt.timestamp))   # if present
      ns.ts = ...                       # envelope alt timestamp

    SDK note: ``CommentEvent.raw`` is the *inner* ``event`` dict
    (lark_oapi/channel/normalize/comment.py:140), NOT the full envelope.
    parse_drive_comment_event walks ``data.event`` (inner dict), so we wrap
    evt.raw under a new SimpleNamespace.event.
    """
    raw = evt.raw if isinstance(evt.raw, dict) else {}

    # SDK normalize_comment stores the inner event dict in evt.raw. If it
    # happens to also contain an "event" key (older payloads/full envelope),
    # prefer that as inner.
    inner_dict = raw.get("event") if isinstance(raw.get("event"), dict) else raw
    header_dict = raw.get("header") if isinstance(raw.get("header"), dict) else None
    envelope_ts = raw.get("ts")

    inner_ns = SimpleNamespace(**inner_dict) if isinstance(inner_dict, dict) else SimpleNamespace()
    header_ns = (
        SimpleNamespace(**header_dict)
        if isinstance(header_dict, dict)
        else SimpleNamespace(create_time=str(evt.timestamp))
    )

    return SimpleNamespace(
        event=inner_ns,
        header=header_ns,
        ts=envelope_ts,
    )
