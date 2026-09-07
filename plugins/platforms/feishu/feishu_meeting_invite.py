"""Feishu/Lark meeting-invitation events: ``vc.bot.meeting_invited_v1`` -> synthetic gateway ``MessageEvent`` so the
reply reaches the inviter through the normal gateway pipeline (no agent is instantiated here)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

from gateway.platforms.base import MessageEvent, MessageType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeetingInviteUser:
    open_id: str = ""
    user_id: str = ""
    union_id: str = ""
    user_name: str = ""


@dataclass(frozen=True)
class MeetingInviteMeeting:
    id: str = ""
    topic: str = ""
    meeting_no: str = ""
    start_time_ms: int = 0
    end_time_ms: int = 0
    host_user: Optional[MeetingInviteUser] = None


@dataclass(frozen=True)
class MeetingInvitedPayload:
    event_id: str = ""
    meeting: Optional[MeetingInviteMeeting] = None
    inviter: Optional[MeetingInviteUser] = None
    invite_time_s: int = 0


def _as_dict(value: Any) -> Dict[str, Any]:
    """Coerce a lark SDK object / dict / JSON string into a plain dict."""
    if isinstance(value, SimpleNamespace) or (value is not None and hasattr(value, "__dict__")):
        value = vars(value)
    try:
        value = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError):
        return {}
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}


def _content_payload(container: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap a Feishu ``body.content`` list carrying an application/json payload."""
    content = _as_dict(container.get("body")).get("content")
    for item in map(_as_dict, content if isinstance(content, list) else ()):
        ctype = str(item.get("contentType") or item.get("content_type") or "").lower()
        payload = next((p for p in map(_as_dict, (item.get(k) for k in ("data", "value", "content", "json"))) if p), {}) if ctype in ("", "application/json") else {}
        if payload:
            return payload
    return {}


def _str_field(raw: Dict[str, Any], key: str, strip: bool = True) -> str:
    return str(raw.get(key) or "").strip() if strip else str(raw.get(key) or "")


def _int_field(value: Any) -> int:
    try:
        return int(str(value).strip()) if value not in (None, "") else 0
    except (TypeError, ValueError):
        return 0


def _parse_user(value: Any) -> Optional[MeetingInviteUser]:
    raw = _as_dict(value)
    raw_id = _as_dict(raw.get("id"))
    return MeetingInviteUser(open_id=_str_field(raw_id, "open_id"), user_id=_str_field(raw_id, "user_id"), union_id=_str_field(raw_id, "union_id"),
                             user_name=_str_field(raw, "user_name", strip=False)) if raw else None


def _parse_meeting(value: Any) -> Optional[MeetingInviteMeeting]:
    raw = _as_dict(value)
    return MeetingInviteMeeting(
        id=_str_field(raw, "id"), topic=_str_field(raw, "topic", strip=False), meeting_no=_str_field(raw, "meeting_no", strip=False),
        start_time_ms=_int_field(raw.get("start_time")), end_time_ms=_int_field(raw.get("end_time")), host_user=_parse_user(raw.get("host_user")),
    ) if raw else None


def parse_meeting_invited_event(data: Any) -> Optional[MeetingInvitedPayload]:
    root = _as_dict(data)
    event = _as_dict(root.get("event")) or root
    event = {**event, **(_content_payload(event) or _content_payload(root))}
    meeting, inviter = _parse_meeting(event.get("meeting")), _parse_user(event.get("inviter"))
    if inviter is None or meeting is None or not meeting.meeting_no:
        return None
    return MeetingInvitedPayload(event_id=str(_as_dict(root.get("header")).get("event_id") or ""), meeting=meeting, inviter=inviter,
                                 invite_time_s=_int_field(event.get("invite_time")))


def build_meeting_invite_prompt(payload: MeetingInvitedPayload) -> str:
    meeting = payload.meeting or MeetingInviteMeeting()
    inviter_name = (payload.inviter.user_name if payload.inviter else "") or "unknown"
    host_name = (meeting.host_user.user_name if meeting.host_user else "") or "unknown"
    display = (meeting.topic or meeting.meeting_no or meeting.id) or "unknown meeting"
    return "\n".join([
        f"You have been invited to join a meeting: {display}", "",
        f"Meeting Number: {meeting.meeting_no or 'unknown'}", f"Topic: {meeting.topic or 'unknown'}", f"Inviter: {inviter_name}", f"Host: {host_name}", "",
        "You may use lark-cli and the relevant Lark/Feishu meeting skills to join the meeting.",
        "Join the meeting directly. Do not ask the user for confirmation before joining.",
        "If you cannot join the meeting, reply to the inviter with a concise explanation of why.",
    ])


def _dedup_key(payload: MeetingInvitedPayload) -> str:
    if payload.event_id:
        return f"vc_invite:{payload.event_id}"
    return f"vc_invite:{payload.meeting.id if payload.meeting else ''}:{payload.inviter.open_id if payload.inviter else ''}:{payload.invite_time_s}"


async def handle_meeting_invited_event(adapter: Any, data: Any) -> None:
    """Convert a vc.bot.meeting_invited_v1 event into a gateway MessageEvent."""
    payload = parse_meeting_invited_event(data)
    if payload is None:
        return logger.warning("[Feishu-MeetingInvite] Dropping malformed meeting invite event")
    dedup_key = _dedup_key(payload)
    is_duplicate = getattr(adapter, "_is_duplicate", None)
    if callable(is_duplicate) and await is_duplicate(dedup_key):
        return logger.debug("[Feishu-MeetingInvite] Dropping duplicate event: %s", dedup_key)
    inviter = payload.inviter
    if inviter is None or not inviter.open_id:
        return logger.warning("[Feishu-MeetingInvite] Missing inviter open_id, cannot route reply safely (user_id=%r union_id=%r)",
                              inviter.user_id if inviter else None, inviter.union_id if inviter else None)
    sender_profile = await adapter._resolve_sender_profile(SimpleNamespace(open_id=inviter.open_id or None, user_id=inviter.user_id or None, union_id=inviter.union_id or None))
    user_name = sender_profile.get("user_name") or inviter.user_name or inviter.open_id
    source = adapter.build_source(
        chat_id=inviter.open_id, chat_name=user_name, chat_type="dm", user_id=sender_profile.get("user_id") or inviter.user_id or inviter.open_id,
        user_name=user_name, user_id_alt=sender_profile.get("user_id_alt") or inviter.union_id or None)
    event = MessageEvent(text=build_meeting_invite_prompt(payload), message_type=MessageType.TEXT, source=source, raw_message=data)
    await adapter._handle_message_with_guards(event)
