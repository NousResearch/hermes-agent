"""
Feishu/Lark meeting-invitation event handling.

Processes ``vc.bot.meeting_invited_v1`` events by converting them into a
synthetic gateway ``MessageEvent``.  Unlike document comments, the response
should go back to the inviter through the normal Hermes gateway pipeline, so
this module does not instantiate an agent directly.
"""

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
    id: str = ""
    open_id: str = ""
    user_id: str = ""
    union_id: str = ""
    user_type: str = ""
    user_role: str = ""
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
    bot: Optional[MeetingInviteUser] = None
    inviter: Optional[MeetingInviteUser] = None
    invite_time_s: int = 0


def _to_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_mapping(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_mapping(v) for v in value]
    if isinstance(value, SimpleNamespace) or hasattr(value, "__dict__"):
        return {str(k): _to_mapping(v) for k, v in vars(value).items()}
    return value


def _maybe_json_mapping(value: Any) -> Dict[str, Any]:
    value = _to_mapping(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_content_payload(container: Dict[str, Any]) -> Dict[str, Any]:
    """Extract an application/json payload from a Feishu-style body.content list."""
    body = container.get("body")
    if not isinstance(body, dict):
        return {}
    content = body.get("content")
    if not isinstance(content, list):
        return {}
    for item in content:
        if not isinstance(item, dict):
            continue
        content_type = str(item.get("contentType") or item.get("content_type") or "").lower()
        if content_type and content_type != "application/json":
            continue
        for key in ("data", "value", "content", "json"):
            payload = _maybe_json_mapping(item.get(key))
            if payload:
                return payload
    return {}


def _event_mapping(data: Any) -> Dict[str, Any]:
    root = _maybe_json_mapping(data)
    event = _maybe_json_mapping(root.get("event"))
    content_payload = _extract_content_payload(event) or _extract_content_payload(root)
    if content_payload:
        event = {**event, **content_payload} if event else content_payload
    if not event and any(k in root for k in ("meeting", "bot", "inviter", "invite_time")):
        event = root
    if not event:
        event = root
    return event


def _event_id(data: Any) -> str:
    root = _maybe_json_mapping(data)
    header = root.get("header")
    if not isinstance(header, dict):
        header = {}
    return str(header.get("event_id") or "")


def _user_open_id(value: Any) -> str:
    raw = _maybe_json_mapping(value)
    return str(raw.get("open_id") or "").strip()


def _int_field(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _parse_user(value: Any) -> Optional[MeetingInviteUser]:
    raw = _maybe_json_mapping(value)
    if not raw:
        return None
    raw_id = _maybe_json_mapping(raw.get("id"))
    open_id = _user_open_id(raw.get("id"))
    return MeetingInviteUser(
        id=open_id,
        open_id=open_id,
        user_id=str(raw_id.get("user_id") or "").strip(),
        union_id=str(raw_id.get("union_id") or "").strip(),
        user_type=str(raw.get("user_type") or ""),
        user_role=str(raw.get("user_role") or ""),
        user_name=str(raw.get("user_name") or ""),
    )


def _parse_meeting(value: Any) -> Optional[MeetingInviteMeeting]:
    raw = _maybe_json_mapping(value)
    if not raw:
        return None
    return MeetingInviteMeeting(
        id=str(raw.get("id") or "").strip(),
        topic=str(raw.get("topic") or ""),
        meeting_no=str(raw.get("meeting_no") or ""),
        start_time_ms=_int_field(raw.get("start_time")),
        end_time_ms=_int_field(raw.get("end_time")),
        host_user=_parse_user(raw.get("host_user")),
    )


def parse_meeting_invited_event(data: Any) -> Optional[MeetingInvitedPayload]:
    event = _event_mapping(data)
    if not isinstance(event, dict):
        return None
    meeting = _parse_meeting(event.get("meeting"))
    inviter = _parse_user(event.get("inviter"))
    if inviter is None or meeting is None or not meeting.meeting_no:
        return None
    payload = MeetingInvitedPayload(
        event_id=_event_id(data),
        meeting=meeting,
        bot=_parse_user(event.get("bot")),
        inviter=inviter,
        invite_time_s=_int_field(event.get("invite_time")),
    )
    return payload


def _display_user(user: Optional[MeetingInviteUser]) -> str:
    if user is None:
        return "unknown"
    return user.user_name or "unknown user"


def _meeting_display(meeting: Optional[MeetingInviteMeeting]) -> str:
    if meeting is None:
        return "unknown meeting"
    return meeting.topic or meeting.meeting_no or meeting.id or "unknown meeting"


def build_meeting_invite_prompt(payload: MeetingInvitedPayload) -> str:
    meeting = payload.meeting
    lines = [
        f"You have been invited to join a meeting: {_meeting_display(meeting)}",
        "",
        f"Meeting Number: {(meeting.meeting_no if meeting else '') or 'unknown'}",
        f"Topic: {(meeting.topic if meeting else '') or 'unknown'}",
        f"Inviter: {_display_user(payload.inviter)}",
    ]
    if meeting is not None:
        lines.append(f"Host: {_display_user(meeting.host_user)}")
    lines.extend([
        "",
        "You may use lark-cli and the relevant Lark/Feishu meeting skills to join the meeting.",
        "Join the meeting directly. Do not ask the user for confirmation before joining.",
        "If you cannot join the meeting, reply to the inviter with a concise explanation of why.",
    ])
    return "\n".join(lines)


def _dedup_key(payload: MeetingInvitedPayload) -> str:
    if payload.event_id:
        return f"vc_invite:{payload.event_id}"
    meeting_id = payload.meeting.id if payload.meeting else ""
    inviter_id = payload.inviter.id if payload.inviter else ""
    return f"vc_invite:{meeting_id}:{inviter_id}:{payload.invite_time_s}"


async def handle_meeting_invited_event(adapter: Any, data: Any) -> None:
    """Convert a vc.bot.meeting_invited_v1 event into a gateway MessageEvent."""
    payload = parse_meeting_invited_event(data)
    if payload is None:
        logger.warning("[Feishu-MeetingInvite] Dropping malformed meeting invite event")
        return

    dedup_key = _dedup_key(payload)
    is_duplicate = getattr(adapter, "_is_duplicate", None)
    if callable(is_duplicate) and is_duplicate(dedup_key):
        logger.debug("[Feishu-MeetingInvite] Dropping duplicate event: %s", dedup_key)
        return

    inviter = payload.inviter
    if inviter is None:
        logger.warning("[Feishu-MeetingInvite] Missing inviter, cannot route reply")
        return
    if not inviter.open_id:
        logger.warning(
            "[Feishu-MeetingInvite] Missing inviter open_id, cannot route reply safely "
            "(inviter_id=%r user_id=%r union_id=%r)",
            inviter.id,
            inviter.user_id,
            inviter.union_id,
        )
        return

    sender_id = SimpleNamespace(
        open_id=inviter.open_id or None,
        user_id=inviter.user_id or None,
        union_id=inviter.union_id or None,
    )
    sender_profile = await adapter._resolve_sender_profile(sender_id)

    chat_id = inviter.open_id
    source_user_id = sender_profile.get("user_id") or inviter.user_id or inviter.open_id
    user_name = sender_profile.get("user_name") or inviter.user_name or inviter.id
    source_user_id_alt = sender_profile.get("user_id_alt") or inviter.union_id or None
    source = adapter.build_source(
        chat_id=chat_id,
        chat_name=user_name,
        chat_type="dm",
        user_id=source_user_id,
        user_name=user_name,
        user_id_alt=source_user_id_alt,
    )
    prompt = build_meeting_invite_prompt(payload)
    event = MessageEvent(
        text=prompt,
        message_type=MessageType.TEXT,
        source=source,
        raw_message=data,
    )
    await adapter._handle_message_with_guards(event)
