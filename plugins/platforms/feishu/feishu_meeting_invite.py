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
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _content_payload(container: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap a Feishu ``body.content`` list carrying an application/json payload."""
    content = _as_dict(container.get("body")).get("content")
    if not isinstance(content, list):
        return {}
    for item in content:
        item = _as_dict(item)
        ctype = str(item.get("contentType") or item.get("content_type") or "").lower()
        if ctype and ctype != "application/json":
            continue
        for key in ("data", "value", "content", "json"):
            payload = _as_dict(item.get(key))
            if payload:
                return payload
    return {}


def _int_field(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _parse_user(value: Any) -> Optional[MeetingInviteUser]:
    raw = _as_dict(value)
    if not raw:
        return None
    raw_id = _as_dict(raw.get("id"))
    return MeetingInviteUser(
        open_id=str(raw_id.get("open_id") or "").strip(),
        user_id=str(raw_id.get("user_id") or "").strip(),
        union_id=str(raw_id.get("union_id") or "").strip(),
        user_name=str(raw.get("user_name") or ""),
    )


def _parse_meeting(value: Any) -> Optional[MeetingInviteMeeting]:
    raw = _as_dict(value)
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
    root = _as_dict(data)
    event = _as_dict(root.get("event"))
    event = event or root
    content = _content_payload(event) or _content_payload(root)
    if content:
        event = {**event, **content}

    meeting = _parse_meeting(event.get("meeting"))
    inviter = _parse_user(event.get("inviter"))
    if inviter is None or meeting is None or not meeting.meeting_no:
        return None

    return MeetingInvitedPayload(
        event_id=str(_as_dict(root.get("header")).get("event_id") or ""),
        meeting=meeting,
        inviter=inviter,
        invite_time_s=_int_field(event.get("invite_time")),
    )


def build_meeting_invite_prompt(payload: MeetingInvitedPayload) -> str:
    meeting = payload.meeting
    inviter_name = (payload.inviter.user_name if payload.inviter else "") or "unknown"
    host_name = (meeting.host_user.user_name if meeting and meeting.host_user else "") or "unknown"
    display = (meeting.topic or meeting.meeting_no or meeting.id) if meeting else "unknown meeting"
    return "\n".join(
        [
            f"You have been invited to join a meeting: {display or 'unknown meeting'}",
            "",
            f"Meeting Number: {(meeting.meeting_no if meeting else '') or 'unknown'}",
            f"Topic: {(meeting.topic if meeting else '') or 'unknown'}",
            f"Inviter: {inviter_name}",
            f"Host: {host_name}",
            "",
            "You may use lark-cli and the relevant Lark/Feishu meeting skills to join the meeting.",
            "Join the meeting directly. Do not ask the user for confirmation before joining.",
            "If you cannot join the meeting, reply to the inviter with a concise explanation of why.",
        ]
    )


def _dedup_key(payload: MeetingInvitedPayload) -> str:
    if payload.event_id:
        return f"vc_invite:{payload.event_id}"
    meeting_id = payload.meeting.id if payload.meeting else ""
    inviter_id = payload.inviter.open_id if payload.inviter else ""
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
    if inviter is None or not inviter.open_id:
        logger.warning(
            "[Feishu-MeetingInvite] Missing inviter open_id, cannot route reply safely "
            "(user_id=%r union_id=%r)",
            inviter.user_id if inviter else None,
            inviter.union_id if inviter else None,
        )
        return

    sender_id = SimpleNamespace(
        open_id=inviter.open_id or None,
        user_id=inviter.user_id or None,
        union_id=inviter.union_id or None,
    )
    sender_profile = await adapter._resolve_sender_profile(sender_id)

    user_name = sender_profile.get("user_name") or inviter.user_name or inviter.open_id
    source = adapter.build_source(
        chat_id=inviter.open_id,
        chat_name=user_name,
        chat_type="dm",
        user_id=sender_profile.get("user_id") or inviter.user_id or inviter.open_id,
        user_name=user_name,
        user_id_alt=sender_profile.get("user_id_alt") or inviter.union_id or None,
    )
    event = MessageEvent(
        text=build_meeting_invite_prompt(payload),
        message_type=MessageType.TEXT,
        source=source,
        raw_message=data,
    )
    await adapter._handle_message_with_guards(event)
