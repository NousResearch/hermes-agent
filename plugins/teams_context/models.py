"""Models and normalization helpers for Teams chat context."""

from __future__ import annotations

import html
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any


_CHAT_RESOURCE_RE = re.compile(
    r"chats(?:\(['\"](?P<quoted_chat>[^'\"]+)['\"]\)|/(?P<path_chat>[^/]+))"
    r"(?:/messages(?:\(['\"](?P<quoted_msg>[^'\"]+)['\"]\)|/(?P<path_msg>[^/]+))?)?",
    re.IGNORECASE,
)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"br", "p", "div", "li"}:
            self.parts.append("\n")

    def get_text(self) -> str:
        text = "".join(self.parts)
        text = html.unescape(text)
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def strip_html(value: str | None) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(value or "")
        parser.close()
        return parser.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", "", html.unescape(value or "")).strip()


def parse_graph_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_chat_resource(resource: str | None) -> tuple[str | None, str | None]:
    match = _CHAT_RESOURCE_RE.search(str(resource or "").strip())
    if not match:
        return None, None
    chat_id = match.group("quoted_chat") or match.group("path_chat")
    message_id = match.group("quoted_msg") or match.group("path_msg")
    return chat_id, message_id


def synthetic_relay_message_id(payload: dict[str, Any], chat_id: str) -> str:
    """Build a stable message id when Power Automate omits the Teams id."""
    material = {
        "chat_id": chat_id,
        "created_at": payload.get("created_at") or payload.get("createdDateTime"),
        "sender_id": payload.get("sender_id"),
        "sender_name": payload.get("sender_name") or payload.get("from"),
        "text": payload.get("text") or payload.get("body"),
        "html": payload.get("html") or payload.get("body_html"),
        "web_url": payload.get("web_url") or payload.get("webUrl") or payload.get("link"),
    }
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    return f"power_automate:{digest}"


@dataclass
class TeamsChatMessage:
    tenant_id: str | None
    chat_id: str
    message_id: str
    sender_id: str | None = None
    sender_name: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    deleted_at: datetime | None = None
    text: str = ""
    html: str | None = None
    web_url: str | None = None
    meeting_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_graph(cls, chat_id: str, payload: dict[str, Any], *, tenant_id: str | None = None) -> "TeamsChatMessage":
        body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
        html_body = str(body.get("content") or "")
        sender = payload.get("from") if isinstance(payload.get("from"), dict) else {}
        user = sender.get("user") if isinstance(sender.get("user"), dict) else {}
        app = sender.get("application") if isinstance(sender.get("application"), dict) else {}
        sender_id = user.get("id") or app.get("id")
        sender_name = user.get("displayName") or app.get("displayName")
        message_id = str(payload.get("id") or "").strip()
        if not message_id:
            raise ValueError("Teams chat message id is required")
        deleted_at = parse_graph_datetime(payload.get("deletedDateTime"))
        return cls(
            tenant_id=tenant_id,
            chat_id=str(chat_id),
            message_id=message_id,
            sender_id=sender_id,
            sender_name=sender_name,
            created_at=parse_graph_datetime(payload.get("createdDateTime")),
            updated_at=parse_graph_datetime(payload.get("lastModifiedDateTime")),
            deleted_at=deleted_at,
            text="" if deleted_at else strip_html(html_body),
            html=html_body if not deleted_at else None,
            web_url=payload.get("webUrl"),
            meeting_id=_extract_meeting_id(payload),
            raw=dict(payload),
        )

    @classmethod
    def from_relay(cls, payload: dict[str, Any]) -> "TeamsChatMessage":
        """Normalize a Power Automate relay payload into a stored message."""
        chat_id = str(
            payload.get("chat_id")
            or payload.get("conversation_id")
            or payload.get("channel_id")
            or "power_automate"
        ).strip()
        message_id = str(payload.get("message_id") or payload.get("id") or "").strip()
        if not message_id:
            message_id = synthetic_relay_message_id(payload, chat_id)
        html_body = payload.get("html") or payload.get("body_html")
        text = str(payload.get("text") or payload.get("body") or "").strip()
        if not text and html_body:
            text = strip_html(str(html_body))
        return cls(
            tenant_id=payload.get("tenant_id"),
            chat_id=chat_id,
            message_id=message_id,
            sender_id=payload.get("sender_id"),
            sender_name=payload.get("sender_name") or payload.get("from"),
            created_at=parse_graph_datetime(
                payload.get("created_at") or payload.get("createdDateTime")
            ),
            updated_at=parse_graph_datetime(
                payload.get("updated_at") or payload.get("lastModifiedDateTime")
            ),
            text=text,
            html=str(html_body) if html_body else None,
            web_url=payload.get("web_url") or payload.get("webUrl") or payload.get("link"),
            meeting_id=payload.get("meeting_id"),
            raw=dict(payload),
        )


def _extract_meeting_id(payload: dict[str, Any]) -> str | None:
    for key in ("meetingId", "onlineMeetingId"):
        value = payload.get(key)
        if value:
            return str(value)
    context = payload.get("eventDetail")
    if isinstance(context, dict):
        for key in ("meetingId", "onlineMeetingId"):
            value = context.get(key)
            if value:
                return str(value)
    return None
