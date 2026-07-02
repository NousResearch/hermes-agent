"""Generic Telegram Business ingestion policy, normalization, and audit support."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


class SecretaryMode(StrEnum):
    OFF = "off"
    AUDIT_ONLY = "audit_only"
    READ_ONLY = "read_only"


SECRETARY_MODES = frozenset(mode.value for mode in SecretaryMode)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_str_set(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        parts = value.split(",")
    else:
        try:
            parts = list(value)
        except TypeError:
            parts = [value]
    return frozenset(str(part).strip() for part in parts if str(part).strip())


class SecretaryTopics(dict):
    """Topic-name to Telegram thread-id mapping with attribute access."""

    def __getattr__(self, name: str) -> str | None:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _coerce_topics(value: Any) -> SecretaryTopics:
    if not isinstance(value, Mapping):
        return SecretaryTopics()
    return SecretaryTopics(
        {
            str(key).strip(): str(topic_id).strip()
            for key, topic_id in value.items()
            if str(key).strip() and str(topic_id).strip()
        }
    )


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    value = getattr(obj, name, default)
    if value is not default:
        return value
    api_kwargs = getattr(obj, "api_kwargs", None)
    if isinstance(api_kwargs, Mapping):
        return api_kwargs.get(name, default)
    return default


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return {}
    if hasattr(obj, "__dict__"):
        return {key: value for key, value in vars(obj).items() if not key.startswith("_")}
    return {}


def _iso_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    return str(value)


def _chat_type(chat: Any) -> str | None:
    value = _get(chat, "type")
    if value is None:
        return None
    return str(value).split(".")[-1].lower() or None


def _user_name(user: Any) -> str | None:
    return _get(user, "username") or _get(user, "full_name") or _get(user, "first_name") or None


def _media_refs(message: Any) -> list[str]:
    refs: list[str] = []
    for attr in ("video", "audio", "voice", "document", "sticker", "animation"):
        media = _get(message, attr)
        file_id = _get(media, "file_id")
        if file_id:
            refs.append(f"{attr}:{file_id}")
    photos = _get(message, "photo") or []
    for photo in photos if isinstance(photos, (list, tuple)) else [photos]:
        file_id = _get(photo, "file_id")
        if file_id:
            refs.append(f"photo:{file_id}")
    return refs


@dataclass(frozen=True)
class SecretaryPolicy:
    """Parsed ``platforms.telegram.extra.secretary`` inbound ingestion policy."""

    enabled: bool = False
    display_name: str = "Hermes"
    mode: SecretaryMode = SecretaryMode.OFF
    owner_chat_id: str | None = None
    topics: SecretaryTopics = field(default_factory=SecretaryTopics)
    allowed_business_connections: frozenset[str] = field(default_factory=frozenset)
    allowed_chats: frozenset[str] = field(default_factory=frozenset)
    redact_secrets: bool = True
    audit_path: Path | None = None

    @classmethod
    def from_config(cls, data: Any) -> "SecretaryPolicy":
        if not isinstance(data, Mapping):
            return cls()
        raw_mode = str(data.get("mode") or "off").strip().lower()
        try:
            mode = SecretaryMode(raw_mode)
        except ValueError as exc:
            allowed = ", ".join(sorted(SECRETARY_MODES))
            raise ValueError(
                f"Invalid Telegram Secretary mode {raw_mode!r}; expected one of: {allowed}"
            ) from exc
        enabled = _coerce_bool(data.get("enabled"), mode is not SecretaryMode.OFF)
        if not enabled:
            mode = SecretaryMode.OFF
        owner_chat_id = data.get("owner_chat_id")
        audit_path_raw = str(data.get("audit_path") or "").strip()
        return cls(
            enabled=enabled,
            display_name=str(data.get("display_name") or "Hermes").strip() or "Hermes",
            mode=mode,
            owner_chat_id=str(owner_chat_id).strip() if owner_chat_id is not None else None,
            topics=_coerce_topics(data.get("topics")),
            allowed_business_connections=_coerce_str_set(data.get("allowed_business_connections")),
            allowed_chats=_coerce_str_set(data.get("allowed_chats")),
            redact_secrets=_coerce_bool(data.get("redact_secrets"), True),
            audit_path=Path(audit_path_raw).expanduser() if audit_path_raw else None,
        )

    @classmethod
    def from_extra(cls, extra: Any) -> "SecretaryPolicy":
        if isinstance(extra, Mapping):
            return cls.from_config(extra.get("secretary"))
        return cls()

    @property
    def active(self) -> bool:
        return self.enabled and self.mode is not SecretaryMode.OFF

    @property
    def accepts_updates(self) -> bool:
        return self.active

    @property
    def reports_to_owner(self) -> bool:
        return bool(self.active and self.owner_chat_id and self.mode is SecretaryMode.READ_ONLY)

    def allows_business_connection(self, business_connection_id: str | None) -> bool:
        if not self.allowed_business_connections:
            return True
        return str(business_connection_id or "").strip() in self.allowed_business_connections

    def allows_chat(self, chat_id: str | None) -> bool:
        if not self.allowed_chats:
            return True
        return str(chat_id or "").strip() in self.allowed_chats

    def allows(self, event: "SecretaryEvent") -> bool:
        return self.allows_business_connection(event.business_connection_id) and self.allows_chat(event.chat_id)


@dataclass(frozen=True)
class SecretaryEvent:
    """Durable normalized audit event for Telegram Business updates."""

    event_type: str
    source: str = "telegram"
    mode: str = "business"
    timestamp: float = field(default_factory=time.time)
    update_id: int | None = None
    business_connection_id: str | None = None
    connection_user_id: str | None = None
    chat_id: str | None = None
    message_id: str | None = None
    message_ids: list[str] = field(default_factory=list)
    from_user_id: str | None = None
    from_username: str | None = None
    chat_type: str | None = None
    text: str | None = None
    caption: str | None = None
    media_refs: list[str] = field(default_factory=list)
    rights: dict[str, Any] = field(default_factory=dict)
    original_timestamp: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    ignored_reason: str | None = None

    @property
    def update_type(self) -> str:
        return self.event_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "source": self.source,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "update_id": self.update_id,
            "business_connection_id": self.business_connection_id,
            "connection_user_id": self.connection_user_id,
            "chat_id": self.chat_id,
            "message_id": self.message_id,
            "message_ids": list(self.message_ids),
            "from_user_id": self.from_user_id,
            "from_username": self.from_username,
            "chat_type": self.chat_type,
            "text": self.text,
            "caption": self.caption,
            "media_refs": list(self.media_refs),
            "rights": dict(self.rights),
            "original_timestamp": self.original_timestamp,
            "raw": dict(self.raw),
            "ignored_reason": self.ignored_reason,
        }

    def redacted(self) -> "SecretaryEvent":
        try:
            from agent.redact import redact_sensitive_text
        except Exception:
            return self
        updates: dict[str, Any] = {}
        for field_name in ("text", "caption"):
            value = getattr(self, field_name)
            if isinstance(value, str) and value:
                updates[field_name] = redact_sensitive_text(value)
        return replace(self, **updates) if updates else self

    def ignored(self, reason: str) -> "SecretaryEvent":
        return replace(self, ignored_reason=reason)


class SecretaryAuditStore:
    """Append-only JSONL audit store for Telegram Business events."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (get_hermes_home() / "gateway" / "telegram_business_ingestion.jsonl")

    def append(self, event: SecretaryEvent) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")
        logger.info(
            "[Telegram Business] Audited %s event for chat=%s message=%s ignored=%s",
            event.event_type,
            event.chat_id,
            event.message_id,
            event.ignored_reason,
        )
        return self.path


def normalize_secretary_update(update: Any) -> SecretaryEvent | None:
    business_connection = getattr(update, "business_connection", None)
    if business_connection is not None:
        return _normalize_business_connection(update, business_connection)
    business_message = getattr(update, "business_message", None)
    if business_message is not None:
        return _normalize_business_message(update, business_message, "business_message")
    edited_business_message = getattr(update, "edited_business_message", None)
    if edited_business_message is not None:
        return _normalize_business_message(update, edited_business_message, "edited_business_message")
    deleted = getattr(update, "deleted_business_messages", None)
    if deleted is not None:
        return _normalize_deleted_business_messages(update, deleted)
    return None


def _normalize_business_connection(update: Any, connection: Any) -> SecretaryEvent:
    user = _get(connection, "user")
    connection_id = str(_get(connection, "id", "") or "") or None
    rights = _to_plain_dict(_get(connection, "rights"))
    can_reply = _get(connection, "can_reply")
    if can_reply is not None:
        rights["can_reply"] = bool(can_reply)
    return SecretaryEvent(
        event_type="business_connection",
        update_id=_get(update, "update_id"),
        business_connection_id=connection_id,
        connection_user_id=str(_get(user, "id", "") or "") or None,
        from_user_id=str(_get(user, "id", "") or "") or None,
        from_username=_user_name(user),
        chat_type="business",
        rights=rights,
        original_timestamp=_iso_timestamp(_get(connection, "date")),
        raw=_to_plain_dict(connection),
    )


def _normalize_business_message(update: Any, message: Any, event_type: str) -> SecretaryEvent:
    chat = _get(message, "chat")
    user = _get(message, "from_user")
    can_reply = _get(message, "can_reply")
    return SecretaryEvent(
        event_type=event_type,
        update_id=_get(update, "update_id"),
        business_connection_id=str(_get(message, "business_connection_id", "") or "") or None,
        chat_id=str(_get(chat, "id", "") or "") or None,
        message_id=str(_get(message, "message_id", "") or "") or None,
        from_user_id=str(_get(user, "id", "") or "") or None,
        from_username=_user_name(user),
        chat_type=_chat_type(chat) or "business",
        text=_get(message, "text"),
        caption=_get(message, "caption"),
        media_refs=_media_refs(message),
        rights={"can_reply": bool(can_reply) if can_reply is not None else False},
        original_timestamp=_iso_timestamp(_get(message, "date")),
        raw=_to_plain_dict(message),
    )


def _normalize_deleted_business_messages(update: Any, deleted: Any) -> SecretaryEvent:
    chat = _get(deleted, "chat")
    message_ids = _get(deleted, "message_ids", []) or []
    return SecretaryEvent(
        event_type="deleted_business_messages",
        update_id=_get(update, "update_id"),
        business_connection_id=str(_get(deleted, "business_connection_id", "") or "") or None,
        chat_id=str(_get(chat, "id", "") or "") or None,
        message_ids=[str(message_id) for message_id in message_ids],
        chat_type=_chat_type(chat) or "business",
        rights={"can_reply": False},
        raw=_to_plain_dict(deleted),
    )


def append_audit_event(event: SecretaryEvent, policy: SecretaryPolicy | None = None) -> Path:
    return SecretaryAuditStore(policy.audit_path if policy else None).append(event)


def render_owner_summary(event: SecretaryEvent, *, assistant_name: str = "Hermes") -> str:
    import html

    msg_ids = event.message_id or ",".join(event.message_ids) or "n/a"
    sender = event.from_username or event.from_user_id or event.connection_user_id or "unknown"
    content = (event.text or event.caption or "").strip()
    if len(content) > 240:
        content = content[:237].rstrip() + "..."
    classification = "inbound.delete" if event.event_type == "deleted_business_messages" else "inbound.text"
    if event.event_type == "business_connection":
        classification = "connection.update"
    summary = content or f"{event.event_type} update"
    action = "Review only; no message was dispatched or sent as the user."
    assistant = assistant_name.strip() or "Hermes"
    lines = [
        f"<b>{html.escape(assistant)}: Telegram Business triage</b>",
        f"<b>Classification:</b> <code>{html.escape(classification)}</code>",
        f"<b>Summary:</b> {html.escape(summary)}",
        f"<b>Suggested next action:</b> {html.escape(action)}",
        "<b>Source:</b>",
        f"- <b>event_type:</b> <code>{html.escape(event.event_type)}</code>",
        f"- <b>business_connection_id:</b> <code>{html.escape(event.business_connection_id or 'unknown')}</code>",
        f"- <b>chat_id:</b> <code>{html.escape(event.chat_id or 'unknown')}</code>",
        f"- <b>message_id:</b> <code>{html.escape(msg_ids)}</code>",
        f"- <b>from:</b> <code>{html.escape(sender)}</code>",
        f"- <b>rights.can_reply:</b> <code>{str(event_can_reply(event)).lower()}</code>",
    ]
    if content:
        lines.append(f"Text: {html.escape(content)}")
    return "\n".join(lines)


def event_can_reply(
    event: SecretaryEvent,
    connection_rights: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """Return whether latest known Telegram Business rights allow a reply."""
    if "can_reply" in event.rights:
        return bool(event.rights.get("can_reply"))
    if connection_rights and event.business_connection_id:
        return bool(connection_rights.get(event.business_connection_id, {}).get("can_reply"))
    return False
