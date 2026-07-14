"""Profile-local Telegram input intake buffer.

C2-B initial scope: a single Telegram forum topic is accepted into one
append-only JSONL file under the active profile. Deduplication deliberately
scans the locked JSONL buffer; this is acceptable for the small initial volume.
A scalable index may be introduced later only after separate approval.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType

logger = logging.getLogger(__name__)

INPUT_CHAT_ID = "-1004298945366"
INPUT_THREAD_ID = "2664"
ACK_TEXT = "✓ прийнято"
SCHEMA_VERSION = 1


class IntakeError(RuntimeError):
    """Sanitized intake persistence failure."""


@dataclass(frozen=True)
class IntakeResult:
    status: str
    record: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    @property
    def duplicate(self) -> bool:
        return self.status == "duplicate"


def default_buffer_path(profile_dir: str | os.PathLike[str] | None = None) -> Path:
    if profile_dir is None:
        profile_dir = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(profile_dir).expanduser() / "intake" / "input_buffer.jsonl"


def _platform_value(event: MessageEvent) -> str:
    source = getattr(event, "source", None)
    platform = getattr(source, "platform", None)
    return getattr(platform, "value", str(platform or ""))


def is_input_intake_event(event: MessageEvent) -> bool:
    source = getattr(event, "source", None)
    if source is None:
        return False
    if _platform_value(event) != Platform.TELEGRAM.value:
        return False
    if str(getattr(source, "chat_id", "")) != INPUT_CHAT_ID:
        return False
    if str(getattr(source, "thread_id", "")) != INPUT_THREAD_ID:
        return False
    if getattr(source, "chat_type", None) == "channel":
        return False
    message_id = getattr(event, "message_id", None) or getattr(source, "message_id", None)
    if message_id is None or str(message_id).strip() == "":
        return False
    try:
        if event.get_command():
            return False
    except Exception:
        if str(getattr(event, "text", "") or "").lstrip().startswith("/"):
            return False
    return True


def dedup_key_for(event: MessageEvent) -> str:
    source = event.source
    return f"telegram:{source.chat_id}:{source.thread_id}:{event.message_id}"


def record_id_for(dedup_key: str) -> str:
    return "intake_" + hashlib.sha256(dedup_key.encode("utf-8")).hexdigest()[:20]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _attachment_from_event(event: MessageEvent) -> tuple[str, Optional[dict[str, Any]]]:
    raw = getattr(event, "raw_message", None)
    msg_type = getattr(event, "message_type", None)
    text = getattr(event, "text", "") or ""

    def base(kind: str) -> dict[str, Any]:
        return {"kind": kind, "has_caption": bool(text)}

    if raw is not None:
        if getattr(raw, "photo", None):
            att = base("photo")
            try:
                photo = raw.photo[-1]
                width = _safe_int(getattr(photo, "width", None))
                height = _safe_int(getattr(photo, "height", None))
                file_size = _safe_int(getattr(photo, "file_size", None))
                if width is not None:
                    att["width"] = width
                if height is not None:
                    att["height"] = height
                if file_size is not None:
                    att["size"] = file_size
            except Exception:
                pass
            return ("photo_caption" if text else "attachment_only"), att

        document = getattr(raw, "document", None)
        if document is not None:
            att = base("document")
            file_name = getattr(document, "file_name", None)
            mime_type = getattr(document, "mime_type", None)
            file_size = _safe_int(getattr(document, "file_size", None))
            if file_name:
                att["file_name"] = str(file_name)
            if mime_type:
                att["mime_type"] = str(mime_type)
            if file_size is not None:
                att["size"] = file_size
            return ("document_caption" if text else "attachment_only"), att

        for attr, kind in (("voice", "voice"), ("audio", "audio"), ("video", "attachment_only"), ("sticker", "attachment_only")):
            obj = getattr(raw, attr, None)
            if obj is None:
                continue
            att = base(attr)
            duration = _safe_int(getattr(obj, "duration", None))
            file_size = _safe_int(getattr(obj, "file_size", None))
            mime_type = getattr(obj, "mime_type", None)
            if duration is not None:
                att["duration"] = duration
            if file_size is not None:
                att["size"] = file_size
            if mime_type:
                att["mime_type"] = str(mime_type)
            return kind, att

    if msg_type == MessageType.TEXT:
        return "text", None
    value = getattr(msg_type, "value", None) or str(msg_type or "attachment")
    if value in {"voice", "audio"}:
        return value, {"kind": value, "has_caption": bool(text)}
    return "attachment_only", {"kind": value, "has_caption": bool(text)}


def build_record(event: MessageEvent, source_session_key: str | None = None) -> dict[str, Any]:
    if not is_input_intake_event(event):
        raise IntakeError("not_input_intake_event")
    source = event.source
    dedup_key = dedup_key_for(event)
    raw = getattr(event, "raw_message", None)
    user = getattr(raw, "from_user", None) if raw is not None else None
    sender_username = getattr(user, "username", None) if user is not None else None
    content_type, attachment = _attachment_from_event(event)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id_for(dedup_key),
        "received_at": utc_now_iso(),
        "platform": "telegram",
        "chat_id": str(source.chat_id),
        "thread_id": str(source.thread_id),
        "message_id": str(event.message_id),
        "sender_id": str(source.user_id) if source.user_id is not None else None,
        "sender_username": str(sender_username) if sender_username else None,
        "text": str(getattr(event, "text", "") or ""),
        "content_type": content_type,
        "status": "received",
        "source_session_key": source_session_key,
        "dedup_key": dedup_key,
        "attachment": attachment,
    }


def append_record_locked(record: dict[str, Any], buffer_path: str | os.PathLike[str]) -> IntakeResult:
    path = Path(buffer_path).expanduser()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)

    flags = os.O_RDWR | os.O_CREAT
    fd = os.open(path, flags, 0o600)
    try:
        os.chmod(path, 0o600)
        with os.fdopen(fd, "r+", encoding="utf-8", newline="") as fh:
            fd = -1
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0)
            dedup_key = record.get("dedup_key")
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise IntakeError(f"malformed_jsonl_line:{line_no}") from exc
                if existing.get("dedup_key") == dedup_key:
                    return IntakeResult("duplicate", existing)

            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            fh.seek(0, os.SEEK_END)
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
            return IntakeResult("accepted", record)
    except IntakeError:
        raise
    except Exception as exc:  # noqa: BLE001 - sanitize caller-visible error
        raise IntakeError(exc.__class__.__name__) from exc
    finally:
        if fd >= 0:
            os.close(fd)


def accept_event(
    event: MessageEvent,
    *,
    source_session_key: str | None,
    buffer_path: str | os.PathLike[str] | None = None,
) -> IntakeResult:
    if not is_input_intake_event(event):
        return IntakeResult("not_intake")
    record = build_record(event, source_session_key)
    path = Path(buffer_path) if buffer_path is not None else default_buffer_path()
    return append_record_locked(record, path)
