"""Local Project Rooms store for Mission Control.

Project Room content is persisted as inert local context only. This module
does not execute, route, publish, or interpret room content as instructions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_cli.mission_control import redact_text, redact_value


MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
ALLOWED_ATTACHMENT_TYPES: dict[str, set[str]] = {
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/plain"},
    ".log": {"text/plain"},
    ".json": {"application/json", "text/plain"},
    ".csv": {"text/csv", "application/vnd.ms-excel", "text/plain"},
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".webp": {"image/webp"},
}
BLOCKED_ATTACHMENT_EXTENSIONS = {
    ".bat",
    ".cmd",
    ".com",
    ".exe",
    ".html",
    ".js",
    ".mjs",
    ".ps1",
    ".sh",
    ".svg",
}

DEFAULT_ROOMS: tuple[dict[str, str], ...] = (
    {
        "title": "Tool & Tally",
        "project_key": "tool-tally",
        "description": "Local planning context for Tool & Tally.",
    },
    {
        "title": "Shorts pipeline",
        "project_key": "shorts-pipeline",
        "description": "Local notes for short-form video workflow.",
    },
    {
        "title": "Longform video",
        "project_key": "longform-video",
        "description": "Local notes for longform video workflow.",
    },
    {
        "title": "Signal Room",
        "project_key": "signal-room",
        "description": "Local signal and research context.",
    },
    {
        "title": "Hermes reliability",
        "project_key": "hermes-reliability",
        "description": "Local reliability and dashboard context.",
    },
    {
        "title": "General / Inbox",
        "project_key": "general-inbox",
        "description": "Local inbox for unsorted project material.",
    },
)

_LOCK = threading.RLock()


class ProjectRoomError(ValueError):
    """Raised for invalid Project Rooms requests."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_dir() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "state" / "mission-control" / "project-rooms"


def audit_path() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "state" / "mission-control" / "project-rooms-audit.jsonl"


def _rooms_path() -> Path:
    return state_dir() / "rooms.json"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(redact_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:64] or "room"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def _load_rooms_unlocked() -> list[dict[str, Any]]:
    path = _rooms_path()
    if not path.exists():
        rooms: list[dict[str, Any]] = []
        created_at = _now_iso()
        for item in DEFAULT_ROOMS:
            slug = _slugify(item["title"])
            rooms.append(
                {
                    "id": f"room_{slug.replace('-', '_')}",
                    "slug": slug,
                    "title": item["title"],
                    "project_key": item["project_key"],
                    "description": item["description"],
                    "trusted_for_execution": False,
                    "inert_context_only": True,
                    "created_at": created_at,
                    "updated_at": created_at,
                    "message_count": 0,
                    "attachment_count": 0,
                }
            )
        _atomic_write_json(path, {"rooms": rooms})
        return rooms
    data = json.loads(path.read_text(encoding="utf-8"))
    rooms = data.get("rooms") if isinstance(data, dict) else []
    if not isinstance(rooms, list):
        raise ProjectRoomError("rooms.json is invalid")
    return [room for room in rooms if isinstance(room, dict)]


def _save_rooms_unlocked(rooms: list[dict[str, Any]]) -> None:
    _atomic_write_json(_rooms_path(), {"rooms": rooms})


def _append_audit(event: str, **fields: Any) -> None:
    record = {
        "timestamp": _now_iso(),
        "event": event,
        "actor": "dashboard",
        "surface": "dashboard",
        "trusted_for_execution": False,
        "inert_context_only": True,
        **fields,
    }
    path = audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(redact_value(record), sort_keys=True) + "\n")


def _room_dir(room_id: str) -> Path:
    if not re.fullmatch(r"room_[a-zA-Z0-9_-]+", room_id):
        raise ProjectRoomError("Invalid room id")
    return state_dir() / room_id


def _messages_path(room_id: str) -> Path:
    return _room_dir(room_id) / "messages.json"


def _attachments_path(room_id: str) -> Path:
    return _room_dir(room_id) / "attachments.json"


def _attachments_dir(room_id: str) -> Path:
    return _room_dir(room_id) / "attachments"


def _load_json_list(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get(key) if isinstance(data, dict) else []
    if not isinstance(items, list):
        raise ProjectRoomError(f"{path.name} is invalid")
    return [item for item in items if isinstance(item, dict)]


def _find_room_unlocked(room_id: str) -> dict[str, Any]:
    for room in _load_rooms_unlocked():
        if room.get("id") == room_id or room.get("slug") == room_id:
            return room
    raise FileNotFoundError(room_id)


def _bump_room_counts_unlocked(room_id: str, *, messages: int = 0, attachments: int = 0) -> None:
    rooms = _load_rooms_unlocked()
    for room in rooms:
        if room.get("id") == room_id:
            room["message_count"] = int(room.get("message_count") or 0) + messages
            room["attachment_count"] = int(room.get("attachment_count") or 0) + attachments
            room["updated_at"] = _now_iso()
            _save_rooms_unlocked(rooms)
            return


def list_rooms() -> dict[str, Any]:
    with _LOCK:
        rooms = _load_rooms_unlocked()
    return {
        "generated_at": _now_iso(),
        "source": "mission_control_project_rooms",
        "source_refs": [str(_rooms_path())],
        "rooms": redact_value(rooms),
        "warnings": [],
    }


def create_room(data: dict[str, Any]) -> dict[str, Any]:
    title = str(data.get("title") or "").strip()
    if not title:
        raise ProjectRoomError("Missing required field: title")
    now = _now_iso()
    with _LOCK:
        rooms = _load_rooms_unlocked()
        existing_slugs = {str(room.get("slug") or "") for room in rooms}
        base_slug = _slugify(title)
        slug = base_slug
        index = 2
        while slug in existing_slugs:
            slug = f"{base_slug}-{index}"
            index += 1
        room = {
            "id": _new_id("room"),
            "slug": slug,
            "title": redact_text(title),
            "project_key": redact_text(str(data.get("project_key") or slug).strip()),
            "description": redact_text(str(data.get("description") or "").strip()),
            "trusted_for_execution": False,
            "inert_context_only": True,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "attachment_count": 0,
        }
        rooms.append(room)
        _save_rooms_unlocked(rooms)
        _append_audit("room_created", room_id=room["id"], room_slug=room["slug"], title=room["title"])
    return redact_value(room)


def list_messages(room_id: str) -> dict[str, Any]:
    with _LOCK:
        room = _find_room_unlocked(room_id)
        messages = _load_json_list(_messages_path(str(room["id"])), "messages")
    return {
        "generated_at": _now_iso(),
        "source": "mission_control_project_room_messages",
        "room": redact_value(room),
        "messages": redact_value(messages),
        "warnings": [],
    }


def add_message(room_id: str, data: dict[str, Any]) -> dict[str, Any]:
    content = str(data.get("content_text") or "").strip()
    if not content:
        raise ProjectRoomError("Missing required field: content_text")
    now = _now_iso()
    with _LOCK:
        room = _find_room_unlocked(room_id)
        canonical_id = str(room["id"])
        messages = _load_json_list(_messages_path(canonical_id), "messages")
        message = {
            "id": _new_id("msg"),
            "room_id": canonical_id,
            "author": redact_text(str(data.get("author") or "dashboard")),
            "role": redact_text(str(data.get("role") or "note")),
            "content_type": redact_text(str(data.get("content_type") or "text")),
            "content_text": redact_text(content),
            "source_refs": redact_value(data.get("source_refs") or []),
            "linked_packet_ids": redact_value(data.get("linked_packet_ids") or []),
            "trusted_for_execution": False,
            "inert_context_only": True,
            "created_at": now,
        }
        messages.insert(0, message)
        _atomic_write_json(_messages_path(canonical_id), {"messages": messages})
        _bump_room_counts_unlocked(canonical_id, messages=1)
        _append_audit("message_added", room_id=canonical_id, message_id=message["id"], role=message["role"])
    return redact_value(message)


def _validate_filename(filename: str) -> str:
    clean = filename.strip()
    if not clean or Path(clean).name != clean or "/" in clean or "\\" in clean:
        raise ProjectRoomError("Unsafe attachment filename")
    suffix = Path(clean).suffix.lower()
    if suffix in BLOCKED_ATTACHMENT_EXTENSIONS or suffix not in ALLOWED_ATTACHMENT_TYPES:
        raise ProjectRoomError("Attachment type is not allowed")
    return clean


def _validate_attachment(filename: str, mime_type: str, content: bytes) -> None:
    if len(content) > MAX_ATTACHMENT_BYTES:
        raise OverflowError("Attachment exceeds maximum size")
    suffix = Path(filename).suffix.lower()
    allowed_mimes = ALLOWED_ATTACHMENT_TYPES.get(suffix, set())
    if mime_type not in allowed_mimes:
        raise ProjectRoomError("Attachment MIME type is not allowed")


def add_attachment(room_id: str, *, filename: str, mime_type: str, content: bytes) -> dict[str, Any]:
    safe_name = _validate_filename(filename)
    clean_mime = (mime_type or "application/octet-stream").split(";", 1)[0].strip().lower()
    _validate_attachment(safe_name, clean_mime, content)
    digest = hashlib.sha256(content).hexdigest()
    now = _now_iso()
    with _LOCK:
        room = _find_room_unlocked(room_id)
        canonical_id = str(room["id"])
        attachment_id = _new_id("att")
        suffix = Path(safe_name).suffix.lower()
        storage_dir = _attachments_dir(canonical_id)
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_path = storage_dir / f"{attachment_id}{suffix}"
        storage_path.write_bytes(content)
        attachments = _load_json_list(_attachments_path(canonical_id), "attachments")
        attachment = {
            "id": attachment_id,
            "room_id": canonical_id,
            "original_filename": redact_text(safe_name),
            "mime_type": clean_mime,
            "size_bytes": len(content),
            "sha256": digest,
            "trusted_for_execution": False,
            "inert_context_only": True,
            "created_at": now,
        }
        record = {**attachment, "storage_filename": storage_path.name}
        attachments.insert(0, record)
        _atomic_write_json(_attachments_path(canonical_id), {"attachments": attachments})
        _bump_room_counts_unlocked(canonical_id, attachments=1)
        _append_audit("attachment_added", room_id=canonical_id, attachment_id=attachment_id, original_filename=safe_name, sha256=digest)
    return redact_value(attachment)


def get_attachment(room_id: str, attachment_id: str) -> dict[str, Any]:
    with _LOCK:
        room = _find_room_unlocked(room_id)
        for attachment in _load_json_list(_attachments_path(str(room["id"])), "attachments"):
            if attachment.get("id") == attachment_id:
                public = dict(attachment)
                public.pop("storage_filename", None)
                return redact_value(public)
    raise FileNotFoundError(attachment_id)


def attachment_file_path(room_id: str, attachment_id: str) -> tuple[Path, dict[str, Any]]:
    with _LOCK:
        room = _find_room_unlocked(room_id)
        canonical_id = str(room["id"])
        for attachment in _load_json_list(_attachments_path(canonical_id), "attachments"):
            if attachment.get("id") == attachment_id:
                filename = str(attachment.get("storage_filename") or "")
                path = _attachments_dir(canonical_id) / filename
                root = _attachments_dir(canonical_id).resolve()
                resolved = path.resolve()
                if not resolved.is_relative_to(root) or not resolved.is_file():
                    raise FileNotFoundError(attachment_id)
                public = dict(attachment)
                public.pop("storage_filename", None)
                return resolved, public
    raise FileNotFoundError(attachment_id)
