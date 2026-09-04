"""Messaging-facing controls for gateway-hosted Bot rooms.

The handlers in :mod:`gateway.slash_commands` deliberately stay thin.  This
module owns parsing, room lookup, bounded presentation, and server-owned actor
identity so Telegram, Signal, WhatsApp, and the other gateway transports share
one behavior contract.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_room_controls, hosted_room_links
from gateway import hosted_rooms
from gateway.hosted_room_control_client import (
    RoomControlClientError,
    RoomControlHTTPClient,
)


MAX_ROOM_CHOICES = 8
MAX_RECENT_MESSAGES = 5
MAX_PREVIEW_CHARS = 180
MAX_GROUP_MEMBERS = 6
MAX_MESSAGING_ROOMS = 4096
GROUP_CHAT_SYNC_META_KEY = "hermes-bots-groups"


def _projected_desktop_rooms(*, profile: str = "default") -> list[dict[str, Any]]:
    """Read the bounded classic-room projection shared by Desktop clients."""

    try:
        import yaml
        from hermes_cli.profiles import get_profile_dir

        profile_meta = Path(get_profile_dir(profile)) / "profile.yaml"
        if not profile_meta.is_file():
            return []
        raw = yaml.safe_load(profile_meta.read_text(encoding="utf-8")) or {}
        ui_meta = raw.get("ui_meta") if isinstance(raw, Mapping) else None
        snapshot = (
            ui_meta.get(GROUP_CHAT_SYNC_META_KEY)
            if isinstance(ui_meta, Mapping)
            else None
        )
        raw_rooms = snapshot.get("rooms") if isinstance(snapshot, Mapping) else None
    except Exception:
        return []
    if not isinstance(raw_rooms, Mapping):
        return []
    try:
        snapshot_version = int(snapshot.get("version") or 0)
    except (TypeError, ValueError):
        snapshot_version = 0

    rooms: list[dict[str, Any]] = []
    for key, raw_room in raw_rooms.items():
        if not isinstance(raw_room, Mapping):
            continue
        hosted = raw_room.get("hosted")
        if isinstance(hosted, str) and hosted.strip():
            continue
        name = _clean_line(raw_room.get("name") or key, limit=200)
        explicit_room_id = str(raw_room.get("roomId") or "").strip()
        room_id = explicit_room_id or (
            str(key).strip() if snapshot_version >= 3 else f"name:{name}"
        )
        if (
            not name
            or not room_id
            or len(room_id) > 200
            or any(char in room_id for char in ("\x00", "\r", "\n"))
        ):
            continue
        authority_hash = str(raw_room.get("desktopAuthorityHash") or "").strip().lower()
        if not re.fullmatch(r"[a-f0-9]{64}", authority_hash):
            authority_hash = ""
        raw_log = raw_room.get("log")
        log = (
            [dict(item) for item in raw_log if isinstance(item, Mapping)]
            if isinstance(raw_log, list)
            else []
        )
        raw_members = raw_room.get("members")
        members = (
            [dict(item) for item in raw_members if isinstance(item, Mapping)]
            if isinstance(raw_members, list)
            else []
        )
        updated_at = max(
            (float(item.get("at") or 0) for item in log),
            default=float(snapshot.get("updatedAt") or 0) / 1000,
        )
        rooms.append({
            "room_id": room_id,
            "name": name,
            "members": members,
            "log": log,
            "created_at": min(
                (float(item.get("at") or 0) for item in log),
                default=updated_at,
            ),
            "updated_at": updated_at,
            "desktop_authority_hash": authority_hash,
            "_room_mode": "desktop",
        })
    return rooms


def _room_profiles(room: Mapping[str, Any]) -> set[str]:
    profiles: set[str] = set()
    members = room.get("members")
    if not isinstance(members, list):
        return profiles
    for member in members:
        if not isinstance(member, Mapping):
            continue
        profile = str(
            member.get("target_profile")
            or member.get("profile")
            or member.get("member_id")
            or ""
        ).strip()
        if profile:
            profiles.add(profile)
        target = member.get("target")
        if isinstance(target, Mapping):
            target_profile = str(
                target.get("target_profile") or target.get("profile") or ""
            ).strip()
            if target_profile:
                profiles.add(target_profile)
    return profiles


def list_messaging_rooms(
    service: Any,
    *,
    profile: str = "default",
) -> list[dict[str, Any]]:
    """Return hosted and reachable classic rooms with stable short numbers."""

    hosted: list[dict[str, Any]] = []
    offset = 0
    while offset < MAX_MESSAGING_ROOMS:
        page = hosted_rooms.list_rooms(
            service.db_path,
            limit=hosted_rooms.MAX_ROOM_LIST_LIMIT,
            offset=offset,
        )
        hosted.extend({**room, "_room_mode": "hosted"} for room in page)
        if len(page) < hosted_rooms.MAX_ROOM_LIST_LIMIT:
            break
        offset += len(page)
    if len(hosted) >= MAX_MESSAGING_ROOMS:
        raise RoomControlError(
            "There are too many Group Chats to list safely. Disband inactive chats and try again."
        )
    # The default profile is the installation owner's master chat. Secondary
    # profiles see only rooms whose frozen roster includes that profile.
    if profile != "default":
        hosted = [room for room in hosted if profile in _room_profiles(room)]
    hosted_ids = {str(room["room_id"]) for room in hosted}
    remote: list[dict[str, Any]] = []
    remote_ids: set[str] = set()
    try:
        peer_links = hosted_room_controls.load_peer_control_links(
            service.db_path,
            limit=hosted_room_controls.MAX_LOAD_LINKS,
        ).links
    except hosted_room_controls.HostedRoomControlError:
        peer_links = ()
    for link in peer_links:
        if link.room_id in hosted_ids or link.room_id in remote_ids:
            continue
        if profile != "default" and not hosted_room_controls.peer_reservation_matches(
            service.db_path,
            room_id=link.room_id,
            member_id=link.member_id,
            target_profile=profile,
            authority_gateway_id=link.authority_gateway_id,
            authority_epoch=link.authority_epoch,
        ):
            continue
        remote_ids.add(link.room_id)
        remote.append({
            "room_id": link.room_id,
            "name": link.room_name,
            "members": [],
            "member_count": link.member_count,
            "authority_gateway_id": link.authority_gateway_id,
            "authority_epoch": link.authority_epoch,
            "created_at": link.created_at,
            "updated_at": link.updated_at,
            "_remote_member_id": link.member_id,
            "_room_mode": "remote",
        })
    desktop = [
        room
        for room in _projected_desktop_rooms(profile=profile)
        if str(room["room_id"]) not in hosted_ids | remote_ids
    ]
    rooms = hosted + remote + desktop
    if not rooms:
        return []
    if len(rooms) > MAX_MESSAGING_ROOMS:
        raise RoomControlError(
            "There are too many Group Chats to list safely. Disband "
            "inactive chats and try again."
        )

    from gateway.desktop_room_mailbox import (
        MAX_ROOM_IDS,
        available_room_ids,
        default_db_path,
        failed_command_counts,
        latest_command_states,
        register_projected_authorities,
    )

    mailbox_db = default_db_path()
    desktop_ids = [room["room_id"] for room in desktop]
    commitments = [
        {
            "room_id": room["room_id"],
            "authority_hash": room["desktop_authority_hash"],
        }
        for room in desktop
        if room.get("desktop_authority_hash")
    ]
    for index in range(0, len(commitments), MAX_ROOM_IDS):
        register_projected_authorities(
            mailbox_db,
            commitments[index : index + MAX_ROOM_IDS],
        )
    available = available_room_ids(mailbox_db, desktop_ids)
    command_states = latest_command_states(mailbox_db, desktop_ids)
    failed_counts = failed_command_counts(mailbox_db, desktop_ids)
    rooms = [
        {
            **room,
            **(
                {
                    "desktop_available": str(room["room_id"]) in available,
                    "desktop_command": command_states.get(str(room["room_id"])),
                    "desktop_failed_commands": failed_counts.get(
                        str(room["room_id"]), 0
                    ),
                }
                if room.get("_room_mode") == "desktop"
                else {}
            ),
        }
        for room in rooms
    ]

    # Keep the numeric reference ledger where #96274 first created it. Moving
    # it to the compatibility mailbox would silently renumber existing group
    # chats after upgrade, making `/group 2 send ...` target the wrong chat.
    db_path = Path(service.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="state.db (room messaging refs)")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS hosted_room_messaging_refs (
                room_ref INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id TEXT NOT NULL UNIQUE
            )"""
        )
        # Existing rooms receive deterministic first-use numbers. AUTOINCREMENT
        # keeps those numbers stable and prevents a disbanded room's reference
        # from being reassigned to unrelated work later.
        for room in sorted(
            rooms,
            key=lambda item: (
                float(item.get("created_at") or 0),
                str(item.get("room_id") or ""),
            ),
        ):
            conn.execute(
                "INSERT OR IGNORE INTO hosted_room_messaging_refs (room_id) VALUES (?)",
                (str(room["room_id"]),),
            )
        refs = {
            str(row["room_id"]): int(row["room_ref"])
            for row in conn.execute(
                "SELECT room_id, room_ref FROM hosted_room_messaging_refs"
            )
        }
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return [{**room, "messaging_ref": refs[str(room["room_id"])]} for room in rooms]


def room_reference(room: Mapping[str, Any]) -> str:
    """Return the short messaging reference, with an internal-id fallback."""

    reference = room.get("messaging_ref")
    if isinstance(reference, int) and reference > 0:
        return str(reference)
    return str(room.get("room_id") or "")


def _frozen_desktop_recipients(room: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the bounded routing identity visible in the trusted projection."""

    raw_members = room.get("members")
    if not isinstance(raw_members, list):
        return []
    recipients: list[dict[str, Any]] = []
    for raw in raw_members[:MAX_GROUP_MEMBERS]:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "").strip()[:128]
        if not name:
            continue
        recipient: dict[str, Any] = {"name": name}
        for key, limit in (
            ("handle", 128),
            ("connectionId", 128),
            ("connectionKind", 64),
            ("connectionLabel", 128),
        ):
            value = str(raw.get(key) or "").strip()
            if value:
                recipient[key] = value[:limit]
        if raw.get("sourceScoped") is True:
            recipient["sourceScoped"] = True
        recipients.append(recipient)
    if len(recipients) < 2:
        raise RoomControlError(
            "This Group Chat’s Bot list is incomplete. Open it in Hermes Desktop and try again."
        )
    return recipients


def _latest_projected_stop_target(room: Mapping[str, Any]) -> tuple[str, str]:
    raw_log = room.get("log")
    if not isinstance(raw_log, list):
        return "", ""
    for entry in reversed(raw_log):
        if not isinstance(entry, Mapping):
            continue
        actor = entry.get("from")
        if not isinstance(actor, Mapping) or actor.get("kind") != "user":
            continue
        thread = str(entry.get("thread") or "").strip()
        if thread:
            message_id = str(entry.get("eventId") or entry.get("id") or "").strip()
            return thread[:128], message_id[:160]
    return "", ""


class RoomControlError(ValueError):
    """A user-actionable hosted-room command error."""


def _room_member_count(room: Mapping[str, Any]) -> int:
    value = room.get("member_count")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    members = room.get("members")
    return len(members) if isinstance(members, list) else 0


def _room_status_icon(status: str) -> str:
    lowered = str(status or "").casefold()
    if any(
        word in lowered
        for word in ("attention", "blocked", "error", "offline", "unavailable")
    ):
        return "⚠️"
    if "connected" in lowered:
        return "⚪"
    if any(
        word in lowered
        for word in ("queued", "running", "stopping", "waiting", "working")
    ):
        return "🟡"
    return "🟢"


def _picker_display_label(value: Any, *, limit: int) -> str:
    """Neutralize rich markup and notification-shaped @mentions in pickers."""

    return _plain_display_label(value, limit=limit)


def room_picker_choices(
    service: Any,
    rooms: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build one bounded native-picker page without exposing room internals."""

    choices: list[dict[str, Any]] = []
    for room in sorted(
        rooms,
        key=lambda item: (
            -float(item.get("updated_at") or item.get("created_at") or 0),
            int(item.get("messaging_ref") or 0),
        ),
    )[:MAX_ROOM_CHOICES]:
        reference = room_reference(room)
        name = _clean_line(room.get("name") or room.get("room_id"), limit=42)
        status = _room_status(service, room)
        count = _room_member_count(room)
        choices.append({
            "value": _room_picker_value(room),
            "label": (
                f"{_room_status_icon(status)} {reference}. "
                f"{_picker_display_label(name, limit=42)} ({count})"
            ),
            "full_width": True,
            "is_current": False,
        })
    return choices


def _room_picker_value(room: Mapping[str, Any]) -> str:
    seed = ":".join((
        str(room.get("_room_mode") or "hosted"),
        str(room.get("connection_id") or room.get("_connection_id") or ""),
        str(room.get("room_id") or ""),
    ))
    return f"room-{hashlib.sha256(seed.encode()).hexdigest()[:16]}"


def resolve_room_picker_choice(
    rooms: list[Mapping[str, Any]],
    value: str,
) -> Mapping[str, Any]:
    """Resolve an exact native-picker room token without number reuse."""

    selected = [room for room in rooms if _room_picker_value(room) == str(value)]
    if len(selected) != 1:
        raise RoomControlError(
            "This Group Chat is no longer available. Run the command again."
        )
    return selected[0]


def _room_participant_lines(room: Mapping[str, Any]) -> list[str]:
    raw_members = room.get("members")
    if not isinstance(raw_members, list):
        return []
    lines: list[str] = []
    for raw in raw_members[:MAX_GROUP_MEMBERS]:
        if not isinstance(raw, Mapping):
            continue
        name = _room_member_name(raw)
        handle = _room_member_handle(raw)
        suffix = f" (`@{handle}`)" if handle else ""
        lines.append(f"• {_plain_display_label(name)}{suffix}")
    return lines


def _room_member_name(member: Mapping[str, Any]) -> str:
    return _clean_line(
        member.get("display_name")
        or member.get("displayName")
        or member.get("name")
        or member.get("handle")
        or "Bot",
        limit=48,
    )


def _room_member_handle(member: Mapping[str, Any]) -> str:
    handle = _clean_line(
        member.get("handle"),
        limit=driver.MAX_IDENTIFIER_CHARS,
    ).lstrip("@")
    return handle if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", handle) else ""


def _room_member_picker_value(
    room: Mapping[str, Any],
    member: Mapping[str, Any],
) -> str:
    seed = json.dumps(
        [
            str(room.get("room_id") or ""),
            str(member.get("member_id") or ""),
            _room_member_handle(member),
            _room_member_name(member),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"p={hashlib.sha256(seed.encode()).hexdigest()[:16]}"


def _room_display_members(
    service: Any, room: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    if room.get("_room_mode") == "remote":
        summary = _remote_summary(service, room)
        remote_room = summary.get("room")
        raw_members = (
            remote_room.get("members") if isinstance(remote_room, Mapping) else None
        )
    else:
        raw_members = room.get("members")
    return [
        member
        for member in (raw_members if isinstance(raw_members, list) else [])
        if isinstance(member, Mapping)
    ][:MAX_GROUP_MEMBERS]


def _room_with_messaging_reference(
    service: Any,
    room: Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(room.get("messaging_ref"), int):
        return room
    room_id = str(room.get("room_id") or "")
    return next(
        (
            candidate
            for candidate in list_messaging_rooms(service)
            if str(candidate.get("room_id") or "") == room_id
        ),
        room,
    )


def room_bot_picker_choices(
    service: Any,
    room: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build a native participant picker without exposing profile internals."""

    room = _room_with_messaging_reference(service, room)
    choices: list[dict[str, Any]] = []
    for member in _room_display_members(service, room):
        name = _room_member_name(member)
        handle = _room_member_handle(member)
        choices.append({
            "value": _room_member_picker_value(room, member),
            "label": f"🤖 {_picker_display_label(name, limit=48)}"
            + (f" · {handle}" if handle else ""),
            "full_width": True,
            "is_current": False,
        })
    return choices


def format_room_bot_list(
    service: Any,
    room: Mapping[str, Any],
    *,
    room_command: str = "/group",
) -> str:
    """Render a bounded participant list with one stable in-roster number."""

    room = _room_with_messaging_reference(service, room)
    members = _room_display_members(service, room)
    name = _clean_line(room.get("name") or room.get("room_id"), limit=72)
    reference = room_reference(room)
    lines = [f"🤖 **Bots in {_plain_display_label(name, limit=72)}**"]
    for index, member in enumerate(members, start=1):
        member_name = _room_member_name(member)
        handle = _room_member_handle(member)
        lines.append(
            f"{index}. **{_plain_display_label(member_name)}**"
            + (f" · `@{handle}`" if handle else "")
        )
    if not members:
        lines.append("No Bots are available in this Group Chat.")
    lines.extend([
        "",
        "────────",
        "🧭 **Controls**",
        f"Bot details: `{room_command} {reference} bot <number>`",
        f"Back to Group Chat: `{room_command} {reference}`",
    ])
    return "\n".join(lines)


def format_room_bot_detail(
    service: Any,
    room: Mapping[str, Any],
    bot_query: str,
    *,
    room_command: str = "/group",
) -> str:
    """Show one participant and only the controls the room contract supports."""

    room = _room_with_messaging_reference(service, room)
    members = _room_display_members(service, room)
    raw_query = str(bot_query or "").strip()
    normalized = raw_query.lstrip("@").casefold()
    selected: Mapping[str, Any] | None = None
    if raw_query.startswith("p="):
        token_matches = [
            member
            for member in members
            if _room_member_picker_value(room, member) == raw_query
        ]
        if len(token_matches) == 1:
            selected = token_matches[0]
    elif not raw_query.startswith("@") and normalized.isdecimal():
        index = int(normalized)
        if 1 <= index <= len(members):
            selected = members[index - 1]
    else:
        matches = [
            member
            for member in members
            if normalized
            in {
                _room_member_handle(member).casefold(),
                _room_member_name(member).casefold(),
                _room_member_picker_value(room, member).casefold(),
            }
        ]
        if len(matches) == 1:
            selected = matches[0]
    if selected is None:
        raise RoomControlError(
            "No Bot in this Group Chat matches that number or handle."
        )

    room_name = _clean_line(room.get("name") or room.get("room_id"), limit=72)
    reference = room_reference(room)
    member_name = _room_member_name(selected)
    handle = _room_member_handle(selected)
    lines = [
        f"🤖 **{_plain_display_label(member_name)}**",
        f"Group Chat: {_plain_display_label(room_name, limit=72)}",
    ]
    if handle:
        lines.append(f"Handle: `@{handle}`")
    lines.extend(["", "────────", "🧭 **Controls**"])
    if handle:
        lines.append(
            f"Message this Bot: `{room_command} {reference} send @{handle} <message>`"
        )
    lines.extend([
        f"All Bots: `{room_command} {reference} bots`",
        f"Back to Group Chat: `{room_command} {reference}`",
    ])
    return "\n".join(lines)


def _room_has_targetable_handle(room: Mapping[str, Any]) -> bool:
    raw_members = room.get("members")
    if not isinstance(raw_members, list):
        return False
    return any(
        isinstance(raw, Mapping)
        and bool(
            re.sub(
                r"[^A-Za-z0-9_.-]",
                "",
                _clean_line(raw.get("handle"), limit=48).lstrip("@"),
            )
        )
        for raw in raw_members[:MAX_GROUP_MEMBERS]
    )


def _remote_control_link(service: Any, room: Mapping[str, Any]):
    room_id = str(room.get("room_id") or "")
    member_id = str(room.get("_remote_member_id") or "")
    try:
        links = hosted_room_controls.load_peer_control_links(
            service.db_path,
            limit=hosted_room_controls.MAX_LOAD_LINKS,
        ).links
    except hosted_room_controls.HostedRoomControlError as exc:
        raise RoomControlError(
            "This Group Chat connection needs repair. Open it in Hermes Desktop "
            "on a connected device."
        ) from exc
    link = next(
        (
            candidate
            for candidate in links
            if candidate.room_id == room_id and candidate.member_id == member_id
        ),
        None,
    )
    if link is None:
        raise RoomControlError(
            "This Group Chat isn’t available here. Open it in Hermes Desktop or "
            "another connected Hermes chat."
        )
    return link


def _remote_error(exc: RoomControlClientError) -> RoomControlError:
    if exc.status_code == 400 and exc.user_message:
        return RoomControlError(exc.user_message)
    if exc.status_code in {401, 403, 404}:
        return RoomControlError(
            "This Group Chat isn’t available here anymore. Open it in Hermes "
            "Desktop or another connected Hermes chat."
        )
    return RoomControlError(
        "This Group Chat can’t be reached right now. Make sure the devices running "
        "its Bots are online, then try again."
    )


def _remote_summary(service: Any, room: Mapping[str, Any]) -> dict[str, Any]:
    link = _remote_control_link(service, room)
    try:
        summary = RoomControlHTTPClient(link).summary()
    except RoomControlClientError as exc:
        raise _remote_error(exc) from exc
    raw_room = summary.get("room")
    if not isinstance(raw_room, Mapping):
        raise RoomControlError("This Group Chat returned invalid status data.")
    try:
        authority_epoch = int(raw_room.get("authority_epoch") or 0)
    except (TypeError, ValueError) as exc:
        raise RoomControlError("This Group Chat returned invalid status data.") from exc
    if (
        str(raw_room.get("room_id") or "") != link.room_id
        or str(raw_room.get("authority_gateway_id") or "") != link.authority_gateway_id
        or authority_epoch != link.authority_epoch
    ):
        raise RoomControlError("This Group Chat returned mismatched status data.")
    raw_members = raw_room.get("members")
    if not isinstance(raw_members, list) or not 1 <= len(raw_members) <= 64:
        raise RoomControlError("This Group Chat returned invalid member data.")
    hosted_room_controls.update_peer_control_metadata(
        service.db_path,
        room_id=link.room_id,
        member_id=link.member_id,
        authority_gateway_id=link.authority_gateway_id,
        authority_epoch=link.authority_epoch,
        room_name=raw_room.get("name") or link.room_name,
        member_count=len(raw_members),
    )
    return summary


def remote_mutate(
    service: Any,
    room: Mapping[str, Any],
    *,
    action: str,
    command_id: str,
    text: str = "",
    actor_display_name: str = "Messaging",
) -> dict[str, Any]:
    link = _remote_control_link(service, room)
    try:
        return RoomControlHTTPClient(link).mutate(
            action=action,
            command_id=command_id,
            text=text,
            actor_display_name=actor_display_name,
        )
    except RoomControlClientError as exc:
        raise _remote_error(exc) from exc


def _retry_receipt_plan(
    db_path: Path,
    *,
    command_id: str,
    room_id: str,
    actor: Mapping[str, Any],
    task_ids: list[str],
) -> tuple[list[str], str | None]:
    """Freeze one transport delivery to one bounded retry decision."""
    from gateway.hosted_room_messaging_retries import (
        MessagingRetryReceiptError,
        retry_receipt_plan,
    )

    try:
        return retry_receipt_plan(
            db_path,
            command_id=command_id,
            room_id=room_id,
            actor=actor,
            task_ids=task_ids,
        )
    except MessagingRetryReceiptError as exc:
        raise RoomControlError(str(exc)) from exc


def _complete_retry_receipt(db_path: Path, *, command_id: str, result: str) -> None:
    from gateway.hosted_room_messaging_retries import complete_retry_receipt

    complete_retry_receipt(db_path, command_id=command_id, result=result)


@dataclass(frozen=True)
class RoomCommand:
    """Parsed mutating ``/group`` subcommand."""

    action: str
    room_query: str
    message: str = ""


class MessagingRoomBackend:
    """Cross-process hosted-room access through the shared durable stores."""

    def __init__(self, *, db_path: Any, service: Any = None) -> None:
        self.db_path = db_path
        self.service = service

    def status(self, room_id: str) -> dict[str, Any]:
        from gateway import hosted_room_messaging_approvals as approvals

        if self.service is not None:
            status = self.service.status(room_id)
            peer_needs_attention = any(
                str(route.get("status") or "") == "needs_reauthorization"
                for route in status.get("peer_routes", [])
                if isinstance(route, Mapping)
            )
            actions = [
                action
                for action in status.get("pending_actions", [])
                if isinstance(action, Mapping) and action.get("kind") != "approval"
            ]
            actions.extend(
                approvals.list_pending_approvals(self.db_path, room_id=room_id)
            )
            return {
                **status,
                "blocked": bool(status.get("blocked") or peer_needs_attention),
                "pending_actions": actions,
            }
        tasks = driver.list_tasks(self.db_path, room_id=room_id)
        counts: dict[str, int] = {}
        for task in tasks:
            status = str(task.get("status") or "")
            counts[status] = counts.get(status, 0) + 1
        try:
            peer_needs_attention = any(
                link.room_id == room_id and link.status == "needs_reauthorization"
                for link in hosted_room_links.load_room_links(self.db_path)
            )
        except Exception:
            peer_needs_attention = True
        return {
            "working": any(
                counts.get(status) for status in ("queued", "running", "stopping")
            ),
            "blocked": bool(counts.get("indeterminate") or peer_needs_attention),
            "counts": counts,
            "pending_actions": [
                {"kind": "retry", "task_id": task["identity"].task_id}
                for task in tasks
                if task.get("status") in {"deferred", "indeterminate"}
            ]
            + approvals.list_pending_approvals(self.db_path, room_id=room_id),
        }

    def send(
        self,
        *,
        room_id: str,
        event_id: str,
        payload: Any,
        actor: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.service is not None:
            send_server_owned = getattr(self.service, "send_server_owned", None)
            if callable(send_server_owned):
                return send_server_owned(
                    room_id=room_id,
                    event_id=event_id,
                    payload=payload,
                    actor=actor,
                )
            return self.service.send(
                room_id=room_id,
                event_id=event_id,
                payload=payload,
                actor=actor,
            )
        room = hosted_rooms.room_state(self.db_path, room_id=room_id)
        local_gateway_id = hosted_rooms.local_authority_gateway_id()
        if str(room["authority_gateway_id"]) != local_gateway_id:
            raise hosted_rooms.AuthorityConflictError(
                "This Group Chat moved to another connected device. Open it there and try again."
            )
        normalized = discussion.validate_user_payload(payload)
        return hosted_rooms.append_event(
            self.db_path,
            room_id=room_id,
            event_id=event_id,
            kind="message.user",
            actor=dict(actor),
            payload=normalized,
            authority_gateway_id=str(room["authority_gateway_id"]),
            authority_epoch=int(room["authority_epoch"]),
        )

    def stop_room(self, room_id: str, *, cancel_id: str) -> int:
        if self.service is not None:
            return self.service.stop_room(room_id, cancel_id=cancel_id)
        room = hosted_rooms.room_state(self.db_path, room_id=room_id)
        local_gateway_id = hosted_rooms.local_authority_gateway_id()
        if str(room["authority_gateway_id"]) != local_gateway_id:
            raise hosted_rooms.AuthorityConflictError(
                "This Group Chat moved to another connected device. Open it there and try again."
            )
        stop_event = hosted_rooms.request_room_stop(
            self.db_path,
            room_id=room_id,
            cancel_id=cancel_id,
            expected_gateway_id=local_gateway_id,
            expected_epoch=int(room["authority_epoch"]),
        )
        stop_seq = int(stop_event["seq"])
        requested = 0
        for task in driver.list_tasks(self.db_path, room_id=room_id):
            if int(task["payload"]["source_event_seq"]) >= stop_seq:
                continue
            for _attempt in range(3):
                current = driver.get_task(self.db_path, task["identity"])
                status = str(current.get("status") or "")
                try:
                    if status == "queued":
                        driver.cancel_task(
                            self.db_path,
                            current["identity"],
                            cancel_id=cancel_id,
                            expected_cancel_generation=int(
                                current["cancel_generation"]
                            ),
                            clock=time.time,
                        )
                        requested += 1
                    elif status in {"running", "indeterminate", "deferred"}:
                        driver.begin_task_cancel(
                            self.db_path,
                            current["identity"],
                            cancel_id=cancel_id,
                            expected_cancel_generation=int(
                                current["cancel_generation"]
                            ),
                            clock=time.time,
                        )
                        requested += 1
                    elif status == "stopping":
                        requested += 1
                    elif (
                        status == "cancelled" and current.get("cancel_id") == cancel_id
                    ):
                        requested += 1
                    break
                except (driver.InvalidTaskTransitionError, driver.StaleTaskError):
                    # Queued work can become running between the list and the
                    # write. Reload and retry under the new generation.
                    continue
        return requested

    def retry_room_task(
        self,
        room_id: str,
        *,
        task_id: str,
        retry_id: str | None = None,
    ) -> dict[str, Any]:
        if self.service is None:
            raise RoomControlError(
                "Retry is available when the device running this Group Chat is online."
            )
        return self.service.retry_room_task(
            room_id,
            task_id=task_id,
            retry_id=retry_id,
        )


def current_room_backend() -> MessagingRoomBackend:
    """Resolve in-process service access or the shared cross-process store."""

    from tui_gateway.methods_groups import get_hosted_room_service

    service = get_hosted_room_service()
    return MessagingRoomBackend(
        db_path=service.db_path
        if service is not None
        else hosted_rooms.default_db_path(),
        service=service,
    )


def _clean_line(value: Any, *, limit: int = MAX_PREVIEW_CHARS) -> str:
    """Collapse untrusted text to one bounded display line."""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _plain_display_label(value: Any, *, limit: int = MAX_PREVIEW_CHARS) -> str:
    """Neutralize markup-shaped labels before placing them in rich layouts."""

    text = _clean_line(value, limit=limit)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[\\`*_{}\[\]#|>~]", "", text).strip()
    return text.replace("@", "＠") or "Unnamed"


def _plain_preview_text(value: Any, *, limit: int = MAX_PREVIEW_CHARS) -> str:
    """Neutralize preview markup without deleting message content."""

    text = _clean_line(value, limit=limit)
    return text.translate(
        str.maketrans({
            "@": "＠",
            "\\": "＼",
            "`": "｀",
            "*": "＊",
            "_": "＿",
            "{": "｛",
            "}": "｝",
            "[": "［",
            "]": "］",
            "#": "＃",
            "|": "｜",
            ">": "＞",
            "~": "～",
        })
    )


def parse_room_command(args: str, *, command_root: str = "/group") -> RoomCommand:
    """Parse the number-first send/retry/stop grammar used by messaging clients."""

    raw = str(args or "").strip()
    entity_first = raw.split(maxsplit=2)
    if len(entity_first) < 2 or not entity_first[0].isdecimal():
        raise RoomControlError(
            f"Use `{command_root} <number> send <message>`, "
            f"`{command_root} <number> retry`, or `{command_root} <number> stop`."
        )
    room_query = entity_first[0]
    action = entity_first[1].casefold()
    remainder = entity_first[2].strip() if len(entity_first) == 3 else ""
    if action == "send":
        message = remainder
        if not message:
            raise RoomControlError(f"Use `{command_root} <number> send <message>`.")
        return RoomCommand("send", room_query, message)
    if action == "stop":
        if remainder:
            raise RoomControlError(f"Use `{command_root} <number> stop`.")
        return RoomCommand("stop", room_query)
    if action == "retry":
        if remainder:
            raise RoomControlError(f"Use `{command_root} <number> retry`.")
        return RoomCommand("retry", room_query)
    if action in {"approve", "deny"}:
        if remainder and re.fullmatch(r"[A-Za-z0-9]{6,16}", remainder) is None:
            raise RoomControlError(
                f"Use `{command_root} <number> {action} [approval code]`."
            )
        return RoomCommand(action, room_query, remainder)
    raise RoomControlError(
        f"Use `{command_root} <number> send <message>`, "
        f"`{command_root} <number> approve`, `{command_root} <number> deny`, "
        f"`{command_root} <number> retry`, or `{command_root} <number> stop`."
    )


def resolve_room(rooms: list[dict[str, Any]], query: str) -> dict[str, Any]:
    """Resolve by stable messaging number, then id/name convenience matches."""

    needle = _clean_line(query, limit=hosted_rooms.MAX_ROOM_NAME_CHARS).casefold()
    if not needle:
        raise RoomControlError("Enter a Group Chat number or name.")

    if needle.isdecimal():
        numeric_ref = int(needle)
        matches = [room for room in rooms if room.get("messaging_ref") == numeric_ref]
        if len(matches) == 1:
            return matches[0]
        raise RoomControlError(f"No Group Chat is numbered {numeric_ref}.")

    if needle.startswith("id:"):
        internal_id = needle.removeprefix("id:")
        matches = [
            room
            for room in rooms
            if str(room.get("room_id") or "").casefold() == internal_id
        ]
        if len(matches) == 1:
            return matches[0]
        raise RoomControlError("No Group Chat matches that internal ID.")

    def _keys(room: Mapping[str, Any]) -> tuple[str, str]:
        return (
            str(room.get("room_id") or "").casefold(),
            str(room.get("name") or "").casefold(),
        )

    for match_kind in ("exact", "prefix", "contains"):
        matches: list[dict[str, Any]] = []
        for room in rooms:
            room_id, name = _keys(room)
            if match_kind == "exact" and needle in {room_id, name}:
                matches.append(room)
            elif match_kind == "prefix" and (
                room_id.startswith(needle) or name.startswith(needle)
            ):
                matches.append(room)
            elif match_kind == "contains" and (needle in room_id or needle in name):
                matches.append(room)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(
                (
                    f"{_clean_line(room.get('name') or room.get('room_id'), limit=48)} "
                    f"[{room_reference(room)}]"
                )
                for room in matches[:MAX_ROOM_CHOICES]
            )
            suffix = "…" if len(matches) > MAX_ROOM_CHOICES else ""
            raise RoomControlError(
                f"That matches several group chats: {names}{suffix}. Enter more of the name."
            )
    raise RoomControlError(f"No group chat matches “{_clean_line(query)}”.")


def _room_status(service: Any, room: Mapping[str, Any]) -> str:
    if room.get("_room_mode") == "desktop":
        command = room.get("desktop_command")
        state = str(command.get("state") or "") if isinstance(command, Mapping) else ""
        if state == "failed":
            return "needs attention"
        if state in {"pending", "claimed"} and room.get("desktop_available"):
            return "applying command"
        return "ready" if room.get("desktop_available") else "waiting for Desktop"
    if room.get("_room_mode") == "remote":
        return "connected"
    room_id = str(room["room_id"])
    status = service.status(room_id)
    if status.get("counts", {}).get("stopping"):
        return "stopping"
    if status.get("blocked"):
        return "needs attention"
    if status.get("working"):
        return "work queued or running"
    state = hosted_rooms.room_state(service.db_path, room_id=room_id)
    since = max(0, int(state.get("latest_seq") or 0) - 80)
    recent = hosted_rooms.read_events(
        service.db_path,
        room_id=room_id,
        since_seq=since,
        limit=80,
    ).get("events", [])
    latest_user = max(
        (int(event["seq"]) for event in recent if event.get("kind") == "message.user"),
        default=0,
    )
    latest_boundary = max(
        (
            int(event["seq"])
            for event in recent
            if event.get("kind") in {"room.activity", "room.stop_requested"}
        ),
        default=0,
    )
    if latest_user > latest_boundary:
        return "waiting for its Bots"
    return "idle"


def _room_action_flags(
    service: Any,
    room: Mapping[str, Any],
    *,
    remote_status: Mapping[str, Any] | None = None,
) -> tuple[bool, bool, bool]:
    """Return Retry/Stop availability and whether prior work is unconfirmed."""

    if room.get("_room_mode") == "desktop":
        command = room.get("desktop_command")
        state = str(command.get("state") or "") if isinstance(command, Mapping) else ""
        action = (
            str(command.get("action") or "") if isinstance(command, Mapping) else ""
        )
        return (
            int(room.get("desktop_failed_commands") or 0) > 0,
            action == "send" and state in {"claimed", "pending"},
            False,
        )

    status: Mapping[str, Any]
    if room.get("_room_mode") == "remote":
        status = remote_status or {}
    else:
        raw_status = service.status(str(room["room_id"]))
        status = raw_status if isinstance(raw_status, Mapping) else {}
    raw_counts = status.get("counts")
    counts = raw_counts if isinstance(raw_counts, Mapping) else {}
    raw_actions = status.get("pending_actions")
    actions = raw_actions if isinstance(raw_actions, list) else []
    retryable = any(
        isinstance(action_row, Mapping) and action_row.get("kind") == "retry"
        for action_row in actions
    ) or bool(counts.get("deferred") or counts.get("indeterminate"))
    stoppable = bool(status.get("working")) and not bool(counts.get("stopping"))
    return retryable, stoppable, bool(counts.get("indeterminate"))


def format_room_list(
    service: Any,
    *,
    rooms: list[dict[str, Any]] | None = None,
    rooms_command: str = "/group",
    page: int = 1,
) -> str:
    """Render a bounded, scan-friendly Group Chat list."""

    rooms = list_messaging_rooms(service) if rooms is None else list(rooms)
    if not rooms:
        return "👥 **No Group Chats yet**\nCreate one in Hermes Desktop first."
    rooms = sorted(rooms, key=lambda room: int(room.get("messaging_ref") or 0))
    if not isinstance(page, int) or page < 1:
        raise RoomControlError("Page numbers start at 1.")
    page_count = max(1, (len(rooms) + MAX_ROOM_CHOICES - 1) // MAX_ROOM_CHOICES)
    if page > page_count:
        raise RoomControlError(f"There are only {page_count} Group Chat pages.")
    start = (page - 1) * MAX_ROOM_CHOICES
    visible_rooms = rooms[start : start + MAX_ROOM_CHOICES]
    heading = (
        "Group Chats"
        if page_count == 1
        else f"Group Chats — page {page} of {page_count}"
    )
    lines = [f"👥 **{heading}**"]
    for room in visible_rooms:
        name = _clean_line(room.get("name") or room.get("room_id"), limit=72)
        member_count = _room_member_count(room)
        status = _room_status(service, room)
        lines.append(
            f"{_room_status_icon(status)} **{room_reference(room)}. "
            f"{_plain_display_label(name, limit=72)}** · "
            f"{status} · {member_count} Bot{'s' if member_count != 1 else ''}"
        )
    if page < page_count:
        lines.append(f"More: `{rooms_command} list {page + 1}`")
    elif page > 1:
        lines.append(f"Previous: `{rooms_command} list {page - 1}`")
    lines.extend([
        "",
        "────────",
        "🧭 **Controls**",
        f"Check: `{rooms_command} <number>`",
        f"Send: `{rooms_command} <number> send <message>`",
        f"Bots: `{rooms_command} <number> bots`",
        f"Retry: `{rooms_command} <number> retry`",
        f"Stop: `{rooms_command} <number> stop`",
    ])
    return "\n".join(lines)


def _event_label(event: Mapping[str, Any], member_names: Mapping[str, str]) -> str:
    raw_actor = event.get("actor")
    actor: Mapping[str, Any] = raw_actor if isinstance(raw_actor, Mapping) else {}
    actor_id = str(actor.get("id") or "")
    display_name = _clean_line(actor.get("display_name"), limit=48)
    if display_name:
        return display_name
    if event.get("kind") == "message.member":
        raw_payload = event.get("payload")
        payload: Mapping[str, Any] = (
            raw_payload if isinstance(raw_payload, Mapping) else {}
        )
        member_id = str(payload.get("member_id") or actor_id)
        return member_names.get(member_id, "Bot")
    return "You"


def format_room_detail(
    service: Any,
    room: Mapping[str, Any],
    *,
    room_command: str = "/group",
    show_approvals: bool = False,
) -> str:
    """Render status plus the latest visible room messages."""

    room_id = str(room["room_id"])
    if not isinstance(room.get("messaging_ref"), int):
        room = next(
            (
                candidate
                for candidate in list_messaging_rooms(service)
                if str(candidate["room_id"]) == room_id
            ),
            dict(room),
        )
    raw_members = room.get("members")
    members: list[Any] = raw_members if isinstance(raw_members, list) else []
    desktop_mode = room.get("_room_mode") == "desktop"
    remote_mode = room.get("_room_mode") == "remote"
    if desktop_mode:
        visible = [
            event for event in room.get("log", []) if isinstance(event, Mapping)
        ][-MAX_RECENT_MESSAGES:]
        member_names: dict[str, str] = {}
    elif remote_mode:
        summary = _remote_summary(service, room)
        remote_room = summary["room"]
        raw_remote_members = remote_room.get("members")
        members = (
            list(raw_remote_members) if isinstance(raw_remote_members, list) else []
        )
        room = {
            **room,
            "name": remote_room.get("name") or room.get("name"),
            "members": members,
            "member_count": len(members),
            "_remote_status": summary.get("status"),
        }
        member_names = {
            str(member.get("member_id") or ""): _clean_line(
                member.get("display_name") or member.get("handle") or "Bot",
                limit=48,
            )
            for member in members
            if isinstance(member, Mapping)
        }
        raw_events = summary.get("events")
        visible = [
            event
            for event in (raw_events if isinstance(raw_events, list) else [])
            if isinstance(event, Mapping)
            and event.get("kind") in {"message.user", "message.member"}
        ][-MAX_RECENT_MESSAGES:]
    else:
        state = hosted_rooms.room_state(service.db_path, room_id=room_id)
        since = max(0, int(state.get("latest_seq") or 0) - 80)
        delta = hosted_rooms.read_events(
            service.db_path,
            room_id=room_id,
            since_seq=since,
            limit=80,
        )
        member_names = {
            str(member.get("member_id") or ""): _clean_line(
                member.get("display_name") or member.get("handle") or "Bot",
                limit=48,
            )
            for member in members
            if isinstance(member, Mapping)
        }
        visible = [
            event
            for event in delta.get("events", [])
            if isinstance(event, Mapping)
            and event.get("kind") in {"message.user", "message.member"}
        ][-MAX_RECENT_MESSAGES:]
    name = _clean_line(room.get("name") or room_id, limit=72)
    status_text = _room_status(service, room)
    action_status: Mapping[str, Any] | None = None
    if remote_mode:
        remote_status = room.get("_remote_status")
        if isinstance(remote_status, Mapping):
            action_status = remote_status
            raw_counts = remote_status.get("counts")
            counts = raw_counts if isinstance(raw_counts, Mapping) else {}
            if counts.get("stopping"):
                status_text = "stopping"
            elif remote_status.get("blocked"):
                status_text = "needs attention"
            elif remote_status.get("working"):
                status_text = "work queued or running"
            else:
                status_text = "idle"
    lines = [
        f"💬 **{_plain_display_label(name, limit=72)}**",
        f"{_room_status_icon(status_text)} {status_text}",
        f"👥 {len(members)} Bot{'s' if len(members) != 1 else ''}",
    ]
    participant_lines = _room_participant_lines(room)
    if participant_lines:
        lines.extend(["", "🤖 **Bots**", *participant_lines])
    if visible:
        lines.extend(["", "🕘 **Recent activity**"])
        for event in visible:
            if desktop_mode:
                source = (
                    event.get("from") if isinstance(event.get("from"), Mapping) else {}
                )
                label = _clean_line(source.get("name") or "You", limit=48)
                text = event.get("text")
            else:
                payload = (
                    event.get("payload")
                    if isinstance(event.get("payload"), Mapping)
                    else {}
                )
                label = _event_label(event, member_names)
                text = payload.get("text")
            lines.append(
                f"• **{_plain_display_label(label, limit=48)}:** "
                f"{_plain_preview_text(text)}"
            )
    else:
        lines.extend(["", "No messages yet."])
    from gateway.hosted_room_messaging_approvals import format_pending_approvals

    if show_approvals:
        approval_section = format_pending_approvals(
            service,
            room,
            room_reference=str(room_reference(room)),
            room_command=room_command,
        )
        if approval_section:
            lines.extend(["", approval_section])
    failed_commands = int(room.get("desktop_failed_commands") or 0)
    show_retry, show_stop, unconfirmed = _room_action_flags(
        service,
        room,
        remote_status=action_status,
    )
    if unconfirmed:
        lines.extend([
            "",
            "A Bot’s last task could not be confirmed. Retrying may repeat its actions.",
        ])
    if desktop_mode and failed_commands > 0:
        lines.append(
            (
                "The latest command could not be applied."
                if failed_commands == 1
                else f"{failed_commands} commands could not be applied."
            )
            + " Retry here or open this Group Chat in Hermes Desktop."
        )
    lines.extend(["", "────────", "🧭 **Controls**"])
    lines.append(f"Send: `{room_command} {room_reference(room)} send <message>`")
    lines.append(f"Bots: `{room_command} {room_reference(room)} bots`")
    if show_retry:
        lines.append(f"Retry: `{room_command} {room_reference(room)} retry`")
    if show_stop:
        lines.append(f"Stop: `{room_command} {room_reference(room)} stop`")
    if _room_has_targetable_handle(room):
        lines.append(
            f"Message one Bot: `{room_command} {room_reference(room)} send @handle <message>`"
        )
    return "\n".join(lines)


def messaging_actor(event: Any, *, gateway_id: str) -> dict[str, str]:
    """Build a stable actor without persisting raw platform user IDs."""

    source = getattr(event, "source", None)
    platform_value = getattr(getattr(source, "platform", None), "value", None)
    platform = _clean_line(platform_value or "messaging", limit=32).casefold()
    raw_user_id = (
        getattr(source, "user_id_alt", None)
        or getattr(event, "user_id", None)
        or getattr(source, "user_id", None)
        or "unknown"
    )
    scope = getattr(source, "scope_id", None) or getattr(source, "guild_id", None)
    chat_id = getattr(source, "chat_id", None)
    digest = hashlib.sha256(
        f"{gateway_id}:{platform}:{scope or ''}:{chat_id or ''}:{raw_user_id}".encode()
    ).hexdigest()[:20]
    raw_name = _clean_line(
        getattr(event, "user_name", None) or getattr(source, "user_name", None),
        limit=48,
    )
    platform_label = platform.replace("_", " ").title()
    display_name = (
        f"{raw_name} via {platform_label}"
        if raw_name and raw_name != str(raw_user_id)
        else platform_label
    )
    return {
        "kind": "user",
        "id": f"messaging:{platform}:{digest}",
        "display_name": display_name,
    }


def _raw_transport_id(event: Any) -> Any:
    """Extract one adapter-owned redelivery key without guessing from text."""

    metadata = getattr(event, "metadata", None)
    candidates: list[Any] = []
    if isinstance(metadata, Mapping):
        candidates.extend(
            metadata.get(key)
            for key in (
                "delivery_id",
                "event_id",
                "message_id",
                "request_id",
                "update_id",
            )
        )
    raw = getattr(event, "raw_message", None)
    payloads = [raw]
    if isinstance(raw, Mapping):
        payloads.extend(
            raw.get(key) for key in ("event", "message", "data", "container")
        )
    for payload in payloads:
        if isinstance(payload, Mapping):
            candidates.extend(
                payload.get(key)
                for key in (
                    "client_msg_id",
                    "trigger_id",
                    "event_id",
                    "message_id",
                    "id",
                    "ts",
                    "event_ts",
                    "timestamp_ms",
                    "timestamp",
                )
            )
    if raw is not None and not isinstance(raw, Mapping):
        candidates.extend(getattr(raw, key, None) for key in ("id", "interaction_id"))
    return next(
        (value for value in candidates if value is not None and str(value).strip()),
        None,
    )


def messaging_event_id(event: Any) -> str:
    """Return a deterministic idempotency key when the transport provides one."""

    source = getattr(event, "source", None)
    platform = getattr(getattr(source, "platform", None), "value", "messaging")
    stable_message_id = (
        getattr(event, "message_id", None)
        or getattr(source, "message_id", None)
        or _raw_transport_id(event)
        or getattr(event, "platform_update_id", None)
    )
    if stable_message_id is None:
        raise RoomControlError(
            "This channel didn’t provide a stable message ID, so Hermes can’t "
            "safely repeat this room command. Try another connected channel."
        )
    material = "|".join(
        str(value or "")
        for value in (
            platform,
            getattr(source, "profile", None) or "default",
            messaging_transport_profile(event),
            getattr(source, "scope_id", None) or getattr(source, "guild_id", None),
            getattr(source, "chat_id", None),
            getattr(source, "thread_id", None),
            getattr(source, "user_id_alt", None)
            or getattr(event, "user_id", None)
            or getattr(source, "user_id", None),
            stable_message_id,
        )
    )
    return f"messaging:{hashlib.sha256(material.encode()).hexdigest()}"


def messaging_transport_profile(event: Any) -> str:
    """Return the locally trusted profile that owns the receiving adapter."""

    source = getattr(event, "source", None)
    adapter_ref = getattr(source, "_transport_adapter_ref", None)
    try:
        adapter = adapter_ref() if callable(adapter_ref) else adapter_ref
    except Exception:
        adapter = None
    if adapter is not None:
        owner = str(getattr(adapter, "_owner_profile", None) or "").strip()
        return owner or "default"
    return str(getattr(source, "profile", None) or "default")


def ensure_text_only(event: Any) -> None:
    """Reject media explicitly until hosted-room attachment transport exists."""

    source = getattr(event, "source", None)
    if (
        getattr(source, "message_had_attachments", False)
        or getattr(event, "media_urls", None)
        or getattr(event, "media_types", None)
    ):
        raise RoomControlError(
            "Attachments from messaging chats aren’t supported yet. Send text only."
        )


def is_machine_authored(event: Any) -> bool:
    """Recognize native and relayed bot/webhook provenance defensively."""

    source = getattr(event, "source", None)
    if getattr(source, "is_bot", False):
        return True
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping) and any(
        metadata.get(key) is True
        for key in ("is_bot", "sender_is_bot", "webhook_sender")
    ):
        return True
    raw = getattr(event, "raw_message", None)
    if isinstance(raw, Mapping):
        if raw.get("bot_id") or raw.get("bot_profile"):
            return True
        if raw.get("subtype") in {"bot_message", "webhook_message"}:
            return True
    for owner_field in ("author", "user"):
        owner = getattr(raw, owner_field, None)
        if getattr(owner, "bot", False) or getattr(owner, "is_bot", False):
            return True
    return False


def is_message_edit(event: Any) -> bool:
    """Reject edited commands even when a platform redelivers them as messages."""

    source = getattr(event, "source", None)
    if getattr(source, "message_is_edit", False):
        return True
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, Mapping) and metadata.get("message_is_edit") is True:
        return True
    raw = getattr(event, "raw_message", None)
    if isinstance(raw, Mapping):
        if raw.get("editMessage") or raw.get("isEdited") is True:
            return True
        if raw.get("subtype") == "message_changed":
            return True
        relation = raw.get("m.relates_to")
        if isinstance(relation, Mapping) and relation.get("rel_type") == "m.replace":
            return True
    return bool(getattr(raw, "edit_date", None) or getattr(raw, "edited_at", None))


def relay_provenance_is_unknown(event: Any) -> bool:
    """Fail closed until a relay producer classifies the inbound author."""

    source = getattr(event, "source", None)
    if not getattr(source, "delivered_via_upstream_relay", False):
        return False
    metadata = getattr(event, "metadata", None)
    return not (
        isinstance(metadata, Mapping)
        and metadata.get("relay_author_classified") is True
        and metadata.get("relay_edit_classified") is True
    )


def _desktop_authority_hash(room: Mapping[str, Any]) -> str:
    authority_hash = str(room.get("desktop_authority_hash") or "").strip().lower()
    if not re.fullmatch(r"[a-f0-9]{64}", authority_hash):
        raise RoomControlError(
            "Open this Group Chat once in the latest Hermes Desktop, then try again."
        )
    return authority_hash


def send_to_room(service: Any, room: Mapping[str, Any], event: Any, text: str) -> str:
    """Append or hand off one idempotent room turn."""

    ensure_text_only(event)
    event_id = messaging_event_id(event)
    name = _clean_line(room.get("name") or room.get("room_id"), limit=72)
    if room.get("_room_mode") == "desktop":
        from gateway.desktop_room_mailbox import enqueue_command, default_db_path

        actor = messaging_actor(
            event,
            gateway_id=hosted_rooms.local_authority_gateway_id(),
        )
        enqueue_command(
            default_db_path(),
            command_id=event_id,
            room_id=str(room["room_id"]),
            authority_hash=_desktop_authority_hash(room),
            action="send",
            payload={
                "message": text,
                "actor_display_name": actor.get("display_name") or "Messaging",
                "recipients": _frozen_desktop_recipients(room),
            },
        )
        if room.get("desktop_available"):
            return f"Queued in {name}."
        return f"Saved for {name}. Open or update Hermes Desktop to continue."
    if room.get("_room_mode") == "remote":
        actor = messaging_actor(
            event,
            gateway_id=hosted_rooms.local_authority_gateway_id(),
        )
        remote_mutate(
            service,
            room,
            action="send",
            command_id=event_id,
            text=text,
            actor_display_name=actor.get("display_name") or "Messaging",
        )
        return f"Queued in {name}."
    service.send(
        room_id=str(room["room_id"]),
        event_id=event_id,
        payload={"text": text, "thread_id": event_id},
        actor=messaging_actor(
            event,
            gateway_id=str(room["authority_gateway_id"]),
        ),
    )
    return f"Queued in {name}."


def stop_room(service: Any, room: Mapping[str, Any], event: Any) -> str:
    """Cancel active room tasks using a transport-derived idempotency key."""

    cancel_id = f"stop:{messaging_event_id(event)}"
    name = _clean_line(room.get("name") or room.get("room_id"), limit=72)
    if room.get("_room_mode") == "desktop":
        from gateway.desktop_room_mailbox import (
            DesktopRoomMailboxError, default_db_path, enqueue_stop_command,
        )

        target_thread_id, target_message_id = _latest_projected_stop_target(room)
        try:
            enqueue_stop_command(
                default_db_path(), command_id=cancel_id, room_id=str(room["room_id"]),
                authority_hash=_desktop_authority_hash(room),
                target_thread_id=target_thread_id, target_message_id=target_message_id,
            )
        except DesktopRoomMailboxError as exc:
            raise RoomControlError(str(exc)) from exc
        if room.get("desktop_available"):
            return f"Stop requested for {name}."
        return f"Stop saved for {name}. Open or update Hermes Desktop to apply it."
    if room.get("_room_mode") == "remote":
        remote_mutate(
            service,
            room,
            action="stop",
            command_id=cancel_id,
        )
        return f"Stop requested for {name}. Active work will stop safely."
    service.stop_room(str(room["room_id"]), cancel_id=cancel_id)
    return f"Stop requested for {name}. Active work will stop safely."


def retry_room(service: Any, room: Mapping[str, Any], event: Any) -> str:
    """Retry bounded failed work after an explicit owner command."""

    name = _clean_line(room.get("name") or room.get("room_id"), limit=72)
    room_id = str(room["room_id"])
    command_id = f"retry:{messaging_event_id(event)}"
    actor = messaging_actor(
        event,
        gateway_id=(
            hosted_rooms.local_authority_gateway_id()
            if room.get("_room_mode") == "desktop"
            else str(room.get("authority_gateway_id") or "")
        ),
    )
    receipt_db = Path(service.db_path)
    if room.get("_room_mode") == "remote":
        result = remote_mutate(
            service,
            room,
            action="retry",
            command_id=command_id,
            actor_display_name=actor.get("display_name") or "Messaging",
        )
        retried = result.get("retried")
        if not isinstance(retried, int):
            processed = result.get("processed")
            if isinstance(processed, int):
                suffix = "task" if processed == 1 else "tasks"
                return f"Retry checked for {name} ({processed} {suffix})."
            summary = result.get("summary")
            retried = (
                int(summary.get("retried") or 0) if isinstance(summary, Mapping) else 0
            )
        suffix = "task" if retried == 1 else "tasks"
        return f"Retry queued for {name} ({retried} {suffix})."
    if room.get("_room_mode") == "desktop":
        from gateway.desktop_room_mailbox import (
            DesktopRoomMailboxError,
            default_db_path,
            retry_failed_commands,
            retryable_command_ids,
        )

        mailbox_db = default_db_path()
        try:
            frozen, completed = _retry_receipt_plan(
                receipt_db,
                command_id=command_id,
                room_id=room_id,
                actor=actor,
                task_ids=[],
            )
        except RoomControlError as exc:
            if str(exc) != "This Group Chat has no failed work to retry.":
                raise
            try:
                target_ids = retryable_command_ids(
                    mailbox_db,
                    room_id=room_id,
                )
            except DesktopRoomMailboxError as mailbox_error:
                raise RoomControlError(
                    "This Group Chat has no failed work to retry."
                ) from mailbox_error
            frozen, completed = _retry_receipt_plan(
                receipt_db,
                command_id=command_id,
                room_id=room_id,
                actor=actor,
                task_ids=[target_id for target_id in target_ids if target_id],
            )
        if completed:
            return completed
        retried = retry_failed_commands(
            mailbox_db,
            room_id=room_id,
            command_ids=frozen,
        )
        count = len(retried)
        noun = "command" if count == 1 else "commands"
        result = f"Retry queued for {name} ({count} {noun})."
        _complete_retry_receipt(receipt_db, command_id=command_id, result=result)
        return result

    pending = [
        action
        for action in service.status(room_id).get("pending_actions", [])
        if isinstance(action, Mapping)
        and action.get("kind") == "retry"
        and str(action.get("task_id") or "")
    ]
    frozen, completed = _retry_receipt_plan(
        receipt_db,
        command_id=command_id,
        room_id=room_id,
        actor=actor,
        task_ids=[str(action["task_id"]) for action in pending[:MAX_ROOM_CHOICES]],
    )
    if completed:
        return completed

    def queue_for_worker() -> None:
        hosted_room_controls.begin_control_retry(
            receipt_db,
            command_id=f"worker:{command_id}",
            room_id=room_id,
            member_id=str(actor.get("id") or "messaging"),
            task_ids=list(frozen),
        )

    for task_id in frozen:
        try:
            service.retry_room_task(
                room_id,
                task_id=task_id,
                retry_id=hosted_room_controls.control_retry_attempt_id(
                    command_id,
                    task_id,
                ),
            )
        except driver.LeaseHeldError:
            queue_for_worker()
            break
        except RoomControlError as exc:
            if str(exc) != (
                "Retry is available when the device running this Group Chat is online."
            ):
                raise
            queue_for_worker()
            break
        except Exception:
            still_pending = {
                str(action.get("task_id") or "")
                for action in service.status(room_id).get("pending_actions", [])
                if isinstance(action, Mapping) and action.get("kind") == "retry"
            }
            if task_id in still_pending:
                queue_for_worker()
                break
    count = len(frozen)
    suffix = "task" if count == 1 else "tasks"
    result = f"Retry queued for {name} ({count} {suffix})."
    _complete_retry_receipt(receipt_db, command_id=command_id, result=result)
    return result
