"""Owner-messaging file access, without delivery, menus, or new storage.

Callers must authorize the current native messaging owner before every call.
Remote control credentials retain member-recipient restrictions in addition to
published viewer visibility; they are not upgraded to unrestricted file viewers.
"""

from __future__ import annotations

import hmac
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from gateway import hosted_room_controls as controls, hosted_rooms
from gateway.hosted_room_file_contract import (
    FileAccessError,
    MANIFEST_FIELDS,
    MAX_FILE_BYTES,
    attachment_id,
    byte_limit,
    catalog_options,
    identifier,
)


def _require_viewer_room(db_path, room):
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore

    conn = hosted_rooms._read_connection(db_path)
    try:
        HostedRoomAttachmentStore._require_viewer_room(
            conn,
            room_id=room["room_id"],
            authority_gateway_id=room["authority_gateway_id"],
            authority_epoch=room["authority_epoch"],
        )
    finally:
        conn.close()


def _owned(backend, expected: Mapping[str, Any], member_id: str | None = None):
    room_id = identifier(expected.get("room_id"))
    if type(expected.get("authority_epoch")) is not int:
        raise FileAccessError("file_access_denied")
    try:
        service = backend.service
        if service is not None and Path(service.db_path) != Path(backend.db_path):
            raise FileAccessError("file_access_denied")
        if service is not None and callable(getattr(service, "_owned_room", None)):
            room = service._owned_room(room_id)
        else:
            room = hosted_rooms.room_state(backend.db_path, room_id=room_id)
        _require_viewer_room(backend.db_path, room)
        if (
            room["authority_gateway_id"] != hosted_rooms.local_authority_gateway_id()
            or room["authority_gateway_id"] != expected.get("authority_gateway_id")
            or room["authority_epoch"] != expected.get("authority_epoch")
            or (
                member_id is not None
                and not any(
                    member.get("member_id") == member_id for member in room["members"]
                )
            )
        ):
            raise FileAccessError("file_access_denied")
        return room
    except FileAccessError:
        raise
    except Exception:
        raise FileAccessError("file_access_denied") from None


@lru_cache(maxsize=8)
def _standalone_store(db_path: str):
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore

    return HostedRoomAttachmentStore(db_path)


def _store(backend):
    try:
        from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    except ImportError:
        raise FileAccessError("file_access_unsupported") from None
    store = getattr(backend.service, "attachments", None)
    if isinstance(store, HostedRoomAttachmentStore):
        if Path(store.db_path) != Path(backend.db_path):
            raise FileAccessError("file_access_denied")
        return store
    return _standalone_store(str(Path(backend.db_path).resolve()))


def _eligible(
    store, *, room_id, event_id, selected_id, member_id, max_bytes=MAX_FILE_BYTES
):
    # Retain the exact-row check after catalogue filtering, without a write lock.
    conn = hosted_rooms._read_connection(store.db_path)
    try:
        row = conn.execute(
            """SELECT state, viewer_access, expires_at, size, recipient_member_ids_json
                 FROM hosted_room_attachments
                WHERE attachment_id=? AND room_id=? AND event_id=?""",
            (selected_id, room_id, event_id),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row["state"] != "committed" or row["viewer_access"] != 1:
        return False
    try:
        if row["expires_at"] is not None:
            expiry = float(row["expires_at"])
            if not math.isfinite(expiry) or expiry <= store.clock():
                return False
        recipients = json.loads(row["recipient_member_ids_json"])
        if (
            not isinstance(recipients, list)
            or not 1 <= len(recipients) <= hosted_rooms.MAX_MEMBERS
            or not all(
                isinstance(value, str) and 0 < len(value) <= 128 for value in recipients
            )
            or len(set(recipients)) != len(recipients)
        ):
            return False
        if member_id is not None and member_id not in recipients:
            return False
        if not 0 < row["size"] <= max_bytes:
            raise FileAccessError("file_too_large")
    except (TypeError, ValueError):
        return False
    return True


def list_local_files(backend, *, room, member_id=None, **options):
    options = catalog_options(options)
    current = _owned(backend, room, member_id)
    store = _store(backend)
    try:
        from gateway.hosted_room_attachments import AttachmentCursorError

        page = store.list_published(
            room_id=current["room_id"],
            authority_gateway_id=current["authority_gateway_id"],
            authority_epoch=current["authority_epoch"],
            recipient_member_id=member_id,
            **options,
        )
    except AttachmentCursorError:
        raise FileAccessError("attachment_cursor_reset_required") from None
    except ValueError:
        raise FileAccessError("file_invalid_request") from None
    page["items"] = [
        item
        for item in page["items"]
        if _eligible(
            store,
            room_id=current["room_id"],
            event_id=item["event_id"],
            selected_id=item["attachment_id"],
            member_id=member_id,
        )
    ]
    _owned(backend, current, member_id)
    return page


def read_local_file(
    backend, *, room, event_id, selected_id, member_id=None, max_bytes=MAX_FILE_BYTES
):
    event_id, selected_id = identifier(event_id), attachment_id(selected_id)
    max_bytes = byte_limit(max_bytes)
    current = _owned(backend, room, member_id)
    store = _store(backend)
    check = dict(
        room_id=current["room_id"],
        event_id=event_id,
        selected_id=selected_id,
        member_id=member_id,
        max_bytes=max_bytes,
    )
    if not _eligible(store, **check):
        raise FileAccessError("file_unavailable")
    from gateway.hosted_room_attachments import (
        AttachmentData,
        AttachmentIntegrityError,
        AttachmentNotFoundError,
    )

    try:
        stored = store.read_viewer(
            room_id=current["room_id"],
            event_id=event_id,
            attachment_id=selected_id,
            recipient_member_id=member_id,
            authority_gateway_id=current["authority_gateway_id"],
            authority_epoch=current["authority_epoch"],
        )
    except AttachmentIntegrityError:
        raise FileAccessError("file_integrity_failed") from None
    except (AttachmentNotFoundError, ValueError):
        raise FileAccessError("file_unavailable") from None
    _owned(backend, current, member_id)
    if not _eligible(store, **check):
        raise FileAccessError("file_unavailable")
    return AttachmentData(
        {
            key: stored.attachment[key]
            for key in MANIFEST_FIELDS | {"event_id", "sha256"}
        },
        stored.data,
    )


def _local_member(room, profile):
    identifier(profile)
    if profile == "default":
        return None
    matches = [
        member["member_id"]
        for member in room["members"]
        if member.get("profile") == profile
        and isinstance(member.get("target") or {}, Mapping)
        and (member.get("target") or {}).get("kind", "local") == "local"
    ]
    if len(matches) != 1:
        raise FileAccessError("file_access_denied")
    return matches[0]


def _remote_link(backend, room, profile):
    from gateway.hosted_room_messaging import _remote_control_link

    identifier(profile)
    if type(room.get("authority_epoch")) is not int:
        raise FileAccessError("file_access_denied")
    try:
        link = _remote_control_link(backend, room)
        if link.authority_gateway_id != room.get(
            "authority_gateway_id"
        ) or link.authority_epoch != room.get("authority_epoch"):
            raise FileAccessError("file_access_denied")
        # An unbound legacy link remains valid for default-owner status, but
        # Files fails with the existing re-registration guidance.
        if link.target_profile is None:
            raise FileAccessError("file_access_denied")
        target_profile = identifier(link.target_profile)
        if profile != "default" and profile != target_profile:
            raise FileAccessError("file_access_denied")
        if controls.peer_reservation_is_revoked(
            backend.db_path,
            room_id=link.room_id,
            member_id=link.member_id,
            target_profile=target_profile,
            authority_gateway_id=link.authority_gateway_id,
            authority_epoch=link.authority_epoch,
        ):
            raise FileAccessError("file_access_denied")
        return link, target_profile
    except FileAccessError:
        raise
    except Exception:
        raise FileAccessError("file_access_denied") from None


def _remote(backend, room, profile, operation, **options):
    from gateway.hosted_room_control_client import RoomControlHTTPClient
    from gateway.hosted_room_file_contract import FileCodeAmbiguous

    link, target_profile = _remote_link(backend, room, profile)
    ambiguous = None
    try:
        result = getattr(RoomControlHTTPClient(link), operation)(
            target_profile=target_profile, **options
        )
    except FileCodeAmbiguous as exc:
        ambiguous = exc
    fresh, fresh_profile = _remote_link(backend, room, profile)
    if (
        fresh_profile != target_profile
        or fresh.home_url != link.home_url
        or not hmac.compare_digest(link.control_token, fresh.control_token)
    ):
        raise FileAccessError("file_access_denied")
    if ambiguous is not None:
        raise ambiguous
    return result


def list_room_files(backend, *, room: Mapping[str, Any], profile="default", **options):
    options = catalog_options(options)
    mode = room.get("_room_mode", "hosted")
    if mode == "remote":
        page = _remote(backend, room, profile, "list_files", **options)
        return {key: value for key, value in page.items() if key != "scope"}
    if mode != "hosted":
        raise FileAccessError(
            "classic_files_on_desktop"
            if mode == "desktop"
            else "file_access_unsupported"
        )
    current = _owned(backend, room)
    member = _local_member(current, profile)
    result = list_local_files(backend, room=current, member_id=member, **options)
    if _local_member(_owned(backend, current), profile) != member:
        raise FileAccessError("file_access_denied")
    return result


def read_room_file(
    backend,
    *,
    room: Mapping[str, Any],
    event_id,
    attachment_id: str,
    profile="default",
    max_bytes=MAX_FILE_BYTES,
):
    identifier(event_id)
    byte_limit(max_bytes)
    mode = room.get("_room_mode", "hosted")
    if mode == "remote":
        return _remote(
            backend,
            room,
            profile,
            "read_file",
            event_id=event_id,
            attachment_id=attachment_id,
            max_bytes=max_bytes,
        )
    if mode != "hosted":
        raise FileAccessError(
            "classic_files_on_desktop"
            if mode == "desktop"
            else "file_access_unsupported"
        )
    current = _owned(backend, room)
    member = _local_member(current, profile)
    result = read_local_file(
        backend,
        room=current,
        event_id=event_id,
        selected_id=attachment_id,
        max_bytes=max_bytes,
        member_id=member,
    )
    if _local_member(_owned(backend, current), profile) != member:
        raise FileAccessError("file_access_denied")
    return result
