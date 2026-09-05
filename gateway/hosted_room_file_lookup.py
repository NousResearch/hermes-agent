"""Bounded exact file-code and latest-reply lookups over canonical room state."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import closing

from gateway import hosted_rooms
from gateway.hosted_room_file_contract import (
    FileCodeAmbiguous,
    FileAccessError,
    identifier,
)

MAX_LOOKUP_CANDIDATES = 50_000
MAX_LOOKUP_MATCHES = 64
MAX_LOOKUP_SQL_STEPS = 4_000_000
LOOKUP_SECONDS = 10.0
_CODE = re.compile(r"^[0-9a-fA-F]{8,64}$")


def selection_digest(room, item):
    values = [
        "group-file-v1",
        room["room_id"],
        room["authority_gateway_id"],
        room["authority_epoch"],
        item["event_id"],
        item["attachment_id"],
    ]
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def validate_code(code):
    if not isinstance(code, str) or _CODE.fullmatch(code) is None:
        raise FileAccessError("file_invalid_request")
    return code.lower()


def _bounded_connection(conn):
    deadline = time.monotonic() + LOOKUP_SECONDS
    steps = 0

    def progress():
        nonlocal steps
        steps += 1000
        return int(steps > MAX_LOOKUP_SQL_STEPS or time.monotonic() >= deadline)

    conn.set_progress_handler(progress, 1000)
    return deadline


def _published_selection(conn, room, candidate, member_id):
    from gateway.hosted_room_attachment_catalog import (
        _event_epoch_is_visible,
        _published_event_attachments,
    )

    event = conn.execute(
        """SELECT seq, event_id, kind, authority_epoch, created_at
             FROM hosted_room_events WHERE room_id=? AND event_id=?""",
        (room["room_id"], candidate["event_id"]),
    ).fetchone()
    if event is None or not _event_epoch_is_visible(
        event["authority_epoch"], room["authority_epoch"]
    ):
        return None
    room_members = {
        member["member_id"]: (
            member.get("display_name")
            or member.get("handle")
            or member.get("profile")
            or member["member_id"]
        ).strip()
        for member in room["members"]
    }
    items = _published_event_attachments(
        conn,
        room_id=room["room_id"],
        event=event,
        room_members=room_members,
        now=time.time(),
        recipient_member_id=member_id or "",
    )
    return next(
        (item for item in items if item["attachment_id"] == candidate["attachment_id"]),
        None,
    )


def resolve_local_file(backend, *, room, code, member_id=None):
    from gateway.hosted_room_file_access import _eligible, _owned, _store

    code = validate_code(code)
    current = _owned(backend, room, member_id)
    store = _store(backend)
    found = []
    matches = 0
    try:
        with closing(hosted_rooms._read_connection(store.db_path)) as conn:
            conn.execute("BEGIN")
            deadline = _bounded_connection(conn)
            candidates = conn.execute(
                """SELECT a.attachment_id, a.event_id FROM hosted_room_attachments a
                    JOIN hosted_room_events e ON e.room_id=a.room_id AND e.event_id=a.event_id
                    WHERE a.room_id=? AND a.state='committed' AND a.viewer_access=1
                      AND (a.expires_at IS NULL OR a.expires_at>?)
                      AND e.kind IN ('message.user','message.member') LIMIT ?""",
                (current["room_id"], time.time(), MAX_LOOKUP_CANDIDATES + 1),
            )
            for inspected, candidate in enumerate(candidates, 1):
                if inspected > MAX_LOOKUP_CANDIDATES or time.monotonic() >= deadline:
                    raise FileAccessError("file_lookup_limit")
                if not selection_digest(current, candidate).startswith(code):
                    continue
                matches += 1
                if matches > MAX_LOOKUP_MATCHES:
                    raise FileAccessError("file_lookup_limit")
                item = _published_selection(conn, current, candidate, member_id)
                if item is not None:
                    found.append(item)
                    if len(found) > 8:
                        raise FileAccessError("file_lookup_limit")
    except sqlite3.OperationalError:
        raise FileAccessError("file_lookup_limit") from None
    _owned(backend, current, member_id)
    found = [
        item
        for item in found
        if _eligible(
            store,
            room_id=current["room_id"],
            event_id=item["event_id"],
            selected_id=item["attachment_id"],
            member_id=member_id,
        )
    ]
    if len(found) > 1:
        found.sort(
            key=lambda item: (
                -item["seq"],
                item["manifest_index"],
                item["attachment_id"],
            )
        )
        raise FileCodeAmbiguous(found)
    if not found or not _eligible(
        store,
        room_id=current["room_id"],
        event_id=found[0]["event_id"],
        selected_id=found[0]["attachment_id"],
        member_id=member_id,
    ):
        raise FileAccessError("file_unavailable")
    return found[0]


def latest_local_reply(backend, *, room, member_id=None):
    from gateway.hosted_room_attachment_catalog import _event_epoch_is_visible
    from gateway.hosted_room_file_access import _owned
    from gateway.hosted_room_shared_message_access import read_local_shared_message

    current = _owned(backend, room, member_id)
    conn = hosted_rooms._read_connection(backend.db_path)
    try:
        _bounded_connection(conn)
        row = conn.execute(
            """SELECT event_id, authority_epoch FROM hosted_room_events
                WHERE room_id=? AND kind='message.member'
                ORDER BY seq DESC LIMIT 1""",
            (current["room_id"],),
        ).fetchone()
    except sqlite3.OperationalError:
        raise FileAccessError("file_lookup_limit") from None
    finally:
        conn.close()
    if row is None or not _event_epoch_is_visible(
        row["authority_epoch"], current["authority_epoch"]
    ):
        raise FileAccessError("file_unavailable")
    return read_local_shared_message(
        backend, room=current, event_id=row["event_id"], member_id=member_id
    )


def _dispatch(backend, room, profile, operation, **options):
    from gateway.hosted_room_file_access import _local_member, _owned, _remote

    identifier(profile)
    mode = room.get("_room_mode", "hosted")
    if mode == "remote":
        return _remote(backend, room, profile, operation, **options)
    if mode != "hosted":
        raise FileAccessError(
            "classic_files_on_desktop"
            if mode == "desktop"
            else "file_access_unsupported"
        )
    current = _owned(backend, room)
    member = _local_member(current, profile)
    function = resolve_local_file if operation == "resolve_file" else latest_local_reply
    try:
        result = function(backend, room=current, member_id=member, **options)
    except FileCodeAmbiguous:
        if _local_member(_owned(backend, current), profile) != member:
            raise FileAccessError("file_access_denied")
        raise
    if _local_member(_owned(backend, current), profile) != member:
        raise FileAccessError("file_access_denied")
    return result


def resolve_file(backend, *, room, code, profile="default"):
    return _dispatch(backend, room, profile, "resolve_file", code=validate_code(code))


def latest_reply(backend, *, room, profile="default"):
    return _dispatch(backend, room, profile, "latest_shared_message")
