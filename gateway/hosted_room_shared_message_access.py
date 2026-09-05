"""Exact bounded canonical Bot-reply reads for the Full reply consumer."""

import json
import math

from gateway import hosted_rooms
from gateway.hosted_room_file_contract import FileAccessError, identifier

MAX_REPLY_TEXT_BYTES = 64 * 1024
MAX_REPLY_RESPONSE_BYTES = 512 * 1024


def validate_reply(value, *, event_id):
    if not isinstance(value, dict) or set(value) != {
        "event_id",
        "seq",
        "producer",
        "text",
        "shared_at",
    }:
        raise FileAccessError("file_invalid_response")
    try:
        if (
            value["event_id"] != event_id
            or type(value["seq"]) is not int
            or value["seq"] < 1
        ):
            raise ValueError
        producer = value["producer"]
        if (
            not isinstance(producer, dict)
            or set(producer) != {"kind", "id", "label"}
            or producer["kind"] != "member"
        ):
            raise ValueError
        identifier(producer["id"])
        if not isinstance(producer["label"], str) or len(producer["label"]) > 200:
            raise ValueError
        if (
            not isinstance(value["text"], str)
            or not value["text"]
            or len(value["text"].encode("utf-8")) > MAX_REPLY_TEXT_BYTES
        ):
            raise ValueError
        if type(value["shared_at"]) not in {int, float} or not math.isfinite(
            value["shared_at"]
        ):
            raise ValueError
    except (ValueError, TypeError, FileAccessError):
        raise FileAccessError("file_invalid_response") from None
    return value


def read_local_shared_message(backend, *, room, event_id, member_id=None):
    from gateway.hosted_room_file_access import _owned

    event_id = identifier(event_id)
    current = _owned(backend, room, member_id)
    conn = hosted_rooms._read_connection(backend.db_path)
    try:
        row = conn.execute(
            """SELECT event_id, seq, actor_json, payload_json, created_at
                 FROM hosted_room_events WHERE room_id=? AND event_id=?
                  AND kind='message.member'
                  AND length(CAST(payload_json AS BLOB))<=?
                  AND length(CAST(actor_json AS BLOB))<=4096""",
            (current["room_id"], event_id, hosted_rooms.MAX_EVENT_JSON_BYTES),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise FileAccessError("file_unavailable")
    try:
        actor = json.loads(row["actor_json"])
        payload = json.loads(row["payload_json"])
        if not isinstance(actor, dict) or not isinstance(payload, dict):
            raise ValueError
        if payload.get("member_id", actor.get("id")) != actor.get("id"):
            raise ValueError
        result = validate_reply(
            {
                "event_id": event_id,
                "seq": row["seq"],
                "text": payload.get("text"),
                "producer": {
                    "kind": actor.get("kind"),
                    "id": actor.get("id"),
                    "label": actor.get("display_name") or actor.get("id"),
                },
                "shared_at": row["created_at"],
            },
            event_id=event_id,
        )
    except (ValueError, TypeError):
        raise FileAccessError("file_unavailable") from None
    _owned(backend, current, member_id)
    return result


def read_shared_message(backend, *, room, event_id, profile="default"):
    from gateway.hosted_room_file_access import _local_member, _owned, _remote

    identifier(event_id)
    mode = room.get("_room_mode", "hosted")
    if mode == "remote":
        return _remote(backend, room, profile, "read_shared_message", event_id=event_id)
    if mode != "hosted":
        raise FileAccessError(
            "classic_files_on_desktop"
            if mode == "desktop"
            else "file_access_unsupported"
        )
    current = _owned(backend, room)
    member = _local_member(current, profile)
    result = read_local_shared_message(
        backend,
        room=current,
        event_id=event_id,
        member_id=member,
    )
    if _local_member(_owned(backend, current), profile) != member:
        raise FileAccessError("file_access_denied")
    return result
