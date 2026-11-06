"""Read-only, bounded Files catalog. Published references without local bytes stay listed;
blob contents are never read and history replication never copies them."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from persistence.gateway.hosted_rooms import AttachmentError, CatalogCursorError
from session_rules.hosted_attachments import fold_catalog_text, validate_manifest
from session_rules.hosted_rooms import identifier

DEFAULT_LIMIT = 8
MAX_LIMIT = 32
MAX_QUERY_CHARS = 255
MAX_YOU_LABEL_CHARS = 64
# The sharer label of a user share whose log actor has no display name; a client shows its own
# word for it and passes that word as ``you_label`` so a search for it finds these shares.
_DEFAULT_YOU = "You"
MAX_CURSOR_CHARS = 4096
# Maximum log rows examined per request, regardless of search selectivity.
EVENT_SCAN_LIMIT = 256
_SHARER_KINDS = {"message.user": "user", "message.member": "member"}
MANIFEST_FIELDS = ("attachment_id", "kind", "name", "size", "mime")
_CURSOR_FIELDS = frozenset({"v", "scope", "snapshot", "seq", "index"})


def _limit(value: Any) -> int:
    if value is None:
        return DEFAULT_LIMIT
    if type(value) is not int or not 1 <= value <= MAX_LIMIT:
        raise AttachmentError(
            f"attachment list limit must be an integer from 1 to {MAX_LIMIT}"
        )
    return value


def _query(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > MAX_QUERY_CHARS:
        raise AttachmentError(
            f"attachment query must be a string of at most {MAX_QUERY_CHARS} characters"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise AttachmentError("attachment query must be valid Unicode") from None
    return fold_catalog_text(value.strip())


def _you_label(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value.strip()) > MAX_YOU_LABEL_CHARS:
        raise AttachmentError(
            f"you_label must be a string of at most {MAX_YOU_LABEL_CHARS} characters"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise AttachmentError("you_label must be valid Unicode") from None
    return fold_catalog_text(value.strip())


def _roster_labels(members_json: Any) -> dict[str, str]:
    try:
        members = json.loads(members_json)
    except (TypeError, ValueError):
        return {}
    labels = {}
    for member in members if isinstance(members, list) else ():
        member_id = member.get("member_id") if isinstance(member, Mapping) else None
        if isinstance(member_id, str):
            label = (
                member.get("display_name")
                or member.get("handle")
                or member.get("profile")
            )
            labels[member_id] = (
                label.strip() if isinstance(label, str) and label.strip() else member_id
            )
    return labels


def _sharer(
    event_kind: Any, actor_json: Any, roster: Mapping[str, str]
) -> dict[str, str] | None:
    """Who shared a message's files, as the room log records it; None if unattributable."""
    kind = _SHARER_KINDS.get(event_kind)
    try:
        actor = json.loads(actor_json)
    except (TypeError, ValueError):
        return None
    if kind is None or not isinstance(actor, Mapping) or actor.get("kind") != kind:
        return None
    sharer_id = actor.get("id")
    if not isinstance(sharer_id, str) or not sharer_id:
        return None
    label = actor.get("display_name")
    label = label.strip() if isinstance(label, str) else ""
    return {
        "kind": kind,
        "id": sharer_id,
        "label": label
        or (_DEFAULT_YOU if kind == "user" else roster.get(sharer_id, sharer_id)),
    }


def _matches(
    name: str, sharer: Mapping[str, str], query: str, producer: str, you: str = ""
) -> bool:
    """``query`` in the file name or the sharer label; for a share labelled the default ``You``,
    also in the reader's own word for it (``you``, folded)."""
    if producer and sharer["id"] != producer:
        return False
    return (
        not query
        or query in fold_catalog_text(name)
        or query in fold_catalog_text(sharer["label"])
        or (
            bool(you)
            and sharer["kind"] == "user"
            and sharer["label"] == _DEFAULT_YOU
            and query in you
        )
    )


def _encode_cursor(**fields: Any) -> str:
    raw = json.dumps({"v": 1, **fields}, separators=(",", ":"), sort_keys=True).encode(
        "ascii"
    )
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _decode_cursor(value: Any, *, scope: str, latest_seq: int) -> dict[str, Any]:
    if not isinstance(value, str) or not value or len(value) > MAX_CURSOR_CHARS:
        raise CatalogCursorError("attachment list cursor is invalid")
    try:
        cursor = json.loads(
            base64.b64decode(
                value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
            )
        )
    except (binascii.Error, ValueError):
        raise CatalogCursorError("attachment list cursor is invalid") from None
    if (
        not isinstance(cursor, dict)
        or set(cursor) != _CURSOR_FIELDS
        or cursor["v"] != 1
        or cursor["scope"] != scope
        or type(cursor["snapshot"]) is not int
        or not 0 <= cursor["snapshot"] <= latest_seq
        or type(cursor["seq"]) is not int
        or not 1 <= cursor["seq"] <= cursor["snapshot"]
        or (
            cursor["index"] is not None
            and (type(cursor["index"]) is not int or cursor["index"] < 0)
        )
    ):
        raise CatalogCursorError("attachment list cursor does not match this request")
    return cursor


def _manifest(value: Any) -> list[dict[str, Any]]:
    try:
        raw = json.loads(value)
        manifest = validate_manifest(raw)
    except (TypeError, ValueError):
        return []
    return manifest if manifest == raw else []


def public_row(row, *, room_id, event_id, entry):
    if (
        row["room_id"] != room_id
        or row["event_id"] != event_id
        or row["state"] != "committed"
        or row["viewer_access"] != 1
        or any(row[key] != entry[key] for key in MANIFEST_FIELDS)
    ):
        return False
    try:
        json.loads(row["recipient_member_ids_json"])
    except (TypeError, ValueError):
        return False
    return True


def published_reference(event, attachment_id, availability):
    if event is None or _sharer(event["kind"], event["actor_json"], {}) is None:
        return False
    try:
        payload = json.loads(event["payload_json"])
        raw = (
            json.dumps(payload.get("attachments"))
            if isinstance(payload, dict)
            else "null"
        )
    except (TypeError, ValueError):
        return False
    return any(
        entry["attachment_id"] == attachment_id
        and availability(entry=entry) is not None
        for entry in _manifest(raw)
    )


def list_published(
    room,
    file_events,
    availability,
    *,
    room_id: str,
    authority_gateway_id: str,
    authority_epoch: int,
    cursor: Any = None,
    limit: Any = None,
    query: Any = None,
    producer_member_id: Any = None,
    you_label: Any = None,
) -> dict[str, Any]:
    """List exact versions in one snapshot; cursors bind the authority and all search filters.

    Search ignores case/accents and includes the reader's localized default You label."""
    limit, query = _limit(limit), _query(query)
    producer = (
        ""
        if producer_member_id is None or producer_member_id == ""
        else identifier(
            producer_member_id, label="producer_member_id", error=AttachmentError
        )
    )
    you = _you_label(you_label)
    if not query:
        you = ""  # only a search uses it: a listing without a query, and its cursor, ignore the word
    scope = hashlib.sha256(
        json.dumps(
            (room_id, authority_gateway_id, authority_epoch, query, producer, you),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    latest = int(room["next_seq"]) - 1
    position = (
        None
        if cursor is None
        else _decode_cursor(cursor, scope=scope, latest_seq=latest)
    )
    snapshot = latest if position is None else position["snapshot"]

    def page(
        items: list[dict[str, Any]], seq: int | None = None, index: int | None = None
    ) -> dict[str, Any]:
        next_cursor = (
            None
            if seq is None
            else _encode_cursor(scope=scope, snapshot=snapshot, seq=seq, index=index)
        )
        return {
            "room_id": room_id,
            "authority": {"gateway_id": authority_gateway_id, "epoch": authority_epoch},
            "snapshot_seq": snapshot,
            "items": items,
            "next_cursor": next_cursor,
            "has_more": next_cursor is not None,
        }

    roster = _roster_labels(room["members_json"])
    events = file_events(snapshot=snapshot, position=position)

    items: list[dict[str, Any]] = []
    for event in events[:EVENT_SCAN_LIMIT]:
        seq, event_id = int(event["seq"]), str(event["event_id"])
        sharer = _sharer(event["kind"], event["actor_json"], roster)
        if sharer is None:
            continue
        for index, entry in enumerate(_manifest(event["manifest_json"])):
            if (
                position is not None
                and seq == position["seq"]
                and position["index"] is not None
                and index <= position["index"]
            ):
                continue
            if not _matches(entry["name"], sharer, query, producer, you):
                continue
            available = availability(event_id=event_id, entry=entry)
            if available is None:
                continue
            if len(items) == limit:
                return page(items, items[-1]["seq"], items[-1]["manifest_index"])
            items.append(
                {
                    **{field: entry[field] for field in MANIFEST_FIELDS},
                    "event_id": event_id,
                    "seq": seq,
                    "manifest_index": index,
                    "producer": sharer,
                    "shared_at": float(event["created_at"]),
                    **({"available": False} if not available else {}),
                }
            )
    if len(events) > EVENT_SCAN_LIMIT:
        return page(items, int(events[EVENT_SCAN_LIMIT - 1]["seq"]))
    return page(items)
