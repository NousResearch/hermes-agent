"""The work-record capture step of the in-memory reference (``HostedRoomReplicaRepository`` internal step ``capture_transition_locked``).

The capture runs inside every driver write that changes a task and inside ``upsert_remote_run_receipt``. A work-record failure never reaches the verb; what does, rolling its write back, is the ``HostedRoomError`` of the roster check (the stored members re-validated) and the ``AuthorityHistoryError`` of the lineage read for a room past epoch 1. The reference runs exactly those two checks, in that order, and stores no work record: the records, their revisions and the authority-transition trigger that freezes them are not readable through any verb.
"""

from __future__ import annotations

import json
from typing import Any

from persistence.gateway.hosted_rooms import AuthorityHistoryError
from session_rules.hosted_rooms import (
    MAX_AUTHORITY_SPANS,
    AuthoritySpan,
    _gateway,
    _validate_members,
    validate_history,
)

from .hosted_room_tables import HostedTables, room_events

__all__ = ["capture_transition_locked"]


def capture_transition_locked(tables: HostedTables, room_id: str) -> None:
    """Run the capture's escaping checks for a live room inside the caller's write (module docstring); a missing or disbanded room captures nothing."""
    room = tables["hosted_rooms"].get(room_id)
    if room is None or room["disbanded_at"] is not None:
        return
    _validate_members(json.loads(room["members_json"]))
    if int(room["authority_epoch"]) != 1:
        _read_history(tables, room_id, gateway_id=room["authority_gateway_id"], epoch=int(room["authority_epoch"]))


def _claims(tables: HostedTables, room_id: str) -> list[dict[str, Any]]:
    # The outgoing owner's fence (authority.lost) and the new owner's claim both open a span.
    rows = [row for row in room_events(tables, room_id)
            if row["kind"] in ("authority.claimed", "authority.lost")][:MAX_AUTHORITY_SPANS]
    if len(rows) >= MAX_AUTHORITY_SPANS:
        raise AuthorityHistoryError("authority history exceeds its bound")
    return rows


# The authority-history read over the reference's event rows.
def _read_history(tables: HostedTables, room_id: str, *, gateway_id: str, epoch: int) -> None:
    rows = _claims(tables, room_id)
    if not rows:
        raise AuthorityHistoryError("current authority has no committed lineage")
    first = json.loads(rows[0]["payload_json"])
    if isinstance(first, dict) and first.get("previous_gateway_id") == "legacy":
        return
    if not isinstance(first, dict) or "previous_gateway_id" not in first:
        raise AuthorityHistoryError("authority history has no previous owner")
    history = [AuthoritySpan(_gateway(first["previous_gateway_id"]), 1, 0)]
    for row in rows:
        actor, payload = json.loads(row["actor_json"]), json.loads(row["payload_json"])
        if actor != {"kind": "system", "id": "authority-control"} or not isinstance(payload, dict):
            raise AuthorityHistoryError("authority claim provenance is invalid")
        expected = history[-1].epoch + 1
        if (payload.get("previous_gateway_id") != history[-1].gateway_id or type(payload.get("authority_epoch")) is not int
                or payload["authority_epoch"] != expected or row["authority_epoch"] != expected):
            raise AuthorityHistoryError("authority claim does not extend its predecessor")
        history.append(AuthoritySpan(_gateway(payload.get("authority_gateway_id")), expected, int(row["seq"])))
    validate_history([span.as_mapping() for span in history], gateway_id=gateway_id, epoch=epoch)
