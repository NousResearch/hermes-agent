"""Bounded replay context for committed authority changes, never election proof.

Reused from the recovery prerequisite; lineage mechanics build on Teknium's
#99047, authored commit 6af548a1f73f26bca54dd2e23f0d278c37f3ba6c.
The passive wire contract applies stricter bounds in hosted_room_passive_lineage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from gateway.hosted_rooms_common import identifier

MAX_AUTHORITY_SPANS = 1024


class AuthorityHistoryError(ValueError):
    """The recorded authority sequence cannot explain the current room."""


@dataclass(frozen=True)
class AuthoritySpan:
    gateway_id: str
    epoch: int
    from_seq: int

    def as_mapping(self) -> dict[str, Any]:
        return {"gateway_id": self.gateway_id, "epoch": self.epoch, "from_seq": self.from_seq}


def _gateway(value: Any) -> str:
    result = identifier(value, label="authority gateway", error=AuthorityHistoryError, max_chars=128)
    if result != value:
        raise AuthorityHistoryError("authority gateway must be canonical")
    return result


def validate_history(value: Any, *, gateway_id: str, epoch: int) -> tuple[AuthoritySpan, ...]:
    """Validate contextual epochs; possession of this data grants no control rights."""
    if value is None:
        return (AuthoritySpan(gateway_id, epoch, 0),)
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_AUTHORITY_SPANS:
        raise AuthorityHistoryError("authority history exceeds its bound or is absent")
    spans = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"gateway_id", "epoch", "from_seq"}:
            raise AuthorityHistoryError("authority history fields are invalid")
        source = _gateway(item["gateway_id"])
        generation, start = item["epoch"], item["from_seq"]
        if type(generation) is not int or not 1 <= generation < 2**63 or type(start) is not int or not 0 <= start < 2**63:
            raise AuthorityHistoryError("authority history coordinates are invalid")
        if not spans:
            if generation != 1 or start != 0:
                raise AuthorityHistoryError("authority history has no verified starting point")
        elif generation != spans[-1].epoch + 1 or start <= spans[-1].from_seq:
            raise AuthorityHistoryError("authority history has a gap or unordered boundary")
        spans.append(AuthoritySpan(source, generation, start))
    if (spans[-1].gateway_id, spans[-1].epoch) != (gateway_id, epoch):
        raise AuthorityHistoryError("authority history does not reach the current owner")
    return tuple(spans)


def at_sequence(history: tuple[AuthoritySpan, ...], seq: int) -> AuthoritySpan:
    for span in reversed(history):
        if seq >= span.from_seq:
            return span
    raise AuthorityHistoryError("event precedes the known authority history")


def origin_gateway_id(room: dict[str, Any]) -> str:
    """Keep the group/session namespace stable when its coordinator changes."""
    history = validate_history(room.get("authority_history"), gateway_id=room["authority_gateway_id"],
                               epoch=room["authority_epoch"])
    return history[0].gateway_id


def read_history_locked(conn, room_id: str, *, gateway_id: str, epoch: int) -> list[dict[str, Any]] | None:
    """Read committed claims within the caller's room-state transaction.

    Legacy adoption is left on its existing path; it is not retroactively
    certified as a history of physical-host transfers.
    """
    rows = conn.execute(
        """SELECT seq,authority_epoch,actor_json,payload_json FROM hosted_room_events
           WHERE room_id=? AND kind='authority.claimed' ORDER BY seq LIMIT ?""",
        (room_id, MAX_AUTHORITY_SPANS),
    ).fetchall()
    if len(rows) >= MAX_AUTHORITY_SPANS:
        raise AuthorityHistoryError("authority history exceeds its bound")
    if not rows:
        if epoch == 1:
            return None
        raise AuthorityHistoryError("current authority has no committed lineage")
    first = json.loads(rows[0]["payload_json"])
    if isinstance(first, dict) and first.get("previous_gateway_id") == "legacy":
        return None
    if not isinstance(first, dict) or "previous_gateway_id" not in first:
        raise AuthorityHistoryError("authority history has no previous owner")
    history = [AuthoritySpan(_gateway(first["previous_gateway_id"]), 1, 0)]
    for row in rows:
        actor, payload = json.loads(row["actor_json"]), json.loads(row["payload_json"])
        if actor != {"kind": "system", "id": "authority-control"} or not isinstance(payload, dict):
            raise AuthorityHistoryError("authority claim provenance is invalid")
        expected = history[-1].epoch + 1
        if (payload.get("previous_gateway_id") != history[-1].gateway_id
                or type(payload.get("authority_epoch")) is not int or payload["authority_epoch"] != expected
                or row["authority_epoch"] != expected):
            raise AuthorityHistoryError("authority claim does not extend its predecessor")
        history.append(AuthoritySpan(_gateway(payload.get("authority_gateway_id")), expected, int(row["seq"])))
    result = [span.as_mapping() for span in history]
    validate_history(result, gateway_id=gateway_id, epoch=epoch)
    return result
