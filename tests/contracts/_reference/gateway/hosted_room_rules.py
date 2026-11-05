"""The pure hosted-room helpers the reference runs inside its transactions, under the names its repositories use.

The helpers come from ``session_rules.hosted_rooms``; ``room_grant_scope_key`` runs its scope key over the typed claims.
"""

from __future__ import annotations

import dataclasses

from persistence.gateway.hosted_rooms import RoomGrantClaims
from session_rules import hosted_rooms as neutral
from session_rules.hosted_rooms import (
    CONTROL_EVENT_BYTE_RESERVE,
    CONTROL_EVENT_COUNT_RESERVE,
    DISBANDED_ROOM_RETENTION_SECONDS,
    MAX_ACTIVE_ROOMS,
    MAX_DISBANDED_ROOM_TOMBSTONES,
    MAX_EVENTS_PER_ROOM,
    MAX_GATEWAY_EVENT_BYTES,
    MAX_LOG_PAGE_BYTES,
    MAX_ROOM_EVENT_BYTES,
    clock,
    compact_json,
    utf8_len,
)

__all__ = [
    "CONTROL_EVENT_BYTE_RESERVE",
    "CONTROL_EVENT_COUNT_RESERVE",
    "CONTROL_EVENT_KINDS",
    "DISBANDED_ROOM_RETENTION_SECONDS",
    "MAX_ACTIVE_ROOMS",
    "MAX_DISBANDED_ROOM_TOMBSTONES",
    "MAX_EVENTS_PER_ROOM",
    "MAX_GATEWAY_EVENT_BYTES",
    "MAX_LOG_PAGE_BYTES",
    "MAX_ROOM_EVENT_BYTES",
    "actor_id",
    "actor_json",
    "clock",
    "compact_json",
    "legacy_members_match",
    "payload_json",
    "replica_event_bytes",
    "room_grant_scope_key",
    "room_id",
    "system_actor_json",
    "utf8_len",
    "validate_members",
    "validate_room_name",
]

room_id = neutral._room_id
actor_json = neutral._actor_json
payload_json = neutral._payload_json
validate_members = neutral._validate_members
replica_event_bytes = neutral._event_bytes
CONTROL_EVENT_KINDS = neutral._CONTROL_EVENT_KINDS
validate_room_name = neutral._validate_room_name
actor_id = neutral._actor_id
system_actor_json = neutral._system_actor_json
legacy_members_match = neutral._legacy_members_match


def room_grant_scope_key(claims: RoomGrantClaims) -> str:
    """The grant scope key over the typed claims' fields."""
    return neutral.room_grant_scope_key(dataclasses.asdict(claims))
