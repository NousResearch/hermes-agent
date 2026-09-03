"""Validation and public error contract for gateway-hosted Group Chats."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 2
MAX_ROOM_ID_CHARS = 128
MAX_EVENT_ID_CHARS = 128
MAX_ROOM_NAME_CHARS = 200
MAX_EVENT_KIND_CHARS = 64
MAX_ACTOR_ID_CHARS = 128
MAX_ACTOR_LABEL_CHARS = 200
MAX_MEMBERS = 128
MAX_MEMBERS_JSON_BYTES = 128 * 1024
MAX_EVENT_JSON_BYTES = 256 * 1024
MAX_LOG_LIMIT = 500
MAX_LOG_PAGE_BYTES = 2 * 1024 * 1024
MAX_ROOM_LIST_LIMIT = 500
MAX_ACTIVE_ROOMS = 256
MAX_DISBANDED_ROOM_TOMBSTONES = 512
DISBANDED_ROOM_RETENTION_SECONDS = 90 * 24 * 60 * 60
MAX_EVENTS_PER_ROOM = 50_000
MAX_ROOM_EVENT_BYTES = 256 * 1024 * 1024
# Leave substantial headroom below the pre-update state.db snapshot ceiling.
# Event accounting does not include SQLite indexes or repeated room ids, so the
# logical budget must stay well below the physical-file limit.
MAX_GATEWAY_EVENT_BYTES = 16 * 1024 * 1024
CONTROL_EVENT_COUNT_RESERVE = 64
CONTROL_EVENT_BYTE_RESERVE = 1024 * 1024
_JOURNAL_MODE_LOCK_RETRIES = 8

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_EVENT_KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_ROOM_SCHEMA_COLUMNS = frozenset({
    "room_id",
    "name",
    "members_json",
    "authority_gateway_id",
    "authority_epoch",
    "next_seq",
    "event_bytes",
    "revision",
    "created_at",
    "updated_at",
    "disbanded_at",
})
_EVENT_SCHEMA_COLUMNS = frozenset({
    "room_id",
    "seq",
    "event_id",
    "kind",
    "actor_json",
    "authority_epoch",
    "payload_json",
    "created_at",
})
_RETIRED_ROOM_SCHEMA_COLUMNS = frozenset({"room_id", "retired_at"})
_LINK_SCHEMA_COLUMNS = frozenset({
    "room_id",
    "member_id",
    "target_url",
    "target_profile",
    "grant",
    "catalog_json",
    "cancellation_scope_id",
    "trace_id",
    "transport_security",
    "status",
    "updated_at",
})
_REMOTE_RUN_SCHEMA_COLUMNS = frozenset({
    "room_id",
    "home_install_id",
    "authority_gateway_id",
    "authority_epoch",
    "member_id",
    "task_id",
    "execution_generation",
    "target_install_id",
    "target_profile",
    "run_id",
    "session_id",
    "created_at",
    "updated_at",
})
_REMOTE_RUN_IDENTITY_COLUMNS = (
    "room_id",
    "home_install_id",
    "authority_gateway_id",
    "authority_epoch",
    "member_id",
    "target_install_id",
    "target_profile",
    "task_id",
    "execution_generation",
)
_REVOKED_GRANT_SCHEMA_COLUMNS = frozenset({
    "scope_key",
    "expires_at",
    "revoked_before",
})
_PEER_RESERVATION_SCHEMA_COLUMNS = frozenset({
    "room_id",
    "member_id",
    "target_profile",
    "authority_gateway_id",
    "authority_epoch",
    "expires_at",
    "revoked_at",
    "created_at",
    "updated_at",
})

_EVENT_KINDS_BY_ACTOR = {
    "user": frozenset({"message.user"}),
    "member": frozenset({"message.member"}),
    "gateway": frozenset({
        "member.unavailable",
        "room.activity",
        "room.stop_requested",
        "turn.deferred",
        "turn.reassigned",
        "turn.cancelled",
        "turn.failed",
        "turn.settled",
        "turn.started",
    }),
    "system": frozenset({
        "authority.claimed",
        "authority.lost",
        "room.created",
        "room.disbanded",
        "room.members_changed",
        "room.renamed",
    }),
}
_ACTOR_FIELDS = frozenset({"kind", "id", "display_name", "profile", "connection_id"})


class HostedRoomError(ValueError):
    """Base class for invalid or conflicting hosted-room operations."""


class RoomNotFoundError(HostedRoomError):
    """Raised when a room does not exist or has been disbanded."""


class RoomHistoryExpiredError(RoomNotFoundError):
    """Raised when a retired room remains reserved after history compaction."""

    reason = "room_history_expired"


class RoomConflictError(HostedRoomError):
    """Raised when an idempotency key is reused for different room state."""


class RoomProbeUnavailableError(HostedRoomError):
    """Raised when a non-blocking ownership probe cannot read the room store."""


class EventConflictError(HostedRoomError):
    """Raised when an event id is reused with different immutable content."""


class AuthorityConflictError(HostedRoomError):
    """Raised when a stale room authority attempts to mutate hosted state."""

    reason = "authority_conflict"


class AuthoritySupersededError(AuthorityConflictError):
    """Raised when a successful authority claim was later superseded."""


def _public_limits():
    """Resolve re-exported limits late to preserve the public patch seam."""
    from gateway import hosted_rooms

    return hosted_rooms


def default_db_path() -> Path:
    """Return the gateway-wide state database for the active install."""
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    root = home.parent.parent if home.parent.name == "profiles" else home
    return root / "state.db"


def local_authority_gateway_id() -> str:
    """Return the stable server-owned identity for hosted-room authority."""
    from hermes_cli.install_identity import get_install_id

    install_id = get_install_id()
    if not install_id:
        raise HostedRoomError("stable gateway install identity is unavailable")
    return _validate_identifier(
        f"install:{install_id}",
        label="authority_gateway_id",
        max_chars=_public_limits().MAX_ACTOR_ID_CHARS,
    )


def _canonical_json(value: Any, *, label: str, max_bytes: int) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise HostedRoomError(f"{label} must be JSON-serializable") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise HostedRoomError(f"{label} is too large")
    return encoded


def _validate_identifier(value: Any, *, label: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise HostedRoomError(f"{label} must be a string")
    value = value.strip()
    if not value or len(value) > max_chars or not _IDENTIFIER_RE.fullmatch(value):
        raise HostedRoomError(f"invalid {label}")
    return value


def user_event_id(client_event_id: Any) -> str:
    """Map a client retry key into the server-owned user-event namespace."""
    normalized = _validate_identifier(
        client_event_id,
        label="event_id",
        max_chars=_public_limits().MAX_EVENT_ID_CHARS,
    )
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"user:{digest}"


def _validate_room_name(value: Any) -> str:
    if not isinstance(value, str):
        raise HostedRoomError("name must be a string")
    value = value.strip()
    if not value or len(value) > _public_limits().MAX_ROOM_NAME_CHARS:
        raise HostedRoomError("invalid room name")
    return value


def _validate_members(value: Any) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(value, list):
        raise HostedRoomError("members must be a list")
    limits = _public_limits()
    if len(value) > limits.MAX_MEMBERS:
        raise HostedRoomError("too many room members")
    members: list[dict[str, Any]] = []
    for member in value:
        if not isinstance(member, dict):
            raise HostedRoomError("each room member must be an object")
        members.append(dict(member))
    encoded = _canonical_json(
        members,
        label="members",
        max_bytes=limits.MAX_MEMBERS_JSON_BYTES,
    )
    return members, encoded


def _legacy_members_match(
    existing_json: str,
    proposed: list[dict[str, Any]],
) -> bool:
    """Allow adoption to add routing metadata an older room could not store."""

    try:
        existing = json.loads(existing_json)
    except (TypeError, ValueError):
        return False
    if not isinstance(existing, list) or len(existing) != len(proposed):
        return False
    for previous, current in zip(existing, proposed, strict=True):
        if not isinstance(previous, dict):
            return False
        previous = dict(previous)
        current = dict(current)
        previous_target = previous.pop("target", None)
        current_target = current.pop("target", None)
        if previous != current:
            return False
        if previous_target not in (None, {}) and previous_target != current_target:
            return False
    return True


def _validate_event_kind(value: Any) -> str:
    if not isinstance(value, str):
        raise HostedRoomError("kind must be a string")
    value = value.strip()
    if (
        not value
        or len(value) > _public_limits().MAX_EVENT_KIND_CHARS
        or not _EVENT_KIND_RE.fullmatch(value)
    ):
        raise HostedRoomError("invalid event kind")
    return value


def _optional_actor_field(actor: dict[str, Any], field: str, max_chars: int) -> str:
    value = actor.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise HostedRoomError(f"actor.{field} must be a string")
    value = value.strip()
    if len(value) > max_chars:
        raise HostedRoomError(f"actor.{field} is too long")
    return value


def _validate_actor(value: Any, *, kind: str) -> tuple[dict[str, str], str]:
    if not isinstance(value, dict):
        raise HostedRoomError("actor must be an object")
    unknown = set(value) - _ACTOR_FIELDS
    if unknown:
        raise HostedRoomError(f"unknown actor fields: {', '.join(sorted(unknown))}")

    actor_kind = value.get("kind")
    if not isinstance(actor_kind, str) or actor_kind not in _EVENT_KINDS_BY_ACTOR:
        raise HostedRoomError("invalid actor.kind")
    if kind not in _EVENT_KINDS_BY_ACTOR[actor_kind]:
        raise HostedRoomError(f"actor kind '{actor_kind}' cannot append '{kind}'")

    limits = _public_limits()
    actor_id = _validate_identifier(
        value.get("id"),
        label="actor.id",
        max_chars=limits.MAX_ACTOR_ID_CHARS,
    )
    actor = {"kind": actor_kind, "id": actor_id}
    for field, max_chars in (
        ("display_name", limits.MAX_ACTOR_LABEL_CHARS),
        ("profile", limits.MAX_ACTOR_ID_CHARS),
        ("connection_id", limits.MAX_ACTOR_ID_CHARS),
    ):
        field_value = _optional_actor_field(value, field, max_chars)
        if field_value:
            actor[field] = field_value
    encoded = _canonical_json(
        actor,
        label="actor",
        max_bytes=4 * 1024,
    )
    return actor, encoded
