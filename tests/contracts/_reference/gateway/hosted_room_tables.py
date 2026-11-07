"""The hosted-room table holder of the in-memory reference, shared by every hosted-room repository.

One placement's hosted-room tables live under the ``"hosted_room_tables"`` key of one namespace: the bound profile's store namespace (``store.scope_store(store.scope)``) for ``'profile'``, and the backend's namespace of the derived root ``shared-state.db`` (``store.backend.file_store(shared_state_path(...))``) for ``'shared'``, so two profiles of one install reach the same shared tables. Inside, each table is a dict under its table name; a per-room table is a dict keyed by ``room_id`` whose value holds that room's rows (``hosted_room_events[room_id][seq]``, ``hosted_room_links[room_id][member_id]``, ``hosted_room_driver_tasks[room_id][task_id]``, ...), so the purge of a disbanded room pops one key per table. ``hosted_rooms``, ``hosted_room_replicas``, ``hosted_room_driver_leases``, ``hosted_room_policy_cursors``, ``hosted_room_policy_transcript_state`` and ``hosted_room_gateway_handoffs`` map ``room_id`` to one row; the tables not keyed by room (``hosted_room_revoked_grants``, ``hosted_room_attachment_blobs``) map their own key to a row, and ``hosted_room_policy_capacity`` maps ``1`` to the file's recorded active-event capacity. Rows are dicts keyed by column name.

A write runs on a deep copy of the namespace and publishes it whole when the verb returns, so a refusal anywhere leaves the tables as they were. Each verb fetches the namespace again: the authority transaction of the profile store swaps its namespace's values.

The functions below are the shared steps over these tables: the room row read with its not-found errors, the authority check, event sizing and capacity, the retention prune with its dependent-row purge, and the room and event dicts.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from persistence.gateway.hosted_rooms import (
    AuthorityConflictError,
    DriverValidationError,
    HostedPlacement,
    HostedRoomError,
    HostedTaskIdentity,
    RoomHistoryExpiredError,
    RoomNotFoundError,
    TaskConflictError,
)
from session_rules.hosted_tasks import normalize_task_payload

from . import hosted_room_rules as rules

__all__ = [
    "DEPENDENT_TABLES",
    "TABLES",
    "HostedTables",
    "event_from_row",
    "find_event",
    "gateway_event_bytes",
    "insert_event",
    "peek_tables",
    "prepare_event",
    "prune_disbanded_rooms_locked",
    "read_tables",
    "require_authority",
    "retire",
    "room_events",
    "room_from_row",
    "room_row",
    "shared_state_path",
    "task_view",
    "write_tables",
]

_NAMESPACE_KEY = "hosted_room_tables"

TABLES = frozenset({
    "hosted_rooms", "hosted_room_events", "hosted_room_retired_ids", "hosted_room_links", "hosted_room_remote_runs",
    "hosted_room_revoked_grants", "hosted_room_peer_reservations", "hosted_room_replicas", "hosted_room_replica_events",
    "hosted_room_driver_leases", "hosted_room_driver_tasks", "hosted_room_policy_cursors", "hosted_room_policy_threads",
    "hosted_room_policy_events", "hosted_room_policy_watermarks", "hosted_room_policy_publications",
    "hosted_room_policy_transcript", "hosted_room_policy_transcript_state", "hosted_room_policy_capacity",
    "hosted_room_attachments", "hosted_room_attachment_blobs", "hosted_room_gateway_handoffs"})

# Purged in this order with a disbanded room.
DEPENDENT_TABLES = (
    "hosted_room_policy_transcript_state", "hosted_room_policy_transcript", "hosted_room_policy_publications",
    "hosted_room_policy_watermarks", "hosted_room_policy_events", "hosted_room_policy_threads",
    "hosted_room_policy_cursors", "hosted_room_driver_tasks", "hosted_room_driver_leases", "hosted_room_remote_runs",
    "hosted_room_links", "hosted_room_peer_reservations", "hosted_room_events")


def shared_state_path(profile_dir: Path) -> Path:
    """The install's root ``shared-state.db`` derived from a profile folder."""
    return (profile_dir.parent.parent if profile_dir.parent.name == "profiles" else profile_dir) / "shared-state.db"


class HostedTables:
    """One placement's hosted-room tables (module docstring); ``tables[name]`` is the named table, created on first touch."""

    def __init__(self, namespace: dict[str, Any]) -> None:
        self._namespace = namespace

    def __getitem__(self, name: str) -> dict[Any, Any]:
        if name not in TABLES:
            raise KeyError(name)
        table: dict[Any, Any] = self._namespace.setdefault(name, {})
        return table

    def rows(self, name: str, room_id: str) -> dict[Any, Any]:
        """The rows of one room in a per-room table, created on first touch."""
        rows: dict[Any, Any] = self[name].setdefault(room_id, {})
        return rows


def _parent(store: Any, placement: HostedPlacement) -> dict[str, Any]:
    if placement == "profile":
        namespace: dict[str, Any] = store.scope_store(store.scope)
        return namespace
    if placement == "shared":
        shared: dict[str, Any] = store.backend.file_store(shared_state_path(Path(store.scope.profile_dir)))
        return shared
    raise ValueError(f"unknown hosted-room placement {placement!r}")


def read_tables(store: Any, placement: HostedPlacement) -> HostedTables:
    """The committed tables of ``placement`` for a read; opening the file creates its tables."""
    return HostedTables(_parent(store, placement).setdefault(_NAMESPACE_KEY, {}))


def peek_tables(store: Any, placement: HostedPlacement) -> HostedTables | None:
    """The committed tables of ``placement``, or None when the file was never opened (the probes never create it)."""
    namespace = _parent(store, placement).get(_NAMESPACE_KEY)
    return None if namespace is None else HostedTables(namespace)


@contextmanager
def write_tables(store: Any, placement: HostedPlacement) -> Iterator[HostedTables]:
    """One write: the block works on a copy of the tables, published only when the block returns."""
    parent = _parent(store, placement)
    staged = copy.deepcopy(parent.setdefault(_NAMESPACE_KEY, {}))
    yield HostedTables(staged)
    parent[_NAMESPACE_KEY] = staged


def room_from_row(row: Mapping[str, Any], *, idempotent: bool = False, latest_seq: bool = True) -> dict[str, Any]:
    """The room dict of a room row; ``latest_seq=False`` is the fresh-create reload, which selects no ``next_seq``."""
    room = {
        "room_id": row["room_id"], "name": row["name"], "members": json.loads(row["members_json"]),
        "authority_gateway_id": row["authority_gateway_id"], "authority_epoch": int(row["authority_epoch"]),
        "revision": int(row["revision"]), "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]), "idempotent": idempotent}
    if latest_seq and row["disbanded_at"] is not None:
        room["disbanded_at"] = float(row["disbanded_at"])
    if latest_seq:
        room["latest_seq"] = int(row["next_seq"]) - 1
    return room


def event_from_row(row: Mapping[str, Any], *, idempotent: bool = False) -> dict[str, Any]:
    """The event dict of an event row, its actor and payload decoded."""
    epoch = row["authority_epoch"]
    return {
        "room_id": row["room_id"], "seq": int(row["seq"]), "event_id": row["event_id"], "kind": row["kind"],
        "actor": json.loads(row["actor_json"]), "authority_epoch": int(epoch) if epoch is not None else None,
        "payload": json.loads(row["payload_json"]), "created_at": float(row["created_at"]), "idempotent": idempotent}


def room_events(tables: HostedTables, room_id: str) -> list[dict[str, Any]]:
    """The room's event rows in sequence order."""
    rows = tables["hosted_room_events"].get(room_id, {})
    return [rows[seq] for seq in sorted(rows)]


def find_event(tables: HostedTables, room_id: str, event_id: str) -> dict[str, Any] | None:
    """The event row of ``(room_id, event_id)``, or None."""
    return next((row for row in tables["hosted_room_events"].get(room_id, {}).values() if row["event_id"] == event_id), None)


def room_row(tables: HostedTables, room_id: str, *, include_disbanded: bool) -> dict[str, Any]:
    """The room row, else the expired-history or not-found refusal."""
    row = tables["hosted_rooms"].get(room_id)
    if row is not None and (include_disbanded or row["disbanded_at"] is None):
        return row
    if row is None and room_id in tables["hosted_room_retired_ids"]:
        raise RoomHistoryExpiredError("Group Chat history expired; room_id remains permanently retired")
    raise RoomNotFoundError("hosted room not found")


def require_authority(room: Mapping[str, Any], gateway_id: str, epoch: int, message: str) -> None:
    """``AuthorityConflictError(message)`` unless the room's authority is ``(gateway_id, epoch)``."""
    if str(room["authority_gateway_id"]) != gateway_id or int(room["authority_epoch"]) != epoch:
        raise AuthorityConflictError(message)


def gateway_event_bytes(tables: HostedTables) -> int:
    """The accounted event bytes of every room row, disbanded ones included."""
    return sum(int(row["event_bytes"]) for row in tables["hosted_rooms"].values())


def prepare_event(tables: HostedTables, room: Mapping[str, Any], event_id: str, kind: str, actor_json: str, payload_json: str, *, allow_control: bool = False) -> int:
    """Size one pending event against the room and gateway budgets, pruning disbanded rooms once when the gateway budget is short; returns its bytes."""
    additional = rules.utf8_len(event_id, kind, actor_json, payload_json)
    count_reserve, byte_reserve = (rules.CONTROL_EVENT_COUNT_RESERVE, rules.CONTROL_EVENT_BYTE_RESERVE) if allow_control else (0, 0)
    gateway_limit = rules.MAX_GATEWAY_EVENT_BYTES + byte_reserve
    if int(room["next_seq"]) - 1 >= rules.MAX_EVENTS_PER_ROOM + count_reserve:
        raise HostedRoomError("This Group Chat reached its history limit. Start a new Group Chat to continue.")
    if int(room["event_bytes"]) + additional > rules.MAX_ROOM_EVENT_BYTES + byte_reserve:
        raise HostedRoomError("This Group Chat reached its storage limit. Start a new Group Chat to continue.")
    if gateway_event_bytes(tables) + additional > gateway_limit:
        prune_disbanded_rooms_locked(tables, now=None, max_gateway_event_bytes=max(0, gateway_limit - additional))
    if gateway_event_bytes(tables) + additional > gateway_limit:
        raise HostedRoomError("Group Chat storage is full on this host. Delete an old Group Chat and try again.")
    return additional


def insert_event(tables: HostedTables, room_id: str, seq: int, event_id: str, kind: str, actor_json: str, epoch: int | None, payload_json: str, now: float) -> None:
    """The ``hosted_room_events`` INSERT."""
    tables.rows("hosted_room_events", room_id)[seq] = {
        "room_id": room_id, "seq": seq, "event_id": event_id, "kind": kind, "actor_json": actor_json,
        "authority_epoch": epoch, "payload_json": payload_json, "created_at": now}


def retire(tables: HostedTables, room_id: str, retired_at: float) -> None:
    """``INSERT OR IGNORE`` of a permanently retired room id."""
    tables["hosted_room_retired_ids"].setdefault(room_id, float(retired_at))


def _disbanded(tables: HostedTables) -> list[dict[str, Any]]:
    return [row for row in tables["hosted_rooms"].values() if row["disbanded_at"] is not None]


def _byte_candidates(tables: HostedTables, max_gateway_event_bytes: int) -> set[str]:
    retained = gateway_event_bytes(tables)
    candidates: set[str] = set()
    if retained <= max_gateway_event_bytes:
        return candidates
    for row in sorted(_disbanded(tables), key=lambda row: (row["disbanded_at"], row["room_id"])):
        candidates.add(row["room_id"])
        retained -= int(row["event_bytes"])
        if retained <= max_gateway_event_bytes:
            break
    return candidates


def prune_disbanded_rooms_locked(tables: HostedTables, *, now: float | None, max_gateway_event_bytes: int | None = None) -> int:
    """The age pass (when ``now`` is set), the tombstone cap and the byte bound choose disbanded rooms; each keeps its retired id and loses its row and dependent rows."""
    disbanded = _disbanded(tables)
    candidates: set[str] = set()
    if now is not None:
        candidates.update(row["room_id"] for row in disbanded if row["disbanded_at"] <= now - rules.DISBANDED_ROOM_RETENTION_SECONDS)
    newest_first = sorted(disbanded, key=lambda row: (-row["disbanded_at"], row["room_id"]))
    candidates.update(row["room_id"] for row in newest_first[rules.MAX_DISBANDED_ROOM_TOMBSTONES:])
    if max_gateway_event_bytes is not None:
        candidates.update(_byte_candidates(tables, max_gateway_event_bytes))
    for room_id in sorted(candidates):
        retire(tables, room_id, tables["hosted_rooms"][room_id]["disbanded_at"])
    for name in DEPENDENT_TABLES:
        for room_id in candidates:
            tables[name].pop(room_id, None)
    for room_id in candidates:
        del tables["hosted_rooms"][room_id]
    return len(candidates)


def _optional(cast: Any) -> Any:
    return lambda value: cast(value) if value is not None else None


# The view casts of the columns after ``payload_digest``; ``result_json`` is exposed as ``result``.
_TASK_VIEW_COLUMNS = (
    "payload_digest", "status", "execution_generation", "cancel_generation", "run_gateway_id", "run_process_generation",
    "run_lease_generation", "cancel_id", "settlement_id", "settlement_status", "result_json", "created_at", "updated_at",
    "started_at", "terminal_at", "indeterminate_at")
_TASK_VIEW_CASTS: dict[str, Any] = {
    "execution_generation": int, "cancel_generation": int, "run_lease_generation": _optional(int),
    "result_json": _optional(json.loads), "created_at": float, "updated_at": float, "started_at": _optional(float),
    "terminal_at": _optional(float), "indeterminate_at": _optional(float)}


def task_view(row: Mapping[str, Any], *, idempotent: bool = False) -> dict[str, Any]:
    """The stored payload re-normalized and checked against its stored JSON, digest and ``source_event_seq`` (``TaskConflictError`` otherwise), then the task view the driver verbs return and the member check reads."""
    try:
        payload, encoded, digest = normalize_task_payload(json.loads(row["payload_json"]))
    except (TypeError, json.JSONDecodeError, DriverValidationError) as exc:
        raise TaskConflictError("stored task payload is invalid") from exc
    if (encoded, digest, payload["source_event_seq"]) != (row["payload_json"], row["payload_digest"], int(row["source_event_seq"])):
        raise TaskConflictError("stored task payload failed its integrity check")
    task: dict[str, Any] = {"identity": HostedTaskIdentity(row["room_id"], row["task_id"], row["thread_id"], row["turn_id"]), "payload": payload}
    for column in _TASK_VIEW_COLUMNS:
        task["result" if column == "result_json" else column] = _TASK_VIEW_CASTS.get(column, lambda value: value)(row[column])
    task["idempotent"] = idempotent
    return task
