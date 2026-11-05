"""The reference hand-off verbs of ``HostedRoomRepository``: the hand-off record in the shared file's ``hosted_room_gateway_handoffs`` (``room_id`` to its row) and the copy and claim of fenced rooms into the bound profile's tables.

A copy takes every room-keyed table except the driver leases; the tables keyed by something else (revocations, attachment blobs, the policy capacity) are never copied. A table whose value per room is one row conflicts when the profile already holds that room's row; a table of rows per room conflicts on a shared inner key. A conflict raises the store's ``PermanentStoreError`` and leaves the profile as it was.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection
from typing import Any

from persistence.core.errors import PermanentStoreError

from . import hosted_room_rules as rules
from .hosted_room_replicas import _control_event
from .hosted_room_tables import TABLES, HostedTables, insert_event, read_tables, write_tables

__all__ = ["HostedRoomHandoffMixin"]

HANDOFF_TABLE = "hosted_room_gateway_handoffs"
# Tables keyed by room whose value is that room's one row.
_ONE_ROW = frozenset({
    "hosted_rooms", "hosted_room_retired_ids", "hosted_room_replicas", "hosted_room_policy_cursors",
    "hosted_room_policy_transcript_state", HANDOFF_TABLE})
_NOT_COPIED = frozenset({
    "hosted_room_driver_leases", "hosted_room_revoked_grants", "hosted_room_attachment_blobs",
    "hosted_room_policy_capacity"})


def _copy_room(source: HostedTables, target: HostedTables, room_id: str) -> None:
    """Copy one room's rows of every copied table, the room row first."""
    for name in sorted(TABLES - _NOT_COPIED, key=lambda name: (name != "hosted_rooms", name)):
        rows = source[name].get(room_id)
        if rows is None:
            continue
        held = target[name]
        if name in _ONE_ROW:
            if room_id in held:
                raise PermanentStoreError(f"UNIQUE constraint failed: {name}.room_id")
            held[room_id] = copy.deepcopy(rows)
            continue
        mine = held.setdefault(room_id, {})
        if mine.keys() & rows.keys():
            raise PermanentStoreError(f"UNIQUE constraint failed: {name}")
        mine.update(copy.deepcopy(rows))


def _claim(tables: HostedTables, room_id: str, *, local_gateway_id: str, handoff_gateway_id: str, now: float) -> None:
    """The ``authority.claimed`` event at the room's next sequence and epoch + 1, and the room's new authority."""
    row = tables["hosted_rooms"][room_id]
    epoch = int(row["authority_epoch"]) + 1
    event_id, kind, actor_json, payload_json = _control_event("claimed", epoch, {
        "previous_gateway_id": handoff_gateway_id, "authority_gateway_id": local_gateway_id, "authority_epoch": epoch})
    insert_event(tables, room_id, int(row["next_seq"]), event_id, kind, actor_json, epoch, payload_json, now)
    row.update(authority_gateway_id=local_gateway_id, authority_epoch=epoch, next_seq=int(row["next_seq"]) + 1,
               revision=int(row["revision"]) + 1, updated_at=now)


class HostedRoomHandoffMixin:
    """The hand-off verbs; the repository provides ``_store``, ``_meta`` and ``_require_epoch``."""

    _store: Any
    _meta: Callable[[], dict[str, str]]
    _require_epoch: Callable[[int], None]

    def fenced_room_ids(self, *, handoff_gateway_id: str) -> frozenset[str]:
        tables = read_tables(self._store, "shared")
        return frozenset(
            room_id for room_id, row in tables["hosted_rooms"].items()
            if row["authority_gateway_id"] == handoff_gateway_id and row["disbanded_at"] is None
            and room_id not in tables[HANDOFF_TABLE])

    def record_room_handoff(self, *, room_ids: Collection[str], now: float | None = None) -> None:
        timestamp = rules.clock(now)
        with write_tables(self._store, "shared") as tables:
            for room_id in sorted(room_ids):
                tables[HANDOFF_TABLE].setdefault(room_id, {"room_id": room_id, "handed_at": timestamp})

    def handed_room_ids(self, *, handoff_gateway_id: str) -> frozenset[str]:
        rooms = read_tables(self._store, "shared")["hosted_rooms"]
        return frozenset(room_id for room_id, row in rooms.items() if row["authority_gateway_id"] == handoff_gateway_id)

    def import_fenced_rooms(self, *, epoch: int | None, room_ids: Collection[str], owner: str, local_gateway_id: str, handoff_gateway_id: str, now: float | None = None) -> int:
        timestamp = rules.clock(now)
        source = read_tables(self._store, "shared")
        with write_tables(self._store, "profile") as tables:
            if epoch is not None:
                self._require_epoch(epoch)
            known = set(tables["hosted_rooms"]) | set(tables["hosted_room_retired_ids"])
            copied = sorted(room_id for room_id in set(room_ids) - known if room_id in source["hosted_rooms"])
            for room_id in copied:
                _copy_room(source, tables, room_id)
            for room_id in copied:
                _claim(tables, room_id, local_gateway_id=local_gateway_id, handoff_gateway_id=handoff_gateway_id,
                       now=timestamp)
        from .hosted_rooms import OWNER_PREFIX

        meta = self._meta()
        for room_id in copied:
            meta.setdefault(OWNER_PREFIX + room_id, owner)
        return len(copied)
