"""The hand-off of a standalone backend's rooms on SQLite: the hand-off table, which a ``shared-state.db`` gains only when a hand-off verb runs on it, as main's gateway import creates it, and the copy of a fenced room's own rows into the gateway's ``state.db`` as main's hand-off copies them (tables the profile lacks are created from the shared file's own DDL; driver leases stay behind)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from persistence.gateway.hosted_rooms import HostedTaskIdentity
from tests.persistence_sqlite.hosted.conftest import NOW, dump, tables


pytestmark = pytest.mark.usefixtures("clock")

MEMBERS = [{"member_id": "m1", "profile": "p1"}]
USER = {"kind": "user", "id": "u1"}
LOCAL, HANDOFF = "install:local", "install:local:gateway-store"
PAYLOAD = {"target_profile": "p1", "target_member_id": "m1", "prompt": "go", "source_event_seq": 1}
# The hand-off table as main's gateway import creates it in shared-state.db.
HANDOFF_SQL = "CREATE TABLE hosted_room_gateway_handoffs (room_id TEXT PRIMARY KEY, handed_at REAL NOT NULL)"


def _table_sql(path: Path, name: str) -> str | None:
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return None if row is None else row[0]


def _read_rooms(store) -> None:
    assert [room["room_id"] for room in store.hosted_rooms.list_rooms(placement="shared")] == ["old"]
    assert all(event["room_id"] == "old"
               for event in store.hosted_rooms.read_events(placement="shared", room_id="old")["events"])
    assert store.hosted_rooms.handed_room_ids(handoff_gateway_id=HANDOFF) == frozenset()


def test_no_room_verb_creates_the_table_until_a_handoff_runs(backend, scopes, root):
    shared = root / "shared-state.db"
    store = backend.open(scopes["a"])
    store.hosted_rooms.create_room(placement="shared", room_id="old", name="Old", members=MEMBERS,
                                   authority_gateway_id=LOCAL, now=NOW)
    store.hosted_rooms.create_room(placement="profile", room_id="mine", name="Mine", members=MEMBERS,
                                   authority_gateway_id=LOCAL, now=NOW)
    assert _table_sql(shared, "hosted_room_gateway_handoffs") is None
    before = shared.read_bytes()
    _read_rooms(store)
    assert shared.read_bytes() == before
    store.hosted_rooms.record_room_handoff(room_ids=frozenset(), now=NOW)
    assert _table_sql(shared, "hosted_room_gateway_handoffs") == HANDOFF_SQL
    assert "hosted_room_gateway_handoffs" not in tables(scopes["a"].profile_dir / "state.db")


def test_import_copies_the_rooms_own_rows_except_its_driver_lease(backend, scopes, root):
    store = backend.open(scopes["root"])
    shared, profile = root / "shared-state.db", root / "state.db"
    store.hosted_rooms.create_room(placement="shared", room_id="desk", name="Desk", members=MEMBERS,
                                   authority_gateway_id=LOCAL, now=NOW)
    store.hosted_rooms.append_event(
        placement="shared", room_id="desk", event_id="e1", kind="message.user", actor=USER,
        payload={"text": "go", "thread_id": "thread-1"}, authority_gateway_id=LOCAL, authority_epoch=1, now=NOW)
    store.hosted_room_driver.acquire_lease(
        placement="shared", room_id="desk", gateway_id=LOCAL, authority_epoch=1, process_generation="p-1",
        ttl_seconds=60, now=NOW)
    store.hosted_room_driver.admit_task(
        HostedTaskIdentity("desk", "task-1", "thread-1", "turn-1"), placement="shared", payload=PAYLOAD, now=NOW)
    store.hosted_room_replicas.demote_room(
        placement="shared", room_id="desk", observed_gateway_id=HANDOFF, observed_epoch=2, local_gateway_id=LOCAL,
        now=NOW)
    assert "hosted_room_driver_tasks" not in tables(profile)

    assert store.hosted_rooms.import_fenced_rooms(
        epoch=None, room_ids=frozenset({"desk"}), owner="alice", local_gateway_id=LOCAL, handoff_gateway_id=HANDOFF,
        now=NOW) == 1

    assert _table_sql(profile, "hosted_room_driver_tasks") == _table_sql(shared, "hosted_room_driver_tasks")
    assert dump(profile, ("hosted_room_driver_tasks",)) == dump(shared, ("hosted_room_driver_tasks",))
    assert dump(shared, ("hosted_room_driver_leases",))["hosted_room_driver_leases"]
    assert "hosted_room_driver_leases" not in tables(profile)
