"""Legacy Group Chat migration preserves the configured policy replay capacity (#149447)."""

import shutil
import sqlite3

from gateway import hosted_rooms
from gateway import hosted_rooms_legacy_import as legacy_import
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint


def test_pre_isolation_rooms_keep_policy_replay_capacity(tmp_path):
    legacy, shared = tmp_path / "state.db", tmp_path / "shared-state.db"
    hosted_rooms.create_room(
        legacy,
        room_id="pre-isolation-room",
        name="Pre-isolation room",
        members=[
            {"member_id": "a", "profile": "default", "handle": "a"},
            {"member_id": "b", "profile": "default", "handle": "b"},
        ],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
    )
    assert HostedRoomPolicyCheckpoint(legacy, max_active_events=128).max_active_events == 128

    assert [room["room_id"] for room in hosted_rooms.list_rooms(shared)] == ["pre-isolation-room"]
    assert HostedRoomPolicyCheckpoint(shared).max_active_events == 128


def test_target_default_capacity_is_merged_rather_than_collided(tmp_path):
    """The real upgrade shape: the target already holds its seeded singleton when the copy runs.

    A live ``state.db`` seeds ``hosted_room_policy_capacity(1, 64)`` the first time a checkpoint opens,
    so the copy meets a row that already exists — a raw INSERT died on ``UNIQUE constraint failed``
    and rolled the whole migration back (#149447). The source's larger value must win, not collide.
    """
    legacy, shared = tmp_path / "state.db", tmp_path / "shared-state.db"
    hosted_rooms.create_room(
        legacy,
        room_id="pre-isolation-room",
        name="Pre-isolation room",
        members=[
            {"member_id": "a", "profile": "default", "handle": "a"},
            {"member_id": "b", "profile": "default", "handle": "b"},
        ],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
    )
    assert HostedRoomPolicyCheckpoint(legacy, max_active_events=128).max_active_events == 128

    # Same schema, stripped to the shape a freshly seeded target has: no rooms, default capacity.
    shutil.copy(legacy, shared)
    with sqlite3.connect(shared) as conn:
        conn.execute("DELETE FROM hosted_room_events")
        conn.execute("DELETE FROM hosted_rooms")
        conn.execute("UPDATE hosted_room_policy_capacity SET max_active_events=64")
        conn.commit()
        assert not legacy_import.settled(conn, shared)

        legacy_import.import_legacy_rooms(conn, shared)

        capacity = conn.execute("SELECT max_active_events FROM hosted_room_policy_capacity").fetchone()[0]
        rooms = [row[0] for row in conn.execute("SELECT room_id FROM hosted_rooms")]
    assert rooms == ["pre-isolation-room"]
    assert capacity == 128
