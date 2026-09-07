"""Every hosted-room opener of state.db must be visible to the live-connection registry.

``is_zeroed_state_db`` consults ``has_live_connection`` before it quarantines
a 0-byte state.db, so an untracked opener is exactly the connection a
concurrent ``SessionDB`` start-up would rename out from under (the SIGBUS
race pinned for the hosted-room store in
``tests/hermes_cli/test_web_server_statedb_boot_race.py``). The policy
checkpoint and the driver open the same file through their own connect
helpers; this file pins the same contract for them: tracked while open,
released once the operation has closed its connection.

The "released afterwards" half matters as much as the first: a tracked
connection that is dropped without ``close()`` keeps its registry entry for
the rest of the process, which would pin ``has_live_connection`` to True and
quietly disable the zeroed-file guard for every later opener.
"""

from __future__ import annotations

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms as rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
from hermes_cli.sqlite_safe_read import has_live_connection


def _create_room(db):
    rooms.create_room(
        db,
        room_id="room-1",
        name="Release room",
        members=[{"profile": "ops", "handle": "ops"}],
        authority_gateway_id="gateway-a",
        now=10,
    )


def test_policy_checkpoint_connections_are_tracked_while_open(tmp_path, monkeypatch):
    """Schema init and every steady-state operation open through the registry."""
    db = tmp_path / "state.db"
    _create_room(db)
    assert not has_live_connection(db)

    seen: list[bool] = []
    real_connect = HostedRoomPolicyCheckpoint._connect

    def connect_and_observe(self):
        conn = real_connect(self)
        seen.append(has_live_connection(db))
        return conn

    monkeypatch.setattr(HostedRoomPolicyCheckpoint, "_connect", connect_and_observe)

    checkpoint = HostedRoomPolicyCheckpoint(db)  # one open: schema initialisation
    assert seen == [True]
    assert not has_live_connection(db)

    checkpoint.compact_completed(room_id="room-1")  # one open: a steady-state write
    assert seen == [True, True]
    assert not has_live_connection(db)


def test_driver_connections_are_tracked_while_open(tmp_path, monkeypatch):
    """The transactional write path and the explicit-close read path both track."""
    db = tmp_path / "state.db"
    _create_room(db)
    assert not has_live_connection(db)

    seen: list[bool] = []
    real_connect = driver._connect

    def connect_and_observe(db_path):
        conn = real_connect(db_path)
        seen.append(has_live_connection(db))
        return conn

    monkeypatch.setattr(driver, "_connect", connect_and_observe)

    driver.acquire_lease(  # write path: _transaction -> _connect
        db,
        room_id="room-1",
        gateway_id="gateway-a",
        authority_epoch=1,
        process_generation="process-a",
        ttl_seconds=30,
        clock=lambda: 100.0,
    )
    assert seen == [True]
    assert not has_live_connection(db)

    driver.list_tasks(db, room_id="room-1")  # read path: _connect + explicit close
    assert seen == [True, True]
    assert not has_live_connection(db)
