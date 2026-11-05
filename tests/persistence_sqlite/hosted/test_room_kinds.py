"""The SQLite hosted-room schema's event-kind index: the rooms step creates it in each placement and restores a dropped one."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from tests.persistence_sqlite.hosted.conftest import NOW


KIND_INDEX = "idx_hosted_room_events_kind"


def _indexes(path) -> set[str]:
    with closing(sqlite3.connect(path)) as conn:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}


@pytest.mark.parametrize("placement", ["shared", "profile"])
def test_the_rooms_step_creates_the_event_kind_index_and_restores_a_missing_one(backend, scopes, placement):
    rooms = backend.open(scopes["a"]).hosted_rooms
    rooms.create_room(placement=placement, room_id="room", name="Room", members=[], authority_gateway_id="gw", now=NOW)
    path = rooms._files.path(placement)
    assert KIND_INDEX in _indexes(path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(f"DROP INDEX {KIND_INDEX}")

    rooms.list_rooms(placement=placement)

    assert KIND_INDEX in _indexes(path)
