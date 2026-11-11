"""The room driver's change rows on a read-only store handle (``open_persistence_reader``) of the SQLite backend.

The change watcher reads each store's live rooms through a reader. The rows need only the rooms table: a file whose
driver tables were never set up (a room created, no task admitted yet) still answers its rooms with no tasks, as a
file without the driver table does on a writable handle. The reference backend has no reader door, so these cases
live here rather than in the contract suite.
"""

from __future__ import annotations

import pytest

from persistence.gateway.hosted_rooms import RoomChangeRows
from persistence_assembly import open_persistence_reader
from tests.persistence_sqlite.hosted._task_fixtures import ROOM, make_hosted, schema, seed_room


@pytest.mark.parametrize("placement", ["profile", "shared"])
def test_a_reader_signs_the_rooms_of_a_file_whose_driver_tables_were_never_set_up(placement):
    hosted = make_hosted("profile-a")
    try:
        seed_room(hosted, placement)
        room = hosted.store.hosted_rooms.room_state(placement=placement, room_id=ROOM)
        home, path = hosted.scope.profile_dir, hosted.path(placement)
    finally:
        hosted.backend.close()
    assert schema(path, "hosted_room_driver_tasks") == []
    with open_persistence_reader(home) as reader:
        rows = reader.hosted_room_driver.room_change_rows(placement=placement)
    assert dict(rows) == {ROOM: RoomChangeRows(room["latest_seq"] + 1, room["revision"], ())}
    assert schema(path, "hosted_room_driver_tasks") == []
