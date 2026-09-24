"""Cross-store history-source identities; run reservation cases in the declared Runtime composition."""

from contextlib import closing
import sqlite3

import pytest

from gateway import hosted_rooms as rooms
from gateway import hosted_rooms_legacy_import as legacy

TABLE = "hosted_room_history_source_reservations"
ROW = ("released-source", "plugin-release", "a" * 64, "pruned-room")


def _reserve(path, row=ROW):
    with rooms._transaction(path, immediate=True) as conn:
        conn.execute(f"INSERT INTO {TABLE} (source_id, source_kind, content_sha256, room_id) VALUES (?, ?, ?, ?)", row)


def _rows(path):
    with sqlite3.connect(path) as conn:
        return conn.execute(f"SELECT source_id, source_kind, content_sha256, room_id FROM {TABLE} ORDER BY source_id").fetchall()


def _marker(path):
    with sqlite3.connect(path) as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_legacy_imports'").fetchone() is None:
            return None
        return conn.execute("SELECT rooms FROM hosted_room_legacy_imports WHERE source='state.db'").fetchone()


def _unsettle(path):
    # Target pre-exists before the legacy file is introduced in this fixture.
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM hosted_room_legacy_imports")


def test_pruned_reservation_only_import_is_durable_and_one_shot(tmp_path):
    source, target = tmp_path / "state.db", tmp_path / "shared-state.db"
    _reserve(source)
    before = _rows(source)
    assert rooms.list_rooms(target) == []
    assert _rows(target) == before
    assert _marker(target) == (0,)
    assert rooms.list_rooms(target) == []
    assert _rows(target) == before == _rows(source)


@pytest.mark.parametrize("target_row", [
    (ROW[0], ROW[1], "b" * 64, ROW[3]),
    (ROW[0], ROW[1], ROW[2], "different-room"),
    ("different-source", ROW[1], ROW[2], ROW[3]),
])
def test_conflicting_target_source_identity_rolls_back(tmp_path, target_row):
    source, target = tmp_path / "state.db", tmp_path / "shared-state.db"
    _reserve(target, target_row)
    _unsettle(target)
    _reserve(source)
    before = _rows(source)
    assert rooms.list_rooms(target) == []
    assert _rows(target) == [target_row]
    assert _rows(source) == before
    assert _marker(target) is None
    assert source in legacy._failed_sources


def test_identical_target_reservation_is_idempotent(tmp_path):
    source, target = tmp_path / "state.db", tmp_path / "shared-state.db"
    _reserve(target)
    _unsettle(target)
    _reserve(source)
    assert rooms.list_rooms(target) == []
    assert _rows(target) == [ROW]
    assert _marker(target) == (0,)


def test_identical_claim_with_already_owned_target_room_is_idempotent(tmp_path):
    source, target = tmp_path / "state.db", tmp_path / "shared-state.db"
    _reserve(target)
    with rooms._transaction(target, immediate=True) as conn:
        conn.execute("INSERT INTO hosted_rooms (room_id, name, members_json, authority_gateway_id, authority_epoch, created_at, updated_at) VALUES (?, 'Existing', '[]', 'owner', 1, 1, 1)", (ROW[3],))
    _unsettle(target)
    _reserve(source)
    assert [r["room_id"] for r in rooms.list_rooms(target)] == [ROW[3]]
    assert _rows(target) == _rows(source) == [ROW]
    assert _marker(target) == (0,)


@pytest.mark.parametrize("namespace", ["authority", "replica", "retired"])
def test_pruned_reservation_refuses_target_room_namespace(tmp_path, namespace):
    source, target = tmp_path / "state.db", tmp_path / "shared-state.db"
    with rooms._transaction(target, immediate=True) as conn:
        if namespace == "retired":
            conn.execute("INSERT INTO hosted_room_retired_ids (room_id, retired_at) VALUES (?, 1)", (ROW[3],))
        else:
            table = "hosted_rooms" if namespace == "authority" else "hosted_room_replicas"
            conn.execute(f"INSERT INTO {table} (room_id, name, members_json, authority_gateway_id, authority_epoch, created_at, updated_at) VALUES (?, 'Existing', '[]', 'owner', 1, 1, 1)", (ROW[3],))
    _unsettle(target)
    _reserve(source)
    assert [r["room_id"] for r in rooms.list_rooms(target)] == ([ROW[3]] if namespace == "authority" else [])
    assert _rows(target) == []
    assert _rows(source) == [ROW]
    assert _marker(target) is None


def test_missing_target_reservation_schema_refuses_settlement(tmp_path):
    source, target = tmp_path / "state.db", tmp_path / "shared-state.db"
    _reserve(source)
    # Lower owner must not create Runtime's target DDL or mark a skipped source settled.
    with closing(sqlite3.connect(target)) as conn:
        conn.execute("CREATE TABLE hosted_room_id_reservations (room_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE hosted_room_retired_ids (room_id TEXT PRIMARY KEY)")
        legacy.import_legacy_rooms(conn, target)
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (TABLE,)).fetchone() is None
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='hosted_room_legacy_imports'").fetchone() is None
    assert _rows(source) == [ROW]


def test_later_copy_failure_rolls_back_reservation_and_parent(tmp_path):
    source, target = tmp_path / "state.db", tmp_path / "shared-state.db"
    with rooms._transaction(source, immediate=True) as conn:
        conn.execute("INSERT INTO hosted_rooms (room_id, name, members_json, authority_gateway_id, authority_epoch, created_at, updated_at) VALUES ('ordinary', 'Ordinary', '[]', 'owner', 1, 1, 1)")
        conn.execute(f"INSERT INTO {TABLE} VALUES (?, ?, ?, ?)", ROW)
        conn.execute("""CREATE TABLE hosted_room_driver_tasks (
            room_id TEXT, task_id TEXT, thread_id TEXT, turn_id TEXT,
            source_event_seq INTEGER, payload_json TEXT, payload_digest TEXT,
            status TEXT, created_at REAL, updated_at REAL)""")
        conn.execute("""INSERT INTO hosted_room_driver_tasks VALUES
            ('ordinary', 'task', 'thread', 'turn', 1, '{}', 'digest', 'alien', 1, 1)""")
    assert rooms.list_rooms(target) == []
    assert _rows(target) == []
    assert _marker(target) is None
    assert _rows(source) == [ROW]
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_rooms").fetchone() == (0,)


def test_old_marker_without_reservation_refuses_target_source_collision(tmp_path):
    source, target = tmp_path / "state.db", tmp_path / "shared-state.db"
    _reserve(target, (ROW[0], ROW[1], "b" * 64, "other-room"))
    _unsettle(target)
    with rooms._transaction(source, immediate=True) as conn:
        conn.execute("INSERT INTO hosted_rooms (room_id, name, members_json, authority_gateway_id, authority_epoch, created_at, updated_at) VALUES (?, 'Imported', '[]', 'owner', 1, 1, 1)", (ROW[3],))
        conn.execute("""INSERT INTO hosted_room_history_imports
            (room_id, source_kind, source_id, content_sha256, history_count,
             held_work_count, held_member_count, imported_at)
            VALUES (?, ?, ?, ?, 0, 0, 0, 1)""", (ROW[3], ROW[1], ROW[0], ROW[2]))
        conn.execute(f"DROP TABLE {TABLE}")
    assert rooms.list_rooms(target) == []
    assert _rows(target) == [(ROW[0], ROW[1], "b" * 64, "other-room")]
    assert _marker(target) is None
    with sqlite3.connect(source) as conn:
        assert conn.execute("SELECT source_id FROM hosted_room_history_imports").fetchone() == (ROW[0],)


def test_old_marker_without_reservation_backfills_in_runtime_after_copy(tmp_path):
    source, target = tmp_path / "state.db", tmp_path / "shared-state.db"
    with rooms._transaction(source, immediate=True) as conn:
        conn.execute("INSERT INTO hosted_rooms (room_id, name, members_json, authority_gateway_id, authority_epoch, created_at, updated_at) VALUES (?, 'Imported', '[]', 'owner', 1, 1, 1)", (ROW[3],))
        conn.execute("""INSERT INTO hosted_room_history_imports
            (room_id, source_kind, source_id, content_sha256, history_count,
             held_work_count, held_member_count, imported_at)
            VALUES (?, ?, ?, ?, 0, 0, 0, 1)""", (ROW[3], ROW[1], ROW[0], ROW[2]))
        conn.execute(f"DROP TABLE {TABLE}")
    assert [r["room_id"] for r in rooms.list_rooms(target)] == [ROW[3]]
    assert _rows(target) == [ROW]
    assert _marker(target) == (1,)


def test_old_provider_without_reservation_table_imports_ordinary_room(tmp_path):
    source, target = tmp_path / "state.db", tmp_path / "shared-state.db"
    with rooms._transaction(source, immediate=True) as conn:
        conn.execute("INSERT INTO hosted_rooms (room_id, name, members_json, authority_gateway_id, authority_epoch, created_at, updated_at) VALUES ('ordinary', 'Ordinary', '[]', 'owner', 1, 1, 1)")
        conn.execute(f"DROP TABLE {TABLE}")
    assert [r["room_id"] for r in rooms.list_rooms(target)] == ["ordinary"]
    assert _marker(target) == (1,)
    assert _rows(target) == []
