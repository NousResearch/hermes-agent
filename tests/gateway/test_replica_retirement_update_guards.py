"""Raw SQLite UPDATE boundaries around real enrolled and retired copies."""

import sqlite3
from contextlib import closing

import pytest

from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms as rooms
from gateway.hosted_room_safety import _prune_disbanded_replicas_locked
from tests.gateway.test_hosted_room_replica_retirement import (
    HOME, MEMBERS, SECRET, TARGET, close, copied_prefix, enroll, notice, pair,
)


def _install_legacy_update_guards(db):
    # Historical trigger DDL only; the database and retirements use repository APIs.
    with closing(sqlite3.connect(db)) as conn, conn:
        for suffix in ("event", "room"):
            for version in ("", "_v2"):
                conn.execute(f"DROP TRIGGER IF EXISTS trg_replica_retired_{suffix}_update{version}")
        conn.execute(f"""CREATE TRIGGER trg_replica_retired_event_update
            BEFORE UPDATE ON hosted_room_replica_events
            WHEN EXISTS (SELECT 1 FROM {retirement.RETIREMENT_TABLE} WHERE room_id=OLD.room_id)
            BEGIN SELECT RAISE(ABORT, 'replica copy is retired'); END""")
        conn.execute(f"""CREATE TRIGGER trg_replica_retired_room_update
            BEFORE UPDATE ON hosted_room_replicas
            WHEN EXISTS (SELECT 1 FROM {retirement.RETIREMENT_TABLE} WHERE room_id=OLD.room_id)
              AND (NEW.room_id IS NOT OLD.room_id OR NEW.name IS NOT OLD.name
                OR NEW.members_json IS NOT OLD.members_json
                OR NEW.authority_gateway_id IS NOT OLD.authority_gateway_id
                OR NEW.authority_epoch IS NOT OLD.authority_epoch
                OR NEW.last_seq IS NOT OLD.last_seq OR NEW.latest_seq IS NOT OLD.latest_seq
                OR NEW.disbanded_at IS NOT OLD.disbanded_at)
            BEGIN SELECT RAISE(ABORT, 'replica copy is retired'); END""")


def _retired_and_active(pair, *, state, legacy):
    home, target = pair
    entry = enroll(pair)
    if state != "before_first_page":
        copied_prefix(pair)
    close(home)
    outgoing = notice(home, entry)
    retirement.retire_copy(
        target, payload=outgoing.payload(), value=outgoing.value, local_gateway_id=TARGET,
    )
    if state == "reclaimed":
        with rooms._transaction(target, immediate=True) as conn:
            assert _prune_disbanded_replicas_locked(conn, now=None, max_replica_event_bytes=0) == 1
    rooms.create_room(home, room_id="active", name="Active", members=MEMBERS, authority_gateway_id=HOME)
    rooms.append_event(
        home, room_id="active", event_id="active-event", kind="message.user",
        actor={"kind": "user", "id": "owner"}, payload={"text": "Active work"},
        authority_gateway_id=HOME, authority_epoch=1,
    )
    replicas.ingest_page(
        target, room_id="active", room_name="Active", members=MEMBERS,
        page=rooms.read_events(home, room_id="active"),
    )
    if legacy:
        _install_legacy_update_guards(target)
    # A normal owner enrollment initializes/migrates an already-populated target DB.
    active = retirement.prepare_home_enrollment(
        home, room_id="active", target_install_id=TARGET,
        endpoint="https://participant.example", local_gateway_id=HOME, secret=SECRET,
    )
    retirement.enroll_target(target, enrollment=active, target_install_id=TARGET)
    retirement.enroll_target(target, enrollment=active, target_install_id=TARGET)
    return target


def _insert_copy(conn, table, room_id):
    row = dict(conn.execute(f"SELECT * FROM {table} WHERE room_id='active'").fetchone())
    row["room_id"] = room_id
    conn.execute(
        f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
        tuple(row.values()),
    )


@pytest.mark.parametrize("legacy", [False, True], ids=["new-schema", "installed-v1"])
@pytest.mark.parametrize("state", ["before_first_page", "reclaimed"])
@pytest.mark.parametrize("table", ["hosted_room_replica_events", "hosted_room_replicas"])
def test_updates_cannot_enter_an_empty_retired_identity(pair, table, state, legacy):
    db = _retired_and_active(pair, state=state, legacy=legacy)
    with closing(sqlite3.connect(db)) as conn:
        conn.row_factory = sqlite3.Row
        assert conn.execute(f"SELECT 1 FROM {table} WHERE room_id='room'").fetchone() is None
        assert conn.execute(
            "SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id='room'",
        ).fetchone()[0] == "replica"
        before = tuple(conn.execute(f"SELECT * FROM {table} WHERE room_id='active'").fetchone())
        with pytest.raises(sqlite3.IntegrityError, match="retired"):
            _insert_copy(conn, table, "room")
            conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="retired"):
            conn.execute(f"UPDATE {table} SET room_id='room' WHERE room_id='active'")
            conn.commit()
        assert conn.execute(f"SELECT 1 FROM {table} WHERE room_id='room'").fetchone() is None
        assert tuple(conn.execute(f"SELECT * FROM {table} WHERE room_id='active'").fetchone()) == before


@pytest.mark.parametrize("legacy", [False, True], ids=["new-schema", "installed-v1"])
def test_source_guards_preserve_bookkeeping_and_unrelated_active_writes(pair, legacy):
    db = _retired_and_active(pair, state="populated", legacy=legacy)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.row_factory = sqlite3.Row
        before = conn.execute("SELECT updated_at,event_bytes FROM hosted_room_replicas WHERE room_id='room'").fetchone()
        conn.execute("""UPDATE hosted_room_replicas
            SET updated_at=updated_at+1,event_bytes=event_bytes+1 WHERE room_id='room'""")
        after = conn.execute("SELECT updated_at,event_bytes FROM hosted_room_replicas WHERE room_id='room'").fetchone()
        assert tuple(after) == (before[0] + 1, before[1] + 1)
        conn.execute("UPDATE hosted_room_replicas SET event_bytes=? WHERE room_id='room'", (before[1],))
        conn.execute("UPDATE hosted_room_replicas SET name='Active update' WHERE room_id='active'")
        conn.execute("UPDATE hosted_room_replica_events SET created_at=created_at+1 WHERE room_id='active'")
        assert conn.execute("SELECT name FROM hosted_room_replicas WHERE room_id='active'").fetchone()[0] == "Active update"
        for table in ("hosted_room_replicas", "hosted_room_replica_events"):
            _insert_copy(conn, table, "unrelated-active")
        for sql in (
            "UPDATE hosted_room_replicas SET room_id='moved' WHERE room_id='room'",
            "UPDATE hosted_room_replica_events SET room_id='moved' WHERE room_id='room'",
            "UPDATE hosted_room_replicas SET name='Rewritten' WHERE room_id='room'",
            "UPDATE hosted_room_replica_events SET payload_json='{}' WHERE room_id='room'",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="retired"):
                conn.execute(sql)
        for table in ("hosted_room_replicas", "hosted_room_replica_events"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE room_id='room'").fetchone()[0] == 1
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE room_id='unrelated-active'").fetchone()[0] == 1
