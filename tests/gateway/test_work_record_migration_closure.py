"""Populated legacy target and orphan evidence survive the real initializer."""

import json
import sqlite3

import pytest

from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_room_work_records as work
from gateway import hosted_rooms as rooms
from tests.gateway.test_hosted_room_work_records import source, capture, HOME  # noqa: F401

TABLES = (work.SOURCE_TABLE, work.TARGET_TABLE, work.PENDING_TABLE)
INVALID = "hosted_room_work_records_invalid"


def legacy_database(source, path, damage, *, orphan=False):
    record = capture(source)
    replicas.ingest_page(path, room_id="room", room_name="Workshop",
        members=rooms.room_state(source, room_id="room")["members"],
        page=rooms.read_events(source, room_id="room"))
    # Install real cleanup triggers before reconstruction, as on an older DB.
    with rooms._transaction(path, immediate=True) as conn:
        work.initialize(conn)
    data = json.dumps(record, indent=2)
    revision, digest = record["revision"], record["digest"]
    if damage == "metadata":
        revision += 7
    if damage == "json":
        data = "{ original invalid json bytes"
    if damage == "payload":
        bad = {**record, "tasks": "invalid"}
        data = json.dumps(bad)
    with sqlite3.connect(path) as raw:
        raw.execute("PRAGMA foreign_keys=OFF")
        for (name,) in raw.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'trg_work_invalid_%'").fetchall():
            raw.execute(f'DROP TRIGGER "{name}"')
        raw.execute(f"DROP TABLE IF EXISTS {INVALID}")
        for table in TABLES:
            raw.execute(f"DROP TABLE {table}")
            extra = ",target_install_id TEXT NOT NULL,route_generation TEXT NOT NULL,status TEXT NOT NULL" if table == work.PENDING_TABLE else ""
            raw.execute(f"CREATE TABLE {table} (room_id TEXT PRIMARY KEY,revision INTEGER NOT NULL,digest TEXT NOT NULL,record_json TEXT NOT NULL{extra})")
        raw.execute(f"INSERT INTO {work.TARGET_TABLE} VALUES (?,?,?,?)", ("room", revision, digest, data))
        if orphan:
            for table in TABLES:
                values = ("orphan", revision, digest, data)
                if table == work.PENDING_TABLE:
                    values += ("original-target", "original-route", "unavailable")
                raw.execute(f"INSERT INTO {table} VALUES ({','.join('?' for _ in values)})", values)
    return record, (revision, digest, data)


@pytest.mark.parametrize("retirement_first", [False, True])
@pytest.mark.parametrize("damage", [None, "metadata", "json", "payload"])
def test_populated_target_migration_preserves_scope_and_cleanup(source, tmp_path, retirement_first, damage):
    target = tmp_path / "legacy.db"
    record, preserved = legacy_database(source, target, damage)
    with rooms._transaction(target, immediate=True) as conn:
        if retirement_first:
            retirement._initialize(conn)
        work.initialize(conn)
        retirement._initialize(conn)
        work.initialize(conn)
        row = conn.execute(f"SELECT * FROM {work.TARGET_TABLE}").fetchone()
        assert (row["revision"], row["digest"], row["record_json"]) == preserved
        summary = work.summary_locked(conn, "room")
        assert summary["source_loss_safe"] is False
        if damage:
            assert row["disposition"] == "invalid"
            assert (row["producer_gateway_id"], row["producer_epoch"]) == ("", 0)
            assert summary["availability"] == "not_retained"
            assert summary["scopes"][0]["availability"] == "invalid"
        else:
            assert summary["producer"] == record["authority"]
            assert summary["digest"] == record["digest"]
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    with sqlite3.connect(target) as raw:
        raw.execute("PRAGMA foreign_keys=OFF")
        if damage:
            with pytest.raises(sqlite3.IntegrityError):
                raw.execute(f"UPDATE {work.TARGET_TABLE} SET record_json='overwritten'")
        raw.execute("UPDATE hosted_room_replicas SET disbanded_at=123 WHERE room_id='room'")
        assert raw.execute(f"SELECT COUNT(*) FROM {work.TARGET_TABLE}").fetchone()[0] == 0


@pytest.mark.parametrize("retirement_first", [False, True])
@pytest.mark.parametrize("bound", ["rows", "bytes"])
def test_orphan_legacy_evidence_is_retained_and_charged_without_fabricated_parents(source, tmp_path, monkeypatch, retirement_first, bound):
    target = tmp_path / "orphan.db"
    _, preserved = legacy_database(source, target, "metadata", orphan=True)
    monkeypatch.setattr(work, "MAX_STORE_ROWS" if bound == "rows" else "MAX_STORE_BYTES", 3 if bound == "rows" else 1)
    with rooms._transaction(target, immediate=True) as conn:
        if retirement_first:
            retirement._initialize(conn)
        work.initialize(conn)
        retirement._initialize(conn)
        work.initialize(conn)
        invalid = [dict(r) for r in conn.execute(f"SELECT * FROM {INVALID} ORDER BY source_table")]
        assert len(invalid) == 3
        assert {r["source_table"] for r in invalid} == set(TABLES)
        for row in invalid:
            assert row["room_id"] == "orphan" and row["disposition"] == "invalid"
            assert (row["revision"], row["digest"], row["record_json"]) == preserved
            assert "producer_gateway_id" not in row
        pending = next(r for r in invalid if r["source_table"] == work.PENDING_TABLE)
        assert (pending["target_install_id"], pending["route_generation"], pending["status"]) == (
            "original-target", "original-route", "unavailable")
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
        assert not conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id='orphan'").fetchone()
        assert not conn.execute("SELECT 1 FROM hosted_room_replicas WHERE room_id='orphan'").fetchone()
        assert work.summary_locked(conn, "orphan")["scopes"][0]["availability"] == "invalid"
    rooms.create_room(target, room_id="new", name="New", members=rooms.room_state(source, room_id="room")["members"], authority_gateway_id=HOME)
    with pytest.raises(work.WorkRecordCapacityError):
        work.capture(target, room_id="new", local_gateway_id=HOME)
    with sqlite3.connect(target) as raw:
        for sql in (f"UPDATE {INVALID} SET record_json='lost'", f"DELETE FROM {INVALID}"):
            with pytest.raises(sqlite3.IntegrityError):
                raw.execute(sql)
        assert raw.execute(f"SELECT COUNT(*) FROM {INVALID}").fetchone()[0] == 3


@pytest.mark.parametrize("cleanup", ["source_disband", "target_retirement"])
def test_orphan_archive_reclamation_is_scoped_to_explicit_owner_cleanup(source, tmp_path, cleanup):
    from gateway import hosted_room_link_records as links
    from tests.gateway.test_hosted_room_replica_retirement import MEMBERS, TARGET, SECRET
    target = tmp_path / "cleanup.db"
    legacy_database(source, target, "json", orphan=True)
    with rooms._transaction(target, immediate=True) as conn:
        work.initialize(conn)
    home = target if cleanup == "source_disband" else tmp_path / "owner.db"
    # Explicit test-owner setup, never performed by migration or inspection.
    rooms.create_room(home, room_id="orphan", name="Owner", members=MEMBERS, authority_gateway_id=HOME)
    if cleanup == "target_retirement":
        entry = retirement.prepare_home_enrollment(home, room_id="orphan", target_install_id=TARGET,
            endpoint="https://participant.example", local_gateway_id=HOME, secret=SECRET)
        retirement.enroll_target(target, enrollment=entry, target_install_id=TARGET)
    scope = dict(room_id="orphan", authority_gateway_id=HOME, authority_epoch=1)
    links.begin_room_link_retirement(home, **scope)
    links.complete_room_link_retirement(home, **scope)
    rooms.disband_room(home, room_id="orphan", expected_gateway_id=HOME, expected_epoch=1)
    if cleanup == "target_retirement":
        notice = retirement.materialize_notice(home, enrollment_id=entry["enrollment_id"],
            local_gateway_id=HOME, secret_loader=lambda: SECRET)
        assert retirement.retire_copy(target, payload=notice.payload(), value=notice.value, local_gateway_id=TARGET)["retired"]
    with rooms._transaction(target) as conn:
        kinds = {r[0] for r in conn.execute(f"SELECT source_table FROM {INVALID} WHERE room_id='orphan'")}
        assert kinds == ({work.TARGET_TABLE} if cleanup == "source_disband" else {work.SOURCE_TABLE, work.PENDING_TABLE})
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"DELETE FROM {INVALID} WHERE room_id='orphan'")


def test_delivery_summary_cannot_label_metadata_mismatch_current_evidence(source):
    with rooms._transaction(source, immediate=True) as conn:
        work.prepare_delivery_locked(conn, room_id="room", target_install_id="target",
            route_generation="generation", local_gateway_id=HOME, through_seq=1)
        conn.execute(f"UPDATE {work.PENDING_TABLE} SET digest='wrong'")
        summary = work.delivery_summaries_locked(conn)[0]
        assert summary["disposition"] == "invalid"
        assert summary["incompleteness"] == ["invalid_work_evidence"]
        assert summary["source_loss_safe"] is False


def test_summary_revalidates_stored_metadata_not_only_record_json(source, tmp_path):
    target = tmp_path / "metadata.db"
    record, _ = legacy_database(source, target, None)
    with rooms._transaction(target, immediate=True) as conn:
        work.initialize(conn)
        conn.execute(f"UPDATE {work.TARGET_TABLE} SET revision=revision+1")
        summary = work.summary_locked(conn, "room")
        assert summary["availability"] == "invalid"
        assert summary["source_loss_safe"] is False
        assert summary["incompleteness"] == ["invalid_work_evidence"]
        assert "tasks" not in summary
        assert record["availability"] == "available"
