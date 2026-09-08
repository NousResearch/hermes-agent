"""Atomic saved-room conversion stays quarantined until reconciliation."""

from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3

import pytest

from gateway import hosted_rooms as rooms, hosted_room_driver as driver, hosted_room_replicas as replicas
from gateway import hosted_room_manual_promotion as promotion, hosted_room_work_records as records
from gateway.hosted_room_manual_recovery import prepare_recovery
from gateway.hosted_room_manual_promotion_schema import TABLE
from tests.gateway.test_hosted_room_replica_ingress import HOME, TARGET, SECRET, MEMBERS, grant, ingest, pair


@pytest.fixture
def saved(pair, monkeypatch):
    return save_pair(pair, monkeypatch)


def save_pair(pair, monkeypatch):
    source, target = pair
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: TARGET)
    driver.list_tasks(source, room_id="room")
    token, _ = grant(target, permissions=("replicate", "work_records"))
    ingest(pair, token)
    record = records.capture(source, room_id="room", local_gateway_id=HOME)
    records.ingest(target, record=record, token=token, secret=SECRET, target_install_id=TARGET, target_profile="reviewer")
    preview = prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
    assert preview["blockers"] == []
    return source, target, token, preview


@pytest.fixture
def saved_v2(saved):
    from gateway import hosted_room_replica_retirement as retirement
    from tests.gateway.test_hosted_room_replica_lineage import SUCCESSOR, transfer
    source, target, _old, _preview = saved
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    enrollment = retirement.prepare_home_enrollment(source, room_id="room", target_install_id=TARGET,
        endpoint="https://participant.example", local_gateway_id=SUCCESSOR, secret=SECRET, enrollment_id="successor")
    retirement.enroll_target(target, enrollment=enrollment, target_install_id=TARGET, authority_history=spans)
    token, _ = grant(target, permissions=("replicate", "work_records"), grant_id="successor",
        home_install_id=SUCCESSOR, authority_gateway_id=SUCCESSOR, authority_epoch=2)
    ingest((source, target), token, page=rooms.read_events(source, room_id="room", replica_version=2))
    current = records.capture(source, room_id="room", local_gateway_id=SUCCESSOR)
    records.ingest(target, record=current, token=token, secret=SECRET, target_install_id=TARGET, target_profile="reviewer")
    fresh = prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
    assert fresh["blockers"] == []
    return source, target, token, fresh


def test_successor_stages_all_producers_without_relabeling_origins(saved_v2):
    target = saved_v2[1]
    before = rows(target, records.TARGET_TABLE)
    members_json = rows(target, "hosted_room_replicas")[0][2]
    assert len(before) == 2
    view = saved_v2[3]
    assert view["work_records"]["tasks"] == []
    assert view["reconciliation_required"] is True
    assert view["accepted_tail"] == "unverified"
    assert view["member_origins"] == [
        {"member_id": "writer", "installation_id": HOME, "profile": "default", "resolved": True},
        {"member_id": "reviewer", "installation_id": TARGET, "profile": "reviewer", "resolved": True}]
    stage(saved_v2)
    with sqlite3.connect(target) as conn:
        data = json.loads(conn.execute(f"SELECT work_record_json FROM {TABLE}").fetchone()[0])
        assert sorted(r["record_json"] for r in data["rows"][records.TARGET_TABLE]) == sorted(r[3] for r in before)
        assert conn.execute("SELECT members_json FROM hosted_rooms").fetchone()[0] == members_json
    assert driver.list_tasks(target, room_id="room") == []


@pytest.mark.parametrize("field,value", [("enrollment_id", "replaced"), ("state", "closed")])
def test_enrollment_change_invalidates_snapshot(saved_v2, field, value):
    from gateway.hosted_room_passive_lineage import ENROLLMENTS
    with sqlite3.connect(saved_v2[1]) as conn:
        conn.execute(f"UPDATE {ENROLLMENTS} SET {field}=? WHERE is_current=1", (value,))
    fresh = prepare_recovery(saved_v2[1], room_id="room", target_gateway_id=TARGET)
    assert fresh["snapshot_id"] != saved_v2[3]["snapshot_id"]
    with pytest.raises(promotion.ManualRecoveryError):
        stage(saved_v2)
    assert rows(saved_v2[1], "hosted_rooms") == []


@pytest.mark.parametrize("seam", ["archive", "target_delete", "second_target_delete", "invalid_delete", "namespace"])
def test_each_transfer_seam_rolls_back_all_evidence(saved_v2, seam):
    from gateway.hosted_room_manual_promotion_schema import initialize
    from gateway.hosted_room_work_storage import INVALID_TABLE
    target = saved_v2[1]
    with rooms._transaction(target, immediate=True) as conn:
        initialize(conn)
        conn.execute("DROP TRIGGER trg_work_invalid_insert")
        conn.execute(f"INSERT INTO {INVALID_TABLE}(source_table,room_id,revision,digest,record_json,disposition) VALUES(?, 'room',3,'opaque','PRIVATE_ARCHIVE','invalid')", (records.TARGET_TABLE,))
    tables = (records.TARGET_TABLE, INVALID_TABLE, "hosted_room_replica_events", "hosted_room_replicas", "hosted_room_id_reservations")
    before = {t: rows(target, t) for t in tables}
    when, table = {"archive": ("INSERT", TABLE), "target_delete": ("DELETE", records.TARGET_TABLE),
                   "second_target_delete": ("DELETE", records.TARGET_TABLE),
                   "invalid_delete": ("DELETE", INVALID_TABLE), "namespace": ("INSERT", "hosted_rooms")}[seam]
    with sqlite3.connect(target) as conn:
        condition = " WHEN OLD.producer_epoch=" + ("1" if seam == "target_delete" else "2") if "target_delete" in seam else ""
        conn.execute(f"CREATE TRIGGER interrupt_transfer AFTER {when} ON {table}{condition} BEGIN SELECT RAISE(ABORT, 'seam interruption'); END")
    fresh = prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
    with pytest.raises(sqlite3.IntegrityError, match="seam interruption"):
        stage(saved_v2, snapshot_id=fresh["snapshot_id"])
    assert {t: rows(target, t) for t in tables} == before
    assert rows(target, "hosted_rooms") == []
    assert rows(target, TABLE) == []


def test_enrollment_replaced_while_staging_waits_is_rechecked(saved_v2, monkeypatch):
    from gateway.hosted_room_passive_lineage import ENROLLMENTS
    from contextlib import contextmanager
    import threading
    entered = threading.Event()
    original = replicas._replica_transaction
    @contextmanager
    def waiting(*args, **kwargs):
        entered.set()
        with original(*args, **kwargs) as conn:
            yield conn
    monkeypatch.setattr(replicas, "_replica_transaction", waiting)
    target = saved_v2[1]
    with ThreadPoolExecutor(max_workers=1) as pool:
        with sqlite3.connect(target) as holder:
            holder.execute("BEGIN IMMEDIATE")
            future = pool.submit(stage, saved_v2)
            assert entered.wait(5)
            holder.execute(f"UPDATE {ENROLLMENTS} SET enrollment_id='new-owner-selection' WHERE is_current=1")
        with pytest.raises(promotion.ManualRecoveryError):
            future.result(8)
    assert rows(target, "hosted_rooms") == []
    assert len(rows(target, records.TARGET_TABLE)) == 2


def test_pending_receipt_rejects_changed_evidence_and_sql_replacement(saved_v2):
    target = saved_v2[1]
    stage(saved_v2)
    with sqlite3.connect(target) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(f"UPDATE {TABLE} SET work_record_json='{{}}'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(f"INSERT OR REPLACE INTO {TABLE} SELECT * FROM {TABLE}")
        conn.execute("DROP TRIGGER trg_recovery_evidence_immutable")
        conn.execute(f"UPDATE {TABLE} SET work_record_json='{{}}'")
    with pytest.raises(RuntimeError, match="recovery record"):
        stage(saved_v2)


def test_staging_leaves_other_room_and_non_target_archive_untouched(saved_v2):
    from gateway.hosted_room_work_storage import INVALID_TABLE
    target = saved_v2[1]
    with sqlite3.connect(target) as conn:
        conn.execute("DROP TRIGGER trg_work_invalid_insert")
        for room_id, source_table in (("other-room", records.TARGET_TABLE), ("room", records.SOURCE_TABLE)):
            conn.execute(f"INSERT INTO {INVALID_TABLE}(source_table,room_id,revision,digest,record_json,disposition) VALUES(?,?,7,'opaque','PRIVATE_OTHER','invalid')", (source_table, room_id))
        before = conn.execute(f"SELECT * FROM {INVALID_TABLE} ORDER BY evidence_id").fetchall()
    stage(saved_v2)
    with sqlite3.connect(target) as conn:
        assert conn.execute(f"SELECT * FROM {INVALID_TABLE} ORDER BY evidence_id").fetchall() == before


def test_historical_closing_fact_blocks_empty_successor(saved_v2):
    target = saved_v2[1]
    with sqlite3.connect(target) as conn:
        row = conn.execute(f"SELECT record_json FROM {records.TARGET_TABLE} WHERE producer_epoch=1").fetchone()
        data = json.loads(row[0])
        data["stop"]["closing"] = True
        data["digest"] = records.digest({k: v for k, v in data.items() if k not in {"revision", "digest"}})
        # Simulated retained historical input, never a production rewrite lease.
        conn.execute(f"DROP TRIGGER trg_{records.TARGET_TABLE}_immutable_v2")
        conn.execute(f"UPDATE {records.TARGET_TABLE} SET record_json=?,digest=? WHERE producer_epoch=1", (records.encode(data), data["digest"]))
    view = prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
    assert view["work_records"]["stop"]["closing"] is False
    assert "group_closing" in view["blockers"]
    with pytest.raises(promotion.ManualRecoveryError):
        stage(saved_v2, snapshot_id=view["snapshot_id"])
    assert rows(target, "hosted_rooms") == []


def test_current_sql_record_scope_mismatch_cannot_stage(saved_v2):
    target = saved_v2[1]
    with sqlite3.connect(target) as conn:
        data = json.loads(conn.execute(f"SELECT record_json FROM {records.TARGET_TABLE} WHERE producer_epoch=2").fetchone()[0])
        data["room_id"] = "other-room"
        data["digest"] = records.digest({k: v for k, v in data.items() if k not in {"revision", "digest"}})
        conn.execute(f"UPDATE {records.TARGET_TABLE} SET record_json=?,digest=? WHERE producer_epoch=2", (records.encode(data), data["digest"]))
    view = prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
    assert "work_records_unavailable" in view["blockers"]
    with pytest.raises(promotion.ManualRecoveryError):
        stage(saved_v2, snapshot_id=view["snapshot_id"])
    assert rows(target, "hosted_rooms") == []


def stage(saved, **overrides):
    return promotion.stage_manual_recovery(saved[1], **{
        "room_id": "room", "recovery_id": "owner-decision", "snapshot_id": saved[3]["snapshot_id"],
        "confirm_previous_host_fenced": True, "confirm_saved_point": True, **overrides,
    })


def rows(path, table):
    with sqlite3.connect(path) as conn:
        return conn.execute(f"SELECT * FROM {table} WHERE room_id='room' ORDER BY rowid").fetchall()


@pytest.mark.parametrize("opaque", ["{not JSON: PRIVATE_ARCHIVE", "null"])
def test_all_original_target_scopes_and_archive_survive_staging(saved_v2, opaque):
    from gateway.hosted_room_work_storage import INVALID_TABLE
    saved = saved_v2
    target = saved[1]
    with sqlite3.connect(target) as conn:
        conn.execute(f"INSERT INTO {records.TARGET_TABLE} (room_id,revision,digest,record_json,producer_gateway_id,producer_epoch,disposition) VALUES('room',2,'opaque',?,'',0,'invalid')", (opaque,))
        conn.execute("DROP TRIGGER trg_work_invalid_insert")
        conn.execute(f"INSERT INTO {INVALID_TABLE}(source_table,room_id,revision,digest,record_json,disposition) VALUES(?, 'room',3,'opaque',?,'invalid')", (records.TARGET_TABLE, opaque))
    before = {}
    with sqlite3.connect(target) as conn:
        conn.row_factory = sqlite3.Row
        for table in (records.TARGET_TABLE, INVALID_TABLE):
            before[table] = [dict(r) for r in conn.execute(f"SELECT * FROM {table} WHERE room_id='room'")]
    fresh = prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
    assert "PRIVATE_ARCHIVE" not in json.dumps(fresh)
    assert "record_json" not in json.dumps(fresh)
    assert fresh["reconciliation_required"] is True
    stage(saved, snapshot_id=fresh["snapshot_id"])
    with sqlite3.connect(target) as conn:
        envelope = json.loads(conn.execute(f"SELECT work_record_json FROM {TABLE}").fetchone()[0])
        assert envelope["object"] == "hermes.group_recovery.evidence"
        assert envelope["version"] == 2
        for table, originals in before.items():
            assert sorted(envelope["rows"][table], key=lambda r: json.dumps(r, sort_keys=True)) == sorted(originals, key=lambda r: json.dumps(r, sort_keys=True))
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE room_id='room'").fetchone()[0] == 0
    assert stage(saved, snapshot_id=fresh["snapshot_id"])["idempotent"] is True
    assert driver.list_tasks(target, room_id="room") == []


@pytest.mark.parametrize("damage", ["column", "blob"])
def test_unsupported_storage_refuses_before_namespace_mutation(saved, damage):
    target = saved[1]
    with sqlite3.connect(target) as conn:
        if damage == "column":
            conn.execute(f"ALTER TABLE {records.TARGET_TABLE} ADD COLUMN future_evidence TEXT")
        else:
            conn.execute(f"UPDATE {records.TARGET_TABLE} SET record_json=CAST(record_json AS BLOB)")
    before = rows(target, records.TARGET_TABLE)
    with pytest.raises((promotion.ManualRecoveryError, records.WorkRecordError, replicas.ReplicaError)):
        fresh = prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
        stage(saved, snapshot_id=fresh["snapshot_id"])
    assert rows(target, records.TARGET_TABLE) == before
    assert rows(target, "hosted_rooms") == []


def test_saved_history_moves_once_without_executing_or_discarding_work_records(saved):
    source, target, _token, preview = saved
    source_history = rows(source, "hosted_room_events")
    copied_history = rows(target, "hosted_room_replica_events")
    copied_work = rows(target, records.TARGET_TABLE)
    result = stage(saved)
    assert result["room_id"] == preview["room_id"]
    assert result["authority_gateway_id"] == TARGET
    assert result["authority_epoch"] == preview["source_authority"]["epoch"] + 1
    assert result["execution_authorized"] is False
    assert result["accepted_tail"] == "unverified"
    assert result["status"] == "pending_reconciliation"
    assert stage(saved) == {**result, "idempotent": True}
    assert rows(source, "hosted_room_events") == source_history
    moved = rows(target, "hosted_room_events")
    assert moved[:-1] == copied_history
    assert moved[-1][3] == "authority.claimed"
    with rooms._transaction(target) as conn:
        record = conn.execute(f"SELECT * FROM {TABLE} WHERE room_id='room'").fetchone()
        assert json.loads(record["work_record_json"])["rows"][records.TARGET_TABLE][0]["record_json"] == copied_work[0][3]
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_replicas WHERE room_id='room'").fetchone()[0] == 0
        assert conn.execute("SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id='room'").fetchone()[0] == "authority"
        expected_bytes = sum(len(str(value).encode("utf-8")) for event in moved for value in (event[2], event[3], event[4], event[6]))
        assert conn.execute("SELECT event_bytes FROM hosted_room_event_budget").fetchone()[0] == expected_bytes
    with pytest.raises(rooms.RoomQuarantinedError):
        rooms.room_state(target, room_id="room")
    with pytest.raises(driver.RoomUnavailableError):
        driver.acquire_lease(target, room_id="room", gateway_id=TARGET, authority_epoch=result["authority_epoch"],
            process_generation="must-not-start", ttl_seconds=30, clock=lambda: 100)
    assert driver.list_tasks(target, room_id="room") == []


@pytest.mark.parametrize("overrides", [
    {"confirm_previous_host_fenced": False}, {"confirm_saved_point": False},
    {"confirm_previous_host_fenced": "true"}, {"snapshot_id": "a" * 64},
])
def test_no_implicit_or_stale_confirmation_can_move_a_copy(saved, overrides):
    target = saved[1]
    before = replicas.replica_state(target, room_id="room")
    with pytest.raises(promotion.ManualRecoveryError):
        stage(saved, **overrides)
    assert replicas.replica_state(target, room_id="room") == before
    assert rows(target, "hosted_rooms") == []


def test_new_copied_task_invalidates_the_selected_snapshot(saved):
    source, target, token, _preview = saved
    task = driver.TaskIdentity("room", "new-task", "thread", "turn")
    driver.admit_task(source, task, payload={"target_profile": "default", "target_member_id": "writer",
        "source_event_seq": rooms.read_events(source, room_id="room")["latest_seq"], "prompt": "PRIVATE_TASK"}, clock=lambda: 100)
    record = records.capture(source, room_id="room", local_gateway_id=HOME)
    records.ingest(target, record=record, token=token, secret=SECRET, target_install_id=TARGET, target_profile="reviewer")
    with pytest.raises(promotion.ManualRecoveryError):
        stage(saved)
    assert rows(target, "hosted_rooms") == []
    refreshed = prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
    assert stage(saved, snapshot_id=refreshed["snapshot_id"])["execution_authorized"] is False
    assert driver.list_tasks(target, room_id="room") == []


def test_failed_transfer_restores_original_copy_and_schema(saved, monkeypatch):
    target = saved[1]
    before = replicas.replica_state(target, room_id="room")
    history, work = rows(target, "hosted_room_replica_events"), rows(target, records.TARGET_TABLE)
    original = promotion._move_events

    def fail_after_move(conn, room_id):
        original(conn, room_id)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(promotion, "_move_events", fail_after_move)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        stage(saved)
    assert replicas.replica_state(target, room_id="room") == before
    assert rows(target, "hosted_room_replica_events") == history
    assert rows(target, records.TARGET_TABLE) == work
    assert rows(target, "hosted_rooms") == []


def test_simultaneous_deliveries_create_one_pending_decision(saved):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: stage(saved), range(2)))
    assert sorted(result["idempotent"] for result in results) == [False, True]
    assert len(rows(saved[1], TABLE)) == 1


def test_completed_transition_cannot_be_reused_by_ordinary_writers(saved):
    target = saved[1]
    stage(saved)
    before = rows(target, "hosted_rooms")
    with rooms._transaction(target, immediate=True) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="reserved"):
            conn.execute("INSERT OR REPLACE INTO hosted_rooms SELECT * FROM hosted_rooms WHERE room_id='room'")
    assert rows(target, "hosted_rooms") == before
    rooms.create_room(target, room_id="unrelated", name="Another group", authority_gateway_id=TARGET,
                      members=[{"profile": "default", "handle": "writer"}])
    assert rooms.room_state(target, room_id="unrelated")["authority_gateway_id"] == TARGET


@pytest.mark.parametrize("state", ["partial", "quarantined", "disbanded", "closing"])
def test_unavailable_or_closing_copy_cannot_be_transferred(saved, state):
    source, target, token, _preview = saved
    if state == "partial":
        with sqlite3.connect(target) as conn:
            conn.execute("UPDATE hosted_room_replicas SET latest_seq=last_seq+1 WHERE room_id='room'")
    elif state == "quarantined":
        with sqlite3.connect(target) as conn:
            conn.execute("UPDATE hosted_room_replicas SET quarantine_reason='unverified' WHERE room_id='room'")
    elif state == "disbanded":
        rooms.disband_room(source, room_id="room", expected_gateway_id=HOME, expected_epoch=1)
        ingest((source, target), token, page=rooms.read_events(source, room_id="room", include_disbanded=True))
    else:
        with rooms._transaction(source, immediate=True) as conn:
            conn.execute("INSERT INTO hosted_room_disband_fences(room_id,authority_gateway_id,authority_epoch,started_at) VALUES(?,?,1,100)",
                         ("room", HOME))
        record = records.capture(source, room_id="room", local_gateway_id=HOME)
        records.ingest(target, record=record, token=token, secret=SECRET, target_install_id=TARGET, target_profile="reviewer")
    current = prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
    assert current["blockers"]
    with pytest.raises(promotion.ManualRecoveryError):
        stage(saved, snapshot_id=current["snapshot_id"])
    assert rows(target, "hosted_rooms") == []


def test_transfer_does_not_temporarily_double_count_the_shared_byte_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(rooms, "MAX_GATEWAY_EVENT_BYTES", 2048)
    source, target = tmp_path / "home.db", tmp_path / "peer.db"
    rooms.create_room(source, room_id="room", name="Workshop", members=MEMBERS, authority_gateway_id=HOME)
    rooms.append_event(source, room_id="room", event_id="large", kind="message.user", actor={"kind": "user", "id": "owner"},
        payload={"text": "x" * 1200}, authority_gateway_id=HOME, authority_epoch=1)
    saved = save_pair((source, target), monkeypatch)
    before = replicas.replica_state(target, room_id="room")["event_bytes"]
    assert rooms.MAX_GATEWAY_EVENT_BYTES / 2 < before < rooms.MAX_GATEWAY_EVENT_BYTES
    stage(saved)
    with sqlite3.connect(target) as conn:
        assert before < conn.execute("SELECT event_bytes FROM hosted_room_event_budget").fetchone()[0] < rooms.MAX_GATEWAY_EVENT_BYTES
