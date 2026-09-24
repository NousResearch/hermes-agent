"""Known-scope permanent admission barriers in the real durable Runs store."""

import json
import sqlite3
from contextlib import closing

import pytest

from gateway.platforms import api_server_run_idempotency as storage
from gateway.platforms.api_server_run_scope import room_run_scope_key
from tests.gateway.test_group_run_scope import IDENTITY


@pytest.fixture
def store(tmp_path):
    value = storage.RunIdempotencyStore(str(tmp_path / "runs.db"))
    try:
        yield value
    finally:
        value.close()


def reserve(store, run_id="run-one", *, identity=IDENTITY, status="queued"):
    return store.reserve(room_run_scope_key(identity), "key-" + run_id, "fingerprint-" + run_id, run_id,
                         {"status": status, "output": "PRIVATE /private/credential", "token": "SECRET"},
                         owner_pid=123, owner_started=456)


def test_freeze_replays_receipts_and_commands_without_exposing_status_bodies(store):
    scope = room_run_scope_key(IDENTITY)
    reserve(store)
    assert store.path == store.path.resolve() and store.durable
    first = store.freeze_room_scope(IDENTITY, "stop-one")
    assert first["identity"] == IDENTITY and first["scope"] == scope
    assert first["runs"] == [{"run_id": "run-one", "status": "queued", "owner_pid": 123, "owner_started": 456}]
    assert first["counts"] == {"total": 1, "terminal": 0, "nonterminal": 1, "unknown": 0}
    assert not first["truncated"] and first["missing_runs"] == 0
    assert first == store.freeze_room_scope(dict(reversed(list(IDENTITY.items()))), "stop-one")
    assert reserve(store)[0] == "reused"
    assert store.reserve(scope, "key-run-one", "different", "never", {"status": "queued"})[0] == "conflict"
    assert store.lookup(scope, "key-run-one", "fingerprint-run-one")[0] == "reused"
    with pytest.raises(storage.GroupRunFrozen) as denied:
        reserve(store, "new-run")
    assert (denied.value.code, denied.value.status) == ("group_work_frozen", 409)
    assert store.is_scope_frozen(scope)
    with store.group_control_open(scope) as allowed:
        assert allowed is False
    store.update_status("run-one", {"status": "cancelled", "output": "PRIVATE"})
    done = store.room_stop_snapshot("stop-one")
    assert done["runs"] == [] and done["counts"]["terminal"] == 1
    assert done["frozen_at"] == first["frozen_at"]
    assert not any(secret in json.dumps(done) for secret in ("PRIVATE", "/private", "SECRET", "fingerprint"))
    with closing(storage.RunIdempotencyStore(str(store.path))) as restarted:
        assert restarted.room_stop_snapshot("stop-one") == done
        assert restarted.freeze_room_scope(IDENTITY, "stop-two")["frozen_at"] == first["frozen_at"]


@pytest.mark.parametrize("field", tuple(IDENTITY))
def test_every_other_participant_identity_remains_open(store, field):
    reserve(store)
    store.freeze_room_scope(IDENTITY, "stop-one")
    other = {**IDENTITY, field: 4 if field == "authority_epoch" else IDENTITY[field] + "-other"}
    assert reserve(store, "other-run", identity=other)[0] == "created"
    with store.group_control_open(room_run_scope_key(other)) as allowed:
        assert allowed is True
    with pytest.raises(storage.GroupStopCommandConflict):
        store.freeze_room_scope(other, "stop-one")


def test_unknown_scope_and_command_never_claim_empty_stop(store):
    with pytest.raises(storage.GroupStopScopeNotFound) as failure:
        store.freeze_room_scope(IDENTITY, "unknown")
    assert (failure.value.code, failure.value.status) == ("group_stop_scope_not_found", 404)
    assert not store.is_scope_frozen(room_run_scope_key(IDENTITY))
    with pytest.raises(storage.GroupStopScopeNotFound):
        store.room_stop_snapshot("unknown")
    with pytest.raises(ValueError):
        store.freeze_room_scope(room_run_scope_key(IDENTITY), "not-an-identity")
    for invalid in (True, " stop", "stop\0suffix", "x" * 129):
        with pytest.raises(ValueError):
            store.freeze_room_scope(IDENTITY, invalid)


def test_waiting_for_approval_is_known_nonterminal_work(store):
    reserve(store, status="waiting_for_approval")
    snapshot = store.freeze_room_scope(IDENTITY, "waiting")
    assert snapshot["counts"] == {"total": 1, "terminal": 0, "nonterminal": 1, "unknown": 0}
    assert snapshot["runs"][0]["status"] == "waiting_for_approval"


def test_scope_and_command_caps_do_not_evict_or_partially_freeze(store):
    store.MAX_GROUP_FREEZES, store.MAX_GROUP_STOP_COMMANDS = 1, 2
    reserve(store)
    other = {**IDENTITY, "room_id": "other"}
    reserve(store, "other-run", identity=other)
    first = store.freeze_room_scope(IDENTITY, "one")
    store.freeze_room_scope(IDENTITY, "two")
    for identity, command in ((other, "three"), (IDENTITY, "three")):
        with pytest.raises(storage.GroupStopCapacity) as error:
            store.freeze_room_scope(identity, command)
        assert error.value.status == 507
    assert store.freeze_room_scope(IDENTITY, "one") == first
    assert not store.is_scope_frozen(room_run_scope_key(other))


def test_snapshot_bounds_unknown_status_and_deleted_evidence_never_look_complete(store):
    store.GROUP_STOP_RUN_LIMIT = 2
    for i in range(4):
        reserve(store, f"run-{i}")
    with closing(sqlite3.connect(store.path)) as conn, conn:
        conn.execute("UPDATE run_idempotency SET status_json='PRIVATE malformed', owner_pid='SECRET' WHERE run_id='run-0'")
    snap = store.freeze_room_scope(IDENTITY, "bounded")
    assert len(snap["runs"]) == 2 and snap["truncated"]
    assert snap["counts"] == {"total": 4, "terminal": 0, "nonterminal": 3, "unknown": 1}
    assert snap["runs"][0] == {"run_id": "run-0", "status": "unknown", "owner_pid": 0, "owner_started": 456}
    assert not any(secret in json.dumps(snap) for secret in ("PRIVATE", "SECRET", "malformed"))
    store.update_status("run-1", {"status": "completed"})
    with closing(sqlite3.connect(store.path)) as conn, conn:
        conn.execute("DELETE FROM run_idempotency")
    missing = store.room_stop_snapshot("bounded")
    assert missing["runs"] == [] and missing["missing_runs"] == 3
    assert missing["truncated"]
    assert missing["counts"] == {"total": 4, "terminal": 1, "nonterminal": 0, "unknown": 3}
    assert store.freeze_room_scope(IDENTITY, "still-known")["missing_runs"] == 3
    with pytest.raises(storage.GroupRunFrozen):
        reserve(store)


def test_raw_legacy_identity_writes_are_fenced_but_status_and_global_pruning_work(store, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(storage.time, "time", lambda: now[0])
    scope, other = room_run_scope_key(IDENTITY), {**IDENTITY, "room_id": "other"}
    reserve(store, "active")
    reserve(store, "terminal", status="completed")
    reserve(store, "other", identity=other)
    with closing(sqlite3.connect(store.path)) as legacy:
        store.freeze_room_scope(IDENTITY, "raw")
        blocked = [
            ("UPDATE run_idempotency SET scope=? WHERE run_id='active'", (room_run_scope_key(other),)),
            ("UPDATE run_idempotency SET scope=? WHERE run_id='other'", (scope,)),
            *[(f"UPDATE run_idempotency SET {column}=? WHERE run_id='active'", (value,)) for column, value in (
                ("idempotency_key", "new"), ("fingerprint", "new"), ("run_id", "new"),
                ("owner_pid", 999), ("owner_started", 999),
            )],
        ]
        for sql, params in blocked:
            with pytest.raises(sqlite3.IntegrityError, match="frozen"):
                legacy.execute(sql, params)
            legacy.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="frozen"):
            legacy.execute("""INSERT INTO run_idempotency(scope,idempotency_key,fingerprint,run_id,status_json,created_at,updated_at)
                VALUES (?,'old-new','fp','old-new','{"status":"queued"}',100,100)""", (scope,))
        legacy.rollback()
    assert store.extend_retention(scope, "active", 101)
    now[0] += store.RETENTION_SECONDS + 1
    assert reserve(store, "other-new", identity=other)[0] == "created"
    snap = store.room_stop_snapshot("raw")
    assert [r["run_id"] for r in snap["runs"]] == ["active"]
    assert snap["counts"]["terminal"] == 1
    with closing(sqlite3.connect(store.path)) as legacy, legacy:
        legacy.execute("DELETE FROM run_idempotency")
    assert store.room_stop_snapshot("raw")["missing_runs"] == 1


def test_failed_freeze_and_failed_control_release_transaction_without_partial_latch(store):
    reserve(store)
    with closing(sqlite3.connect(store.path)) as conn, conn:
        conn.execute("""CREATE TRIGGER fail_command BEFORE INSERT ON group_run_stop_commands
            BEGIN SELECT RAISE(ABORT,'PRIVATE /private/database'); END""")
    with pytest.raises(storage.GroupStopStorageUnavailable) as failure:
        store.freeze_room_scope(IDENTITY, "retry")
    assert "PRIVATE" not in str(failure.value) and failure.value.status == 503
    assert not store.is_scope_frozen(room_run_scope_key(IDENTITY))
    with store._lock:
        assert not store._conn.in_transaction
    with closing(sqlite3.connect(store.path)) as conn, conn:
        conn.execute("DROP TRIGGER fail_command")
    with pytest.raises(RuntimeError, match="decision failed"):
        with store.group_control_open(room_run_scope_key(IDENTITY)) as allowed:
            assert allowed
            raise RuntimeError("decision failed")
    assert store.freeze_room_scope(IDENTITY, "retry")["counts"]["nonterminal"] == 1


def test_existing_schema_migrates_without_reconstructing_legacy_scope(tmp_path):
    path = tmp_path / "legacy.db"
    scope = room_run_scope_key(IDENTITY)
    with closing(sqlite3.connect(path)) as legacy:
        legacy.execute("""CREATE TABLE run_idempotency(scope TEXT NOT NULL,idempotency_key TEXT NOT NULL,
            fingerprint TEXT NOT NULL,run_id TEXT NOT NULL,status_json TEXT NOT NULL,
            created_at REAL NOT NULL,updated_at REAL NOT NULL,PRIMARY KEY(scope,idempotency_key))""")
        sql = "INSERT INTO run_idempotency VALUES (?,?,?,?,?,?,?)"
        legacy.execute(sql, (scope, "key", "fp", "legacy-run", '{"status":"running","output":"PRIVATE"}', 1, 1))
        legacy.commit()
        with closing(storage.RunIdempotencyStore(str(path))) as upgraded:
            snap = upgraded.freeze_room_scope(IDENTITY, "upgrade")
            assert snap["runs"] == [{"run_id": "legacy-run", "status": "running", "owner_pid": 0, "owner_started": 0}]
            assert upgraded.lookup(scope, "key", "fp")[0] == "reused"
            with pytest.raises(sqlite3.IntegrityError, match="frozen"):
                legacy.execute("""INSERT INTO run_idempotency(scope,idempotency_key,fingerprint,run_id,status_json,created_at,updated_at)
                    VALUES (?,?,?,?,?,?,?)""", (scope, "late", "fp", "late", '{"status":"queued"}', 1, 1))


def test_memory_store_does_not_claim_a_durable_stop():
    with closing(storage.RunIdempotencyStore(":memory:")) as memory:
        assert memory.path is None and not memory.durable
        reserve(memory)
        for operation in (
            lambda: memory.freeze_room_scope(IDENTITY, "stop"),
            lambda: memory.room_stop_snapshot("stop"),
            lambda: memory.is_scope_frozen(room_run_scope_key(IDENTITY)),
        ):
            with pytest.raises(storage.GroupStopStorageUnavailable):
                operation()
        with pytest.raises(storage.GroupStopStorageUnavailable):
            with memory.group_control_open(room_run_scope_key(IDENTITY)):
                pytest.fail("memory control gate must not open")


def test_oversized_or_invalid_legacy_summary_cannot_expose_body_or_claim_terminal(store):
    scope = room_run_scope_key(IDENTITY)
    store.reserve(scope, "large", "fp", "large-run", {"status": "completed", "output": "PRIVATE" * 160000})
    store.reserve(scope, "invalid-id", "fp", "/private/credential", {"status": "queued"})
    snapshot = store.freeze_room_scope(IDENTITY, "bounded-status")
    assert snapshot["counts"] == {"total": 2, "terminal": 0, "nonterminal": 0, "unknown": 2}
    assert snapshot["truncated"]
    assert snapshot["runs"] == [{"run_id": "large-run", "status": "unknown", "owner_pid": 0, "owner_started": 0}]
    assert len(json.dumps(snapshot)) < 2000
    assert "PRIVATE" not in json.dumps(snapshot) and "/private" not in json.dumps(snapshot)
