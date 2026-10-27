"""Causal controls for the source-owned preservation runtime census."""

import json
import time

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as runtime


def _store(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("primary", source="test")
    epoch = runtime.begin_runtime_epoch(db, instance_id="instance-a")
    return db, epoch


def _deadline():
    return time.monotonic() + 2.0


def _admit(db, epoch, *, session_id="primary", request_id="request", principal_id="synthetic-owner"):
    return runtime.admit_session_input(
        db,
        epoch=epoch,
        principal_id=principal_id,
        session_id=session_id,
        request_id=request_id,
        payload={"text": f"payload-{request_id}"},
    )


def _snapshot(db, epoch, **kwargs):
    options = {"max_records": 128, "max_bytes": 2 * 1024 * 1024}
    options.update(kwargs)
    return runtime.snapshot_runtime_admission_workers(
        db, epoch=epoch, deadline=_deadline(), **options,
    )


def test_snapshot_reads_pending_admissions_workers_and_producer_encoded_receipt_bytes(tmp_path):
    db, epoch = _store(tmp_path)
    try:
        worker = runtime.register_worker_execution(
            db, epoch=epoch, execution_id="worker-a", session_id="primary", generation=0,
            kind="compute", adoption_secret="synthetic-adoption-secret",
        )
        runtime.mutate_worker_execution(
            db, epoch=epoch, execution_id=worker["execution_id"], session_id="primary", generation=0,
            sequence=1, operation="transcript.append",
            payload={"messages": [{"role": "assistant", "content": "réponse 漢字"}]},
        )
        first = _admit(db, epoch, request_id="request-a")
        second = _admit(db, epoch, request_id="request-b")

        snapshot = _snapshot(db, epoch)

        assert snapshot["schema"] == "runtime-admission-snapshot-v1"
        assert snapshot["epoch"] == epoch
        assert snapshot["instance_id"] == "instance-a"
        assert [row["admission_id"] for row in snapshot["admissions"]] == [
            first["admission_id"], second["admission_id"],
        ]
        assert all(row["status"] == "queued" for row in snapshot["admissions"])
        assert snapshot["workers"] == [{
            "execution_id": "worker-a", "session_id": "primary", "kind": "compute",
            "owner_epoch": epoch, "generation": 0, "status": "running", "last_sequence": 1,
        }]
        stored_receipt = db._read_one(
            "SELECT payload_digest, result_json FROM worker_receipts WHERE execution_id=? AND sequence=1",
            ("worker-a",),
        )
        assert snapshot["worker_receipts"] == [{
            "execution_id": "worker-a", "sequence": 1,
            "payload_digest": stored_receipt["payload_digest"],
            "result_bytes": len(stored_receipt["result_json"].encode("utf-8", errors="surrogatepass")),
        }]
        assert snapshot["counts"]["records"] == 4
        assert snapshot["counts"]["encoded_bytes"] >= snapshot["worker_receipts"][0]["result_bytes"]
        assert snapshot["counts"]["admissions_by_status"] == {"queued": 2, "started": 0, "unknown": 0}
        assert snapshot["counts"]["workers_by_status"] == {"registered": 0, "running": 1, "unknown": 0}
    finally:
        db.close()


def test_snapshot_keeps_prior_epoch_unknown_rows_without_authority_registry(tmp_path):
    db, epoch = _store(tmp_path)
    db_path = db.db_path
    try:
        admitted = _admit(db, epoch)
        claimed = runtime.claim_session_input(db, epoch=epoch, session_id="primary")
        assert claimed["admission_id"] == admitted["admission_id"]
        worker = runtime.register_worker_execution(
            db, epoch=epoch, execution_id="worker-old", session_id="primary",
            generation=claimed["generation"], kind="child", adoption_secret="synthetic-adoption-secret",
        )
        runtime.mutate_worker_execution(
            db, epoch=epoch, execution_id=worker["execution_id"], session_id="primary",
            generation=claimed["generation"], sequence=1, operation="transcript.append",
            payload={"messages": [{"role": "assistant", "content": "before restart"}]},
        )
        replacement_epoch = runtime.begin_runtime_epoch(db, instance_id="instance-b")
        assert runtime.recover_session_inputs(db, epoch=replacement_epoch) == 1
        db.close()
        db = SessionDB(db_path=db_path)

        snapshot = _snapshot(db, replacement_epoch)

        admission = snapshot["admissions"][0]
        assert admission["target_session_id"] == "primary"
        assert admission["status"] == "unknown"
        assert admission["owner_epoch"] == epoch
        assert admission["generation"] == claimed["generation"]
        worker_row = snapshot["workers"][0]
        assert worker_row["execution_id"] == "worker-old"
        assert worker_row["status"] == "unknown"
        assert worker_row["owner_epoch"] == epoch
        assert snapshot["worker_receipts"][0]["execution_id"] == "worker-old"
        assert snapshot["instance_id"] == "instance-b"
    finally:
        db.close()


def test_snapshot_accepts_physical_lineage_child_without_tail_equality(tmp_path):
    db, epoch = _store(tmp_path)
    try:
        admitted = _admit(db, epoch)
        claimed = runtime.claim_session_input(db, epoch=epoch, session_id="primary")
        assert claimed["admission_id"] == admitted["admission_id"]
        worker = runtime.register_worker_execution(
            db, epoch=epoch, execution_id="worker-publish", session_id="primary",
            generation=claimed["generation"], kind="child", adoption_secret="synthetic-adoption-secret",
        )
        holder = "synthetic-compression-holder"
        runtime.mutate_worker_execution(
            db, epoch=epoch, execution_id=worker["execution_id"], session_id="primary",
            generation=claimed["generation"], sequence=1, operation="compression.lock.acquire",
            payload={"holder": holder, "ttl_seconds": 300.0},
        )
        runtime.mutate_worker_execution(
            db, epoch=epoch, execution_id=worker["execution_id"], session_id="primary",
            generation=claimed["generation"], sequence=2, operation="compression.publish",
            payload={
                "child_session_id": "physical-child", "source": "test",
                "messages": [{"role": "assistant", "content": "physical handoff"}],
                "model": None, "model_config": None, "system_prompt": None, "cwd": None,
                "profile_name": None, "compression_lock_holder": holder,
                "require_compression_lease": True, "require_lease_refresh": False,
                "lease_ttl_seconds": 300.0, "watermark": None, "watermark_ceiling": None,
            },
        )
        row = _snapshot(db, epoch)["admissions"][0]
        assert row["target_session_id"] == "primary"
        assert row["lineage"] == ["primary", "physical-child"]
    finally:
        db.close()


def test_snapshot_uses_one_shared_record_budget_and_utf8_bytes(tmp_path):
    db, epoch = _store(tmp_path)
    try:
        worker = runtime.register_worker_execution(
            db, epoch=epoch, execution_id="worker-budget", session_id="primary", generation=0,
            kind="compute", adoption_secret="synthetic-adoption-secret",
        )
        runtime.mutate_worker_execution(
            db, epoch=epoch, execution_id=worker["execution_id"], session_id="primary", generation=0,
            sequence=1, operation="transcript.append",
            payload={"messages": [{"role": "assistant", "content": "budget"}]},
        )
        _admit(db, epoch, request_id="budget-admission")
        with pytest.raises(runtime.RuntimeStoreError) as exc:
            _snapshot(db, epoch, max_records=1)
        assert exc.value.reason == "snapshot_limit"
        assert not db._conn.in_transaction

        unicode_session = "é" * 32
        db.create_session(unicode_session, source="test")
        unicode_admission = _admit(
            db, epoch, session_id=unicode_session, request_id="请求-1", principal_id="主" * 32,
        )
        positive = _snapshot(db, epoch)
        record = next(row for row in positive["admissions"] if row["admission_id"] == unicode_admission["admission_id"])
        utf8_bytes = len(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8", errors="surrogatepass"
        ))
        unicode_chars = len(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        assert utf8_bytes > unicode_chars
        with pytest.raises(runtime.RuntimeStoreError) as exc:
            _snapshot(db, epoch, max_bytes=utf8_bytes - 1)
        assert exc.value.reason == "snapshot_limit"
        assert not db._conn.in_transaction
    finally:
        db.close()


def test_snapshot_refuses_closed_owner_without_reopen(tmp_path):
    db, epoch = _store(tmp_path)
    db.close()
    with pytest.raises(runtime.RuntimeStoreError) as exc:
        runtime.snapshot_runtime_admission_workers(db, epoch=epoch, deadline=_deadline())
    assert exc.value.reason == "snapshot_closed"
    assert db._conn is None
    assert db._read_conns_closed


def test_snapshot_lock_wait_is_deadline_bounded_and_recovers(tmp_path):
    db, epoch = _store(tmp_path)
    try:
        db._lock.acquire()
        try:
            with pytest.raises(runtime.RuntimeStoreError) as exc:
                runtime.snapshot_runtime_admission_workers(db, epoch=epoch, deadline=_deadline())
            assert exc.value.reason == "snapshot_deadline"
        finally:
            db._lock.release()
        assert not db._conn.in_transaction
        assert _snapshot(db, epoch)["epoch"] == epoch
    finally:
        db.close()


def test_snapshot_rejects_midcall_epoch_drift_and_next_call_recovers(tmp_path, monkeypatch):
    db, epoch = _store(tmp_path)
    try:
        _admit(db, epoch)
        original_read_epoch = runtime._snapshot_read_epoch
        drifted = False

        def drift_once(owner, deadline):
            nonlocal drifted
            if not drifted:
                drifted = True
                runtime.begin_runtime_epoch(owner, instance_id="instance-drift")
            return original_read_epoch(owner, deadline)

        monkeypatch.setattr(runtime, "_snapshot_read_epoch", drift_once)
        with pytest.raises(runtime.RuntimeStoreError) as exc:
            _snapshot(db, epoch)
        assert exc.value.reason == "stale_epoch"
        assert not db._conn.in_transaction
        monkeypatch.setattr(runtime, "_snapshot_read_epoch", original_read_epoch)
        current = db._read_one("SELECT epoch FROM runtime_epoch WHERE singleton=1")["epoch"]
        assert _snapshot(db, current)["epoch"] == current
    finally:
        db.close()


def test_snapshot_refuses_malformed_lineage_and_leaves_writer_usable(tmp_path):
    db, epoch = _store(tmp_path)
    try:
        admitted = _admit(db, epoch)
        db._execute_write(lambda conn: conn.execute(
            "UPDATE session_admissions SET lineage_json=? WHERE admission_id=?",
            (json.dumps(["unrelated-session"]), admitted["admission_id"]),
        ))
        with pytest.raises(runtime.RuntimeStoreError) as exc:
            _snapshot(db, epoch)
        assert exc.value.reason == "snapshot_schema"
        assert not db._conn.in_transaction
        assert _admit(db, epoch, request_id="after-malformed")
    finally:
        db.close()


def test_snapshot_refuses_nonobject_receipt_and_leaves_worker_writer_usable(tmp_path):
    db, epoch = _store(tmp_path)
    try:
        worker = runtime.register_worker_execution(
            db, epoch=epoch, execution_id="worker-malformed", session_id="primary", generation=0,
            kind="compute", adoption_secret="synthetic-adoption-secret",
        )
        runtime.mutate_worker_execution(
            db, epoch=epoch, execution_id=worker["execution_id"], session_id="primary", generation=0,
            sequence=1, operation="transcript.append",
            payload={"messages": [{"role": "assistant", "content": "valid"}]},
        )
        db._execute_write(lambda conn: conn.execute(
            "UPDATE worker_receipts SET result_json=? WHERE execution_id=? AND sequence=1",
            ("[]", worker["execution_id"]),
        ))
        with pytest.raises(runtime.RuntimeStoreError) as exc:
            _snapshot(db, epoch)
        assert exc.value.reason == "snapshot_schema"
        assert not db._conn.in_transaction
        runtime.mutate_worker_execution(
            db, epoch=epoch, execution_id=worker["execution_id"], session_id="primary", generation=0,
            sequence=2, operation="transcript.append",
            payload={"messages": [{"role": "assistant", "content": "after refusal"}]},
        )
    finally:
        db.close()


def test_snapshot_refuses_missing_runtime_schema_without_reopening_or_leaking(tmp_path):
    db, epoch = _store(tmp_path)
    try:
        db._execute_write(lambda conn: conn.execute(
            "ALTER TABLE runtime_epoch RENAME TO runtime_epoch_missing"))
        with pytest.raises(runtime.RuntimeStoreError) as exc:
            _snapshot(db, epoch)
        assert exc.value.reason == "snapshot_schema"
        assert not db._conn.in_transaction
    finally:
        db.close()


def test_snapshot_refuses_same_column_session_admissions_view_before_idle(tmp_path):
    db, epoch = _store(tmp_path)
    hidden = False
    try:
        admitted = _admit(db, epoch, request_id="before-view")
        positive = _snapshot(db, epoch)
        assert positive["counts"]["admissions_by_status"]["queued"] == 1
        assert positive["admissions"][0]["admission_id"] == admitted["admission_id"]
        original = dict(db._read_one(
            "SELECT * FROM session_admissions WHERE admission_id=?", (admitted["admission_id"],),
        ))
        columns = tuple(row[1] for row in db._conn.execute("PRAGMA table_info(session_admissions)"))

        def hide_live_rows(conn):
            conn.execute("ALTER TABLE session_admissions RENAME TO synthetic_held_admissions")
            conn.execute("CREATE VIEW session_admissions AS SELECT * FROM synthetic_held_admissions WHERE 0")

        db._execute_write(hide_live_rows)
        hidden = True
        assert tuple(row[1] for row in db._conn.execute("PRAGMA table_info(session_admissions)")) == columns
        assert db._read_one(
            "SELECT type FROM sqlite_master WHERE name='session_admissions'",
        )["type"] == "view"
        assert dict(db._read_one(
            "SELECT * FROM synthetic_held_admissions WHERE admission_id=?", (admitted["admission_id"],),
        )) == original
        observed = None
        refusal = None
        refusal_field = None
        refusal_detail = None
        try:
            observed = _snapshot(db, epoch)
        except runtime.RuntimeStoreError as exc:
            refusal = exc.reason
            refusal_field = exc.field
            refusal_detail = exc.detail
        assert not db._conn.in_transaction

        def restore_real_relation(conn):
            conn.execute("DROP VIEW session_admissions")
            conn.execute("ALTER TABLE synthetic_held_admissions RENAME TO session_admissions")

        db._execute_write(restore_real_relation)
        hidden = False
        assert dict(db._read_one(
            "SELECT * FROM session_admissions WHERE admission_id=?", (admitted["admission_id"],),
        )) == original
        assert db._read_one(
            "SELECT count(*) AS total FROM sqlite_master WHERE name='synthetic_held_admissions'",
        )["total"] == 0
        assert db._read_one(
            "SELECT type FROM sqlite_master WHERE name='session_admissions'",
        )["type"] == "table"
        restored = runtime.get_session_admission(db, admission_id=admitted["admission_id"])
        assert restored["admission_id"] == original["admission_id"]
        assert restored["status"] == original["status"]
        assert restored["payload"] == json.loads(original["payload_json"])
        assert _admit(db, epoch, request_id="after-view")["status"] == "queued"
        assert _snapshot(db, epoch)["counts"]["admissions_by_status"]["queued"] == 2
        assert not db._conn.in_transaction
        print(json.dumps({"diagnostic": "runtime-relation-kind", "sourceType": "view",
                          "heldQueuedRows": 1, "returnedCounts": observed.get("counts") if observed else None,
                          "refusal": refusal, "refusalField": refusal_field,
                          "allOriginalAdmissionValuesRetained": True,
                          "restoredWriterSucceeded": True}), flush=True)
        assert refusal == "snapshot_schema"
        assert "session_admissions" in (refusal_field or "") + (refusal_detail or "")
    finally:
        try:
            if hidden:
                def restore_on_failure(conn):
                    conn.execute("DROP VIEW session_admissions")
                    conn.execute("ALTER TABLE synthetic_held_admissions RENAME TO session_admissions")
                db._execute_write(restore_on_failure)
        finally:
            db.close()
