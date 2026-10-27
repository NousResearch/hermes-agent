"""Diagnostic: source-owned receipt budgets must precede receipt JSON parsing.

AGC-1663 private causal probe only; no production snapshot correction is included.
"""
import json
import time

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as runtime


def _snapshot(db, epoch, max_bytes=65536):
    return runtime.snapshot_runtime_admission_workers(
        db, epoch=epoch, deadline=time.monotonic() + 30.0,
        max_records=16, max_bytes=max_bytes,
    )


def _prepared_store(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("primary", source="test")
    epoch = runtime.begin_runtime_epoch(db, instance_id="synthetic-owner")
    runtime.register_worker_execution(
        db, epoch=epoch, execution_id="worker", session_id="primary", generation=0,
        kind="compute", adoption_secret="synthetic-adoption-secret",
    )
    result = runtime.mutate_worker_execution(
        db, epoch=epoch, execution_id="worker", session_id="primary", generation=0,
        sequence=1, operation="transcript.append",
        payload={"messages": [{"role": "assistant", "content": "synthetic first write"}]},
    )
    assert result["count"] == 1
    return db, epoch


def _following_real_write(db, epoch, busy_timeout):
    assert not db._conn.in_transaction
    assert db._conn.execute("PRAGMA busy_timeout").fetchone()[0] == busy_timeout
    result = runtime.mutate_worker_execution(
        db, epoch=epoch, execution_id="worker", session_id="primary", generation=0,
        sequence=2, operation="transcript.append",
        payload={"messages": [{"role": "assistant", "content": "synthetic follow-up write"}]},
    )
    assert result["count"] == 1
    assert db._read_one(
        "SELECT last_sequence FROM worker_executions WHERE execution_id=?", ("worker",),
    )["last_sequence"] == 2
    assert not db._conn.in_transaction


def test_real_worker_receipt_snapshot_positive_and_following_writer(tmp_path):
    db, epoch = _prepared_store(tmp_path)
    try:
        busy_timeout = db._conn.execute("PRAGMA busy_timeout").fetchone()[0]
        snapshot = _snapshot(db, epoch)
        receipt = db._read_one(
            "SELECT result_json FROM worker_receipts WHERE execution_id=? AND sequence=1",
            ("worker",),
        )
        assert snapshot["counts"]["records"] == 2
        assert snapshot["worker_receipts"][0]["result_bytes"] == len(
            receipt["result_json"].encode("utf-8"))
        assert snapshot["workers"][0]["status"] == "running"
        _following_real_write(db, epoch, busy_timeout)
    finally:
        db.close()


@pytest.mark.parametrize("valid_json", [True, False], ids=["valid-oversize", "malformed-oversize"])
def test_oversize_receipt_refuses_budget_before_body_json_parse(tmp_path, valid_json):
    db, epoch = _prepared_store(tmp_path)
    try:
        baseline = _snapshot(db, epoch)
        budget = baseline["counts"]["encoded_bytes"] + 128
        body = json.dumps({"synthetic": "a" * 16384})
        if not valid_json:
            body = body[:-1]
        assert len(body.encode("utf-8")) > budget
        changed = db._execute_write(lambda conn: conn.execute(
            "UPDATE worker_receipts SET result_json=? WHERE execution_id=? AND sequence=1",
            (body, "worker"),
        ).rowcount)
        assert changed == 1
        busy_timeout = db._conn.execute("PRAGMA busy_timeout").fetchone()[0]
        with pytest.raises(runtime.RuntimeStoreError) as exc:
            _snapshot(db, epoch, max_bytes=budget)
        # Check cleanup before the causal assertion, including on the expected old-source RED.
        _following_real_write(db, epoch, busy_timeout)
        print(json.dumps({"diagnostic": "receipt-budget-order", "validJson": valid_json,
                          "baselineEncodedBytes": baseline["counts"]["encoded_bytes"],
                          "maxBytes": budget, "resultBytes": len(body.encode("utf-8")),
                          "refusal": exc.value.reason, "followupWriterSucceeded": True}), flush=True)
        assert exc.value.reason == "snapshot_limit"
    finally:
        db.close()
