"""B3 boundaries: imported digests retain, but never impersonate, executions."""
import json
import sqlite3

import pytest

from cron import delivery_outcome, delivery_queue as queue, executions, jobs, scheduler


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "cron/deliveries.db")
    monkeypatch.setattr(scheduler, "load_config", dict)
    monkeypatch.setattr(scheduler, "run_job", lambda *a, **kw: (True, "raw", "body", None))
    monkeypatch.setattr(queue, "_owner_is_live", lambda *a: False)
    return tmp_path


@pytest.mark.parametrize("keep", [0, 1000])
@pytest.mark.parametrize("pre_upgrade", [True, False])
def test_abandoned_import_settles_the_original_run(store, monkeypatch, keep, pre_upgrade):
    job = jobs.create_job(prompt="p", schedule="every 1h", deliver="bot-chat")
    scheduler.run_one_job(job)
    execution = executions.latest_execution(job["id"])["id"]
    digest = "a" * 64
    with queue._transaction() as conn:
        conn.execute("UPDATE deliveries SET execution_id=? WHERE execution_id=?", (digest, execution))
    if not pre_upgrade:
        assert queue.claim_next()["execution_id"] == digest
        queue._ACTIVE_DELIVERIES.discard(digest)
    # Simulate a predecessor claim without the new attribution column contents.
    with sqlite3.connect(queue.DELIVERY_DB) as conn:
        conn.execute("UPDATE deliveries SET status='delivering', owner_process_id='dead', owner_pid=999999")
        if pre_upgrade:
            conn.execute("UPDATE deliveries SET bot_chat_receipt=NULL")
    monkeypatch.setattr(queue, "MAX_TERMINAL_DELIVERIES", keep)
    assert queue.recover_abandoned() == 1
    assert executions.get_execution(execution)["delivery_outcome"] == "failed"
    assert jobs.get_job(job["id"])["last_status"] == "delivery_failed"
    assert delivery_outcome.settled_outcome(execution) == "failed"
    assert delivery_outcome.settled_outcome(digest) is None
    row = queue.get_status(digest)
    ref = json.loads(row["bot_chat_receipt"])
    assert ref["execution_id"] == execution and ref["job_id"] == job["id"]
    assert row.get("job_json", "{}") == "{}" and row.get("content", "") == ""
    assert queue.recover_abandoned() == 0
    assert queue.claim_next() is None
    if keep == 0:
        with queue._transaction() as conn:
            assert conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM delivery_tombstones").fetchone()[0] == 1


def old_database(home, status, payload):
    path = home / "cron/deliveries.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE deliveries (
            execution_id TEXT PRIMARY KEY, job_json TEXT NOT NULL, content TEXT NOT NULL,
            for_failure INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL CHECK(status IN ('pending','delivering','delivered','failed','unknown')),
            owner_process_id TEXT, owner_pid INTEGER, owner_started_at INTEGER,
            created_at TEXT NOT NULL, finished_at TEXT, error TEXT)""")
        conn.execute("""CREATE TABLE delivery_tombstones (execution_id TEXT PRIMARY KEY,
            terminal_status TEXT NOT NULL CHECK(terminal_status IN ('delivered','failed','unknown')),
            finished_at TEXT)""")
        if status == "tombstone":
            conn.execute("INSERT INTO delivery_tombstones VALUES (?, 'delivered', '2026-01-01')", ("b" * 64,))
        else:
            conn.execute("""INSERT INTO deliveries
                (execution_id, job_json, content, status, created_at, finished_at)
                VALUES (?, ?, 'private payload', ?, '2026-01-01', '2026-01-01')""",
                ("b" * 64, json.dumps(payload), status))


@pytest.mark.parametrize("status", ["delivering", "unknown", "failed", "delivered"])
def test_old_schema_backfills_before_terminal_pruning(store, monkeypatch, status):
    old_database(store, status, {"id": "job", "execution_id": "original-execution",
        "_bot_chat_pending": str(store), "_bot_chat_delivery_id": "c" * 64})
    row = queue.get_execution_status("original-execution")
    assert row is not None, "schema migration did not backfill the original execution"
    assert row["execution_id"] == "b" * 64
    ref = json.loads(row["bot_chat_receipt"])
    assert ref == {"execution_id": "original-execution", "job_id": "job",
                   "home": str(store), "delivery_id": "c" * 64}
    assert delivery_outcome.settled_outcome("b" * 64) is None
    monkeypatch.setattr(queue, "MAX_TERMINAL_DELIVERIES", 0)
    with queue._transaction() as conn:
        conn.execute("UPDATE deliveries SET status='unknown'")
        queue._prune_terminal_unlocked(conn)
    assert json.loads(queue.get_status("b" * 64)["bot_chat_receipt"]) == ref
    assert delivery_outcome.settled_outcome("original-execution") == "failed"
    assert delivery_outcome.settled_outcome("b" * 64) is None


@pytest.mark.parametrize("status", ["unknown", "tombstone"])
def test_scrubbed_digest_does_not_invent_execution_identity(store, status):
    old_database(store, status, {})
    row = queue.get_status("b" * 64)
    assert row["bot_chat_receipt"]
    ref = json.loads(row["bot_chat_receipt"])
    assert ref["execution_id"] is None
    assert ref["reference_error"] == "original execution identity unavailable"
    assert queue.get_execution_status("original-execution") is None
    assert delivery_outcome.settled_outcome("b" * 64) is None
    assert queue.get_status("b" * 64)["status"] == row["status"]
