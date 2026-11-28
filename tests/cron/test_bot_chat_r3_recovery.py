"""Keep original execution identity when fencing and migrating digest owners."""
import json
import sqlite3

import pytest
from cron import delivery_queue as queue, delivery_outcome, executions, jobs, scheduler


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "cron/deliveries.db")
    monkeypatch.setattr(scheduler, "load_config", dict)
    monkeypatch.setattr(scheduler, "run_job", lambda job, **kw: (True, "raw", "body", None))
    monkeypatch.setattr(queue, "_owner_is_live", lambda *args: False)
    return tmp_path


@pytest.mark.parametrize("pre_upgrade", [False, True])
@pytest.mark.parametrize("keep", [0, 1000])
def test_abandoned_digest_settles_original_job_and_ledger(store, monkeypatch, pre_upgrade, keep):
    job = jobs.create_job(prompt="p", schedule="every 1h", deliver="bot-chat")
    scheduler.run_one_job(job)
    execution = executions.latest_execution(job["id"])["id"]
    key = "a" * 64
    with queue._transaction() as conn:
        conn.execute("UPDATE deliveries SET execution_id=? WHERE execution_id=?", (key, execution))
    if not pre_upgrade:
        assert queue.claim_next()["execution_id"] == key
        queue._ACTIVE_DELIVERIES.discard(key)
    # A predecessor claim did not yet have bot_chat_receipt. Keep real job_json.
    with sqlite3.connect(queue.DELIVERY_DB) as conn:
        conn.execute("UPDATE deliveries SET status='delivering', owner_process_id='gone', owner_pid=999999")
        if pre_upgrade:
            conn.execute("UPDATE deliveries SET bot_chat_receipt=NULL")
    monkeypatch.setattr(queue, "MAX_TERMINAL_DELIVERIES", keep)
    assert queue.recover_abandoned() == 1
    saved = jobs.get_job(job["id"])
    assert saved["last_status"] == "delivery_failed", "original job remains delivery_queued"
    assert executions.get_execution(execution)["delivery_outcome"] == "failed"
    assert delivery_outcome.settled_outcome(execution) == "failed"
    assert delivery_outcome.settled_outcome(key) is None
    row = queue.get_status(key)
    ref = json.loads(row["bot_chat_receipt"])
    assert ref["execution_id"] == execution and ref["job_id"] == job["id"]
    assert row.get("job_json", "{}") == "{}" and row.get("content", "") == ""
    assert queue.claim_next() is None
    assert queue.recover_abandoned() == 0


def _pre_upgrade_database(home, status, job):
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
                VALUES (?, ?, 'private body', ?, '2026-01-01', '2026-01-01')""",
                ("b" * 64, json.dumps(job), status))


@pytest.mark.parametrize("status", ["delivering", "unknown"])
def test_schema_backfills_specialized_reference_before_scrubbing(store, monkeypatch, status):
    _pre_upgrade_database(store, status, {"id": "job", "execution_id": "original-run",
        "_bot_chat_pending": str(store), "_bot_chat_delivery_id": "c" * 64})
    # Schema initialization alone must recover references, before a claim or prune.
    row = queue.get_status("b" * 64)
    assert row["bot_chat_receipt"], "schema migration lost the available original execution"
    reference = json.loads(row["bot_chat_receipt"])
    assert reference == {"execution_id": "original-run", "job_id": "job",
                         "delivery_id": "c" * 64, "home": str(store)}
    monkeypatch.setattr(queue, "MAX_TERMINAL_DELIVERIES", 0)
    with queue._transaction() as conn:
        conn.execute("UPDATE deliveries SET status='unknown'")
        queue._prune_terminal_unlocked(conn)
    assert json.loads(queue.get_status("b" * 64)["bot_chat_receipt"]) == reference
    assert delivery_outcome.settled_outcome("original-run") == "failed"
    assert delivery_outcome.settled_outcome("b" * 64) is None


@pytest.mark.parametrize("status", ["unknown", "tombstone"])
def test_already_scrubbed_digest_preserves_unknown_original_identity(store, status):
    _pre_upgrade_database(store, status, {})
    row = queue.get_status("b" * 64)
    assert row["bot_chat_receipt"], "unrecoverable original identity must be explicit"
    reference = json.loads(row["bot_chat_receipt"])
    assert reference["execution_id"] is None
    assert reference["reference_error"] == "original execution identity unavailable"
    assert queue.get_execution_status("original-run") is None
    assert delivery_outcome.settled_outcome("b" * 64) is None
    assert queue.get_status("b" * 64)["status"] == row["status"]
