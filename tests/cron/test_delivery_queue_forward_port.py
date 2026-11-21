"""Approved durable queue invariants on the v0.23 schema."""
import json
import sqlite3
from contextlib import closing

import pytest

from cron import delivery_queue as queue


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "cron" / "deliveries.db")
    return tmp_path


@pytest.mark.parametrize("existing", ["upstream", "partial", "approved"])
def test_schema_migration_preserves_rows_and_retry_metadata(store, existing):
    queue._path().parent.mkdir(parents=True)
    with closing(sqlite3.connect(queue._path())) as conn, conn:
        conn.execute("""CREATE TABLE deliveries (
            execution_id TEXT PRIMARY KEY, job_json TEXT NOT NULL, content TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending','delivering','delivered','failed','unknown','suppressed')),
            owner_process_id TEXT, owner_pid INTEGER, owner_started_at INTEGER,
            created_at TEXT NOT NULL, finished_at TEXT, error TEXT)""")
        conn.execute("""CREATE TABLE delivery_tombstones (
            execution_id TEXT PRIMARY KEY, terminal_status TEXT NOT NULL CHECK(terminal_status IN
            ('delivered','failed','unknown','suppressed')), finished_at TEXT)""")
        if existing != "upstream":
            conn.execute("ALTER TABLE deliveries ADD COLUMN payload_hash TEXT")
        if existing == "approved":
            conn.execute("ALTER TABLE deliveries ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0")
            conn.execute("ALTER TABLE deliveries ADD COLUMN next_attempt_at REAL DEFAULT 0")
            conn.execute("ALTER TABLE delivery_tombstones ADD COLUMN payload_hash TEXT")
        for state in ("pending", "delivering", "suppressed", "delivered"):
            conn.execute("INSERT INTO deliveries(execution_id,job_json,content,status,created_at) VALUES (?, '{}','kept',?,'2026-09-17')", (state, state))
        conn.execute("INSERT INTO delivery_tombstones(execution_id,terminal_status) VALUES ('old','unknown')")
        if existing == "approved":
            conn.execute("UPDATE deliveries SET retry_count=12,next_attempt_at=NULL,payload_hash='kept-hash' WHERE execution_id='pending'")
    pending = queue.get_status("pending")
    assert {"payload_hash", "retry_count", "next_attempt_at"} <= pending.keys()
    assert pending["content"] == "kept"
    assert pending["retry_count"] == (12 if existing == "approved" else 0)
    if existing == "approved":
        assert pending["next_attempt_at"] is None
        assert pending["payload_hash"] == "kept-hash"
    assert queue.get_status("delivering")["status"] == "delivering"
    assert queue.get_status("suppressed")["status"] == "suppressed"
    assert queue.get_status("old")["status"] == "unknown"
    with queue._transaction() as conn:
        assert "payload_hash" in {r[1] for r in conn.execute("PRAGMA table_info(delivery_tombstones)")}
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2


@pytest.mark.parametrize("state", ["pending", "delivering", "delivered", "suppressed", "unknown"])
@pytest.mark.parametrize("change", ["message", "destination", "category"])
def test_hashed_admission_rejects_collision_in_every_state(store, state, change):
    args = dict(rendered_message='[Daily] redacted body', destination=store, category="result")
    original = queue.enqueue("identity", {"id": "job"}, "body", **args)
    assert original["payload_hash"]
    with queue._transaction() as conn:
        conn.execute("UPDATE deliveries SET status=? WHERE execution_id='identity'", (state,))
    assert queue.enqueue("identity", {"id": "job"}, "body", **args)["status"] == state
    args[{"message": "rendered_message", "destination": "destination", "category": "category"}[change]] = {
        "message": "[Renamed] redacted body", "destination": store / "other", "category": "diagnostic"
    }[change]
    with pytest.raises(ValueError, match="different payload"):
        queue.enqueue("identity", {"id": "job"}, "body", **args)


def test_explicit_hash_is_validated_even_when_new_call_omits_it(store):
    queue.enqueue("identity", {}, "body", payload_hash="approved-digest")
    with pytest.raises(ValueError, match="different payload"):
        queue.enqueue("identity", {}, "changed")


@pytest.mark.parametrize("keep", [0, 1])
def test_terminal_scrub_keeps_collision_hash_in_rows_and_tombstones(store, monkeypatch, keep):
    monkeypatch.setattr(queue, "MAX_TERMINAL_DELIVERIES", keep)
    queue.enqueue("identity", {"id": "job", "prompt": "private"}, "body", payload_hash="digest")
    queue.claim_next()
    assert queue._finish("identity", error=None, suppressed=True)
    row = queue.get_status("identity")
    assert row["payload_hash"] == "digest"
    assert row["status"] == "suppressed"
    assert not row.get("content") and row.get("job_json", "{}") == "{}"
    assert queue.enqueue("identity", {}, "body", payload_hash="digest")["status"] == "suppressed"
    with pytest.raises(ValueError, match="different payload"):
        queue.enqueue("identity", {}, "changed", payload_hash="changed-digest")


@pytest.mark.parametrize("count,due", [(1, 1001), (12, None), (12, 0)])
def test_claim_skips_future_and_exhausted_rows(store, monkeypatch, count, due):
    import time
    monkeypatch.setattr(time, "time", lambda: 1000)
    queue.enqueue("parked", {}, "body")
    with queue._transaction() as conn:
        conn.execute("UPDATE deliveries SET retry_count=?,next_attempt_at=? WHERE execution_id='parked'", (count, due))
    queue.enqueue("ready", {}, "other")
    assert queue.claim_next()["execution_id"] == "ready"
    assert queue.claim_next() is None
    if count < 12:
        monkeypatch.setattr(time, "time", lambda: 1001)
        assert queue.claim_next()["execution_id"] == "parked"


@pytest.mark.parametrize("fence", ["same", "process", "pid", "terminal", "identity", "untyped"])
def test_refusal_deferral_is_typed_and_owner_fenced(store, fence):
    queue.enqueue("identity", {"_bot_chat_pending": str(store)}, "body")
    queue.claim_next()
    with queue._transaction() as conn:
        if fence == "process":
            conn.execute("UPDATE deliveries SET owner_process_id='replacement'")
        elif fence == "pid":
            conn.execute("UPDATE deliveries SET owner_pid=-1")
        elif fence == "terminal":
            conn.execute("UPDATE deliveries SET status='unknown'")
    before = queue.get_status("identity")
    refusal = "not admitted" if fence == "untyped" else queue.KnownNotAdmitted("not admitted")
    changed = queue._defer_known_refusal("other" if fence == "identity" else "identity", refusal)
    after = queue.get_status("identity")
    assert changed is (fence == "same")
    if fence == "same":
        assert after["status"] == "pending" and after["retry_count"] == 1
        assert after["owner_pid"] is None and after["owner_process_id"] is None
    else:
        assert after == before


def test_typed_refusal_backoff_parks_after_twelve_attempts(store, monkeypatch):
    clock = [1000]
    monkeypatch.setattr(queue.time, "time", lambda: clock[0])
    queue.enqueue("identity", {"_bot_chat_pending": str(store)}, "body")
    for attempt, delay in enumerate([30, 60, 120, 240, 300, 300, 300, 300, 300, 300, 300, None], 1):
        assert queue.drain(lambda *args: queue.KnownNotAdmitted("refused")) == 1
        row = queue.get_status("identity")
        assert row["status"] == "pending" and row["retry_count"] == attempt
        assert row["content"] == "body"
        assert row["next_attempt_at"] == (clock[0] + delay if delay else None)
        assert queue.claim_next() is None
        clock[0] = row["next_attempt_at"] or 1000000
    assert queue.claim_next() is None


@pytest.mark.parametrize("failure", ["not admitted", ValueError("not admitted"), TimeoutError("timeout")])
def test_unproven_bot_chat_failures_are_terminal_unknown(store, failure):
    queue.enqueue("identity", {"_bot_chat_pending": str(store)}, "body")
    def send(*args):
        if isinstance(failure, Exception):
            raise failure
        return failure
    assert queue.drain(send) == 1
    row = queue.get_status("identity")
    assert row["status"] == "unknown" and row["retry_count"] == 0
    assert queue.drain(send) == 0


def test_typed_refusal_does_not_replay_mixed_target_jobs(store):
    queue.enqueue("identity", {"deliver": ["bot-chat", "telegram:1"]}, "body")
    refusal = queue.KnownNotAdmitted("refused")
    assert queue.drain(lambda *args: refusal) == 1
    assert queue.get_status("identity")["status"] == "failed"
    assert queue.claim_next() is None


@pytest.mark.parametrize("count,due", [(1, 999999999999), (12, None)])
def test_claim_update_rechecks_retry_eligibility(store, monkeypatch, count, due):
    queue.enqueue("identity", {}, "body")
    connect = sqlite3.connect
    class InterleavedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            cursor = super().execute(sql, parameters)
            if sql.startswith("SELECT execution_id FROM deliveries WHERE status='pending'"):
                # The SELECT has evaluated eligibility; simulate a refusal
                # published in the check-to-claim gap before the guarded UPDATE.
                super().execute("UPDATE deliveries SET retry_count=?,next_attempt_at=?", (count, due))
            return cursor
    monkeypatch.setattr(queue.sqlite3, "connect", lambda *args, **kwargs: connect(*args, **kwargs, factory=InterleavedConnection))
    assert queue.claim_next() is None
    assert queue.get_status("identity")["status"] == "pending"


@pytest.mark.parametrize("change", ["body", "name", "destination", "category"])
def test_specialized_envelope_computes_hash_when_caller_does_not_supply_one(store, change):
    from cron import bot_chat_legacy
    job = {"id": "job", "name": "Daily", "_bot_chat_pending": str(store)}
    row = queue.enqueue("identity", job, "body")
    assert row["payload_hash"] == bot_chat_legacy.payload_hash(job, "body", store)
    changed = dict(job)
    if change == "name":
        changed["name"] = "Renamed"
    if change == "destination":
        changed["_bot_chat_pending"] = str(store / "other")
    with pytest.raises(ValueError, match="different payload"):
        queue.enqueue("identity", changed, "changed" if change == "body" else "body",
                      for_failure=change == "category")
