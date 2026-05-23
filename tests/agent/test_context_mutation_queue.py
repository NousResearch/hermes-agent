import pytest

from agent.context_dag_engine import DAGContextEngine
from agent.context_dag_store import ContextDAGStore
from agent.context_mutation_queue import (
    ContextMutationQueue,
    MutationWorker,
    default_dag_mutation_handlers,
    process_next_mutation,
)
from hermes_state import SessionDB


def _db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", "test")
    db.create_session("s2", "test")
    return db


def test_enqueue_is_idempotent_and_records_attempt_defaults(tmp_path):
    db = _db(tmp_path)
    try:
        queue = ContextMutationQueue(ContextDAGStore(db))
        first = queue.enqueue(
            "s1",
            "rebuild_projection",
            {"reason": "manual"},
            idempotency_key="same",
            max_attempts=4,
        )
        second = queue.enqueue(
            "s1",
            "rebuild_projection",
            {"reason": "ignored-on-dedupe"},
            idempotency_key="same",
            max_attempts=4,
        )

        assert second.id == first.id
        assert second.status == "queued"
        assert second.payload == {"reason": "manual"}
        assert second.attempts == 0
        assert second.max_attempts == 4
        with db._lock:
            count = db._conn.execute(
                "SELECT COUNT(*) FROM context_mutation_log WHERE session_id = ? AND idempotency_key = ?",
                ("s1", "same"),
            ).fetchone()[0]
        assert count == 1
    finally:
        db.close()


def test_worker_claims_and_completes_one_job(tmp_path):
    db = _db(tmp_path)
    try:
        store = ContextDAGStore(db)
        queue = ContextMutationQueue(store)
        job = queue.enqueue("s1", "rebuild_projection", {"projection": [{"role": "system", "content": "p"}]})
        worker = MutationWorker(queue, handlers={"rebuild_projection": lambda job: {"ok": True, "job": job.id}})

        result = worker.run_once()

        assert result is not None
        assert result.status == "succeeded"
        assert result.attempts == 1
        assert result.metadata["result"] == {"ok": True, "job": job.id}
        assert queue.get(job.id).status == "succeeded"
    finally:
        db.close()


def test_worker_retries_then_marks_dead_and_keeps_error(tmp_path):
    db = _db(tmp_path)
    try:
        queue = ContextMutationQueue(ContextDAGStore(db))
        job = queue.enqueue("s1", "compact_leaf", {"span": [1, 2]}, max_attempts=2)
        worker = MutationWorker(queue, handlers={"compact_leaf": lambda job: (_ for _ in ()).throw(RuntimeError("boom"))})

        first = worker.run_once()
        second = worker.run_once()
        third = worker.run_once()

        assert first.status == "queued"
        assert first.attempts == 1
        assert "boom" in first.error
        assert second.status == "dead"
        assert second.attempts == 2
        assert "boom" in second.error
        assert third is None
    finally:
        db.close()


def test_per_session_claim_lock_skips_running_same_session_but_allows_other_session(tmp_path):
    db = _db(tmp_path)
    try:
        queue = ContextMutationQueue(ContextDAGStore(db))
        first = queue.enqueue("s1", "reconcile_transcript", {"n": 1}, idempotency_key="s1-a")
        second = queue.enqueue("s1", "reconcile_transcript", {"n": 2}, idempotency_key="s1-b")
        third = queue.enqueue("s2", "reconcile_transcript", {"n": 3}, idempotency_key="s2-a")

        claimed_first = queue.claim_next(worker_id="w1")
        claimed_second = queue.claim_next(worker_id="w2")

        assert claimed_first.id == first.id
        assert claimed_first.status == "running"
        assert claimed_second.id == third.id
        assert claimed_second.session_id == "s2"
        assert queue.get(second.id).status == "queued"
    finally:
        db.close()


def test_process_next_mutation_disabled_by_default(tmp_path):
    db = _db(tmp_path)
    try:
        queue = ContextMutationQueue(ContextDAGStore(db))
        job = queue.enqueue("s1", "rebuild_projection", {})

        assert process_next_mutation(queue, enabled=False, handlers={"rebuild_projection": lambda job: {"ok": True}}) is None
        assert queue.get(job.id).status == "queued"
    finally:
        db.close()


def test_default_handlers_reconcile_transcript_without_llm_or_rewrite(tmp_path):
    db = _db(tmp_path)
    try:
        store = ContextDAGStore(db)
        queue = ContextMutationQueue(store)
        queue.enqueue("s1", "reconcile_transcript", {"messages": [{"role": "user", "content": "from queue"}]})

        result = MutationWorker(queue, handlers=default_dag_mutation_handlers(store)).run_once()

        assert result.status == "succeeded"
        assert result.metadata["result"]["inserted"] == 1
        assert [m["content"] for m in db.get_messages("s1")] == ["from queue"]
    finally:
        db.close()


def test_unknown_job_type_fails_safely_without_transcript_rewrite(tmp_path):
    db = _db(tmp_path)
    try:
        original = db.append_message("s1", "user", "raw stays")
        queue = ContextMutationQueue(ContextDAGStore(db))
        queue.enqueue("s1", "unknown", {}, max_attempts=1)

        result = MutationWorker(queue, handlers={}).run_once()

        assert result.status == "dead"
        assert "No handler" in result.error
        rows = db.get_messages("s1")
        assert [(r["id"], r["content"]) for r in rows] == [(original, "raw stays")]
    finally:
        db.close()


def test_dag_engine_queues_reconcile_on_compress_only_when_enabled(tmp_path):
    db = _db(tmp_path)
    try:
        messages = [{"role": "user", "content": "queued raw"}]
        engine = DAGContextEngine(session_db=db, mutation_queue_enabled=True)
        engine.on_session_start("s1")

        result = engine.compress(messages)

        assert result.messages == messages
        assert db.get_messages("s1") == []
        jobs = ContextMutationQueue(ContextDAGStore(db)).list_jobs(session_id="s1")
        assert len(jobs) == 1
        assert jobs[0].operation == "reconcile_transcript"
        assert jobs[0].payload["messages"] == messages
        assert engine.get_status()["mutation_queue_enabled"] is True

        inline = DAGContextEngine(session_db=db, mutation_queue_enabled=False)
        inline.on_session_start("s2")
        inline.compress([{"role": "user", "content": "inline raw"}])
        assert [m["content"] for m in db.get_messages("s2")] == ["inline raw"]
    finally:
        db.close()
