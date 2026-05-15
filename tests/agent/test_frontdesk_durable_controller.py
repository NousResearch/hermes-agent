"""Tests for the durable frontdesk controller store foundation."""

import stat

import pytest

from agent.frontdesk_store import (
    JOB_CANCELLED,
    JOB_QUEUED,
    JOB_REVIEWER,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    JOB_WORKER,
    REVIEW_REJECTED,
    FrontdeskStore,
)
from agent.task_registry import (
    FRONTDESK_CANCELLED,
    FRONTDESK_DONE_PRESENTED,
    FRONTDESK_ERROR,
    FRONTDESK_QUEUED,
    FRONTDESK_REVIEW_PASSED,
    FRONTDESK_RUNNING_WORKER,
    FRONTDESK_WORKER_DONE_PENDING_REVIEW,
    REVIEW_PASSED,
)


def test_task_job_event_persist_across_restart(tmp_path):
    path = tmp_path / "frontdesk.sqlite3"
    store = FrontdeskStore(path)
    task, worker = store.create_task_with_worker_job(
        "write a durable controller",
        session_key="session-a",
        origin={"platform": "gateway", "chat_id": "chat-1"},
    )
    claimed = store.claim_job(kind=JOB_WORKER, lease_owner="worker-a", lease_seconds=30)
    assert claimed is not None
    store.heartbeat_job(
        claimed.id,
        lease_owner="worker-a",
        attempt=claimed.attempt,
        extend_seconds=30,
    )
    store.close()

    reloaded = FrontdeskStore(path)
    try:
        restored_task = reloaded.get_task(task.id)
        restored_worker = reloaded.get_job(worker.id)
        assert restored_task is not None
        assert restored_task.user_goal == "write a durable controller"
        assert restored_task.session_key == "session-a"
        assert restored_task.origin["platform"] == "gateway"
        assert restored_task.state == FRONTDESK_RUNNING_WORKER
        assert restored_worker is not None
        assert restored_worker.state == JOB_RUNNING
        assert restored_worker.lease_owner == "worker-a"
        events = reloaded.list_events(task_id=task.id)
        assert [event.event_type for event in events][:2] == [
            "task_created",
            "worker_job_enqueued",
        ]
        assert "job_heartbeat" in {event.event_type for event in events}
    finally:
        reloaded.close()


def test_enqueue_task_and_worker_job_atomic(tmp_path, monkeypatch):
    store = FrontdeskStore(tmp_path / "frontdesk.sqlite3")

    def fail_insert_job(**kwargs):  # noqa: ARG001
        raise RuntimeError("simulated enqueue failure")

    monkeypatch.setattr(store, "_insert_job", fail_insert_job)
    with pytest.raises(RuntimeError, match="simulated enqueue failure"):
        store.create_task_with_worker_job("must not half-write")

    assert store.list_tasks() == []
    assert store.list_jobs() == []
    store.close()


def test_worker_success_queues_reviewer_not_done(tmp_path):
    store = FrontdeskStore(tmp_path / "frontdesk.sqlite3")
    task, _worker = store.create_task_with_worker_job("implement phase one")
    claimed = store.claim_job(kind=JOB_WORKER, lease_owner="worker-a", lease_seconds=30)
    assert claimed is not None

    completed, reviewer = store.complete_worker_job(
        claimed.id,
        success=True,
        lease_owner="worker-a",
        attempt=claimed.attempt,
        exit_status=0,
        result={"summary": "worker finished"},
        artifacts=[{"path": "agent/frontdesk_store.py", "type": "source", "size": 10}],
    )

    assert completed.state == JOB_SUCCEEDED
    assert reviewer is not None
    assert reviewer.kind == JOB_REVIEWER
    assert reviewer.state == JOB_QUEUED
    restored_task = store.get_task(task.id)
    assert restored_task is not None
    assert restored_task.state == FRONTDESK_WORKER_DONE_PENDING_REVIEW
    assert restored_task.state != FRONTDESK_DONE_PRESENTED
    assert [artifact.path for artifact in store.list_artifacts(task_id=task.id)] == [
        "agent/frontdesk_store.py"
    ]
    store.close()


def test_duplicate_completion_is_idempotent(tmp_path):
    store = FrontdeskStore(tmp_path / "frontdesk.sqlite3")
    task, _worker = store.create_task_with_worker_job("complete once")
    claimed = store.claim_job(kind=JOB_WORKER, lease_owner="worker-a", lease_seconds=30)
    assert claimed is not None

    first_completed, first_reviewer = store.complete_worker_job(
        claimed.id,
        success=True,
        lease_owner="worker-a",
        attempt=claimed.attempt,
        result={"summary": "done"},
        artifacts=[{"path": "summary.md", "type": "summary"}],
    )
    second_completed, second_reviewer = store.complete_worker_job(
        claimed.id,
        success=True,
        lease_owner="worker-a",
        attempt=claimed.attempt,
        result={"summary": "done again"},
        artifacts=[{"path": "duplicate.md", "type": "summary"}],
    )

    assert first_completed.id == second_completed.id
    assert first_completed.state == second_completed.state == JOB_SUCCEEDED
    assert first_reviewer is not None
    assert second_reviewer is not None
    assert first_reviewer.id == second_reviewer.id
    assert store.get_task(task.id).state == FRONTDESK_WORKER_DONE_PENDING_REVIEW
    assert len(store.list_jobs(task_id=task.id, kind=JOB_REVIEWER)) == 1
    assert [artifact.path for artifact in store.list_artifacts(task_id=task.id)] == [
        "summary.md"
    ]
    store.close()


def test_expired_lease_transitions_to_recovering_or_requeue(tmp_path):
    store = FrontdeskStore(tmp_path / "frontdesk.sqlite3")
    task, _worker = store.create_task_with_worker_job("recover stale work")
    claimed = store.claim_job(
        kind=JOB_WORKER,
        lease_owner="worker-a",
        lease_seconds=10,
        now=100.0,
        pid=123,
        session_id="worker-session",
    )
    assert claimed is not None
    assert claimed.state == JOB_RUNNING

    recovered = store.recover_expired_leases(now=111.0)

    assert [job.id for job in recovered] == [claimed.id]
    recovered_job = store.get_job(claimed.id)
    assert recovered_job is not None
    assert recovered_job.state == JOB_QUEUED
    assert recovered_job.lease_owner is None
    assert recovered_job.lease_expires_at is None
    assert recovered_job.pid is None
    assert store.get_task(task.id).state == FRONTDESK_QUEUED

    reclaimed = store.claim_job(kind=JOB_WORKER, lease_owner="worker-b", lease_seconds=10)
    assert reclaimed is not None
    assert reclaimed.id == claimed.id
    assert reclaimed.attempt == 2
    store.close()


def test_reviewer_completion_allows_presentation_only_after_pass(tmp_path):
    store = FrontdeskStore(tmp_path / "frontdesk.sqlite3")
    task, _worker = store.create_task_with_worker_job("review gate")
    worker = store.claim_job(kind=JOB_WORKER, lease_owner="worker-a", lease_seconds=30)
    assert worker is not None
    _completed, reviewer = store.complete_worker_job(worker.id, success=True, lease_owner="worker-a", attempt=worker.attempt)
    assert reviewer is not None

    with pytest.raises(ValueError, match="before review passes"):
        store.mark_done_presented(task.id)

    reviewer_claim = store.claim_job(kind=JOB_REVIEWER, lease_owner="reviewer-a", lease_seconds=30)
    assert reviewer_claim is not None
    store.complete_reviewer_job(
        reviewer_claim.id,
        review_status=REVIEW_PASSED,
        lease_owner="reviewer-a",
        attempt=reviewer_claim.attempt,
        result={"summary": "safe to present"},
    )
    assert store.get_task(task.id).state == FRONTDESK_REVIEW_PASSED

    presented = store.mark_done_presented(task.id)
    assert presented.state == FRONTDESK_DONE_PRESENTED
    store.close()


def test_completion_requires_current_running_lease_owner(tmp_path):
    store = FrontdeskStore(tmp_path / "frontdesk.sqlite3")
    task, _worker = store.create_task_with_worker_job("stale worker")
    first_claim = store.claim_job(
        kind=JOB_WORKER,
        lease_owner="worker-a",
        lease_seconds=10,
        now=100.0,
    )
    assert first_claim is not None
    store.recover_expired_leases(now=111.0)

    with pytest.raises(ValueError, match="running"):
        store.complete_worker_job(first_claim.id, success=True, lease_owner="worker-a", attempt=first_claim.attempt)

    second_claim = store.claim_job(kind=JOB_WORKER, lease_owner="worker-a", lease_seconds=10)
    assert second_claim is not None
    with pytest.raises(ValueError, match="attempt"):
        store.heartbeat_job(
            second_claim.id,
            lease_owner="worker-a",
            attempt=first_claim.attempt,
            extend_seconds=10,
        )
    with pytest.raises(ValueError, match="attempt"):
        store.complete_worker_job(
            second_claim.id,
            success=True,
            lease_owner="worker-a",
            attempt=first_claim.attempt,
        )
    with pytest.raises(ValueError, match="lease owner"):
        store.complete_worker_job(
            second_claim.id,
            success=True,
            lease_owner="worker-b",
            attempt=second_claim.attempt,
        )

    completed, reviewer = store.complete_worker_job(
        second_claim.id,
        success=True,
        lease_owner="worker-a",
        attempt=second_claim.attempt,
    )
    assert completed.state == JOB_SUCCEEDED
    assert reviewer is not None
    with pytest.raises(ValueError, match="attempt"):
        store.complete_worker_job(
            second_claim.id,
            success=True,
            lease_owner="worker-a",
            attempt=first_claim.attempt,
        )
    assert store.get_task(task.id).state == FRONTDESK_WORKER_DONE_PENDING_REVIEW
    store.close()


def test_reviewer_reject_sets_error_state(tmp_path):
    store = FrontdeskStore(tmp_path / "frontdesk.sqlite3")
    task, _worker = store.create_task_with_worker_job("review reject")
    worker = store.claim_job(kind=JOB_WORKER, lease_owner="worker-a", lease_seconds=30)
    assert worker is not None
    _completed, reviewer = store.complete_worker_job(worker.id, success=True, lease_owner="worker-a", attempt=worker.attempt)
    assert reviewer is not None
    reviewer_claim = store.claim_job(kind=JOB_REVIEWER, lease_owner="reviewer-a", lease_seconds=30)
    assert reviewer_claim is not None

    store.complete_reviewer_job(
        reviewer_claim.id,
        review_status=REVIEW_REJECTED,
        lease_owner="reviewer-a",
        attempt=reviewer_claim.attempt,
        result={"summary": "reject unsafe output"},
    )

    assert store.get_task(task.id).state == FRONTDESK_ERROR
    with pytest.raises(ValueError, match="before review passes"):
        store.mark_done_presented(task.id)
    store.close()


def test_store_files_are_private(tmp_path):
    path = tmp_path / "frontdesk.sqlite3"
    store = FrontdeskStore(path)
    store.create_task_with_worker_job("private durable state")

    for suffix in ("", "-wal", "-shm"):
        file_path = path.with_name(path.name + suffix)
        if file_path.exists():
            mode = stat.S_IMODE(file_path.stat().st_mode)
            assert mode == 0o600
    store.close()


def test_cancel_request_is_idempotent(tmp_path):
    store = FrontdeskStore(tmp_path / "frontdesk.sqlite3")
    task, _worker = store.create_task_with_worker_job("cancel once")
    first = store.request_cancel(task.id, reason="user stop")
    first_events = store.list_events(task_id=task.id)
    second = store.request_cancel(task.id, reason="duplicate stop")
    second_events = store.list_events(task_id=task.id)

    assert first.state == second.state == FRONTDESK_CANCELLED
    assert first.cancel_requested_at == second.cancel_requested_at
    assert [event.event_type for event in first_events] == [event.event_type for event in second_events]
    assert [event.event_type for event in second_events].count("task_cancel_requested") == 1
    assert store.list_jobs(task_id=task.id)[0].state == JOB_CANCELLED
    store.close()


def test_running_cancel_allows_worker_cancel_completion_retry(tmp_path):
    store = FrontdeskStore(tmp_path / "frontdesk.sqlite3")
    task, _worker = store.create_task_with_worker_job("cancel running")
    claimed = store.claim_job(kind=JOB_WORKER, lease_owner="worker-a", lease_seconds=30)
    assert claimed is not None

    cancelled_task = store.request_cancel(task.id, reason="user stop")
    assert cancelled_task.state == FRONTDESK_CANCELLED

    completed, reviewer = store.complete_worker_job(
        claimed.id,
        success=False,
        cancelled=True,
        lease_owner="worker-a",
        attempt=claimed.attempt,
    )

    assert completed.state == JOB_CANCELLED
    assert reviewer is None
    assert store.get_task(task.id).state == FRONTDESK_CANCELLED
    store.close()
