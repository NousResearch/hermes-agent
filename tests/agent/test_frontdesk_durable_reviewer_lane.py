import pytest

from agent.frontdesk_live import run_one_durable_frontdesk_review
from agent.frontdesk_store import (
    JOB_QUEUED,
    JOB_REVIEWER,
    JOB_SUCCEEDED,
    JOB_WORKER,
    REVIEW_UNSAFE,
    FrontdeskStore,
)
from agent.task_registry import (
    FRONTDESK_ERROR,
    FRONTDESK_REVIEW_FAILED_NEEDS_ITERATION,
    FRONTDESK_REVIEW_PASSED,
    FRONTDESK_WORKER_DONE_PENDING_REVIEW,
    REVIEW_FAILED,
    REVIEW_PASSED,
)


def _queue_reviewer(
    db_path,
    *,
    worker_result: dict | None = None,
    artifacts: list[dict] | None = None,
):
    store = FrontdeskStore(db_path)
    try:
        task, _worker = store.create_task_with_worker_job("review durable worker output")
        worker_claim = store.claim_job(
            kind=JOB_WORKER,
            lease_owner="worker-a",
            lease_seconds=30,
        )
        assert worker_claim is not None
        completed, reviewer = store.complete_worker_job(
            worker_claim.id,
            success=True,
            lease_owner="worker-a",
            attempt=worker_claim.attempt,
            exit_status=0,
            result=worker_result or {"summary": "worker completed"},
            artifacts=artifacts
            or [{"path": "summary.md", "type": "summary", "size": 17}],
        )
        assert completed.state == JOB_SUCCEEDED
        assert reviewer is not None
        assert reviewer.state == JOB_QUEUED
        assert store.get_task(task.id).state == FRONTDESK_WORKER_DONE_PENDING_REVIEW
        return task.id, completed.id, reviewer.id
    finally:
        store.close()


def test_reviewer_pass_transitions_task_to_review_passed_not_presented(tmp_path):
    db_path = tmp_path / "frontdesk.sqlite3"
    task_id, worker_job_id, reviewer_job_id = _queue_reviewer(db_path)

    result = run_one_durable_frontdesk_review(path=db_path, lease_owner="reviewer-a")

    assert result is not None
    assert result["worker_job"]["id"] == worker_job_id
    assert result["reviewer_job"]["id"] == reviewer_job_id
    assert result["reviewer_job"]["state"] == JOB_SUCCEEDED
    assert result["review_result"]["review_status"] == REVIEW_PASSED
    assert result["review_result"]["artifact_pointers"] == [
        {
            "id": result["review_result"]["artifact_pointers"][0]["id"],
            "job_id": worker_job_id,
            "path": "summary.md",
            "type": "summary",
            "import_status": "pending",
            "size": 17,
        }
    ]

    store = FrontdeskStore(db_path)
    try:
        task = store.get_task(task_id)
        assert task is not None
        assert task.state == FRONTDESK_REVIEW_PASSED
        assert task.state != "done_presented"
    finally:
        store.close()


def test_reviewer_reject_or_unsafe_is_non_presentable(tmp_path):
    db_path = tmp_path / "frontdesk.sqlite3"
    task_id, _worker_job_id, _reviewer_job_id = _queue_reviewer(db_path)

    result = run_one_durable_frontdesk_review(
        path=db_path,
        lease_owner="reviewer-a",
        review_adapter=lambda context: {
            "review_status": REVIEW_UNSAFE,
            "summary": f"unsafe output for {context['task']['id']}",
        },
    )

    assert result is not None
    assert result["review_result"]["review_status"] == REVIEW_UNSAFE
    store = FrontdeskStore(db_path)
    try:
        task = store.get_task(task_id)
        assert task is not None
        assert task.state == FRONTDESK_ERROR
        with pytest.raises(ValueError, match="before review passes"):
            store.mark_done_presented(task_id)
    finally:
        store.close()


def test_reviewer_failed_verdict_is_recorded_non_presentable(tmp_path):
    db_path = tmp_path / "frontdesk.sqlite3"
    task_id, _worker_job_id, _reviewer_job_id = _queue_reviewer(db_path)

    result = run_one_durable_frontdesk_review(
        path=db_path,
        lease_owner="reviewer-a",
        review_adapter=lambda context: {
            "review_status": REVIEW_FAILED,
            "summary": f"failed review for {context['task']['id']}",
        },
    )

    assert result is not None
    assert result["review_result"]["review_status"] == REVIEW_FAILED
    store = FrontdeskStore(db_path)
    try:
        task = store.get_task(task_id)
        assert task is not None
        assert task.state == FRONTDESK_REVIEW_FAILED_NEEDS_ITERATION
        with pytest.raises(ValueError, match="before review passes"):
            store.mark_done_presented(task_id)
    finally:
        store.close()


def test_reviewer_adapter_exception_records_failed_review(tmp_path):
    db_path = tmp_path / "frontdesk.sqlite3"
    task_id, _worker_job_id, reviewer_job_id = _queue_reviewer(db_path)

    def broken_adapter(context):  # noqa: ARG001
        raise RuntimeError("review adapter exploded")

    result = run_one_durable_frontdesk_review(
        path=db_path,
        lease_owner="reviewer-a",
        review_adapter=broken_adapter,
    )

    assert result is not None
    assert result["reviewer_job"]["id"] == reviewer_job_id
    assert result["reviewer_job"]["state"] == JOB_SUCCEEDED
    assert result["review_result"]["review_status"] == REVIEW_FAILED
    assert result["review_result"]["error"] == "review adapter exploded"
    store = FrontdeskStore(db_path)
    try:
        task = store.get_task(task_id)
        assert task is not None
        assert task.state == FRONTDESK_REVIEW_FAILED_NEEDS_ITERATION
        with pytest.raises(ValueError, match="before review passes"):
            store.mark_done_presented(task_id)
    finally:
        store.close()


def test_reviewer_claim_token_blocks_stale_completion_after_recovery(tmp_path):
    db_path = tmp_path / "frontdesk.sqlite3"
    task_id, _worker_job_id, reviewer_job_id = _queue_reviewer(db_path)
    store = FrontdeskStore(db_path)
    try:
        stale_claim = store.claim_job(
            kind=JOB_REVIEWER,
            lease_owner="reviewer-stale",
            lease_seconds=5,
            now=100.0,
        )
        assert stale_claim is not None
        assert stale_claim.id == reviewer_job_id
        recovered = store.recover_expired_leases(now=106.0)
        assert [job.id for job in recovered] == [reviewer_job_id]
        with pytest.raises(ValueError, match="running"):
            store.complete_reviewer_job(
                reviewer_job_id,
                review_status=REVIEW_PASSED,
                lease_owner="reviewer-stale",
                attempt=stale_claim.attempt,
                result={"summary": "stale pass"},
            )
        recovered_task = store.get_task(task_id)
        assert recovered_task is not None
        assert recovered_task.state == FRONTDESK_WORKER_DONE_PENDING_REVIEW
    finally:
        store.close()

    result = run_one_durable_frontdesk_review(path=db_path, lease_owner="reviewer-current")

    assert result is not None
    assert result["reviewer_job"]["attempt"] == stale_claim.attempt + 1
    assert result["review_result"]["review_status"] == REVIEW_PASSED
    store = FrontdeskStore(db_path)
    try:
        with pytest.raises(ValueError, match="attempt"):
            store.complete_reviewer_job(
                reviewer_job_id,
                review_status=REVIEW_PASSED,
                lease_owner="reviewer-current",
                attempt=stale_claim.attempt,
                result={"summary": "old attempt duplicate"},
            )
    finally:
        store.close()


def test_run_one_review_returns_none_when_no_queued_reviewer_job(tmp_path):
    db_path = tmp_path / "frontdesk.sqlite3"
    store = FrontdeskStore(db_path)
    try:
        store.create_task_with_worker_job("worker has not completed")
    finally:
        store.close()

    assert run_one_durable_frontdesk_review(path=db_path, lease_owner="reviewer-a") is None
