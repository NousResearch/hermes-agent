import json

import pytest

from agent.frontdesk_live import run_one_durable_frontdesk_review
from agent.frontdesk_review_surface import (
    get_durable_frontdesk_task,
    list_durable_frontdesk_tasks,
    present_durable_frontdesk_task,
    record_durable_frontdesk_discard,
    record_durable_frontdesk_import,
)
from agent.frontdesk_store import (
    ARTIFACT_DISCARDED,
    ARTIFACT_IMPORT_REQUESTED,
    JOB_REVIEWER,
    JOB_WORKER,
    REVIEW_UNSAFE,
    FrontdeskStore,
)
from agent.task_registry import (
    FRONTDESK_DONE_PRESENTED,
    FRONTDESK_REVIEW_FAILED_NEEDS_ITERATION,
    FRONTDESK_REVIEW_PASSED,
    FRONTDESK_WORKER_DONE_PENDING_REVIEW,
    REVIEW_FAILED,
    REVIEW_PASSED,
)


def _queue_worker_success(db_path, *, artifact_path: str = "summary.md") -> str:
    store = FrontdeskStore(db_path)
    try:
        task, _worker = store.create_task_with_worker_job(
            "produce reviewed artifact",
            session_key="session-a",
            origin={"platform": "test"},
        )
        worker = store.claim_job(kind=JOB_WORKER, lease_owner="worker-a", lease_seconds=30)
        assert worker is not None
        completed, reviewer = store.complete_worker_job(
            worker.id,
            success=True,
            lease_owner="worker-a",
            attempt=worker.attempt,
            exit_status=0,
            result={"summary": "worker completed"},
            artifacts=[{"path": artifact_path, "type": "summary", "size": 17}],
        )
        assert completed.task_id == task.id
        assert reviewer is not None
        return task.id
    finally:
        store.close()


def _review_pass(db_path, *, artifact_path: str = "summary.md") -> str:
    task_id = _queue_worker_success(db_path, artifact_path=artifact_path)
    result = run_one_durable_frontdesk_review(path=db_path, lease_owner="reviewer-a")
    assert result is not None
    assert result["review_result"]["review_status"] == REVIEW_PASSED
    return task_id


def _review_failed(db_path, *, status: str = REVIEW_FAILED) -> str:
    task_id = _queue_worker_success(db_path)
    result = run_one_durable_frontdesk_review(
        path=db_path,
        lease_owner="reviewer-a",
        review_adapter=lambda _context: {
            "review_status": status,
            "summary": f"review verdict: {status}",
        },
    )
    assert result is not None
    return task_id


def _event_count(db_path, task_id: str, event_type: str) -> int:
    store = FrontdeskStore(db_path)
    try:
        return sum(1 for event in store.list_events(task_id=task_id) if event.event_type == event_type)
    finally:
        store.close()


def test_list_durable_frontdesk_tasks_includes_review_state_and_artifacts(tmp_path):
    db_path = tmp_path / "frontdesk.sqlite3"
    task_id = _review_pass(db_path)

    payload = list_durable_frontdesk_tasks(path=db_path, session_key="session-a")

    json.dumps(payload, allow_nan=False)
    assert payload["path"] == str(db_path)
    assert len(payload["tasks"]) == 1
    item = payload["tasks"][0]
    assert item["task"]["id"] == task_id
    assert item["state"] == FRONTDESK_REVIEW_PASSED
    assert item["latest_jobs"]["worker"]["kind"] == JOB_WORKER
    assert item["latest_jobs"]["reviewer"]["kind"] == JOB_REVIEWER
    assert item["review_result"]["review_status"] == REVIEW_PASSED
    assert item["artifact_pointers"] == [
        {
            "id": item["artifact_pointers"][0]["id"],
            "job_id": item["artifact_pointers"][0]["job_id"],
            "path": "summary.md",
            "type": "summary",
            "import_status": "pending",
            "size": 17,
        }
    ]
    assert item["presentable"] is True
    assert item["importable"] is True


def test_present_review_passed_task_marks_done_presented(tmp_path):
    db_path = tmp_path / "frontdesk.sqlite3"
    task_id = _review_pass(db_path)

    payload = present_durable_frontdesk_task(task_id, path=db_path)

    json.dumps(payload, allow_nan=False)
    assert payload["presented"] is True
    assert payload["already_presented"] is False
    assert payload["task"]["state"] == FRONTDESK_DONE_PRESENTED
    assert payload["events"][-1]["event_type"] == "task_done_presented"

    second = present_durable_frontdesk_task(task_id, path=db_path)
    assert second["presented"] is True
    assert second["already_presented"] is True
    assert _event_count(db_path, task_id, "task_done_presented") == 1


def test_store_presentation_is_idempotent_without_pre_read(tmp_path):
    db_path = tmp_path / "frontdesk.sqlite3"
    task_id = _review_pass(db_path)
    store = FrontdeskStore(db_path)
    try:
        first, first_already = store.mark_done_presented_with_status(task_id)
        second, second_already = store.mark_done_presented_with_status(task_id)
    finally:
        store.close()

    assert first.state == FRONTDESK_DONE_PRESENTED
    assert second.state == FRONTDESK_DONE_PRESENTED
    assert first_already is False
    assert second_already is True
    assert _event_count(db_path, task_id, "task_done_presented") == 1


def test_present_before_review_pass_fails(tmp_path):
    pending_db = tmp_path / "pending.sqlite3"
    pending_task_id = _queue_worker_success(pending_db)
    assert get_durable_frontdesk_task(pending_task_id, path=pending_db)["state"] == (
        FRONTDESK_WORKER_DONE_PENDING_REVIEW
    )
    with pytest.raises(ValueError, match="before review passes"):
        present_durable_frontdesk_task(pending_task_id, path=pending_db)
    with pytest.raises(ValueError, match="before review completes"):
        record_durable_frontdesk_discard(pending_task_id, path=pending_db)

    failed_db = tmp_path / "failed.sqlite3"
    failed_task_id = _review_failed(failed_db)
    assert get_durable_frontdesk_task(failed_task_id, path=failed_db)["state"] == (
        FRONTDESK_REVIEW_FAILED_NEEDS_ITERATION
    )
    with pytest.raises(ValueError, match="before review passes"):
        present_durable_frontdesk_task(failed_task_id, path=failed_db)

    unsafe_db = tmp_path / "unsafe.sqlite3"
    unsafe_task_id = _review_failed(unsafe_db, status=REVIEW_UNSAFE)
    with pytest.raises(ValueError, match="before review passes"):
        present_durable_frontdesk_task(unsafe_task_id, path=unsafe_db)
    with pytest.raises(ValueError, match="before review passes"):
        record_durable_frontdesk_import(unsafe_task_id, path=unsafe_db)


def test_import_decision_is_idempotent_and_non_destructive(tmp_path):
    db_path = tmp_path / "frontdesk.sqlite3"
    artifact_file = tmp_path / "artifact; touch should-not-run.md"
    artifact_file.write_text("reviewed content", encoding="utf-8")
    task_id = _review_pass(db_path, artifact_path=str(artifact_file))

    payload = record_durable_frontdesk_import(
        task_id,
        path=db_path,
        metadata={"operator": "test"},
    )
    first_event_count = _event_count(db_path, task_id, "task_import_decision_recorded")
    second = record_durable_frontdesk_import(task_id, path=db_path)

    assert first_event_count == 1
    assert _event_count(db_path, task_id, "task_import_decision_recorded") == 1
    assert payload["import_decision"]["status"] == ARTIFACT_IMPORT_REQUESTED
    assert payload["import_decision"]["applied"] is False
    assert second["artifact_pointers"][0]["import_status"] == ARTIFACT_IMPORT_REQUESTED
    assert artifact_file.read_text(encoding="utf-8") == "reviewed content"
    assert not (tmp_path / "should-not-run.md").exists()


def test_discard_decision_is_idempotent_and_non_destructive(tmp_path):
    db_path = tmp_path / "frontdesk.sqlite3"
    artifact_file = tmp_path / "discard-me.md"
    artifact_file.write_text("leave me alone", encoding="utf-8")
    task_id = _review_pass(db_path, artifact_path=str(artifact_file))

    payload = record_durable_frontdesk_discard(
        task_id,
        path=db_path,
        metadata={"operator": "test"},
    )
    first_event_count = _event_count(db_path, task_id, "task_discard_decision_recorded")
    second = record_durable_frontdesk_discard(task_id, path=db_path)

    assert first_event_count == 1
    assert _event_count(db_path, task_id, "task_discard_decision_recorded") == 1
    assert payload["discard_decision"]["status"] == ARTIFACT_DISCARDED
    assert payload["discard_decision"]["deleted"] is False
    assert second["artifact_pointers"][0]["import_status"] == ARTIFACT_DISCARDED
    assert artifact_file.read_text(encoding="utf-8") == "leave me alone"
