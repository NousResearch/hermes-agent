from pathlib import Path

import pytest

from agent.frontdesk_live import (
    handle_frontdesk_live_input,
    recover_durable_frontdesk_store,
)
from agent.frontdesk_store import (
    JOB_FAILED,
    JOB_QUEUED,
    JOB_REVIEWER,
    JOB_SUCCEEDED,
    JOB_WORKER,
    FrontdeskStore,
)
from agent.orchestration_runtime import get_orchestration_runtime
from agent.task_registry import (
    FRONTDESK_DONE_PRESENTED,
    FRONTDESK_ERROR,
    FRONTDESK_QUEUED,
    FRONTDESK_WORKER_DONE_PENDING_REVIEW,
)


class Owner:
    frontdesk_live_enabled = True


@pytest.fixture(autouse=True)
def _disable_durable_env(monkeypatch):
    monkeypatch.delenv("HERMES_FRONTDESK_DURABLE_STORE", raising=False)


def _artifact_paths(tmp_path: Path, task_id: str | None) -> dict[str, Path]:
    worker_dir = tmp_path / "workers" / (task_id or "untracked")
    worker_dir.mkdir(parents=True, exist_ok=True)
    return {
        "last_message": worker_dir / "last-message.txt",
        "summary": worker_dir / "summary.md",
        "stderr": worker_dir / "stderr.log",
    }


def _start_worker(owner: Owner, text: str):
    result = handle_frontdesk_live_input(
        owner,
        text,
        session_key="session-a",
        source_surface="gateway",
    )
    assert result is not None
    assert result.action == "worker_started"
    runtime = get_orchestration_runtime(owner)
    assert runtime is not None
    assert result.worker_id is not None
    assert runtime.worker_registry.wait(result.worker_id, timeout=2.0)
    return runtime, result


def test_durable_gate_off_preserves_existing_worker_start_behavior(tmp_path, monkeypatch):
    owner = Owner()
    owner._frontdesk_durable_store_path = str(tmp_path / "frontdesk.sqlite3")

    monkeypatch.setattr(
        "agent.frontdesk_live._worker_artifact_paths",
        lambda task_id: _artifact_paths(tmp_path, task_id),
    )
    monkeypatch.setattr(
        "agent.frontdesk_live._run_default_worker_subprocess",
        lambda *args, **kwargs: "worker done",
    )

    runtime, result = _start_worker(owner, "워커 레인에 배당해서 이 회귀를 조사해줘")

    assert result.task_id is not None
    assert runtime.task_registry.get_task(result.task_id).result["summary"] == "worker done"
    assert not (tmp_path / "frontdesk.sqlite3").exists()


def test_durable_worker_start_records_task_and_claimed_job(tmp_path, monkeypatch):
    db_path = tmp_path / "frontdesk.sqlite3"
    owner = Owner()
    owner.frontdesk_durable_store_enabled = True
    owner._frontdesk_durable_store_path = str(db_path)

    monkeypatch.setattr(
        "agent.frontdesk_live._worker_artifact_paths",
        lambda task_id: _artifact_paths(tmp_path, task_id),
    )

    def fake_worker(goal, token, **kwargs):  # noqa: ARG001
        kwargs["on_process_start"](4321)
        kwargs["last_message_path"].write_text("last", encoding="utf-8")
        kwargs["summary_path"].write_text("worker done", encoding="utf-8")
        kwargs["stderr_path"].write_text("", encoding="utf-8")
        return "worker done"

    monkeypatch.setattr("agent.frontdesk_live._run_default_worker_subprocess", fake_worker)

    _runtime, result = _start_worker(owner, "워커 레인에 배당해서 이 회귀를 조사해줘")

    store = FrontdeskStore(db_path)
    try:
        tasks = store.list_tasks()
        assert len(tasks) == 1
        task = tasks[0]
        assert task.id == result.task_id
        assert task.session_key == "session-a"
        assert task.state == FRONTDESK_WORKER_DONE_PENDING_REVIEW
        assert task.origin["in_memory_task_id"] == result.task_id
        assert task.origin["in_memory_worker_id"] == result.worker_id
        assert task.origin["source_surface"] == "gateway"

        worker_jobs = store.list_jobs(task_id=task.id, kind=JOB_WORKER)
        assert len(worker_jobs) == 1
        worker = worker_jobs[0]
        assert worker.state == JOB_SUCCEEDED
        assert worker.attempt == 1
        assert worker.lease_owner
        assert worker.pid == "4321"
        assert worker.session_id == result.worker_id
        assert worker.result["summary"] == "worker done"

        reviewers = store.list_jobs(task_id=task.id, kind=JOB_REVIEWER)
        assert len(reviewers) == 1
        assert reviewers[0].state == JOB_QUEUED
        assert {artifact.artifact_type for artifact in store.list_artifacts(task_id=task.id)} == {
            "last_message",
            "summary",
            "stderr",
        }
    finally:
        store.close()


def test_durable_worker_failure_records_error_not_presented(tmp_path, monkeypatch):
    db_path = tmp_path / "frontdesk.sqlite3"
    owner = Owner()
    owner.frontdesk_durable_store_enabled = True
    owner._frontdesk_durable_store_path = str(db_path)

    monkeypatch.setattr(
        "agent.frontdesk_live._worker_artifact_paths",
        lambda task_id: _artifact_paths(tmp_path, task_id),
    )

    def failing_worker(goal, token, **kwargs):  # noqa: ARG001
        kwargs["on_process_start"](9876)
        raise RuntimeError("boom")

    monkeypatch.setattr("agent.frontdesk_live._run_default_worker_subprocess", failing_worker)

    _runtime, result = _start_worker(owner, "워커 레인에 배당해서 이 회귀를 조사해줘")

    store = FrontdeskStore(db_path)
    try:
        task = store.get_task(result.task_id)
        assert task is not None
        assert task.state == FRONTDESK_ERROR
        assert task.state != FRONTDESK_DONE_PRESENTED
        worker = store.list_jobs(task_id=task.id, kind=JOB_WORKER)[0]
        assert worker.state == JOB_FAILED
        assert worker.result["error"] == "boom"
        assert store.list_jobs(task_id=task.id, kind=JOB_REVIEWER) == []
    finally:
        store.close()


def test_recover_durable_frontdesk_store_requeues_expired_worker_job(tmp_path):
    db_path = tmp_path / "frontdesk.sqlite3"
    store = FrontdeskStore(db_path)
    try:
        task, _worker = store.create_task_with_worker_job("recover stale worker")
        claimed = store.claim_job(
            kind=JOB_WORKER,
            lease_owner="worker-a",
            lease_seconds=5,
            now=100.0,
            pid=123,
            session_id="session-worker",
        )
        assert claimed is not None
    finally:
        store.close()

    recovered = recover_durable_frontdesk_store(path=db_path, now=106.0)

    assert recovered["path"] == str(db_path)
    assert [job["id"] for job in recovered["recovered_jobs"]] == [claimed.id]
    assert recovered["recovered_jobs"][0]["state"] == JOB_QUEUED
    assert recovered["recovered_jobs"][0]["lease_owner"] is None
    assert recovered["tasks"][0]["state"] == FRONTDESK_QUEUED

    store = FrontdeskStore(db_path)
    try:
        with pytest.raises(ValueError, match="running"):
            store.complete_worker_job(
                claimed.id,
                success=True,
                lease_owner="worker-a",
                attempt=claimed.attempt,
            )
    finally:
        store.close()


def test_durable_bridge_init_failure_preserves_in_memory_worker_result(tmp_path, monkeypatch):
    db_path = tmp_path / "frontdesk.sqlite3"
    owner = Owner()
    owner.frontdesk_durable_store_enabled = True
    owner._frontdesk_durable_store_path = str(db_path)

    monkeypatch.setattr(
        "agent.frontdesk_live._worker_artifact_paths",
        lambda task_id: _artifact_paths(tmp_path, task_id),
    )
    monkeypatch.setattr(
        "agent.frontdesk_live._run_default_worker_subprocess",
        lambda *args, **kwargs: "worker done despite durable init failure",
    )

    def broken_create(self, *args, **kwargs):  # noqa: ARG001
        raise RuntimeError("durable create failed")

    monkeypatch.setattr(FrontdeskStore, "create_task_with_worker_job", broken_create)

    runtime, result = _start_worker(owner, "워커 레인에 배당해서 이 회귀를 조사해줘")

    assert result.task_id is not None
    task = runtime.task_registry.get_task(result.task_id)
    assert task is not None
    assert task.result is not None
    assert task.result["summary"] == "worker done despite durable init failure"


def test_durable_store_open_failure_preserves_in_memory_worker_result(tmp_path, monkeypatch):
    db_path = tmp_path / "frontdesk.sqlite3"
    owner = Owner()
    owner.frontdesk_durable_store_enabled = True
    owner._frontdesk_durable_store_path = str(db_path)

    monkeypatch.setattr(
        "agent.frontdesk_live._worker_artifact_paths",
        lambda task_id: _artifact_paths(tmp_path, task_id),
    )
    monkeypatch.setattr(
        "agent.frontdesk_live._run_default_worker_subprocess",
        lambda *args, **kwargs: "worker done despite durable open failure",
    )

    class BrokenFrontdeskStore:
        def __init__(self, *args, **kwargs):  # noqa: D107, ARG002
            raise RuntimeError("durable open failed")

    monkeypatch.setattr("agent.frontdesk_store.FrontdeskStore", BrokenFrontdeskStore)

    runtime, result = _start_worker(owner, "워커 레인에 배당해서 이 회귀를 조사해줘")

    assert result.task_id is not None
    task = runtime.task_registry.get_task(result.task_id)
    assert task is not None
    assert task.result is not None
    assert task.result["summary"] == "worker done despite durable open failure"


def test_durable_complete_failure_does_not_break_success_result(tmp_path, monkeypatch):
    db_path = tmp_path / "frontdesk.sqlite3"
    owner = Owner()
    owner.frontdesk_durable_store_enabled = True
    owner._frontdesk_durable_store_path = str(db_path)

    monkeypatch.setattr(
        "agent.frontdesk_live._worker_artifact_paths",
        lambda task_id: _artifact_paths(tmp_path, task_id),
    )
    monkeypatch.setattr(
        "agent.frontdesk_live._run_default_worker_subprocess",
        lambda *args, **kwargs: "worker done even if durable complete fails",
    )

    def broken_complete(self, *args, **kwargs):  # noqa: ARG001
        raise RuntimeError("durable complete failed")

    monkeypatch.setattr("agent.frontdesk_live._DurableWorkerBridge.complete", broken_complete)

    runtime, result = _start_worker(owner, "워커 레인에 배당해서 이 회귀를 조사해줘")

    assert result.task_id is not None
    task = runtime.task_registry.get_task(result.task_id)
    assert task is not None
    assert task.result is not None
    assert task.result["summary"] == "worker done even if durable complete fails"


def test_durable_complete_failure_preserves_original_worker_error(tmp_path, monkeypatch):
    db_path = tmp_path / "frontdesk.sqlite3"
    owner = Owner()
    owner.frontdesk_durable_store_enabled = True
    owner._frontdesk_durable_store_path = str(db_path)

    monkeypatch.setattr(
        "agent.frontdesk_live._worker_artifact_paths",
        lambda task_id: _artifact_paths(tmp_path, task_id),
    )

    def failing_worker(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("original worker failure")

    monkeypatch.setattr("agent.frontdesk_live._run_default_worker_subprocess", failing_worker)

    def broken_complete(self, *args, **kwargs):  # noqa: ARG001
        raise RuntimeError("durable complete failed")

    monkeypatch.setattr("agent.frontdesk_live._DurableWorkerBridge.complete", broken_complete)

    runtime, result = _start_worker(owner, "워커 레인에 배당해서 이 회귀를 조사해줘")

    assert result.task_id is not None
    task = runtime.task_registry.get_task(result.task_id)
    assert task is not None
    assert task.result is not None
    assert task.result["status"] == "failed"
    assert task.result["error"] == "original worker failure"