import json
from pathlib import Path
from unittest.mock import patch

from agent.job_protocol import build_background_job_envelope, build_completion_envelope


def _background_envelope() -> dict:
    from agent.background_jobs import BackgroundTaskRequest

    return build_background_job_envelope(
        BackgroundTaskRequest(
            task_id="bg_1",
            prompt="map the repo",
            origin="gateway",
            platform="telegram",
            session_id="bg_1",
            user_id="123",
            chat_id="456",
            thread_id="789",
        )
    )


def test_enqueue_job_writes_queue_file(tmp_path):
    from agent.minions_reference import enqueue_job

    ack = enqueue_job(_background_envelope(), spool_dir=tmp_path)

    queued_file = tmp_path / "queue" / "background" / "bg_1.json"
    assert queued_file.exists()
    assert json.loads(queued_file.read_text(encoding="utf-8"))["task_id"] == "bg_1"
    assert ack["task_id"] == "bg_1"
    assert ack["backend"] == "reference-minions"
    assert ack["queue"] == "background"


def test_process_next_job_runs_default_worker_and_delivers(tmp_path):
    from agent.minions_reference import enqueue_job, process_next_job

    enqueue_job(_background_envelope(), spool_dir=tmp_path)

    with patch("agent.minions_reference.deliver_completion", return_value=None) as deliver_mock:
        completion = process_next_job(spool_dir=tmp_path)

    assert completion is not None
    assert completion["task_id"] == "bg_1"
    assert completion["status"] == "succeeded"
    assert "map the repo" in (completion.get("final_output") or "")
    completed_file = tmp_path / "completed" / "background" / "bg_1.json"
    assert completed_file.exists()
    deliver_mock.assert_called_once_with(completion, adapters=None, loop=None)


def test_process_next_job_uses_custom_runner(tmp_path):
    from agent.minions_reference import enqueue_job, process_next_job

    enqueue_job(_background_envelope(), spool_dir=tmp_path)

    def runner(envelope: dict) -> dict:
        return build_completion_envelope(
            kind=envelope["kind"],
            task_id=envelope["task_id"],
            status="succeeded",
            callback=envelope["callback"],
            summary="custom done",
            final_output="custom output",
        )

    with patch("agent.minions_reference.deliver_completion", return_value=None):
        completion = process_next_job(spool_dir=tmp_path, runner=runner)

    assert completion["summary"] == "custom done"
    assert completion["final_output"] == "custom output"


def test_process_next_job_returns_none_when_queue_empty(tmp_path):
    from agent.minions_reference import process_next_job

    assert process_next_job(spool_dir=tmp_path) is None
