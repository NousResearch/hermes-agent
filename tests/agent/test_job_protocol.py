import json

from agent.background_jobs import BackgroundTaskRequest, CronTaskRequest, DelegationTaskRequest
from agent.job_protocol import (
    PROTOCOL_VERSION,
    build_background_job_envelope,
    build_completion_envelope,
    build_cron_job_envelope,
    build_delegation_job_envelope,
)


def test_background_job_envelope_includes_platform_callback():
    envelope = build_background_job_envelope(
        BackgroundTaskRequest(
            task_id="bg_1",
            prompt="map the repo",
            origin="gateway",
            platform="telegram",
            session_id="bg_1",
            user_id="123",
            user_name="victor",
            chat_id="456",
            thread_id="789",
        )
    )

    assert envelope["version"] == PROTOCOL_VERSION
    assert envelope["kind"] == "background"
    assert envelope["task_id"] == "bg_1"
    assert envelope["callback"]["type"] == "platform"
    assert envelope["callback"]["target"] == {
        "platform": "telegram",
        "chat_id": "456",
        "thread_id": "789",
    }


def test_delegation_job_envelope_includes_session_callback():
    envelope = build_delegation_job_envelope(
        DelegationTaskRequest(
            task_id="delegate_1",
            goal="research idea",
            context="ctx",
            toolsets=["web"],
            model="gpt-test",
            max_iterations=12,
            parent_session_id="session_123",
            parent_platform="telegram",
            task_index=1,
            task_count=3,
        )
    )

    assert envelope["kind"] == "delegation"
    assert envelope["callback"] == {
        "type": "session",
        "session_id": "session_123",
        "platform": "telegram",
    }
    assert envelope["payload"]["goal"] == "research idea"
    assert envelope["payload"]["task_index"] == 1


def test_cron_job_envelope_prefers_deliver_target():
    envelope = build_cron_job_envelope(
        CronTaskRequest(
            task_id="cron_1",
            job_id="job_1",
            job_name="daily digest",
            prompt="summarize",
            schedule_display="0 9 * * *",
            deliver="telegram:-1001:55",
            origin={"platform": "telegram", "chat_id": "123", "thread_id": "44"},
            skills=["blogwatcher"],
        )
    )

    assert envelope["kind"] == "cron"
    assert envelope["callback"]["type"] == "platform"
    assert envelope["callback"]["target"] == {
        "platform": "telegram",
        "chat_id": "-1001",
        "thread_id": "55",
    }
    assert envelope["payload"]["job_id"] == "job_1"


def test_completion_envelope_keeps_callback_and_artifacts():
    callback = {"type": "platform", "target": {"platform": "telegram", "chat_id": "456", "thread_id": None}}
    envelope = build_completion_envelope(
        kind="background",
        task_id="bg_1",
        status="succeeded",
        callback=callback,
        summary="done",
        final_output="all good",
        artifacts=[{"kind": "file", "path": "/tmp/out.md"}],
        metadata={"attempt": 2},
    )

    assert envelope == {
        "version": PROTOCOL_VERSION,
        "kind": "background",
        "task_id": "bg_1",
        "status": "succeeded",
        "summary": "done",
        "final_output": "all good",
        "artifacts": [{"kind": "file", "path": "/tmp/out.md"}],
        "metadata": {"attempt": 2},
        "callback": callback,
    }


def test_completion_envelope_is_json_serializable():
    envelope = build_completion_envelope(
        kind="cron",
        task_id="cron_1",
        status="failed",
        callback={"type": "none"},
        error="boom",
    )
    json.dumps(envelope)
