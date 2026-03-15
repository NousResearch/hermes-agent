import asyncio
from pathlib import Path

import pytest

from gateway.collaboration import (
    CollaborationJob,
    CollaborationStore,
    InternalGatewayEvent,
    create_collaboration_job,
    resolve_target_alias,
)
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def test_create_collaboration_job_records_correlation_fields(tmp_path):
    store = CollaborationStore(tmp_path / "collaboration.db")
    job = create_collaboration_job(
        store=store,
        requester_session_key="agent:main:webhook:dm:caller",
        target_session_key="agent:main:webhook:dm:researcher",
        target_agent="hermes-2",
        task_text="Research OpenClaw's handoff model",
    )
    assert job.status == "pending"
    assert job.requester_session_key == "agent:main:webhook:dm:caller"
    assert job.target_session_key == "agent:main:webhook:dm:researcher"
    assert job.job_id


def test_collaboration_job_round_trips_through_sqlite(tmp_path):
    db_path = tmp_path / "collaboration.db"
    store = CollaborationStore(db_path)
    job = create_collaboration_job(
        store=store,
        requester_session_key="agent:main:webhook:dm:caller",
        target_session_key="agent:main:webhook:dm:researcher",
        target_agent="hermes-2",
        task_text="Summarize findings",
    )

    reloaded = CollaborationStore(db_path)
    restored = reloaded.get_job(job.job_id)
    assert restored is not None
    assert restored.task_text == "Summarize findings"
    assert restored.status == "pending"


def test_resolve_target_alias_uses_explicit_registry():
    config = GatewayConfig()
    config.collaboration = {
        "enabled": True,
        "targets": {
            "hermes-2": {
                "platform": "webhook",
                "chat_id": "hermes-2",
                "display_name": "Research",
            }
        },
    }

    resolved = resolve_target_alias(config, "hermes-2", requester_session_key="agent:main:webhook:dm:caller")
    assert resolved["chat_id"] == "hermes-2"


@pytest.mark.asyncio
async def test_runner_routes_collaboration_request_to_internal_queue(tmp_path):
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    await runner.route_collaboration_request(
        requester_session_key="agent:main:webhook:dm:caller",
        target_session_key="agent:main:webhook:dm:researcher",
        job_id="job-123",
        task_text="Summarize latest findings",
        lineage=[],
    )

    event = runner._internal_event_queue.get_nowait()
    assert event.kind == "collaboration_request"
    assert event.job_id == "job-123"
    assert event.session_key == "agent:main:webhook:dm:researcher"


@pytest.mark.asyncio
async def test_runner_internal_event_consumer_delivers_event(tmp_path):
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    delivered = []

    async def fake_deliver(event):
        delivered.append(event)

    runner._deliver_internal_event = fake_deliver
    event = InternalGatewayEvent(
        kind="collaboration_result",
        session_key="agent:main:webhook:dm:caller",
        job_id="job-123",
        payload={"result_text": "done"},
    )
    await runner._internal_event_queue.put(event)
    await runner._process_next_internal_event()

    assert len(delivered) == 1
    assert delivered[0].payload["result_text"] == "done"


@pytest.mark.asyncio
async def test_runner_resolves_wait_when_collaboration_result_arrives(tmp_path):
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    handle = runner.register_collaboration_wait("job-123", "agent:main:webhook:dm:caller")

    await runner._handle_collaboration_result_event(
        InternalGatewayEvent(
            kind="collaboration_result",
            session_key="agent:main:webhook:dm:caller",
            job_id="job-123",
            payload={"result_text": '{"status":"completed","result":"Artifact ready"}'},
        )
    )

    assert handle["future"].result(timeout=0.1)["result_text"]


@pytest.mark.asyncio
async def test_runner_executes_collaboration_request_end_to_end(tmp_path):
    config = GatewayConfig(sessions_dir=tmp_path / "sessions")
    config.collaboration = {
        "enabled": True,
        "targets": {
            "hermes-2": {"platform": "webhook", "chat_id": "hermes-2", "display_name": "Research"}
        },
    }
    runner = GatewayRunner(config)
    runner._event_loop = asyncio.get_running_loop()
    source = SessionSource(
        platform=__import__("gateway.config", fromlist=["Platform"]).Platform.WEBHOOK,
        chat_id="hermes-2",
        chat_type="dm",
        user_id="hermes-2",
        user_name="Research",
    )
    target_session_key = build_session_key(source)
    handle = runner.register_collaboration_wait("job-123", "agent:main:webhook:dm:caller")

    async def fake_run_agent(message, context_prompt, history, source, session_id, session_key=None):
        assert "Hermes collaboration request" in message
        return {"final_response": "Research result"}

    runner._run_agent = fake_run_agent
    await runner.route_collaboration_request(
        requester_session_key="agent:main:webhook:dm:caller",
        target_session_key=target_session_key,
        job_id="job-123",
        task_text="Research this",
        lineage=[],
        requester_agent="hermes-1",
        target_source=source.to_dict(),
    )

    payload = await asyncio.wait_for(asyncio.wrap_future(handle["future"]), timeout=1)
    assert "Research result" in payload["result_text"]
