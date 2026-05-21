from agent.swarm_honcho import build_swarm_honcho_summary, persist_swarm_honcho_summary
from agent.swarm_state import SwarmJob


def test_build_swarm_honcho_summary_redacts_secrets_and_child_scratchpads():
    job = SwarmJob.create("ship it", created_at="2026-01-01T00:00:00+00:00", metadata={"api_key": "secret"})
    task = job.add_task("draft", task_id="t1")
    task.result = {"summary": "drafted", "scratchpad": "private chain", "token": "secret"}
    job.transition("completed")

    summary = build_swarm_honcho_summary(job)

    assert "ship it" in summary["content"]
    assert "drafted" in summary["content"]
    assert "private chain" not in summary["content"]
    assert "secret" not in summary["content"]
    assert summary["metadata"]["job_id"] == job.job_id


def test_persist_swarm_honcho_summary_noops_unless_enabled():
    calls = []

    result = persist_swarm_honcho_summary(
        SwarmJob.create("noop", created_at="2026-01-01T00:00:00+00:00"),
        enabled=False,
        writer=lambda payload: calls.append(payload),
    )

    assert result["persisted"] is False
    assert result["reason"] == "disabled"
    assert calls == []


def test_persist_swarm_honcho_summary_uses_injected_writer_when_enabled():
    calls = []
    job = SwarmJob.create("remember", created_at="2026-01-01T00:00:00+00:00")

    result = persist_swarm_honcho_summary(job, enabled=True, writer=lambda payload: calls.append(payload))

    assert result["persisted"] is True
    assert calls[0]["metadata"]["job_id"] == job.job_id
