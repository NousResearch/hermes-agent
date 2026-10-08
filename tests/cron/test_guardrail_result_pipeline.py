"""Runtime hard stops must stay failures through cron's archive, session and ledger."""
from types import SimpleNamespace

import pytest

import cron.scheduler as scheduler
from agent.tool_guardrails import ToolGuardrailDecision
from cron.executions import list_executions
from cron.jobs import create_job, get_job


PARTIAL = "Partial evidence: seven drives passed; one failed. Coverage is incomplete."


def _halt_result():
    decision = ToolGuardrailDecision(
        action="block", code="loop_subagent_cap", tool_name="delegate_task", count=50,
        message="Blocked delegate_task: this turn has already spawned 50 subagents (limit 50).",
    )
    return {
        "completed": True, "failed": False, "turn_exit_reason": "guardrail_halt",
        "final_response": PARTIAL, "guardrail": decision.to_metadata(), "api_calls": 16,
        "messages": [{"role": "assistant", "content": PARTIAL}],
    }


@pytest.fixture
def pipeline(monkeypatch):
    ended, audits, delivered = [], [], []
    db = SimpleNamespace(
        get_compression_tip=lambda _sid: None,
        set_session_title=lambda *_a: True,
        session_lifecycle_statuses=lambda ids: {sid: "complete" for sid in ids},
        end_session=lambda sid, reason: ended.append(reason),
        close=lambda: None,
    )
    monkeypatch.setattr(scheduler, "_open_cron_session_db", lambda _job: db)
    monkeypatch.setattr(
        scheduler, "_resolve_cron_agent_setup",
        lambda *_a: SimpleNamespace(blocked=None, model="test-model", fallback_notice=None),
    )
    monkeypatch.setattr(
        scheduler, "_construct_cron_agent",
        lambda *_a, **kw: SimpleNamespace(session_id=kw["session_id"]),
    )
    monkeypatch.setattr(scheduler, "_teardown_cron_agent", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        scheduler, "_FireAudit",
        lambda *_a: SimpleNamespace(write=lambda result, error: audits.append((result, error))),
    )

    def deliver(job, content, **kw):
        delivered.append((content, kw["for_failure"]))
        return job.get("test_delivery_error")

    monkeypatch.setattr(scheduler, "_deliver_result", deliver)
    return SimpleNamespace(ended=ended, audits=audits, delivered=delivered)


@pytest.mark.parametrize("delivery_error", [None, "send failed: 502"])
def test_halt_fails_through_archive_session_delivery_and_ledger(monkeypatch, pipeline, delivery_error):
    result = _halt_result()
    monkeypatch.setattr(scheduler, "_run_agent_with_watchdog", lambda *_a, **_kw: result)
    job = create_job("Verify all coverage", "1d", name="Coverage", deliver="telegram")
    job["test_delivery_error"] = delivery_error

    assert scheduler.run_one_job(job) is True

    stored = get_job(job["id"])
    assert stored["last_status"] == "error"
    assert "loop_subagent_cap" in stored["last_error"]
    assert "count=50" in stored["last_error"]
    assert stored["last_delivery_error"] == delivery_error
    execution = list_executions(job_id=job["id"])[0]
    assert execution["status"] == "failed"
    assert execution["delivery_outcome"] == ("failed" if delivery_error else "delivered")
    assert pipeline.ended == ["cron_guardrail_halt"]
    assert pipeline.audits[0][0]["guardrail"] == result["guardrail"]
    assert pipeline.audits[0][0]["api_calls"] == 16
    notice, for_failure = pipeline.delivered[0]
    assert for_failure is True
    assert "loop_subagent_cap" in notice and "count=50" in notice
    assert "50 retries" not in notice
    from cron.jobs import get_cron_output_dir

    archived = next((get_cron_output_dir() / job["id"]).glob("*.md")).read_text()
    assert PARTIAL in archived
    assert "loop_subagent_cap" in archived


@pytest.mark.parametrize("result", [
    {"turn_exit_reason": "guardrail_halt"},
    {"guardrail": {"action": "halt", "code": "same_tool_failure_halt", "count": 8}},
    {"guardrail": {"action": "block", "code": "loop_subagent_cap", "count": 50}},
])
def test_halt_cannot_be_hidden_by_success_flags_or_silence(result):
    from agent.turn_explainers import TurnExplainersMixin

    result = dict(completed=True, failed=False, final_response="[SILENT]", **result)
    with pytest.raises(RuntimeError, match="guardrail_halt"):
        scheduler._final_response_from_result(result, "job", "Coverage", TurnExplainersMixin)


@pytest.mark.parametrize("text,completed,reason", [
    ("Finished coverage.", True, "text_response(finish_reason=stop)"),
    ("[SILENT]", True, "text_response(finish_reason=stop)"),
    ("Iteration-limit handoff.", False, "max_iterations_reached(5)"),
])
def test_completed_silent_and_iteration_handoffs_keep_their_contract(text, completed, reason):
    from agent.turn_explainers import TurnExplainersMixin

    result = dict(completed=completed, failed=False, final_response=text, turn_exit_reason=reason)
    assert scheduler._final_response_from_result(result, "job", "Coverage", TurnExplainersMixin) == text
