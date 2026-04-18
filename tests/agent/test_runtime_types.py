from agent.runtime_types import (
    ArtifactRecord,
    DelegationRecord,
    InterruptionRecord,
    RunEventRecord,
    RunRecord,
    RunStepRecord,
)


def test_run_record_create_sets_defaults():
    record = RunRecord.create(
        session_id="session-1",
        parent_run_id=None,
        source="cli",
        user_intent="test intent",
        state="intake",
        next_step=None,
    )

    assert record.id
    assert record.session_id == "session-1"
    assert record.state == "intake"
    assert record.started_at > 0
    assert record.metadata == {}


def test_runtime_records_can_serialize_to_db_dicts():
    run = RunRecord.create(
        session_id="session-1",
        parent_run_id=None,
        source="cli",
        user_intent="test intent",
        state="executing",
        next_step="call_tool",
    )
    step = RunStepRecord.create(
        run_id=run.id,
        step_index=0,
        step_type="model_call",
        status="started",
    )
    event = RunEventRecord.create(
        run_id=run.id,
        step_id=step.id,
        event_type="StepStarted",
        payload={"foo": "bar"},
    )
    interruption = InterruptionRecord.create(
        run_id=run.id,
        step_id=step.id,
        reason_type="waiting_user",
        waiting_on="clarify",
        snapshot={"message": "need user answer"},
        resumable=True,
        status="open",
    )
    delegation = DelegationRecord.create(
        parent_run_id=run.id,
        child_session_id="child-1",
        goal="review this",
        context_summary="delegated for review",
        allowed_toolsets=["file"],
        side_effect_policy="read_only",
        expected_output_type="summary",
        verification_status="pending",
        status="started",
    )
    artifact = ArtifactRecord.create(
        run_id=run.id,
        step_id=step.id,
        artifact_type="report",
        path_or_ref="/tmp/report.md",
        produced_by="assistant",
        purpose="final response",
        is_final=True,
        delivered=False,
    )

    assert run.to_db_dict()["state"] == "executing"
    assert step.to_db_dict()["step_type"] == "model_call"
    assert event.to_db_dict()["payload_json"] == '{"foo": "bar"}'
    assert interruption.to_db_dict()["snapshot_json"] == '{"message": "need user answer"}'
    assert delegation.to_db_dict()["allowed_toolsets_json"] == '["file"]'
    assert artifact.to_db_dict()["artifact_type"] == "report"
