from pathlib import Path

from hermes_state import SessionDB
from agent.run_state_store import RunStateStore
from agent.runtime_types import (
    ArtifactRecord,
    DelegationRecord,
    InterruptionRecord,
    RunEventRecord,
    RunRecord,
    RunStepRecord,
)


def test_run_state_store_round_trip(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-1", "cli")
        store = RunStateStore(db)

        run = RunRecord.create(
            session_id="session-1",
            parent_run_id=None,
            source="cli",
            user_intent="do the thing",
            state="intake",
            next_step=None,
        )
        store.create_run(run)

        step = RunStepRecord.create(
            run_id=run.id,
            step_index=0,
            step_type="model_call",
            status="started",
        )
        store.create_step(step)

        event = RunEventRecord.create(
            run_id=run.id,
            step_id=step.id,
            event_type="StepStarted",
            payload={"iteration": 0},
        )
        store.append_event(event)

        interruption = InterruptionRecord.create(
            run_id=run.id,
            step_id=step.id,
            reason_type="waiting_user",
            waiting_on="clarify",
            snapshot={"pending": True},
            resumable=True,
            status="open",
        )
        store.create_interruption(interruption)

        delegation = DelegationRecord.create(
            parent_run_id=run.id,
            child_session_id="child-1",
            goal="investigate",
            context_summary="check the logs",
            allowed_toolsets=["terminal"],
            side_effect_policy="read_only",
            expected_output_type="summary",
            verification_status="pending",
            status="started",
        )
        store.create_delegation(delegation)

        artifact = ArtifactRecord.create(
            run_id=run.id,
            step_id=step.id,
            artifact_type="log",
            path_or_ref="/tmp/log.txt",
            produced_by="delegate",
            purpose="debug evidence",
            is_final=False,
            delivered=False,
        )
        store.create_artifact(artifact)

        store.finish_step(step.id, status="completed", output_summary="ok")
        store.resume_interruption(interruption.id)
        store.finish_delegation(delegation.id, status="completed", verification_status="verified")
        store.finish_run(run.id, final_status="completed", state="completed")

        run_row = db._conn.execute("SELECT * FROM runs WHERE id = ?", (run.id,)).fetchone()
        step_row = db._conn.execute("SELECT * FROM run_steps WHERE id = ?", (step.id,)).fetchone()
        event_row = db._conn.execute("SELECT * FROM run_events WHERE id = ?", (event.id,)).fetchone()
        interruption_row = db._conn.execute("SELECT * FROM interruptions WHERE id = ?", (interruption.id,)).fetchone()
        delegation_row = db._conn.execute("SELECT * FROM delegations WHERE id = ?", (delegation.id,)).fetchone()
        artifact_row = db._conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact.id,)).fetchone()

        assert run_row["final_status"] == "completed"
        assert step_row["status"] == "completed"
        assert event_row["event_type"] == "StepStarted"
        assert interruption_row["status"] == "resumed"
        assert delegation_row["verification_status"] == "verified"
        assert artifact_row["path_or_ref"] == "/tmp/log.txt"
    finally:
        db.close()
