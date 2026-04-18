import json
from pathlib import Path

from hermes_state import SessionDB
from agent.run_state_store import RunStateStore
from agent.runtime_types import RunEventRecord, RunRecord, RunStepRecord


def test_build_proposal_from_failed_tool_run_and_register_artifact(tmp_path: Path):
    from agent.evolution_loop import build_proposal_from_run, persist_proposal_artifact

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-1", "cli")
        store = RunStateStore(db)
        run = RunRecord.create(
            session_id="session-1",
            parent_run_id=None,
            source="cli",
            user_intent="debug broken tool",
            state="failed",
            next_step=None,
            final_status="failed",
        )
        store.create_run(run)
        step = RunStepRecord.create(
            run_id=run.id,
            step_index=0,
            step_type="tool_execution",
            status="failed",
            tool_name="web_search",
            error="timeout",
        )
        store.create_step(step)
        event = RunEventRecord.create(
            run_id=run.id,
            step_id=step.id,
            event_type="ToolCallFailed",
            payload={"tool_name": "web_search", "error": "timeout"},
        )
        store.append_event(event)

        proposal = build_proposal_from_run(db, run.id, target_kind="doc", target_ref="docs/runtime.md")

        assert proposal.source_run_id == run.id
        assert proposal.source_session_id == "session-1"
        assert proposal.status == "draft"
        assert proposal.evidence["step_ids"] == [step.id]
        assert proposal.evidence["event_types"] == ["ToolCallFailed"]
        assert proposal.risk_level == "low"
        assert proposal.requires_human_approval is True

        artifact = persist_proposal_artifact(
            db,
            proposal,
            output_dir=tmp_path / "artifacts",
            produced_by="assistant",
        )

        artifact_path = Path(artifact.path_or_ref)
        assert artifact.artifact_type == "evolution_proposal"
        assert artifact_path.exists()
        payload = json.loads(artifact_path.read_text())
        assert payload["proposal_id"] == proposal.proposal_id
        detail = db.get_run(run.id)
        assert detail is not None
        assert detail["artifacts"][0]["artifact_type"] == "evolution_proposal"
    finally:
        db.close()


def test_build_proposal_from_run_rejects_weak_evidence(tmp_path: Path):
    from agent.evolution_loop import build_proposal_from_run

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-1", "cli")
        detailless_run = db.list_runs_for_session("session-1")
        assert detailless_run == []

        run = build_proposal_from_run(db, "missing-run", target_kind="doc", target_ref="docs/runtime.md")
        assert run is None
    finally:
        db.close()


def test_build_proposal_rejects_disallowed_target_kind(tmp_path: Path):
    from agent.evolution_loop import build_proposal_from_run

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-1", "cli")
        store = RunStateStore(db)
        run = RunRecord.create(
            session_id="session-1",
            parent_run_id=None,
            source="cli",
            user_intent="debug broken tool",
            state="failed",
            next_step=None,
            final_status="failed",
        )
        store.create_run(run)
        step = RunStepRecord.create(
            run_id=run.id,
            step_index=0,
            step_type="tool_execution",
            status="failed",
            tool_name="web_search",
            error="timeout",
        )
        store.create_step(step)
        event = RunEventRecord.create(
            run_id=run.id,
            step_id=step.id,
            event_type="ToolCallFailed",
            payload={"tool_name": "web_search", "error": "timeout"},
        )
        store.append_event(event)

        proposal = build_proposal_from_run(db, run.id, target_kind="code", target_ref="run_agent.py")
        assert proposal is None
    finally:
        db.close()


def test_build_proposal_requires_failed_step_and_failure_event(tmp_path: Path):
    from agent.evolution_loop import build_proposal_from_run

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-1", "cli")
        store = RunStateStore(db)
        run = RunRecord.create(
            session_id="session-1",
            parent_run_id=None,
            source="cli",
            user_intent="non failure run",
            state="completed",
            next_step=None,
            final_status="completed",
        )
        store.create_run(run)
        step = RunStepRecord.create(
            run_id=run.id,
            step_index=0,
            step_type="tool_execution",
            status="completed",
            tool_name="web_search",
        )
        store.create_step(step)
        event = RunEventRecord.create(
            run_id=run.id,
            step_id=step.id,
            event_type="ToolCallCompleted",
            payload={"tool_name": "web_search", "result": "ok"},
        )
        store.append_event(event)

        proposal = build_proposal_from_run(db, run.id, target_kind="doc", target_ref="docs/runtime.md")
        assert proposal is None
    finally:
        db.close()


def test_request_and_resolve_proposal_approval_round_trip(tmp_path: Path):
    from agent.evolution_loop import (
        build_proposal_from_run,
        persist_proposal_artifact,
        request_proposal_approval,
        resolve_proposal_approval,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-1", "cli")
        store = RunStateStore(db)
        run = RunRecord.create(
            session_id="session-1",
            parent_run_id=None,
            source="cli",
            user_intent="debug broken tool",
            state="failed",
            next_step=None,
            final_status="failed",
        )
        store.create_run(run)
        step = RunStepRecord.create(
            run_id=run.id,
            step_index=0,
            step_type="tool_execution",
            status="failed",
            tool_name="web_search",
            error="timeout",
        )
        store.create_step(step)
        event = RunEventRecord.create(
            run_id=run.id,
            step_id=step.id,
            event_type="ToolCallFailed",
            payload={"tool_name": "web_search", "error": "timeout"},
        )
        store.append_event(event)

        proposal = build_proposal_from_run(db, run.id, target_kind="doc", target_ref="docs/runtime.md")
        artifact = persist_proposal_artifact(db, proposal, output_dir=tmp_path / "artifacts")

        interruption = request_proposal_approval(db, artifact.path_or_ref, run.id, step_id=step.id)
        proposal_after_request = json.loads(Path(artifact.path_or_ref).read_text())
        assert proposal_after_request["status"] == "pending_approval"
        assert interruption.reason_type == "waiting_external"
        assert interruption.waiting_on == "proposal_approval"
        assert interruption.snapshot["proposal_id"] == proposal.proposal_id

        updated = resolve_proposal_approval(db, artifact.path_or_ref, interruption.id, decision="approved")
        assert updated["status"] == "approved"

        detail = db.get_run(run.id)
        assert detail is not None
        assert detail["interruptions"][0]["status"] == "resumed"
    finally:
        db.close()
