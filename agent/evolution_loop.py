from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from hermes_state import SessionDB
from agent.evolution_types import EvolutionProposal
from agent.run_state_store import RunStateStore
from agent.runtime_types import ArtifactRecord, InterruptionRecord

_ALLOWED_TARGET_KINDS = {"skill", "prompt", "doc"}


def _read_proposal(path_or_ref: str) -> dict:
    return json.loads(Path(path_or_ref).read_text(encoding="utf-8"))


def _write_proposal(path_or_ref: str, payload: dict) -> None:
    Path(path_or_ref).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_proposal_from_run(
    db: SessionDB,
    run_id: str,
    *,
    target_kind: str,
    target_ref: str,
) -> Optional[EvolutionProposal]:
    detail = db.get_run(run_id)
    if not detail:
        return None
    if target_kind not in _ALLOWED_TARGET_KINDS:
        return None

    run = detail["run"]
    steps = detail["steps"]
    events = detail["events"]

    failed_step_ids = [step["id"] for step in steps if step.get("status") == "failed"]
    event_types = sorted({event["event_type"] for event in events})
    if not failed_step_ids:
        return None
    if "ToolCallFailed" not in event_types and "StepFailed" not in event_types:
        return None

    tool_names = [step.get("tool_name") for step in steps if step.get("tool_name")]
    primary_tool = tool_names[0] if tool_names else "unknown_tool"
    problem_summary = f"Run {run_id} failed around {primary_tool}; create bounded follow-up proposal."
    change_summary = f"Review {target_ref} to reduce recurrence of {primary_tool} failures."
    proposed_patch_summary = f"Draft a minimal {target_kind} update focused on prechecks, fallback, or operator guidance."
    verification_plan = "Read back proposal artifact and run targeted regression before any apply step."

    return EvolutionProposal.create(
        source_run_id=run_id,
        source_session_id=run["session_id"],
        target_kind=target_kind,
        target_ref=target_ref,
        problem_summary=problem_summary,
        evidence={
            "run_id": run_id,
            "step_ids": failed_step_ids,
            "event_types": event_types,
            "artifact_refs": [artifact["path_or_ref"] for artifact in detail["artifacts"]],
            "delegation_ids": [delegation["id"] for delegation in detail["delegations"]],
        },
        change_summary=change_summary,
        proposed_patch_summary=proposed_patch_summary,
        verification_plan=verification_plan,
        risk_level="low",
        requires_human_approval=True,
    )


def persist_proposal_artifact(
    db: SessionDB,
    proposal: EvolutionProposal,
    *,
    output_dir: Path,
    produced_by: str = "assistant",
) -> ArtifactRecord:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{proposal.proposal_id}.json"
    artifact_path.write_text(
        json.dumps(proposal.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    artifact = ArtifactRecord.create(
        run_id=proposal.source_run_id,
        step_id=None,
        artifact_type="evolution_proposal",
        path_or_ref=str(artifact_path),
        produced_by=produced_by,
        purpose="autonomy_loop_proposal",
        is_final=False,
        delivered=False,
    )
    db._execute_write(lambda conn: conn.execute(
        """
        INSERT INTO artifacts (
            id, run_id, step_id, artifact_type, path_or_ref,
            produced_by, purpose, is_final, delivered, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            artifact.id,
            artifact.run_id,
            artifact.step_id,
            artifact.artifact_type,
            artifact.path_or_ref,
            artifact.produced_by,
            artifact.purpose,
            artifact.is_final,
            artifact.delivered,
            artifact.created_at,
        ),
    ))
    return artifact


def request_proposal_approval(
    db: SessionDB,
    proposal_path: str,
    run_id: str,
    *,
    step_id: str | None = None,
) -> InterruptionRecord:
    proposal = _read_proposal(proposal_path)
    proposal["status"] = "pending_approval"
    _write_proposal(proposal_path, proposal)

    interruption = InterruptionRecord.create(
        run_id=run_id,
        step_id=step_id,
        reason_type="waiting_external",
        waiting_on="proposal_approval",
        snapshot={
            "proposal_id": proposal["proposal_id"],
            "proposal_path": proposal_path,
            "decision": None,
        },
        resumable=True,
        status="open",
    )
    RunStateStore(db).create_interruption(interruption)
    return interruption


def resolve_proposal_approval(
    db: SessionDB,
    proposal_path: str,
    interruption_id: str,
    *,
    decision: str,
) -> dict:
    if decision not in {"approved", "rejected"}:
        raise ValueError(f"Unsupported proposal approval decision: {decision}")

    proposal = _read_proposal(proposal_path)
    proposal["status"] = decision
    _write_proposal(proposal_path, proposal)
    RunStateStore(db).resume_interruption(interruption_id)
    return proposal
