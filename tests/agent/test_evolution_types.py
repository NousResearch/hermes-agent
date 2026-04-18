from agent.evolution_types import EvolutionProposal


def test_evolution_proposal_create_sets_defaults_and_serializes():
    proposal = EvolutionProposal.create(
        source_run_id="run-1",
        source_session_id="session-1",
        target_kind="doc",
        target_ref=".hermes/plans/README.md",
        problem_summary="tool failure repeated",
        evidence={
            "run_id": "run-1",
            "step_ids": ["step-1"],
            "event_types": ["ToolCallFailed"],
            "artifact_refs": ["/tmp/export.json"],
            "delegation_ids": [],
        },
        change_summary="tighten docs",
        proposed_patch_summary="update README routing",
        verification_plan="read back file and run tests",
        risk_level="low",
        requires_human_approval=True,
    )

    payload = proposal.to_dict()

    assert proposal.proposal_id
    assert proposal.status == "draft"
    assert payload["source_run_id"] == "run-1"
    assert payload["evidence"]["event_types"] == ["ToolCallFailed"]
    assert payload["requires_human_approval"] is True
