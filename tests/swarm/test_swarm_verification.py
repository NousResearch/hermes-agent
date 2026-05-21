from agent.swarm_state import EvidenceRequirement, RoutingPlan, SwarmJob
from agent.swarm_verify import build_verification_plan, apply_verification_results


def test_build_verification_plan_includes_child_results_and_blockers():
    job = SwarmJob.create("implement and test", created_at="2026-01-01T00:00:00+00:00")
    job.add_task("code", task_id="t1")
    job.add_task("deploy", task_id="t2", permission_required=True)
    job.tasks[0].result = {"summary": "implemented", "evidence": "pytest passed"}
    job.tasks[1].status = "blocked"
    plan = RoutingPlan(mode="swarm", reason="parallel", verification_required=True)

    verification = build_verification_plan(job, plan)

    assert verification.required is True
    assert "child_result_review" in [step["id"] for step in verification.steps]
    assert "permission_blocker_review" in [step["id"] for step in verification.steps]
    assert verification.summary()["status"] == "pending"


def test_apply_verification_results_persists_evals_and_transitions():
    job = SwarmJob.create("verify me", created_at="2026-01-01T00:00:00+00:00")
    plan = RoutingPlan(mode="direct", reason="simple", verification_required=False)
    verification = build_verification_plan(job, plan)

    apply_verification_results(job, verification, {"direct_sanity_check": (True, "ok")})

    assert job.evals[0].name == "direct_sanity_check"
    assert job.evals[0].passed is True
    assert job.metadata["swarm_verification"]["status"] == "passed"


def test_apply_verification_results_marks_failed_verification_without_overwriting_failed_job():
    job = SwarmJob.create("verify failure", created_at="2026-01-01T00:00:00+00:00")
    job.transition("failed")
    plan = RoutingPlan(mode="swarm", reason="parallel", verification_required=True)
    verification = build_verification_plan(job, plan)

    apply_verification_results(job, verification, {"final_consistency_check": {"passed": False, "details": "missing evidence"}})

    assert job.status == "failed"
    assert job.metadata["swarm_verification"]["status"] == "failed"


def test_verification_plan_requires_code_config_evidence():
    job = SwarmJob.create("implement config patch", created_at="2026-01-01T00:00:00+00:00")
    job.add_task("Edit code", description="patch config and run tests")

    verification = build_verification_plan(job, RoutingPlan(mode="swarm", reason="code work", verification_required=True))

    step_ids = [step["id"] for step in verification.steps]
    assert "code_config_verification" in step_ids


def test_verification_plan_requires_dry_run_payload_contract_for_pipes():
    job = SwarmJob.create("trigger n8n docker workflow", created_at="2026-01-01T00:00:00+00:00")

    verification = build_verification_plan(job, RoutingPlan(mode="pipe", reason="n8n pipe", verification_required=True))

    step_ids = [step["id"] for step in verification.steps]
    assert "dry_run_payload_contract" in step_ids


def test_verification_plan_requires_human_review_for_external_actions():
    job = SwarmJob.create("send client-facing email", created_at="2026-01-01T00:00:00+00:00")

    verification = build_verification_plan(job, RoutingPlan(mode="swarm", reason="external send", verification_required=True))

    step_ids = [step["id"] for step in verification.steps]
    assert "human_approval_review" in step_ids


def test_verification_plan_maps_explicit_evidence_requirements_to_steps():
    job = SwarmJob.create("research current docs and prove it", created_at="2026-01-01T00:00:00+00:00")
    plan = RoutingPlan(
        mode="swarm",
        reason="research",
        verification_required=True,
        evidence_requirements=[
            EvidenceRequirement("citation", "Cite current source"),
            EvidenceRequirement("artifact", "Return output file path"),
            EvidenceRequirement("human_approval", "Get Garrett approval before send"),
        ],
    )

    verification = build_verification_plan(job, plan)

    step_ids = [step["id"] for step in verification.steps]
    assert "evidence_citation" in step_ids
    assert "evidence_artifact" in step_ids
    assert "evidence_human_approval" in step_ids
    citation_step = next(step for step in verification.steps if step["id"] == "evidence_citation")
    assert citation_step["evidence_kind"] == "citation"
