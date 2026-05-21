import json

from agent.swarm_state import EvidenceRequirement, EvalResult, PermissionGrant, RoutingPlan, RoutingTelemetry, SwarmJob


def test_swarm_job_create_sets_defaults_and_stable_shape():
    job = SwarmJob.create(
        "please check the logs",
        platform="slack",
        user_id="U1",
        chat_id="C1",
        session_id="S1",
        created_at="2026-05-21T00:00:00+00:00",
    )

    assert job.job_id.startswith("swarm_")
    assert job.status == "received"
    assert job.created_at == "2026-05-21T00:00:00+00:00"
    assert job.updated_at == job.created_at
    assert job.original_request == "please check the logs"
    assert job.tasks == []
    assert job.audit == []

    same = SwarmJob.create(
        "please check the logs",
        platform="slack",
        user_id="U1",
        chat_id="C1",
        session_id="S1",
        created_at="2026-05-21T00:00:00+00:00",
    )
    assert same.job_id == job.job_id


def test_add_task_appends_task_and_audit_event():
    job = SwarmJob.create("do work")
    task = job.add_task("Research", "look up docs", mode="swarm", toolsets=["web"])

    assert task in job.tasks
    assert task.status == "queued"
    assert job.audit[-1].event_type == "task_added"
    assert job.audit[-1].metadata["task_id"] == task.task_id


def test_transition_records_status_change_and_audit_event():
    job = SwarmJob.create("do work")
    job.transition("triaging", metadata={"reason": "shadow"})

    assert job.status == "triaging"
    assert job.audit[-1].event_type == "status_changed"
    assert job.audit[-1].metadata["from"] == "received"
    assert job.audit[-1].metadata["to"] == "triaging"
    assert job.audit[-1].metadata["reason"] == "shadow"


def test_json_round_trip_preserves_nested_data():
    job = SwarmJob.create("deploy after approval", platform="slack", user_id="U1")
    grant = PermissionGrant("perm_deploy", "Deploy production", scope={"env": "prod"})
    job.permissions.append(grant)
    job.evals.append(EvalResult("unit", True, "passed"))
    job.routing_plan = RoutingPlan(
        mode="swarm",
        reason="multi-step",
        suggested_tasks=[{"title": "Review code"}],
        permission_requests=[grant],
        verification_required=True,
    )
    job.add_task("Review code", permission_required=True)

    encoded = json.loads(json.dumps(job.to_dict()))
    restored = SwarmJob.from_dict(encoded)

    assert restored.to_dict() == job.to_dict()


def test_routing_plan_round_trip_preserves_anti_illusion_metadata():
    requirement = EvidenceRequirement(
        kind="citation",
        description="Cite the source used for current facts",
        source="router",
        metadata={"reason": "research"},
    )
    telemetry = RoutingTelemetry(
        complexity_risk=2,
        context_pressure=3,
        illusion_risk=3,
        required_scaffold="parallel_review",
        weak_output_risk=True,
        signals={"cross_session_reference": True},
    )
    plan = RoutingPlan(
        mode="swarm",
        reason="needs a small panel",
        verification_required=True,
        evidence_requirements=[requirement],
        routing_telemetry=telemetry,
        panel_policy={"considered": True, "required": True, "roles": ["scout", "skeptic"]},
    )

    restored = RoutingPlan.from_dict(json.loads(json.dumps(plan.to_dict())))

    assert restored.to_dict() == plan.to_dict()
    assert restored.evidence_requirements[0].kind == "citation"
    assert restored.routing_telemetry.context_pressure == 3
    assert restored.panel_policy["roles"] == ["scout", "skeptic"]


def test_routing_plan_from_dict_accepts_v1_snapshots_without_anti_illusion_fields():
    restored = RoutingPlan.from_dict({"mode": "direct", "reason": "old snapshot"})

    assert restored.mode == "direct"
    assert restored.evidence_requirements == []
    assert restored.routing_telemetry.to_dict() == RoutingTelemetry().to_dict()
    assert restored.panel_policy == {}


def test_routing_plan_to_dict_accepts_serialized_new_field_shapes():
    plan = RoutingPlan(
        mode="swarm",
        reason="serialized",
        evidence_requirements=[{"kind": "citation", "description": "cite"}],
        routing_telemetry={"complexity_risk": 1, "required_scaffold": "checklist"},
    )

    encoded = plan.to_dict()

    assert encoded["evidence_requirements"][0]["kind"] == "citation"
    assert encoded["routing_telemetry"]["complexity_risk"] == 1
