from agent.swarm_state import AuditEvent, EvidenceRequirement, PermissionGrant, RoutingPlan, SwarmJob
from agent.swarm_status import format_swarm_status, load_swarm_status_text, redact_secrets
from agent.swarm_store import SwarmStore


def test_formats_active_jobs_with_status_tasks_blockers_and_last_event():
    job = SwarmJob.create("Research A and B", created_at="2026-01-01T00:00:00+00:00")
    job.routing_plan = RoutingPlan(mode="swarm", reason="parallel research")
    job.transition("running")
    job.add_task("Research A", status="running")
    job.add_task("Send report", permission_required=True, status="blocked")
    job.audit.append(AuditEvent("checkpoint", "waiting on approval", metadata={"safe": "yes"}))

    text = format_swarm_status([job])

    assert "Swarm operator status" in text
    assert f"{job.job_id}: running" in text
    assert "Research A" in text
    assert "blockers:" in text
    assert "permission required for Send report" in text
    assert "last event: checkpoint" in text


def test_redacts_secrets_from_audit_metadata():
    job = SwarmJob.create("Check status", created_at="2026-01-01T00:00:00+00:00")
    job.audit.append(
        AuditEvent(
            "tool_result",
            "received metadata",
            metadata={"api_key": "sk-live", "nested": {"token": "abc", "count": 2}},
        )
    )

    text = format_swarm_status([job])

    assert "sk-live" not in text
    assert "abc" not in text
    assert "[REDACTED]" in text
    assert redact_secrets({"password": "p"})["password"] == "[REDACTED]"


def test_shows_permission_requests_clearly():
    job = SwarmJob.create("Deploy this", created_at="2026-01-01T00:00:00+00:00")
    grant = PermissionGrant(
        permission_id="perm_deploy",
        description="Permission required before deploy",
        scope={"target": "prod", "credential_token": "secret"},
    )
    job.permissions.append(grant)

    text = format_swarm_status([job])

    assert "permission requests:" in text
    assert "perm_deploy: requested" in text
    assert "Permission required before deploy" in text
    assert "prod" in text
    assert "secret" not in text


def test_load_swarm_status_text_does_not_fallback_to_other_sessions():
    job = SwarmJob.create("private other session", session_id="session-a", created_at="2026-01-01T00:00:00+00:00")
    job.transition("running")
    SwarmStore().save_job(job)

    assert load_swarm_status_text(session_id="session-b") == ""
    assert "private other session" in load_swarm_status_text(session_id="session-a")


def test_status_exposes_missing_evidence_from_synthesis():
    job = SwarmJob.create("research", created_at="2026-01-01T00:00:00+00:00")
    job.routing_plan = RoutingPlan(
        mode="swarm",
        reason="needs proof",
        evidence_requirements=[EvidenceRequirement("citation", "Cite sources")],
        verification_required=True,
    )
    task = job.add_task("research", status="needs_review")
    task.result = {"summary": "done", "claims": ["claim"], "evidence": []}
    job.transition("partially_completed")

    text = format_swarm_status([job], include_completed=True)

    assert "evidence:" in text
    assert "citation: missing" in text
    assert "safe to present complete: no" in text
