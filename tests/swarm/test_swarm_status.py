from agent.swarm_state import AuditEvent, PermissionGrant, RoutingPlan, SwarmJob
from agent.swarm_status import format_swarm_status, redact_secrets


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
