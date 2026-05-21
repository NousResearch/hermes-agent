import json

from agent.swarm_executor import build_delegate_tasks, execute_swarm
from agent.swarm_state import EvidenceRequirement, RoutingPlan, SwarmJob


def _plan(tasks, mode="swarm"):
    return RoutingPlan(mode=mode, reason="test", suggested_tasks=tasks, verification_required=True)


def test_executes_at_most_max_children_independent_tasks():
    job = SwarmJob.create("do several things", created_at="2026-01-01T00:00:00+00:00")
    plan = _plan([
        {"title": "one"},
        {"title": "two"},
        {"title": "three"},
    ])
    calls = []

    def delegate_fn(**kwargs):
        calls.append(kwargs)
        return {"results": [{"status": "completed"}, {"status": "completed"}]}

    result = execute_swarm(job, plan, delegate_fn, max_children=2)

    assert len(result.dispatched) == 2
    assert len(calls[0]["tasks"]) == 2
    assert job.status == "completed"


def test_build_delegate_tasks_preserves_declared_toolsets():
    job = SwarmJob.create("research and code", created_at="2026-01-01T00:00:00+00:00")
    plan = _plan([
        {"title": "research", "description": "look up docs", "toolsets": ["web"]},
        {"title": "code", "description": "inspect files", "toolsets": ["terminal", "file"]},
    ])

    tasks = build_delegate_tasks(plan, job, max_children=3)

    assert tasks[0]["toolsets"] == ["web"]
    assert tasks[1]["toolsets"] == ["terminal", "file"]


def test_build_delegate_tasks_injects_required_evidence_contract():
    job = SwarmJob.create("research with citations", created_at="2026-01-01T00:00:00+00:00")
    plan = RoutingPlan(
        mode="swarm",
        reason="research",
        suggested_tasks=[{"title": "research", "description": "find sources"}],
        verification_required=True,
        evidence_requirements=[
            EvidenceRequirement("citation", "Cite source URLs"),
            EvidenceRequirement("artifact", "Return any output path"),
        ],
    )

    tasks = build_delegate_tasks(plan, job, max_children=3)

    context = tasks[0]["context"]
    assert "Required evidence" in context
    assert "citation" in context
    assert "artifact" in context
    assert "claims" in context
    assert "side_effects_performed" in context


def test_aggregates_results_into_job_summary_data():
    job = SwarmJob.create("parallel work", created_at="2026-01-01T00:00:00+00:00")
    plan = _plan([{"title": "alpha"}, {"title": "beta"}])

    def delegate_fn(**kwargs):
        return json.dumps({"results": [{"summary": "A done"}, {"summary": "B done"}]})

    result = execute_swarm(job, plan, delegate_fn, max_children=3)

    assert result.results == [{"summary": "A done"}, {"summary": "B done"}]
    assert job.metadata["swarm_execution"]["results"][0]["summary"] == "A done"
    assert job.tasks[0].result is not None
    assert job.tasks[0].result["summary"] == "A done"


def test_partial_child_failure_marks_partially_completed_unless_critical():
    job = SwarmJob.create("parallel work", created_at="2026-01-01T00:00:00+00:00")
    plan = _plan([{"title": "alpha"}, {"title": "beta"}])

    def delegate_fn(**kwargs):
        return {"results": [{"summary": "A done"}, {"error": "boom", "status": "failed"}]}

    execute_swarm(job, plan, delegate_fn, max_children=3)

    assert job.status == "partially_completed"
    assert job.tasks[1].status == "failed"


def test_critical_child_failure_marks_failed():
    job = SwarmJob.create("parallel work", created_at="2026-01-01T00:00:00+00:00")
    plan = _plan([{"title": "alpha", "critical": True}])

    def delegate_fn(**kwargs):
        return {"results": [{"error": "boom", "status": "failed"}]}

    execute_swarm(job, plan, delegate_fn, max_children=3)

    assert job.status == "failed"


def test_permission_required_tasks_remain_blocked_and_are_not_dispatched():
    job = SwarmJob.create("send an email", created_at="2026-01-01T00:00:00+00:00")
    plan = _plan([
        {"title": "draft", "permission_required": False},
        {"title": "send", "permission_required": True},
    ])
    calls = []

    def delegate_fn(**kwargs):
        calls.append(kwargs)
        return {"results": [{"summary": "drafted"}]}

    result = execute_swarm(job, plan, delegate_fn, max_children=3)

    assert len(calls[0]["tasks"]) == 1
    assert calls[0]["tasks"][0]["goal"] == "draft"
    assert result.blocked[0]["title"] == "send"
    assert job.tasks[1].status == "blocked"
    assert job.status == "partially_completed"


def test_weak_child_output_without_required_evidence_is_marked_needs_review():
    job = SwarmJob.create("research and cite sources", created_at="2026-01-01T00:00:00+00:00")
    plan = RoutingPlan(
        mode="swarm",
        reason="research",
        suggested_tasks=[{"title": "research"}],
        verification_required=True,
        evidence_requirements=[EvidenceRequirement("citation", "Cite sources")],
    )

    def delegate_fn(**kwargs):
        return {"results": [{"summary": "Looks good"}]}

    execute_swarm(job, plan, delegate_fn, max_children=3)

    assert job.tasks[0].status == "needs_review"
    assert job.tasks[0].result["weak_output"]["weak"] is True
    assert "no_evidence" in job.tasks[0].result["weak_output"]["reasons"]
    assert job.status == "partially_completed"


def test_weak_output_detector_enforces_dry_run_kind_not_generic_evidence():
    job = SwarmJob.create("run pipe", created_at="2026-01-01T00:00:00+00:00")
    plan = RoutingPlan(
        mode="pipe",
        reason="pipe",
        suggested_tasks=[{"title": "dry run"}],
        verification_required=True,
        evidence_requirements=[EvidenceRequirement("dry_run", "Show dry-run output")],
    )

    def delegate_fn(**kwargs):
        return {"results": [{"summary": "done", "evidence": "some notes"}]}

    execute_swarm(job, plan, delegate_fn, max_children=3)

    assert job.tasks[0].status == "needs_review"
    assert "missing_dry_run" in job.tasks[0].result["weak_output"]["reasons"]


def test_weak_output_detector_accepts_dict_evidence_requirements():
    job = SwarmJob.create("research", created_at="2026-01-01T00:00:00+00:00")
    plan = RoutingPlan(
        mode="swarm",
        reason="research",
        suggested_tasks=[{"title": "research"}],
        verification_required=True,
        evidence_requirements=[{"kind": "citation", "description": "Cite sources"}],
    )

    def delegate_fn(**kwargs):
        return {"results": [{"summary": "done", "evidence": "generic"}]}

    execute_swarm(job, plan, delegate_fn, max_children=3)

    assert job.tasks[0].status == "needs_review"
    assert "missing_citation" in job.tasks[0].result["weak_output"]["reasons"]


def test_valid_structured_evidence_packet_is_not_marked_weak_and_synthesis_persists():
    job = SwarmJob.create("research", created_at="2026-01-01T00:00:00+00:00")
    plan = RoutingPlan(
        mode="swarm",
        reason="research",
        suggested_tasks=[{"title": "research"}],
        verification_required=True,
        evidence_requirements=[EvidenceRequirement("citation", "Cite sources")],
    )

    def delegate_fn(**kwargs):
        return {
            "results": [
                {
                    "summary": "researched",
                    "claims": ["claim is sourced"],
                    "evidence": [{"kind": "citation", "url": "https://example.com/source"}],
                }
            ]
        }

    execute_swarm(job, plan, delegate_fn, max_children=3)

    assert job.tasks[0].status == "completed"
    assert "weak_output" not in job.tasks[0].result
    assert job.metadata["swarm_synthesis"]["verified_claims"] == ["claim is sourced"]
    assert job.metadata["swarm_synthesis"]["safe_to_present_complete"] is True


def test_synthesis_persists_when_all_tasks_are_blocked_before_dispatch():
    job = SwarmJob.create("send report", created_at="2026-01-01T00:00:00+00:00")
    plan = RoutingPlan(
        mode="swarm",
        reason="approval needed",
        suggested_tasks=[{"title": "send", "permission_required": True}],
        verification_required=True,
        evidence_requirements=[EvidenceRequirement("human_approval", "Garrett approves send")],
    )

    def delegate_fn(**kwargs):
        raise AssertionError("blocked tasks should not dispatch")

    execute_swarm(job, plan, delegate_fn, max_children=3)

    assert job.status == "awaiting_permission"
    assert job.metadata["swarm_synthesis"]["safe_to_present_complete"] is False
    assert job.metadata["swarm_synthesis"]["blocked_tasks"] == [job.tasks[0].task_id]
