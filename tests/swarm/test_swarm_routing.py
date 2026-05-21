from agent.swarm_router import route_request


def test_simple_explanatory_prompt_routes_direct():
    plan = route_request("Explain why the sky is blue")

    assert plan.mode == "direct"
    assert plan.permission_requests == []
    assert plan.verification_required is False


def test_multiple_independent_research_and_review_routes_swarm():
    plan = route_request("Research the API docs and review the code for security issues")

    assert plan.mode == "swarm"
    assert plan.suggested_tasks
    assert plan.verification_required is True


def test_more_than_three_procedural_steps_routes_script_candidate():
    plan = route_request("Collect the logs, then parse errors, validate counts, verify source of truth, and summarize")

    assert plan.mode == "script"
    assert plan.metadata["step_count"] > 3
    assert plan.verification_required is True


def test_external_send_deploy_destructive_wording_requires_permission():
    plan = route_request("Deploy the service and delete the old release")

    assert plan.permission_requests
    assert plan.metadata["permission_required"] is True
    assert plan.verification_required is True


def test_n8n_docker_wording_routes_pipe_with_permission_gate():
    plan = route_request("Run the n8n workflow in docker compose")

    assert plan.mode == "pipe"
    assert any(grant.permission_id == "perm_pipe_execution" for grant in plan.permission_requests)
    assert plan.verification_required is True


def test_cross_session_context_pressure_is_first_class_telemetry():
    plan = route_request(
        "Using the other session and earlier thread, review the architecture, compare files, and propose next steps",
        platform_context={"thread_message_count": 18, "compressed_context": True, "attachment_count": 2},
    )

    telemetry = plan.routing_telemetry
    assert telemetry.context_pressure >= 3
    assert telemetry.signals["cross_session_reference"] is True
    assert telemetry.signals["compressed_context"] is True
    assert telemetry.required_scaffold in {"externalize_state", "parallel_review", "script"}


def test_external_actions_add_human_approval_evidence_requirement():
    plan = route_request("Draft and send the client-facing email, then deploy the update")

    kinds = [item.kind for item in plan.evidence_requirements]
    assert "human_approval" in kinds
    assert plan.routing_telemetry.complexity_risk >= 2
    assert plan.routing_telemetry.required_scaffold == "approval"


def test_procedural_source_of_truth_work_adds_command_or_dry_run_evidence():
    plan = route_request("Collect CSV exports, parse every row, validate counts against source of truth, verify totals, and summarize")

    kinds = [item.kind for item in plan.evidence_requirements]
    assert plan.mode == "script"
    assert "command_output" in kinds
    assert plan.routing_telemetry.required_scaffold == "script"


def test_complex_research_forces_role_separated_panel_policy():
    plan = route_request("Get a model council to research the options, critique failure modes, and synthesize a recommendation")

    assert plan.mode == "swarm"
    assert plan.panel_policy["considered"] is True
    assert plan.panel_policy["required"] is True
    assert "skeptic" in plan.panel_policy["roles"]
    assert "citation" in [item.kind for item in plan.evidence_requirements]


def test_platform_context_non_numeric_values_do_not_break_routing():
    plan = route_request("Explain profile setup", platform_context={"thread_message_count": "many", "attachment_count": "unknown", "source_count": None})

    assert plan.mode == "direct"
    assert plan.routing_telemetry.context_pressure >= 0


def test_code_only_swarm_does_not_require_citation_evidence():
    plan = route_request("Implement the router change and test the executor")

    kinds = [item.kind for item in plan.evidence_requirements]
    assert "test" in kinds
    assert "citation" not in kinds


def test_profile_word_does_not_trigger_code_file_evidence():
    plan = route_request("Explain the profile settings")

    assert "test" not in [item.kind for item in plan.evidence_requirements]


def test_code_review_swarm_requires_tests_not_citations():
    plan = route_request("Review the code and test the executor")

    kinds = [item.kind for item in plan.evidence_requirements]
    assert "test" in kinds
    assert "citation" not in kinds


def test_plural_tests_signal_counts_as_test_evidence_requirement():
    plan = route_request("Analyze this function and write tests")

    kinds = [item.kind for item in plan.evidence_requirements]
    assert "test" in kinds
    assert "citation" not in kinds
