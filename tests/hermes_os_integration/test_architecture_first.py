from hermes_os_integration.architecture_first import (
    ARCHITECTURE_ORDER,
    DashboardRequirements,
    WorkflowDefinition,
    assumption_challenge_output,
    check_agent_ownership,
    check_architecture_order,
    create_grill_me_session,
    dashboard_feedback,
    dashboard_readiness,
    existing_project_review_targets,
    load_constitution,
    project_document_templates,
    render_review_report,
    review_architecture,
    runtime_delegation_readiness,
    specification_to_tasks,
    validate_architecture_review_request,
    validate_artifact_ingestion,
    validate_workflow_definition,
)


def test_constitution_and_architecture_order_block_premature_code():
    constitution = load_constitution()
    assert "Business logic before implementation." in constitution["rules"]

    violation = check_architecture_order(["business_system"], "implementation")
    assert violation.severity == "block"
    assert "control_plane" in violation.missing_prerequisites
    assert ARCHITECTURE_ORDER[-2] == "implementation"


def test_architect_review_contract_and_report():
    request, error = validate_architecture_review_request({
        "project_id": "sample-project",
        "project_path": "/workspace/projects/sample-project",
        "present_documents": ["PROJECT.md"],
        "completed_stages": ["business_system"],
    })

    assert error is None
    report = review_architecture(request)
    assert report.blocked is True
    assert "DOMAIN.md" in report.missing_documents
    assert "Architecture Review" in render_review_report(report)


def test_grill_me_blocks_unanswered_categories():
    session = create_grill_me_session("project-1")
    output = assumption_challenge_output(session)

    assert "business" in session.questions
    assert output["blocks_task_generation"] is True
    assert "business" in output["unanswered_categories"]


def test_project_templates_and_spec_to_task_contract():
    templates = project_document_templates()
    assert "ARCHITECTURE.md" in templates

    tasks, error = specification_to_tasks({
        "spec_id": "spec-1",
        "summary": "architecture gate",
        "acceptance_criteria": ["Blocks premature coding"],
    }, architecture_approved=True)

    assert error is None
    assert tasks[0]["traceability"] == "spec-1"


def test_workflow_and_dashboard_gates():
    workflow, error = validate_workflow_definition(WorkflowDefinition(
        trigger="new thesis",
        inputs=["ticker"],
        steps=["research", "review"],
        outputs=["report"],
        approvals=["human-review"],
        metrics=["cycle_time"],
        failure_states=["missing evidence"],
        escalation_rules=["notify owner"],
    ))

    assert error is None
    assert workflow.steps == ["research", "review"]

    readiness = dashboard_readiness(DashboardRequirements(
        daily_visibility=["thesis status"],
        success_metrics=["coverage"],
        failure_indicators=["stale thesis"],
        opportunity_indicators=["valuation gap"],
        required_reports=["weekly status"],
    ))

    assert readiness["ready"] is True
    assert dashboard_feedback("coverage", "too low")["traceability"] == "dashboard://coverage"


def test_agent_boundaries_artifacts_and_delegation_gate():
    allowed, error = check_agent_ownership(["research", "source_of_truth_state"])
    assert allowed is None
    assert error.code == "permission_denied"

    artifact = validate_artifact_ingestion({
        "artifact_id": "a1",
        "agent_id": "agent-1",
        "schema": "ResearchReport",
        "content_ref": "file://report.md",
    })
    assert artifact.accepted is True

    readiness = runtime_delegation_readiness(["business_system"], dry_run=False)
    assert readiness["ready"] is False


def test_existing_project_review_targets_include_workspace_projects():
    targets = existing_project_review_targets()
    assert targets == {}
