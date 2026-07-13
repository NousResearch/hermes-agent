import json
from types import SimpleNamespace

from hermes_cli.main import cmd_ask_col
from hermes_os_integration.conversational import (
    ChatEnvelope,
    INTENT_ARCHITECTURE,
    INTENT_NEW_PROJECT,
    INTENT_RESEARCH,
    chief_of_staff_plan,
    col_dashboard_panels,
    route_intent,
)
from hermes_os_integration.dashboard import build_project_dashboard


def test_routes_new_project_into_launch_workflow():
    route = route_intent("Build a CRM for wholesalers")

    assert route.intent == INTENT_NEW_PROJECT
    assert route.workflow == "new_project_launch"
    assert route.confidence > 0.8


def test_routes_research_and_architecture_requests():
    assert route_intent("Research ERP competitors").intent == INTENT_RESEARCH
    assert route_intent("Review architecture for this project").intent == INTENT_ARCHITECTURE


def test_chief_of_staff_plan_is_coordination_not_direct_work(tmp_path):
    (tmp_path / "project.md").write_text("# Sample\n", encoding="utf-8")
    plan = chief_of_staff_plan(
        ChatEnvelope(
            message="Build a CRM for wholesalers",
            project_id="sample",
            active_goal="Project launch",
            active_initiative="CRM",
        ),
        project_path=str(tmp_path),
    )

    assert plan.route.workflow == "new_project_launch"
    assert [step.step_id for step in plan.steps][:5] == [
        "grill-me",
        "prd",
        "architecture",
        "plan",
        "tasks",
    ]
    assert any(item.requires_approval for item in plan.steps)
    assert any(item.layer == "coordination" for item in plan.delegations)
    assert any(item.layer == "management" for item in plan.delegations)
    assert any(item.layer == "worker" for item in plan.delegations)
    assert "Chief of Staff coordinates" in " ".join(plan.guardrails)


def test_col_dashboard_panels_expose_context_workflow_and_agents(tmp_path):
    panels = col_dashboard_panels(str(tmp_path))
    panel_ids = {panel["panel_id"] for panel in panels}

    assert "col-active-context" in panel_ids
    assert "col-chief-of-staff" in panel_ids
    assert "col-workflow-preview" in panel_ids
    assert "col-agent-hierarchy" in panel_ids


def test_project_dashboard_includes_col_panels(tmp_path):
    summary = build_project_dashboard(str(tmp_path))
    panel_ids = {panel["panel_id"] for panel in summary["panels"]}

    assert "col-chief-of-staff" in panel_ids
    assert "col-workflow-preview" in panel_ids


def test_cmd_ask_col_prints_json_plan(tmp_path, capsys):
    args = SimpleNamespace(
        message=["Build", "a", "CRM", "for", "wholesalers"],
        project=str(tmp_path),
        user_id="operator",
        session_id="test-session",
        goal="Launch",
        initiative="COL",
        live=False,
    )

    cmd_ask_col(args)
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert payload["route"]["workflow"] == "new_project_launch"
    assert payload["request"]["dry_run"] is True
    assert payload["request"]["session_id"] == "test-session"

