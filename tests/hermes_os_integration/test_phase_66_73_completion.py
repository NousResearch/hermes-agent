import json

from hermes_os_integration.conversational import (
    ChatEnvelope,
    ConversationalWorkflow,
    active_agent_activity,
    active_session_store,
    agent_context_package,
    agent_failure_fallback,
    approval_prompt_contract,
    ask_command_contract,
    chat_cli_contract,
    chat_error_envelope,
    chat_help_examples,
    chief_of_staff_api_contract,
    chief_of_staff_dashboard,
    chief_of_staff_decision_loop,
    chief_of_staff_memory_handoff,
    chief_of_staff_plan,
    chief_of_staff_policy,
    chief_of_staff_role,
    cli_smoke_contracts,
    contextual_command_recommendation,
    command_first_to_chat_first_rollout,
    core_project_memory_loader,
    create_conversational_session,
    create_conversational_workflow,
    crm_launch_success_scenario,
    conversational_user_guide,
    dashboard_status_widgets,
    delegation_protocol_for,
    discover_chat_commands,
    dynamic_command_regression_tests,
    generated_command_permission_check,
    hermes_chat_page_contract,
    intent_evaluation_set,
    interactive_shell_contract,
    load_transcript_contract,
    memory_freshness_warnings,
    model_assisted_intent_fallback,
    multi_agent_trace,
    natural_language_skill_invocation,
    open_webui_integration_contract,
    parse_conversational_shortcut,
    plan_agent_assignment,
    project_identity_memory,
    project_switch_from_chat,
    project_switcher_contract,
    recommendation_panel,
    redact_memory_context,
    retrieve_decision_progress,
    review_queue_view,
    reviewer_approval_flow,
    rolling_conversation_memory,
    save_transcript_contract,
    skill_manifest_index,
    slash_command_aliases,
    streaming_status_updates,
    task_backlog_view,
    transcript_from_turns,
    transition_conversational_workflow,
    ui_test_contracts,
    validate_artifact_handoff,
    workflow_preview,
    workflow_recovery_contract,
    workflow_registry,
    working_context_builder,
)
from hermes_os_integration.phase_completion import complete_phases, completion_summary, phase_statuses, task_ids_for_phases


def test_phase_66_chat_cli_surface_contracts(tmp_path):
    session = create_conversational_session("s1", "operator", "alpha")
    transcript = transcript_from_turns(session, [{"role": "user", "content": "/plan build"}])

    assert chat_cli_contract()["command"] == "hermes chat"
    assert ask_command_contract("Build CRM")["dry_run"] is True
    assert interactive_shell_contract(feature_enabled=True)["mode"] == "chat-shell"
    assert parse_conversational_shortcut("/plan build")["route"] == "plan"
    assert streaming_status_updates(chief_of_staff_plan(ChatEnvelope(message="Build CRM")))[0]["event"] == "workflow.step"
    assert save_transcript_contract(transcript)["path"].endswith("s1.json")
    assert load_transcript_contract("s1")["required"] is False
    assert project_switch_from_chat("switch to alpha", ["alpha"])["project_id"] == "alpha"
    assert chat_error_envelope("unknown_intent", "Unknown")["ok"] is False
    assert chat_help_examples()[0].startswith("hermes ask")
    assert cli_smoke_contracts()[0]["command"] == "hermes chat --help"


def test_phase_67_chief_of_staff_contracts(tmp_path):
    envelope = ChatEnvelope(message="Build a CRM", project_id="alpha", session_id="s1")
    plan = chief_of_staff_plan(envelope, project_path=str(tmp_path))
    workflow = create_conversational_workflow(plan.route)

    assert chief_of_staff_role()["role"] == "Chief of Staff"
    assert chief_of_staff_policy("write_code")["allowed"] is False
    assert chief_of_staff_decision_loop(envelope, str(tmp_path))["loop"] == ["understand", "route", "plan", "delegate", "track", "report"]
    assert chief_of_staff_memory_handoff(plan)["session_id"] == "s1"
    assert approval_prompt_contract(plan)["required"] is True
    assert recommendation_panel(plan)["panel_id"] == "col-recommendations"
    assert workflow.status == "planned"


def test_phase_68_intent_routing_contracts():
    examples = intent_evaluation_set()

    assert examples[0]["expected"] == "new_project"
    assert model_assisted_intent_fallback("unclear")["enabled"] is False
    assert chief_of_staff_plan(ChatEnvelope(message="Continue this project")).route.intent == "existing_project"
    assert chief_of_staff_plan(ChatEnvelope(message="Research providers")).route.intent == "research"
    assert chief_of_staff_plan(ChatEnvelope(message="Review architecture")).route.intent == "architecture"
    assert chief_of_staff_plan(ChatEnvelope(message="Implement task-123")).route.intent == "task_work"
    assert chief_of_staff_plan(ChatEnvelope(message="???")).route.requires_clarification is True


def test_phase_69_workflow_engine_contracts():
    route = chief_of_staff_plan(ChatEnvelope(message="Build a CRM")).route
    workflow = create_conversational_workflow(route)
    running = transition_conversational_workflow(workflow, "running")
    checkpointed = __import__("hermes_os_integration.conversational", fromlist=["checkpoint_workflow"]).checkpoint_workflow(running, "grill-me", "artifact://grill")

    assert "new_project_launch" in workflow_registry()
    assert running.status == "running"
    assert checkpointed.checkpoints[0]["step_id"] == "grill-me"
    assert workflow_preview(workflow)["approvals"]
    assert workflow_recovery_contract(workflow, "resume")["valid"] is True
    assert create_conversational_workflow(chief_of_staff_plan(ChatEnvelope(message="Review architecture")).route).route == "architecture_review"


def test_phase_70_memory_layer_contracts(tmp_path):
    (tmp_path / "project.md").write_text("# Project\n", encoding="utf-8")
    session = create_conversational_session("s1", "operator", "alpha", goal="Goal", initiative="Initiative")
    memory = core_project_memory_loader(str(tmp_path))
    workflow = ConversationalWorkflow("wf", "research", "planned", [])

    assert project_identity_memory("alpha", str(tmp_path))["project_id"] == "alpha"
    assert "project.md" in memory["loaded"]
    assert working_context_builder(memory)["source_count"] == 1
    assert retrieve_decision_progress([{"topic": "db", "confidence": 0.9}], topic="db", min_confidence=0.8)
    assert rolling_conversation_memory([{"content": "What next?"}])["unresolved_questions"]
    assert active_session_store(session)["active_goal"] == "Goal"
    assert memory_freshness_warnings(memory)
    assert redact_memory_context({"api_token": "secret"})["api_token"] == "<redacted>"
    assert agent_context_package({"api_token": "secret"}, {"task": "t1"}, workflow)["project_memory"]["api_token"] == "<redacted>"


def test_phase_71_agent_hierarchy_delegation_contracts():
    plan = chief_of_staff_plan(ChatEnvelope(message="Build a CRM"))
    step = plan.steps[0]
    protocol = delegation_protocol_for(step, "planner")
    assignment = plan_agent_assignment(step, [{"role": "planner", "confidence": 0.95}])

    assert "management" in __import__("hermes_os_integration.conversational", fromlist=["conversational_agent_roles"]).conversational_agent_roles()
    assert protocol.agent_role == "planner"
    assert assignment.confidence == 0.95
    assert validate_artifact_handoff(protocol)["valid"] is True
    assert multi_agent_trace(plan.delegations)[0]["layer"] == "coordination"
    assert reviewer_approval_flow("artifact://x")["requires_reason"] is True
    assert agent_failure_fallback("engineer", reason="unavailable")["escalate"] is True
    assert active_agent_activity([assignment])["count"] == 1


def test_phase_72_chat_ui_dashboard_contracts():
    session = create_conversational_session("s1", "operator", "alpha", initiative="COL")
    plan = chief_of_staff_plan(ChatEnvelope(message="Build a CRM", project_id="alpha", session_id="s1"))
    workflow = create_conversational_workflow(plan.route)
    assignment = plan_agent_assignment(plan.steps[0], [{"role": "planner", "confidence": 0.9}])

    assert chief_of_staff_api_contract()["mode"] == "contract"
    assert open_webui_integration_contract(session)["streaming"] is True
    assert hermes_chat_page_contract(session, workflow)["persistent_conversation"] is True
    assert project_switcher_contract(["alpha"], active_project="alpha")["active_project"] == "alpha"
    assert task_backlog_view([{"project_id": "alpha"}], project_id="alpha")["count"] == 1
    assert review_queue_view([{"status": "pending"}])["actions"][0] == "approve"
    assert recommendation_panel(plan)["recommendations"]
    assert active_agent_activity([assignment])["assignments"][0]["agent_role"] == "planner"
    assert dashboard_status_widgets(session, open_tasks=2)["open_tasks"] == 2
    assert len(ui_test_contracts()) == 5


def test_phase_73_dynamic_commands_and_launch_success():
    plan = chief_of_staff_plan(ChatEnvelope(message="Build a CRM", project_id="crm"))

    assert "plan" in skill_manifest_index()["skills"]
    assert slash_command_aliases()["/plan"] == "plan"
    assert natural_language_skill_invocation("create plan for app")["skill"] == "plan"
    assert contextual_command_recommendation("crm", "new_project_launch", [])["command"] == "/grill-me"
    assert generated_command_permission_check("deploy app")["requires_approval"] is True
    assert chief_of_staff_dashboard(plan)["active_project"] == "crm"
    assert crm_launch_success_scenario()["workflow"] == "new_project_launch"
    assert conversational_user_guide()["sections"][0] == "ask"
    assert command_first_to_chat_first_rollout()["migration"]
    assert dynamic_command_regression_tests()[0]["case"] == "dynamic command recommendation"
    assert discover_chat_commands()["commands"]


def test_phase_66_to_73_completion_tracking(tmp_path):
    (tmp_path / ".hermes").mkdir()
    (tmp_path / "TASKS.md").write_text(
        "\n".join(f"- `task-{number:03d}`: Task {number}" for number in range(398, 478)),
        encoding="utf-8",
    )
    (tmp_path / ".hermes" / "tasks.json").write_text(json.dumps({"tasks": []}), encoding="utf-8")

    result = complete_phases(tmp_path, range(66, 74))
    statuses = phase_statuses(json.loads((tmp_path / ".hermes" / "tasks.json").read_text(encoding="utf-8")), range(66, 74))
    summary = completion_summary(tmp_path, range(66, 74))

    assert task_ids_for_phases([66, 73])[0] == "task-398"
    assert task_ids_for_phases([66, 73])[-1] == "task-477"
    assert result["completed"] == 80
    assert result["percent"] == 100
    assert summary["completed"] == 80
    assert all(status.percent == 100 for status in statuses)
