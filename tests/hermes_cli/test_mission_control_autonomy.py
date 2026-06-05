from __future__ import annotations

import dataclasses
from pathlib import Path


ALLOWED_WEBSITE_GENERATED = (
    "website/static/api/skills-meta.json",
    "website/static/api/skills.json",
)


def _envelope(**overrides):
    from hermes_cli.mission_control_autonomy import (
        ApprovalTier,
        LaneState,
        TaskControlEnvelopeModel,
    )

    payload = {
        "active_lane": "autonomous lane runner MVP implementation",
        "mode": "code/test only",
        "lane_state": LaneState.ACTIVE,
        "approval_tier": ApprovalTier.CODE_TEST,
        "repo_path": "/home/jenny/.hermes/hermes-context-routing-e1d-integration",
        "branch": "mission-control-os-stateful-foundation",
        "allowed_actions": ("read_files", "edit_files", "run_focused_tests", "py_compile", "diff_check", "secret_scan"),
        "forbidden_actions": ("network", "subprocess", "dashboard_wiring", "production_wiring", "commit"),
        "allowed_files": (
            "hermes_cli/mission_control_autonomy.py",
            "tests/hermes_cli/test_mission_control_autonomy.py",
        ),
        "allowed_start_gate_dirty_files": ALLOWED_WEBSITE_GENERATED,
        "focused_test_files": ("tests/hermes_cli/test_mission_control_autonomy.py",),
        "stop_condition": "stop before commit",
    }
    payload.update(overrides)
    return TaskControlEnvelopeModel(**payload)


def test_start_gate_allows_only_expected_repo_branch_and_allowed_dirty_files():
    from hermes_cli.mission_control_autonomy import validate_start_gate

    decision = validate_start_gate(
        _envelope(),
        repo_path="/home/jenny/.hermes/hermes-context-routing-e1d-integration",
        branch="mission-control-os-stateful-foundation",
        dirty_files=ALLOWED_WEBSITE_GENERATED,
    )

    assert decision.allowed is True
    assert decision.reason == "start_gate_passed"
    assert decision.approval_tier.value == "code_test"
    assert decision.violations == ()
    assert decision.evidence_cards[0].kind == "start_gate"
    assert decision.evidence_cards[0].inert_context_only is True
    assert decision.evidence_cards[0].authorizing is False


def test_start_gate_denies_unexpected_dirty_files_without_touching_filesystem():
    from hermes_cli.mission_control_autonomy import validate_start_gate

    decision = validate_start_gate(
        _envelope(),
        repo_path="/home/jenny/.hermes/hermes-context-routing-e1d-integration",
        branch="mission-control-os-stateful-foundation",
        dirty_files=ALLOWED_WEBSITE_GENERATED + ("hermes_cli/web_server.py",),
    )

    assert decision.allowed is False
    assert decision.reason == "start_gate_blocked"
    assert decision.violations == ("unexpected_dirty_file:hermes_cli/web_server.py",)


def test_start_gate_denies_wrong_repo_or_branch():
    from hermes_cli.mission_control_autonomy import validate_start_gate

    wrong_repo = validate_start_gate(
        _envelope(),
        repo_path="/tmp/other",
        branch="mission-control-os-stateful-foundation",
        dirty_files=(),
    )
    wrong_branch = validate_start_gate(
        _envelope(),
        repo_path="/home/jenny/.hermes/hermes-context-routing-e1d-integration",
        branch="other",
        dirty_files=(),
    )

    assert wrong_repo.allowed is False
    assert wrong_repo.violations == ("repo_path_mismatch",)
    assert wrong_branch.allowed is False
    assert wrong_branch.violations == ("branch_mismatch",)


def test_tool_decision_allows_scoped_edit_and_focused_validation_requests():
    from hermes_cli.mission_control_autonomy import ToolRequest, decide_tool_request

    edit_decision = decide_tool_request(
        _envelope(),
        ToolRequest(
            tool_name="apply_patch",
            action="edit_files",
            target_files=("hermes_cli/mission_control_autonomy.py",),
            writes=True,
        ),
    )
    test_decision = decide_tool_request(
        _envelope(),
        ToolRequest(
            tool_name="pytest",
            action="run_focused_tests",
            target_files=("tests/hermes_cli/test_mission_control_autonomy.py",),
            executes=True,
        ),
    )

    assert edit_decision.allowed is True
    assert edit_decision.reason == "tool_request_allowed"
    assert test_decision.allowed is True
    assert test_decision.reason == "tool_request_allowed"


def test_tool_decision_denies_forbidden_actions_and_out_of_scope_files():
    from hermes_cli.mission_control_autonomy import ToolRequest, decide_tool_request

    deploy = decide_tool_request(
        _envelope(),
        ToolRequest(tool_name="deploy", action="deploy", executes=True),
    )
    production_wiring = decide_tool_request(
        _envelope(),
        ToolRequest(
            tool_name="apply_patch",
            action="production_wiring",
            target_files=("run_agent.py",),
            writes=True,
        ),
    )
    outside_file = decide_tool_request(
        _envelope(),
        ToolRequest(
            tool_name="apply_patch",
            action="edit_files",
            target_files=("hermes_cli/web_server.py",),
            writes=True,
        ),
    )

    assert deploy.allowed is False
    assert deploy.violations == ("action_not_allowed:deploy",)
    assert production_wiring.allowed is False
    assert production_wiring.violations == (
        "forbidden_action:production_wiring",
        "write_outside_allowed_files:run_agent.py",
    )
    assert outside_file.allowed is False
    assert outside_file.violations == ("write_outside_allowed_files:hermes_cli/web_server.py",)


def test_tool_decision_denies_network_and_unfocused_tests():
    from hermes_cli.mission_control_autonomy import ToolRequest, decide_tool_request

    network = decide_tool_request(
        _envelope(),
        ToolRequest(tool_name="curl", action="network", uses_network=True),
    )
    broad_tests = decide_tool_request(
        _envelope(),
        ToolRequest(
            tool_name="pytest",
            action="run_focused_tests",
            target_files=("tests/hermes_cli/test_mission_control.py",),
            executes=True,
        ),
    )

    assert network.allowed is False
    assert network.violations == ("forbidden_action:network", "network_not_allowed")
    assert broad_tests.allowed is False
    assert broad_tests.violations == (
        "test_outside_focused_scope:tests/hermes_cli/test_mission_control.py",
    )


def test_autonomy_models_are_frozen_dataclasses_and_module_has_no_runtime_wiring_symbols():
    from hermes_cli import mission_control_autonomy as autonomy

    for model in (
        autonomy.TaskControlEnvelopeModel,
        autonomy.ToolRequest,
        autonomy.GuardDecision,
        autonomy.EvidenceCardModel,
    ):
        assert dataclasses.is_dataclass(model)
        assert model.__dataclass_params__.frozen is True

    source = Path("hermes_cli/mission_control_autonomy.py").read_text(encoding="utf-8")
    forbidden_symbols = {
        "subprocess",
        "urllib",
        "requests",
        "httpx",
        "socket",
        "Path(",
        ".read_text",
        ".write_text",
        "FastAPI",
        "APIRouter",
        "add_api_route",
        "run_agent",
        "handle_function_call",
    }
    for symbol in forbidden_symbols:
        assert symbol not in source


def test_stored_task_control_envelope_converts_to_inert_policy_model():
    from hermes_cli.mission_control_autonomy import (
        ApprovalTier,
        LaneState,
        task_control_envelope_model_from_record,
    )

    model = task_control_envelope_model_from_record(
        {
            "status": "active",
            "title": "Fallback lane",
            "mode": "implement-slice",
            "allowed_actions": ["read_files", "edit_files", "run_focused_tests"],
            "forbidden_actions": ["deploy", "restart_service"],
            "repo_context": {
                "path": "/repo",
                "branch": "feature-branch",
            },
            "lane_lock": {"active_lane": "Hermes OS autonomy read-model wiring"},
            "metadata": {
                "approval_tier": "code/test only",
                "allowed_files": ["hermes_cli/mission_control_autonomy.py"],
                "forbidden_files": ["website/static/api/skills.json"],
                "allowed_start_gate_dirty_files": ["website/static/api/skills.json"],
                "focused_test_files": ["tests/hermes_cli/test_mission_control_autonomy.py"],
                "stop_condition": "stop before commit",
            },
        }
    )

    assert model.active_lane == "Hermes OS autonomy read-model wiring"
    assert model.mode == "implement-slice"
    assert model.lane_state is LaneState.ACTIVE
    assert model.approval_tier is ApprovalTier.CODE_TEST
    assert model.repo_path == "/repo"
    assert model.branch == "feature-branch"
    assert model.allowed_actions == ("read_files", "edit_files", "run_focused_tests")
    assert model.forbidden_actions == ("deploy", "restart_service")
    assert model.allowed_files == ("hermes_cli/mission_control_autonomy.py",)
    assert model.forbidden_files == ("website/static/api/skills.json",)
    assert model.allowed_start_gate_dirty_files == ("website/static/api/skills.json",)
    assert model.focused_test_files == ("tests/hermes_cli/test_mission_control_autonomy.py",)
    assert model.stop_condition == "stop before commit"


def test_policy_read_summaries_are_decision_only_and_classify_next_action():
    from hermes_cli.mission_control_autonomy import (
        ApprovalTier,
        ToolRequest,
        decide_tool_request,
        next_action_decision_summary,
        summarize_guard_decision,
        task_control_envelope_model_from_record,
        validate_start_gate,
    )

    model = task_control_envelope_model_from_record(
        {
            "status": "active",
            "title": "Read-only lane",
            "mode": "inspection-only",
            "allowed_actions": ["read_files"],
            "forbidden_actions": ["deploy"],
            "repo_context": {"path": "/repo", "branch": "main"},
            "metadata": {"approval_tier": "read_only"},
        }
    )
    start_gate = validate_start_gate(model, repo_path="/repo", branch="main", dirty_files=())
    guard_summary = summarize_guard_decision(start_gate)

    assert guard_summary == {
        "allowed": True,
        "approval_tier": "read_only",
        "reason": "start_gate_passed",
        "violations": [],
        "execution_enabled": False,
        "trusted_for_execution": False,
        "inert_context_only": True,
    }
    assert next_action_decision_summary(model, start_gate)["kind"] == "auto"

    code_model = dataclasses.replace(model, approval_tier=ApprovalTier.CODE_TEST)
    code_start_gate = validate_start_gate(code_model, repo_path="/repo", branch="main", dirty_files=())
    assert next_action_decision_summary(code_model, code_start_gate)["kind"] == "one_click_approval"

    blocked = summarize_guard_decision(
        validate_start_gate(code_model, repo_path="/repo", branch="wrong", dirty_files=())
    )
    assert blocked["allowed"] is False
    assert blocked["approval_tier"] == "forbidden"
    assert next_action_decision_summary(code_model, blocked)["kind"] == "forbidden"

    tool_summary = summarize_guard_decision(
        decide_tool_request(
            code_model,
            ToolRequest(tool_name="deploy", action="deploy", executes=True),
        )
    )
    assert tool_summary["allowed"] is False
    assert tool_summary["execution_enabled"] is False
