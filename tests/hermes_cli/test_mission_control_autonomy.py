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
