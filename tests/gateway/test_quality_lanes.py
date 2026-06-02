from __future__ import annotations

from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_context, build_session_context_prompt
from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE


def test_quality_lane_section_required_for_code_change_report():
    from gateway.quality_lanes import (
        require_quality_lane_section,
        validate_quality_lane_section,
    )

    section = require_quality_lane_section(
        "Make code changes and run tests before the final report.",
        verification_summary="pytest tests/gateway/test_quality_lanes.py",
    )

    assert "## Quality lanes" in section
    assert "Task classification: high-rigor" in section
    assert "Required lanes:" in section
    assert "Implementation lane result:" in section
    assert "Review lane result:" in section
    assert "Verification lane result:" in section
    assert "Safety lane result:" in section
    assert "Remaining risks:" in section
    assert validate_quality_lane_section(section)["valid"] is True


def test_quality_lane_section_required_for_restart_report():
    from gateway.quality_lanes import require_quality_lane_section

    section = require_quality_lane_section(
        "Verify restart and deployment runtime state.",
        verification_summary="systemctl --user show hermes-gateway.service",
    )

    assert "Deployment/runtime lane result:" in section
    assert "restart required/performed/not performed" in section


def test_quality_lane_fallback_when_subagent_unavailable():
    from gateway.quality_lanes import require_quality_lane_section

    section = require_quality_lane_section(
        "Clean up a repo worktree.",
        subagent_available=False,
        subagent_invoked=False,
    )

    assert "real subagent used: no" in section
    assert "Subagent unavailable/not invoked; checklist fallback used." in section


def test_no_claim_real_subagent_without_delegate_execution():
    from gateway.quality_lanes import require_quality_lane_section

    section = require_quality_lane_section(
        "Review and commit code.",
        subagent_available=True,
        subagent_invoked=False,
    )

    assert "real subagent used: no" in section
    assert "real subagent used: yes" not in section


def test_goal_task_receives_quality_lane_requirement():
    prompt = CONTINUATION_PROMPT_TEMPLATE.format(goal="make code changes and verify them")

    assert "Quality lanes" in prompt
    assert "Task classification" in prompt
    assert "Subagent unavailable/not invoked; checklist fallback used." in prompt


def test_session_prompt_includes_enforceable_quality_lane_requirement():
    source = SessionSource(platform=Platform.DISCORD, chat_id="channel-1")
    ctx = build_session_context(source, GatewayConfig())

    prompt = build_session_context_prompt(ctx)

    assert "## Quality Lanes for High-Rigor Work" in prompt
    assert "Quality lanes section is required in final reports" in prompt
    assert "Do not claim real subagents ran unless delegation actually ran" in prompt


def test_recovery_report_includes_quality_lane_requirement():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._build_repo_identity_guard = lambda _record: {
        "ok": True,
        "lines": ["Repo identity guard passed"],
        "repo_path": "/tmp/repo",
        "branch": "main",
        "head": "abc123",
    }
    record = SimpleNamespace(
        task_summary="restart/deployment verification",
        command="systemctl status",
        expected_commit=None,
        final_report_path=None,
    )

    report = runner._build_active_execute_recovery_report(record, "not_found", None)

    assert "## Quality lanes" in report
    assert "Task classification: high-rigor" in report
    assert "Deployment/runtime lane result:" in report
