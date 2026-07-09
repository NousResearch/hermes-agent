"""Usage-guard compact handoff contract tests."""

from __future__ import annotations

import json


def test_compact_handoff_packet_contains_required_fields_without_secrets():
    from hermes_cli.usage_guard import (
        REQUIRED_COMPACT_HANDOFF_FIELDS,
        build_compact_handoff_packet,
        validate_compact_handoff_packet,
    )

    packet = build_compact_handoff_packet(
        task_id="task-123",
        phase="fix-tests",
        touched_files=["hermes_cli/usage_guard.py"],
        recent_diff_summary="Added bounded usage guard helpers.",
        failing_tests=["tests/hermes_cli/test_usage_guard.py"],
        missing_symbols=["usage_guard_active"],
        blocked_commands=["rg --files"],
        declared_artifacts=[],
        next_small_step="Patch one narrow failing assertion.",
        must_not_repeat=["Do not run full-repo dumps."],
    )

    assert set(REQUIRED_COMPACT_HANDOFF_FIELDS).issubset(packet)
    assert validate_compact_handoff_packet(packet) == []
    assert "sk-proj-" not in json.dumps(packet)


def test_compact_handoff_packet_rejects_obvious_secret_values():
    from hermes_cli.usage_guard import validate_compact_handoff_packet

    packet = {
        "task_id": "task-123",
        "phase": "fix",
        "touched_files": [],
        "recent_diff_summary": "token sk-proj-abcdefghijklmnopqrstuvwxyz",
        "failing_tests": [],
        "missing_symbols": [],
        "blocked_commands": [],
        "declared_artifacts": [],
        "next_small_step": "continue",
        "must_not_repeat": [],
    }

    errors = validate_compact_handoff_packet(packet)

    assert any("secret" in error.lower() for error in errors)


def test_no_tool_spin_requires_db_and_runtime_log_evidence():
    from hermes_cli.usage_guard import NoToolSpinEvidence, classify_no_tool_spin

    db_only = classify_no_tool_spin(
        NoToolSpinEvidence(
            api_call_count=5,
            db_tool_call_count=0,
            runtime_tool_activity_seen=True,
            same_session_resume_attempted=True,
        )
    )
    assert db_only.is_no_tool_spin is False
    assert "runtime logs show tool activity" in db_only.reason

    both = classify_no_tool_spin(
        NoToolSpinEvidence(
            api_call_count=5,
            db_tool_call_count=0,
            runtime_tool_activity_seen=False,
            same_session_resume_attempted=True,
        )
    )
    assert both.is_no_tool_spin is True


def test_compact_instruction_after_patch_names_required_packet_fields():
    from hermes_cli.usage_guard import (
        REQUIRED_COMPACT_HANDOFF_FIELDS,
        compact_instruction_after_code_patch,
    )

    instruction = compact_instruction_after_code_patch(
        task_id="task-123",
        touched_files=["a.py", "b.py"],
    )

    assert "patch one narrow issue" in instruction
    assert "compact handoff packet" in instruction
    for field in REQUIRED_COMPACT_HANDOFF_FIELDS:
        assert field in instruction


def test_usage_guard_blocks_actual_broad_read_file_tool(tmp_path):
    from hermes_cli.usage_guard import (
        activate_usage_guard_after_warning,
        reset_usage_guard_for_tests,
    )
    from tools.file_tools import read_file_tool

    target = tmp_path / "large.py"
    target.write_text("\n".join(f"line {i}" for i in range(300)), encoding="utf-8")
    try:
        activate_usage_guard_after_warning(task_id="task-read", reason="test")

        result = json.loads(
            read_file_tool(str(target), offset=1, limit=500, task_id="task-read")
        )

        assert result["usage_guard"] == "active"
        assert "broad read_file" in result["error"]
    finally:
        reset_usage_guard_for_tests()


def test_usage_guard_blocks_actual_broad_search_tool():
    from hermes_cli.usage_guard import (
        activate_usage_guard_after_warning,
        reset_usage_guard_for_tests,
    )
    from tools.file_tools import search_tool

    try:
        activate_usage_guard_after_warning(task_id="task-search", reason="test")

        result = json.loads(
            search_tool(
                pattern="*",
                target="files",
                path=".",
                limit=50,
                task_id="task-search",
            )
        )

        assert result["usage_guard"] == "active"
        assert "broad repository searches" in result["error"]
    finally:
        reset_usage_guard_for_tests()


def test_usage_guard_blocks_actual_broad_terminal_command():
    from hermes_cli.usage_guard import (
        activate_usage_guard_after_warning,
        reset_usage_guard_for_tests,
    )
    from tools.terminal_tool import terminal_tool

    try:
        activate_usage_guard_after_warning(task_id="task-terminal", reason="test")

        result = json.loads(
            terminal_tool("find .", task_id="task-terminal", timeout=5)
        )

        assert result["usage_guard"] == "active"
        assert result["exit_code"] == -1
        assert "broad terminal enumeration" in result["error"]
    finally:
        reset_usage_guard_for_tests()
