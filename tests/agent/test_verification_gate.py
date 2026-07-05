"""Tests for multi-agent/post-task verification gate."""

from __future__ import annotations


def test_verification_gate_rejects_success_claim_without_required_path():
    from agent.verification_gate import verify_postconditions

    result = verify_postconditions(
        summary="created the requested file and tests passed",
        required_paths=["missing.py"],
        command_results=[],
        root=None,
    )

    assert result["passed"] is False
    assert result["failures"][0]["kind"] == "missing_path"


def test_verification_gate_accepts_existing_file_and_passing_command(tmp_path):
    from agent.verification_gate import verify_postconditions

    target = tmp_path / "out.py"
    target.write_text("print('ok')\n", encoding="utf-8")

    result = verify_postconditions(
        summary="done",
        required_paths=[str(target)],
        command_results=[{"command": "pytest", "exit_code": 0, "output": "1 passed"}],
        root=tmp_path,
    )

    assert result["passed"] is True
    assert result["evidence"]["paths_verified"] == [str(target)]
    assert result["evidence"]["commands_verified"][0]["status"] == "passed"


def test_verification_gate_rejects_path_traversal_outside_root(tmp_path):
    from agent.verification_gate import verify_postconditions

    outside = tmp_path.parent / "outside-proof.txt"
    outside.write_text("not in workspace", encoding="utf-8")

    result = verify_postconditions(
        summary="verified outside file",
        required_paths=["../outside-proof.txt", str(outside)],
        command_results=[],
        root=tmp_path,
    )

    assert result["passed"] is False
    assert [failure["kind"] for failure in result["failures"]] == ["path_outside_root", "path_outside_root"]
    assert result["evidence"]["paths_verified"] == []


def test_verification_gate_rejects_failing_command_even_with_confident_summary(tmp_path):
    from agent.verification_gate import verify_postconditions

    result = verify_postconditions(
        summary="all tests passed successfully",
        required_paths=[],
        command_results=[{"command": "pytest", "exit_code": 1, "output": "failed"}],
        root=tmp_path,
    )

    assert result["passed"] is False
    assert result["failures"][0]["kind"] == "command_failed"
