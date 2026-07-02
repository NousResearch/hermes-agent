from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_max_turn_closeout_state_contains_latest_child_and_bounded_resume_prompt(hermes_home):
    from hermes_cli.closeout_state import write_closeout_state
    from hermes_cli.closure_artifacts import read_closure_artifact

    path = write_closeout_state(
        session_id="root-session",
        latest_session_id="child-session",
        parent_lineage=["root-session", "compressed-session", "child-session"],
        task_id="t_paid_live",
        final_response="PR is not merged yet; CI not checked.",
        pr_url="https://github.com/example/repo/pull/123",
        head_sha="abc123def456",
        merge_status="not_merged",
        ci_status="not_checked",
        remaining_closeout_tasks=["rerun bounded review", "watch post-main CI"],
        changed_files=["hermes_cli/closeout_state.py"],
    )

    data = read_closure_artifact(path)

    assert data["status"] == "recoverable_incomplete"
    assert data["session_id"] == "root-session"
    assert data["latest_session_id"] == "child-session"
    assert data["parent_lineage"] == ["root-session", "compressed-session", "child-session"]
    assert data["pr_url"] == "https://github.com/example/repo/pull/123"
    assert data["head_sha"] == "abc123def456"
    assert data["merge_status"] == "not_merged"
    assert data["ci_status"] == "not_checked"
    assert data["remaining_closeout_tasks"] == ["rerun bounded review", "watch post-main CI"]
    assert "Do not perform live provider" in data["safe_bounded_resume_prompt"]
    assert "bloated parent history" in data["safe_bounded_resume_prompt"]


def test_closeout_final_response_with_unmerged_pr_is_recoverable_incomplete():
    from hermes_cli.closeout_state import classify_closeout_response

    verdict = classify_closeout_response(
        "Implementation done, but PR not merged and CI not checked.",
        pr_url="https://github.com/example/repo/pull/123",
        merge_status="not_merged",
        ci_status="not_checked",
    )

    assert verdict["status"] == "recoverable_incomplete"
    assert "pr_not_merged" in verdict["reasons"]
    assert "ci_not_checked" in verdict["reasons"]


def test_invalid_zero_budget_review_child_blocks_final_success(hermes_home):
    from hermes_cli.closeout_state import write_closeout_state
    from hermes_cli.closure_artifacts import read_closure_artifact

    invalid_child = {
        "task_index": 0,
        "child_session_id": None,
        "review_evidence_status": "blocked_zero_budget",
        "goal_preview": "Review paid-live diff",
    }
    path = write_closeout_state(
        session_id="root-session",
        latest_session_id="child-session",
        task_id="t_paid_live",
        final_response="All done.",
        invalid_review_children=[invalid_child],
        remaining_closeout_tasks=[],
    )
    data = read_closure_artifact(path)

    assert data["status"] == "recoverable_incomplete"
    assert data["invalid_review_children"][0]["review_evidence_status"] == "blocked_zero_budget"
    assert "invalid_review_child" in data["closeout_reasons"]


def test_runtime_closeout_resume_command_targets_latest_child(hermes_home, capsys):
    from hermes_cli.closeout_state import write_closeout_state
    from hermes_cli.runtime_cli import _cmd_closeout_resume_command, _cmd_closeout_status

    write_closeout_state(
        session_id="root-session",
        latest_session_id="child-session",
        task_id="t_paid_live",
        final_response="PR not merged.",
        remaining_closeout_tasks=["finish closeout"],
    )

    status_rc = _cmd_closeout_status(
        SimpleNamespace(session="root-session", task_id=None, json=False)
    )
    status_out = capsys.readouterr().out
    assert status_rc == 0
    assert "status=recoverable_incomplete" in status_out

    resume_rc = _cmd_closeout_resume_command(
        SimpleNamespace(session="root-session", task_id=None, hermes_command="hermes")
    )
    resume_out = capsys.readouterr().out

    assert resume_rc == 0
    assert "--resume child-session" in resume_out
    assert "root-session" not in resume_out.split("--resume", 1)[1].split("--max-turns", 1)[0]
    assert "Do not perform live provider" in resume_out


def test_resume_command_uses_powershell_literal_query_quote():
    from hermes_cli.closeout_state import build_closeout_resume_command

    command = build_closeout_resume_command(
        {
            "session_id": "root-session",
            "latest_session_id": "child-session",
            "safe_bounded_resume_prompt": "Check $env:TOKEN, \"quoted\", and don't expand anything.",
        }
    )

    assert '--query "' not in command
    assert "--query 'Check $env:TOKEN, \"quoted\", and don''t expand anything.'" in command


def test_ci_command_builder_uses_commit_filter():
    from hermes_cli.closeout_state import build_commit_ci_check_command

    command = build_commit_ci_check_command("abc123def456")

    assert command == ["gh", "run", "list", "--commit", "abc123def456", "--limit", "20"]
    assert ["gh", "run", "list", "sha", "abc123def456"] != command


def test_post_merge_main_ci_required_before_complete():
    from hermes_cli.closeout_state import classify_closeout_response

    pending = classify_closeout_response(
        "PR merged.",
        merge_status="merged",
        ci_status="not_checked",
    )
    green = classify_closeout_response(
        "PR merged and main CI green.",
        merge_status="merged",
        ci_status="success",
    )

    assert pending["status"] == "recoverable_incomplete"
    assert "post_main_ci_not_green" in pending["reasons"]
    assert green["status"] == "complete_candidate"
