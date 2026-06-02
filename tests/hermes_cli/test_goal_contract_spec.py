"""Static checks for the G1 goal-contract vocabulary."""

from __future__ import annotations


def test_goal_contract_spec_required_values():
    from hermes_cli import goal_contract_spec as spec

    assert spec.GOAL_CONTRACT_CONTROL_FIELDS == frozenset(
        {
            "stop_after_status_report",
        }
    )
    assert spec.APPROVAL_STATES == frozenset(
        {
            "none",
            "active",
            "expired",
            "revoked",
            "completed",
            "blocked",
        }
    )
    assert spec.APPROVAL_SLICE_SOURCE_FIELDS == frozenset(
        {
            "created_by",
            "created_from",
            "raw_user_approval",
            "created_at",
        }
    )
    assert spec.CREATED_FROM_VALUES == frozenset(
        {
            "manual_command",
            "ui_form",
            "template",
        }
    )
    assert spec.PRESET_NAMES == frozenset(
        {
            "discussion-only",
            "inspection-only",
            "implement-slice",
            "commit-only",
            "local-smoke-test",
            "stop-state-only",
        }
    )
    assert spec.PRESET_CONTROL_FIELDS == {
        "stop-state-only": frozenset({"stop_after_status_report"}),
    }


def test_goal_contract_spec_supporting_vocabularies_are_canonical():
    from hermes_cli import goal_contract_spec as spec

    assert spec.ACTION_CATEGORIES == frozenset(
        {
            "discuss",
            "plan",
            "inspect_repo",
            "read_files",
            "search_files",
            "edit_files",
            "run_focused_tests",
            "run_broad_tests",
            "run_build",
            "run_lint",
            "run_dev_server",
            "browser_qa",
            "install_dependencies",
            "change_config",
            "touch_secrets",
            "commit",
            "push",
            "open_pr",
            "deploy",
            "restart_service",
            "public_bind",
            "oauth_connector",
            "external_network",
            "destructive_git",
        }
    )
    assert spec.CHECKPOINT_KINDS == frozenset(
        {
            "stop_after_status_report",
            "stop_after_plan",
            "stop_after_inspection_report",
            "stop_after_implementation_report",
            "stop_after_validation_report",
            "stop_after_local_commit_report",
            "stop_on_scope_expansion",
            "stop_on_validation_failure",
            "stop_on_unrelated_dirty_files",
            "stop_on_dependency_change_needed",
            "stop_on_restart_or_deploy_needed",
            "stop_on_user_message_conflict",
        }
    )
    assert spec.COMMIT_POLICY_VALUES
    assert spec.RESTART_POLICY_VALUES
    assert spec.DEPLOY_POLICY_VALUES
    assert spec.PUSH_POLICY_VALUES
