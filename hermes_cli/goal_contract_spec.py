"""Static G1 vocabulary for goal contracts and approval slices.

This module is intentionally data-only. It must not import Hermes runtime
modules or change runtime goal, approval, CLI, gateway, persistence, or UI
behavior.
"""

from __future__ import annotations

from typing import Literal


GOAL_CONTRACT_CONTROL_FIELDS = frozenset(
    {
        "stop_after_status_report",
    }
)

APPROVAL_STATES = frozenset(
    {
        "none",
        "active",
        "expired",
        "revoked",
        "completed",
        "blocked",
    }
)
ApprovalState = Literal["none", "active", "expired", "revoked", "completed", "blocked"]

APPROVAL_SLICE_SOURCE_FIELDS = frozenset(
    {
        "created_by",
        "created_from",
        "raw_user_approval",
        "created_at",
    }
)

CREATED_FROM_VALUES = frozenset(
    {
        "manual_command",
        "ui_form",
        "template",
    }
)
CreatedFrom = Literal["manual_command", "ui_form", "template"]

PRESET_NAMES = frozenset(
    {
        "discussion-only",
        "inspection-only",
        "implement-slice",
        "commit-only",
        "local-smoke-test",
        "stop-state-only",
    }
)
PRESET_CONTROL_FIELDS = {
    "stop-state-only": frozenset({"stop_after_status_report"}),
}
PresetName = Literal[
    "discussion-only",
    "inspection-only",
    "implement-slice",
    "commit-only",
    "local-smoke-test",
    "stop-state-only",
]

ACTION_CATEGORIES = frozenset(
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
ActionCategory = Literal[
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
]

CHECKPOINT_KINDS = frozenset(
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
CheckpointKind = Literal[
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
]

COMMIT_POLICY_VALUES = frozenset({"forbidden", "allowed", "requires_approval"})
RESTART_POLICY_VALUES = frozenset({"forbidden", "allowed", "requires_approval"})
DEPLOY_POLICY_VALUES = frozenset({"forbidden", "allowed", "requires_approval"})
PUSH_POLICY_VALUES = frozenset({"forbidden", "allowed", "requires_approval"})

PolicyValue = Literal["forbidden", "allowed", "requires_approval"]

__all__ = [
    "ACTION_CATEGORIES",
    "APPROVAL_SLICE_SOURCE_FIELDS",
    "APPROVAL_STATES",
    "CHECKPOINT_KINDS",
    "COMMIT_POLICY_VALUES",
    "CREATED_FROM_VALUES",
    "DEPLOY_POLICY_VALUES",
    "GOAL_CONTRACT_CONTROL_FIELDS",
    "PRESET_CONTROL_FIELDS",
    "PRESET_NAMES",
    "PUSH_POLICY_VALUES",
    "RESTART_POLICY_VALUES",
    "ActionCategory",
    "ApprovalState",
    "CheckpointKind",
    "CreatedFrom",
    "PolicyValue",
    "PresetName",
]
