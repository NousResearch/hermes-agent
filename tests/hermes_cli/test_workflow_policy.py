from __future__ import annotations

import pytest
import yaml

from hermes_cli.workflow.errors import WorkflowValidationError
from hermes_cli.workflow.policy import CANONICAL_ROLES, load_policy, policy_path_for_workspace


def test_missing_policy_returns_default_with_warning(tmp_path):
    result = load_policy(tmp_path)

    assert result.ok
    assert result.policy.using_default is True
    assert result.policy.path == tmp_path / ".hermes" / "workflow.yaml"
    assert result.policy.data["project"]["name"] == tmp_path.name
    assert result.policy.board == "default"
    assert [issue.code for issue in result.warnings] == ["policy_missing"]


def test_valid_policy_merges_with_defaults(tmp_path):
    policy_path = policy_path_for_workspace(tmp_path)
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "project": {"name": "PepChat", "board": "pepchat"},
                "roles": {"engineer": "pepchat-engineer", "integrator": None},
                "dag": {"max_parallel_engineers": 2},
            }
        ),
        encoding="utf-8",
    )

    result = load_policy(tmp_path)

    assert result.ok
    assert result.policy.using_default is False
    assert result.policy.data["project"] == {"name": "PepChat", "board": "pepchat"}
    assert result.policy.roles["engineer"] == "pepchat-engineer"
    assert result.policy.roles["integrator"] is None
    # Unspecified canonical roles come from the safe defaults.
    assert set(result.policy.roles) == set(CANONICAL_ROLES)
    assert result.policy.roles["architect"] == "architect"
    assert result.policy.data["dag"]["max_parallel_engineers"] == 2
    assert result.policy.data["worktrees"]["root"] == ".worktrees"


def test_invalid_yaml_returns_structured_error_and_default(tmp_path):
    policy_path = policy_path_for_workspace(tmp_path)
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text("version: [unterminated\n", encoding="utf-8")

    result = load_policy(tmp_path)

    assert not result.ok
    assert result.policy.using_default is True
    assert result.errors[0].code == "policy_invalid_yaml"
    assert str(policy_path) in result.errors[0].path


def test_strict_invalid_yaml_raises_validation_error(tmp_path):
    policy_path = policy_path_for_workspace(tmp_path)
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text("version: [unterminated\n", encoding="utf-8")

    with pytest.raises(WorkflowValidationError) as exc_info:
        load_policy(tmp_path, strict=True)

    assert exc_info.value.issues[0].code == "policy_invalid_yaml"


def test_unknown_gate_setting_is_structured_error(tmp_path):
    policy_path = policy_path_for_workspace(tmp_path)
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "review": {
                    "gates": {
                        "small": {"review": "robot-vibes"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = load_policy(tmp_path)

    assert not result.ok
    assert any(issue.code == "unknown_gate_setting" for issue in result.errors)
    assert any("review.gates.small.review" in issue.path for issue in result.errors)


def test_unsafe_worktree_root_is_rejected(tmp_path):
    policy_path = policy_path_for_workspace(tmp_path)
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(
        yaml.safe_dump({"version": 1, "worktrees": {"root": "../outside"}}),
        encoding="utf-8",
    )

    result = load_policy(tmp_path)

    assert not result.ok
    assert any(issue.code == "unsafe_worktree_root" for issue in result.errors)
