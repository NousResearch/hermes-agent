import pytest

from agent.conversation_worktree_policy import (
    ConversationWorktreePolicyError,
    resolve_conversation_worktree_policy,
)


def test_top_level_policy_wins_over_legacy_desktop_block(tmp_path):
    cfg = {
        "conversation_worktree": {
            "enabled": True,
            "source_worktree": str(tmp_path / "new"),
            "worktree_root": str(tmp_path / "new-worktrees"),
        },
        "desktop": {
            "conversation_worktree": {
                "enabled": False,
                "source_worktree": str(tmp_path / "old"),
                "worktree_root": str(tmp_path / "old-worktrees"),
            }
        },
    }

    policy = resolve_conversation_worktree_policy(cfg)

    assert policy.enabled is True
    assert policy.source_worktree == (tmp_path / "new").resolve()
    assert policy.worktree_root == (tmp_path / "new-worktrees").resolve()
    assert policy.legacy_location is False


def test_legacy_desktop_policy_is_read_when_top_level_is_absent(tmp_path):
    cfg = {
        "desktop": {
            "conversation_worktree": {
                "enabled": True,
                "source_worktree": str(tmp_path),
                "worktree_root": str(tmp_path / "worktrees"),
            }
        }
    }

    policy = resolve_conversation_worktree_policy(cfg)

    assert policy.enabled is True
    assert policy.legacy_location is True


def test_default_only_top_level_sentinel_falls_back_to_legacy_policy(tmp_path):
    policy = resolve_conversation_worktree_policy(
        {
            "conversation_worktree": None,
            "desktop": {
                "conversation_worktree": {
                    "enabled": True,
                    "source_worktree": str(tmp_path / "stable"),
                    "worktree_root": str(tmp_path / "worktrees"),
                }
            },
        }
    )

    assert policy.enabled is True
    assert policy.legacy_location is True


@pytest.mark.parametrize(
    "config, expected_field",
    [
        ({"conversation_worktree": {"enabled": True}}, "source_worktree"),
        (
            {
                "conversation_worktree": {
                    "enabled": True,
                    "source_worktree": "/stable",
                }
            },
            "worktree_root",
        ),
    ],
)
def test_enabled_policy_requires_source_and_root(config, expected_field):
    with pytest.raises(ConversationWorktreePolicyError, match=expected_field):
        resolve_conversation_worktree_policy(config)


def test_enabled_policy_rejects_relative_paths(tmp_path):
    with pytest.raises(ConversationWorktreePolicyError, match="source_worktree"):
        resolve_conversation_worktree_policy(
            {
                "conversation_worktree": {
                    "enabled": True,
                    "source_worktree": "relative-source",
                    "worktree_root": str(tmp_path / "worktrees"),
                }
            }
        )


def test_enabled_policy_requires_retention_and_safe_values(tmp_path):
    with pytest.raises(ConversationWorktreePolicyError, match="retain_until_explicit_cleanup"):
        resolve_conversation_worktree_policy(
            {
                "conversation_worktree": {
                    "enabled": True,
                    "source_worktree": str(tmp_path / "stable"),
                    "worktree_root": str(tmp_path / "worktrees"),
                    "retain_until_explicit_cleanup": False,
                }
            }
        )

    with pytest.raises(ConversationWorktreePolicyError, match="branch_prefix"):
        resolve_conversation_worktree_policy(
            {
                "conversation_worktree": {
                    "enabled": True,
                    "source_worktree": str(tmp_path / "stable"),
                    "worktree_root": str(tmp_path / "worktrees"),
                    "branch_prefix": "hermes//session",
                }
            }
        )

    with pytest.raises(ConversationWorktreePolicyError, match="bootstrap_command"):
        resolve_conversation_worktree_policy(
            {
                "conversation_worktree": {
                    "enabled": True,
                    "source_worktree": str(tmp_path / "stable"),
                    "worktree_root": str(tmp_path / "worktrees"),
                    "bootstrap_command": "python3 bootstrap.py",
                }
            }
        )


def test_bootstrap_requires_non_empty_command_when_enabled(tmp_path):
    with pytest.raises(ConversationWorktreePolicyError, match="bootstrap_command"):
        resolve_conversation_worktree_policy(
            {
                "conversation_worktree": {
                    "enabled": True,
                    "source_worktree": str(tmp_path / "stable"),
                    "worktree_root": str(tmp_path / "worktrees"),
                    "bootstrap": True,
                    "bootstrap_command": [],
                }
            }
        )


def test_branch_prefix_rejects_invalid_ref_component(tmp_path):
    with pytest.raises(ConversationWorktreePolicyError, match="branch_prefix"):
        resolve_conversation_worktree_policy(
            {
                "conversation_worktree": {
                    "enabled": True,
                    "source_worktree": str(tmp_path / "stable"),
                    "worktree_root": str(tmp_path / "worktrees"),
                    "branch_prefix": "hermes/.session",
                }
            }
        )


@pytest.mark.parametrize("control_character", ["\x01", "\x7f"])
def test_branch_prefix_rejects_ascii_control_characters(tmp_path, control_character):
    with pytest.raises(ConversationWorktreePolicyError, match="branch_prefix"):
        resolve_conversation_worktree_policy(
            {
                "conversation_worktree": {
                    "enabled": True,
                    "source_worktree": str(tmp_path / "stable"),
                    "worktree_root": str(tmp_path / "worktrees"),
                    "branch_prefix": f"hermes/{control_character}session",
                }
            }
        )


@pytest.mark.parametrize("field, value", [("bootstrap_timeout", 0), ("create_timeout", float("inf"))])
def test_policy_rejects_non_positive_or_non_finite_timeouts(tmp_path, field, value):
    with pytest.raises(ConversationWorktreePolicyError, match=field):
        resolve_conversation_worktree_policy(
            {
                "conversation_worktree": {
                    "enabled": True,
                    "source_worktree": str(tmp_path / "stable"),
                    "worktree_root": str(tmp_path / "worktrees"),
                    field: value,
                }
            }
        )


def test_disabled_default_policy_needs_no_paths():
    policy = resolve_conversation_worktree_policy({})

    assert policy.enabled is False
    assert policy.source_worktree is None
    assert policy.worktree_root is None
    assert policy.retain_until_explicit_cleanup is True


def test_resolved_policy_is_immutable(tmp_path):
    policy = resolve_conversation_worktree_policy(
        {
            "conversation_worktree": {
                "enabled": True,
                "source_worktree": str(tmp_path / "stable"),
                "worktree_root": str(tmp_path / "worktrees"),
                "bootstrap": True,
                "bootstrap_command": ["python3", "bootstrap.py"],
                "bootstrap_timeout": 120,
                "create_timeout": 30,
            }
        }
    )

    assert policy.source_worktree == (tmp_path / "stable").resolve()
    assert policy.bootstrap_command == ("python3", "bootstrap.py")
    assert policy.bootstrap_timeout == 120.0
    assert policy.create_timeout == 30.0
    with pytest.raises(AttributeError):
        policy.enabled = False
