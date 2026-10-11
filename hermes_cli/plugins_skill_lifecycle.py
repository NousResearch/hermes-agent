"""Skill mutation lifecycle hook catalog shared by the plugin facade and callers.

Keeping this catalog separate from the large plugin loader makes the hook names a
small, importable contract while ``hermes_cli.plugins`` continues to re-export
the names used by plugins and tests.
"""

from __future__ import annotations

SKILL_MUTATION_ACTIONS = ("create", "edit", "patch", "write_file", "remove_file", "delete")
SKILL_MUTATION_GUARD_HOOKS = frozenset(
    f"pre_skill_{action}:guard" for action in SKILL_MUTATION_ACTIONS)
SKILL_MUTATION_PRE_HOOKS = frozenset(
    f"pre_skill_{action}" for action in SKILL_MUTATION_ACTIONS)
SKILL_MUTATION_POST_HOOKS = frozenset(
    f"post_skill_{action}" for action in SKILL_MUTATION_ACTIONS)
SKILL_MUTATION_HOOKS = (
    SKILL_MUTATION_GUARD_HOOKS | SKILL_MUTATION_PRE_HOOKS | SKILL_MUTATION_POST_HOOKS)
SKILL_MUTATION_SHELL_UNSUPPORTED_HOOKS = (
    SKILL_MUTATION_GUARD_HOOKS | SKILL_MUTATION_PRE_HOOKS)

__all__ = (
    "SKILL_MUTATION_ACTIONS", "SKILL_MUTATION_GUARD_HOOKS", "SKILL_MUTATION_PRE_HOOKS",
    "SKILL_MUTATION_POST_HOOKS", "SKILL_MUTATION_HOOKS", "SKILL_MUTATION_SHELL_UNSUPPORTED_HOOKS",
)
