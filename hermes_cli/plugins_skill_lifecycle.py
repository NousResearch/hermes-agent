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

# Whole-batch decision point: fires once per ``skill_manage`` operations[] call
# BEFORE any mutation (before the write-approval gate, snapshots and locks), so
# a policy plugin sees every operation together — type, target path and full
# replacement/patch text — and decides on the combination, not per op. A valid
# approval is bound to the batch payload: the payload digest is sent TO the hook
# and an ``approve`` directive must echo it back, so an approval minted for one
# payload cannot be replayed against a modified batch or a subset.
SKILL_BATCH_HOOK = "pre_skill_batch"

# Directive skill hooks whose dispatch is bounded (and, when declared required
# via ``skill_policy.required``, fail-closed). Kept separate from the per-op
# catalog above: the per-op surface stays orthogonal and untouched.
SKILL_POLICY_HOOKS = frozenset({SKILL_BATCH_HOOK})

__all__ = (
    "SKILL_MUTATION_ACTIONS", "SKILL_MUTATION_GUARD_HOOKS", "SKILL_MUTATION_PRE_HOOKS",
    "SKILL_MUTATION_POST_HOOKS", "SKILL_MUTATION_HOOKS", "SKILL_MUTATION_SHELL_UNSUPPORTED_HOOKS",
    "SKILL_BATCH_HOOK", "SKILL_POLICY_HOOKS",
)
