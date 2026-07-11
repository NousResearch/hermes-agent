"""Gate tests for hermes_cli/prune_plan.py — plan_worktree_prune().

Spec (must match exactly):
  plan_worktree_prune(worktrees: list[dict]) -> list[str]
  - Each entry MUST have keys: "id" (str), "merged" (bool), "dirty" (bool),
    "age_days" (int|float). A missing required key raises
    ValueError mentioning the entry's "id" when it is present, else the
    entry's index.
  - An entry is pruned iff: merged is True AND dirty is False AND
    age_days >= 1 (boundary: exactly 1 day IS pruned).
  - Unmerged worktrees are NEVER pruned, regardless of age or cleanliness.
  - Dirty worktrees are NEVER pruned, even when merged and old.
  - Extra/unknown keys are ignored.
  - Returns the pruned ids sorted lexicographically; input order must not
    matter and the input list must not be mutated.
"""
import pytest

from hermes_cli.prune_plan import plan_worktree_prune


def wt(id, merged, dirty, age_days, **extra):
    d = {"id": id, "merged": merged, "dirty": dirty, "age_days": age_days}
    d.update(extra)
    return d


def test_basic_prune_and_keep():
    plan = plan_worktree_prune([
        wt("b", True, False, 3),
        wt("a", True, False, 2.5),
        wt("keep-young", True, False, 0.5),
        wt("keep-dirty", True, True, 30),
        wt("keep-unmerged", False, False, 400),
    ])
    assert plan == ["a", "b"]


def test_boundary_exactly_one_day():
    assert plan_worktree_prune([wt("x", True, False, 1)]) == ["x"]
    assert plan_worktree_prune([wt("x", True, False, 0.999)]) == []


def test_unmerged_never_pruned_even_dirty_old():
    assert plan_worktree_prune([
        wt("u1", False, True, 999),
        wt("u2", False, False, 999),
    ]) == []


def test_empty_input():
    assert plan_worktree_prune([]) == []


def test_extra_keys_ignored():
    assert plan_worktree_prune([wt("x", True, False, 2, branch="b", note=1)]) == ["x"]


def test_missing_key_names_id():
    with pytest.raises(ValueError, match="wt-7"):
        plan_worktree_prune([{"id": "wt-7", "merged": True, "dirty": False}])


def test_missing_key_names_index_when_no_id():
    with pytest.raises(ValueError, match="0"):
        plan_worktree_prune([{"merged": True, "dirty": False, "age_days": 2}])


def test_input_not_mutated_and_order_independent():
    items = [wt("z", True, False, 2), wt("a", True, False, 2)]
    snapshot = [dict(i) for i in items]
    assert plan_worktree_prune(items) == ["a", "z"]
    assert items == snapshot
