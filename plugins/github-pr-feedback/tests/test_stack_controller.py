import pytest

from github_pr_feedback.stack import StackEntry
from github_pr_feedback.stack_controller import _ordered


def test_stack_entries_are_ordered_from_base_not_input_order():
    entries = (
        StackEntry("codex/child", "codex/parent", "Child", ""),
        StackEntry("codex/parent", "stable", "Parent", ""),
        StackEntry("codex/grandchild", "codex/child", "Grandchild", ""),
    )
    assert [entry.branch for entry in _ordered(entries, "stable")] == [
        "codex/parent",
        "codex/child",
        "codex/grandchild",
    ]


def test_unreachable_stack_is_rejected():
    entries = (StackEntry("codex/child", "codex/parent", "Child", ""),)
    with pytest.raises(ValueError, match="no reachable"):
        _ordered(entries, "stable")
