"""A delegated orchestrator is still the owning parent of its own children."""
import json

import pytest

from agent.subagent_lifecycle import bind_subagent_parent
from test_progress import setup, report


@pytest.mark.parametrize("first_turn", [False, True])
def test_nested_parent_receives_checkpoint_and_can_review(setup, first_turn):
    plugin, ctx, parent, child = setup
    from tools.delegate_tool_deadline import ReviewedDeadline
    from plugins.subagent_progress.supervision import Supervision
    child._delegate_reviewed_deadline = ReviewedDeadline(600)
    rid = report(plugin)["checkpoint_id"]
    parent._delegate_depth = 1
    parent.valid_tool_names = ["report_progress", "review_subagent_progress"]
    plugin.current_child = lambda: parent
    with bind_subagent_parent(parent):
        context = plugin.context(session_id=parent.session_id, platform="subagent", is_first_turn=first_turn)
    assert context is not None and "Input checked" in context["context"]
    if first_turn:
        assert "Use report_progress" in context["context"]
    assert plugin.context(session_id=parent.session_id, platform="subagent") is None
    result = json.loads(Supervision(plugin).review({
        "checkpoint_id": rid, "decision": "approve", "reason": "Evidence inspected",
        "evidence_checked": ["evidence.json"],
    }))
    assert result["success"] and child._delegate_reviewed_deadline.renewals == 1
