"""Regression test for issue #75394 Mode C: empty/whitespace-only action in summary.

The bug: `actions` list can contain entries like ' updated' or 'Skill create'
where the skill_name or label is empty. After the dedup-join at
agent/background_review.py:931 the summary prints as:
  "💾 Self-improvement review:  updated"   (empty label + action)
  "💾 Self-improvement review: Skill create"  (empty skill_name)

Fix: filter empty/whitespace-only entries before the dedup-join at the
background review callback site in agent/background_review.py.
"""

import inspect

from run_agent import AIAgent
from agent.background_review import summarize_background_review_actions


# 1. The summarize function still legitimately produces "Skill create" /
#    " updated" in degenerate cases (those are caller-side bugs that the
#    downstream caller now strips). Confirm the bug reproduces through
#    summarize_background_review_actions alone.
def test_summarize_can_emit_empty_skill_action():
    """Confirm the upstream source can produce 'Skill create' (degenerate path).

    This is the symptom the join-site fix has to clean up. The test reproduces
    the exact scenario the user reported: a skill_manage call with empty
    name and no _change, no message content -> summarize returns 'Skill create'.
    """
    import json

    def _assistant_tool_call(tcid, name, arguments):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tcid,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }

    def _tool_msg(tcid, payload):
        return {
            "role": "tool",
            "tool_call_id": tcid,
            "content": json.dumps(payload),
        }

    msgs = [
        _assistant_tool_call(
            "c1", "skill_manage", {"action": "create", "name": ""}
        ),
        _tool_msg(
            "c1",
            {"success": True, "message": "", "_change": {}},
        ),
    ]
    actions = summarize_background_review_actions(
        msgs, prior_snapshot=[], notification_mode="verbose"
    )
    # The summary function can produce 'Skill create' (empty skill name).
    # This is acceptable; the fix is at the join site, not in summarize.
    assert "Skill create" in actions, (
        "Test premise broken: 'Skill create' no longer emitted by summarize. "
        "If this happens the underlying bug may have been fixed at the source "
        "— review and consider deleting this test."
    )


# 2. The fix lives at the join site. Confirm the source contains the filter.
def test_join_site_filters_empty_actions():
    """The dedup-join at the marker-print site must filter empty entries."""
    import agent.background_review as br_module

    src = inspect.getsource(br_module)
    # The fix must filter actions where `a` is empty OR whitespace-only
    # BEFORE the dict.fromkeys dedup. The pattern matches the canonical
    # fix from the issue: `a for a in actions if a and a.strip()`.
    assert (
        "for a in actions if a and a.strip()" in src
    ), (
        "Fix missing: the dedup-join at agent/background_review.py "
        "must filter empty/whitespace-only entries. The exact pattern "
        "`for a in actions if a and a.strip()` is expected."
    )


def test_valid_skill_action_still_emitted_by_summarize():
    """A normal skill create with name + description must still appear."""
    import json

    def _assistant_tool_call(tcid, name, arguments):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tcid,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }

    def _tool_msg(tcid, payload):
        return {
            "role": "tool",
            "tool_call_id": tcid,
            "content": json.dumps(payload),
        }

    msgs = [
        _assistant_tool_call(
            "c3",
            "skill_manage",
            {"action": "create", "name": "my-skill"},
        ),
        _tool_msg(
            "c3",
            {
                "success": True,
                "message": "",
                "_change": {"description": "useful skill"},
            },
        ),
    ]
    actions = summarize_background_review_actions(
        msgs, prior_snapshot=[], notification_mode="verbose"
    )
    assert any("my-skill" in a for a in actions), (
        f"valid skill action lost: {actions!r}"
    )


def test_valid_memory_action_still_emitted_by_summarize():
    """A normal memory add must still produce 'Memory ➕ <preview>'."""
    import json

    def _assistant_tool_call(tcid, name, arguments):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tcid,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }

    def _tool_msg(tcid, payload):
        return {
            "role": "tool",
            "tool_call_id": tcid,
            "content": json.dumps(payload),
        }

    msgs = [
        _assistant_tool_call(
            "c4", "memory", {"action": "add", "content": "hello world"}
        ),
        _tool_msg(
            "c4",
            {"success": True, "message": "Entry added.", "target": "memory"},
        ),
    ]
    actions = summarize_background_review_actions(
        msgs, prior_snapshot=[], notification_mode="verbose"
    )
    assert any("Memory" in a and "hello world" in a for a in actions), (
        f"valid memory action lost: {actions!r}"
    )