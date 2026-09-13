"""Tests for AIAgent._summarize_background_review_actions.

Regression coverage for issue #14944: the background memory/skill review used
to re-surface tool results that were already present in the conversation
history before the review started (e.g. an earlier "Cron job '...' created.").
"""

import json

from run_agent import AIAgent


_summarize = AIAgent._summarize_background_review_actions


def _tool_msg(tool_call_id, payload):
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(payload),
    }


def _skill_call(call_id, operations):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "skill_manage",
                    "arguments": json.dumps({"operations": operations}),
                },
            }
        ],
    }


def test_skips_prior_tool_messages_by_tool_call_id():
    """Stale 'created' tool result from prior history must not be re-surfaced."""
    prior_payload = {"success": True, "message": "Cron job 'remind-me' created."}
    new_payload = {
        "success": True,
        "message": "Entry added",
        "target": "user",
    }

    snapshot = [
        {"role": "user", "content": "create a reminder"},
        _tool_msg("call_old", prior_payload),
        {"role": "assistant", "content": "done"},
    ]
    review_messages = list(snapshot) + [
        {"role": "user", "content": "<review prompt>"},
        _tool_msg("call_new", new_payload),
    ]

    actions = _summarize(review_messages, snapshot)

    assert "Cron job 'remind-me' created." not in actions
    assert "User profile updated" in actions


def test_includes_genuinely_new_actions():
    new_payload = {
        "success": True,
        "message": "Memory entry created.",
    }
    review_messages = [_tool_msg("call_new", new_payload)]

    actions = _summarize(review_messages, prior_snapshot=[])

    assert actions == ["Memory entry created."]


def test_falls_back_to_content_equality_when_tool_call_id_missing():
    """If a tool message has no tool_call_id, match prior entries by content."""
    payload = {"success": True, "message": "Cron job 'X' created."}
    raw = json.dumps(payload)
    prior_msg = {"role": "tool", "content": raw}  # no tool_call_id
    review_messages = [
        {"role": "tool", "content": raw},  # same content -> stale, skip
        _tool_msg("call_new", {"success": True, "message": "Skill created."}),
    ]

    actions = _summarize(review_messages, [prior_msg])

    assert "Cron job 'X' created." not in actions
    assert "Skill created." in actions



def test_handles_non_json_tool_content_gracefully():
    review_messages = [
        {"role": "tool", "tool_call_id": "x", "content": "not-json"},
        _tool_msg("call_y", {"success": True, "message": "Memory updated."}),
    ]

    actions = _summarize(review_messages, [])

    assert actions == ["Memory updated."]


def test_empty_inputs():
    assert _summarize([], []) == []
    assert _summarize(None, None) == []



def test_removed_or_replaced_relabels_by_target():
    review_messages = [
        _tool_msg(
            "c1",
            {"success": True, "message": "Entry removed.", "target": "user"},
        ),
        _tool_msg(
            "c2",
            {"success": True, "message": "Entry replaced.", "target": "memory"},
        ),
    ]

    actions = _summarize(review_messages, [])

    assert "User profile updated" in actions
    assert "Memory updated" in actions


def test_skill_batch_without_message_surfaces_ops_summary():
    """Batch skill_manage results carry operations_applied/results and no message —
    the non-verbose summary must still surface the writes (#104506)."""
    operations = [
        {"action": "patch", "name": "deploy", "old_string": "a", "new_string": "b"},
        {
            "action": "write_file",
            "name": "deploy",
            "file_path": "references/api.md",
            "file_content": "x",
        },
    ]
    batch_result = {
        "success": True,
        "operations_applied": 2,
        "results": [
            {"name": "deploy", "action": "patch", "success": True},
            {"name": "deploy", "action": "write_file", "success": True},
        ],
    }
    review_messages = [
        _skill_call("call_b1", operations),
        _tool_msg("call_b1", batch_result),
    ]

    actions = _summarize(review_messages, [])

    assert actions == [
        "Skill 'deploy' patched",
        "Skill 'deploy' written (references/api.md)",
    ]


def test_skill_batch_without_message_or_details_falls_back():
    """A message-less skill result whose call arguments are unavailable still
    surfaces a generic line instead of vanishing (#104506)."""
    batch_result = {"success": True, "operations_applied": 1, "results": []}
    review_messages = [
        _skill_call(
            "call_b2",
            [
                {
                    "action": "patch",
                    "name": "deploy",
                    "old_string": "a",
                    "new_string": "b",
                }
            ],
        ),
        _tool_msg("call_b2", batch_result),
    ]
    # Simulate unparsable call arguments: detail missing for this tool_call_id.
    review_messages[0]["tool_calls"][0]["function"]["arguments"] = "{not json"

    actions = _summarize(review_messages, [])

    assert actions == ["Skill updated"]


def test_staged_skill_write_is_not_surfaced():
    """A write staged for approval is not an applied action — it must stay silent
    even though its response says success=True (#104506)."""
    staged = {
        "success": True,
        "staged": True,
        "pending_id": "p1",
        "gist": "batch(1 ops: patch) on deploy",
        "message": (
            "Staged for approval (skills.write_approval is on). "
            "Not yet saved — review with /skills pending."
        ),
    }
    operations = [
        {"action": "patch", "name": "deploy", "old_string": "a", "new_string": "b"}
    ]
    review_messages = [_skill_call("call_s1", operations), _tool_msg("call_s1", staged)]

    actions = _summarize(review_messages, [])

    assert actions == []
