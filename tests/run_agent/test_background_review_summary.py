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


def test_staged_skill_write_surfaced_as_staged_not_applied():
    """hc-376: a staged (not yet applied) background-review skill write must be
    reported honestly — never as a landed create/patch."""
    tcid = "call_stage"
    assistant = {
        "role": "assistant",
        "tool_calls": [{
            "id": tcid,
            "function": {
                "name": "skill_manage",
                "arguments": json.dumps({"action": "patch", "name": "douyin-batch"}),
            },
        }],
    }
    staged = _tool_msg(tcid, {
        "success": True,
        "staged": True,
        "pending_id": "ab12cd34",
        "gist": "patch douyin-batch",
        "message": (
            "Staged for approval — background self-improvement skill writes "
            "are gated. Not yet saved — review with /skills pending."
        ),
    })

    actions = _summarize([assistant, staged], prior_snapshot=[])

    assert len(actions) == 1
    assert "staged for approval" in actions[0].lower()
    assert "douyin-batch" in actions[0]
    # Must NOT claim the write landed.
    assert "patched" not in actions[0].lower()


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


def test_ignores_failed_tool_results():
    bad = {"success": False, "message": "something created but failed"}
    review_messages = [_tool_msg("call_new", bad)]

    actions = _summarize(review_messages, [])

    assert actions == []


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


def test_added_message_relabels_by_target():
    review_messages = [
        _tool_msg(
            "c1",
            {"success": True, "message": "Entry added to store.", "target": "memory"},
        )
    ]

    actions = _summarize(review_messages, [])

    assert actions == ["Memory updated"]


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
