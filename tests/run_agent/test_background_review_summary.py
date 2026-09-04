"""Tests for AIAgent._summarize_background_review_actions.

Regression coverage for issue #14944: the background memory/skill review used
to re-surface tool results that were already present in the conversation
history before the review started (e.g. an earlier "Cron job '...' created.").
"""

import json

from agent.background_review import summarize_background_review_actions
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




def _skill_batch_review():
    """Batch skill_manage call: operations list, no top-level action."""
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_skill_batch",
                    "function": {
                        "name": "skill_manage",
                        "arguments": json.dumps(
                            {
                                "operations": [
                                    {"action": "create", "name": "alpha", "content": "A new skill body"},
                                    {
                                        "action": "patch",
                                        "name": "beta",
                                        "old_string": "old text",
                                        "new_string": "new text",
                                    },
                                    {"action": "write_file", "name": "alpha", "file_path": "references/x.md", "file_content": "data"},
                                ]
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_skill_batch",
            "content": json.dumps(
                {
                    "success": True,
                    "message": "batch(3 ops: create, patch, write_file) on alpha, beta",
                }
            ),
        },
    ]


def test_skill_batch_renders_per_op_lines():
    """Issue #102853: a batch skill_manage call must render per-op previews,
    never the empty-action "Skill " line."""
    verbose = summarize_background_review_actions(
        _skill_batch_review(), [], notification_mode="verbose"
    )
    assert verbose, "expected at least one rendered line"
    joined = "\n".join(verbose)
    assert "Skill 'alpha' created" in joined
    assert "Skill 'beta' patched" in joined
    assert "Skill 'alpha' wrote file" in joined
    assert not any(line == "Skill " or line.rstrip() == "Skill" for line in verbose), verbose
    assert not any("Skill ?" in line for line in verbose), verbose


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
