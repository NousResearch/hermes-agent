"""Tests for AIAgent._summarize_background_review_actions.

Regression coverage for issue #14944: the background memory/skill review used
to re-surface tool results that were already present in the conversation
history before the review started (e.g. an earlier "Cron job '...' created.").
"""

import json

import pytest

from hermes_constants import get_hermes_home
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


@pytest.mark.parametrize(("language", "expected"), [
    ("en", "Skill 'demo' patched (references/example.md)"),
    ("zh", "技能 'demo' 已修补 (references/example.md)"),
    ("ko", "스킬 'demo' 패치됨 (references/example.md)"),
])
def test_flat_and_batched_skill_patches_report_the_same_file(monkeypatch, language, expected):
    from tools.skill_manager_tool import skill_manage

    monkeypatch.setenv("HERMES_LANGUAGE", language)
    root = get_hermes_home() / "skills" / "demo"
    path = "references/example.md"
    target = root / path
    target.parent.mkdir(parents=True)
    (root / "SKILL.md").write_text("---\nname: demo\ndescription: Example skill.\n---\n# Demo\n", encoding="utf-8")
    target.write_text("first", encoding="utf-8")

    for batched, old, new in [(False, "first", "second"), (True, "second", "third")]:
        op = dict(action="patch", name="demo", file_path=path, old_string=old, new_string=new)
        args = {"operations": [op]} if batched else op
        raw = skill_manage(action="", name="", operations=[op]) if batched else skill_manage(**op)
        assert target.read_text(encoding="utf-8") == new, raw
        call = {"id": "patch", "function": {"name": "skill_manage", "arguments": json.dumps(args)}}
        messages = [
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "patch", "content": raw},
        ]
        assert _summarize(messages, []) == [expected]


@pytest.mark.parametrize(("result", "expected"), [
    ({"success": True, "staged": True}, []),
    ({"success": False, "error": "blocked"}, []),
    ({"success": True, "_archived": True}, ["Skill 'demo' removed"]),
    ({"success": True}, ["Skill 'demo' deleted"]),
])
def test_flat_delete_notification_reflects_the_outcome(monkeypatch, result, expected):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    args = json.dumps({"action": "delete", "name": "demo"})
    call = {"id": "delete", "function": {"name": "skill_manage", "arguments": args}}
    messages = [{"role": "assistant", "tool_calls": [call]}, _tool_msg("delete", result)]
    assert _summarize(messages, []) == expected
