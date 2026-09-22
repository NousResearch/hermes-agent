"""Tests for AIAgent._summarize_background_review_actions.

Regression coverage for issue #14944: the background memory/skill review used
to re-surface tool results that were already present in the conversation
history before the review started (e.g. an earlier "Cron job '...' created.").
"""

import json

import pytest

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


# ---------------------------------------------------------------------------
# The summary is a static notification, so its wording comes from the i18n
# catalog (display.language / HERMES_LANGUAGE) instead of hardcoded English.
# ---------------------------------------------------------------------------


def _skill_call_messages(action="patch", name="demo-skill", message="Skill updated."):
    """Review-fork messages for one applied ``skill_manage`` operation."""
    call_id = "call_skill"
    payload = {
        "success": True,
        "operations_applied": True,
        "message": message,
        "results": [{"success": True, "action": action, "name": name}],
    }
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "skill_manage",
                        "arguments": json.dumps({"name": name, "action": action}),
                    },
                }
            ],
        },
        _tool_msg(call_id, payload),
    ]


@pytest.fixture
def chinese_notifications(monkeypatch):
    """Pin the notification language to Chinese (the env override documented for display.language)."""
    from agent import i18n

    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    i18n.reset_language_cache()
    yield
    i18n.reset_language_cache()


def test_summary_lines_follow_display_language(chinese_notifications):
    """Skill/memory action lines render in the configured language."""
    assert _summarize(_skill_call_messages(), []) == ["技能「demo-skill」已修补"]

    memory_messages = [_tool_msg("c9", {"success": True, "message": "Entry added", "target": "user"})]
    assert _summarize(memory_messages, []) == ["用户画像已更新"]


def test_published_notification_follows_display_language(chinese_notifications):
    """The published "💾 …" line is the catalog's, so both the CLI print and the gateway push localize."""
    from agent.background_review import _publish_review_summary

    class _Agent:
        def __init__(self):
            self.printed = []
            self.pushed = []
            self.background_review_callback = self.pushed.append

        def _safe_print(self, text):
            self.printed.append(text)

    agent = _Agent()
    _publish_review_summary(agent, ["技能「demo-skill」已修补"])

    assert agent.printed == ["  💾 自我改进评审：技能「demo-skill」已修补"]
    assert agent.pushed == ["💾 自我改进评审：技能「demo-skill」已修补"]
