"""Tests for SkillOpt trace harvesting."""

from __future__ import annotations


class FakeDB:
    def __init__(self):
        self.queries = []

    def search_messages(self, query, role_filter=None, limit=20, sort=None, include_inactive=False):
        self.queries.append((query, role_filter, limit, sort, include_inactive))
        return [
            {"session_id": "s1", "message_id": 10, "source": "discord", "content": "Loaded skill_view systematic-debugging"},
            {"session_id": "cron1", "message_id": 20, "source": "cron", "content": "cron scaffold skill_view systematic-debugging"},
        ]

    def get_messages_around(self, session_id, message_id, window=4):
        return {
            "window": [
                {"id": message_id - 1, "role": "user", "content": "fix bug"},
                {"id": message_id, "role": "assistant", "content": "using skill"},
                {"id": message_id + 1, "role": "tool", "tool_name": "terminal", "content": "pytest failed"},
            ],
            "messages_before": 1,
            "messages_after": 1,
        }


def test_harvest_skill_traces_excludes_cron_and_returns_context_windows():
    from agent.skillopt_harvest import harvest_skill_traces

    db = FakeDB()
    traces = harvest_skill_traces(db, "systematic-debugging", limit=5)

    assert len(traces) == 1
    assert traces[0]["session_id"] == "s1"
    assert traces[0]["anchor_message_id"] == 10
    assert traces[0]["messages"][0]["content"] == "fix bug"
    assert db.queries[0][0] == '"systematic-debugging" OR systematic-debugging OR skill_view OR skill_manage'


def test_harvest_skill_traces_rejects_empty_skill_name():
    from agent.skillopt_harvest import harvest_skill_traces

    try:
        harvest_skill_traces(FakeDB(), "")
    except ValueError as exc:
        assert "skill_name" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_distill_trace_to_skill_extracts_commands_and_generalizes_steps():
    from agent.skillopt_harvest import distill_trace_to_skill

    trace = {
        "skill_name": "debug-python-tests",
        "messages": [
            {"role": "user", "content": "Fix the pytest failure"},
            {"role": "assistant", "content": "I will reproduce it."},
            {"role": "tool", "tool_name": "terminal", "content": "pytest tests/test_demo.py -q\nFAILED"},
            {"role": "tool", "tool_name": "patch", "content": "*** Begin Patch\n*** Update File: demo.py"},
            {"role": "tool", "tool_name": "terminal", "content": "pytest tests/test_demo.py -q\n1 passed"},
        ],
    }

    distilled = distill_trace_to_skill(trace)

    assert distilled["name"] == "debug-python-tests"
    assert "pytest tests/test_demo.py -q" in distilled["commands"]
    assert any("Reproduce" in step for step in distilled["steps"])
    assert "SKILL.md" in distilled["skill_markdown"]
