"""Tests for agent/outcome_tracker.py"""
from __future__ import annotations

import json
from unittest.mock import patch


def _skill_view_msg(skill_name: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_test",
                "type": "function",
                "function": {
                    "name": "skill_view",
                    "arguments": json.dumps({"name": skill_name}),
                },
            }
        ],
    }


def _other_tool_msg(tool_name: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"function": {"name": tool_name, "arguments": json.dumps({"path": "/tmp/x"})}}
        ],
    }


class TestExtractSkillViews:
    def test_extracts_single_skill_name(self):
        from agent.outcome_tracker import _extract_skill_views
        assert _extract_skill_views([_skill_view_msg("coding-skill")]) == {"coding-skill"}

    def test_extracts_multiple_unique_skills(self):
        from agent.outcome_tracker import _extract_skill_views
        msgs = [_skill_view_msg("skill-a"), _skill_view_msg("skill-b")]
        assert _extract_skill_views(msgs) == {"skill-a", "skill-b"}

    def test_deduplicates_repeated_views(self):
        from agent.outcome_tracker import _extract_skill_views
        msgs = [_skill_view_msg("skill-a"), _skill_view_msg("skill-a")]
        assert _extract_skill_views(msgs) == {"skill-a"}

    def test_ignores_other_tool_calls(self):
        from agent.outcome_tracker import _extract_skill_views
        msgs = [_other_tool_msg("read_file"), _other_tool_msg("bash")]
        assert _extract_skill_views(msgs) == set()

    def test_ignores_non_assistant_roles(self):
        from agent.outcome_tracker import _extract_skill_views
        msgs = [
            {"role": "user", "content": "use the coding skill"},
            {"role": "tool", "content": "skill content", "tool_call_id": "call_1"},
        ]
        assert _extract_skill_views(msgs) == set()

    def test_handles_empty_messages(self):
        from agent.outcome_tracker import _extract_skill_views
        assert _extract_skill_views([]) == set()

    def test_handles_malformed_arguments_gracefully(self):
        from agent.outcome_tracker import _extract_skill_views
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "skill_view", "arguments": "not-valid-json"}}
                ],
            }
        ]
        assert _extract_skill_views(msgs) == set()

    def test_handles_missing_tool_calls_key(self):
        from agent.outcome_tracker import _extract_skill_views
        assert _extract_skill_views([{"role": "assistant", "content": "hello"}]) == set()


class TestRecordSessionOutcome:
    def test_calls_bump_use_for_each_skill(self):
        from agent.outcome_tracker import record_session_outcome
        bumped: list[str] = []
        with patch("tools.skill_usage.bump_use", side_effect=lambda n: bumped.append(n)):
            record_session_outcome([_skill_view_msg("skill-a"), _skill_view_msg("skill-b")])
        assert set(bumped) == {"skill-a", "skill-b"}

    def test_no_op_on_empty_messages(self):
        from agent.outcome_tracker import record_session_outcome
        with patch("tools.skill_usage.bump_use") as mock_bump:
            record_session_outcome([])
        mock_bump.assert_not_called()

    def test_no_op_when_no_skill_views(self):
        from agent.outcome_tracker import record_session_outcome
        with patch("tools.skill_usage.bump_use") as mock_bump:
            record_session_outcome([_other_tool_msg("read_file")])
        mock_bump.assert_not_called()

    def test_bump_use_failure_does_not_raise(self):
        from agent.outcome_tracker import record_session_outcome
        with patch("tools.skill_usage.bump_use", side_effect=RuntimeError("disk full")):
            record_session_outcome([_skill_view_msg("my-skill")])  # must not raise


class TestShutdownHook:
    def test_shutdown_memory_provider_calls_record_session_outcome(self):
        """shutdown_memory_provider calls record_session_outcome after our edit.

        NOTE: This test is EXPECTED TO FAIL in Task 1. The hook in run_agent.py
        will be added in Task 2. Write the test now so Task 2 can verify it passes.
        """
        import run_agent as ra

        recorded: list[list] = []

        class FakeAgent:
            _memory_manager = None
            session_id = "test"
            shutdown_memory_provider = ra.AIAgent.shutdown_memory_provider

        messages = [_skill_view_msg("test-skill")]
        with patch("agent.outcome_tracker.record_session_outcome",
                   side_effect=lambda m: recorded.append(m)):
            FakeAgent().shutdown_memory_provider(messages)

        assert len(recorded) == 1
        assert recorded[0] == messages
