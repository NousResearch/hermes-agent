"""Regression tests for the post-ceiling session wedge.

A turn that exhausts all 4 length-continuation attempts must leave the
session usable: the next user message issues a fresh upstream request,
inherits no continuation counter, and the partial text that WAS received
is surfaced instead of dropped.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import PARTIAL_STREAM_STUB_ID, FINISH_REASON_LENGTH


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _stub(content):
    from tests.run_agent.test_run_agent import _mock_assistant_msg
    return SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content=content),
            finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=None,
    )


def _run(agent, message, history=None):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message, conversation_history=history)


class TestContinuationCeilingWedge:
    def _exhaust_ceiling(self, agent):
        agent.client.chat.completions.create.side_effect = [
            _stub("part one "), _stub("part two "),
            _stub("part three "), _stub("part four."),
        ]
        return _run(agent, "write me a long report")

    def test_partial_text_surfaced_at_ceiling(self, loop_agent):
        result = self._exhaust_ceiling(loop_agent)
        assert result["completed"] is False
        assert result["partial"] is True
        assert "part one" in (result["final_response"] or "")
        assert "part four" in (result["final_response"] or "")

    def test_new_user_message_issues_fresh_request(self, loop_agent):
        """Core regression: after the ceiling, a new user turn must reach
        the provider instead of replaying wedge state."""
        from tests.run_agent.test_run_agent import _mock_response

        result1 = self._exhaust_ceiling(loop_agent)
        assert "truncated after 4 continuation attempts" in (result1.get("error") or "")
        calls_after_turn1 = loop_agent.client.chat.completions.create.call_count
        assert calls_after_turn1 == 4

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content="Hello! How can I help?", finish_reason="stop"),
        ]
        result2 = _run(loop_agent, "hi", history=result1["messages"])

        assert loop_agent.client.chat.completions.create.call_count == calls_after_turn1 + 1, (
            "A new user message after the continuation ceiling must issue "
            "exactly one fresh upstream request."
        )
        assert result2["completed"] is True
        assert result2["final_response"] == "Hello! How can I help?"
        assert not result2.get("error")

    def test_ceiling_replaces_scaffolding_with_settled_turn(self, loop_agent):
        """The persisted tail must not keep the continuation scaffolding.
        Unanswered "continue" nudges make every later turn resume the
        truncated response and re-exhaust the same ceiling."""
        result = self._exhaust_ceiling(loop_agent)
        msgs = result["messages"]

        nudges = [
            m for m in msgs
            if m.get("role") == "user"
            and "Continue exactly where you left off" in (m.get("content") or "")
        ]
        assert nudges == [], (
            "Continuation nudges must not survive the ceiling exit — they "
            "steer every subsequent turn back into the truncated response."
        )

        assistants = [m for m in msgs if m.get("role") == "assistant"]
        assert len(assistants) == 1, (
            "The fragment trail must collapse into one settled assistant turn."
        )
        assert msgs[-1]["role"] == "assistant"
        content = msgs[-1]["content"] or ""
        for part in ("part one", "part two", "part three", "part four"):
            assert part in content, "Stitched partial must keep every fragment."

    def test_ceiling_not_labeled_network_error(self, loop_agent):
        """A finish_reason='length' stub is a truncation, not a network
        error — the user-facing message must not blame the network."""
        printed = []
        original = loop_agent._vprint

        def _capture(text, **kwargs):
            printed.append(str(text))
            return original(text, **kwargs)

        with patch.object(loop_agent, "_vprint", side_effect=_capture):
            self._exhaust_ceiling(loop_agent)

        network_lines = [line for line in printed if "network error" in line.lower()]
        assert network_lines == [], (
            "Truncation must not be reported as a network error: "
            f"{network_lines!r}"
        )
        assert any("truncated" in line.lower() for line in printed), (
            "The user-facing message must name the truncation."
        )

    def test_continuation_requests_carry_no_marks(self, loop_agent):
        """The scaffolding marks are Hermes bookkeeping. The centrally
        sanitized api_messages must never carry them — only the
        chat-completions transport strips underscore keys, so anthropic
        and bedrock requests would otherwise send them to the provider."""
        from tests.run_agent.test_run_agent import _mock_response

        seen_api_messages = []
        original = loop_agent._build_api_kwargs

        def _spy(api_messages, tools_for_api=None):
            seen_api_messages.append([dict(m) for m in api_messages if isinstance(m, dict)])
            return original(api_messages, tools_for_api=tools_for_api)

        loop_agent.client.chat.completions.create.side_effect = [
            _stub("part one "), _stub("part two "),
            _mock_response(content="the rest.", finish_reason="stop"),
        ]
        with patch.object(loop_agent, "_build_api_kwargs", side_effect=_spy):
            result = _run(loop_agent, "write me a long report")

        assert result["completed"] is True
        assert len(seen_api_messages) >= 3, "Expected continuation attempts 2+."
        marked = [
            (idx, key)
            for idx, batch in enumerate(seen_api_messages)
            for m in batch
            for key in m
            if str(key).startswith("_length_continuation")
        ]
        assert marked == [], (
            f"Continuation marks leaked into outgoing api_messages: {marked!r}"
        )

    def test_prior_turn_marked_message_survives_later_ceiling(self, loop_agent):
        """A mark that reached disk mid-crash and got reloaded on a PRIOR
        turn's message must never be deleted by a later turn's ceiling
        cleanup — the cleanup is scoped to the current turn."""
        reloaded_history = [
            {"role": "user", "content": "earlier question"},
            {
                "role": "assistant",
                "content": "earlier answer fragment",
                "_length_continuation_fragment": True,
            },
        ]
        loop_agent.client.chat.completions.create.side_effect = [
            _stub("wedge one "), _stub("wedge two "),
            _stub("wedge three "), _stub("wedge four."),
        ]
        result = _run(loop_agent, "another long report", history=reloaded_history)

        assert "truncated after 4 continuation attempts" in (result.get("error") or "")
        prior = [
            m for m in result["messages"]
            if m.get("role") == "assistant"
            and "earlier answer fragment" in (m.get("content") or "")
        ]
        assert len(prior) == 1, (
            "The prior turn's reloaded message must survive the later "
            "turn's ceiling cleanup."
        )

    def test_new_turn_does_not_inherit_continuation_counter(self, loop_agent):
        """A single truncation on the turn AFTER the ceiling must get its
        own full 4-attempt budget, not the exhausted counter."""
        from tests.run_agent.test_run_agent import _mock_response

        result1 = self._exhaust_ceiling(loop_agent)
        loop_agent.client.chat.completions.create.side_effect = [
            _stub("second turn partial "),
            _mock_response(content="and the rest.", finish_reason="stop"),
        ]
        result2 = _run(loop_agent, "try again", history=result1["messages"])

        assert result2["completed"] is True, (
            "One truncation on a fresh turn must continue (1/4), not fail "
            "with an inherited exhausted counter."
        )
        assert "second turn partial" in result2["final_response"]
        assert "and the rest." in result2["final_response"]


class TestTruncatedPartJoining:
    """#78577 — parts joined with no separator glued text together."""

    def test_glued_parts_get_a_newline(self):
        from agent.conversation_loop import _join_truncated_parts
        assert _join_truncated_parts(
            ["Edited index.html", "Review the 5 changes"]
        ) == "Edited index.html\nReview the 5 changes"

    def test_existing_whitespace_is_not_doubled(self):
        from agent.conversation_loop import _join_truncated_parts
        assert _join_truncated_parts(["line one\n", "line two"]) == "line one\nline two"
        assert _join_truncated_parts(["word", " next"]) == "word next"

    def test_degenerate_inputs(self):
        from agent.conversation_loop import _join_truncated_parts
        assert _join_truncated_parts([]) == ""
        assert _join_truncated_parts(["only"]) == "only"
        assert _join_truncated_parts(["a", "", "b"]) == "a\nb"
