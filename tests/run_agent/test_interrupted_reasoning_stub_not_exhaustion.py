"""Regression tests: interrupted reasoning streams must NOT be classified as
Thinking Budget Exhausted.

A stream that dies mid-reasoning returns as a partial-stream stub
(``PARTIAL_STREAM_STUB_ID``) labelled finish_reason="length" — a Hermes classification of a
dropped connection, not wire proof the budget was exhausted (the 08:32 live logs show no
finish_reason and no usage).

Pinned contract:

- a reasoning-only stub (closed or unterminated) is retried (bounded); a valid retry completes;
- repeated reasoning-only stubs end at the ceiling with an honest error that never claims an
  output-token limit;
- a REAL length truncation that is reasoning-only (normal id) still aborts with Thinking
  Budget Exhausted;
- a stub's visible text is preserved with edge whitespace intact;
- the stub gate disables ONLY the thinking-exhaustion heuristic — repetition detection stays.
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


_THINK_ONLY = "<thinking>Let me work through this carefully and thoroughly</thinking>"


def _stub(content):
    """Partial-stream stub (dropped connection): no wire finish_reason/usage."""
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


def _reasoning_only_length_response():
    """REAL length truncation (normal id, not a stub): reasoning, no visible text."""
    from tests.run_agent.test_run_agent import _mock_assistant_msg

    return SimpleNamespace(
        id="chatcmpl-real-length",
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content=_THINK_ONLY),
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


class TestInterruptedReasoningStubNotExhaustion:
    def test_reasoning_only_stub_then_valid_response_completes(self, loop_agent):
        """Closed reasoning-only stub: continue and complete — not exhaustion abort."""
        from tests.run_agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _stub(_THINK_ONLY),
            _mock_response(content="Here is the final answer.", finish_reason="stop"),
        ]
        result = _run(loop_agent, "solve this")

        assert result["completed"] is True
        assert "final answer" in (result["final_response"] or "")
        assert "Thinking Budget" not in (result["final_response"] or "")
        assert "exhaust" not in (result["final_response"] or "").lower()
        assert loop_agent.client.chat.completions.create.call_count == 2

        # The reasoning must never leak into the persisted transcript.
        for m in result["messages"]:
            if isinstance(m, dict) and m.get("role") == "assistant":
                assert "Let me work through" not in (m.get("content") or "")

    def test_unterminated_reasoning_only_stub_then_valid_response(self, loop_agent):
        """The live 08:32 shape: the stream died BEFORE the closing tag. Must continue
        (bounded), not abort as exhaustion."""
        from tests.run_agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _stub("<thinking>Let me analyze the request step by step and"),
            _mock_response(content="The answer is 42.", finish_reason="stop"),
        ]
        result = _run(loop_agent, "what is the answer")

        assert result["completed"] is True
        assert "The answer is 42." in (result["final_response"] or "")
        assert "Thinking Budget" not in (result["final_response"] or "")
        assert loop_agent.client.chat.completions.create.call_count == 2
        for m in result["messages"]:
            if isinstance(m, dict) and m.get("role") == "assistant":
                assert "step by step" not in (m.get("content") or "")

    def test_repeated_reasoning_only_stubs_bounded_honest_error(self, loop_agent):
        """Four reasoning-only stubs: bounded ceiling exit with an honest dropped-stream
        error — no claim the model hit an output-token limit."""
        loop_agent.client.chat.completions.create.side_effect = [
            _stub("<thinking>still thinking and"),
            _stub("<thinking>still thinking and"),
            _stub("<thinking>still thinking and"),
            _stub("<thinking>still thinking and"),
        ]
        result = _run(loop_agent, "write me a long report")

        assert result["completed"] is False
        assert result["partial"] is True
        assert loop_agent.client.chat.completions.create.call_count == 4
        assert "truncated after 4 continuation attempts" in (result.get("error") or "")
        assert "output length limit" not in (result.get("error") or "").lower()
        assert "output token" not in (result.get("error") or "").lower()
        assert "Thinking Budget" not in (result["final_response"] or "")
        assert "output length limit" not in (result["final_response"] or "").lower()


class TestRealLengthReasoningOnlyStillExhaustion:
    def test_real_length_reasoning_only_aborts_exhaustion(self, loop_agent):
        """Control: a REAL length truncation (normal id, wire finish_reason) that is
        reasoning-only must STILL abort with Thinking Budget Exhausted — the stub gate
        must not weaken the real case."""
        loop_agent.client.chat.completions.create.side_effect = [
            _reasoning_only_length_response(),
        ]
        result = _run(loop_agent, "write me a long report")

        assert result["completed"] is False
        assert result["partial"] is True
        assert "Thinking Budget Exhausted" in (result["final_response"] or "")
        assert loop_agent.client.chat.completions.create.call_count == 1, (
            "A real reasoning-only length truncation must abort immediately, "
            "not burn continuation retries."
        )


class TestStubRepetitionStillAborts:
    def test_stub_repetition_loop_still_aborts(self, loop_agent):
        """Direct regression: the stub gate must not disable repetition detection — a stub
        whose visible text is a repetition loop still aborts immediately."""
        _repeated = "The quick brown fox jumps over the lazy dog near the riverbank 12345 " * 8
        loop_agent.client.chat.completions.create.side_effect = [_stub(_repeated)]
        result = _run(loop_agent, "write me a long report")

        assert result["completed"] is False
        assert result["partial"] is True
        assert "Repetition" in (result["final_response"] or "")
        assert loop_agent.client.chat.completions.create.call_count == 1, (
            "A repetition-dominated stub must abort immediately, not burn continuation retries."
        )


class TestVisiblePartialRecoveryPreserved:
    def test_stub_with_visible_text_keeps_partial(self, loop_agent):
        """A stub carrying visible text is stitched into the final response on a
        successful continuation (regression guard)."""
        from tests.run_agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _stub("Here is my partial answer: "),
            _mock_response(content="the rest of the answer.", finish_reason="stop"),
        ]
        result = _run(loop_agent, "explain it")

        assert result["completed"] is True
        assert "Here is my partial answer:" in (result["final_response"] or "")
        assert "the rest of the answer." in (result["final_response"] or "")

    def test_ceiling_stitch_keeps_fragment_edge_whitespace(self, loop_agent):
        """Stored visible fragments keep their edge whitespace: a trailing space stays the
        stitching point, so the ceiling partial is joined without an injected newline."""
        loop_agent.client.chat.completions.create.side_effect = [
            _stub("One two three "),
            _stub("One two three "),
            _stub("One two three "),
            _stub("One two three "),
        ]
        result = _run(loop_agent, "write me a long report")

        assert result["completed"] is False
        assert loop_agent.client.chat.completions.create.call_count == 4
        assert (result["final_response"] or "") == (
            "One two three One two three One two three One two three"
        )
