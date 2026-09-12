"""Regression tests for the hard-stop design of ``handle_max_iterations`` (#36246).

Before this patch, ``handle_max_iterations`` appended its runtime summary request as
a plain ``role="user"`` row to the persistent ``messages`` list. That had two
visible consequences:

1. The model received a ``role="user"`` row that read like a user complaint
   ("You've reached the maximum...") and paraphrased the budget cut-off as user
   speech in the summary turn ("the user stopped me mid-task").
2. The synthetic user row was persisted verbatim to ``state.db`` and survived
   into later turns, poisoning subsequent context.

The patch routes the iteration-limit instruction through ``role="system"`` on
the wire payload only (not into ``messages``), adds ``tool_choice="none"`` on
the Chat Completions + Codex Responses paths, and adds a response-side scrub
helper that rejects tool-call leakage on every api_mode.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _mock_response(
    content="Hello",
    finish_reason="stop",
    tool_calls=None,
    usage=None,
):
    """Mirror tests/run_agent/test_run_agent.py::_mock_response; trimmed to what we need."""
    if tool_calls:
        msg = SimpleNamespace(content=content or "", tool_calls=tool_calls, reasoning=None)
    else:
        msg = SimpleNamespace(content=content, tool_calls=None, reasoning=None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    if usage:
        resp.usage = SimpleNamespace(**usage)
    else:
        resp.usage = None
    return resp


def _tool_call_stub(name="read_file", args="{}"):
    """A single fake tool_call matching OpenAI's wire shape."""
    return SimpleNamespace(
        id="call_test",
        type="function",
        function=SimpleNamespace(name=name, arguments=args),
    )


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked OpenAI client and tool loading.

    Mirrors the fixture in tests/run_agent/test_run_agent.py but slimmed for
    the focused assertions below.
    """
    with (
        patch(
            "model_tools.get_tool_definitions", return_value=[]
        ),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        return a


def _tool_defs_minimal():
    # Empty tool list keeps the helper happy without pulling model_tools fixtures
    return []


# ===================================================================
# Group 1: persistent messages must NOT carry the synthetic user row
# ===================================================================


class TestPersistentMessagesClean:
    def test_no_synthetic_user_row_after_handle_max_iterations(self, agent):
        """The synthetic nudge MUST NOT be appended to ``messages`` (#36246).

        Before the patch, ``append_message(messages, {"role": "user", "content":
        MAX_ITERATIONS_SUMMARY_REQUEST})`` made the iteration-limit instruction
        part of the persistent history, so subsequent turns saw it as a real
        user turn.
        """
        from agent.context_compressor import MAX_ITERATIONS_SUMMARY_REQUEST

        agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
        messages = [{"role": "user", "content": "do the thing"}]

        result = agent._handle_max_iterations(messages, 60)

        assert result == "Summary"
        # No row in the persistent history may match the synthetic content as a user turn.
        user_rows_with_nudge = [
            m for m in messages
            if m.get("role") == "user" and m.get("content") == MAX_ITERATIONS_SUMMARY_REQUEST
        ]
        assert user_rows_with_nudge == [], (
            f"persistent messages contain synthetic user row(s): {user_rows_with_nudge}"
        )

    def test_synthetic_nudge_only_appears_in_assistant_summary_row(self, agent):
        """The single assistant row appended after a successful summary is fine.

        Verifies we didn't accidentally suppress the legitimate assistant summary
        append while removing the user append.
        """
        agent.client.chat.completions.create.return_value = _mock_response(
            content="Here is the summary."
        )
        messages = [{"role": "user", "content": "do the thing"}]

        agent._handle_max_iterations(messages, 60)

        assistant_summaries = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_summaries) == 1
        assert assistant_summaries[0]["content"] == "Here is the summary."


# ===================================================================
# Group 2: wire payload uses role=system, not role=user
# ===================================================================


class TestWirePayloadRoleSystem:
    def test_chat_completions_wire_payload_uses_system_role_for_nudge(self, agent):
        """The nudge must land on a role=system message, not role=user (#36246)."""
        from agent.context_compressor import MAX_ITERATIONS_SUMMARY_REQUEST

        agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
        messages = [{"role": "user", "content": "do the thing"}]

        agent._handle_max_iterations(messages, 60)

        sent_msgs = agent.client.chat.completions.create.call_args.kwargs.get("messages", [])
        # No user row may carry the synthetic nudge content.
        user_rows_with_nudge = [
            m for m in sent_msgs
            if m.get("role") == "user" and MAX_ITERATIONS_SUMMARY_REQUEST in str(m.get("content", ""))
        ]
        assert user_rows_with_nudge == [], (
            f"wire payload still sends nudge as role=user: {user_rows_with_nudge}"
        )
        # A system message may carry the nudge (combined with cached system prompt).
        system_rows = [m for m in sent_msgs if m.get("role") == "system"]
        assert len(system_rows) >= 1, "expected at least one system message on the wire"
        assert any(
            MAX_ITERATIONS_SUMMARY_REQUEST in str(m.get("content", ""))
            for m in system_rows
        ), "nudge must be on a system message"

    def test_chat_completions_sends_tool_choice_none(self, agent):
        """Wire-level hard stop: Chat Completions must receive tool_choice='none' (#36246)."""
        agent.client.chat.completions.create.return_value = _mock_response(content="Summary")
        messages = [{"role": "user", "content": "do the thing"}]

        agent._handle_max_iterations(messages, 60)

        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert kwargs.get("tool_choice") == "none", (
            f"expected tool_choice='none' on the wire, got {kwargs.get('tool_choice')!r}"
        )


# ===================================================================
# Group 3: scrub helper rejects tool-call leakage on the response side
# ===================================================================


class TestScrubHelper:
    def test_response_with_tool_calls_returns_fallback_not_prose(self, agent):
        """If the model ignores tool_choice='none' and emits tool_calls, the
        runtime returns the fixed fallback string instead of the model's
        mid-thought prose (#36246, #36239)."""
        agent.client.chat.completions.create.return_value = _mock_response(
            content="let me do one more cross-check",
            tool_calls=[_tool_call_stub()],
        )
        messages = [{"role": "user", "content": "do the thing"}]

        result = agent._handle_max_iterations(messages, 60)

        assert result == "I reached the iteration limit and couldn't generate a summary."
        # No assistant row should have been appended (fallback is not a real summary).
        assistant_rows = [m for m in messages if m.get("role") == "assistant"]
        assert assistant_rows == [], f"fallback path should not append an assistant row; got {assistant_rows}"

    def test_fallback_on_first_call_skips_retry(self, agent):
        """If the first call already triggered the fallback, the runtime does
        NOT make a second retry call (same model state guarantees another
        tool_calls payload) (#36246)."""
        agent.client.chat.completions.create.return_value = _mock_response(
            content="",
            tool_calls=[_tool_call_stub()],
        )
        messages = [{"role": "user", "content": "do the thing"}]

        result = agent._handle_max_iterations(messages, 60)

        assert "couldn't" in result.lower() or "I reached" in result
        assert agent.client.chat.completions.create.call_count == 1, (
            f"expected 1 retry-eligible call after fallback; got "
            f"{agent.client.chat.completions.create.call_count}"
        )

    def test_empty_summary_still_retries_once(self, agent):
        """Sanity: an empty first response (no tool_calls) still triggers the
        single retry we had before the patch — we only suppress retry on the
        fallback path, not on empty content."""
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content=""),
            _mock_response(content="Summary"),
        ]
        messages = [{"role": "user", "content": "do the thing"}]

        result = agent._handle_max_iterations(messages, 60)

        assert result == "Summary"
        assert agent.client.chat.completions.create.call_count == 2


# ===================================================================
# Group 4: API failure path unchanged
# ===================================================================


class TestApiFailure:
    def test_api_failure_returns_error_string(self, agent):
        """Regression: the chat-completion raising still surfaces the same
        'couldn't summarize' error string as before the patch."""
        agent.client.chat.completions.create.side_effect = Exception("API down")
        messages = [{"role": "user", "content": "do the thing"}]

        with patch("agent.relay_llm.complete_logical_call") as complete_logical:
            result = agent._handle_max_iterations(messages, 60)

        assert "error" in result.lower()
        assert "API down" in result
        complete_logical.assert_called_once()
        assert complete_logical.call_args.kwargs == {"outcome": "failed"}