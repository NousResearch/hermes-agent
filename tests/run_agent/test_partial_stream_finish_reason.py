"""Regression tests for issue #30963 — partial-stream stub finish_reason.

Pins the contract:

- text-only partial stream → stub.finish_reason == "length" so the
  conversation loop's existing length-continuation path can keep the
  agent moving against an unfinished goal.
- partial mid-tool-call → stub.finish_reason == "stop" so the loop
  hands control back to the user (matches the user-visible warning
  "Ask me to retry if you want to continue").
- conversation_loop's length-continuation prompt distinguishes a real
  output-length truncation from a partial-stream-stub network error
  via response.id.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers (mirrors test_streaming.py) ────────────────────────────────────

def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content, tool_calls=tool_calls,
        reasoning_content=None, reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=None, usage=None)


def _make_tool_call_delta(index=0, tc_id=None, name=None, arguments=None):
    func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=func)


def _make_agent():
    from run_agent import AIAgent
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


# ── Stub finish_reason ────────────────────────────────────────────────────

class TestPartialStreamStubFinishReason:
    """The stub returned by interruptible_streaming_api_call when the
    upstream connection dies mid-flight."""

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_text_only_partial_returns_length(self, _mock_close, mock_create, monkeypatch):
        """#30963: text-only partials must classify as length so the loop
        keeps continuing instead of exiting with budget remaining."""

        def _stalling_stream():
            yield _make_stream_chunk(content="Here's my answer so far")
            raise RuntimeError("simulated upstream stall")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _stalling_stream()
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._current_streamed_assistant_text = "Here's my answer so far"

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        response = agent._interruptible_streaming_api_call({})

        assert response.id == "partial-stream-stub"
        assert response.choices[0].finish_reason == "length", (
            "Text-only partial streams must use finish_reason=length so the "
            "conversation loop continues from where the network died "
            "(issue #30963)."
        )
        assert response.choices[0].message.content == "Here's my answer so far"
        assert response.choices[0].message.tool_calls is None

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_partial_tool_call_keeps_stop(self, _mock_close, mock_create, monkeypatch):
        """Mid-tool-call partials keep finish_reason=stop on purpose — the
        warning text asks the user to drive the retry, not the agent."""

        def _stalling_stream():
            yield _make_stream_chunk(content="Let me write the audit: ")
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_1", name="write_file"),
            ])
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments='{"path": "/tmp/x", '),
            ])
            raise RuntimeError("simulated upstream stall")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _stalling_stream()
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._fire_stream_delta = lambda text: None
        agent._current_streamed_assistant_text = "Let me write the audit: "

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        response = agent._interruptible_streaming_api_call({})

        assert response.id == "partial-stream-stub"
        assert response.choices[0].finish_reason == "stop", (
            "Partial mid-tool-call must keep finish_reason=stop — the warning "
            "appended to content asks the user to retry, so the agent must "
            "not auto-replay a tool call with possible side-effects."
        )
        content = response.choices[0].message.content or ""
        assert "Stream stalled mid tool-call" in content
        assert "write_file" in content


# ── Length-continuation prompt branching ──────────────────────────────────

class TestLengthContinuationPromptBranching:
    """When finish_reason=length, the continuation prompt that reaches the
    model has to tell the truth: real truncation vs. network interruption.
    Lying ("you were truncated") on a partial-stream stub leads the model
    to no-op ("I wasn't truncated, I'm done"), defeating recovery."""

    def _simulate_branch(self, response_id: str) -> str:
        """Return the continuation prompt text the loop would inject for
        a `finish_reason=length` response with the given id. Mirrors the
        exact branch in agent/conversation_loop.py."""
        response = SimpleNamespace(id=response_id)
        if getattr(response, "id", "") == "partial-stream-stub":
            return (
                "[System: The previous response was cut off by a "
                "network error mid-stream. Continue exactly where "
                "you left off. Do not restart or repeat prior text. "
                "Finish the answer directly.]"
            )
        return (
            "[System: Your previous response was truncated by the output "
            "length limit. Continue exactly where you left off. Do not "
            "restart or repeat prior text. Finish the answer directly.]"
        )

    def test_partial_stream_stub_uses_network_prompt(self):
        prompt = self._simulate_branch("partial-stream-stub")
        assert "network error mid-stream" in prompt
        assert "output length limit" not in prompt

    def test_real_truncation_uses_length_prompt(self):
        prompt = self._simulate_branch("chatcmpl-abc123")
        assert "output length limit" in prompt
        assert "network error" not in prompt

    def test_no_id_falls_through_to_length_prompt(self):
        prompt = self._simulate_branch("")
        assert "output length limit" in prompt


# ── Integration: live conversation loop ───────────────────────────────────

@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_run_agent's fixture)
    so we can stage a stub + continuation pair on .chat.completions.create."""
    from run_agent import AIAgent
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
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
        a.tool_delay = 0
        a.compression_enabled = False
        a.save_trajectories = False
        return a


class TestConversationLoopPartialStreamContinuation:
    """End-to-end: a partial-stream stub feeds the loop and the loop
    asks for continuation instead of exiting with finish_reason=stop."""

    def test_partial_stream_stub_does_not_exit_loop_immediately(self, loop_agent):
        """The stub from chat_completion_helpers used to exit the loop with
        text_response(finish_reason=stop). Now finish_reason=length routes
        through length_continue_retries — the loop persists the partial
        content and asks the model to continue."""

        from tests.run_agent.test_run_agent import _mock_response, _mock_assistant_msg

        # First API call: the partial-stream stub (length on partial-stream-stub id).
        partial_stub = SimpleNamespace(
            id="partial-stream-stub",
            model="test/model",
            choices=[SimpleNamespace(
                index=0,
                message=_mock_assistant_msg(content="The first half of "),
                finish_reason="length",
            )],
            usage=None,
        )
        # Second API call: model continues with the rest, clean stop.
        continuation = _mock_response(
            content="the answer is forty-two.", finish_reason="stop",
        )

        loop_agent.client.chat.completions.create.side_effect = [
            partial_stub, continuation,
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("ask me something")

        # The loop made TWO API calls (stub + continuation), not one.
        assert loop_agent.client.chat.completions.create.call_count == 2, (
            "Partial-stream-stub must trigger a continuation API call, not "
            "exit the loop after one call."
        )
        # The continuation prompt the loop appended must be the network-error
        # variant, not the "output length limit" lie — otherwise the model
        # no-ops with "I wasn't truncated, I'm done."
        # We assert it indirectly by inspecting the second-call kwargs.
        second_call_kwargs = loop_agent.client.chat.completions.create.call_args_list[1]
        msgs = second_call_kwargs.kwargs.get("messages") or second_call_kwargs.args[0].get("messages")
        last_user = next(
            (m for m in reversed(msgs) if m.get("role") == "user"), None,
        )
        assert last_user is not None
        assert "network error mid-stream" in (last_user.get("content") or ""), (
            "Continuation prompt for partial-stream-stub must mention the "
            "network error, not the 'output length limit'."
        )

        # And the final response stitches both halves together.
        assert "first half of" in result["final_response"]
        assert "forty-two" in result["final_response"]
