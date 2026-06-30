"""Regression tests for issue #30963 — partial-stream stub finish_reason.

Pins the contract:

- text-only partial stream → stub.finish_reason == "length" so the
  conversation loop's existing length-continuation path can keep the
  agent moving against an unfinished goal.
- partial mid-tool-call → stub.finish_reason == "length" so the loop
  triggers continuation machinery with targeted chunking guidance
  instead of ending the turn immediately.
- conversation_loop's length-continuation prompt distinguishes a real
  output-length truncation from a partial-stream-stub network error
  via response.id.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import PARTIAL_STREAM_STUB_ID, FINISH_REASON_LENGTH
from agent.conversation_loop import _get_continuation_prompt


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

        assert response.id == PARTIAL_STREAM_STUB_ID
        assert response.choices[0].finish_reason == FINISH_REASON_LENGTH, (
            "Text-only partial streams must use finish_reason=length so the "
            "conversation loop continues from where the network died "
            "(issue #30963)."
        )
        assert response.choices[0].message.content == "Here's my answer so far"
        assert response.choices[0].message.tool_calls is None

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_partial_tool_call_uses_length(self, _mock_close, mock_create, monkeypatch):
        """Mid-tool-call partials now use finish_reason=length so the
        conversation loop's continuation machinery fires — bounded 3-retry
        with guidance to break output into smaller chunks (#31998).
        tool_calls=None is preserved, so no tool auto-executes."""

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

        assert response.id == PARTIAL_STREAM_STUB_ID
        assert response.choices[0].finish_reason == FINISH_REASON_LENGTH, (
            "Partial mid-tool-call must use finish_reason=length so the "
            "continuation machinery fires instead of ending the turn "
            "immediately (#31998)."
        )
        assert response.choices[0].message.tool_calls is None, (
            "tool_calls must remain None (no auto-execution of side-effectful "
            "tool calls)."
        )
        # The stub should carry dropped tool names for continuation prompt
        assert getattr(response, "_dropped_tool_names", None) == ["write_file"]
        content = response.choices[0].message.content or ""
        assert "Stream stalled mid tool-call" in content
        assert "write_file" in content


# ── Clean stream-end mid-tool-call (no exception, no finish_reason) ─────────

class TestCleanStreamEndMidToolCall:
    """The upstream closes the SSE stream cleanly after delivering a tool
    name + the opening '{' of its arguments — NO exception, NO finish_reason,
    NO [DONE].  Observed live on NVIDIA Nemotron Ultra via the Nous dedicated
    endpoint: it stalls/drops during large tool-arg generation.

    The mock-builder must NOT stamp this as finish_reason='length' (which
    routes it through the max_tokens-boost truncation path and finally
    reports the misleading 'Response truncated due to output length limit').
    It must route through the partial-stream-stub path so the loop reports
    an honest mid-tool-call drop and asks the model to chunk its output.
    """

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_no_finish_reason_partial_tool_args_routes_to_stub(
        self, _mock_close, mock_create, monkeypatch,
    ):
        def _clean_ending_stream():
            # Reasoning + tool name + the lone opening brace, then the
            # generator simply RETURNS (StopIteration) — no raise, no
            # finish_reason chunk, no [DONE].
            yield _make_stream_chunk(content="\n")
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_x", name="execute_code"),
            ])
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments="{"),
            ])
            # falls off the end — clean close, no terminator

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = (
            lambda *a, **kw: _clean_ending_stream()
        )
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._fire_stream_delta = lambda text: None

        response = agent._interruptible_streaming_api_call({})

        assert response.id == PARTIAL_STREAM_STUB_ID, (
            "A clean stream-end mid tool-call (no finish_reason) must be "
            "tagged as a partial-stream stub, not a 'stream-<uuid>' "
            "truncation — otherwise the loop reports the false 'output "
            "length limit' error."
        )
        assert response.choices[0].finish_reason == FINISH_REASON_LENGTH
        assert response.choices[0].message.tool_calls is None, (
            "Incomplete tool args must never auto-execute."
        )
        assert getattr(response, "_dropped_tool_names", None) == ["execute_code"]

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_real_length_truncation_still_uses_uuid_id(
        self, _mock_close, mock_create, monkeypatch,
    ):
        """Control: when the provider DOES send finish_reason='length' with
        partial tool args, it is a genuine output cap — keep the existing
        non-stub behaviour (boost max_tokens and retry)."""

        def _capped_stream():
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_y", name="execute_code"),
            ])
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments="{"),
            ])
            # Provider explicitly reports the output cap.
            yield _make_stream_chunk(finish_reason="length")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = (
            lambda *a, **kw: _capped_stream()
        )
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._fire_stream_delta = lambda text: None

        response = agent._interruptible_streaming_api_call({})

        assert response.id != PARTIAL_STREAM_STUB_ID, (
            "A provider-reported finish_reason='length' is a real output cap "
            "and must keep the existing truncation path, not the stream-drop "
            "stub path."
        )
        assert response.id.startswith("stream-")
        assert response.choices[0].finish_reason == FINISH_REASON_LENGTH


# ── Length-continuation prompt branching ──────────────────────────────────

class TestLengthContinuationPromptBranching:
    """When finish_reason=length, the continuation prompt that reaches the
    model has to tell the truth: real truncation vs. network interruption
    vs. dropped tool call (#31998).  Three distinct prompts now exist."""

    def _simulate_branch(self, response_id: str, dropped_tools=None) -> str:
        """Return the continuation prompt text the loop would inject for
        a `finish_reason=length` response with the given id."""
        is_partial = response_id == PARTIAL_STREAM_STUB_ID
        return _get_continuation_prompt(is_partial, dropped_tools)

    def test_partial_stream_stub_uses_network_prompt(self):
        prompt = self._simulate_branch(PARTIAL_STREAM_STUB_ID)
        assert "network error mid-stream" in prompt
        assert "output length limit" not in prompt

    def test_real_truncation_uses_length_prompt(self):
        prompt = self._simulate_branch("chatcmpl-abc123")
        assert "output length limit" in prompt
        assert "network error" not in prompt

    def test_no_id_falls_through_to_length_prompt(self):
        prompt = self._simulate_branch("")
        assert "output length limit" in prompt

    def test_dropped_tool_call_uses_chunking_prompt(self):
        """When the stub dropped a tool call, the continuation prompt
        must guide the model to break its output into smaller chunks
        instead of retrying the same large tool call (#31998)."""
        prompt = self._simulate_branch(
            PARTIAL_STREAM_STUB_ID, dropped_tools=["write_file"],
        )
        assert "too large" in prompt
        assert "break" in prompt.lower()
        assert "write_file" in prompt
        assert "network error" not in prompt
        assert "output length limit" not in prompt


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

        from tests.run_agent._run_agent_helpers import _mock_response, _mock_assistant_msg

        # First API call: the partial-stream stub (length on partial-stream-stub id).
        partial_stub = SimpleNamespace(
            id=PARTIAL_STREAM_STUB_ID,
            model="test/model",
            choices=[SimpleNamespace(
                index=0,
                message=_mock_assistant_msg(content="The first half of "),
                finish_reason=FINISH_REASON_LENGTH,
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


class TestContentFilterStallActivatesFallback:
    """Regression for #32421: a provider output-layer content safety filter
    (e.g. MiniMax ``output new_sensitive (1027)``) terminates a streaming
    response mid-delivery.  The raw error is swallowed into a
    finish_reason=length partial-stream stub, so before the fix the loop
    burned 3 continuation retries against the SAME primary (re-hitting the
    content-deterministic filter every time) and gave up with
    ``"Response remained truncated after 3 continuation attempts"`` — the
    configured fallback chain was never consulted.

    The fix has three layers:
      1. error_classifier classifies ``new_sensitive`` as
         ``content_policy_blocked``.
      2. interruptible_streaming_api_call runs the swallowed error through
         that classifier and stamps the stub ``_content_filter_terminated``.
      3. the conversation loop reads the tag and activates fallback BEFORE
         burning any continuation retries.
    """

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_streaming_call_tags_content_filter_stub(
        self, _mock_close, mock_create, monkeypatch,
    ):
        """Layer 2: the real streaming path stamps _content_filter_terminated
        when the swallowed error matches a content-filter pattern."""

        def _minimax_stall():
            yield _make_stream_chunk(content="Writing the file: ")
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_1", name="write_file"),
            ])
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments='{"path": "/tmp/x", '),
            ])
            raise RuntimeError("output new_sensitive (1027) [MiniMax-M2.7]")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = (
            lambda *a, **kw: _minimax_stall()
        )
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._fire_stream_delta = lambda text: None
        agent._current_streamed_assistant_text = "Writing the file: "

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        response = agent._interruptible_streaming_api_call({})

        assert response.id == PARTIAL_STREAM_STUB_ID
        assert getattr(response, "_content_filter_terminated", False) is True, (
            "MiniMax new_sensitive stream stall must tag the stub so the loop "
            "can route to fallback (#32421)."
        )

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_plain_network_stall_not_tagged(
        self, _mock_close, mock_create, monkeypatch,
    ):
        """A plain network stall (no content-filter signature) must NOT be
        tagged — it should still use the normal continuation path, not
        switch providers."""

        def _network_stall():
            yield _make_stream_chunk(content="Writing the file: ")
            raise RuntimeError("connection reset by peer")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = (
            lambda *a, **kw: _network_stall()
        )
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._fire_stream_delta = lambda text: None
        agent._current_streamed_assistant_text = "Writing the file: "

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        response = agent._interruptible_streaming_api_call({})

        assert response.id == PARTIAL_STREAM_STUB_ID
        assert getattr(response, "_content_filter_terminated", False) is False, (
            "A plain network stall must not be misclassified as a content "
            "filter — that would needlessly switch providers."
        )

    def test_tagged_stub_activates_fallback_first_pass(self, loop_agent):
        """Layer 3: a tagged stub activates fallback on the FIRST pass, with
        zero continuation retries burned, and the fallback provider then
        completes the turn."""
        from tests.run_agent._run_agent_helpers import _mock_assistant_msg, _mock_response

        def _filter_stub():
            return SimpleNamespace(
                id=PARTIAL_STREAM_STUB_ID,
                model="minimax/MiniMax-M2.7",
                choices=[SimpleNamespace(
                    index=0,
                    message=_mock_assistant_msg(content="Writing the file..."),
                    finish_reason=FINISH_REASON_LENGTH,
                )],
                usage=None,
                _dropped_tool_names=["write_file"],
                _content_filter_terminated=True,
            )

        recovery = _mock_response(
            content="Done on the fallback provider.", finish_reason="stop",
        )
        loop_agent.client.chat.completions.create.side_effect = [
            _filter_stub(), recovery,
        ]
        loop_agent._fallback_chain = [
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.7"},
        ]
        loop_agent._fallback_index = 0
        fb_calls = {"n": 0}

        def _fake_activate(reason=None):
            fb_calls["n"] += 1
            loop_agent._fallback_index = len(loop_agent._fallback_chain)
            return True

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
            patch.object(loop_agent, "_try_activate_fallback",
                         side_effect=_fake_activate),
        ):
            result = loop_agent.run_conversation("write me a long file")

        assert fb_calls["n"] == 1, (
            "Content-filter-tagged stub must activate fallback exactly once, "
            "on the first pass — not after exhausting continuation retries."
        )
        assert result["final_response"] == "Done on the fallback provider."
        assert result["completed"] is True

    def test_tagged_stub_no_fallback_falls_through(self, loop_agent):
        """When no fallback chain is configured, a tagged stub falls through
        to the normal continuation path (best-effort) rather than crashing."""
        from tests.run_agent._run_agent_helpers import _mock_assistant_msg, _mock_response

        def _filter_stub():
            return SimpleNamespace(
                id=PARTIAL_STREAM_STUB_ID,
                model="minimax/MiniMax-M2.7",
                choices=[SimpleNamespace(
                    index=0,
                    message=_mock_assistant_msg(content="partial "),
                    finish_reason=FINISH_REASON_LENGTH,
                )],
                usage=None,
                _dropped_tool_names=["write_file"],
                _content_filter_terminated=True,
            )

        recovery = _mock_response(content="recovered text", finish_reason="stop")
        loop_agent.client.chat.completions.create.side_effect = [
            _filter_stub(), recovery,
        ]
        # No fallback chain configured.
        loop_agent._fallback_chain = []
        loop_agent._fallback_index = 0
        fb_calls = {"n": 0}

        def _fake_activate(reason=None):
            fb_calls["n"] += 1
            return False

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
            patch.object(loop_agent, "_try_activate_fallback",
                         side_effect=_fake_activate),
        ):
            result = loop_agent.run_conversation("write me a long file")

        # Fallback was not attempted (empty chain gates it out); the loop
        # continued normally and produced a response.
        assert fb_calls["n"] == 0, (
            "With an empty fallback chain, the loop must not even call "
            "_try_activate_fallback — it should fall through to continuation."
        )
        assert result["completed"] is True
