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




# ── Clean stream-end before any argument byte arrives (#80498) ─────────────

class TestCleanStreamEndBeforeAnyToolArgs:
    """The upstream closes the SSE stream cleanly right after delivering the
    tool NAME — not a single byte of the arguments delta ever arrived, no
    exception, no finish_reason, no [DONE].

    Before the fix, an empty ``arguments`` string skipped the
    truncated-JSON check entirely (it only ran when ``arguments and
    arguments.strip()``), so ``has_truncated_tool_args`` stayed False. With
    no other guard catching this shape, the stub-builder fell through to
    ``effective_finish_reason = finish_reason or "stop"`` and returned a
    normal "stop" turn carrying a tool call with ``arguments=""`` — which
    the dispatch boundary silently coerces to "{}" and executes with no
    retry (#80498, e.g. ``write_file`` running with no arguments).
    """

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_empty_tool_args_routes_to_stub_not_silent_empty_object(
        self, _mock_close, mock_create, monkeypatch,
    ):
        def _clean_ending_stream():
            # Tool name arrives, then the generator simply RETURNS
            # (StopIteration) before any arguments delta chunk — no raise,
            # no finish_reason chunk, no [DONE].
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_x", name="write_file"),
            ])
            # falls off the end — clean close, no terminator, zero args bytes

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = (
            lambda *a, **kw: _clean_ending_stream()
        )
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._fire_stream_delta = lambda text: None

        response = agent._interruptible_streaming_api_call({})

        assert response.id == PARTIAL_STREAM_STUB_ID, (
            "A tool call whose arguments never started streaming before a "
            "clean stream end must be tagged as a partial-stream stub, not "
            "silently returned as a completed 'stop' turn with empty "
            "arguments (#80498)."
        )
        assert response.choices[0].finish_reason == FINISH_REASON_LENGTH
        assert response.choices[0].message.tool_calls is None, (
            "A tool call with zero argument bytes delivered must never "
            "auto-execute with a silently substituted empty object."
        )
        assert getattr(response, "_dropped_tool_names", None) == ["write_file"]


# ── Mixed response: one complete call, one dropped (#80498) ─────────────────

class TestMixedToolCallsOneDroppedOneComplete:
    """A single response can carry more than one tool call in parallel. If
    the stream dies before one call's arguments ever start while a sibling
    call in the SAME response already completed with valid, parseable
    arguments, the whole response is still discarded as one partial-stream
    stub — ``_build_partial_stream_stub`` unconditionally sets
    ``tool_calls=None``, so the valid sibling is not selectively kept.

    This all-or-nothing behavior predates #80498 (it already applied to the
    partial-but-nonempty-JSON case); this test just locks it in now that the
    zero-byte case routes through the same stub path, so a future change
    towards per-tool-call granularity doesn't silently regress either case.
    """

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_valid_sibling_tool_call_is_discarded_with_dropped_one(
        self, _mock_close, mock_create, monkeypatch,
    ):
        def _clean_ending_stream():
            # Tool call 0: complete, valid JSON arguments.
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_ok", name="read_file"),
            ])
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments='{"path": "a.txt"}'),
            ])
            # Tool call 1: name arrives, then the stream dies before a
            # single argument byte — the #80498 zero-byte case.
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=1, tc_id="call_x", name="write_file"),
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

        assert response.id == PARTIAL_STREAM_STUB_ID
        assert response.choices[0].finish_reason == FINISH_REASON_LENGTH
        assert response.choices[0].message.tool_calls is None, (
            "The valid 'read_file' call must not be selectively kept — the "
            "whole response is discarded as one stub so the retry "
            "regenerates every call in it together."
        )
        assert getattr(response, "_dropped_tool_names", None) == [
            "read_file", "write_file",
        ], (
            "_dropped_tool_names lists every tool call in the response, "
            "not just the one that actually failed to complete."
        )


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


    def test_tagged_stub_activates_fallback_first_pass(self, loop_agent):
        """Layer 3: a tagged stub activates fallback on the FIRST pass, with
        zero continuation retries burned, and the fallback provider then
        completes the turn."""
        from tests.run_agent.test_run_agent import _mock_assistant_msg, _mock_response

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



class TestEmptyPartialStreamStubNotPersisted:
    """Regression for the session-poisoning bug hit with moonshotai/kimi-k3
    via OpenRouter (2026-07-20): a stream dropped mid-``write_file`` tool
    call before ANY text was delivered.  The partial-stream-stub carries
    ``content=""`` and ``tool_calls=None``, so the loop's truncation path
    took the "no tool calls" branch and appended
    ``{"role": "assistant", "content": ""}`` to history before the
    continuation user-message.  Moonshot rejects empty assistant content
    ("the message at position N with role 'assistant' must not be empty")
    with HTTP 400 on the very next replay — and since the message is
    persisted, EVERY subsequent turn re-fails: session unrecoverable.

    Fix layer 1 (conversation_loop): an empty partial-stream stub must not
    be appended as an interim assistant message — only the continuation
    user-message is.
    """

    def test_empty_stub_only_appends_continuation_user_message(self, loop_agent):
        from tests.run_agent.test_run_agent import _mock_response, _mock_assistant_msg

        # First API call: empty partial-stream stub — stream died mid
        # tool-call args with zero text delivered.
        empty_stub = SimpleNamespace(
            id=PARTIAL_STREAM_STUB_ID,
            model="test/model",
            choices=[SimpleNamespace(
                index=0,
                message=_mock_assistant_msg(content=""),
                finish_reason=FINISH_REASON_LENGTH,
            )],
            usage=None,
            _dropped_tool_names=["write_file"],
        )
        # Second API call: the model answers normally after the nudge.
        recovery = _mock_response(content="Done — wrote it in chunks.",
                                  finish_reason="stop")

        loop_agent.client.chat.completions.create.side_effect = [
            empty_stub, recovery,
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("make me a webpage")

        assert loop_agent.client.chat.completions.create.call_count == 2

        # Inspect the history replayed on the SECOND call: there must be NO
        # empty-content assistant message anywhere — that is the exact shape
        # Moonshot 400s on.
        second_call_kwargs = loop_agent.client.chat.completions.create.call_args_list[1]
        msgs = second_call_kwargs.kwargs.get("messages") or second_call_kwargs.args[0].get("messages")
        empty_assistants = [
            m for m in msgs
            if m.get("role") == "assistant" and not m.get("content")
        ]
        assert empty_assistants == [], (
            "Empty partial-stream stub must not be persisted as an "
            "empty-content assistant message — strict providers (Moonshot/"
            "Kimi) reject the replay with HTTP 400 and poison the session."
        )

        # The continuation nudge is still appended as a user message, and
        # it's the chunking variant (dropped tool call), not the length lie.
        last_user = next(
            (m for m in reversed(msgs) if m.get("role") == "user"), None,
        )
        assert last_user is not None
        assert "too large" in (last_user.get("content") or "")
        assert "output length limit" not in (last_user.get("content") or "")

        assert result["completed"] is True



class TestBuildAssistantMessageEmptyContentPad:
    """Layer 2 was consolidated into the class owner: the builder stores
    textless turns AS-IS (no write-time pad — a pad here broke codex
    commentary turns and forked the concept).  Wire safety is owned by
    ``repair_empty_non_final_messages`` inside ``sanitize_api_messages``.
    These tests pin the builder's store-as-is contract."""

    def _agent_for_builder(self):
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
        return a

    def test_empty_content_stored_as_is(self):
        from agent.chat_completion_helpers import build_assistant_message
        from tests.run_agent.test_run_agent import _mock_assistant_msg

        agent = self._agent_for_builder()
        msg = build_assistant_message(agent, _mock_assistant_msg(content=""), "stop")
        assert msg["content"] == "", (
            "Builder must store textless turns as-is — wire repair is owned "
            "by repair_empty_non_final_messages at the send boundary."
        )


    def test_tool_call_turn_content_left_empty(self):
        from agent.chat_completion_helpers import build_assistant_message
        from tests.run_agent.test_run_agent import _mock_assistant_msg, _mock_tool_call

        agent = self._agent_for_builder()
        msg = build_assistant_message(
            agent,
            _mock_assistant_msg(content="", tool_calls=[_mock_tool_call()]),
            "tool_calls",
        )
        assert msg["content"] == ""
        assert msg["tool_calls"]

    def test_non_empty_content_unchanged(self):
        from agent.chat_completion_helpers import build_assistant_message
        from tests.run_agent.test_run_agent import _mock_assistant_msg

        agent = self._agent_for_builder()
        msg = build_assistant_message(agent, _mock_assistant_msg(content="hi"), "stop")
        assert msg["content"] == "hi"


class TestSendTimeEmptyAssistantPad:
    """Durable repair for ALREADY-poisoned persisted sessions: a partial
    -stream-stub row written by an older build (content:'' ,
    finish_reason:'length') is rebuilt to content:'' on every reload —
    ``_rows_to_conversation`` strips whitespace, so a DB-side pad cannot
    survive.  The class owner ``repair_empty_non_final_messages`` (inside
    ``sanitize_api_messages``, the pre-send chokepoint) must repair the
    empty textless assistant turn at the serialization boundary, so a
    RESUMED poisoned session replays cleanly against strict providers
    (Moonshot/Kimi HTTP 400 "message ... with role 'assistant' must not
    be empty" / Anthropic "all messages must have non-empty content")."""

    def _run_one_turn_with_history(self, loop_agent, history):
        from tests.run_agent.test_run_agent import _mock_response
        loop_agent.client.chat.completions.create.return_value = _mock_response(
            content="ok", finish_reason="stop",
        )
        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            loop_agent.run_conversation(
                "continue", conversation_history=history,
            )
        kwargs = loop_agent.client.chat.completions.create.call_args_list[0]
        return kwargs.kwargs.get("messages") or kwargs.args[0].get("messages")

    def test_poisoned_resumed_history_repaired_on_send(self, loop_agent):
        # Byte-shape of a persisted poisoned session:
        # user -> assistant('' , finish_reason='length', NO tool_calls) -> user.
        poisoned = [
            {"role": "user", "content": "make me a webpage"},
            {"role": "assistant", "content": "", "finish_reason": "length"},
            {"role": "user", "content": "please proceed"},
        ]
        sent = self._run_one_turn_with_history(loop_agent, poisoned)
        empties = [
            m for m in sent
            if m.get("role") == "assistant"
            and not m.get("tool_calls")
            and not (m.get("content") or "").strip()
        ]
        assert empties == [], (
            "A resumed session carrying a persisted empty partial-stream "
            "stub must be repaired at the send boundary — strict providers "
            "reject the replay with HTTP 400 otherwise."
        )
        stub = next(
            (m for m in sent if m.get("role") == "assistant"
             and not m.get("tool_calls")),
            None,
        )
        assert stub is not None and stub["content"] == "[response interrupted]"

    def test_tool_call_turn_not_padded_on_send(self, loop_agent):
        history = [
            {"role": "user", "content": "search something"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "and now?"},
        ]
        sent = self._run_one_turn_with_history(loop_agent, history)
        tc_turn = next(
            (m for m in sent if m.get("role") == "assistant" and m.get("tool_calls")),
            None,
        )
        assert tc_turn is not None
        assert tc_turn["content"] == "", (
            "Tool-call turns are exempt from the pad: content:'' alongside "
            "tool_calls is accepted by every provider and normalizing it "
            "would alter prompt-cache keys."
        )


class TestSendTimePadMultimodalSafety:
    """Regression: the send-time repair must skip non-string (list) assistant
    content instead of crashing — a forked session whose new user turn
    attaches an image hit AttributeError: 'list' object has no attribute
    'strip' inside an earlier pad loop.

    The repair is now owned by ``repair_empty_non_final_messages``, whose
    ``_msg_has_payload`` treats a list with any typed block as payload —
    multimodal turns are never rewritten.  This test drives a multimodal
    history through the loop and asserts (a) no crash, and (b) the
    assistant turn's text is neither dropped nor replaced.
    """

    def test_multimodal_assistant_content_not_touched(self, loop_agent):
        from tests.run_agent.test_run_agent import _mock_response
        multimodal = [
            {"role": "user", "content": "look at this"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "I see an image"},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "animate it"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ]},
        ]
        loop_agent.client.chat.completions.create.return_value = _mock_response(
            content="ok", finish_reason="stop",
        )
        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation(
                "animate it", conversation_history=multimodal,
            )
        assert result["completed"] is True
        kwargs = loop_agent.client.chat.completions.create.call_args_list[0]
        sent = kwargs.kwargs.get("messages") or kwargs.args[0].get("messages")
        # The assistant turn survives with its text intact — regardless of
        # whether upstream passes flattened it to a str or kept the list.
        mm = next(
            m for m in sent
            if m.get("role") == "assistant" and not m.get("tool_calls")
        )
        c = mm["content"]
        if isinstance(c, list):
            assert c == [{"type": "text", "text": "I see an image"}], (
                "Multimodal assistant list content must pass through untouched."
            )
        else:
            assert "I see an image" in (c or ""), (
                "Flattened multimodal assistant text must survive the repair."
            )

    def test_repair_owner_skips_list_content_directly(self):
        """Unit-shape check against the REAL owner: multimodal list content
        (the exact AttributeError shape) passes through untouched; a textless
        str turn is repaired; tool-call turns are exempt."""
        from agent.agent_runtime_helpers import repair_empty_non_final_messages
        api_messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "user", "content": "trailing turn keeps the above non-final"},
        ]
        out = repair_empty_non_final_messages(api_messages)
        assert out[0]["content"] == [{"type": "text", "text": "hi"}]
        assert out[1]["content"] == "[response interrupted]"
        assert out[2]["content"] == ""
        # input list untouched (repair is copy-on-write)
        assert api_messages[1]["content"] == ""
