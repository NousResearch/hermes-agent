"""Regression tests for first_chunk_at in the post_api_request hook (#98555).

interruptible_streaming_api_call records ``first_chunk_at`` in its per-attempt
stream diagnostics; on stream success the value is stashed on the agent as
``_last_api_first_chunk_at`` and forwarded by the conversation loop as the
``first_chunk_at`` kwarg of the ``post_api_request`` plugin hook (TTFB =
first_chunk_at - started_at).

Contract under test:

- A successful streamed response emits a non-null ``first_chunk_at`` in the
  post_api_request payload, refreshed per attempt.
- Non-streamed responses, failed streams, and partial-stream stubs emit None.
- A prior API call's timestamp can never leak into the next call's payload
  (the loop resets ``_last_api_first_chunk_at`` before every attempt).
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tests.agent.test_run_agent import (
    _make_tool_defs,
    _mock_response,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_stream_chunk(content=None, finish_reason=None, model=None, usage=None):
    """Mock streaming chunk matching OpenAI's ChatCompletionChunk shape."""
    delta = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


# Seconds the fake stream spends decoding between its first delta and its terminal frame.
_DECODE_GAP_S = 0.05
# A sleep can wake a few ms early on Windows, so the window lower bound stays loose.
_MIN_MEASURED_S = 0.04


def _make_stream_iterator(chunks, decode_gap_s=_DECODE_GAP_S):
    """Yield fake chunks with a real decode window between the deltas.

    Windows resolves adjacent clock reads to the same tick, so a back-to-back
    in-memory stream measures a 0.0 s window and the contract records it as
    unmeasured. The gap is the seconds this stream actually spends decoding.
    """
    for index, chunk in enumerate(chunks):
        if index:
            time.sleep(decode_gap_s)
        yield chunk


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked OpenAI client and tool loading."""
    with (
        patch(
            "model_tools.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
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


def _run_with_hooks(agent, message="hi"):
    """Run a conversation with pre/post api hooks recorded; return post calls."""
    hook_calls = []

    def _record_hook(name, **kwargs):
        hook_calls.append((name, kwargs))
        return []

    with (
        patch(
            "hermes_cli.lifecycle.has_hook",
            side_effect=lambda name: name in {"pre_api_request", "post_api_request"},
        ),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=_record_hook),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(message)

    post = [kw for name, kw in hook_calls if name == "post_api_request"]
    return result, post


# ── Streaming helper level ───────────────────────────────────────────────


class TestStreamHelperStashesFirstChunkAt:
    """interruptible_streaming_api_call's success/failure stash behavior."""

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_stream_success_sets_timestamp(self, _mock_close, mock_create, agent):
        chunks = [
            _make_stream_chunk(content="Hello"),
            _make_stream_chunk(content="!", finish_reason="stop", model="test-model"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_stream_iterator(chunks)
        mock_create.return_value = mock_client
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._last_api_first_chunk_at = None  # as the loop does per attempt

        before = time.time()
        response = agent._interruptible_streaming_api_call({})
        after = time.time()

        assert response.choices[0].message.content == "Hello!"
        stamped = agent._last_api_first_chunk_at
        assert isinstance(stamped, float)
        assert before <= stamped <= after
        # Decode window: opened by the first real delta (the "Hello" chunk) and
        # closed when the terminal frame was processed — never by wire-only data.
        decode_seconds = agent._last_api_decode_seconds
        assert isinstance(decode_seconds, float)
        assert _MIN_MEASURED_S <= decode_seconds < 1.0

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_failed_stream_leaves_timestamp_none(
        self, _mock_close, mock_create, agent
    ):
        """A stream that dies before any chunk must not stamp a timestamp."""

        def _dead_stream():
            raise ConnectionError("upstream closed before first byte")
            yield  # pragma: no cover

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _dead_stream()
        mock_create.return_value = mock_client
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._last_api_first_chunk_at = None

        with pytest.raises(Exception):
            agent._interruptible_streaming_api_call({})

        assert agent._last_api_first_chunk_at is None
        assert agent._last_api_decode_seconds is None  # a dead stream never measured a window

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_partial_stream_stub_leaves_timestamp_none(
        self, _mock_close, mock_create, agent
    ):
        """A mid-stream drop returns the partial stub WITHOUT stamping the
        agent — the partial-stub return path precedes the success stash."""

        def _stalling_stream():
            yield _make_stream_chunk(content="partial text")
            raise ConnectionError("connection dropped mid-stream")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _stalling_stream()
        mock_create.return_value = mock_client
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._last_api_first_chunk_at = None
        # Register a stream consumer so deltas are marked as delivered,
        # routing the failure through the partial-stream stub path.
        agent.stream_delta_callback = lambda _text: None

        response = agent._interruptible_streaming_api_call({})

        from hermes_constants import PARTIAL_STREAM_STUB_ID

        assert response.id == PARTIAL_STREAM_STUB_ID
        assert agent._last_api_first_chunk_at is None
        assert agent._last_api_decode_seconds is None

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_wire_only_chunks_do_not_start_the_decode_window(
        self, _mock_close, mock_create, agent
    ):
        """A role-only wire chunk carries no token: the decode window must not
        open on it — only real delta data counts as the first output token."""

        def _role_only_then_dies():
            yield _make_stream_chunk(model="test-model")  # delta carries no content
            raise ConnectionError("dropped before any token arrived")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _role_only_then_dies()
        mock_create.return_value = mock_client
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._last_api_first_chunk_at = None
        agent.stream_delta_callback = lambda _text: None

        with pytest.raises(Exception):
            agent._interruptible_streaming_api_call({})

        assert agent._last_api_decode_seconds is None

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_reasoning_delta_opens_the_decode_window(
        self, _mock_close, mock_create, agent
    ):
        """Reasoning output is real model output: the window starts at the
        first reasoning delta even when no visible text token ever arrives."""
        think = _make_stream_chunk(model="test-model")
        think.choices[0].delta.reasoning_content = "thinking first"
        end = _make_stream_chunk(finish_reason="stop", model="test-model")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_stream_iterator(
            [think, end]
        )
        mock_create.return_value = mock_client
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._last_api_first_chunk_at = None

        agent._interruptible_streaming_api_call({})

        decode_seconds = agent._last_api_decode_seconds
        assert isinstance(decode_seconds, float)
        assert _MIN_MEASURED_S <= decode_seconds < 1.0


# ── Conversation loop / hook payload level ───────────────────────────────


class TestPostApiRequestFirstChunkAtPayload:
    """The first_chunk_at kwarg observed by post_api_request subscribers."""

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_streamed_success_emits_non_null_first_chunk_at(
        self, _mock_close, mock_create, agent
    ):
        chunks = [
            _make_stream_chunk(content="Hello"),
            _make_stream_chunk(
                content=" world", finish_reason="stop", model="test-model"
            ),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_stream_iterator(chunks)
        mock_create.return_value = mock_client
        # A registered stream consumer forces the streaming path even though
        # agent.client is a Mock.
        agent.stream_delta_callback = lambda _text: None

        result, post = _run_with_hooks(agent)

        assert result["final_response"] == "Hello world"
        assert len(post) == 1
        assert "first_chunk_at" in post[0]
        stamped = post[0]["first_chunk_at"]
        assert isinstance(stamped, float)
        # The payload window covers this call's own gap: measured seconds, never 0.0.
        assert isinstance(post[0]["decode_seconds"], float)
        assert _MIN_MEASURED_S <= post[0]["decode_seconds"] < 1.0
        # TTFB must be derivable and sane: started_at <= first_chunk_at <= ended_at.
        assert post[0]["started_at"] <= stamped <= post[0]["ended_at"]

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_each_streamed_attempt_emits_its_own_timestamp(
        self, _mock_close, mock_create, agent
    ):
        """Two sequential streamed calls each report their own (monotonic)
        first-chunk timestamp — the payload reflects the latest attempt."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            iter([
                _make_stream_chunk(content="first"),
                _make_stream_chunk(
                    content=" answer", finish_reason="stop", model="test-model"
                ),
            ]),
            iter([
                _make_stream_chunk(content="second"),
                _make_stream_chunk(
                    content=" answer", finish_reason="stop", model="test-model"
                ),
            ]),
        ]
        mock_create.return_value = mock_client
        agent.stream_delta_callback = lambda _text: None

        _, post_a = _run_with_hooks(agent, "first question")
        _, post_b = _run_with_hooks(agent, "second question")

        assert len(post_a) == 1 and len(post_b) == 1
        first, second = post_a[0]["first_chunk_at"], post_b[0]["first_chunk_at"]
        assert isinstance(first, float) and isinstance(second, float)
        assert second >= first

    def test_non_streamed_response_emits_none(self, agent):
        """Non-streaming path (Mock client, no stream consumers) → None."""
        agent.client.chat.completions.create.return_value = _mock_response(
            content="Done", finish_reason="stop"
        )

        result, post = _run_with_hooks(agent)

        assert result["final_response"] == "Done"
        assert len(post) == 1
        assert "first_chunk_at" in post[0]
        assert post[0]["first_chunk_at"] is None
        assert post[0]["decode_seconds"] is None

    def test_prior_attempt_timestamp_cannot_leak_into_next_call(self, agent):
        """A stale timestamp left on the agent by an earlier API call must be
        reset before the next attempt — a non-streamed follow-up call reports
        None, never the previous call's value."""
        agent._last_api_first_chunk_at = 1756000000.512  # stale prior attempt
        agent._last_api_decode_seconds = 9.9  # stale prior decode window
        agent.client.chat.completions.create.return_value = _mock_response(
            content="Fresh answer", finish_reason="stop"
        )

        result, post = _run_with_hooks(agent)

        assert result["final_response"] == "Fresh answer"
        assert len(post) == 1
        assert post[0]["first_chunk_at"] is None
        assert post[0]["decode_seconds"] is None
        assert agent._last_api_first_chunk_at is None
        assert agent._last_api_decode_seconds is None

    @pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_decode_seconds_emit_none(self, agent, invalid):
        from agent.turn_response_intake import _sanitize_decode_seconds

        assert _sanitize_decode_seconds(invalid) is None
