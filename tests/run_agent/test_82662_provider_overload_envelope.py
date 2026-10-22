"""Regression tests for issue #82662 — provider error envelope in final response.

Some providers report failures as ordinary assistant text instead of a
structured error / SSE frame: "[Error] Our servers are currently overloaded.
Please try again later." arrives as the message content of a chat completion
with finish_reason="stop". Before this fix Hermes treated it as a successful
final answer (delivered verbatim on Telegram), the turn ended, and NO
retry/fallback/recovery ever ran.  Distinct from #58017 (structured SSE
frames) which does not cover inline text.

Pins the contract:

- ``looks_like_provider_error_envelope`` is CONSERVATIVE: short text + a
  leading "[Error]" tag + a clear service-failure marker are all required;
  "[Error] The file does not exist. Please try again." stays False.
- The conversation loop marks such responses invalid BEFORE finish-reason
  normalization, so the existing retry/fallback path runs and the envelope
  is never persisted as a successful answer (finish_reason="stop" does NOT
  override text detection).
- The gateway rewrites the envelope to a safe provider-failure reply instead
  of delivering it raw to chat surfaces (defense in depth).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.chat_completion_helpers import (
    looks_like_provider_error_envelope,
    _could_be_provider_error_envelope_prefix,
    _StreamEnvelopeBuffer,
    interruptible_streaming_api_call,
)
from gateway.config import Platform
from gateway.run import (
    _looks_like_gateway_provider_error,
    _sanitize_gateway_final_response,
)

OVERLOAD_ENVELOPE = (
    "[Error] Our servers are currently overloaded. Please try again later."
)


def _mock_response(content="Hello", finish_reason="stop"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _mock_chunk(content=None, finish_reason=None, tool_calls=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_tool_defs(*names: str) -> list:
    """Build minimal tool definition list accepted by AIAgent.__init__."""
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked OpenAI client and tool loading."""
    with (
        patch(
            "run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")
        ),
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
        return a


def _setup_agent(agent):
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False


def _make_fast_time_mock():
    """Return a mock time module where sleep loops exit instantly."""
    mock_time = MagicMock()
    _t = [1000.0]

    def _advancing_time():
        _t[0] += 500.0  # jump 500s per call so sleep_end is always in the past
        return _t[0]

    mock_time.time.side_effect = _advancing_time
    mock_time.sleep = MagicMock()  # no-op
    mock_time.monotonic.return_value = 12345.0
    return mock_time


# ---------------------------------------------------------------------------
# Unit — conservative envelope detector
# ---------------------------------------------------------------------------


class TestProviderErrorEnvelopeDetector:
    def test_exact_envelope_detected(self):
        assert looks_like_provider_error_envelope(OVERLOAD_ENVELOPE) is True

    def test_uppercase_envelope_detected(self):
        assert (
            looks_like_provider_error_envelope(
                "[ERROR] OUR SERVERS ARE CURRENTLY OVERLOADED. "
                "PLEASE TRY AGAIN LATER."
            )
            is True
        )

    def test_symbol_prefix_envelope_detected(self):
        assert looks_like_provider_error_envelope(f"* {OVERLOAD_ENVELOPE}") is True
        assert looks_like_provider_error_envelope(f"> {OVERLOAD_ENVELOPE}") is True

    @pytest.mark.parametrize(
        "variant",
        [
            "[Error] The service is at capacity right now. Please retry shortly.",
            "[Error] Too many requests. Slow down and try again.",
            "[Error] Our service is temporarily unavailable. Try again later.",
            "[Error] Internal server error. Please try again later.",
            "[Error] Service unavailable. Try again in a few minutes.",
        ],
    )
    def test_service_marker_variants_detected(self, variant):
        assert looks_like_provider_error_envelope(variant) is True

    def test_normal_content_not_detected(self):
        assert (
            looks_like_provider_error_envelope("Here is the answer to your question.")
            is False
        )

    def test_error_tag_without_marker_not_detected(self):
        assert (
            looks_like_provider_error_envelope(
                "[Error] The file does not exist. Please try again."
            )
            is False
        )

    def test_error_tag_file_not_found_not_detected(self):
        assert (
            looks_like_provider_error_envelope(
                "[Error] I could not find the file you asked about."
            )
            is False
        )

    def test_long_text_mentioning_overloaded_not_detected(self):
        long_text = (
            "The provider reported that its servers were overloaded for a "
            "moment, but here is the full explanation of what happened and "
            "what it means for your request. "
            * 5
        )
        assert len(long_text) > 400
        assert looks_like_provider_error_envelope(long_text) is False

    @pytest.mark.parametrize("empty", ["", "   ", None])
    def test_empty_and_none_not_detected(self, empty):
        assert looks_like_provider_error_envelope(empty) is False


# ---------------------------------------------------------------------------
# Integration — conversation loop routes the envelope through retry/fallback
# ---------------------------------------------------------------------------


class TestProviderOverloadEnvelopeConversationLoop:
    def test_overload_envelope_retries_and_recovers(self, agent):
        """First attempt returns the overload envelope → retried → recovered."""
        _setup_agent(agent)
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content=OVERLOAD_ENVELOPE),
            _mock_response(content="recovered"),
        ]
        relay_attempts = []
        logical_completions = []

        def execute(request, callback, **kwargs):
            relay_attempts.append(kwargs)
            return callback(request)

        from agent import conversation_loop as _conv_loop

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time", _make_fast_time_mock()),
            patch.object(_conv_loop, "time", _make_fast_time_mock()),
            patch("agent.relay_llm.execute", side_effect=execute),
            patch(
                "agent.relay_llm.complete_logical_call",
                side_effect=lambda request_id, *, outcome: logical_completions.append(
                    (request_id, outcome)
                ),
            ),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert len(relay_attempts) == 2
        assert "recovered" in result["final_response"]
        request_ids = {
            attempt["metadata"]["api_request_id"] for attempt in relay_attempts
        }
        assert len(request_ids) == 1
        assert logical_completions == [(request_ids.pop(), "success")]

    def test_overload_envelope_every_attempt_fails_turn(self, agent):
        """Envelope on every attempt → retries exhaust → turn fails, never a success."""
        _setup_agent(agent)
        agent.client.chat.completions.create.return_value = _mock_response(
            content=OVERLOAD_ENVELOPE
        )
        relay_attempts = []
        hook_messages = []

        def execute(request, callback, **kwargs):
            relay_attempts.append(kwargs)
            return callback(request)

        def record_hook(**kwargs):
            hook_messages.append(kwargs.get("error_message", ""))

        from agent import conversation_loop as _conv_loop

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_invoke_api_request_error_hook", side_effect=record_hook),
            patch("run_agent.time", _make_fast_time_mock()),
            patch.object(_conv_loop, "time", _make_fast_time_mock()),
            patch("agent.relay_llm.execute", side_effect=execute),
        ):
            result = agent.run_conversation("hello")

        assert result.get("completed") is False
        assert result.get("failed") is True
        assert "error" in result
        assert "Invalid API response" in result["error"]
        # The envelope was never accepted: every attempt went through the
        # invalid-response path (proved by the error hook receiving the
        # envelope detail + multiple relay attempts).
        assert len(relay_attempts) >= 2
        assert any("provider error envelope" in m for m in hook_messages)

    def test_normal_stop_response_unchanged_single_attempt(self, agent):
        """Ordinary finish_reason=stop content is untouched: 1 attempt, success."""
        _setup_agent(agent)
        agent.client.chat.completions.create.return_value = _mock_response(
            content="A perfectly normal answer."
        )
        relay_attempts = []

        def execute(request, callback, **kwargs):
            relay_attempts.append(kwargs)
            return callback(request)

        from agent import conversation_loop as _conv_loop

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time", _make_fast_time_mock()),
            patch.object(_conv_loop, "time", _make_fast_time_mock()),
            patch("agent.relay_llm.execute", side_effect=execute),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert len(relay_attempts) == 1
        assert "A perfectly normal answer." in result["final_response"]

    def test_error_tag_without_marker_is_success(self, agent):
        """'[Error]' WITHOUT a service marker is legitimate content — no retry."""
        _setup_agent(agent)
        agent.client.chat.completions.create.return_value = _mock_response(
            content="[Error] The file does not exist. Please try again."
        )
        relay_attempts = []

        def execute(request, callback, **kwargs):
            relay_attempts.append(kwargs)
            return callback(request)

        from agent import conversation_loop as _conv_loop

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time", _make_fast_time_mock()),
            patch.object(_conv_loop, "time", _make_fast_time_mock()),
            patch("agent.relay_llm.execute", side_effect=execute),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert len(relay_attempts) == 1
        assert "[Error] The file does not exist." in result["final_response"]


# ---------------------------------------------------------------------------
# Gateway — defense in depth: never deliver the envelope raw to chat surfaces
# ---------------------------------------------------------------------------


class TestProviderErrorEnvelopeGatewayDefense:
    def test_gateway_detector_true_for_envelope(self):
        assert _looks_like_gateway_provider_error(OVERLOAD_ENVELOPE) is True

    def test_gateway_detector_false_for_normal(self):
        assert _looks_like_gateway_provider_error("A perfectly normal answer.") is False
        assert (
            _looks_like_gateway_provider_error(
                "[Error] The file does not exist. Please try again."
            )
            is False
        )

    def test_gateway_sanitize_rewrites_envelope_to_safe_reply(self):
        sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, OVERLOAD_ENVELOPE)
        # The raw envelope must never reach a chat surface...
        assert "overloaded" not in sanitized
        assert "try again later" not in sanitized
        # ...and the rewrite must be the safe provider-failure reply.
        assert "model provider" in sanitized


# ---------------------------------------------------------------------------
# Streaming unit and integration — prefix detection and buffering (_StreamEnvelopeBuffer)
# ---------------------------------------------------------------------------


class TestProviderErrorEnvelopePrefix:
    @pytest.mark.parametrize("empty", ["", "   ", None])
    def test_empty_and_none_prefix_is_candidate(self, empty):
        assert _could_be_provider_error_envelope_prefix(empty) is True

    @pytest.mark.parametrize(
        "candidate",
        [
            "[",
            "[e",
            "[E",
            "[err",
            "[Error]",
            "[ERROR]",
            "* [Error]",
            "> [error]",
            "— [error]",
            "   [error]",
        ],
    )
    def test_valid_error_prefixes_are_candidates(self, candidate):
        assert _could_be_provider_error_envelope_prefix(candidate) is True

    @pytest.mark.parametrize(
        "divergent",
        [
            "Hello",
            "Sure, I can help you with that.",
            "[Info] Starting task...",
            "[Warning] High memory usage.",
            "The error occurred in line 4.",
        ],
    )
    def test_divergent_prefixes_are_rejected(self, divergent):
        assert _could_be_provider_error_envelope_prefix(divergent) is False

    def test_too_long_or_too_many_newlines_rejected(self):
        long_prefix = "[" + "e" * 405
        assert _could_be_provider_error_envelope_prefix(long_prefix) is False

        multiline_prefix = "[\n\n\n\n\nerror]"
        assert _could_be_provider_error_envelope_prefix(multiline_prefix) is False


class TestStreamEnvelopeBufferUnit:
    def test_divergent_stream_flushes_all_and_fires_first_delta(self):
        first_fired = []
        stream_deltas = []
        flag = {"yes": False}
        buf = _StreamEnvelopeBuffer(
            lambda: first_fired.append(True),
            stream_deltas.append,
            flag,
        )

        # Feed "[" which matches prefix -> buffering continues
        buf.feed("[")
        assert first_fired == []
        assert stream_deltas == []
        assert flag["yes"] is False

        # Feed "He" -> combined is "[He" which diverges from "[error]"!
        buf.feed("He")
        assert len(first_fired) == 1
        assert stream_deltas == ["[", "He"]
        assert flag["yes"] is True

        # Subsequent deltas pass through directly
        buf.feed("llo")
        assert stream_deltas == ["[", "He", "llo"]

    def test_exact_envelope_drops_buffer_on_finish(self):
        first_fired = []
        stream_deltas = []
        flag = {"yes": False}
        buf = _StreamEnvelopeBuffer(
            lambda: first_fired.append(True),
            stream_deltas.append,
            flag,
        )

        buf.feed("[Error] Our servers are ")
        buf.feed("currently overloaded. ")
        buf.feed("Please try again later.")

        assert first_fired == []
        assert stream_deltas == []

        buf.finish()
        # Full envelope was matched; buffer is dropped, never delivered!
        assert first_fired == []
        assert stream_deltas == []
        assert flag["yes"] is False

    def test_non_envelope_error_tag_flushes_on_finish(self):
        first_fired = []
        stream_deltas = []
        flag = {"yes": False}
        buf = _StreamEnvelopeBuffer(
            lambda: first_fired.append(True),
            stream_deltas.append,
            flag,
        )

        buf.feed("[Error] The file ")
        buf.feed("does not exist.")

        assert first_fired == []
        assert stream_deltas == []

        buf.finish()
        # Not a service failure envelope -> flushes completely on finish
        assert len(first_fired) == 1
        assert stream_deltas == ["[Error] The file ", "does not exist."]
        assert flag["yes"] is True

    def test_tool_call_flushes_buffer(self):
        first_fired = []
        stream_deltas = []
        buf = _StreamEnvelopeBuffer(
            lambda: first_fired.append(True),
            stream_deltas.append,
        )

        buf.feed("[")
        assert first_fired == []
        assert stream_deltas == []

        buf.flush()
        assert len(first_fired) == 1
        assert stream_deltas == ["["]


class TestStreamingProviderOverloadEnvelopeIntegration:
    def test_streaming_multi_chunk_overload_envelope_not_leaked(self, agent):
        """Reviewer finding: multi-chunk provider overload envelope must not leak into stream callbacks."""
        _setup_agent(agent)
        stream_deltas = []
        first_delta_calls = []

        agent.stream_delta_callback = stream_deltas.append

        chunks = [
            _mock_chunk(content="[Error] Our servers are "),
            _mock_chunk(content="currently overloaded. Please try again later.", finish_reason="stop"),
        ]

        class _MockStream:
            final_response = None
            output_modified = False

            def __iter__(self):
                return iter(chunks)

            def close(self):
                pass

        with (
            patch("agent.chat_completion_helpers.should_use_direct_api_call", return_value=False),
            patch("agent.relay_llm.stream", return_value=_MockStream()),
        ):
            resp = interruptible_streaming_api_call(
                agent,
                {"messages": [{"role": "user", "content": "hi"}]},
                on_first_delta=lambda: first_delta_calls.append(True),
            )

        assert stream_deltas == []
        assert first_delta_calls == []
        assert resp.choices[0].message.content == OVERLOAD_ENVELOPE

    def test_completed_response_overload_envelope_not_leaked(self, agent):
        """Completed-response-instead-of-stream branch must not emit overload envelope."""
        _setup_agent(agent)
        stream_deltas = []
        first_delta_calls = []

        agent.stream_delta_callback = stream_deltas.append

        completed_resp = _mock_response(content=OVERLOAD_ENVELOPE)

        class _MockCompletedStream:
            final_response = completed_resp
            output_modified = False

            def __iter__(self):
                return iter([])

            def close(self):
                pass

        with (
            patch("agent.chat_completion_helpers.should_use_direct_api_call", return_value=False),
            patch("agent.relay_llm.stream", return_value=_MockCompletedStream()),
        ):
            resp = interruptible_streaming_api_call(
                agent,
                {"messages": [{"role": "user", "content": "hi"}]},
                on_first_delta=lambda: first_delta_calls.append(True),
            )

        assert stream_deltas == []
        assert first_delta_calls == []
        assert resp.choices[0].message.content == OVERLOAD_ENVELOPE

    def test_streaming_normal_content_emits_deltas(self, agent):
        """Normal multi-chunk stream emits deltas and fires on_first_delta."""
        _setup_agent(agent)
        stream_deltas = []
        first_delta_calls = []

        agent.stream_delta_callback = stream_deltas.append

        chunks = [
            _mock_chunk(content="Hello "),
            _mock_chunk(content="world!", finish_reason="stop"),
        ]

        class _MockNormalStream:
            final_response = None
            output_modified = False

            def __iter__(self):
                return iter(chunks)

            def close(self):
                pass

        with (
            patch("agent.chat_completion_helpers.should_use_direct_api_call", return_value=False),
            patch("agent.relay_llm.stream", return_value=_MockNormalStream()),
        ):
            resp = interruptible_streaming_api_call(
                agent,
                {"messages": [{"role": "user", "content": "hi"}]},
                on_first_delta=lambda: first_delta_calls.append(True),
            )

        assert stream_deltas == ["Hello ", "world!"]
        assert len(first_delta_calls) == 1
        assert resp.choices[0].message.content == "Hello world!"

    def test_streaming_error_tag_without_marker_emits_deltas(self, agent):
        """Legitimate error tag without service marker must flush and emit deltas."""
        _setup_agent(agent)
        stream_deltas = []
        first_delta_calls = []

        agent.stream_delta_callback = stream_deltas.append

        chunks = [
            _mock_chunk(content="[Error] The file "),
            _mock_chunk(content="does not exist.", finish_reason="stop"),
        ]

        class _MockErrorStream:
            final_response = None
            output_modified = False

            def __iter__(self):
                return iter(chunks)

            def close(self):
                pass

        with (
            patch("agent.chat_completion_helpers.should_use_direct_api_call", return_value=False),
            patch("agent.relay_llm.stream", return_value=_MockErrorStream()),
        ):
            resp = interruptible_streaming_api_call(
                agent,
                {"messages": [{"role": "user", "content": "hi"}]},
                on_first_delta=lambda: first_delta_calls.append(True),
            )

        assert stream_deltas == ["[Error] The file ", "does not exist."]
        assert len(first_delta_calls) == 1
        assert resp.choices[0].message.content == "[Error] The file does not exist."
