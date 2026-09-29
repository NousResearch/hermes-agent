"""test_run_agent_streaming: split from the former tests/agent/test_run_agent.py monolith (#99982)."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from tests.agent._run_agent_helpers import _make_chunk, _make_tc_delta, _mock_response


def _provider_sse_429_text(
    code="Throttling.AllocationQuota",
    message="Allocated quota exceeded.",
):
    return (
        "id:1\n"
        "event:error\n"
        ":HTTP_STATUS/429\n"
        f'data:{{"request_id":"req-123","code":"{code}","message":"{message}"}}'
    )

def _provider_sse_error_text(status=503, code="ServiceUnavailable", message="Busy"):
    return (
        "event: error\n"
        f'data:{{"status":{status},"request_id":"req-456","code":"{code}",'
        f'"message":"{message}"}}'
    )

def _provider_bare_sse_error_text(
    code="rate_limit_exceeded",
    message="Rate limit exceeded.",
):
    return f'data: {{"error":{{"code":"{code}","message":"{message}"}}}}\n'


class TestStreamingApiCall:
    """Tests for _streaming_api_call — voice TTS streaming pipeline."""

    def test_content_assembly(self, agent):
        chunks = [
            _make_chunk(content="Hel"),
            _make_chunk(content="lo "),
            _make_chunk(content="World"),
            _make_chunk(finish_reason="stop"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)
        callback = MagicMock()
        agent.stream_delta_callback = callback

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == "Hello World"
        assert resp.choices[0].finish_reason == "stop"
        assert callback.call_count == 3
        callback.assert_any_call("Hel")
        callback.assert_any_call("lo ")
        callback.assert_any_call("World")

    def test_error_finish_http_status_429_stream_raises_rate_limit(self, agent):
        error_text = _provider_sse_429_text()
        chunks = [
            _make_chunk(content=error_text[:5]),
            _make_chunk(content=error_text[5:]),
            _make_chunk(finish_reason="error_finish"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)
        agent.stream_delta_callback = MagicMock()

        with pytest.raises(Exception) as exc_info:
            agent._interruptible_streaming_api_call({"messages": []})

        exc = exc_info.value
        assert getattr(exc, "status_code", None) == 429
        assert "Throttling.AllocationQuota" in str(exc)
        assert getattr(exc, "body", {})["error"]["code"] == "Throttling.AllocationQuota"
        agent.stream_delta_callback.assert_not_called()

    def test_error_finish_sse_data_status_raises_provider_status(self, agent):
        chunks = [
            _make_chunk(content=_provider_sse_error_text()),
            _make_chunk(finish_reason="error_finish"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)
        agent.stream_delta_callback = MagicMock()

        with pytest.raises(Exception) as exc_info:
            agent._interruptible_streaming_api_call({"messages": []})

        exc = exc_info.value
        assert getattr(exc, "status_code", None) == 503
        assert getattr(exc, "body", {})["error"]["code"] == "ServiceUnavailable"
        assert "Busy" in str(exc)
        agent.stream_delta_callback.assert_not_called()

    def test_error_finish_bare_sse_error_payload_raises_provider_error(self, agent):
        chunks = [
            _make_chunk(content=_provider_bare_sse_error_text()),
            _make_chunk(finish_reason="error_finish"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)
        agent.stream_delta_callback = MagicMock()

        with pytest.raises(Exception) as exc_info:
            agent._interruptible_streaming_api_call({"messages": []})

        exc = exc_info.value
        assert getattr(exc, "status_code", None) is None
        assert getattr(exc, "body", {})["error"]["code"] == "rate_limit_exceeded"
        assert "Rate limit exceeded" in str(exc)
        agent.stream_delta_callback.assert_not_called()

    def test_choiceless_error_chunk_raises_provider_stream_error(self, agent):
        """DeepInfra-style in-stream error: choices=None + error_type/error_message.

        Regression for #65631: the choiceless-chunk skip silently dropped
        error-bearing chunks, the stream ended empty, and the caller got a
        misleading EmptyStreamError plus pointless retries of the same bad
        request. The chunk must instead surface as ProviderStreamError so
        the classifier sees the real provider error.
        """
        err_chunk = SimpleNamespace(
            model="test/model",
            choices=None,
            error_type="400 BadRequestError",
            error_message="context length exceeded",
        )
        agent.client.chat.completions.create.return_value = iter([err_chunk])
        agent.stream_delta_callback = MagicMock()

        with pytest.raises(Exception) as exc_info:
            agent._interruptible_streaming_api_call({"messages": []})

        exc = exc_info.value
        assert type(exc).__name__ == "ProviderStreamError"
        assert getattr(exc, "status_code", None) == 400
        assert "context length exceeded" in str(exc)
        agent.stream_delta_callback.assert_not_called()

    def test_choiceless_usage_only_chunk_still_skipped(self, agent):
        """Usage-only final chunks (choices empty, no error fields) keep flowing."""
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        chunks = [
            _make_chunk(content="Hi"),
            _make_chunk(finish_reason="stop"),
            SimpleNamespace(model="test/model", choices=[], usage=usage),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)
        agent.stream_delta_callback = MagicMock()

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == "Hi"
        assert resp.choices[0].finish_reason == "stop"

    def test_named_non_json_sse_error_preserves_provider_message(self, agent):
        """SDK-level plain-text SSE errors retain their actionable message."""
        import httpx
        from openai import OpenAI, Stream
        from openai.types.chat import ChatCompletionChunk
        from agent.chat_completion_helpers import ProviderStreamError
        from agent.error_classifier import PROVIDER_STREAM_NON_JSON_ERROR_CODE

        provider_message = (
            "request validation failed: unsupported reasoning_effort"
        )
        request = httpx.Request(
            "POST",
            "https://provider.example/v1/chat/completions",
        )
        response = httpx.Response(
            200,
            request=request,
            headers={"x-request-id": "req-plain-text"},
            content=(
                f"event: error\ndata: {provider_message}\n\n"
            ).encode("utf-8"),
        )
        agent.stream_delta_callback = MagicMock()

        with OpenAI(api_key="test-key", max_retries=0) as sdk_client:
            stream = Stream(
                cast_to=ChatCompletionChunk,
                response=response,
                client=sdk_client,
            )
            agent.client.chat.completions.create.return_value = stream

            with pytest.raises(ProviderStreamError) as exc_info:
                agent._interruptible_streaming_api_call({"messages": []})

        exc = exc_info.value
        assert exc.status_code is None
        assert exc.body["error"]["code"] == PROVIDER_STREAM_NON_JSON_ERROR_CODE
        assert exc.body["error"]["message"] == provider_message
        assert exc.raw_text == provider_message
        assert exc.response.headers["x-request-id"] == "req-plain-text"
        assert isinstance(exc.__cause__, json.JSONDecodeError)
        agent.stream_delta_callback.assert_not_called()

    def test_named_non_json_sse_error_force_redacts_secrets(self, agent):
        """SDK-level SSE errors cannot expose credentials in exceptions."""
        import httpx
        from openai import OpenAI, Stream
        from openai.types.chat import ChatCompletionChunk
        from agent.chat_completion_helpers import ProviderStreamError

        secret = "sk-" + ("a" * 48)
        request = httpx.Request(
            "POST",
            "https://provider.example/v1/chat/completions",
        )
        response = httpx.Response(
            200,
            request=request,
            content=(
                "event: error\n"
                f"data: request validation failed: token={secret}\n\n"
            ).encode("utf-8"),
        )
        agent.stream_delta_callback = MagicMock()

        with patch("agent.redact._REDACT_ENABLED", False):
            with OpenAI(api_key="test-key", max_retries=0) as sdk_client:
                stream = Stream(
                    cast_to=ChatCompletionChunk,
                    response=response,
                    client=sdk_client,
                )
                agent.client.chat.completions.create.return_value = stream

                with pytest.raises(ProviderStreamError) as exc_info:
                    agent._interruptible_streaming_api_call({"messages": []})

        assert secret not in str(exc_info.value)
        assert secret not in exc_info.value.raw_text
        assert secret not in exc_info.value.body["error"]["message"]
        assert "sk-" in exc_info.value.body["error"]["message"]
        agent.stream_delta_callback.assert_not_called()

    def test_provider_error_prefix_like_normal_text_flushes_to_callback(self, agent):
        chunks = [
            _make_chunk(content="id: product-42\n"),
            _make_chunk(content="is ready"),
            _make_chunk(finish_reason="stop"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)
        agent.stream_delta_callback = MagicMock()

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == "id: product-42\nis ready"
        assert [
            call.args[0] for call in agent.stream_delta_callback.call_args_list
        ] == ["id: product-42\n", "is ready"]

    def test_full_bailian_sse_error_example_with_stop_is_literal_text(self, agent):
        error_text = _provider_sse_429_text(message="Example error payload.")
        split_at = len(error_text) // 2
        chunks = [
            _make_chunk(content=error_text[:split_at]),
            _make_chunk(content=error_text[split_at:]),
            _make_chunk(finish_reason="stop"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)
        agent.stream_delta_callback = MagicMock()

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == error_text
        assert [
            call.args[0] for call in agent.stream_delta_callback.call_args_list
        ] == [error_text[:split_at], error_text[split_at:]]

    def test_bare_sse_error_payload_with_stop_is_literal_text(self, agent):
        error_text = _provider_bare_sse_error_text(message="Example error payload.")
        chunks = [
            _make_chunk(content=error_text),
            _make_chunk(finish_reason="stop"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)
        agent.stream_delta_callback = MagicMock()

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == error_text
        assert [
            call.args[0] for call in agent.stream_delta_callback.call_args_list
        ] == [error_text]

    def test_bare_sse_error_payload_without_finish_reason_is_literal_text(self, agent):
        error_text = _provider_bare_sse_error_text(message="Example error payload.")
        chunks = [_make_chunk(content=error_text)]
        agent.client.chat.completions.create.return_value = iter(chunks)
        agent.stream_delta_callback = MagicMock()

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == error_text
        # Current main treats every text-only stream without a terminal finish
        # signal as a partial response. The SSE-shaped text remains literal,
        # but is withheld from the callback so the retry path can own delivery.
        assert resp.choices[0].finish_reason == "length"
        agent.stream_delta_callback.assert_not_called()

    def test_run_conversation_retries_stream_error_finish_rate_limit(self, agent):
        first_attempt = iter([
            _make_chunk(content=_provider_sse_429_text()),
            _make_chunk(finish_reason="error_finish"),
        ])
        second_attempt = iter([
            _make_chunk(content="Recovered"),
            _make_chunk(finish_reason="stop"),
        ])
        agent.client.chat.completions.create.side_effect = [first_attempt, second_attempt]
        agent.stream_delta_callback = MagicMock()
        agent._persist_session = lambda *args, **kwargs: None
        agent._save_trajectory = lambda *args, **kwargs: None

        import agent.conversation_loop as _conversation_loop
        import agent.retry_utils as _retry_utils

        with (
            patch.object(_retry_utils, "jittered_backoff", return_value=0.0),
            patch.object(
                _retry_utils,
                "adaptive_rate_limit_backoff",
                return_value=(0.0, None),
            ),
            patch.object(_conversation_loop.time, "sleep", return_value=None),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "Recovered"
        assert agent.client.chat.completions.create.call_count == 2
        assert not any(
            "HTTP_STATUS/429" in str(call.args[0])
            for call in agent.stream_delta_callback.call_args_list
        )

    def test_tool_call_accumulation(self, agent):
        # Per OpenAI streaming spec, function names are delivered atomically
        # in the first chunk; only `arguments` is fragmented across chunks.
        # The accumulator uses assignment for names (immune to MiniMax/NIM
        # resends of the full name) and `+=` for arguments.
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_1", "web_search", '{"q":')]),
            _make_chunk(tool_calls=[_make_tc_delta(0, None, None, '"test"}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 1
        assert tc[0].function.name == "web_search"
        assert tc[0].function.arguments == '{"q":"test"}'
        assert tc[0].id == "call_1"

    def test_multiple_tool_calls(self, agent):
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_a", "search", '{}')]),
            _make_chunk(tool_calls=[_make_tc_delta(1, "call_b", "read", '{}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 2
        assert tc[0].function.name == "search"
        assert tc[1].function.name == "read"

    def test_truncated_tool_call_args_no_finish_reason_routes_to_stub(self, agent):
        # Stream delivers a tool call with incomplete JSON args and then ENDS
        # with no finish_reason (the SSE just stops — no terminator, no
        # [DONE]).  This is an upstream mid-tool-call drop, NOT an output cap.
        # The builder must route it through the partial-stream-stub path
        # (id=PARTIAL_STREAM_STUB_ID, tool_calls=None so it can't execute,
        # finish_reason=length so the loop's continuation machinery fires with
        # chunking guidance) rather than stamping a normal 'length' truncation.
        from hermes_constants import PARTIAL_STREAM_STUB_ID
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_1", "write_file", '{"path":"x.txt","content":"hel')]),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.id == PARTIAL_STREAM_STUB_ID
        assert resp.choices[0].finish_reason == "length"
        assert resp.choices[0].message.tool_calls is None
        assert getattr(resp, "_dropped_tool_names", None) == ["write_file"]

    def test_truncated_tool_call_args_with_length_finish_reason_upgrades(self, agent):
        # Control: when the provider explicitly reports finish_reason='length'
        # alongside incomplete tool args, it IS a genuine output cap.  Keep the
        # existing behaviour — tool_calls preserved, finish_reason 'length' —
        # so the max_tokens-boost truncation retry path still applies.
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_1", "write_file", '{"path":"x.txt","content":"hel')]),
            _make_chunk(finish_reason="length"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 1
        assert tc[0].function.name == "write_file"
        assert tc[0].function.arguments == '{"path":"x.txt","content":"hel'
        assert resp.choices[0].finish_reason == "length"

    @pytest.mark.parametrize("finish_reason", [None, "tool_calls"])
    def test_cut_tool_args_are_repaired_only_once_the_provider_finished(self, agent, finish_reason):
        # Cut after the first digit of "timeout": 600. Every string is closed, so the
        # prefix repairs to valid JSON that carries timeout=6. Without a finish_reason
        # nothing says the model was done: retry, never run what happened to arrive.
        from hermes_constants import PARTIAL_STREAM_STUB_ID
        raw = '{"command": "make deploy", "timeout": 6'
        chunks = [_make_chunk(tool_calls=[_make_tc_delta(0, "call_1", "terminal", raw)])]
        if finish_reason:
            chunks.append(_make_chunk(finish_reason=finish_reason))
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        if finish_reason is None:
            assert resp.id == PARTIAL_STREAM_STUB_ID
            assert resp.choices[0].message.tool_calls is None
            assert resp._dropped_tool_names == ["terminal"]
        else:
            args = resp.choices[0].message.tool_calls[0].function.arguments
            assert json.loads(args) == {"command": "make deploy", "timeout": 6}

    def test_ollama_reused_index_separate_tool_calls(self, agent):
        """Ollama sends every tool call at index 0 with different ids.

        Without the fix, names and arguments get concatenated into one slot.
        """
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_a", "search", '{"q":"hello"}')]),
            # Second tool call at the SAME index 0, but different id
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_b", "read_file", '{"path":"x.py"}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 2, f"Expected 2 tool calls, got {len(tc)}: {[t.function.name for t in tc]}"
        assert tc[0].function.name == "search"
        assert tc[0].function.arguments == '{"q":"hello"}'
        assert tc[0].id == "call_a"
        assert tc[1].function.name == "read_file"
        assert tc[1].function.arguments == '{"path":"x.py"}'
        assert tc[1].id == "call_b"

    def test_ollama_reused_index_streamed_args(self, agent):
        """Ollama with streamed arguments across multiple chunks at same index."""
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_a", "search", '{"q":')]),
            _make_chunk(tool_calls=[_make_tc_delta(0, None, None, '"hello"}')]),
            # New tool call, same index 0
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_b", "read", '{}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 2
        assert tc[0].function.name == "search"
        assert tc[0].function.arguments == '{"q":"hello"}'
        assert tc[1].function.name == "read"
        assert tc[1].function.arguments == '{}'

    def test_content_and_tool_calls_together(self, agent):
        chunks = [
            _make_chunk(content="I'll search"),
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_1", "search", '{}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == "I'll search"
        assert len(resp.choices[0].message.tool_calls) == 1

    def test_empty_content_returns_none(self, agent):
        chunks = [_make_chunk(finish_reason="stop")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content is None
        assert resp.choices[0].message.tool_calls is None

    @pytest.mark.parametrize("carrier", ["reasoning_content", "reasoning"])
    def test_reasoning_only_in_delta_model_extra_counts_as_stream_output(self, agent, carrier):
        """Reasoning that reaches the stream only via ``delta.model_extra`` is real output:
        the empty-stream guard must not fire and the text must survive (#56516)."""
        def _extra_delta(text):
            return SimpleNamespace(content=None, tool_calls=None, model_extra={carrier: text})

        chunks = [
            SimpleNamespace(model="m", choices=[SimpleNamespace(delta=_extra_delta("thinking "), finish_reason=None)]),
            SimpleNamespace(model="m", choices=[SimpleNamespace(delta=_extra_delta("only"), finish_reason="length")]),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content is None
        assert resp.choices[0].message.reasoning_content == "thinking only"
        assert resp.choices[0].finish_reason == "length"

    def test_final_response_object_replays_reasoning_from_model_extra(self, agent):
        """The 'completed response instead of an iterator' branch reads reasoning through the
        same ``model_extra`` fallback as the delta path, so it is still shown (#56516)."""
        message = SimpleNamespace(content="done", tool_calls=None, model_extra={"reasoning": "thought"})
        final = SimpleNamespace(model="m", choices=[SimpleNamespace(message=message, finish_reason="stop")])
        agent.client.chat.completions.create.return_value = final
        agent.reasoning_callback = MagicMock()

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp is final
        agent.reasoning_callback.assert_called_once_with("thought")

    def test_model_name_captured(self, agent):
        chunks = [
            _make_chunk(content="Hi", model="gpt-4o"),
            _make_chunk(finish_reason="stop", model="gpt-4o"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.model == "gpt-4o"

    def test_stream_kwarg_injected(self, agent):
        chunks = [_make_chunk(content="x"), _make_chunk(finish_reason="stop")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        agent._interruptible_streaming_api_call({"messages": [], "model": "test"})

        call_kwargs = agent.client.chat.completions.create.call_args
        assert call_kwargs[1].get("stream") is True or call_kwargs.kwargs.get("stream") is True

    def test_api_exception_propagates_no_non_streaming_fallback(self, agent):
        """When streaming fails before any deltas, error propagates to the main retry loop."""
        agent.client.chat.completions.create.side_effect = ConnectionError("fail")
        # Prevent stream retry logic from replacing the mock client
        with patch.object(agent, "_replace_primary_openai_client", return_value=False):
            # The fallback also uses the same client, so it'll fail too
            with pytest.raises(ConnectionError, match="fail"):
                agent._interruptible_streaming_api_call({"messages": []})

class TestAnthropicInterruptHandler:
    """_interruptible_api_call must handle Anthropic mode when interrupted."""


    def test_interruptible_anthropic_interrupt_never_closes_shared_client(self, agent):
        """#67142: a non-streaming Anthropic interrupt must abort the
        request-local client from the poll thread, never close/rebuild the
        shared _anthropic_client (which raced a live SSL BIO and corrupted an
        unrelated SQLite DB via TLS-FD recycling).

        Replaces the former source-reading assertion (which asserted the old,
        now-removed rebuild-on-interrupt behavior) with a behavior test.
        """
        import time
        from unittest.mock import MagicMock
        from agent.chat_completion_helpers import interruptible_api_call

        agent.api_mode = "anthropic_messages"
        agent._interrupt_requested = False
        agent._anthropic_client = MagicMock()
        agent._rebuild_anthropic_client = MagicMock()
        request_client = MagicMock()
        agent._create_request_anthropic_client = MagicMock(return_value=request_client)
        agent._abort_request_anthropic_client = MagicMock()
        agent._close_request_anthropic_client = MagicMock()

        def _create(_api_kwargs, *, client):
            assert client is request_client
            agent._interrupt_requested = True
            time.sleep(0.5)
            raise RuntimeError("forced close would have happened")

        agent._anthropic_messages_create = MagicMock(side_effect=_create)

        t0 = time.time()
        with pytest.raises(InterruptedError):
            interruptible_api_call(agent, {"model": "x", "messages": []})
        elapsed = time.time() - t0

        assert elapsed < 3.0, f"interrupt took {elapsed:.1f}s — should be near-instant"
        # The shared client is never closed/rebuilt from the poll thread.
        agent._anthropic_client.close.assert_not_called()
        agent._rebuild_anthropic_client.assert_not_called()
        # The poll (stranger) thread aborts the request-local client's socket.
        agent._abort_request_anthropic_client.assert_called_once_with(
            request_client, reason="interrupt_abort"
        )

class TestEmptySSEFrameTurnRecovery:
    """A degraded gateway answers every streaming request with a contentless ``data:``
    frame. The SDK turns that into ``JSONDecodeError(doc='')`` → ``Provider stream returned
    non-JSON SSE data`` and the turn died after 3 identical streaming retries. The turn must
    instead complete on the automatic non-streaming retry."""

    def test_turn_completes_on_the_non_streaming_retry(self, agent):
        import httpx
        from openai import OpenAI, Stream
        from openai.types.chat import ChatCompletionChunk

        request = httpx.Request("POST", "https://gw.example/v1/chat/completions")
        empty_frame = httpx.Response(
            200, request=request, headers={"x-request-id": "req-empty"}, content=b"data:\n\n"
        )
        # The real SDK decoder, so the test exercises the exact production rejection.
        agent.client.chat.completions.create.return_value = Stream(
            cast_to=ChatCompletionChunk,
            response=empty_frame,
            client=OpenAI(api_key="test-key", max_retries=0),
        )
        agent.stream_delta_callback = MagicMock()  # a consumer: the loop prefers streaming

        attempts = []

        def _non_streaming(api_kwargs):
            attempts.append("non_streaming")
            return _mock_response(content="Recovered")

        agent._interruptible_api_call = _non_streaming
        agent._persist_session = lambda *args, **kwargs: None
        agent._save_trajectory = lambda *args, **kwargs: None
        warnings = []
        agent.status_callback = lambda kind, message: warnings.append((kind, message))

        with patch("run_agent.time.sleep", return_value=None):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "Recovered"
        assert agent._disable_streaming is True
        # Exactly one retry, and it went out on the non-streaming channel: the stream was
        # attempted once and never re-entered (the old behaviour retried it 3 times).
        assert attempts == ["non_streaming"]
        assert agent.client.chat.completions.create.call_count == 1
        assert any(kind == "warn" for kind, _msg in warnings)

class TestVprintForceOnErrors:
    """Error/warning messages must be visible during streaming TTS."""

    def test_forced_message_shown_during_tts(self, agent):
        agent._stream_callback = lambda x: None
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(a)):
            agent._vprint("error msg", force=True)
        assert len(printed) == 1
