"""Regression tests for Codex Responses transport recovery."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import httpx
from openai import APIConnectionError

from agent.codex_runtime import run_codex_stream


class _EventStream:
    final_response = None

    def __init__(self, events):
        self._events = list(events)
        self.closed = False

    def __iter__(self):
        return iter(self._events)

    def close(self):
        self.closed = True


class CodexStreamTransportRecoveryTests(TestCase):
    def test_broken_pipe_before_stream_open_retries_with_fresh_primary_client(self):
        request = httpx.Request(
            "POST",
            "https://chatgpt.com/backend-api/codex/responses",
            content=b"request-body",
        )
        read_error = httpx.ReadError("[Errno 32] Broken pipe", request=request)
        connection_error = APIConnectionError(request=request)
        connection_error.__cause__ = read_error

        first_client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_kwargs: (_ for _ in ()).throw(connection_error))
        )
        message = SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text="recovered")],
        )
        second_stream = _EventStream(
            [
                SimpleNamespace(type="response.output_item.done", item=message),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        id="resp_recovered",
                        status="completed",
                        usage=None,
                    ),
                ),
            ]
        )
        second_client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_kwargs: second_stream)
        )

        clients = [first_client]
        replacements = []
        agent = SimpleNamespace(
            model="gpt-5.6-sol",
            provider="openai-codex",
            session_id="transport-recovery",
            is_subagent=False,
            _fallback_index=0,
            _current_api_request_id="req-1",
            _interrupt_requested=False,
            _codex_streamed_text_parts=[],
            _codex_stream_last_event_ts=0.0,
            _ensure_primary_openai_client=lambda **_kwargs: clients[-1],
            _replace_primary_openai_client=lambda **_kwargs: (
                replacements.append(_kwargs.get("reason")),
                clients.append(second_client),
                True,
            )[-1],
            _fire_stream_delta=lambda _text: None,
            _fire_reasoning_delta=lambda _text: None,
            _fire_streamed_codex_commentary=lambda _text: None,
            _touch_activity=lambda _text: None,
            _client_log_context=lambda: "provider=openai-codex model=gpt-5.6-sol",
        )

        with patch("agent.relay_llm.stream", side_effect=lambda _request, opener, **_kwargs: opener(_request)):
            response = run_codex_stream(agent, {"model": "gpt-5.6-sol"})

        self.assertEqual(replacements, ["codex_stream_transport_retry"])
        self.assertIsNotNone(response)
        assert response is not None
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.id, "resp_recovered")
        self.assertEqual(response.output, [message])
        self.assertTrue(second_stream.closed)

    def test_read_error_after_terminal_response_does_not_retry_inference(self):
        request = httpx.Request(
            "POST",
            "https://chatgpt.com/backend-api/codex/responses",
        )
        message = SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text="already complete")],
        )

        class _PostTerminalReadErrorStream(_EventStream):
            def __iter__(self):
                yield from super().__iter__()
                raise httpx.ReadError("[Errno 32] Broken pipe", request=request)

        stream = _PostTerminalReadErrorStream(
            [
                SimpleNamespace(type="response.output_item.done", item=message),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        id="resp_complete",
                        status="completed",
                        usage=None,
                    ),
                ),
            ]
        )
        calls = []
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **_kwargs: (calls.append(_kwargs), stream)[-1]
            )
        )
        agent = SimpleNamespace(
            model="gpt-5.6-sol",
            provider="openai-codex",
            session_id="post-terminal-read-error",
            is_subagent=False,
            _fallback_index=0,
            _current_api_request_id="req-2",
            _interrupt_requested=False,
            _codex_streamed_text_parts=[],
            _codex_stream_last_event_ts=0.0,
            _ensure_primary_openai_client=lambda **_kwargs: client,
            _replace_primary_openai_client=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("completed inference must not replace client or retry")
            ),
            _fire_stream_delta=lambda _text: None,
            _fire_reasoning_delta=lambda _text: None,
            _fire_streamed_codex_commentary=lambda _text: None,
            _touch_activity=lambda _text: None,
            _client_log_context=lambda: "provider=openai-codex model=gpt-5.6-sol",
        )

        with patch("agent.relay_llm.stream", side_effect=lambda _request, opener, **_kwargs: opener(_request)):
            response = run_codex_stream(agent, {"model": "gpt-5.6-sol"})

        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(response)
        assert response is not None
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.id, "resp_complete")
        self.assertEqual(response.output, [message])
        self.assertTrue(stream.closed)

    def test_explicit_client_keeps_existing_single_retry_semantics(self):
        request = httpx.Request(
            "POST",
            "https://chatgpt.com/backend-api/codex/responses",
        )
        attempts = []
        message = SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text="explicit recovered")],
        )
        recovered_stream = _EventStream(
            [
                SimpleNamespace(type="response.output_item.done", item=message),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        id="resp_explicit",
                        status="completed",
                        usage=None,
                    ),
                ),
            ]
        )

        def _create(**_kwargs):
            attempts.append(_kwargs)
            if len(attempts) == 1:
                raise httpx.ReadError("temporary read failure", request=request)
            return recovered_stream

        explicit_client = SimpleNamespace(
            responses=SimpleNamespace(create=_create)
        )
        agent = SimpleNamespace(
            model="gpt-5.6-sol",
            provider="openai-codex",
            session_id="explicit-client-retry",
            is_subagent=False,
            _fallback_index=0,
            _current_api_request_id="req-3",
            _interrupt_requested=False,
            _codex_streamed_text_parts=[],
            _codex_stream_last_event_ts=0.0,
            _ensure_primary_openai_client=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("explicit client must be used")
            ),
            _replace_primary_openai_client=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("caller-owned client must not be replaced")
            ),
            _fire_stream_delta=lambda _text: None,
            _fire_reasoning_delta=lambda _text: None,
            _fire_streamed_codex_commentary=lambda _text: None,
            _touch_activity=lambda _text: None,
            _client_log_context=lambda: "provider=openai-codex model=gpt-5.6-sol",
        )

        with patch("agent.relay_llm.stream", side_effect=lambda _request, opener, **_kwargs: opener(_request)):
            response = run_codex_stream(
                agent,
                {"model": "gpt-5.6-sol"},
                client=explicit_client,
            )

        self.assertEqual(len(attempts), 2)
        self.assertIsNotNone(response)
        assert response is not None
        self.assertEqual(response.id, "resp_explicit")
        self.assertEqual(response.output, [message])

    def test_failed_rebuild_still_retries_on_existing_client(self):
        request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
        connection_error = APIConnectionError(request=request)
        connection_error.__cause__ = httpx.ReadError("[Errno 32] Broken pipe", request=request)
        message = SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text="same client")],
        )
        stream = _EventStream(
            [
                SimpleNamespace(type="response.output_item.done", item=message),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(id="resp_same", status="completed", usage=None),
                ),
            ]
        )
        attempts = []

        def _create(**_kwargs):
            attempts.append(_kwargs)
            if len(attempts) == 1:
                raise connection_error
            return stream

        shared_client = SimpleNamespace(responses=SimpleNamespace(create=_create))
        replacements = []
        agent = SimpleNamespace(
            model="gpt-5.6-sol",
            provider="openai-codex",
            session_id="failed-rebuild",
            is_subagent=False,
            _fallback_index=0,
            _current_api_request_id="req-4",
            _interrupt_requested=False,
            _codex_streamed_text_parts=[],
            _codex_stream_last_event_ts=0.0,
            _ensure_primary_openai_client=lambda **_kwargs: shared_client,
            _replace_primary_openai_client=lambda **_kwargs: replacements.append(_kwargs.get("reason")) and False,
            _fire_stream_delta=lambda _text: None,
            _fire_reasoning_delta=lambda _text: None,
            _fire_streamed_codex_commentary=lambda _text: None,
            _touch_activity=lambda _text: None,
            _client_log_context=lambda: "provider=openai-codex model=gpt-5.6-sol",
        )

        with patch("agent.relay_llm.stream", side_effect=lambda _request, opener, **_kwargs: opener(_request)):
            response = run_codex_stream(agent, {"model": "gpt-5.6-sol"})

        self.assertEqual(replacements, ["codex_stream_transport_retry"])
        self.assertEqual(len(attempts), 2)
        assert response is not None
        self.assertEqual(response.id, "resp_same")
        self.assertEqual(response.output, [message])
