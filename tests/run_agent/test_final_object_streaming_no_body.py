from __future__ import annotations

from types import SimpleNamespace

from agent.chat_completion_helpers import interruptible_streaming_api_call


class _CompletedObjectClient:
    def __init__(self, response):
        self._response = response
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **_kwargs):
        return self._response


def test_completed_response_object_does_not_stream_answer_body(monkeypatch) -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="canonical answer", reasoning_content=None))]
    )
    agent = SimpleNamespace(
        provider="moa",
        model="default",
        api_mode="chat_completions",
        base_url="https://example.invalid/v1",
        _disable_streaming=False,
        _create_request_openai_client=lambda **_kwargs: _CompletedObjectClient(response),
        _close_request_openai_client=lambda *_args, **_kwargs: None,
        _capture_rate_limits=lambda _response: None,
        _capture_credits=lambda _response: None,
        _check_openrouter_cache_status=lambda _response: None,
        _stream_diag_init=lambda: {},
        _stream_diag_capture_response=lambda *_args, **_kwargs: None,
        _stream_diag_finalize=lambda *_args, **_kwargs: None,
        _fire_reasoning_delta=lambda text: None,
        _fire_stream_delta=lambda text: streamed.append(text),
        _is_provider_stream_parse_error=lambda _exc: False,
        _interrupt_requested=False,
        _touch_activity=lambda _message: None,
    )
    streamed: list[str] = []
    monkeypatch.setattr("agent.chat_completion_helpers.get_provider_stale_timeout", lambda *_args, **_kwargs: 1)

    result = interruptible_streaming_api_call(agent, {"stream": True})

    assert result is response
    assert agent._disable_streaming is True
    assert streamed == []
