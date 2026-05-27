from types import SimpleNamespace

import pytest


class _StreamRaisesNoneOutputTypeError:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        raise TypeError("'NoneType' object is not iterable")


class _StreamRaisesOtherTypeError:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        raise TypeError("different type error")


class _FakeResponses:
    def __init__(self, stream):
        self._stream = stream

    def stream(self, **kwargs):
        return self._stream


class _FakeAgent:
    _interrupt_requested = False
    _codex_streamed_text_parts = []

    def __init__(self):
        self.fallback_calls = []

    def _touch_activity(self, *_args, **_kwargs):
        pass

    def _fire_stream_delta(self, *_args, **_kwargs):
        pass

    def _fire_reasoning_delta(self, *_args, **_kwargs):
        pass

    def _client_log_context(self):
        return "provider=openai-codex"

    def _run_codex_create_stream_fallback(self, api_kwargs, client=None):
        self.fallback_calls.append((api_kwargs, client))
        return SimpleNamespace(status="failed", output=None)


def test_codex_stream_falls_back_when_sdk_parser_sees_output_none():
    from agent.codex_runtime import run_codex_stream

    agent = _FakeAgent()
    client = SimpleNamespace(responses=_FakeResponses(_StreamRaisesNoneOutputTypeError()))
    response = run_codex_stream(agent, {"model": "gpt-5.5"}, client=client)

    assert response.status == "failed"
    assert agent.fallback_calls == [({"model": "gpt-5.5"}, client)]


def test_codex_stream_does_not_swallow_unrelated_type_error():
    from agent.codex_runtime import run_codex_stream

    agent = _FakeAgent()
    client = SimpleNamespace(responses=_FakeResponses(_StreamRaisesOtherTypeError()))

    with pytest.raises(TypeError, match="different type error"):
        run_codex_stream(agent, {"model": "gpt-5.5"}, client=client)

    assert agent.fallback_calls == []
