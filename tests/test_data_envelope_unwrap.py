"""Tests for data-envelope unwrapping in non-streaming chat-completion dispatch.

Catches a provider quirk: some OpenAI-wire providers (e.g. clinepass) wrap
non-streaming responses in a top-level ``{"data": {...}, "success": true}``
envelope. The OpenAI SDK tolerates this by returning a ChatCompletion with
empty id/choices, which the agent loop rejects. These tests pin the unwrap
behaviour that keeps every OpenAI-wire provider round-tripping cleanly.
"""

from types import SimpleNamespace

from openai.types.chat import ChatCompletion

from agent.chat_completion_helpers import (
    _create_with_data_envelope_unwrap,
    _dispatch_nonstreaming_api_request,
)


WRAPPED = {
    "data": {
        "id": "gen_x",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "cline-pass/deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello from envelope"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "success": True,
}

STANDARD = {
    "id": "gen_y",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "some/provider-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello standard"},
            "finish_reason": "stop",
        }
    ],
}


class _FakeRawResponse:
    def __init__(self, text):
        self.text = text


class _FakeRawAccessor:
    def __init__(self, raw_text):
        self._raw_text = raw_text

    def create(self, *args, **kwargs):
        return _FakeRawResponse(self._raw_text)


class _FakeCompletions:
    """Mimics the parts of an OpenAI client's chat.completions the code uses."""

    def __init__(self, raw_text):
        self.with_raw_response = _FakeRawAccessor(raw_text)

    def create(self, *args, **kwargs):
        raise AssertionError("plain create() should not be hit when unwrapping")


class _FakeClient:
    def __init__(self, raw_text, host="api.cline.bot"):
        self.base_url = SimpleNamespace(host=host)
        self.chat = SimpleNamespace(completions=_FakeCompletions(raw_text))


class _FakeAgent:
    api_mode = "chat_completions"
    provider = "not-moa"


def _import_fast_json_body(body):
    import json

    return json.dumps(body)


def _make_client_factory(raw_text):
    client = _FakeClient(raw_text)
    return lambda reason, kind="openai": client


def test_unwrap_envelope_returns_inner_choices():
    raw = _import_fast_json_body(WRAPPED)
    result = _create_with_data_envelope_unwrap(_FakeClient(raw), {})
    assert isinstance(result, ChatCompletion)
    assert result.id == "gen_x"
    assert result.choices[0].message.content == "hello from envelope"


def test_no_envelope_passes_standard_body_through():
    raw = _import_fast_json_body(STANDARD)
    result = _create_with_data_envelope_unwrap(_FakeClient(raw), {})
    assert isinstance(result, ChatCompletion)
    assert result.id == "gen_y"
    assert result.choices[0].message.content == "hello standard"


def test_dispatch_unwraps_envelope_through_make_client():
    raw = _import_fast_json_body(WRAPPED)
    result = _dispatch_nonstreaming_api_request(
        _FakeAgent(), {}, make_client=_make_client_factory(raw)
    )
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "hello from envelope"


def test_dispatch_passes_standard_body_through():
    raw = _import_fast_json_body(STANDARD)
    result = _dispatch_nonstreaming_api_request(
        _FakeAgent(), {}, make_client=_make_client_factory(raw)
    )
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "hello standard"