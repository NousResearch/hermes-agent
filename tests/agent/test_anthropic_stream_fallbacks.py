"""Invariants: Anthropic stream failures fall back or stub in a shape the loop accepts."""

from types import SimpleNamespace

import pytest

from agent.anthropic_adapter import create_anthropic_message
from agent.chat_completion_helpers import _build_partial_stream_stub
from agent.transports.anthropic import AnthropicTransport


class _BrokenStream:
    def __init__(self, exc):
        self._exc = exc

    def __enter__(self):
        raise self._exc

    def __exit__(self, *a):
        return False


def _client(exc):
    created = []
    messages = SimpleNamespace(
        stream=lambda **kw: _BrokenStream(exc),
        create=lambda **kw: created.append(kw) or SimpleNamespace(content=[], stop_reason="end_turn"),
    )
    return SimpleNamespace(messages=messages), created


@pytest.mark.parametrize("exc", [
    RuntimeError('Unexpected event order, got content_block_delta before "message_start"'),  # #72833
    AttributeError("'NoneType' object has no attribute 'output_tokens'"),  # #60683 MiniMax usage:null
])
def test_known_stream_breakage_falls_back_to_create(exc):
    client, created = _client(exc)
    create_anthropic_message(client, {"model": "m", "messages": [], "max_tokens": 8, "stream": True})
    assert len(created) == 1 and "stream" not in created[0]


def test_unrelated_stream_error_still_raises():
    client, created = _client(AttributeError("'Foo' object has no attribute 'bar'"))
    with pytest.raises(AttributeError):
        create_anthropic_message(client, {"model": "m", "messages": [], "max_tokens": 8})
    assert created == []


@pytest.mark.parametrize("content,overflow", [("partial answer", False), (None, True)])
def test_anthropic_partial_stub_passes_transport_validation(content, overflow):
    # #45908: the stub must survive AnthropicTransport.validate_response in anthropic_messages mode.
    stub = _build_partial_stream_stub("assistant", content, None, "m", None,
                                      overflow_terminal=overflow, api_mode="anthropic_messages")
    transport = AnthropicTransport()
    assert transport.validate_response(stub)
    assert transport.response_finish_reason(stub) == "length"
    assert stub._overflow_terminal is overflow
