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


def test_known_stream_breakage_falls_back_to_create():
    # #72833: custom provider SSE emits deltas before message_start.
    client, created = _client(RuntimeError('Unexpected event order, got content_block_delta before "message_start"'))
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


def _minimax_sdk_stream():
    """Real SDK MessageStream over MiniMax-style usage:null message_start/message_delta events."""
    from anthropic import NOT_GIVEN
    from anthropic._models import construct_type_unchecked
    from anthropic.lib.streaming import MessageStream
    from anthropic.types import RawMessageStreamEvent

    raw = [construct_type_unchecked(type_=RawMessageStreamEvent, value=v) for v in (
        {"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant",
                                              "model": "MiniMax-M2", "content": [], "usage": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": None},
        {"type": "message_stop"},
    )]
    return MessageStream(iter(raw), output_format=NOT_GIVEN)


class _Ctx:
    def __init__(self, inner):
        self._inner = inner

    def __enter__(self):
        return self._inner

    def __exit__(self, *a):
        return False


def test_null_usage_aux_stream_is_normalized_without_create_retry():
    # #60683 aux path: usage:null must be normalized in-stream, not crash and re-bill via create().
    created = []
    messages = SimpleNamespace(stream=lambda **kw: _Ctx(_minimax_sdk_stream()),
                               create=lambda **kw: created.append(kw))
    final = create_anthropic_message(SimpleNamespace(messages=messages),
                                     {"model": "m", "messages": [], "max_tokens": 8})
    assert final.content[0].text == "hi" and final.stop_reason == "end_turn"
    assert created == []


def _anthropic_agent(stream_factory, provider="custom"):
    from run_agent import AIAgent

    agent = AIAgent(api_key="k", base_url="https://api.minimax.io/anthropic", model="MiniMax-M2",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.api_mode, agent.provider, agent._interrupt_requested = "anthropic_messages", provider, False
    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: stream_factory()))
    agent._anthropic_client = client
    agent._create_request_anthropic_client = lambda *a, **k: client
    return agent


def test_null_usage_stream_is_normalized_before_sdk_accumulation():
    # #60683 main turn: MiniMax sends usage:null on message_start/message_delta; the SDK's
    # accumulate_event() crashes mid-iteration unless _call_anthropic normalizes the raw events.
    agent = _anthropic_agent(lambda: _Ctx(_minimax_sdk_stream()))
    final = agent._interruptible_streaming_api_call({"model": "MiniMax-M2", "messages": [], "max_tokens": 8})
    assert final.content[0].text == "hi" and final.stop_reason == "end_turn"


@pytest.mark.parametrize("provider,disabled", [("custom", True), ("bedrock", False)])
def test_main_turn_event_order_error_disables_streaming(provider, disabled):
    # #72833 main turn: a custom anthropic_messages provider's out-of-order SSE must switch the
    # retry to non-streaming; Bedrock keeps turn_recovery's Converse fallback instead.
    err = RuntimeError('Unexpected event order, got content_block_delta before "message_start"')
    agent = _anthropic_agent(lambda: _BrokenStream(err), provider=provider)
    with pytest.raises(RuntimeError, match="Unexpected event order"):
        agent._interruptible_streaming_api_call({"model": "MiniMax-M2", "messages": [], "max_tokens": 8})
    assert bool(getattr(agent, "_disable_streaming", False)) is disabled
