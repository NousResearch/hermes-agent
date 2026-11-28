"""Explicit terminal failures must survive stream and turn recovery (offline)."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from run_agent import AIAgent


class ScriptedStream:
    response = None

    def __init__(self, chunks, error):
        self.chunks, self.error = chunks, error
        self.closed = False

    def __iter__(self):
        yield from self.chunks
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


def chunk(content=None, tool_calls=None, finish_reason=None):
    return NS(choices=[NS(index=0, delta=NS(content=content, tool_calls=tool_calls,
              reasoning_content=None, reasoning=None), finish_reason=finish_reason)], model=None, usage=None)


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    agent = AIAgent(api_key="fixture", provider="custom", base_url="https://example.com/v1",
                    model="fixture-model", api_mode="chat_completions", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=3,
                    save_trajectories=False, stream_delta_callback=lambda text: visible.append(text))
    agent.compression_enabled = False
    visible, streams, requests = [], [], []
    error = ConnectionError("response validation failed")
    error.__cause__ = ValueError("invalid final envelope")
    error.hermes_terminal_stream_error = True
    script = []

    def create(**kwargs):
        requests.append(kwargs)
        # A second request succeeds, making an unwanted retry/continuation observable.
        stream = ScriptedStream(script if len(requests) == 1 else [chunk("replayed", finish_reason="stop")],
                                error if len(requests) == 1 else None)
        streams.append(stream)
        return stream

    client = NS(chat=NS(completions=NS(create=create)))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    close_client = Mock()
    monkeypatch.setattr(agent, "_close_request_openai_client", close_client)
    monkeypatch.setattr(agent, "_persist_session", lambda *a, **kw: None)
    monkeypatch.setattr(agent, "_save_trajectory", lambda *a, **kw: None)
    fallback = Mock(return_value=False)
    monkeypatch.setattr(agent, "_try_activate_fallback", fallback)
    return NS(agent=agent, error=error, script=script, requests=requests, streams=streams,
              visible=visible, close_client=close_client, fallback=fallback)


@pytest.mark.parametrize("after_delta", [False, True])
def test_terminal_error_propagates_from_stream(harness, after_delta):
    h = harness
    if after_delta:
        h.script.append(chunk("visible prefix"))
    cause = h.error.__cause__
    with pytest.raises(ConnectionError) as caught:
        h.agent._interruptible_streaming_api_call({"model": "fixture-model", "messages": []})
    assert caught.value is h.error
    assert caught.value.__cause__ is cause
    assert len(h.requests) == 1
    assert h.visible == (["visible prefix"] if after_delta else [])
    assert all(stream.closed for stream in h.streams)
    h.close_client.assert_called_once()


@pytest.mark.parametrize("after_delta", [False, True])
def test_terminal_error_stops_real_turn(harness, after_delta):
    h = harness
    if after_delta:
        h.script.append(chunk("visible prefix"))
    with pytest.raises(ConnectionError) as caught:
        h.agent.run_conversation("Give a short answer.")
    assert caught.value is h.error
    assert len(h.requests) == 1
    h.fallback.assert_not_called()
    assert all(stream.closed for stream in h.streams)


@pytest.mark.parametrize("marker", [False, None, 0, 1, "true", "missing"])
@pytest.mark.parametrize("after_delta", [False, True])
def test_legacy_stream_recovery_is_unchanged(harness, marker, after_delta, monkeypatch):
    h = harness
    if marker == "missing":
        del h.error.hermes_terminal_stream_error
    else:
        h.error.hermes_terminal_stream_error = marker
    if after_delta:
        h.script.append(chunk("visible prefix"))
    monkeypatch.setattr("agent.chat_completion_helpers._wait_stream_retry_backoff", lambda *args: None)
    response = h.agent._interruptible_streaming_api_call({"model": "fixture-model", "messages": []})
    if after_delta:
        assert response.id == "partial-stream-stub"
        assert response.choices[0].finish_reason == "length"
        assert response.choices[0].message.tool_calls is None
        assert len(h.requests) == 1
    else:
        assert response.choices[0].message.content == "replayed"
        assert len(h.requests) == 2
    assert all(stream.closed for stream in h.streams)


@pytest.mark.parametrize("marker", [1, "missing"])
def test_legacy_turn_continues_partial_response(harness, marker):
    h = harness
    if marker == "missing":
        del h.error.hermes_terminal_stream_error
    else:
        h.error.hermes_terminal_stream_error = marker
    h.script.append(chunk("visible prefix"))
    result = h.agent.run_conversation("Give a short answer.")
    assert len(h.requests) == 2
    assert "replayed" in result["final_response"]


def test_terminal_tool_fragment_never_dispatches(harness, monkeypatch):
    h = harness
    h.script.extend([chunk("visible prefix"), chunk(tool_calls=[
        NS(index=0, id="call_fixture", function=NS(name="terminal", arguments='{"command":'))])])
    dispatch = Mock()
    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    with pytest.raises(ConnectionError) as caught:
        h.agent.run_conversation("Give a short answer.")
    assert caught.value is h.error
    assert len(h.requests) == 1
    dispatch.assert_not_called()
    h.fallback.assert_not_called()
    assert all(stream.closed for stream in h.streams)


def test_cancellation_still_wins_and_cleans_up(harness):
    h = harness

    def interrupt_on_delta(text):
        h.visible.append(text)
        h.agent.interrupt()

    h.agent.stream_delta_callback = interrupt_on_delta
    h.script.append(chunk("visible prefix"))
    with pytest.raises(InterruptedError):
        h.agent._interruptible_streaming_api_call({"model": "fixture-model", "messages": []})
    assert len(h.requests) == 1
    assert all(stream.closed for stream in h.streams)


def test_anthropic_terminal_error_closes_context_and_stops_turn(harness, monkeypatch):
    h = harness
    h.agent.api_mode = "anthropic_messages"
    h.agent._anthropic_thinking_budget = None
    stream = ScriptedStream([NS(type="content_block_delta", index=0,
                            delta=NS(type="text_delta", text="visible prefix"))], h.error)
    manager = Mock()
    manager.__enter__ = Mock(return_value=stream)
    manager.__exit__ = Mock(return_value=False)
    open_stream = Mock(return_value=manager)
    client = NS(messages=NS(stream=open_stream))
    monkeypatch.setattr(h.agent, "_create_request_anthropic_client", lambda **kw: client)
    close_client = Mock()
    monkeypatch.setattr(h.agent, "_close_request_anthropic_client", close_client)
    cause = h.error.__cause__
    with pytest.raises(ConnectionError) as caught:
        h.agent.run_conversation("Give a short answer.")
    assert caught.value is h.error
    assert caught.value.__cause__ is cause
    assert h.visible == ["visible prefix"]
    open_stream.assert_called_once()
    manager.__exit__.assert_called_once()
    close_client.assert_called_once()
    assert stream.closed
    h.fallback.assert_not_called()
