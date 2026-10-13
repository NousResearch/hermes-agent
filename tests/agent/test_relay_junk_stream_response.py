"""A relay's junk answer to ``stream=True`` must not kill the turn with an AttributeError.

AgentRouter intermittently answers a streaming request with bytes that are not a chat-completion
stream: a bare ``data: null`` SSE event (the SDK hands the loop a ``None`` chunk) and, when it
ignores ``stream=True`` entirely, a completed body that carries no ``choices`` at all.

Both shapes used to crash the stream loop on ``chunk.choices`` /
``final_response.choices`` with ``'NoneType' object has no attribute 'choices'`` — an opaque
Python AttributeError that burned the retry budget and ended unattended runs (cron deliveries,
Bot Chat turns) instead of retrying the connection. Neither shape says anything about whether
the route can stream, so neither may disable streaming for the session either.

Contract under test:

- A null chunk mid-stream is skipped; the rest of the stream is delivered normally.
- A stream made only of null chunks surfaces as an empty stream (retryable, honest message).
- A completed response with no choices raises EmptyStreamError *without* switching the session
  to non-streaming.
- A legitimate completed response keeps switching to non-streaming and keeps its content.
- ``_replay_final_response`` keeps failing loudly on an unreplayable probe response: the
  non-streaming unmask probe depends on that raise to keep the provider's real error.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.errors import EmptyStreamError
from run_agent import AIAgent


@pytest.fixture()
def agent():
    """Minimal AIAgent with a mocked provider client (no tui_gateway import chain)."""
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://agentrouter.org/v1",
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


def _make_stream_chunk(content=None, finish_reason=None, model=None, usage=None):
    """Mock streaming chunk matching OpenAI's ChatCompletionChunk shape."""
    delta = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None, reasoning=None)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _streaming_call(agent, create_return):
    client = MagicMock()
    client.chat.completions.create.return_value = create_return
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return patch("run_agent.AIAgent._create_request_openai_client", return_value=client), client


@patch("run_agent.AIAgent._close_request_openai_client")
def test_null_chunk_mid_stream_is_skipped_not_fatal(_mock_close, agent):
    """``data: null`` between two good chunks: the answer still arrives, whole."""
    def _stream():
        yield _make_stream_chunk(content="Hel")
        yield None  # the relay's null event
        yield _make_stream_chunk(content="lo", finish_reason="stop", model="m")

    created, _client = _streaming_call(agent, _stream())
    with created:
        response = agent._interruptible_streaming_api_call({})

    assert response.choices[0].message.content == "Hello"


@patch("run_agent.AIAgent._close_request_openai_client")
def test_stream_of_only_null_chunks_is_an_empty_stream(_mock_close, agent):
    """Nothing but junk: an empty stream (retryable), never an AttributeError."""
    created, _client = _streaming_call(agent, iter([None, None]))
    with created, pytest.raises(EmptyStreamError):
        agent._interruptible_streaming_api_call({})


@patch("run_agent.AIAgent._close_request_openai_client")
def test_completed_response_without_choices_raises_empty_stream(_mock_close, agent):
    """The non-stream body with no choices: retry the connection, don't disable streaming.

    (An object that is not even shaped like a response is a different shape, handled by the
    relay's completed-response predicate, not here.)"""
    created, _client = _streaming_call(agent, SimpleNamespace(id="no-choices", choices=None))
    with created, pytest.raises(EmptyStreamError):
        agent._interruptible_streaming_api_call({})

    assert agent._disable_streaming is False


@patch("run_agent.AIAgent._close_request_openai_client")
def test_completed_response_with_choices_still_switches_to_non_streaming(_mock_close, agent):
    """A real non-streaming answer keeps its behavior (and its content)."""
    message = SimpleNamespace(content="hello", tool_calls=None, reasoning_content=None, reasoning=None)
    completed = SimpleNamespace(
        id="ok", choices=[SimpleNamespace(message=message, finish_reason="stop")], model="m")
    created, _client = _streaming_call(agent, completed)
    with created:
        response = agent._interruptible_streaming_api_call({})

    assert response.choices[0].message.content == "hello"
    assert agent._disable_streaming is True


def test_unreplayable_probe_response_still_fails_loudly(agent):
    """``_replay_final_response`` keeps reading ``choices`` directly, on purpose.

    Its other caller is the non-streaming unmask probe (``_handle_stream_error``), which
    depends on an unreplayable probe raising: its ``except Exception`` turns that into
    ``return False`` with the provider's real 5xx still in ``result['error']``. A defensive
    ``getattr`` there would instead leave ``response=None, error=None`` — a state the base
    cannot produce, and one that hides whether the gateway is 5xx-ing or returning junk.
    """
    from agent import chat_completion_helpers as helpers

    call = helpers._StreamingCall(agent, {"model": "m", "messages": []}, None)

    with pytest.raises(AttributeError):
        call._replay_final_response(SimpleNamespace(id="junk"))  # no ``choices`` attribute
