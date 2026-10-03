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
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.errors import EmptyStreamError
from tests.agent.test_first_chunk_at_hook import (  # noqa: F401  (shared agent fixture)
    _make_stream_chunk,
    agent,
)


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
