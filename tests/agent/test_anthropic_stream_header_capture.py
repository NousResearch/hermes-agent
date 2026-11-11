"""The streamed Anthropic main turn records its response headers.

Rate-limit, credits and plan-window state live only in response headers. The non-streaming
``create_anthropic_message(on_response=...)`` path already captured them; the streamed main
turn, which most Anthropic turns take, did not.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock


def test_anthropic_stream_captures_response_rate_limit_headers():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "anthropic_messages"
    agent._interrupt_requested = False
    assert agent.get_rate_limit_state() is None

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.__iter__ = MagicMock(return_value=iter([SimpleNamespace(type="message_stop")]))
    mock_stream.get_final_message.return_value = SimpleNamespace(content=[], stop_reason="end_turn")
    mock_stream.response = SimpleNamespace(headers={
        "x-ratelimit-limit-requests": "50",
        "x-ratelimit-remaining-requests": "49",
        "x-ratelimit-reset-requests": "60",
    })

    agent._anthropic_client = MagicMock()
    agent._anthropic_client.messages.stream.return_value = mock_stream
    agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client

    agent._interruptible_streaming_api_call({})

    state = agent.get_rate_limit_state()
    assert state is not None, "streamed response headers were never captured"
    assert state.requests_min.limit == 50 and state.requests_min.remaining == 49
