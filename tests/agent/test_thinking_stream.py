"""Regression tests for OpenAI-compatible ``delta.thinking`` streams."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="x",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    setattr(agent, "api_mode", "chat_completions")
    return agent


def _chunk(
    *,
    content=None,
    thinking=None,
    reasoning_content=None,
    reasoning=None,
    finish_reason=None,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(
                    content=content,
                    thinking=thinking,
                    tool_calls=None,
                    reasoning_content=reasoning_content,
                    reasoning=reasoning,
                ),
                finish_reason=finish_reason,
            )
        ],
        model="test/model",
        usage=None,
    )


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.response = SimpleNamespace(headers={})

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        return None


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_thinking_stream_fires_reasoning_and_is_preserved(mock_close, mock_create):
    stream = _FakeStream(
        [
            _chunk(thinking="first "),
            _chunk(thinking="second"),
            _chunk(content="42", finish_reason="stop"),
        ]
    )

    client = MagicMock()
    client.chat.completions.create.return_value = stream
    mock_create.return_value = client

    agent = _make_agent()
    reasoning_deltas = []
    text_deltas = []
    setattr(agent, "reasoning_callback", reasoning_deltas.append)
    setattr(agent, "stream_delta_callback", text_deltas.append)

    response = agent._interruptible_streaming_api_call({})

    assert response is not None
    assert reasoning_deltas == ["first ", "second"]
    assert response.choices[0].message.reasoning_content == "first second"
    assert response.choices[0].message.content == "42"
    assert text_deltas == ["42"]


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_empty_thinking_stream_does_not_emit_reasoning(mock_close, mock_create):
    client = MagicMock()
    client.chat.completions.create.return_value = _FakeStream(
        [_chunk(thinking=""), _chunk(content="42", finish_reason="stop")]
    )
    mock_create.return_value = client

    agent = _make_agent()
    reasoning_deltas = []
    setattr(agent, "reasoning_callback", reasoning_deltas.append)

    response = agent._interruptible_streaming_api_call({})

    assert response is not None
    assert reasoning_deltas == []
    assert response.choices[0].message.reasoning_content is None


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_stream_prefers_reasoning_content_over_thinking(mock_close, mock_create):
    client = MagicMock()
    client.chat.completions.create.return_value = _FakeStream(
        [
            _chunk(reasoning_content="standard", thinking="fallback"),
            _chunk(content="42", finish_reason="stop"),
        ]
    )
    mock_create.return_value = client

    agent = _make_agent()
    reasoning_deltas = []
    setattr(agent, "reasoning_callback", reasoning_deltas.append)

    response = agent._interruptible_streaming_api_call({})

    assert response is not None
    assert reasoning_deltas == ["standard"]
    assert response.choices[0].message.reasoning_content == "standard"


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_stream_thinking_nested_in_model_extra(mock_close, mock_create):
    # Some relays deliver ``thinking`` only as an undeclared pydantic field
    # (model_extra) on streaming deltas; the accumulator must see it there.
    stream = _FakeStream(
        [
            _chunk(thinking="nested "),
            _chunk(thinking="thought"),
            _chunk(content="42", finish_reason="stop"),
        ]
    )
    for chunk, value in zip(stream._chunks, ["nested ", "thought"]):
        delta = chunk.choices[0].delta
        delta.thinking = None
        delta.model_extra = {"thinking": value}

    client = MagicMock()
    client.chat.completions.create.return_value = stream
    mock_create.return_value = client

    agent = _make_agent()
    reasoning_deltas = []
    setattr(agent, "reasoning_callback", reasoning_deltas.append)

    response = agent._interruptible_streaming_api_call({})

    assert response is not None
    assert reasoning_deltas == ["nested ", "thought"]
    assert response.choices[0].message.reasoning_content == "nested thought"
