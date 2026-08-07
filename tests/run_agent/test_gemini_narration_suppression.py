"""Tests for issue #76997: Vertex Gemini interim narration suppression.

Vertex's OpenAI-compat layer for Gemini 3.x emits the model's pre-tool
narration ("Locating Files", ...) as ordinary ``delta.content``. The fix
buffers that content for Gemini-family models on the OpenAI-compat path and
routes it to reasoning when the turn carries tool calls, so it is neither
streamed as normal assistant content nor persisted into history. Native Gemini
and non-Gemini providers keep the previous behavior.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None, model=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=None)


def _make_tool_call_delta(index=0, tc_id=None, name=None, arguments=None):
    func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=func)


def _build_agent(model, base_url, provider=None):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url=base_url,
        model=model,
        provider=provider or "openai",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


_VERTEX_OPENAI_COMPAT_URL = (
    "https://us-central1-aiplatform.googleapis.com/v1beta1/"
    "projects/p/locations/us-central1/endpoints/openapi"
)


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_gemini_vertex_narration_routed_to_reasoning(mock_close, mock_create):
    """Pre-tool narration on Vertex Gemini is not visible content nor persisted."""
    chunks = [
        _make_stream_chunk(content="Locating Files", model="gemini-3.5-flash"),
        _make_stream_chunk(content="Exploring", model="gemini-3.5-flash"),
        _make_stream_chunk(
            tool_calls=[_make_tool_call_delta(0, "call_1", "read_file", '{"path":"/tmp/a"}')],
            model="gemini-3.5-flash",
        ),
        _make_stream_chunk(finish_reason="tool_calls", model="gemini-3.5-flash"),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(chunks)
    mock_create.return_value = mock_client

    agent = _build_agent("gemini-3.5-flash", _VERTEX_OPENAI_COMPAT_URL, provider="vertex")

    response = agent._interruptible_streaming_api_call({})
    msg = response.choices[0].message

    # Narration must NOT appear as visible/persisted content...
    assert not msg.content or "Locating" not in msg.content
    assert not msg.content or "Exploring" not in msg.content
    # ...but IS preserved as reasoning.
    assert msg.reasoning_content and "Locating Files" in msg.reasoning_content
    assert "Exploring" in msg.reasoning_content
    # Tool call intact.
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].function.name == "read_file"


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_gemini_vertex_plain_text_answer_released(mock_close, mock_create):
    """A Gemini OpenAI-compat turn WITHOUT tool calls keeps its content."""
    chunks = [
        _make_stream_chunk(content="The answer", model="gemini-3.5-flash"),
        _make_stream_chunk(content=" is 42.", finish_reason="stop", model="gemini-3.5-flash"),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(chunks)
    mock_create.return_value = mock_client

    agent = _build_agent("gemini-3.5-flash", _VERTEX_OPENAI_COMPAT_URL, provider="vertex")

    response = agent._interruptible_streaming_api_call({})
    assert response.choices[0].message.content == "The answer is 42."
    assert response.choices[0].message.tool_calls is None


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_native_gemini_streaming_unchanged(mock_close, mock_create):
    """Native Gemini path (no OpenAI-compat) keeps token-by-token content."""
    chunks = [
        _make_stream_chunk(content="Locating Files", model="gemini-3.5-flash"),
        _make_stream_chunk(
            tool_calls=[_make_tool_call_delta(0, "call_1", "read_file", '{"path":"/tmp/a"}')],
            model="gemini-3.5-flash",
        ),
        _make_stream_chunk(finish_reason="tool_calls", model="gemini-3.5-flash"),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(chunks)
    mock_create.return_value = mock_client

    agent = _build_agent(
        "gemini-3.5-flash",
        "https://generativelanguage.googleapis.com/v1beta",
        provider="gemini",
    )

    response = agent._interruptible_streaming_api_call({})
    msg = response.choices[0].message
    # Native path is not buffered: narration remains content (pre-existing shape).
    assert msg.content and "Locating Files" in msg.content
    assert msg.tool_calls is not None


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_non_gemini_vertex_behavior_unchanged(mock_close, mock_create):
    """Non-Gemini models on the same OpenAI-compat path keep current behavior."""
    chunks = [
        _make_stream_chunk(content="Reading file...", model="gpt-5.5"),
        _make_stream_chunk(
            tool_calls=[_make_tool_call_delta(0, "call_1", "read_file", '{"path":"/tmp/a"}')],
            model="gpt-5.5",
        ),
        _make_stream_chunk(finish_reason="tool_calls", model="gpt-5.5"),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(chunks)
    mock_create.return_value = mock_client

    agent = _build_agent("gpt-5.5", _VERTEX_OPENAI_COMPAT_URL, provider="vertex")

    response = agent._interruptible_streaming_api_call({})
    msg = response.choices[0].message
    # Pre-tool text remains as content (current behavior for non-Gemini).
    assert msg.content and "Reading file..." in msg.content
    assert msg.tool_calls is not None
