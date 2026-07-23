from unittest.mock import MagicMock
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from agent.chat_completion_helpers import interruptible_streaming_api_call

def test_gemini_openai_compat_separates_parallel_calls_with_heuristic():
    """
    OpenAI-compatible Gemini endpoints emit parallel tool calls with the same index and no tool call id.
    The heuristic must separate them based on a new function name appearing after the previous slot has arguments.
    """
    chunks = [
        ChatCompletionChunk(
            id="chatcmpl-1", choices=[Choice(index=0, delta=ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(name="search", arguments=""))]
            ))], created=1, model="gemini-2.5-flash", object="chat.completion.chunk"
        ),
        ChatCompletionChunk(
            id="chatcmpl-1", choices=[Choice(index=0, delta=ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments="{\"q\": \"A\"}"))]
            ))], created=1, model="gemini-2.5-flash", object="chat.completion.chunk"
        ),
        ChatCompletionChunk(
            id="chatcmpl-1", choices=[Choice(index=0, delta=ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(name="search", arguments=""))]
            ))], created=1, model="gemini-2.5-flash", object="chat.completion.chunk"
        ),
        ChatCompletionChunk(
            id="chatcmpl-1", choices=[Choice(index=0, delta=ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments="{\"q\": \"B\"}"))]
            ))], created=1, model="gemini-2.5-flash", object="chat.completion.chunk"
        )
    ]

    def fake_stream(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    mock_agent = MagicMock()
    mock_agent.model_name = "gemini-2.5-flash"
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_stream
    mock_agent._create_request_openai_client.return_value = mock_client
    mock_agent._interrupt_requested = False
    mock_agent.worker_manager = None
    mock_agent.id = "agent-1"
    mock_agent.api_mode = "chat_completions"

    response = interruptible_streaming_api_call(mock_agent, {})
    
    assert len(response.choices) == 1
    tool_calls = response.choices[0].message.tool_calls
    assert len(tool_calls) == 2
    
    assert tool_calls[0].function.name == "search"
    assert tool_calls[0].function.arguments == '{"q": "A"}'
    
    assert tool_calls[1].function.name == "search"
    assert tool_calls[1].function.arguments == '{"q": "B"}'
