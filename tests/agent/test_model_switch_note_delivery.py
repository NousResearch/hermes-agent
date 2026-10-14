"""Model-switch routing guidance must not become assistant commentary or replay."""

from types import SimpleNamespace

import pytest

from agent.turn_response_intake import normalize_model_response
from run_agent import AIAgent


NOTE = (
    "[Note: model was just switched from local/model to MiniMax-M3 via MiniMax. "
    "Adjust your self-identification accordingly.]"
)
ONCE_NOTE = NOTE.replace(
    "Adjust your", "This override applies to the next turn only. Adjust your"
)
ECHO_NOTE = NOTE.replace("local/model", "locl/model").replace("accordingly.]", "accordingly.)]")


@pytest.fixture
def agent():
    return AIAgent(
        api_key="test-key", base_url="https://openrouter.ai/api/v1",
        model="test/model", quiet_mode=True, skip_context_files=True, skip_memory=True,
    )


@pytest.mark.parametrize("note", [NOTE, ONCE_NOTE, ECHO_NOTE])
@pytest.mark.parametrize("progress", ["", "Checking the next file."])
def test_tool_response_keeps_work_and_tools_without_echoing_routing_guidance(agent, note, progress):
    call = SimpleNamespace(
        id="call_1", type="function",
        function=SimpleNamespace(name="execute_code", arguments='{"code":"1 + 1"}'),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=note + "\n\n" + progress, tool_calls=[call]),
            finish_reason="tool_calls",
        )], usage=None, model="test/model",
    )
    verdict = normalize_model_response(
        agent, response=response, messages=[], api_messages=[], conversation_history=[],
        api_call_count=1, api_duration=0, api_start_time=0, api_request_id="request_1",
        effective_task_id="test", turn_id="turn_1",
    )
    assert verdict.action == "fallthrough"
    row = agent._build_assistant_message(verdict.assistant_message, verdict.finish_reason)
    assert row["content"] == progress
    assert row["tool_calls"][0]["function"]["name"] == "execute_code"
    delivered = []
    agent.interim_assistant_callback = lambda text, **kwargs: delivered.append(text)
    agent._emit_interim_assistant_message(row)
    assert delivered == ([progress] if progress else [])


@pytest.mark.parametrize("note", [NOTE, ONCE_NOTE, ECHO_NOTE])
@pytest.mark.parametrize("chunk_size", [1, 7, 97])
def test_streamed_notice_is_hidden_before_any_delta_reaches_the_ui(agent, note, chunk_size):
    delivered = []
    agent.stream_delta_callback = delivered.append
    for content, expected in [(note, ""), (note + "\n\n", ""), (note + "\n\nChecking the next file.", "Checking the next file.")]:
        agent._reset_stream_delivery_tracking()
        delivered.clear()
        for start in range(0, len(content), chunk_size):
            agent._fire_stream_delta(content[start:start + chunk_size])
        agent._reset_stream_delivery_tracking()
        assert "".join(delivered) == expected


def test_hidden_notice_does_not_consume_the_separator_after_tools(agent):
    delivered = []
    agent.stream_delta_callback = delivered.append
    agent._stream_needs_break = True
    for char in "<think>private reasoning</think>" + NOTE:
        agent._fire_stream_delta(char)
    assert delivered == []
    assert agent._stream_needs_break
    agent._fire_stream_delta("Checking the next file.")
    assert "".join(delivered) == "\n\nChecking the next file."


@pytest.mark.parametrize("with_tools", [False, True])
def test_minimax_anthropic_response_keeps_work_without_persisting_the_notice(agent, with_tools):
    from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

    agent.api_mode = "anthropic_messages"
    blocks = [TextBlock(type="text", text=NOTE + "\n\nChecking the next file.")]
    if with_tools:
        blocks.append(ToolUseBlock(type="tool_use", id="call_1", name="execute_code", input={"code": "1 + 1"}))
    response = Message(
        id="msg_1", type="message", role="assistant", model="MiniMax-M3", content=blocks,
        stop_reason="tool_use" if with_tools else "end_turn",
        usage=Usage(input_tokens=10, output_tokens=20),
    )
    verdict = normalize_model_response(
        agent, response=response, messages=[], api_messages=[], conversation_history=[],
        api_call_count=1, api_duration=0, api_start_time=0, api_request_id="request_1",
        effective_task_id="test", turn_id="turn_1",
    )
    row = agent._build_assistant_message(verdict.assistant_message, verdict.finish_reason)
    assert row["content"] == "Checking the next file."
    assert bool(row.get("tool_calls")) == with_tools


@pytest.mark.parametrize("text", [
    "[Note: keep this ordinary user-facing note.]",
    "[Note: model was just switched from prose without the routing terminator.]",
    "```text\n" + NOTE + "\n```",
    "The diagnostic example is " + NOTE,
    "[Note: model was just switched from unfinished",
    "[Note: model was just switched from " + "x" * 4096,
])
def test_ordinary_notes_and_quoted_examples_remain_visible(agent, text):
    delivered = []
    agent.stream_delta_callback = delivered.append
    for char in text:
        agent._fire_stream_delta(char)
    agent._reset_stream_delivery_tracking()
    assert "".join(delivered) == text
    assert agent._interim_assistant_visible_text({"role": "assistant", "content": text}) == text
