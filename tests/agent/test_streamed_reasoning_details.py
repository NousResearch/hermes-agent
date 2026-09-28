"""Streamed ``reasoning_details`` survive to the assembled response (replay continuity).

The non-streaming path always kept OpenRouter's ``reasoning_details`` (signatures,
encrypted blocks a provider needs back on the next turn); the streaming chunk loop
dropped them. Consecutive text/summary fragments merge into one logical entry,
encrypted entries stay discrete.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.reasoning_summaries import append_streamed_reasoning_detail


def _make_chunk(content=None, finish_reason=None, model=None, reasoning_details=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None, reasoning=None)
    if reasoning_details is not None:
        delta.reasoning_details = reasoning_details
    return SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
                           model=model, usage=usage)


def test_fragments_merge_per_block_and_backfill_signature():
    acc = []
    append_streamed_reasoning_detail(acc, {"type": "reasoning.text", "text": "The user "})
    append_streamed_reasoning_detail(acc, SimpleNamespace(type="reasoning.text", text="wants X.", signature="sig1"))
    append_streamed_reasoning_detail(acc, {"type": "reasoning.encrypted", "data": "AAAA"})
    append_streamed_reasoning_detail(acc, {"type": "reasoning.encrypted", "data": "BBBB"})
    append_streamed_reasoning_detail(acc, {"type": "reasoning.summary", "summary": "s1 "})
    append_streamed_reasoning_detail(acc, {"type": "reasoning.summary", "summary": "s2"})
    assert [d["type"] for d in acc] == [
        "reasoning.text", "reasoning.encrypted", "reasoning.encrypted", "reasoning.summary"]
    assert acc[0] == {"type": "reasoning.text", "text": "The user wants X.", "signature": "sig1"}
    assert acc[3]["summary"] == "s1 s2"


def _agent():
    from run_agent import AIAgent
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_streamed_details_land_on_final_message_and_persist(_mock_close, mock_create):
    chunks = [
        _make_chunk(reasoning_details=[{"type": "reasoning.text", "text": "I should "}]),
        _make_chunk(reasoning_details=[{"type": "reasoning.text", "text": "answer.", "signature": "sigZ"}]),
        _make_chunk(content="Hello!", finish_reason="stop", model="test-model"),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(chunks)
    mock_create.return_value = mock_client

    agent = _agent()
    response = agent._interruptible_streaming_api_call({})
    msg = response.choices[0].message
    assert msg.content == "Hello!"
    assert msg.reasoning_details == [{"type": "reasoning.text", "text": "I should answer.", "signature": "sigZ"}]
    # The persisted assistant dict (what gets replayed next turn) carries them too.
    persisted = agent._build_assistant_message(msg, "stop")
    assert persisted["reasoning_details"] == msg.reasoning_details


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_no_details_leaves_attribute_absent(_mock_close, mock_create):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter([
        _make_chunk(content="plain", finish_reason="stop", model="test-model")])
    mock_create.return_value = mock_client
    response = _agent()._interruptible_streaming_api_call({})
    assert not hasattr(response.choices[0].message, "reasoning_details")
