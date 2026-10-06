"""Mid-stream repetition guard: a runaway loop is cut at a 12k-char threshold, legitimate long text is not.

Regression for LKP-1014: one kimi-k3 completion streamed ~138k chars of repeated text for 38 minutes
before the provider closed the socket; nothing stopped reading it.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_constants import PARTIAL_STREAM_STUB_ID


def _chunk(content=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
                           model="moonshotai/kimi-k3", usage=None)


def _agent():
    from run_agent import AIAgent

    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="moonshotai/kimi-k3",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


def _stream(texts, consumed, finish=True):
    def gen():
        for text in texts:
            consumed.append(len(text))
            yield _chunk(content=text)
        if finish:
            yield _chunk(content="", finish_reason="stop")
    return gen()


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_runaway_loop_is_cut_at_the_first_threshold(mock_close, mock_create, caplog):
    unit = "The court held that the motion is denied because the movant lacks standing here.\n"
    consumed = []
    client = MagicMock()
    # 2,000 chunks (~170k chars) and no finish_reason: only the guard can end this early
    client.chat.completions.create.return_value = _stream([unit] * 2000, consumed, finish=False)
    mock_create.return_value = client

    response = _agent()._interruptible_streaming_api_call({})

    assert response.id == PARTIAL_STREAM_STUB_ID and response._repetition_terminated is True
    assert response.choices[0].finish_reason == "length"
    assert 12_000 <= sum(consumed) < 13_000, "stopped at the first 12k threshold"
    assert "repetition loop detected mid-stream after" in caplog.text


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_long_distinct_answer_streams_to_the_end(mock_close, mock_create):
    lines = [f"{i}. Exhibit {i} (Bates ACME-{i:06d}) is a letter dated 2024-{i % 12 + 1:02d}-{i % 28 + 1:02d} "
             f"from counsel about item {i * 7}.\n" for i in range(400)]
    assert sum(map(len, lines)) > 30_000
    consumed = []
    client = MagicMock()
    client.chat.completions.create.return_value = _stream(lines, consumed)
    mock_create.return_value = client

    response = _agent()._interruptible_streaming_api_call({})

    assert response.id != PARTIAL_STREAM_STUB_ID and not getattr(response, "_repetition_terminated", False)
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == "".join(lines)
