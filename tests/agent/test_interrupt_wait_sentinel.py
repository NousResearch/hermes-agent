from agent.agent_runtime_helpers_placeholders import hidden_interrupt_placeholder_row
from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX
from agent.message_sanitization import close_interrupted_tool_sequence

SENTINEL = f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}{26.2:.1f}s elapsed)."


def _tool_tail():
    return [
        {"role": "user", "content": "did you read it?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "# report"},
    ]


def test_wait_sentinel_closes_the_tail_with_the_hidden_placeholder():
    for final in (SENTINEL, "  " + SENTINEL + "\n"):
        messages = _tool_tail()
        assert close_interrupted_tool_sequence(messages, final) is True
        row = messages[-1]
        assert {k: row[k] for k in ("role", "content", "display_kind", "api_content")} == hidden_interrupt_placeholder_row()
        assert INTERRUPT_WAITING_FOR_MODEL_PREFIX not in str(row)


def test_real_partial_reply_still_wins():
    messages = _tool_tail()
    close_interrupted_tool_sequence(messages, "Checking the transcript")
    assert messages[-1]["content"] == "Checking the transcript"
    assert messages[-1].get("display_kind") != "hidden"
