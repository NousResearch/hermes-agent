"""The Responses current-turn boundary is anchored on this turn's user row, not on prefix
equality with the client history (#89891).

The loop repairs host-fed history before its first call (consecutive assistant rows merge,
stray tool results drop) and compaction rewrites it, so ``result["messages"]`` legitimately
stops sharing a prefix with ``conversation_history``.  A prefix match then reported 0 and the
whole transcript became "the current turn": earlier ``function_call`` items were replayed as
this turn's output and the stored history doubled on every chained turn.
"""

from agent.agent_runtime_helpers import repair_message_sequence
from agent.message_metadata import append_message
from gateway.platforms.api_server import APIServerAdapter


class _Agent:
    api_mode = "chat_completions"
    session_id = "s"


def _tool_turn(messages, call_id, answer):
    append_message(messages, {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]})
    append_message(messages, {"role": "tool", "tool_call_id": call_id, "content": "ok"})
    append_message(messages, {"role": "assistant", "content": answer, "finish_reason": "stop"})


def test_repaired_earlier_rows_keep_only_this_turn_as_output():
    # A stateless client replays two consecutive assistant items; the loop merges them, so the
    # transcript is one row shorter than the client history from index 1 on.
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
               {"role": "assistant", "content": "again"}]
    prompt = "Q1: run the tool"
    messages = list(history)
    append_message(messages, {"role": "user", "content": prompt})
    repair_message_sequence(_Agent(), messages)
    _tool_turn(messages, "call-current", "done")
    result = {"messages": messages}

    start = APIServerAdapter._response_messages_turn_start_index(history, prompt, result)
    items = APIServerAdapter._extract_output_items(result, start_index=start)
    assert messages[start - 1]["content"] == prompt
    assert [i["type"] for i in items] == ["function_call", "function_call_output", "message"]
    # Stored as the agent's transcript, not client history + transcript again.
    stored = APIServerAdapter._build_response_conversation_history(history, prompt, result, "done")
    assert [m["role"] for m in stored] == ["user", "assistant", "user", "assistant", "tool", "assistant"]


def test_compacted_transcript_keeps_only_this_turn_as_output():
    # Compaction replaced the oldest rows with a summary carrier and kept the recent
    # tool-bearing turn verbatim: nothing before the current user row is this turn's output.
    history = [{"role": "user", "content": "old ask"}, {"role": "assistant", "content": "old answer"},
               {"role": "user", "content": "recent ask"}]
    messages = [{"role": "user", "content": "[Compressed summary of earlier turns]"},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": "recent ask"}]
    _tool_turn(messages, "call-old", "recent answer")
    history = history + [dict(m) for m in messages[3:]]
    append_message(messages, {"role": "user", "content": "Q2"})
    _tool_turn(messages, "call-current", "answer 2")
    result = {"messages": messages, "_compressed": True}

    start = APIServerAdapter._response_messages_turn_start_index(history, "Q2", result)
    items = APIServerAdapter._extract_output_items(result, start_index=start)
    assert [i["type"] for i in items] == ["function_call", "function_call_output", "message"]
    assert items[0]["call_id"] == "call-current"
