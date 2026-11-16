"""Stored /v1/responses transcripts must not duplicate history across chained turns.

The agent's ``result["messages"]`` copies of the prior history carry ``timestamp`` /
``_db_persisted`` / ``reasoning`` / ``finish_reason`` that the API layer's bare
``{"role", "content"}`` dicts never have, so a whole-dict prefix check failed every turn and
the stored history grew 3 -> 8 -> 17 instead of 2 -> 4 -> 6 (#95137, #101644, #82513).
"""

from agent.agent_runtime_helpers import repair_message_sequence
from agent.message_metadata import append_message
from gateway.platforms.api_server import APIServerAdapter


class _Agent:
    api_mode = "chat_completions"
    session_id = "s"


def _agent_turn(history, user_message, answer):
    """Real producers: turn_context appends the user row via append_message (stamps timestamp)."""
    messages = list(history)
    append_message(messages, {"role": "user", "content": user_message})
    repair_message_sequence(_Agent(), messages)
    append_message(messages, {"role": "assistant", "content": answer, "reasoning": "r", "finish_reason": "stop"})
    return {"messages": messages}


def test_chained_responses_turns_store_each_message_once():
    build = APIServerAdapter._build_response_conversation_history
    history, counts = [], []
    for turn, prompt in enumerate(["Remember TOKEN-ABC.", "What token?", "Third"]):
        answer = f"answer {turn}"
        history = build(history, prompt, _agent_turn(history, prompt, answer), answer)
        counts.append(len(history))
    assert counts == [2, 4, 6]
    assert [m["role"] for m in history] == ["user", "assistant"] * 3


def test_suffix_only_result_still_appends_to_history():
    build = APIServerAdapter._build_response_conversation_history
    prior = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    # Mocked/legacy paths return only this turn's rows (no user row): appended after the input.
    result = {"messages": [{"role": "assistant", "content": "d"}]}
    stored = build(prior, "c", result, "d")
    assert [m["content"] for m in stored] == ["a", "b", "c", "d"]
