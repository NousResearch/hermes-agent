"""A compaction merge/flip must not leave assistant reasoning sidecars on a non-assistant row.

Real-world failure: after context compaction merged its summary into a carrier row, the row
kept ``reasoning``/``reasoning_content`` while carrying ``role="user"``. llama.cpp (and other
strict OpenAI-compatible backends) reject that with HTTP 400 ``invalid_messages``
("'reasoning_content' is only valid as a string on assistant messages"), the error is not
retryable, and every later turn of the session replays the same rejected history.
"""

from agent.context_compressor import ContextCompressor, _drop_stale_reasoning_fields
from agent.transports.chat_completions import ChatCompletionsTransport


# --- sanitizer guard (defense in depth) ---------------------------------------------------

def test_non_assistant_rows_never_replay_reasoning_keys():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "merged carrier",
         "reasoning": "stale", "reasoning_content": "stale", "reasoning_details": [{"type": "x"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "out", "reasoning_content": "stale"},
    ]
    out = ChatCompletionsTransport().convert_messages(messages)
    by_index = {i: m for i, m in enumerate(out)}
    for i in (1, 2):
        row = by_index[i]
        assert "reasoning" not in row and "reasoning_content" not in row, row
        assert "reasoning_details" not in row, row
    assert "content" in by_index[1] and by_index[1]["content"] == "merged carrier"


def test_assistant_rows_keep_reasoning_content_by_default():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok", "reasoning_content": "because"},
    ]
    out = ChatCompletionsTransport().convert_messages(messages)
    assistant = [m for m in out if m.get("role") == "assistant"][0]
    assert assistant.get("reasoning_content") == "because"


# --- compressor: merged carrier row --------------------------------------------------------

def _bare_compressor() -> ContextCompressor:
    comp = ContextCompressor.__new__(ContextCompressor)
    comp._summary_has_user_turn = True
    return comp


def test_merge_into_tail_row_drops_reasoning_sidecars():
    comp = _bare_compressor()
    carrier = {"role": "user", "content": "old tail text",
               "reasoning": "minted for pre-merge text",
               "reasoning_content": "minted for pre-merge text",
               "reasoning_details": [{"type": "resp_item"}]}
    comp._merge_summary_into_tail_row(carrier, "SUMMARY TEXT", "user", force_user_leading=False)
    assert "reasoning" not in carrier and "reasoning_content" not in carrier
    assert "reasoning_details" not in carrier
    assert carrier["content"].startswith("[PRIOR CONTEXT")
    assert "SUMMARY TEXT" in carrier["content"]


def test_force_user_leading_merge_drops_reasoning_sidecars():
    comp = _bare_compressor()
    carrier = {"role": "user", "content": "request text", "reasoning_content": "stale"}
    comp._merge_summary_into_tail_row(carrier, "SUMMARY TEXT", "user", force_user_leading=True)
    assert "reasoning_content" not in carrier


def test_drop_helper_is_noop_without_keys():
    msg = {"role": "user", "content": "x"}
    _drop_stale_reasoning_fields(msg)
    assert msg == {"role": "user", "content": "x"}
