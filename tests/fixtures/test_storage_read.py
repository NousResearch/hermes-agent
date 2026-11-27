"""The read helpers return the session row, the stored message rows and the replay messages as dicts."""
from persistence.conversation.messages import ConversationRequest
from tests.fixtures.storage_read import conversation, session_row, transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
MESSAGES = ({"role": "user", "content": "hi"}, {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
            {"role": "tool", "content": "out", "tool_call_id": "c1", "tool_name": "f"})


def test_session_row_is_the_row_dict_or_none(tmp_path):
    store = storage_handle(tmp_path)
    seed_session(store, "s", "cli", model="m", title="T", end_reason="user_exit")
    row = session_row(store, "s")
    assert (row["id"], row["source"], row["model"], row["title"], row["title_source"], row["end_reason"]) == (
        "s", "cli", "m", "T", "user", "user_exit")
    assert (row["hidden"], row["archived"], row["pinned"]) == (0, 0, 0)
    assert session_row(store, "missing") is None


def test_transcript_rows_are_the_stored_rows_in_order(tmp_path):
    store = storage_handle(tmp_path)
    seed_session(store, "s", "cli")
    ids = seed_messages(store, "s", *MESSAGES)
    rows = transcript_rows(store, "s")
    assert [(row["id"], row["session_id"], row["role"], row["content"]) for row in rows] == [
        (ids[0], "s", "user", "hi"), (ids[1], "s", "assistant", ""), (ids[2], "s", "tool", "out")]
    assert (rows[1]["tool_calls"], rows[2]["tool_call_id"]) == ([TOOL_CALL], "c1")
    assert {(row["active"], row["compacted"]) for row in rows} == {(1, 0)}


def test_conversation_is_the_openai_format_replay(tmp_path):
    store = storage_handle(tmp_path)
    seed_session(store, "s", "cli")
    ids = seed_messages(store, "s", *MESSAGES)
    messages = conversation(store, "s")
    assert [(m["role"], m["content"]) for m in messages] == [("user", "hi"), ("assistant", ""), ("tool", "out")]
    assert (messages[1]["tool_calls"], messages[2]["tool_call_id"], messages[2]["tool_name"]) == ([TOOL_CALL], "c1", "f")
    assert [m["_row_id"] for m in conversation(store, "s", ConversationRequest(include_row_ids=True))] == ids
