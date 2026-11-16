"""A compaction carries a tool result forward as a shortened stub (same ``message_uid``). Every display
read paints that tool result once, as the archived full original, while the model keeps the stub and
no stored row changes (#132939, #135655)."""
import pytest

from hermes_state import SessionDB
from hermes_state_timeline import get_session_messages_around

FULL = "FULL " * 1000
STUB = "[read_file] pruned stub"


@pytest.fixture()
def db(tmp_path):
    instance = SessionDB(tmp_path / "state.db")
    yield instance
    instance.close()


def _painted_tools(db, sid):
    _model, resumed = db.get_resume_conversations(sid)
    paged = db.get_messages(sid, include_compacted=True)
    shown = db.get_messages_as_conversation(sid, include_compacted=True)
    around = get_session_messages_around(db, sid, paged[0]["id"])["messages"]
    return [[m.get("content") for m in read if m["role"] == "tool"] for read in (resumed, paged, shown, around)]


def test_compacted_tool_result_paints_once_as_its_archived_original(db):
    sid = "stub-display"
    db.create_session(sid, source="test")
    db.append_messages_batch(sid, [
        {"role": "user", "content": "read it", "timestamp": 100.0},
        {"role": "assistant", "content": "", "timestamp": 101.0,
         "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "content": FULL, "tool_call_id": "call-1", "tool_name": "read_file", "timestamp": 102.0,
         "display_metadata": {"tool_result_metadata": {"lines": 1000}}},
        {"role": "assistant", "content": "done reading", "timestamp": 103.0},
    ])
    for generation in (1, 2):  # the stub is carried again by a later compaction
        history = db.get_messages_as_conversation(sid, include_row_ids=True)
        tool = next(m for m in history if m["role"] == "tool")
        tool["content"] = STUB
        db.archive_and_compact(sid, [{"role": "user", "content": f"summary {generation}", "timestamp": 150.0 + generation},
                                     *[m for m in history if not m.get("_compressed_summary")]])
        for bodies in _painted_tools(db, sid):
            assert bodies == [FULL]
        assert db.display_message_count(sid) == len(db.get_messages(sid, include_compacted=True))
        model_tool = [m for m in db.get_messages_as_conversation(sid) if m["role"] == "tool"]
        assert [m["content"] for m in model_tool] == [STUB]
        assert model_tool[0].get("display_metadata") == {"tool_result_metadata": {"lines": 1000}}
