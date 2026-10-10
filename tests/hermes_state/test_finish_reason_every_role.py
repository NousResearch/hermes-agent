"""finish_reason survives on every role."""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


def test_finish_reason_is_read_back_on_a_tool_row(db):
    db.create_session("s1", source="cli")
    db.append_message("s1", "user", "q")
    db.append_message("s1", "assistant", "a", finish_reason="tool_calls")
    db.append_message("s1", "tool", "partial", tool_call_id="c1", finish_reason="length")
    rows = db.get_messages_as_conversation("s1")
    by_role = {m["role"]: m for m in rows}
    assert by_role["assistant"].get("finish_reason") == "tool_calls"  # unchanged path
    assert by_role["tool"].get("finish_reason") == "length"

