import time
from pathlib import Path
try:
    import pytest
    fixture = pytest.fixture
except ImportError:
    fixture = lambda f: f

from hermes_state import SessionDB


@fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_state.db"
    db = SessionDB(db_path=db_file)
    yield db
    db.close()


def test_clear_session_messages_full(temp_db, tmp_path):
    sid = "test_sess_full_clear"
    temp_db.create_session(
        session_id=sid,
        source="cli",
        model="gpt-5",
    )
    temp_db.set_session_title(sid, "My Project Session")
    temp_db.set_session_pinned(sid, True)

    # Append some messages
    temp_db.append_message(sid, role="user", content="Hello", token_count=10)
    temp_db.append_message(sid, role="assistant", content="Hi there!", token_count=20)
    temp_db.append_message(sid, role="user", content="Do task", token_count=15)
    temp_db.append_message(
        sid,
        role="assistant",
        content="",
        tool_calls=[{"id": "call_1", "name": "terminal", "args": {"command": "ls"}}],
        token_count=30,
    )
    temp_db.append_message(
        sid,
        role="tool",
        content="file1.txt\nfile2.txt",
        tool_call_id="call_1",
        token_count=50,
    )

    # Verify messages and counters before clear
    msgs_before = temp_db.get_messages(sid)
    assert len(msgs_before) == 5
    sess_before = temp_db.get_session(sid)
    assert sess_before["message_count"] == 5
    assert sess_before["pinned"] == 1
    assert sess_before["title"] == "My Project Session"

    # Create dummy transcript files
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / f"{sid}.json").write_text("{}")
    (sessions_dir / f"{sid}.jsonl").write_text("{}")

    # Clear messages
    cleared = temp_db.clear_session_messages(sid, sessions_dir=sessions_dir)
    assert cleared is True

    # Verify messages and counters after clear
    msgs_after = temp_db.get_messages(sid)
    assert len(msgs_after) == 0

    sess_after = temp_db.get_session(sid)
    assert sess_after["message_count"] == 0
    assert sess_after["tool_call_count"] == 0
    assert sess_after["input_tokens"] == 0
    assert sess_after["output_tokens"] == 0
    # Metadata preserved
    assert sess_after["title"] == "My Project Session"
    assert sess_after["pinned"] == 1
    assert sess_after["source"] == "cli"

    # Files removed
    assert not (sessions_dir / f"{sid}.json").exists()
    assert not (sessions_dir / f"{sid}.jsonl").exists()


def test_clear_session_messages_keep_last_n(temp_db):
    sid = "test_sess_keep_last"
    temp_db.create_session(session_id=sid, source="cli")
    temp_db.set_session_title(sid, "Keep Last Test")

    for i in range(10):
        temp_db.append_message(sid, role="user" if i % 2 == 0 else "assistant", content=f"msg_{i}")

    assert len(temp_db.get_messages(sid)) == 10

    # Keep only the last 3 messages
    cleared = temp_db.clear_session_messages(sid, keep_last_n=3)
    assert cleared is True

    msgs_after = temp_db.get_messages(sid)
    assert len(msgs_after) == 3
    assert [m["content"] for m in msgs_after] == ["msg_7", "msg_8", "msg_9"]

    sess_after = temp_db.get_session(sid)
    assert sess_after["message_count"] == 3
    assert sess_after["title"] == "Keep Last Test"


def test_clear_session_messages_before_timestamp(temp_db):
    sid = "test_sess_before_ts"
    temp_db.create_session(session_id=sid, source="cli")
    temp_db.set_session_title(sid, "Before Timestamp Test")

    t0 = 1000.0
    temp_db.append_message(sid, role="user", content="old_1", timestamp=t0)
    temp_db.append_message(sid, role="assistant", content="old_2", timestamp=t0 + 10)
    temp_db.append_message(sid, role="user", content="new_1", timestamp=t0 + 100)
    temp_db.append_message(sid, role="assistant", content="new_2", timestamp=t0 + 110)

    assert len(temp_db.get_messages(sid)) == 4

    # Delete messages before t0 + 50
    cleared = temp_db.clear_session_messages(sid, before_timestamp=t0 + 50)
    assert cleared is True

    msgs_after = temp_db.get_messages(sid)
    assert len(msgs_after) == 2
    assert [m["content"] for m in msgs_after] == ["new_1", "new_2"]

    sess_after = temp_db.get_session(sid)
    assert sess_after["message_count"] == 2


def test_clear_session_messages_not_found(temp_db):
    cleared = temp_db.clear_session_messages("nonexistent_session")
    assert cleared is False


def test_delete_session_messages_selective(temp_db):
    sid = "test_sess_selective"
    temp_db.create_session(session_id=sid, source="cli")
    temp_db.set_session_title(sid, "Selective Delete")

    m1 = temp_db.append_message(sid, role="user", content="msg 1")
    m2 = temp_db.append_message(sid, role="assistant", content="msg 2")
    m3 = temp_db.append_message(sid, role="user", content="msg 3")

    assert len(temp_db.get_messages(sid)) == 3

    # Delete message 2
    deleted = temp_db.delete_session_messages(sid, [m2])
    assert deleted == 1

    msgs_after = temp_db.get_messages(sid)
    assert len(msgs_after) == 2
    assert [m["id"] for m in msgs_after] == [m1, m3]

    sess_after = temp_db.get_session(sid)
    assert sess_after["message_count"] == 2


def test_list_sessions_rich_cleared_session_visibility(temp_db):
    # 1. Blank draft session (no messages, no title, not pinned) -> hidden with min_message_count=1
    temp_db.create_session("draft_1", source="cli")

    # 2. Real session with messages -> visible
    temp_db.create_session("active_1", source="cli")
    temp_db.append_message("active_1", role="user", content="hello")

    # 3. Cleared session with a title -> visible
    temp_db.create_session("cleared_1", source="cli")
    temp_db.set_session_title("cleared_1", "My Saved Chat")

    # 4. Cleared session pinned -> visible
    temp_db.create_session("pinned_1", source="cli")
    temp_db.set_session_pinned("pinned_1", True)

    sessions = temp_db.list_sessions_rich(min_message_count=1, order_by_last_active=True)
    ids = [s["id"] for s in sessions]

    assert "draft_1" not in ids
    assert "active_1" in ids
    assert "cleared_1" in ids
    assert "pinned_1" in ids
    assert temp_db.session_count(min_message_count=1) == 3

