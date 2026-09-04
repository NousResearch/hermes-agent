"""Lifecycle status classification for session pickers.

Covers ``classify_session_status`` (pure last-message shape → status) and
``SessionDB.session_lifecycle_statuses`` (batched last-message lookup), plus
the delete wiring the picker's 'd' key relies on.
"""

import pytest

from hermes_state import SessionDB
from hermes_state_sessions import (
    SESSION_STATUS_COMPLETE,
    SESSION_STATUS_EMPTY,
    SESSION_STATUS_ERROR,
    SESSION_STATUS_INTERRUPTED,
    classify_session_status,
)


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


# ---------------------------------------------------------------------------
# Pure classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "role,has_tool_calls,finish_reason,expected",
    [
        ("assistant", False, "stop", SESSION_STATUS_COMPLETE),
        ("assistant", False, None, SESSION_STATUS_COMPLETE),
        ("assistant", False, "length", SESSION_STATUS_COMPLETE),
        ("assistant", True, "tool_calls", SESSION_STATUS_INTERRUPTED),
        ("user", False, None, SESSION_STATUS_INTERRUPTED),
        ("tool", False, None, SESSION_STATUS_INTERRUPTED),
        ("assistant", False, "error", SESSION_STATUS_ERROR),
        ("assistant", True, "error", SESSION_STATUS_ERROR),
        ("user", False, "agent_error", SESSION_STATUS_ERROR),
        ("system", False, None, SESSION_STATUS_COMPLETE),
        (None, False, None, SESSION_STATUS_COMPLETE),
    ],
)
def test_classify_session_status(role, has_tool_calls, finish_reason, expected):
    assert classify_session_status(role, has_tool_calls, finish_reason) == expected


# ---------------------------------------------------------------------------
# DB-backed batch classification
# ---------------------------------------------------------------------------

def test_session_lifecycle_statuses_shapes(db):
    # complete: normal user → assistant exchange
    db.create_session("s_complete", source="cli")
    db.append_message("s_complete", "user", "hi")
    db.append_message("s_complete", "assistant", "hello", finish_reason="stop")

    # interrupted: user asked, no reply landed
    db.create_session("s_user_tail", source="cli")
    db.append_message("s_user_tail", "user", "are you there?")

    # interrupted: assistant fired tool calls, no tool result followed
    db.create_session("s_pending_tool", source="cli")
    db.append_message("s_pending_tool", "user", "run it")
    db.append_message(
        "s_pending_tool",
        "assistant",
        None,
        tool_calls=[{"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}],
        finish_reason="tool_calls",
    )

    # complete: full tool round-trip then final assistant reply
    db.create_session("s_tool_roundtrip", source="cli")
    db.append_message("s_tool_roundtrip", "user", "run it")
    db.append_message(
        "s_tool_roundtrip",
        "assistant",
        None,
        tool_calls=[{"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}],
        finish_reason="tool_calls",
    )
    db.append_message("s_tool_roundtrip", "tool", "ok", tool_call_id="c1")
    db.append_message("s_tool_roundtrip", "assistant", "done", finish_reason="stop")

    # interrupted: tool result present but assistant never consumed it
    db.create_session("s_tool_tail", source="cli")
    db.append_message("s_tool_tail", "user", "run it")
    db.append_message(
        "s_tool_tail",
        "assistant",
        None,
        tool_calls=[{"id": "c2", "function": {"name": "terminal", "arguments": "{}"}}],
        finish_reason="tool_calls",
    )
    db.append_message("s_tool_tail", "tool", "ok", tool_call_id="c2")

    # error: last message carries an error finish_reason
    db.create_session("s_error", source="cli")
    db.append_message("s_error", "user", "hi")
    db.append_message("s_error", "assistant", "boom", finish_reason="error")

    # empty: session row exists, zero messages
    db.create_session("s_empty", source="cli")

    statuses = db.session_lifecycle_statuses(
        [
            "s_complete",
            "s_user_tail",
            "s_pending_tool",
            "s_tool_roundtrip",
            "s_tool_tail",
            "s_error",
            "s_empty",
        ]
    )
    assert statuses == {
        "s_complete": SESSION_STATUS_COMPLETE,
        "s_user_tail": SESSION_STATUS_INTERRUPTED,
        "s_pending_tool": SESSION_STATUS_INTERRUPTED,
        "s_tool_roundtrip": SESSION_STATUS_COMPLETE,
        "s_tool_tail": SESSION_STATUS_INTERRUPTED,
        "s_error": SESSION_STATUS_ERROR,
        "s_empty": SESSION_STATUS_EMPTY,
    }


def test_session_lifecycle_statuses_empty_input(db):
    assert db.session_lifecycle_statuses([]) == {}
    assert db.session_lifecycle_statuses([None, ""]) == {}


def test_session_lifecycle_statuses_unknown_id(db):
    # Unknown ids classify as 'empty' (no messages), never raise.
    assert db.session_lifecycle_statuses(["nope"]) == {
        "nope": SESSION_STATUS_EMPTY
    }


# ---------------------------------------------------------------------------
# Picker helpers (status annotation + delete wiring)
# ---------------------------------------------------------------------------

def test_annotate_session_statuses(db):
    from hermes_cli.sessions_cmd_browse import _annotate_session_statuses, _session_status_tag

    db.create_session("s1", source="cli")
    db.append_message("s1", "user", "hi")
    db.append_message("s1", "assistant", "hello", finish_reason="stop")
    db.create_session("s2", source="cli")
    db.append_message("s2", "user", "hi")

    rows = [{"id": "s1"}, {"id": "s2"}]
    _annotate_session_statuses(rows, db)
    assert rows[0]["_status"] == SESSION_STATUS_COMPLETE
    assert rows[1]["_status"] == SESSION_STATUS_INTERRUPTED

    # No db → rows untouched, tag falls back to '-'
    bare = [{"id": "s1"}]
    _annotate_session_statuses(bare, None)
    assert "_status" not in bare[0]
    assert _session_status_tag(bare[0].get("_status")) == "-"

    # Tag mapping
    assert _session_status_tag(SESSION_STATUS_COMPLETE) == "done"
    assert _session_status_tag(SESSION_STATUS_INTERRUPTED) == "intr"
    assert _session_status_tag(SESSION_STATUS_ERROR) == "err"
    assert _session_status_tag(SESSION_STATUS_EMPTY) == "empty"


def test_delete_session_removes_session_and_messages(db, tmp_path):
    db.create_session("doomed", source="cli")
    db.append_message("doomed", "user", "hi")
    db.append_message("doomed", "assistant", "hello", finish_reason="stop")

    assert db.delete_session("doomed", sessions_dir=tmp_path / "sessions") is True
    assert db.get_session("doomed") is None
    remaining = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?", ("doomed",)
    ).fetchone()[0]
    assert remaining == 0
    # Deleting again reports False (not found)
    assert db.delete_session("doomed") is False
