"""Lifecycle status classification for session pickers.

Covers ``classify_session_status`` (pure last-message shape → status) and
``session_lifecycle_statuses`` (batched last-message lookup), plus
the delete wiring the picker's 'd' key relies on.
"""

import pytest

from persistence.conversation.sessions import SessionLifecycleStatus
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    SESSION_STATUS_COMPLETE,
    SESSION_STATUS_ERROR,
    SESSION_STATUS_INTERRUPTED,
    classify_session_status,
    create_session,
    delete_session,
    get_session,
    session_lifecycle_statuses,
)
from tests.persistence_sqlite.conftest import store_for

# The status of a session with no message row: the backend writes it as a literal.
SESSION_STATUS_EMPTY = SessionLifecycleStatus.EMPTY.value


@pytest.fixture
def db(backend, scope_a):
    return backend.open(scope_a).holder()


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
    create_session(db, "s_complete", source="cli")
    append_message(db, "s_complete", "user", "hi")
    append_message(db, "s_complete", "assistant", "hello", finish_reason="stop")

    # interrupted: user asked, no reply landed
    create_session(db, "s_user_tail", source="cli")
    append_message(db, "s_user_tail", "user", "are you there?")

    # interrupted: assistant fired tool calls, no tool result followed
    create_session(db, "s_pending_tool", source="cli")
    append_message(db, "s_pending_tool", "user", "run it")
    append_message(
        db,
        "s_pending_tool",
        "assistant",
        None,
        tool_calls=[{"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}],
        finish_reason="tool_calls",
    )

    # complete: full tool round-trip then final assistant reply
    create_session(db, "s_tool_roundtrip", source="cli")
    append_message(db, "s_tool_roundtrip", "user", "run it")
    append_message(
        db,
        "s_tool_roundtrip",
        "assistant",
        None,
        tool_calls=[{"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}],
        finish_reason="tool_calls",
    )
    append_message(db, "s_tool_roundtrip", "tool", "ok", tool_call_id="c1")
    append_message(db, "s_tool_roundtrip", "assistant", "done", finish_reason="stop")

    # interrupted: tool result present but assistant never consumed it
    create_session(db, "s_tool_tail", source="cli")
    append_message(db, "s_tool_tail", "user", "run it")
    append_message(
        db,
        "s_tool_tail",
        "assistant",
        None,
        tool_calls=[{"id": "c2", "function": {"name": "terminal", "arguments": "{}"}}],
        finish_reason="tool_calls",
    )
    append_message(db, "s_tool_tail", "tool", "ok", tool_call_id="c2")

    # error: last message carries an error finish_reason
    create_session(db, "s_error", source="cli")
    append_message(db, "s_error", "user", "hi")
    append_message(db, "s_error", "assistant", "boom", finish_reason="error")

    # empty: session row exists, zero messages
    create_session(db, "s_empty", source="cli")

    statuses = session_lifecycle_statuses(
        db,
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
    assert session_lifecycle_statuses(db, []) == {}
    assert session_lifecycle_statuses(db, [None, ""]) == {}


def test_session_lifecycle_statuses_unknown_id(db):
    # Unknown ids classify as 'empty' (no messages), never raise.
    assert session_lifecycle_statuses(db, ["nope"]) == {
        "nope": SESSION_STATUS_EMPTY
    }


# ---------------------------------------------------------------------------
# Picker helpers (status annotation + delete wiring)
# ---------------------------------------------------------------------------

def test_annotate_session_statuses(db, scope_a):
    from hermes_cli.sessions_cmd_browse import _annotate_session_statuses, _session_status_tag
    from persistence_assembly import open_persistence_reader

    create_session(db, "s1", source="cli")
    append_message(db, "s1", "user", "hi")
    append_message(db, "s1", "assistant", "hello", finish_reason="stop")
    create_session(db, "s2", source="cli")
    append_message(db, "s2", "user", "hi")

    rows = [{"id": "s1"}, {"id": "s2"}]
    with open_persistence_reader(store_for(scope_a)) as store:
        _annotate_session_statuses(rows, store)
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
    create_session(db, "doomed", source="cli")
    append_message(db, "doomed", "user", "hi")
    append_message(db, "doomed", "assistant", "hello", finish_reason="stop")

    media_root = str(tmp_path / "media")
    assert delete_session(db, "doomed", sessions_dir=tmp_path / "sessions", media_root=media_root) == ("doomed",)
    assert get_session(db, "doomed") is None
    remaining = db._read_one(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?", ("doomed",)
    )[0]
    assert remaining == 0
    # Deleting again reports nothing deleted (not found)
    assert delete_session(db, "doomed", media_root=media_root) == ()
