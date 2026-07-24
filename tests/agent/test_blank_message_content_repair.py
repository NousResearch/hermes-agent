"""Regression: heal blank assistant text blocks that wedge a session.

An assistant message stored with empty/whitespace ``content`` while it carries
``tool_calls`` serializes to an empty text content block. The Anthropic Messages
API (and Bedrock) reject a request containing a blank text block with HTTP 400
``text content blocks must contain non-whitespace text``. Because the blank is
in stored history it replays every turn, so the session hangs with no output.

The request-path adapters coerce blanks on send, but rows persisted before that
coercion existed stay wedged. ``repair_blank_message_content`` heals them in
place. This test uses synthetic rows only.
"""
import sqlite3

import pytest

from hermes_state import SessionDB, repair_blank_message_content


def _seed(db_path, rows):
    """Insert (role, content, tool_calls) rows into a real, initialized DB."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'test', 0)"
        )
        for i, (role, content, tool_calls) in enumerate(rows):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, timestamp) "
                "VALUES ('s1', ?, ?, ?, ?)",
                (role, content, tool_calls, float(i)),
            )
        conn.commit()
    finally:
        conn.close()


def _contents(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        return [r[0] for r in conn.execute(
            "SELECT content FROM messages ORDER BY timestamp"
        ).fetchall()]
    finally:
        conn.close()


@pytest.fixture()
def db_path(tmp_path):
    p = tmp_path / "state.db"
    SessionDB(db_path=p).close()  # initialize schema
    return p


TOOL_CALLS = '[{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]'


def test_heals_blank_assistant_toolcall_content(db_path):
    _seed(db_path, [
        ("user", "hi", None),
        ("assistant", "", TOOL_CALLS),        # poison: empty + tool_calls
        ("assistant", "   \n", TOOL_CALLS),   # poison: whitespace + tool_calls
        ("assistant", None, TOOL_CALLS),      # poison: NULL + tool_calls
    ])
    report = repair_blank_message_content(db_path, backup=False)
    assert report["repaired"] is True
    assert report["affected"] == 3
    assert report["sessions"] == 1
    contents = _contents(db_path)
    assert contents[0] == "hi"
    for healed in contents[1:]:
        assert healed and healed.strip(), f"blank survived: {healed!r}"


def test_leaves_real_content_and_plain_empty_untouched(db_path):
    _seed(db_path, [
        ("assistant", "real answer", TOOL_CALLS),  # real text: keep verbatim
        ("assistant", "", None),                   # empty but NO tool_calls: not this class
        ("user", "", None),                        # user empty: never touched
    ])
    report = repair_blank_message_content(db_path, backup=False)
    assert report["affected"] == 0
    assert report["repaired"] is False
    contents = _contents(db_path)
    assert contents[0] == "real answer"
    assert contents[1] == ""  # untouched
    assert contents[2] == ""  # untouched


def test_check_only_reports_without_writing(db_path):
    _seed(db_path, [("assistant", "", TOOL_CALLS)])
    report = repair_blank_message_content(db_path, check_only=True, backup=False)
    assert report["affected"] == 1
    assert report["repaired"] is False
    assert _contents(db_path) == [""]  # unchanged


def test_missing_db_is_noop(tmp_path):
    report = repair_blank_message_content(tmp_path / "nope.db")
    assert report == {
        "repaired": False, "affected": 0, "sessions": 0,
        "backup_path": None, "error": None,
    }
