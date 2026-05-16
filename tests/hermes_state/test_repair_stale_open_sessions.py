import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _age_session(db: SessionDB, session_id: str, timestamp: float) -> None:
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (timestamp, session_id),
    )
    db._conn.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id = ?",
        (timestamp, session_id),
    )
    db._conn.commit()


def _end_reason(db: SessionDB, session_id: str):
    row = db._conn.execute(
        "SELECT end_reason FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return row[0]


def test_repair_stale_open_sessions_closes_terminal_rows_only(db):
    old = time.time() - 10_000

    db.create_session("assistant-stop", source="tui")
    db.append_message(
        "assistant-stop",
        role="assistant",
        content="done",
        finish_reason="stop",
    )

    db.create_session("tool-result", source="tui")
    db.append_message("tool-result", role="tool", content="ok")

    db.create_session("empty-old", source="tui")

    db.create_session("user-waiting", source="tui")
    db.append_message("user-waiting", role="user", content="still waiting")

    db.create_session("assistant-tool-call", source="tui")
    db.append_message(
        "assistant-tool-call",
        role="assistant",
        content="calling tool",
        finish_reason="tool_calls",
    )

    for session_id in (
        "assistant-stop",
        "tool-result",
        "empty-old",
        "user-waiting",
        "assistant-tool-call",
    ):
        _age_session(db, session_id, old)

    repaired = db.repair_stale_open_sessions(
        older_than_seconds=60,
        end_reason="test_repair",
    )

    assert repaired == 3
    assert _end_reason(db, "assistant-stop") == "test_repair"
    assert _end_reason(db, "tool-result") == "test_repair"
    assert _end_reason(db, "empty-old") == "test_repair"
    assert _end_reason(db, "user-waiting") is None
    assert _end_reason(db, "assistant-tool-call") is None


def test_repair_stale_open_sessions_leaves_recent_terminal_rows_open(db):
    db.create_session("recent", source="tui")
    db.append_message("recent", role="assistant", content="done", finish_reason="stop")

    repaired = db.repair_stale_open_sessions(older_than_seconds=60)

    assert repaired == 0
    assert _end_reason(db, "recent") is None
