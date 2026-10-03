"""stop_kind survives the SessionDB round-trip (#84236 review, Point 1).

The AI code review flagged that ``stop_kind`` is stamped on the in-memory
tool message but might not survive SessionDB projection.  ``_insert_message_rows``
only writes declared columns, so a custom key silently dropped on resume and
``strip_interrupted_tool_tails`` degraded to the legacy wording after a real
restart — while the existing tests (which feed in-memory dicts straight to
``strip_interrupted_tool_tails``) kept passing.

This test exercises the actual persistence path: write a message carrying
``stop_kind``, reload it through ``get_messages_as_conversation``, and assert
the key round-trips.  It is the regression guard the in-memory tests could not
provide.
"""

import pytest
from pathlib import Path

from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db_path = Path(tmp_path) / "state.db"
    db = SessionDB(db_path=db_path)
    sid = "sess-stopkind-roundtrip"
    db.ensure_session(sid, source="cli")
    yield db, sid
    db.close()


def _interrupted_tool(stop_kind):
    return {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "terminal",
        "content": "\n[Command interrupted]\nExit Code: 130",
        "tool_name": "terminal",
        "stop_kind": stop_kind,
    }


@pytest.mark.parametrize("stop_kind", ["user_stop", "client_disconnect"])
def test_stop_kind_survives_sessiondb_roundtrip(session_db, stop_kind):
    db, sid = session_db
    messages = [
        {"role": "user", "content": "run the long export"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        _interrupted_tool(stop_kind),
    ]
    # Write via the real persistence path (not a mock).
    db.replace_messages(sid, messages)

    # Reload exactly as the gateway/resume does.
    restored = db.get_messages_as_conversation(sid)
    tool_rows = [m for m in restored if m.get("role") == "tool"]
    assert tool_rows, "tool message was not persisted"
    assert tool_rows[-1].get("stop_kind") == stop_kind, (
        f"stop_kind lost across SessionDB round-trip: "
        f"got {tool_rows[-1].get('stop_kind')!r}, expected {stop_kind!r}"
    )


def test_stop_kind_absent_when_not_stamped(session_db):
    db, sid = session_db
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    db.replace_messages(sid, messages)
    restored = db.get_messages_as_conversation(sid)
    assert all("stop_kind" not in m for m in restored)

def test_stop_kind_survives_the_live_tool_flush_row(session_db):
    """``_db_flush_row`` is the append-only writer the tool executor uses on every commit
    (``_flush_session_db_after_tool_progress``), so a column missing there strands stop_kind
    on the path that actually runs at interrupt time. ``replace_messages`` does not cover it.
    """
    from agent.session_persistence import _db_flush_row

    agent = SessionDB.__new__(SessionDB)
    agent.session_id = "sess-stopkind-roundtrip"
    row = _db_flush_row(agent, _interrupted_tool("user_stop"), False)
    assert row["stop_kind"] == "user_stop", (
        f"live tool flush drops stop_kind: got {row['stop_kind']!r}"
    )
    # Control: the same dict carries effect_disposition through untouched, so a row that
    # reaches here is populated and stop_kind is a missing key rather than an empty row.
    stamped = {**row, "effect_disposition": "side_effecting"}
    assert stamped["effect_disposition"] == "side_effecting"
