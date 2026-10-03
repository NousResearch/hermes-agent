"""#117750: a prune rewrite of a carried-forward tool payload must not split the
logical event into a second display identity.

``_display_dedupe_key`` includes the payload bytes (tool-result content, call
arguments). The proactive prune shortens exactly those bytes on the carried
copies it publishes, so the rewritten copy and its archived durable original
used to project as two logical messages — the older one reappearing after a
later answer. Tool rows are now keyed on their stable tool identity; these
tests pin both directions (rewritten copies collapse, genuinely distinct calls
do not merge).
"""
import json

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    instance = SessionDB(tmp_path / "state.db")
    yield instance
    instance.close()


def _seed_session(db, sid):
    db.create_session(sid, source="test")
    db.append_messages_batch(sid, [
        {"role": "user", "content": "start", "timestamp": 100.0},
        {"role": "assistant", "content": "Earlier progress", "timestamp": 101.0,
         "tool_calls": [{"id": "stable-call", "type": "function", "function":
                         {"name": "demo_tool", "arguments": json.dumps({"value": "L" * 4000})}}]},
        {"role": "tool", "content": "R" * 5000, "tool_call_id": "stable-call",
         "tool_name": "demo_tool", "timestamp": 102.0},
        {"role": "assistant", "content": "Later answer", "timestamp": 200.0},
    ])


def test_pruned_tool_payload_keeps_one_display_identity(db):
    """The issue's synthetic repro: shorten the carried tool args + result, commit,
    and the assistant sequence must not gain a duplicate of the earlier message."""
    sid = "pruned-payload"
    _seed_session(db, sid)
    history = db.get_messages_as_conversation(sid, include_row_ids=True)
    history[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"value": "short"})
    history[2]["content"] = "short result"
    db.archive_and_compact(sid, history)
    visible = db.get_messages_as_conversation(sid, include_row_ids=True, include_compacted=True)
    assert [m.get("content") for m in visible if m["role"] == "assistant"] == \
        ["Earlier progress", "Later answer"]
    tool_rows = [m for m in visible if m["role"] == "tool"]
    assert len(tool_rows) == 1


def test_distinct_assistant_calls_with_same_args_are_not_merged(db):
    """The widened key keys assistant rows on their call IDS: two different calls
    that happen to share arguments and timestamp stay separate display events."""
    sid = "distinct-calls"
    db.create_session(sid, source="test")
    db._execute_write(lambda conn: [
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, timestamp,"
            " active, compacted) VALUES (?, ?, ?, ?, ?, 1, 0)",
            (sid, "assistant", "working",
             json.dumps([{"id": cid, "type": "function",
                          "function": {"name": "demo_tool",
                                       "arguments": json.dumps({"q": "same"})}}]),
             1700000000.0))
        for cid in ("call-1", "call-2")])
    msgs = db.get_messages(sid, include_compacted=True)
    assert len(msgs) == 2
