"""#117750: a prune rewrite of a carried-forward tool call must not split the
logical event into a second display identity.

``_display_dedupe_key`` includes the payload bytes (call arguments). The
proactive prune shortens exactly those bytes on the carried copies it
publishes, so the rewritten assistant copy and its archived durable original
used to project as two logical messages — the older one reappearing after a
later answer. Tool-calling assistant rows are now keyed on their stable call
ids; tool RESULT rows keep their content key so the archived full output is
never folded into its pruned stub. These tests pin both directions (rewritten
calls collapse without losing the original output, distinct calls do not merge).
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
    and the assistant sequence must not gain a duplicate of the earlier message
    while the archived full tool output stays in compacted history and exports."""
    sid = "pruned-payload"
    _seed_session(db, sid)
    history = db.get_messages_as_conversation(sid, include_row_ids=True)
    history[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"value": "short"})
    history[2]["content"] = "short result"
    db.archive_and_compact(sid, history)
    visible = db.get_messages_as_conversation(sid, include_row_ids=True, include_compacted=True)
    assert [m.get("content") for m in visible if m["role"] == "assistant"] == \
        ["Earlier progress", "Later answer"]
    exported = db.export_session(sid, include_compacted=True)["messages"]
    for messages in (visible, exported):
        assert "R" * 5000 in [m.get("content") for m in messages if m["role"] == "tool"]


def test_distinct_idless_assistant_calls_are_not_merged(db):
    """Only a complete set of call ids replaces the content key: two id-less calls
    sharing a timestamp but differing in arguments stay separate display events."""
    sid = "idless-calls"
    db.create_session(sid, source="test")
    db.append_messages_batch(sid, [
        {"role": "assistant", "content": "working", "timestamp": 1700000000.0,
         "tool_calls": [{"type": "function", "function":
                         {"name": "demo_tool", "arguments": json.dumps({"q": q})}}]}
        for q in ("a", "b")])
    assert len(db.get_messages(sid, include_compacted=True)) == 2
