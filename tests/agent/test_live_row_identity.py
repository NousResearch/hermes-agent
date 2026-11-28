"""A tool round's interim and tool events name the stored row they are (agent/live_row_identity.py).

The round persists its assistant row before it emits the row's commentary or runs its calls, so each
event can carry that row's ``row_id`` / ``message_uid`` (and a call its ``tool_call_uid``). The tests
read the identity the way the tui_gateway emitters do, from inside the callbacks of a real agent
writing to a real store, and compare it with the rows the store holds.
"""

import json
from unittest.mock import patch

from agent.live_row_identity import (
    LIVE_ROW_ATTR, final_reply_row_identity, interim_row_identity, reply_row_payload, reply_row_result_fields, tool_call_row_identity,
)
from tests.agent.test_tool_call_incremental_persistence import (
    _attach_real_session_db, _make_agent, _mock_response, _mock_tool_call,
)
from tests.fixtures.storage_read import transcript_rows


def _run_two_rounds(tmp_path):
    agent = _make_agent()
    db = _attach_real_session_db(agent, tmp_path / "state.db", "live-row-identity")
    # Both rounds use the provider id ``call_0``: the store keeps them apart, so must the events.
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="Looking now.", finish_reason="tool_calls", tool_calls=[_mock_tool_call(call_id="call_0")]),
        _mock_response(content="Second look.", finish_reason="tool_calls", tool_calls=[_mock_tool_call(call_id="call_0")]),
        _mock_response(content="done", finish_reason="stop"),
    ]
    events = []
    agent.interim_assistant_callback = lambda text, already_streamed=False: events.append(
        ("interim", text, interim_row_identity(agent)))
    agent.tool_start_callback = lambda call_id, name, args: events.append(
        ("tool.start", call_id, tool_call_row_identity(agent, call_id)))
    agent.tool_complete_callback = lambda call_id, name, args, result: events.append(
        ("tool.complete", call_id, tool_call_row_identity(agent, call_id)))
    try:
        with patch("model_tools.handle_function_call", return_value='{"ok": true}'):
            result = agent.run_conversation("look")
        rows = transcript_rows(db, "live-row-identity")
    finally:
        db.close()
    agent.result = result
    return agent, events, rows


def test_interim_and_tool_events_name_the_stored_row_they_are(tmp_path):
    agent, events, rows = _run_two_rounds(tmp_path)

    calls = [row for row in rows if row["role"] == "assistant" and row["tool_calls"]]
    assert [row["content"] for row in calls] == ["Looking now.", "Second look."]
    expected = []
    for row in calls:
        call_id = row["tool_calls"][0]["id"]
        stored = {"row_id": row["id"], "message_uid": row["message_uid"]}
        call_uid = json.loads(row["tool_call_uids"])[call_id]
        expected += [("interim", row["content"], stored),
                     ("tool.start", call_id, {**stored, "tool_call_uid": call_uid}),
                     ("tool.complete", call_id, {**stored, "tool_call_uid": call_uid})]
    assert events == expected
    # The repeated provider id is two occurrences with their own rows and uids.
    assert len({identity["row_id"] for _kind, _id, identity in events}) == 2
    assert len({identity.get("tool_call_uid") for kind, _id, identity in events if kind != "interim"}) == 2
    # Outside a round nothing names a row: a later callback cannot borrow a finished round's identity.
    assert getattr(agent, LIVE_ROW_ATTR) is None
    assert interim_row_identity(agent) == {} and tool_call_row_identity(agent, "call_0") == {}


def test_a_row_that_is_not_durable_names_nothing():
    from agent.live_row_identity import live_assistant_row

    agent = type("Agent", (), {})()
    live = {"role": "assistant", "content": "x", "_row_id": 7,
            "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "t", "arguments": "{}"}}]}
    with live_assistant_row(agent, live):
        assert interim_row_identity(agent) == {}  # no durable marker yet: an in-memory projection
    live["_db_persisted"] = True
    with live_assistant_row(agent, live):
        assert interim_row_identity(agent) == {"row_id": 7}
        assert tool_call_row_identity(agent, "call_1") == {}  # a call this row does not hold


def test_a_settled_turn_names_its_reply_row_for_hosts_that_settle_elsewhere(tmp_path):
    """The session authority and a managed worker carry the reply's row on the turn result."""
    agent, _events, rows = _run_two_rounds(tmp_path)
    reply = rows[-1]
    assert reply["role"] == "assistant" and reply["content"] == "done"

    fields = reply_row_result_fields(agent.result, [])
    assert fields == {"reply_row_id": reply["id"], "reply_message_uid": reply["message_uid"]}
    assert reply_row_payload(fields) == {"row_id": reply["id"], "message_uid": reply["message_uid"]}
    # A failed turn, or a delivered text the stored row does not hold, names no reply row.
    assert reply_row_result_fields({**agent.result, "failed": True}, []) == {}
    assert reply_row_result_fields({**agent.result, "final_response": "rewritten"}, []) == {}
    # The reply row of an earlier turn is never this turn's.
    assert reply_row_result_fields(agent.result, agent.result["messages"]) == {}
    assert reply_row_payload({"reply_row_id": True}) == {} and reply_row_payload({"reply_row_id": "9"}) == {}
    # An empty reply is no reply to name, even when the stored row is empty too.
    empty = {**reply, "_row_id": reply["id"], "_db_persisted": True, "content": ""}
    assert final_reply_row_identity([empty], [], "", "complete") == {}
