"""A Todo result whose newest copy a rewind removed in a compression child is not restored from the parent's copy."""

import json
import sqlite3

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from tests.persistence_sqlite.conftest import store_for

TODO = json.dumps({"todos": [{"id": "1", "content": "ship it", "status": "pending"}], "revision": 1})
CALL = {"id": "call-todo", "type": "function", "function": {"name": "todo", "arguments": "{}"}}


def _turn():
    return [
        NewMessage("user", "plan it", timestamp=100.0, message_uid="uid-user"),
        NewMessage("assistant", "", timestamp=101.0, tool_calls=(CALL,), message_uid="uid-assistant"),
        NewMessage("tool", TODO, timestamp=102.0, tool_name="todo", tool_call_id="call-todo", message_uid="uid-tool"),
    ]


def test_a_todo_plan_undone_in_a_compression_child_stays_gone(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("parent", "cli"))
    store.messages.append_batch("parent", _turn())
    store.sessions.create(NewSession("child", "cli", parent_session_id="parent"))
    store.messages.append_batch("child", _turn())
    with sqlite3.connect(store_for(scope_a)) as connection:
        connection.execute("UPDATE sessions SET end_reason = 'compression', "
                           "ended_at = (SELECT started_at FROM sessions WHERE id = 'child') WHERE id = 'parent'")
    assert store.messages.latest_todo_result("child", max_chars=10_000) == TODO

    with sqlite3.connect(store_for(scope_a)) as connection:
        connection.execute("UPDATE messages SET active = 0, compacted = 0 WHERE session_id = 'child'")

    assert store.messages.latest_todo_result("child", max_chars=10_000) is None
    assert store.messages.latest_todo_result("parent", max_chars=10_000) == TODO
