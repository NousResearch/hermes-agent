"""The bounded Todo read preserves pairing, lineage and explicit-clear semantics."""

import json


from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.core.identity import SessionId
from tests.contracts.conversation.test_messages import Script, SESSION, CHILD, OTHER

MAX = 10_000


def todo(script, sid, call_id, state, index):
    return script.msg(sid, "assistant", "", index, tool_calls=({
        "id": call_id, "type": "function", "function": {"name": "todo_list", "arguments": "{}"},
    },)).msg(sid, "tool", json.dumps(state), index + 1, tool_call_id=call_id, tool_name="todo_list")


def test_latest_todo_result_preserves_lineage_and_pairing(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    state = {"todos": [{"id": "a", "content": "plan", "status": "pending"}], "revision": 1}
    empty = {"todos": [], "revision": 2}
    script = todo(Script().session(SESSION), SESSION, "plan", state, 1)
    script.end(SESSION, "compression").session(CHILD, parent=SESSION)
    script.apply(backend, scope_a)
    assert json.loads(store_a.messages.latest_todo_result(CHILD, max_chars=MAX)) == state
    assert store_b.messages.latest_todo_result(CHILD, max_chars=MAX) is None
    assert store_a.messages.latest_todo_result(SessionId("absent"), max_chars=MAX) is None

    todo(Script(), CHILD, "clear", empty, 3).apply(backend, scope_a)
    assert json.loads(store_a.messages.latest_todo_result(CHILD, max_chars=MAX)) == empty
    (Script().msg(CHILD, "user", "unrelated", 5)
           .msg(CHILD, "tool", json.dumps(state), 6, tool_call_id="plan", tool_name="todo_list")).apply(backend, scope_a)
    assert json.loads(store_a.messages.latest_todo_result(CHILD, max_chars=MAX)) == empty

    Script().session(OTHER, parent=SESSION, branch=True).apply(backend, scope_a)
    assert store_a.messages.latest_todo_result(OTHER, max_chars=MAX) is None
    todo(Script(), OTHER, "branch-plan", state, 7).apply(backend, scope_a)
    todo(Script(), OTHER, "invalid", {"todos": [], "revision": "invalid"}, 9).apply(backend, scope_a)
    assert json.loads(store_a.messages.latest_todo_result(OTHER, max_chars=MAX)) == state
    # A result longer than max_chars is skipped like an invalid one.
    assert json.loads(store_a.messages.latest_todo_result(OTHER, max_chars=len(json.dumps(state)))) == state
    assert store_a.messages.latest_todo_result(OTHER, max_chars=len(json.dumps(state)) - 1) is None


def test_a_plan_whose_newer_copy_was_removed_is_not_restored_from_an_older_copy(backend, scope_a):
    """A compression child copies the parent's Todo turn (same uids); rewinding the child's copy hides the plan there, and the parent's archived copy does not bring it back. The parent itself still reads its own copy."""
    store = backend.open(scope_a)
    state = json.dumps({"todos": [{"id": "1", "content": "ship it", "status": "pending"}], "revision": 1})
    call = {"id": "call-todo", "type": "function", "function": {"name": "todo", "arguments": "{}"}}
    turn = [
        NewMessage("user", "plan it", timestamp=100.0, message_uid="todo-uid-user"),
        NewMessage("assistant", "", timestamp=101.0, tool_calls=(call,), message_uid="todo-uid-assistant"),
        NewMessage("tool", state, timestamp=102.0, tool_name="todo", tool_call_id="call-todo", message_uid="todo-uid-tool"),
    ]
    store.sessions.create(NewSession("todo-parent", "cli"))
    store.messages.append_batch("todo-parent", turn)
    store.sessions.end("todo-parent", "compression")
    store.sessions.create(NewSession("todo-child", "cli", parent_session_id="todo-parent"))
    copied = store.messages.append_batch("todo-child", turn)
    assert store.messages.latest_todo_result("todo-child", max_chars=MAX) == state

    for outcome in copied.outcomes:
        assert store.messages.deactivate_message("todo-child", outcome.id) is True

    assert store.messages.latest_todo_result("todo-child", max_chars=MAX) is None
    assert store.messages.latest_todo_result("todo-parent", max_chars=MAX) == state
