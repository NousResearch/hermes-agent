"""A waiting turn must refresh cached tasks after durable lease admission.

Regression coverage extending PR #135722 with real isolated SessionDB handles.
No model request or provider initialization is required.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import agent.turn_facade_lease as lease_module
from agent.turn_facade_lease import admit_durable_turn_lease
from agent.turn_context import _hydrate_from_history
from hermes_state import SessionDB
from run_agent import AIAgent
from tools.todo_tool import TodoStore


ITEM = {"id": "task", "content": "finish task", "status": "pending"}
SUMMARY = [{"role": "user", "content": "compressed context"}]


def pair(call_id, items, revision):
    payload = {"todos": items}
    if revision is not None:
        payload["revision"] = revision
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "todo_list", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "tool_name": "todo_list",
            "content": json.dumps(payload),
        },
    ]


@pytest.fixture
def native_agent(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("validation-session", source="test")
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "validation-session"
    agent._session_db = db
    agent._todo_store = TodoStore()
    agent._persist_disabled = False
    agent._user_turn_count = 1  # Exclude unrelated usage-anchor recovery.
    agent.quiet_mode = True
    agent.log_prefix = ""
    yield agent
    # No agent constructor, live configuration, provider or live database is used.
    agent._session_db = None
    db.close()


def persist(agent, call_id, items, revision):
    messages = pair(call_id, items, revision)
    for message in messages:
        agent._session_db.append_message(agent.session_id, **message)
    content = agent._session_db.get_latest_todo_result(agent.session_id)
    assert json.loads(content)["todos"] == items
    return messages


@pytest.mark.parametrize(
    "latest_items",
    [[{**ITEM, "status": "completed"}], []],
    ids=["completed", "cleared"],
)
def test_waiting_turn_refreshes_populated_todos_after_admission(
    native_agent, monkeypatch, latest_items
):
    """A real failed SQLite lease attempt gates the intervening durable write."""
    monkeypatch.setattr(lease_module, "LEASE_WAIT_SECONDS", 10.0)
    old_history = persist(native_agent, "old", [ITEM], 3)
    native_agent._todo_store.restore([ITEM], revision=3)
    holder_db = native_agent._session_db
    holder = "validation-previous-holder"
    assert holder_db.acquire_session_turn_lease(
        native_agent.session_id, holder, wait_seconds=0
    )
    contended = threading.Event()
    native_agent._emit_status = lambda text: (
        contended.set() if "Another Hermes process" in text else None
    )
    native_agent._interrupt_requested = False
    with SessionDB(holder_db.db_path) as waiting_db:
        native_agent._session_db = waiting_db

        def admitted_read():
            admission = admit_durable_turn_lease(
                native_agent,
                session_id=native_agent.session_id,
                relay_turn_id="validation-next-turn",
                task_context={
                    "platform": "desktop",
                    "session_id": native_agent.session_id,
                },
                conversation_history=old_history,
            )
            assert admission.early_result is None
            assert admission.lease is not None
            try:
                # Establish that native admission really reloaded the new durable pair.
                newest = native_agent._latest_todo_response(
                    admission.conversation_history
                )
                assert newest == (latest_items, 8)
                _hydrate_from_history(native_agent, admission.conversation_history)
                assert native_agent._todo_store.snapshot() == {
                    "todos": latest_items,
                    "revision": 8,
                }
                native_agent._todo_store.write([{**ITEM, "content": "next task edit"}])
                assert native_agent._todo_store.snapshot()["revision"] == 9
            finally:
                admission.lease.release()

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(admitted_read)
                try:
                    assert contended.wait(timeout=5)
                    assert not future.done()
                    for message in pair("new", latest_items, 8):
                        holder_db.append_message(
                            native_agent.session_id, **message, turn_lease_holder=holder
                        )
                    holder_db.release_session_turn_lease(
                        native_agent.session_id, holder
                    )
                    future.result(timeout=12)
                finally:
                    native_agent._interrupt_requested = True
                    holder_db.release_session_turn_lease(
                        native_agent.session_id, holder
                    )
        finally:
            native_agent._session_db = holder_db


def test_native_owner_excludes_persistence_disabled_db(native_agent, monkeypatch):
    persist(native_agent, "current", [ITEM], 8)
    native_agent._persist_disabled = True
    native_agent._todo_store.restore([ITEM], revision=3)

    def forbidden_read(*args, **kwargs):
        pytest.fail("persistence-disabled caller read canonical database")

    monkeypatch.setattr(
        native_agent._session_db, "get_latest_todo_result", forbidden_read
    )
    _hydrate_from_history(native_agent, SUMMARY)
    assert native_agent._todo_store.snapshot() == {"todos": [ITEM], "revision": 3}
