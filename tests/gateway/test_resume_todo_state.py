"""``session.resume`` carries the session's todo snapshot (``todo_state``): a viewer attaching to a session
with a plan (the classic CLI's /resume or ``-c``) shows it before the next ``todo.updated``.

Real store, authority, local session and ``AuthorityConnection.resume`` (the helpers of
``test_resume_display_history``).
"""
import json
import sys

import pytest

from tests.gateway.test_resume_display_history import _local_session, _resume

TODOS = [{'id': '1', 'content': 'write the fix', 'status': 'completed'},
         {'id': '2', 'content': 'ship it', 'status': 'in_progress'}]


def _todo_turn(db, sid, call_id, todos, revision):
    call = {'id': call_id, 'type': 'function',
            'function': {'name': 'todo_list', 'arguments': json.dumps({'todos': todos})}}
    db.append_message(sid, 'assistant', '', tool_calls=[call])
    db.append_message(sid, 'tool', json.dumps({'todos': todos, 'revision': revision}), tool_name='todo_list',
                      tool_call_id=call_id)


@pytest.mark.asyncio
async def test_resume_carries_the_newest_todo_snapshot(tmp_path, monkeypatch):
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.db
    assert (await _resume(authority, actor, ref))['todo_state'] is None  # no plan yet
    db.append_message(physical, 'user', 'plan it')
    _todo_turn(db, physical, 'call-1', TODOS[:1], 1)
    _todo_turn(db, physical, 'call-2', TODOS, 2)
    db.append_message(physical, 'assistant', 'planned')
    assert (await _resume(authority, actor, ref))['todo_state'] == {'todos': TODOS, 'revision': 2}


def test_the_home_module_reads_the_snapshot_when_imported_directly():
    """``tool_progress`` bodies are also rebound onto ``tui_gateway.server``; the module's own copy (what the
    gateway imports) must not lean on the server's globals for ``json``/``logger``."""
    from tui_gateway.tool_progress import _todo_state_from_history
    call = {'id': 'call-1', 'type': 'function', 'function': {'name': 'todo_list', 'arguments': '{}'}}
    history = [{'role': 'assistant', 'content': '', 'tool_calls': [call]},
               {'role': 'tool', 'tool_call_id': 'call-1', 'content': json.dumps({'todos': TODOS, 'revision': 2})}]
    assert _todo_state_from_history(history) == {'todos': TODOS, 'revision': 2}


@pytest.mark.asyncio
async def test_resume_leaves_the_gateway_processs_stdout_alone(tmp_path, monkeypatch):
    """Importing ``tui_gateway.server`` (the stdio JSON-RPC sidecar) points ``sys.stdout`` at stderr, so the
    resume path must not import it: a CLI sharing the process would print to stderr from then on."""
    import tui_gateway
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    _todo_turn(authority.db, physical, 'call-1', TODOS, 2)
    monkeypatch.delitem(sys.modules, 'tui_gateway.server', raising=False)
    monkeypatch.delattr(tui_gateway, 'server', raising=False)
    monkeypatch.setattr(sys, 'stdout', sys.stdout)  # restored even if the import below swaps it
    stdout = sys.stdout
    assert (await _resume(authority, actor, ref))['todo_state'] == {'todos': TODOS, 'revision': 2}
    assert sys.stdout is stdout
    assert 'tui_gateway.server' not in sys.modules


def test_a_later_null_resume_clears_the_clients_todo_seed():
    """The client keeps the reply's snapshot until the next one: a later resume of a session with no plan
    (``todo_state: null``) clears it, so a fresh ``TodoStore`` is not seeded with a plan that is gone. A
    reply without the field (an older gateway) keeps it."""
    from hermes_cli.gateway_connection import SessionState
    state = SessionState("local-abc")
    state.adopt_snapshot({"todo_state": {"todos": TODOS, "revision": 2}})
    assert state.todo_state == {"todos": TODOS, "revision": 2}
    state.adopt_snapshot({})
    assert state.todo_state == {"todos": TODOS, "revision": 2}
    state.adopt_snapshot({"todo_state": None})
    assert state.todo_state is None
