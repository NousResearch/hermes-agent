"""``session.resume`` carries the session's todo snapshot (``todo_state``): a viewer attaching to a session
with a plan (the classic CLI's /resume or ``-c``) shows it before the next ``todo.updated``.

Real store, authority, local session and ``AuthorityConnection.resume`` (the helpers of
``test_resume_display_history``).
"""
import json
import sys
from types import SimpleNamespace

import pytest

from tests.gateway.test_resume_display_history import _local_session, _resume
from tests.fixtures.storage_seed import seed_messages

pytestmark = pytest.mark.usefixtures('authority_cwd_in_tmp')  # symbolic profile_id: a relative home

TODOS = [{'id': '1', 'content': 'write the fix', 'status': 'completed'},
         {'id': '2', 'content': 'ship it', 'status': 'in_progress'}]


def _todo_turn(db, sid, call_id, todos, revision):
    call = {'id': call_id, 'type': 'function',
            'function': {'name': 'todo_list', 'arguments': json.dumps({'todos': todos})}}
    seed_messages(db, sid, {'role': 'assistant', 'content': '', 'tool_calls': [call]},
                  {'role': 'tool', 'content': json.dumps({'todos': todos, 'revision': revision}),
                   'tool_name': 'todo_list', 'tool_call_id': call_id})


@pytest.mark.asyncio
async def test_resume_carries_the_newest_todo_snapshot(tmp_path, monkeypatch):
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.persistence
    assert (await _resume(authority, actor, ref))['todo_state'] is None  # no plan yet
    seed_messages(db, physical, {'role': 'user', 'content': 'plan it'})
    _todo_turn(db, physical, 'call-1', TODOS[:1], 1)
    _todo_turn(db, physical, 'call-2', TODOS, 2)
    seed_messages(db, physical, {'role': 'assistant', 'content': 'planned'})
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
    _todo_turn(authority.persistence, physical, 'call-1', TODOS, 2)
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


async def _running_turn(tmp_path, monkeypatch, platform):
    """An earlier turn left an unfinished plan; a later turn on a *platform* source is claimed and its
    TurnRunner bound to the authority, as the drain does before running it."""
    from gateway.config import Platform
    from gateway.session_authority import admission_row
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.persistence
    seed_messages(db, physical, {'role': 'user', 'content': 'plan it'})
    _todo_turn(db, physical, 'call-1', TODOS, 2)
    seed_messages(db, physical, {'role': 'assistant', 'content': 'stopped halfway'})
    db.admissions.admit(epoch=authority.epoch, principal_id='owner', session_id=ref.session_id,
                        request_id='next-turn', payload={'text': 'carry on'})
    row = admission_row(db.admissions.claim(epoch=authority.epoch, session_id=ref.session_id))
    authority.executing[ref.session_id] = row['generation']  # what the claim stamps before running
    assert _bind_turn(authority, ref, getattr(Platform, platform))._approval_owner == (
        authority, ref.session_id, row['generation'])
    return authority, actor, ref, physical, row


def _bind_turn(authority, ref, platform):
    """A TurnRunner for *ref*'s running execution, bound to the authority as the turn path builds it."""
    from gateway.run_turn_runner import TurnRunner
    source = SimpleNamespace(platform=platform, chat_id=ref.session_id)
    return TurnRunner(SimpleNamespace(session_authority=authority), SimpleNamespace(source=source, session_id=ref.session_id))


@pytest.mark.asyncio
async def test_a_local_turns_resume_says_whether_that_turn_wrote_the_plan(tmp_path, monkeypatch):
    """#158324: an earlier turn ended on an unfinished plan, which its end took off the client's live panel. A
    viewer attaching while a later local turn runs is told that turn did not write it, and told so once it does."""
    authority, actor, ref, physical, row = await _running_turn(tmp_path, monkeypatch, 'LOCAL')
    resumed = await _resume(authority, actor, ref)
    assert resumed['running'] is True
    assert resumed['todo_state'] == {'todos': TODOS, 'revision': 2, 'from_running_turn': False}

    written = [{**TODOS[1], 'status': 'completed'}]
    authority.publish_execution(ref.session_id, row['generation'], 'todo.updated', {'todos': written, 'revision': 3})
    _todo_turn(authority.persistence, physical, 'call-2', written, 3)
    resumed = await _resume(authority, actor, ref)
    assert resumed['todo_state'] == {'todos': written, 'revision': 3, 'from_running_turn': True}


@pytest.mark.asyncio
async def test_a_turn_that_publishes_no_todo_updates_leaves_the_snapshot_unmarked(tmp_path, monkeypatch):
    """A messaging turn's todo writes never reach the stream as ``todo.updated``: its resume cannot tell whose
    plan it is, so it says nothing and the client keeps restoring the plan while the turn runs."""
    authority, actor, ref, _physical, _row = await _running_turn(tmp_path, monkeypatch, 'TELEGRAM')
    resumed = await _resume(authority, actor, ref)
    assert resumed['running'] is True
    assert resumed['todo_state'] == {'todos': TODOS, 'revision': 2}


@pytest.mark.asyncio
async def test_a_queued_follow_up_keeps_the_plan_its_execution_already_wrote(tmp_path, monkeypatch):
    """A queued follow-up runs a second TurnRunner inside the same execution generation (``_run_agent_queued_followup``,
    one completion for both): the plan the first part wrote, still shown live, is that execution's own."""
    from gateway.config import Platform
    authority, actor, ref, physical, row = await _running_turn(tmp_path, monkeypatch, 'LOCAL')
    written = [{**TODOS[1], 'status': 'completed'}]
    authority.publish_execution(ref.session_id, row['generation'], 'todo.updated', {'todos': written, 'revision': 3})
    _todo_turn(authority.persistence, physical, 'call-2', written, 3)
    _bind_turn(authority, ref, Platform.LOCAL)
    resumed = await _resume(authority, actor, ref)
    assert resumed['todo_state'] == {'todos': written, 'revision': 3, 'from_running_turn': True}
