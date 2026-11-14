"""A resumed canonical session renders through main's display projection (TUI parity, PR #106742).

Main's TUI sidecar answered ``session.resume`` with ``tui_gateway.session_history._history_to_messages``:
compaction handoffs stripped, ``/steer`` rows unwrapped to the user's words, tool rows labelled from the
paired assistant ``tool_calls``. The canonical snapshot carries raw stored rows (the controls need their
``content``/``row_id``), so a client that renders asks for ``display_projection`` and receives the same
projection main produced, alongside the raw rows.
"""
import json
from types import SimpleNamespace

import pytest


async def _native(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': str(tmp_path), 'instance_id': 'owner'})
    return store._db, native


def _seed(db):
    from agent.context_compressor import SUMMARY_PREFIX
    from agent.prompt_builder import steer_user_row
    db.create_session('old-tui', source='tui', model='fake-model')
    db.append_message('old-tui', 'user', SUMMARY_PREFIX + '\nUser Messages (verbatim, newest first)\nEND OF CONTEXT SUMMARY')
    db.append_message('old-tui', 'user', 'run the tool')
    db.append_message('old-tui', 'assistant', '', tool_calls=[{
        'id': 'call_1', 'type': 'function',
        'function': {'name': 'read_file', 'arguments': json.dumps({'path': 'notes.txt'})}}])
    db.append_message('old-tui', 'tool', 'file body', tool_name='read_file', tool_call_id='call_1')
    steer = steer_user_row('focus on the numbers')
    db.append_message('old-tui', 'user', steer['content'], display_kind=steer['display_kind'])
    db.append_message('old-tui', 'assistant', 'done')
    db.end_session('old-tui', 'tui_shutdown')


@pytest.mark.asyncio
async def test_resume_display_projection_matches_main(tmp_path, monkeypatch):
    db, native = await _native(tmp_path, monkeypatch)
    _seed(db)
    reply = await native.dispatch({'id': 1, 'method': 'session.resume',
                                   'params': {'session_id': 'old-tui', 'display_projection': True}})
    assert 'result' in reply, reply
    result = reply['result']
    shown = result['display_messages']
    # Main's projection, byte-for-byte, over the same stored rows.
    from tui_gateway import server
    raw = db.get_messages_as_conversation('old-tui', include_row_ids=True)
    assert shown == server._history_to_messages(raw, profile_home=str(tmp_path))
    texts = json.dumps(shown)
    assert 'CONTEXT COMPACTION' not in texts and 'END OF CONTEXT SUMMARY' not in texts
    assert 'OUT-OF-BAND' not in texts
    steer = [m for m in shown if m.get('display_kind') == 'steer']
    assert [m['text'].strip() for m in steer] == ['focus on the numbers']
    tool = next(m for m in shown if m['role'] == 'tool')
    assert (tool['name'], tool['context']) == ('read_file', 'notes.txt')
    # The raw rows stay for rewind targeting (content + row_id).
    assert all('row_id' in m for m in result['messages'])
    assert result['message_count'] == len(result['messages'])


@pytest.mark.asyncio
async def test_resume_without_projection_is_unchanged(tmp_path, monkeypatch):
    db, native = await _native(tmp_path, monkeypatch)
    _seed(db)
    reply = await native.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 'old-tui'}})
    assert 'result' in reply, reply
    assert 'display_messages' not in reply['result']
