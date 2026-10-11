"""Mounting a finalized local session is a read; the first admitted turn reopens it (#85303)."""
from types import SimpleNamespace

import pytest

from gateway.session_contract import Submission
from gateway.session_controls import AuthorityConnection


async def _ended_local_session(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local import create_local_session

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    ref = create_local_session(authority, connection.actor, {'request_id': 'r', 'source': 'tui',
                                                             'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
    authority.db.append_message(ref.session_id, 'user', 'earlier turn')
    authority.db.end_session(ref.session_id, 'tui_shutdown')
    return authority, connection, ref


@pytest.mark.asyncio
async def test_resuming_a_finalized_local_session_leaves_it_ended(tmp_path, monkeypatch):
    authority, connection, ref = await _ended_local_session(tmp_path, monkeypatch)
    try:
        reply = await connection.dispatch({'id': 1, 'method': 'session.resume',
                                           'params': {'session_id': ref.session_id}})
        assert reply['result']['messages'][0]['content'] == 'earlier turn', reply
        row = authority.db.get_session(ref.session_id)
        assert (row['end_reason'], row['ended_at'] is not None) == ('tui_shutdown', True), \
            'a mount with no new activity must not re-light a finalized row'
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_first_admitted_turn_reopens_a_finalized_local_session_before_execution(tmp_path, monkeypatch):
    authority, connection, ref = await _ended_local_session(tmp_path, monkeypatch)
    seen = []

    async def execute(authority, ref, row):
        current = authority.db.get_session(ref.session_id)
        seen.append((current['ended_at'], current['end_reason']))
        return 'ok'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        await authority.submit(connection.actor, Submission('turn', ref, {'text': 'continue'}, 'queue'))
        await authority.sessions[ref.session_id].task
        # SessionStore routes a stamped row as stale and answers with a FRESH session, so the
        # turn must find the row live before the runner ever routes it.
        assert seen == [(None, None)]
    finally:
        await connection.close()
        authority.db.close()
