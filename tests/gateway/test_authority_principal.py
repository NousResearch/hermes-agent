"""Server-stamped grants retain their scope through connection admission."""
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.mark.asyncio
@pytest.mark.parametrize('binding', ['read-only', 'wrong-profile', 'stale-instance'])
async def test_connection_preserves_server_grant(tmp_path, binding):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned',
                                     instance_id='current', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(None, 'route')
        identity = {'user_id': 'human', 'profile_id': 'owned', 'instance_id': 'current',
                    'capabilities': ['session:read']}
        if binding == 'wrong-profile':
            identity['profile_id'] = 'other'
        elif binding == 'stale-instance':
            identity['instance_id'] = 'previous'
        connection = AuthorityConnection(authority, object(), identity)
        try:
            resumed = await connection.dispatch({'id': 1, 'method': 'session.resume',
                                                 'params': {'session_id': 's'}})
            assert ('result' in resumed) == (binding == 'read-only'), resumed
            submitted = await connection.dispatch({'id': 2, 'method': 'prompt.submit',
                                                    'params': {'session_id': 's', 'submission_id': 'denied'}})
            assert submitted['error']['message'] in {'permission_denied', 'profile_mismatch'}, submitted
        finally:
            await connection.close()


@pytest.mark.asyncio
async def test_question_response_does_not_grant_execution_control(tmp_path):
    from gateway.session_contract import Principal, SessionRef
    from hermes_state_runtime import RuntimeStoreError, admit_session_input, claim_session_input
    from tools import clarify_gateway

    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned',
                                     instance_id='current', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(None, 'route')
        admit_session_input(db, epoch=epoch, principal_id='human', session_id='s',
                            request_id='input', payload={'text': 'question'})
        row = claim_session_input(db, epoch=epoch, session_id='s')
        generation = row['generation']
        ref = SessionRef('owned', 's')
        actor = Principal('human', 'owned', frozenset({'session:read', 'session:respond'}), 'viewer')
        await authority.attach(actor, ref)
        entry = clarify_gateway.register('scoped-question', 'scoped-route', 'Choose', ['a', 'b'])
        try:
            authority.register_clarify('s', generation, entry)
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                await authority.interrupt(actor, ref, generation)
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                await authority.respond(actor, ref, generation, entry.clarify_id, {'choice': 'once'})
            await authority.respond(actor, ref, generation, entry.clarify_id, {'answer': 'a'}, kind='clarify')
            assert clarify_gateway.wait_for_response(entry.clarify_id, timeout=.1) == 'a'
        finally:
            clarify_gateway.clear_session(entry.session_key)
