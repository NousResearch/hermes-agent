"""Server-stamped grants retain their scope through connection admission."""
from types import SimpleNamespace

import pytest
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session


@pytest.mark.asyncio
@pytest.mark.parametrize('binding', ['read-only', 'wrong-profile', 'stale-instance'])
async def test_connection_preserves_server_grant(tmp_path, binding):
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'test')
        epoch = db.runtime.begin_epoch(instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned',
                                     instance_id='current', epoch=epoch, persistence=storage_handle(tmp_path))
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
    from gateway.session_contract import SessionActor, SessionRef
    from persistence.core.errors import AuthorityRefusal
    from tools import clarify_gateway

    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'test')
        epoch = db.runtime.begin_epoch(instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned',
                                     instance_id='current', epoch=epoch, persistence=storage_handle(tmp_path))
        authority.sessions['s'] = LiveSession(None, 'route')
        db.admissions.admit(epoch=epoch, principal_id='human', session_id='s',
                            request_id='input', payload={'text': 'question'})
        row = db.admissions.claim(epoch=epoch, session_id='s')
        generation = row.generation
        ref = SessionRef('owned', 's')
        actor = SessionActor('human', 'owned', frozenset({'session:read', 'session:respond'}), 'viewer')
        await authority.attach(actor, ref)
        entry = clarify_gateway.register('scoped-question', 'scoped-route', 'Choose', ['a', 'b'])
        try:
            authority.register_clarify('s', generation, entry)
            with pytest.raises(AuthorityRefusal, match='permission_denied'):
                await authority.interrupt(actor, ref, generation)
            with pytest.raises(AuthorityRefusal, match='permission_denied'):
                await authority.respond(actor, ref, generation, entry.clarify_id, {'choice': 'once'})
            await authority.respond(actor, ref, generation, entry.clarify_id, {'answer': 'a'}, kind='clarify')
            assert clarify_gateway.wait_for_response(entry.clarify_id, timeout=.1) == 'a'
        finally:
            clarify_gateway.clear_session(entry.session_key)
