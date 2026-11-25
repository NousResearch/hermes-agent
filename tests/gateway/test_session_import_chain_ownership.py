"""Import parent links and chain-aware deletes never cross principals (review M4)."""
from types import SimpleNamespace

import pytest

from gateway.session_authority import SessionAuthority
from gateway.session_contract import SessionActor, SessionRef
from gateway.session_mutations import mutate_session
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

_CAPS = frozenset({'session:read', 'session:control', 'session:create'})


@pytest.mark.asyncio
async def test_foreign_compressed_root_survives_import_and_delete_of_a_child(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as db:
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='p', instance_id='owner',
                                     epoch=db.runtime.begin_epoch(instance_id='owner'), persistence=storage_handle(tmp_path))
        attacker, victim = (SessionActor(name, 'p', _CAPS, name) for name in ('attacker', 'victim'))

        async def mutate(actor, sid, request_id, operation, payload):
            row = session_row(db, sid) or {'runtime_revision': 0, 'runtime_generation': 0}
            params = {'session_id': sid, 'request_id': request_id, 'expected_revision': row['runtime_revision'],
                      'operation': operation, 'payload': payload}
            if operation == 'delete':
                params['expected_generation'] = row['runtime_generation']
            return await mutate_session(authority, actor, SessionRef('p', sid), params)

        # A messaging conversation another principal owns, compressed once.
        route = dict(source='telegram', session_key='agent:main:telegram:dm:7', chat_id='7', user_id='7')
        seed_session(db, 'victim_root', **route)
        db.sessions.end('victim_root', 'compression')
        seed_session(db, 'victim_tip', parent_session_id='victim_root', **route)
        # Imported history bound to the victim principal, also a compression chain.
        await mutate(victim, 'v_root', 'victim-import', 'import', {'sessions': [
            {'id': 'v_root', 'source': 'cli', 'end_reason': 'compression'},
            {'id': 'v_tip', 'source': 'cli', 'parent_session_id': 'v_root'}]})

        imported = await mutate(attacker, 'att', 'attack', 'import', {'sessions': [
            {'id': 'att', 'source': 'cli', 'parent_session_id': 'victim_root'},
            {'id': 'att2', 'source': 'cli', 'parent_session_id': 'v_root'}]})
        assert imported['detached'] == 2
        assert session_row(db, 'att')['parent_session_id'] is None
        assert session_row(db, 'att2')['parent_session_id'] is None
        assert (await mutate(attacker, 'att', 'delete', 'delete', {}))['deleted_ids'] == ['att']

        # Links forged before this guard (or by an offline ``hermes sessions import``) never carry a
        # delete into a chain the actor is not authorized for: the walk stops at the foreign row.
        raw_holder(db)._execute_write(lambda c: c.execute("UPDATE sessions SET parent_session_id='v_root' WHERE id='att2'"))
        assert (await mutate(attacker, 'att2', 'delete2', 'delete', {}))['deleted_ids'] == ['att2']
        await mutate(attacker, 'att3', 'attack3', 'import', {'sessions': [{'id': 'att3', 'source': 'cli'}]})
        raw_holder(db)._execute_write(lambda c: c.execute("UPDATE sessions SET parent_session_id='victim_root' WHERE id='att3'"))
        assert (await mutate(attacker, 'att3', 'delete3', 'delete', {}))['deleted_ids'] == ['att3']
        assert all(db.sessions.get(sid) for sid in ('victim_root', 'victim_tip', 'v_root', 'v_tip'))
