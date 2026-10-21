"""Import parent links and chain-aware deletes never cross principals (review M4)."""
from types import SimpleNamespace

import pytest

from gateway.session_authority import SessionAuthority
from gateway.session_contract import Principal, SessionRef
from gateway.session_mutations import mutate_session
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch

_CAPS = frozenset({'session:read', 'session:control', 'session:create'})


@pytest.mark.asyncio
async def test_foreign_compressed_root_survives_import_and_delete_of_a_child(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='p', instance_id='owner',
                                     db=db, epoch=begin_runtime_epoch(db, instance_id='owner'))
        attacker, victim = (Principal(name, 'p', _CAPS, name) for name in ('attacker', 'victim'))

        async def mutate(actor, sid, request_id, operation, payload):
            row = db.get_session(sid) or {'runtime_revision': 0, 'runtime_generation': 0}
            params = {'session_id': sid, 'request_id': request_id, 'expected_revision': row['runtime_revision'],
                      'operation': operation, 'payload': payload}
            if operation == 'delete':
                params['expected_generation'] = row['runtime_generation']
            return await mutate_session(authority, actor, SessionRef('p', sid), params)

        # A messaging conversation another principal owns, compressed once.
        route = dict(source='telegram', session_key='agent:main:telegram:dm:7', chat_id='7', user_id='7')
        db.create_session('victim_root', **route)
        db.end_session('victim_root', 'compression')
        db.create_session('victim_tip', parent_session_id='victim_root', **route)
        # Imported history bound to the victim principal, also a compression chain.
        await mutate(victim, 'v_root', 'victim-import', 'import', {'sessions': [
            {'id': 'v_root', 'source': 'cli', 'end_reason': 'compression'},
            {'id': 'v_tip', 'source': 'cli', 'parent_session_id': 'v_root'}]})

        imported = await mutate(attacker, 'att', 'attack', 'import', {'sessions': [
            {'id': 'att', 'source': 'cli', 'parent_session_id': 'victim_root'},
            {'id': 'att2', 'source': 'cli', 'parent_session_id': 'v_root'}]})
        assert imported['detached'] == 2
        assert db.get_session('att')['parent_session_id'] is None
        assert db.get_session('att2')['parent_session_id'] is None
        assert (await mutate(attacker, 'att', 'delete', 'delete', {}))['deleted_ids'] == ['att']

        # Links forged before this guard (or by an offline ``hermes sessions import``) never carry a
        # delete into a chain the actor is not authorized for: the walk stops at the foreign row.
        db._execute_write(lambda c: c.execute("UPDATE sessions SET parent_session_id='v_root' WHERE id='att2'"))
        assert (await mutate(attacker, 'att2', 'delete2', 'delete', {}))['deleted_ids'] == ['att2']
        await mutate(attacker, 'att3', 'attack3', 'import', {'sessions': [{'id': 'att3', 'source': 'cli'}]})
        db._execute_write(lambda c: c.execute("UPDATE sessions SET parent_session_id='victim_root' WHERE id='att3'"))
        assert (await mutate(attacker, 'att3', 'delete3', 'delete', {}))['deleted_ids'] == ['att3']
        assert all(db.get_session(sid) for sid in ('victim_root', 'victim_tip', 'v_root', 'v_tip'))
