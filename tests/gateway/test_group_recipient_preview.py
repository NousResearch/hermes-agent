"""The advisory projection must use the real room owner and first-round resolver."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.hosted_rooms import create_room, local_authority_gateway_id
from gateway.session_controls import AuthorityConnection
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.mark.parametrize(('text', 'expected'), [
    ('hello', ['one', 'two']),
    ('@unknown', ['one', 'two']),
    ('@two @unknown', ['two']),
    ('@TWO @two @one', ['one', 'two']),
    ('@everyone @two', ['one', 'two']),
    ('alice@two', ['two']),
    ('`@two`', ['two']),
    ('@two.', ['one', 'two']),
])
def test_owner_gets_real_first_round_selection_without_writes(tmp_path, monkeypatch, text, expected):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('model:\n  default: fixture\n')
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), instance_id='owner', events={},
                                    epoch=begin_runtime_epoch(db, instance_id='test'))
        service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
        monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True, 'stopping': False})
        alice = AuthorityConnection(authority, object(), {'user_id': 'alice'})
        service.authorize_room(alice.actor.subject, 'room', create=True)
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=local_authority_gateway_id(),
                    members=[{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
                             {'member_id': 'two', 'profile': 'two', 'handle': 'two'}])
        with db._read_ctx() as conn:
            before = tuple(conn.iterdump())
        async def call():
            return await alice.dispatch({'id': 1, 'method': 'groups.preview',
                                         'params': {'room_id': 'room', 'text': text}})
        reply = asyncio.run(call())
        assert 'result' in reply, reply
        assert [m['member_id'] for m in reply['result']['members']] == expected
        with db._read_ctx() as conn:
            assert tuple(conn.iterdump()) == before


def test_secondary_profile_authority_boundaries_are_read_only(tmp_path, monkeypatch):
    root = tmp_path / 'home'
    home = root / 'profiles' / 'helper'
    home.mkdir(parents=True)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(root))
    (root / 'config.yaml').write_text('model:\n  default: fixture\n')
    (home / 'config.yaml').write_text('model:\n  default: fixture\n')
    with SessionDB(home / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(home), instance_id='owner', events={},
                                    epoch=begin_runtime_epoch(db, instance_id='test'))
        service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
        monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True, 'stopping': False})
        owner = AuthorityConnection(authority, object(), {'user_id': 'owner', 'provider': 'basic'})
        no_read = AuthorityConnection(authority, object(), {'user_id': 'owner', 'capabilities': []})
        wrong_actor = AuthorityConnection(authority, object(), {'user_id': 'owner', 'profile_id': 'foreign'})
        gateway_id = local_authority_gateway_id()
        members = [{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
                   {'member_id': 'two', 'profile': 'two', 'handle': 'two'}]
        for room_id, room_gateway, grant in [('owned', gateway_id, True), ('unowned', gateway_id, False),
                                              ('foreign', 'install:foreign', True), ('closed', gateway_id, True)]:
            if grant:
                service.authorize_room(owner.actor.subject, room_id, create=True)
            create_room(db.db_path, room_id=room_id, name=room_id, authority_gateway_id=room_gateway, members=members)
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            conn.execute("UPDATE hosted_rooms SET disbanded_at=1 WHERE room_id='closed'")
        with db._read_ctx() as conn:
            before = tuple(conn.iterdump())
        def call(connection=owner, **params):
            return asyncio.run(connection.dispatch({'id': 1, 'method': 'groups.preview',
                'params': {'profile': 'helper', 'room_id': 'owned', 'text': '@two', **params}}))
        success = call()
        assert [m['member_id'] for m in success['result']['members']] == ['two']
        assert call(profile='default')['error']['message'] == 'profile_mismatch'
        assert call(no_read)['error']['message'] == 'permission_denied'
        assert call(wrong_actor)['error']['message'] == 'profile_mismatch'
        for room_id in ('unowned', 'foreign', 'closed'):
            assert 'error' in call(room_id=room_id)
        monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True, 'stopping': True})
        assert call()['error']['message'] == 'runtime_coordination_required'
        monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True, 'stopping': False})
        monkeypatch.setattr(service, 'db_path', str(root / 'foreign.db'))
        assert call()['error']['message'] == 'profile_mismatch'
        monkeypatch.setattr(service, 'db_path', db.db_path)
        monkeypatch.setattr(service, 'authorize_room', None)
        assert call()['error']['message'] == 'permission_denied'
        monkeypatch.undo()
        with db._read_ctx() as conn:
            assert tuple(conn.iterdump()) == before


def test_preview_refuses_foreign_unowned_inactive_and_bad_input(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('model:\n  default: fixture\n')
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), instance_id='owner', events={},
                                    epoch=begin_runtime_epoch(db, instance_id='test'))
        service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
        monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True, 'stopping': False})
        alice = AuthorityConnection(authority, object(), {'user_id': 'alice'})
        bob = AuthorityConnection(authority, object(), {'user_id': 'bob'})
        service.authorize_room(alice.actor.subject, 'room', create=True)
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=local_authority_gateway_id(),
                    members=[{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
                             {'member_id': 'two', 'profile': 'two', 'handle': 'two'}])

        async def call(connection, **extra):
            return await connection.dispatch({'id': 1, 'method': 'groups.preview',
                'params': {'room_id': 'room', 'text': '@two', **extra}})

        with db._read_ctx() as conn:
            before = tuple(conn.iterdump())
        assert asyncio.run(call(bob))['error']['message'] == 'permission_denied'
        assert asyncio.run(call(alice, room_id='unowned'))['error']['message'] == 'permission_denied'
        assert asyncio.run(call(alice, profile='foreign'))['error']['message'] == 'profile_mismatch'
        assert asyncio.run(call(alice, text='漢' * 21846))['error']['message'] == 'invalid_params'
        assert asyncio.run(call(alice, unexpected=True))['error']['message'] == 'invalid_params'
        authority.hosted_room_service = None
        assert asyncio.run(call(alice))['error']['message'] == 'permission_denied'
        authority.hosted_room_service = service
        monkeypatch.setattr(service.runtime, 'status', lambda: {'running': False})
        assert asyncio.run(call(alice))['error']['message'] == 'runtime_coordination_required'
        with db._read_ctx() as conn:
            assert tuple(conn.iterdump()) == before
