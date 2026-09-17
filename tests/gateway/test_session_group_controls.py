"""Canonical group management is profile-bound and never starts inference."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket


def test_group_management_on_ordinary_authenticated_socket(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    (home / 'profiles' / 'helper').mkdir(parents=True)
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'hosted_rooms': {'profiles': {'helper': str(home / 'profiles' / 'helper')}},
        'model': {'provider': 'custom', 'default': 'no-inference',
                  'base_url': 'http://127.0.0.1:1/v1'},
        'auxiliary': {'title_generation': {'enabled': False}},
    }))
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
                             PYTHONPATH=str(root), PYTHONUNBUFFERED='1')

    async def probe(desc):
        with pytest.raises(InvalidStatus):
            async with connect(desc['api_origin'].replace('http:', 'ws:') + '/api/ws'):
                pass
        async with websocket(home, desc) as ws:
            capabilities = await rpc(ws, 'groups.capabilities')
            assert 'result' in capabilities, capabilities
            assert 'groups.create' in capabilities['result']['methods']
            members = [{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
                       {'member_id': 'two', 'profile': 'helper', 'handle': 'two'}]
            created = await rpc(ws, 'groups.create', room_id='room-fixture', name='First', members=members)
            assert 'result' in created, created
            room = created['result']['room']
            retry = (await rpc(ws, 'groups.create', room_id='room-fixture', name='First', members=members))['result']['room']
            assert retry['room_id'] == room['room_id'] and retry['idempotent']
            renamed = await rpc(ws, 'groups.rename', room_id='room-fixture', event_id='rename-fixture', name='Renamed')
            assert renamed['result']['room']['name'] == 'Renamed', renamed
            listed = await rpc(ws, 'groups.list')
            assert [r['name'] for r in listed['result']['rooms']] == ['Renamed'], listed
            state = await rpc(ws, 'groups.state', room_id='room-fixture')
            log = await rpc(ws, 'groups.log', room_id='room-fixture')
            assert state['result']['room']['latest_seq'] == log['result']['cursor']
            denied = await rpc(ws, 'groups.rename', profile='foreign', room_id='room-fixture', event_id='forbidden', name='Wrong')
            assert denied['error']['message'] == 'profile_mismatch', denied
            profiles = await rpc(ws, 'profiles.list', include_sessions=False)
            assert [p['path'] for p in profiles['result']['profiles']] == [str(home), str(home / 'profiles' / 'helper')], profiles
            disbanded = await rpc(ws, 'groups.disband', room_id='room-fixture')
            assert 'result' in disbanded, disbanded
            assert (await rpc(ws, 'groups.list'))['result']['rooms'] == []
            assert (await rpc(ws, 'session.list'))['result']['sessions'] == []
            print(json.dumps({'group_management': 'passed', 'profiles': profiles['result'], 'capabilities': capabilities['result']}))

    with daemon(root, home, env, barrier=False) as (_, desc):
        asyncio.run(probe(desc))


def test_profile_discovery_and_group_controls_enforce_actor_scope(tmp_path, monkeypatch):
    home = tmp_path / 'state'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HOME', str(tmp_path / 'user'))
    (home / 'config.yaml').write_text('model:\n  default: fixture-model\n  provider: custom\n')
    (home / 'profile.yaml').write_text('display_name: Fixture\nui_meta:\n  hermes-bots:\n    color: blue\n')
    with SessionDB(home / 'state.db') as db:
        db.create_session('private', source='cli', user_id='another')
        db.set_session_title('private', 'Bot Chat')
        authority = SimpleNamespace(profile_id=str(home), instance_id='owner', db=db, events={}, sessions={})

        async def probe():
            async def call(method, params=None, **identity):
                connection = AuthorityConnection(authority, object(), {'user_id': 'reader', **identity})
                return await connection.dispatch({'id': 'request', 'method': method, 'params': params or {}})
            for method in ('profiles.list', 'groups.list', 'groups.create'):
                for identity in ({'user_id': ''}, {'capabilities': []}, {'profile_id': 'foreign'}):
                    result = await call(method, **identity)
                    assert result['error']['message'] in {'permission_denied', 'profile_mismatch'}, result
            denied = await call('groups.create', capabilities=['session:read'])
            assert denied['error']['message'] == 'permission_denied', denied
            result = await call('profiles.list', capabilities=['session:read'])
            assert 'result' in result, result
            row, = result['result']['profiles']
            assert row['model'] == 'fixture-model'
            assert row['display_name'] == 'Fixture'
            assert row['ui_meta']['hermes-bots']['color'] == 'blue'
            assert row['canonical_session'] is None
            assert row['last_session'] is None
            assert db.get_session('private')['title'] == 'Bot Chat'
            db.set_session_title('private', 'Other Bot Chat')
            from gateway.session_identity import authenticated_subject
            db.create_session('mine', source='gui',
                              user_id=authenticated_subject({'user_id': 'reader'}), chat_id='local-mine')
            db.set_session_title('mine', 'Bot Chat')
            db.set_session_hidden('mine', True)
            db.append_message('mine', 'user', 'owned preview')
            own = (await call('profiles.list', capabilities=['session:read']))['result']['profiles'][0]
            assert own['canonical_session']['id'] == 'mine'
            assert own['canonical_session']['preview'] == 'owned preview'
            for method in ('profiles.list', 'groups.list'):
                denied = await call(method, {'profile': 'foreign'})
                assert denied['error']['message'] == 'profile_mismatch', denied
            assert (await call('profiles.delete'))['error']['message'] == 'invalid_params'
        asyncio.run(probe())
