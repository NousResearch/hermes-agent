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
            assert (await call('profiles.delete'))['error']['code'] == -32601  # unknown method, not a bad argument
        asyncio.run(probe())


def test_groups_list_pages_the_actor_visible_rooms(tmp_path):
    """keeltrace: pagination is applied to the rooms the actor may see, not to the room table
    before the authorization filter (which returned empty/short pages ahead of visible rooms)."""
    from gateway import hosted_rooms as rooms
    from gateway.session_group_controls import _group
    from hermes_state_runtime import RuntimeStoreError
    db_path = tmp_path / 'state.db'
    with SessionDB(db_path) as db:
        for index, name in enumerate(['f1', 'f2', 'm1', 'f3', 'm2', 'm3']):
            rooms.create_room(db_path, room_id=name, name=name, members=[], authority_gateway_id='gw',
                              now=1000.0 - index)  # list order = updated_at DESC = this order

        def authorize_room(subject, room_id):
            if not room_id.startswith('m'):
                raise RuntimeStoreError('permission_denied')
        service = SimpleNamespace(db_path=db_path, authorize_room=authorize_room,
                                  runtime=SimpleNamespace(status=lambda: {'running': True}))
        authority = SimpleNamespace(db=db, hosted_room_service=service)
        actor = SimpleNamespace(subject='me')
        pages, params = [], {'limit': 2}
        while True:
            page = _group(authority, actor, 'groups.list', params)
            pages.append([room['room_id'] for room in page['rooms']])
            if page['next_offset'] is None:
                break
            params = {'limit': 2, 'offset': page['next_offset']}
        assert pages == [['m1', 'm2'], ['m3']], pages


def test_profiles_list_names_every_profile_on_disk_and_its_install(tmp_path, monkeypatch):
    """RT-38 (#140417): with one gateway per profile (no multiplexing, no hosted_rooms entries) the
    authority's roster must still name every profile under this root, and carry ``install_id`` as
    the pre-authority ``profiles.list`` did. Desktop's roster merge drops every union agent the
    active source did not return, so a default-only answer showed one bot after ``hermes update``."""
    home = tmp_path / 'state'
    for name in ('helper', 'writer'):
        (home / 'profiles' / name).mkdir(parents=True)
        (home / 'profiles' / name / 'config.yaml').write_text('model:\n  default: m\n')
    (home / 'profiles' / 'writer' / 'profile.yaml').write_text('display_name: Writer Bot\n')
    (home / 'profiles' / 'ghost').mkdir()  # no identity marker: never listed
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HOME', str(tmp_path / 'user'))
    (home / 'config.yaml').write_text('model:\n  default: fixture-model\n')
    with SessionDB(home / 'state.db') as db:
        db.create_session('theirs', source='cli', user_id='another')
        writer_db = SessionDB(home / 'profiles' / 'writer' / 'state.db')
        writer_db.create_session('writer-chat', source='gui', user_id='another')
        writer_db.set_session_title('writer-chat', 'Bot Chat')
        writer_db.close()
        authority = SimpleNamespace(profile_id=str(home), instance_id='owner', db=db, events={}, sessions={})

        async def probe():
            connection = AuthorityConnection(authority, object(), {'user_id': 'reader'})
            return await connection.dispatch({'id': 'r', 'method': 'profiles.list', 'params': {}})
        result = asyncio.run(probe())['result']
    from hermes_cli.install_identity import get_install_id
    assert result['install_id'] and result['install_id'] == get_install_id()
    rows = {row['name']: row for row in result['profiles']}
    assert list(rows) == ['default', 'helper', 'writer'], rows
    assert rows['writer']['display_name'] == 'Writer Bot'
    # Discovery is metadata only: a profile this socket does not route to adds no session fields.
    assert 'canonical_session' not in rows['writer'] and 'last_session' not in rows['writer']


def _roster_home(tmp_path, monkeypatch):
    home = tmp_path / 'state'
    for name in ('helper', 'writer'):
        (home / 'profiles' / name).mkdir(parents=True)
        (home / 'profiles' / name / 'config.yaml').write_text('model:\n  default: m\n')
    (home / 'profiles' / 'writer' / 'profile.yaml').write_text('display_name: Writer Bot\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HOME', str(tmp_path / 'user'))
    (home / 'config.yaml').write_text('model:\n  default: fixture-model\n')
    return home


def _list_profiles(home):
    with SessionDB(home / 'state.db') as db:
        authority = SimpleNamespace(profile_id=str(home), instance_id='owner', db=db, events={}, sessions={})

        async def probe():
            connection = AuthorityConnection(authority, object(), {'user_id': 'reader'})
            return await connection.dispatch({'id': 'r', 'method': 'profiles.list', 'params': {}})
        return asyncio.run(probe())


def test_profiles_list_survives_an_undecodable_sibling_profile_yaml(tmp_path, monkeypatch, caplog):
    """@dokterdok on #140972: a sibling profile.yaml saved in a legacy Windows code page raised
    UnicodeDecodeError and failed the healthy default's whole roster. The roster still answers,
    that profile keeps a basic row, and the unreadable file is logged."""
    home = _roster_home(tmp_path, monkeypatch)
    (home / 'profiles' / 'helper' / 'profile.yaml').write_bytes('display_name: Andr\xe9\n'.encode('cp1252'))
    with caplog.at_level('WARNING', logger='gateway.session_group_controls'):
        reply = _list_profiles(home)
    assert 'result' in reply, reply
    rows = {row['name']: row for row in reply['result']['profiles']}
    assert list(rows) == ['default', 'helper', 'writer']
    assert rows['helper']['display_name'] == 'helper'
    assert rows['writer']['display_name'] == 'Writer Bot'
    assert any('helper' in record.getMessage() and 'profile.yaml' in record.getMessage()
               for record in caplog.records)


def test_profiles_list_reuses_its_disk_scan_until_a_profile_changes(tmp_path, monkeypatch):
    """@kshitijk4poor on #140972: the Bots pane polls profiles.list every 5 s, and each call
    re-enumerated profiles/ and re-parsed every profile.yaml. An unchanged tree is served from
    cache; a new profile or an edited profile.yaml is picked up on the next call."""
    import os

    from hermes_cli import profiles
    import hermes_yaml
    home = _roster_home(tmp_path, monkeypatch)
    scans, parses = [], []
    real_iter, real_load = profiles._iter_named_profile_dirs, hermes_yaml.safe_load
    monkeypatch.setattr(profiles, '_iter_named_profile_dirs',
                        lambda *a, **k: scans.append(1) or real_iter(*a, **k))
    monkeypatch.setattr(hermes_yaml, 'safe_load', lambda text: parses.append(text) or real_load(text))

    def sibling_parses():
        return sum('Writer' in str(text) for text in parses)

    _list_profiles(home)
    first_scans, first_parses = len(scans), sibling_parses()
    assert first_scans and first_parses
    _list_profiles(home)
    assert (len(scans), sibling_parses()) == (first_scans, first_parses)

    meta = home / 'profiles' / 'writer' / 'profile.yaml'
    meta.write_text('display_name: Renamed Writer\n')
    stat = meta.stat()
    os.utime(meta, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    (home / 'profiles' / 'scout').mkdir()
    (home / 'profiles' / 'scout' / 'config.yaml').write_text('model:\n  default: m\n')
    rows = {row['name']: row for row in _list_profiles(home)['result']['profiles']}
    assert rows['writer']['display_name'] == 'Renamed Writer'
    assert 'scout' in rows


def test_profiles_list_cache_sees_changes_inside_one_mtime_tick(tmp_path, monkeypatch):
    """Independent review on #140972, then CI: the stat caches missed a change made inside one
    kernel timestamp tick. Linux stamps mtime AND ctime from the coarse clock, and a dir's size and
    inode survive a file appearing and vanishing in it, so every stat field can be unchanged. The
    test forces that case on every run by making every stat key identical (one frozen tick)."""
    from gateway import session_group_controls as g
    home = _roster_home(tmp_path, monkeypatch)
    root = home / 'profiles'
    monkeypatch.setattr(g, '_stat_key', lambda stat: ('one-tick',))

    lone = root / 'lone'
    lone.mkdir()
    (lone / 'profile.yaml').write_text('display_name: Lone\n')
    assert 'lone' in [p.name for p in g._named_profile_dirs()]
    # Deleting the profile's only identity file, inside the same tick, drops it from the roster.
    (lone / 'profile.yaml').unlink()
    assert 'lone' not in [p.name for p in g._named_profile_dirs()]
    # Recreating a different marker inside the same tick lists it again.
    (lone / 'SOUL.md').write_text('hi\n')
    assert 'lone' in [p.name for p in g._named_profile_dirs()]

    # An edit that keeps every stat field serves the new content.
    meta = root / 'writer' / 'profile.yaml'
    meta.write_text('display_name: AAAA\n')
    assert g._discovered_profile_meta('writer', meta) == {'display_name': 'AAAA'}
    meta.write_text('display_name: BBBB\n')
    assert g._discovered_profile_meta('writer', meta) == {'display_name': 'BBBB'}
    meta.unlink()
    assert g._discovered_profile_meta('writer', meta) == {}
