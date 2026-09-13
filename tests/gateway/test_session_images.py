"""Canonical uploads stage bytes only after viewer and owner authorization."""
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nL8AAAAASUVORK5CYII=')


@pytest.mark.asyncio
async def test_upload_is_authorized_private_staging_not_legacy_execution(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'wrong-home'))
    home = tmp_path / 'owner'
    home.mkdir()
    with SessionDB(db_path=home / 'state.db') as db:
        db.create_session('s', source='test')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(home),
            instance_id='current', db=db, epoch=begin_runtime_epoch(db, instance_id='current'))
        owner = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
        live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=Platform.LOCAL,
            user_id=owner.actor.subject), 'route')
        owner.subscriptions['s'] = 'test-viewer'
        live.subscribers['test-viewer'] = owner.actor
        params = {'session_id': 's', 'content_base64': base64.b64encode(PNG).decode(), 'filename': '../../shot.png'}
        async def upload(connection=owner, **changes):
            return await connection.dispatch({'id': 1, 'method': 'image.attach_bytes', 'params': {**params, **changes}})
        first = await upload()
        assert 'result' in first, first
        result = first['result']
        path = Path(result['path'])
        assert path.parent == home / 'cache/images' and path.read_bytes() == PNG
        assert path.stat().st_mode & 0o777 == 0o600
        assert result['mime'] == 'image/png' and result['bytes'] == len(PNG)
        second = await upload()
        assert second['result']['path'] != str(path)
        assert live.task is None and not db.get_messages('s')
        with db._read_ctx() as conn:
            assert conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
        before = sorted((home / 'cache/images').iterdir())
        for identity, subscribed, reason in [
            ({'user_id': 'owner'}, False, 'permission_denied'),
            ({'user_id': 'owner', 'capabilities': ['session:read']}, True, 'permission_denied'),
            ({'user_id': 'other'}, True, 'permission_denied'),
            ({'user_id': 'owner', 'profile_id': 'foreign'}, True, 'profile_mismatch'),
        ]:
            peer = AuthorityConnection(authority, object(), identity)
            if subscribed:
                peer.subscriptions['s'] = 'test-authority-boundary'
            denied = await upload(peer)
            assert denied['error']['message'] == reason, denied
        for changes in [{'session_id': ''}, {'session_id': ['s']}, {'profile_home': str(tmp_path)},
                        {'content_base64': 'not base64'}, {'content_base64': ''},
                        {'content_base64': base64.b64encode(b'x' * (25 * 1024 * 1024 + 1)).decode()},
                        {'filename': 'bad.bmp'}, {'filename': 'bad.svg'},
                        {'content_base64': base64.b64encode(b'BMbad').decode(), 'filename': ''}]:
            denied = await upload(**changes)
            assert denied['error']['message'] == 'invalid_params', denied
        assert sorted((home / 'cache/images').iterdir()) == before
        assert not (tmp_path / 'wrong-home/cache/images').exists()
        for method in ('image.attach', 'clipboard.paste'):
            denied = await owner.dispatch({'id': 2, 'method': method, 'params': params})
            assert denied['error']['message'] == 'invalid_params'
        await owner.close()



def test_authenticated_daemon_upload_retains_exact_admitted_bytes(tmp_path):
    import asyncio
    from contextlib import closing
    import json
    import sqlite3
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus
    from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket

    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    config = {'gateway': {'multiplex_profiles': False},
              'model': {'provider': 'custom', 'default': 'fixture', 'base_url': 'http://127.0.0.1:1/v1'},
              'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}
    (home / 'config.yaml').write_text(json.dumps(config))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', PYTHONUNBUFFERED='1')

    async def exercise(desc):
        url = desc['api_origin'].replace('http:', 'ws:') + '/api/ws'
        with pytest.raises(InvalidStatus):
            async with connect(url):
                pass
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='image-owner', source='tui', cwd=str(user), toolsets=[])
            assert 'result' in created, created
            sid = created['result']['session_id']
            params = dict(session_id=sid, filename='client.png', content_base64=base64.b64encode(PNG).decode())
            uploaded = await rpc(ws, 'image.attach_bytes', **params)
            assert 'result' in uploaded, uploaded
            image = uploaded['result']
            path = Path(image['path'])
            assert path.parent == home / 'cache/images' and path.read_bytes() == PNG
            async with websocket(home, desc) as detached:
                denied = await rpc(detached, 'image.attach_bytes', **params)
                assert denied['error']['message'] == 'permission_denied', denied
            for changes in ({'profile_home': str(user)}, {'filename': 'bad.bmp'}, {'content_base64': 'bad!'}):
                denied = await rpc(ws, 'image.attach_bytes', **{**params, **changes})
                assert denied['error']['message'] == 'invalid_params', denied
            assert list((home / 'cache/images').iterdir()) == [path]
            with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                assert db.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
            submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='image-input',
                                  text='Retain upload', attachments=[{'path': str(path), 'mime': image['mime']}])
            assert 'result' in submitted, submitted
            with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                payload = json.loads(db.execute('SELECT payload_json FROM session_admissions WHERE request_id=?', ('image-input',)).fetchone()[0])
            retained = payload['attachments_v1']['media'][0]
            path.write_bytes(b'client staging changed after ACK')
            assert Path(retained['path']).read_bytes() == PNG
            assert payload['attachments_v1']['media_types'] == ['image/png']
            print(json.dumps({'authenticated_public_ws': True, 'session_id': sid,
                              'owner_path': str(path), 'retained': retained, 'no_admission_on_upload': True}))
    with daemon(root, home, env, barrier=False) as (_, desc):
        asyncio.run(exercise(desc))
