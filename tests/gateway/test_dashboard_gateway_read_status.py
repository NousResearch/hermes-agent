"""Password-authenticated standalone dashboards acknowledge through the live owner.

A valid mutation snapshot previously still led to HTTP 409 exclusive-maintenance
refusal whenever a separate multiplex gateway owned the selected profile.
"""
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import uuid

import httpx
import pytest

from tests.gateway.test_normal_runtime_boot import control
from tests.waiting import wait_until


def _stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.platforms('linux')
def test_dashboard_read_status_uses_owner_without_claiming_history(tmp_path):
    from hermes_state import SessionDB
    from hermes_state_mutation_binding import BINDING_PREFIX
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    model = {'model': {'provider': 'custom', 'default': 'm', 'base_url': 'http://127.0.0.1:1/v1'},
             'auxiliary': {'title_generation': {'enabled': False}}}
    (home / 'config.yaml').write_text(json.dumps({**model,
        'gateway': {'multiplex_profiles': True},
        'dashboard': {'public_url': 'http://wing.example.test', 'basic_auth': {
            'username': 'probe-operator', 'password': 'synthetic-probe-password',
            'secret': 'synthetic-probe-session-secret-0123456789'}},
        'plugins': {'enabled': ['basic']}}))
    pace = home / 'profiles' / 'pace'
    pace.mkdir(parents=True)
    (pace / 'config.yaml').write_text(json.dumps(model))
    for owner in (home, pace):
        db = SessionDB(db_path=owner / 'state.db')
        try:
            for sid in ('unclaimed-answer', 'native-owned-answer'):
                db.create_session(sid, source='tui')
                db.append_message(sid, 'user', 'Saved question')
                db.append_message(sid, 'assistant', 'Saved answer')
            binding = json.dumps([str(owner), f'uid:{os.getuid()}'], separators=(',', ':'))
            db._execute_write(lambda conn: conn.execute(
                'INSERT INTO state_meta(key,value) VALUES(?,?)', (BINDING_PREFIX + 'native-owned-answer', binding)))
        finally:
            db.close()
    root = Path(__file__).resolve().parents[2]
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if key in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               HERMES_RUNTIME_DIR=str(tmp_path / 'runtime'), PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    processes = []
    with (tmp_path / 'gateway.log').open('w+') as gateway_log, (tmp_path / 'dashboard.log').open('w+') as dashboard_log:
        try:
            gateway = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=root, env=env,
                                       stdin=subprocess.DEVNULL, stdout=gateway_log, stderr=subprocess.STDOUT)
            processes.append(gateway)
            descriptor = {}
            def ready():
                nonlocal descriptor
                if gateway.poll() is not None:
                    gateway_log.seek(0)
                    pytest.fail(gateway_log.read()[-5000:])
                try:
                    descriptor = control(home, 'identify')
                    return descriptor.get('state') == 'ready'
                except (OSError, ValueError, AssertionError):
                    return False
            wait_until(ready, timeout=60, what='gateway owner readiness')
            with socket.socket() as reserve:
                reserve.bind(('127.0.0.1', 0))
                port = reserve.getsockname()[1]
            command = ('from hermes_cli.plugins import discover_plugins; discover_plugins(); '
                       'from hermes_cli.web_server import start_server; '
                       f'start_server(host="127.0.0.1",port={port},open_browser=False)')
            dashboard = subprocess.Popen([sys.executable, '-c', command], cwd=root, env=env,
                                         stdin=subprocess.DEVNULL, stdout=dashboard_log, stderr=subprocess.STDOUT)
            processes.append(dashboard)
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=20,
                              headers={'Host': 'wing.example.test'}) as client:
                def serving():
                    if dashboard.poll() is not None:
                        dashboard_log.seek(0)
                        pytest.fail(dashboard_log.read()[-5000:])
                    try:
                        return client.get('/api/status').status_code == 200
                    except httpx.TransportError:
                        return False
                wait_until(serving, timeout=60, what='standalone dashboard readiness')
                assert client.get('/api/sessions/unclaimed-answer/mutation-snapshot',
                                  params={'profile': 'pace'}).status_code == 401
                login = client.post('/auth/password-login', json={
                    'provider': 'basic', 'username': 'probe-operator', 'password': 'synthetic-probe-password'})
                assert login.status_code == 200, login.text
                for sid in ('unclaimed-answer', 'native-owned-answer'):
                    snapshot = client.get(f'/api/sessions/{sid}/mutation-snapshot', params={'profile': 'pace'})
                    assert snapshot.status_code == 200, snapshot.text
                    body = {'profile': 'pace', 'unread': False, 'request_id': uuid.uuid4().hex,
                            'expected_revision': snapshot.json()['runtime_revision'],
                            'expected_generation': snapshot.json()['runtime_generation']}
                    result = client.patch(f'/api/sessions/{sid}', json=body)
                    assert result.status_code == 200 and result.json().get('ok') is True, result.text
                    replay = client.patch(f'/api/sessions/{sid}', json=body)
                    assert replay.status_code == 200 and replay.json() == result.json(), replay.text
                    conflicting = client.patch(f'/api/sessions/{sid}', json={**body, 'request_id': uuid.uuid4().hex})
                    assert conflicting.status_code == 409, conflicting.text
                    unread = client.patch(f'/api/sessions/{sid}', json={**body, 'unread': True,
                        'request_id': uuid.uuid4().hex, 'expected_revision': result.json()['revision']})
                    assert unread.status_code == 200 and unread.json()['unread'] is True, unread.text
                    # Read status is shared metadata; it neither claims old history nor lends its control.
                    rename = client.patch(f'/api/sessions/{sid}', json={
                        'profile': 'pace', 'title': 'Must not be forwarded', 'request_id': uuid.uuid4().hex,
                        'expected_revision': unread.json()['revision']})
                    assert rename.status_code == 409, rename.text
                envelope = {'instance_id': descriptor['instance_id'], 'profile_id': str(pace),
                    'principal_id': 'test-dashboard-operator', 'params': {
                        'session_id': 'unclaimed-answer', 'request_id': uuid.uuid4().hex,
                        'expected_revision': 2, 'expected_generation': 0,
                        'operation': 'sidebar', 'payload': {'unread': False}}}
                invalid = {**envelope, 'params': {**envelope['params'], 'payload': {'title': 'Forbidden'}}}
                assert control(home, 'session-read-status', invalid) == {'error': 'invalid_params'}
                stale = {**envelope, 'instance_id': 'retired-owner'}
                assert control(home, 'session-read-status', stale) == {'error': 'runtime_coordination_required'}
                foreign = {**envelope, 'profile_id': str(tmp_path / 'unserved-profile')}
                assert control(home, 'session-read-status', foreign) == {'error': 'profile_mismatch'}
                assert not (tmp_path / 'unserved-profile').exists()
                assert control(home, 'identify')['instance_id'] == descriptor['instance_id']
                for owner in (home, pace):
                    with sqlite3.connect(f'file:{owner / "state.db"}?mode=ro', uri=True) as conn:
                        assert conn.execute('SELECT value FROM state_meta WHERE key=?',
                            (BINDING_PREFIX + 'unclaimed-answer',)).fetchone() is None
                        assert conn.execute('SELECT value FROM state_meta WHERE key=?',
                            (BINDING_PREFIX + 'native-owned-answer',)).fetchone()[0] == json.dumps(
                                [str(owner), f'uid:{os.getuid()}'], separators=(',', ':'))
                        assert conn.execute('SELECT last_read_at FROM sessions WHERE id=?',
                            ('unclaimed-answer',)).fetchone()[0] in (None, 0)
                        assert conn.execute('SELECT runtime_revision FROM sessions WHERE id=?',
                            ('unclaimed-answer',)).fetchone()[0] == (2 if owner == pace else 0)
        finally:
            for process in reversed(processes):
                _stop(process)


@pytest.mark.asyncio
async def test_shared_read_status_does_not_grant_history_control(tmp_path):
    from types import SimpleNamespace
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('old-answer', source='tui')
        epoch = begin_runtime_epoch(db, instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(tmp_path),
                                     instance_id='current', db=db, epoch=epoch)
        identity = {'user_id': 'remote-viewer', 'provider': 'basic'}
        viewer = AuthorityConnection(authority, object(), identity)
        operator = AuthorityConnection(authority, object(), identity, operator=True)
        async def mutate(connection, operation, payload, revision):
            return await connection.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
                'session_id': 'old-answer', 'request_id': uuid.uuid4().hex,
                'expected_revision': revision, 'operation': operation, 'payload': payload}})
        try:
            denied = await mutate(viewer, 'sidebar', {'unread': False}, 0)
            assert denied['error']['message'] == 'permission_denied', denied
            accepted = await mutate(operator, 'sidebar', {'unread': False}, 0)
            assert accepted['result']['unread'] is False, accepted
            denied = await mutate(operator, 'sidebar', {'unread': False, 'title': 'Forbidden'}, 1)
            assert denied['error']['message'] == 'permission_denied', denied
            denied = await mutate(viewer, 'sidebar', {'unread': True}, 1)
            assert denied['error']['message'] == 'permission_denied', denied
            assert db.get_session('old-answer')['runtime_revision'] == 1
            assert db.get_session('old-answer')['title'] is None
        finally:
            await viewer.close()
            await operator.close()
