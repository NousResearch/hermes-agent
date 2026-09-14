"""Discovery stays transport-independent and cannot select another profile."""
import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from tests.gateway.test_normal_runtime_boot import control


async def _legacy_ws_snapshot(root, home, env, requests):
    legacy_home = home.parent / 'legacy-state'
    legacy_home.mkdir(mode=0o700)
    shutil.copy2(home / 'config.yaml', legacy_home / 'config.yaml')
    for directory in ('skills', 'plugins'):
        shutil.copytree(home / directory, legacy_home / directory)
    legacy_env = {**env, 'HERMES_HOME': str(legacy_home),
                  'HERMES_DASHBOARD_SESSION_TOKEN': 'disposable-discovery-token'}
    with (home.parent / 'legacy-serve.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'hermes_cli.main', 'serve',
                                    '--host', '127.0.0.1', '--port', '0', '--skip-build', '--isolated'],
                                   cwd=root, env=legacy_env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 40
            port = None
            output = ''
            while process.poll() is None and time.monotonic() < deadline:
                log.seek(0)
                output = log.read()
                if match := re.search(r'HERMES_BACKEND_READY port=(\d+)', output):
                    port = int(match.group(1))
                    break
                await asyncio.sleep(.1)
            assert port, output
            results = []
            async with connect(f'ws://127.0.0.1:{port}/api/ws?token=disposable-discovery-token') as ws:
                for rid, (method, params) in enumerate(requests):
                    await ws.send(json.dumps({'jsonrpc': '2.0', 'id': rid, 'method': method, 'params': params}))
                    async with asyncio.timeout(20):
                        while True:
                            reply = json.loads(await ws.recv())
                            if reply.get('id') == rid:
                                assert 'result' in reply, reply
                                results.append(reply['result'])
                                break
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=20) == 0
            return results, {'legacy_ws_pid': process.pid, 'legacy_ws_port': port,
                             'legacy_ws_exit': process.returncode}
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@pytest.mark.linux_only
def test_ordinary_daemon_discovery_matches_legacy_extensions(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'discovery-fixture',
                  'base_url': 'http://127.0.0.1:1/v1'},
        'auxiliary': {'title_generation': {'enabled': False}},
        'quick_commands': {'probe-quick': {'type': 'alias', 'target': '/help',
                                          'description': 'Disposable quick discovery'}},
        'plugins': {'enabled': ['probe_plugin']},
    }))
    skill = home / 'skills' / 'probe-skill'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\nname: probe-skill\ndescription: Disposable skill discovery\n---\n# Probe\nOnly a discovery fixture.\n')
    plugin = home / 'plugins' / 'probe_plugin'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text('name: probe_plugin\nversion: 0.1.0\ndescription: Discovery fixture\n')
    (plugin / '__init__.py').write_text(
        "def register(ctx):\n    ctx.register_command('probe-plugin', lambda args: args, description='Disposable plugin discovery')\n")
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    requests = [('commands.catalog', {}), *[('complete.slash', {'text': text})
        for text in ('/', '/probe', '/probe-quick', '/help', '/details c', 'plain')]]
    legacy_code = '''import json, sys
from tui_gateway import server
requests = json.loads(sys.argv[1])
results = [server._methods[method](i, params)['result'] for i, (method, params) in enumerate(requests)]
sys.__stdout__.write(json.dumps(results) + '\\n')
'''

    async def probe(descriptor):
        ws_baseline, legacy_receipt = await _legacy_ws_snapshot(root, home, env, requests)
        assert ws_baseline == expected
        url = descriptor['api_origin'].replace('http:', 'ws:') + '/api/ws'
        with pytest.raises(InvalidStatus):
            async with connect(url):
                pass
        ticket = control(home, 'session-ticket', {
            'profile_id': str(home), 'instance_id': descriptor['instance_id'], 'purpose': 'interactive'})
        async with connect(url, subprotocols=['hermes-gateway-v1',
                           'hermes-gateway-ticket.' + ticket['ticket']]) as ws:
            async def rpc(method, params):
                await ws.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
                async with asyncio.timeout(20):
                    while True:
                        reply = json.loads(await ws.recv())
                        if reply.get('id') == method:
                            return reply
            for (method, params), baseline in zip(requests, expected):
                reply = await rpc(method, params)
                got = reply.get('result')
                if got != baseline and isinstance(got, dict) and 'skills' in got:
                    a, b = got['skills'], baseline['skills']
                    delta = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
                    raise AssertionError(f'skills delta daemon vs legacy: {delta}')
                assert got == baseline, reply
            for method in ('commands.catalog', 'complete.slash'):
                reply = await rpc(method, {'profile': 'foreign-profile', **({'text': '/probe'} if method == 'complete.slash' else {})})
                assert reply['error']['message'] == 'profile_mismatch', reply
        return {**legacy_receipt, 'legacy_pid_exit': legacy.returncode, 'catalog_pairs': len(pairs),
                'skill_count': expected[0]['skill_count'], 'compared_requests': len(requests),
                'unauthenticated_ws': 'rejected', 'foreign_profile': 'rejected'}

    with (tmp_path / 'gateway.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            descriptor = {}
            deadline = time.monotonic() + 40
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    descriptor = control(home, 'identify')
                    if descriptor.get('state') == 'ready':
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            log.seek(0)
            assert descriptor.get('state') == 'ready', (descriptor, log.read())
            # Ordinary startup syncs bundled skills; compare both transports on that same snapshot.
            legacy = subprocess.run([sys.executable, '-c', legacy_code, json.dumps(requests)],
                                    cwd=root, env=env, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=45)
            assert legacy.returncode == 0, legacy.stderr
            expected = json.loads(legacy.stdout.splitlines()[-1])
            pairs = dict(expected[0]['pairs'])
            for name in ('/probe-skill', '/probe-quick', '/probe-plugin'):
                assert name in pairs, (name, expected[0]['warning'])
            assert {'probe-skill', 'probe-plugin'} <= {
                row['text'].strip().lstrip('/') for row in expected[2]['items']}
            receipt = asyncio.run(probe(descriptor))
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=20) == 0
            print(json.dumps({**receipt, 'pid': process.pid, 'daemon_exit': process.returncode,
                              'instance_id': descriptor['instance_id'], 'api_origin': descriptor['api_origin']}))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@pytest.mark.asyncio
async def test_discovery_authorizes_before_scanning(tmp_path, monkeypatch):
    from agent import skill_commands
    from hermes_cli import plugins
    from gateway.session_controls import AuthorityConnection

    def forbidden(*args, **kwargs):
        pytest.fail('unauthorized request reached discovery')
    monkeypatch.setattr(skill_commands, 'scan_skill_commands', forbidden)
    monkeypatch.setattr(plugins, 'get_plugin_commands', forbidden)
    authority = SimpleNamespace(instance_id='owned', profile_id=str(tmp_path), events={})
    for identity in ({}, {'user_id': 'reader', 'capabilities': []},
                     {'user_id': 'reader', 'profile_id': 'foreign'},
                     {'user_id': 'reader', 'instance_id': 'stale'}):
        connection = AuthorityConnection(authority, object(), identity)
        try:
            for method in ('commands.catalog', 'complete.slash'):
                reply = await connection.dispatch({'id': 1, 'method': method,
                    'params': {'text': '/probe'} if method == 'complete.slash' else {}})
                assert reply['error']['message'] == 'permission_denied', reply
        finally:
            await connection.close()
