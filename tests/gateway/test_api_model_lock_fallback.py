"""A confirmed Browser model lock on a canonical API turn never falls back (main parity).

One ordinary daemon with the API server: the locked primary answers 429 ``insufficient_quota``
and ``fallback_providers`` names a healthy backup. Main builds the locked agent with
``fallback_model=None`` and fails a turn served by another runtime; the shared owner must too.
"""
import asyncio
import json
import os
from pathlib import Path
import socket

import aiohttp
import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon
from tests.gateway.test_managed_worker_construct import Backup, Primary, _serve


@pytest.mark.platforms("linux")
def test_locked_session_chat_does_not_fall_back_to_the_profile_chain(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    primary, backup = _serve(Primary), _serve(Backup)
    primary.refuse = True
    purl, burl = (f'http://127.0.0.1:{s.server_port}/v1' for s in (primary, backup))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        api_port = sock.getsockname()[1]
    config = {'model': {'provider': 'custom', 'default': 'primary-model', 'base_url': purl},
              'gateway': {'multiplex_profiles': False,
                          'platforms': {'api_server': {'enabled': True, 'port': api_port, 'host': '127.0.0.1'}}},
              'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)},
              'platform_toolsets': {'api_server': []},
              'fallback_providers': [{'provider': 'custom', 'model': 'backup-model', 'base_url': burl,
                                      'api_key': 'loopback-only'}]}
    (home / 'config.yaml').write_text(json.dumps(config))
    env = {**child_env(), **{k: os.environ[k] for k in ('TIRITH_ENABLED',) if k in os.environ}}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=purl, API_SERVER_KEY='lock-owned-secret',
               API_SERVER_ENABLED='true', API_SERVER_PORT=str(api_port), PYTHONUNBUFFERED='1')

    async def exercise():
        headers = {'Authorization': 'Bearer lock-owned-secret'}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as client:
            base = f'http://127.0.0.1:{api_port}/api/sessions'
            async with client.post(base, json={'id': 'locked', 'title': 'Locked'}, headers=headers) as response:
                assert response.status in (200, 201), await response.text()
            async with client.post(f'{base}/locked/chat', headers=headers, json={
                    'message': 'NEED_FALLBACK', 'provider': 'custom', 'model': 'primary-model',
                    'require_model_lock': True}) as response:
                return response.status, await response.text()
    try:
        with daemon(root, home, env, barrier=False):
            status, text = asyncio.run(exercise())
    finally:
        for server in (primary, backup):
            server.shutdown()
            server.server_close()
    assert primary.requests, (status, text)
    assert backup.requests == [], 'a confirmed model lock was served by the profile fallback chain'
    body = json.loads(text)
    assert body['status'] == 'failed' and 'BACKUP_OK' not in text, body
    assert body['usage']['runtime']['model_lock'] == 'confirmed', body


def test_locked_turn_served_by_another_runtime_settles_failed(api):
    """Main's settlement refusal: a locked turn whose agent ran on another provider/model is a
    failed turn carrying the mismatch, never a completed answer from the wrong runtime."""
    from types import SimpleNamespace
    from gateway.run_turn_runner import _settle_api_runtime_lock
    from gateway.session_api_turn import api_execution

    agent = SimpleNamespace(provider='custom', model='backup-model', _hermes_api_runtime={'provider': 'custom'})
    result, captured = {'final_response': 'BACKUP_OK', 'completed': True}, {'usage': {}}
    token = api_execution.set({'adapter': api, 'settings': {
        'confirmed_runtime_lock': True, 'route': {'provider': 'custom', 'model': 'primary-model'},
        'route_source': 'raw_request'}})
    try:
        _settle_api_runtime_lock(agent, result, captured)
    finally:
        api_execution.reset(token)
    assert result['failed'] is True and result['completed'] is False and result['final_response'] == ''
    assert 'confirmed model lock runtime mismatch' in result['error']
