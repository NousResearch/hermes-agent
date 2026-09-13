"""Native config/model reads cross the authenticated owner without executing."""
import asyncio
import json

import pytest
from websockets.asyncio.client import connect

from tests.gateway.test_session_busy_controls import owner, admissions
from tests.gateway.fixtures.local_recovery_probe import rpc, websocket


@pytest.mark.linux_only
def test_client_config_projection_keeps_session_policy_and_secrets_private(tmp_path):
    with owner(tmp_path) as (home, peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='config', source='tui',
                                 toolsets=[], reasoning='low'))['result']['session_id']
                import yaml
                config = yaml.safe_load((home / 'config.yaml').read_text())
                config.update(voice={'record_key': 'ctrl+r', 'submit_mode': 'draft', 'api_key': 'PRIVATE_VOICE'},
                              paste_collapse_threshold=12)
                config['display'].update(tui_theme='light', bell_on_complete=True)
                config['approvals'] = {'mode': 'manual'}
                (home / 'config.yaml').write_text(yaml.safe_dump(config))
                before = (home / 'config.yaml').read_bytes()
                full = await rpc(ws, 'config.get', key='full', session_id=sid)
                assert full.get('result', {}).get('config', {}).get('voice', {}).get('record_key') == 'ctrl+r', full
                assert 'PRIVATE_VOICE' not in json.dumps(full)
                assert full['result']['config']['display']['bell_on_complete'] is True
                mtime = (await rpc(ws, 'config.get', key='mtime', session_id=sid))['result']
                assert mtime['mtime'] > 0 and mtime['mcp_rev']
                config['mcp_servers'] = {'later': {'command': 'not-started'}}
                (home / 'config.yaml').write_text(yaml.safe_dump(config))
                before = (home / 'config.yaml').read_bytes()
                assert (await rpc(ws, 'config.get', key='mtime', session_id=sid))['result']['mcp_rev'] == mtime['mcp_rev']
                assert (await rpc(ws, 'config.get', key='full'))['result']['config']['voice']['record_key'] == 'ctrl+r'
                assert (await rpc(ws, 'config.get', key='reasoning', session_id=sid))['result']['value'] == 'low'
                assert (await rpc(ws, 'config.get', key='project', session_id=sid, cwd=str(home)))['result']['cwd'] == str(home)
                assert (await rpc(ws, 'config.get', key='approvals.mode'))['result']['value'] == 'manual'
                assert (await rpc(ws, 'config.get', key='theme'))['result']['value'] == 'light'
                assert (await rpc(ws, 'config.set', key='busy', session_id=sid, value='queue'))['result']['value'] == 'queue'
                assert (await rpc(ws, 'config.get', key='busy', session_id=sid))['result']['value'] == 'queue'
                for method, params, reason in [
                    ('config.get', {'key': 'full', 'profile': 'foreign'}, 'profile_mismatch'),
                    ('config.get', {'key': 'full', 'session_id': sid, 'source': 'gui'}, 'invalid_params'),
                    ('config.set', {'key': 'model', 'session_id': sid, 'value': 'other'}, 'invalid_params'),
                ]:
                    assert (await rpc(ws, method, **params))['error']['message'] == reason
                other_url = desc['api_origin'].replace('http:', 'ws:') + '/api/ws?token=other-controls-actor'
                async with connect(other_url) as other:
                    # Verified dashboard operators receive the same redacted projection.
                    shared = await rpc(other, 'config.get', key='full', session_id=sid)
                    current = await rpc(ws, 'config.get', key='full', session_id=sid)
                    assert 'result' in shared, shared
                    assert shared['result'] == current['result']
                    assert 'PRIVATE_VOICE' not in json.dumps(shared)
                    assert (await rpc(other, 'config.get', key='reasoning', session_id=sid))['result']['value'] == 'low'
                assert (home / 'config.yaml').read_bytes() == before
                assert not admissions(home) and not peer.requests
        asyncio.run(probe())


@pytest.mark.linux_only
def test_model_options_uses_frozen_selection_without_composer_global_writes(tmp_path):
    with owner(tmp_path) as (home, peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='picker', source='tui',
                                 model='frozen-picker', toolsets=[]))['result']['session_id']
                before = (home / 'config.yaml').read_bytes()
                result = await rpc(ws, 'model.options', session_id=sid, include_unconfigured=True)
                assert result.get('result', {}).get('model') == 'frozen-picker', result
                assert result['result']['providers']
                assert all('slug' in row and isinstance(row['models'], list) for row in result['result']['providers'])
                assert 'loopback-only' not in json.dumps(result)
                assert (await rpc(ws, 'model.options', session_id=sid, profile='foreign'))['error']['message'] == 'profile_mismatch'
                assert (await rpc(ws, 'model.options', session_id=sid, refresh='yes'))['error']['message'] == 'invalid_params'
                other_url = desc['api_origin'].replace('http:', 'ws:') + '/api/ws?token=other-controls-actor'
                async with connect(other_url) as other:
                    shared = await rpc(other, 'model.options', session_id=sid)
                    current = await rpc(ws, 'model.options', session_id=sid)
                    assert shared.get('result', {}).get('model') == 'frozen-picker', shared
                    assert shared['result'] == current['result']
                    assert 'loopback-only' not in json.dumps(shared)
                assert (home / 'config.yaml').read_bytes() == before
                assert not admissions(home) and not peer.requests
        asyncio.run(probe())
