"""SDK-v1 selectors against the real owner, retained through a daemon restart.

Only slash commands are submitted: fixture credentials never reach an external model.
"""
import asyncio
from contextlib import asynccontextmanager
import json
import os
import signal
import sys

import psutil
import pytest
import pytest_asyncio

from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.gateway.test_normal_runtime_boot import control
from tests.waiting import async_wait_until


def _options(frame):
    assert 'result' in frame, frame
    options = {option['id']: option for option in frame['result']['configOptions']}
    assert set(options) == {'model', 'reasoning', 'fast'}
    for option in options.values():
        assert option['currentValue'] in {row['value'] for row in option['options']}
    assert options['model']['category'] == 'model'
    assert options['reasoning']['category'] == 'thought_level'
    assert options['fast']['category'] == 'model_config'
    return options


def _updated(acp, sid, key, value, after=0):
    return any(frame.get('method') == 'session/update'
               and frame['params']['sessionId'] == sid
               and frame['params']['update'].get('sessionUpdate') == 'config_option_update'
               and any(option['id'] == key and option['currentValue'] == value
                       for option in frame['params']['update']['configOptions'])
               for frame in acp.frames[after:])


async def _set(acp, sid, key, value):
    start = len(acp.frames)
    options = _options(await acp.rpc('session/set_config_option', sessionId=sid, configId=key, value=value))
    assert options[key]['currentValue'] == value
    await async_wait_until(lambda: _updated(acp, sid, key, value, start), what='ConfigOptionUpdate')
    return options


async def _slash(acp, sid, command):
    return await acp.rpc('session/prompt', sessionId=sid, prompt=[{'type': 'text', 'text': command}])


@asynccontextmanager
async def _restarted(daemon, tmp_path):
    home, descriptor, env, root = daemon
    original = psutil.Process(descriptor['pid'])
    os.kill(original.pid, signal.SIGINT)
    await asyncio.to_thread(original.wait, 20)
    with (tmp_path / 'controls-restart.log').open('w') as log:
        process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'gateway.run',
            cwd=root, env=env, stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            current = {}

            def ready():
                nonlocal current
                try:
                    current = control(home, 'identify')
                except (OSError, ValueError):
                    return False
                return current.get('state') == 'ready' and current['instance_id'] != descriptor['instance_id']

            await async_wait_until(ready, timeout=40, interval=.1, what='restarted gateway ready')
            yield home, current, env, root
        finally:
            if process.returncode is None:
                process.send_signal(signal.SIGINT)
                try:
                    await asyncio.wait_for(process.wait(), 20)
                except TimeoutError:
                    process.kill()
                    await process.wait()


@pytest_asyncio.fixture
async def routed_daemon(daemon, tmp_path):
    # The shared fixture deliberately exports a loopback OPENAI_BASE_URL. Remove
    # that override in a fresh process to test the configured official/proxy route.
    daemon[2].pop('OPENAI_BASE_URL')
    async with _restarted(daemon, tmp_path) as routed:
        yield routed


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_peer_model_mutation_refreshes_attached_acp_options(daemon, tmp_path, model_peer):
    from acp_adapter.model_catalog import encode_model_choice
    home, _, _, _ = daemon
    config_before = (home / 'config.yaml').read_bytes()
    async with viewer(daemon) as ws, editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        created = await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[])
        _options(created)
        sid = created['result']['sessionId']
        before = await ws.rpc('session.resume', session_id=sid)
        start = len(acp.frames)
        params = dict(session_id=sid, request_id='peer-model', operation='model',
                      payload={'model': 'peer-selected-fixture'}, expected_revision=before['revision'],
                      expected_generation=before['execution_generation'])
        applied = await ws.rpc('session.mutate', **params)
        assert applied['model'] == params['payload']['model']
        after = await ws.rpc('session.resume', session_id=sid)

        async def events():
            return (await ws.rpc('session.events.since', session_id=sid, replay_epoch=before['replay_epoch'],
                                 last_sequence=before['last_sequence']))['events']

        published = await events()
        infos = [event['payload'] for event in published if event['type'] == 'session.info'
                 and event['payload'].get('model') == applied['model']]
        assert len(infos) == 1, published
        fields = ('model', 'provider', 'base_url', 'api_mode', 'reasoning_effort', 'reasoning_effort_wire',
                  'service_tier', 'fast', 'cwd', 'launch_request', 'profile_id')
        assert {key: infos[0][key] for key in fields} == {key: after['info'][key] for key in fields}
        choice = encode_model_choice(infos[0]['provider'], infos[0]['model'])
        await async_wait_until(lambda: _updated(acp, sid, 'model', choice, start), what='peer model option update')
        # Exact retry republishes current info for repair, without another one-shot mutation event.
        assert await ws.rpc('session.mutate', **params) == applied
        retried = await ws.rpc('session.resume', session_id=sid)
        assert retried['revision'] == after['revision']
        replay = await events()
        assert len([event for event in replay if event['type'] == 'session.updated']) == 1
        assert len([event for event in replay if event['type'] == 'session.info'
                    and event['payload'].get('model') == applied['model']]) == 2
        assert retried['messages'] == before['messages'] == []
        assert retried['info']['launch_request'] == before['info']['launch_request']
        assert (home / 'config.yaml').read_bytes() == config_before
    assert model_peer.requests == [], 'Peer model selection must not run inference'


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('daemon', [
    {'model': {'provider': 'openai-api', 'default': 'gpt-6.1-sol',
               'base_url': 'https://proxy.example/v1', 'api_mode': 'chat_completions', 'api_key': 'fixture-only'},
     'agent': {'reasoning_effort': 'low', 'service_tier': 'ultrafast'}},
    {'model': {'provider': 'openai-api', 'default': 'gpt-6.1-sol',
               'base_url': 'https://api.openai.com/v1', 'api_mode': 'codex_responses', 'api_key': 'fixture-only'},
     'agent': {'reasoning_effort': 'low', 'service_tier': 'auto'}},
], indirect=True, ids=['unsupported-proxy', 'official-route'])
async def test_config_options_retain_owner_choices_over_real_acp_wire(routed_daemon, tmp_path, model_peer):
    daemon = routed_daemon
    home, _, _, _ = daemon
    config_before = (home / 'config.yaml').read_bytes()
    configured = json.loads(config_before)
    supported = configured['model']['base_url'] == 'https://api.openai.com/v1'
    async with viewer(daemon) as ws, editor(daemon, tmp_path) as acp:
        initialized = await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        assert initialized['result']['protocolVersion'] == 1
        created = await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[])
        options = _options(created)
        sid = created['result']['sessionId']
        assert options['fast']['currentValue'] == configured['agent']['service_tier']
        offered = {row['value'] for row in options['fast']['options']}
        assert ('fast' in offered) is supported
        info = await ws.rpc('session.info', session_id=sid)
        assert (info['base_url'], info['api_mode']) == (
            configured['model']['base_url'], configured['model']['api_mode'])
        assert 'api_key' not in info and 'access_token' not in info
        assert 'fixture-only' not in json.dumps(info)
        baseline = await ws.rpc('session.resume', session_id=sid)
        await _set(acp, sid, 'reasoning', 'high')
        assert (await ws.rpc('session.info', session_id=sid))['reasoning_effort'] == 'high'
        start = len(acp.frames)
        assert 'result' in await _slash(acp, sid, '/reasoning low')
        await async_wait_until(lambda: _updated(acp, sid, 'reasoning', 'low', start), what='text reasoning update')
        assert (await ws.rpc('session.info', session_id=sid))['reasoning_effort'] == 'low'
        await _set(acp, sid, 'reasoning', 'high')
        if supported:
            await _set(acp, sid, 'fast', 'fast')
            assert (await ws.rpc('session.info', session_id=sid))['service_tier'] == 'priority'
            assert 'result' in await _slash(acp, sid, '/fast cold')
            retained_word, retained_tier = 'cold', 'cold'
        else:
            before_refusal = await ws.rpc('session.resume', session_id=sid)
            for response in [await acp.rpc('session/set_config_option', sessionId=sid, configId='fast', value='fast'),
                             await _slash(acp, sid, '/fast fast')]:
                assert 'error' in response and 'not available' in json.dumps(response), response
            after_refusal = await ws.rpc('session.resume', session_id=sid)
            assert after_refusal['revision'] == before_refusal['revision']
            assert after_refusal['info']['service_tier'] == 'ultrafast'
            assert 'result' in await _slash(acp, sid, '/fast normal')
            retained_word, retained_tier = 'normal', None
        retained = await ws.rpc('session.resume', session_id=sid)
        assert retained['info']['service_tier'] == retained_tier
        assert retained['revision'] > baseline['revision']
        assert retained['messages'] == baseline['messages'] == []
        assert retained['info']['launch_request'] == baseline['info']['launch_request']
        assert retained['info']['yolo'] == baseline['info']['yolo'] is False
        for command in ('/fast fast --global', '/reasoning low --global'):
            refused = await _slash(acp, sid, command)
            assert 'error' in refused and 'session-scoped' in json.dumps(refused), refused
        assert (await ws.rpc('session.resume', session_id=sid))['revision'] == retained['revision']
        forked = await acp.rpc('session/fork', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
        assert _options(forked)['fast']['currentValue'] == retained_word
        child = forked['result']['sessionId']
        await _set(acp, child, 'reasoning', 'low')
        await _set(acp, child, 'fast', 'normal')
        assert (await ws.rpc('session.info', session_id=sid))['reasoning_effort'] == 'high'
        assert (home / 'config.yaml').read_bytes() == config_before
    # Changing the disposable profile default proves cold resume reads the retained
    # session policy; fresh owner/editor processes have no in-memory control state.
    configured['agent'].update(reasoning_effort='minimal', service_tier='fast')
    (home / 'config.yaml').write_text(json.dumps(configured))
    cold_config = (home / 'config.yaml').read_bytes()
    async with _restarted(daemon, tmp_path) as cold:
        async with editor(cold, tmp_path) as acp, viewer(cold) as ws:
            await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
            resumed = _options(await acp.rpc('session/resume', sessionId=sid, cwd=str(tmp_path), mcpServers=[]))
            assert (resumed['reasoning']['currentValue'], resumed['fast']['currentValue']) == ('high', retained_word)
            loaded = _options(await acp.rpc('session/load', sessionId=child, cwd=str(tmp_path), mcpServers=[]))
            assert (loaded['reasoning']['currentValue'], loaded['fast']['currentValue']) == ('low', 'normal')
            await _set(acp, child, 'reasoning', 'medium')
            assert (await ws.rpc('session.info', session_id=sid))['reasoning_effort'] == 'high'
            assert (await ws.rpc('session.resume', session_id=sid))['messages'] == []
            assert (home / 'config.yaml').read_bytes() == cold_config
    assert model_peer.requests == [], 'Slash-only acceptance must not make model requests'
