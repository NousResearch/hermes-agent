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
    assert {'model', 'reasoning_effort', 'edit_approval_policy'} <= set(options)
    for option in options.values():
        assert option['currentValue'] in {row['value'] for row in option['options']}
    assert options['model']['category'] == 'model'
    assert options['reasoning_effort']['category'] == 'thought_level'
    if 'service_tier' in options:
        assert options['service_tier']['category'] == 'other'
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
    if key in options:
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
        if supported:
            assert options['service_tier']['currentValue'] == configured['agent']['service_tier']
        offered = {row['value'] for row in options.get('service_tier', {}).get('options', [])}
        assert ('fast' in offered) is supported
        info = await ws.rpc('session.info', session_id=sid)
        assert 'api_key' not in info and 'access_token' not in info
        assert 'fixture-only' not in json.dumps(info)
        baseline = await ws.rpc('session.resume', session_id=sid)
        await _set(acp, sid, 'reasoning_effort', 'high')
        assert (await ws.rpc('session.info', session_id=sid))['reasoning_effort'] == 'high'
        start = len(acp.frames)
        assert 'result' in await _slash(acp, sid, '/reasoning low')
        await async_wait_until(lambda: _updated(acp, sid, 'reasoning_effort', 'low', start), what='text reasoning update')
        assert (await ws.rpc('session.info', session_id=sid))['reasoning_effort'] == 'low'
        await _set(acp, sid, 'reasoning_effort', 'high')
        if supported:
            await _set(acp, sid, 'service_tier', 'fast')
            assert (await ws.rpc('session.info', session_id=sid))['service_tier'] == 'priority'
            assert 'result' in await _slash(acp, sid, '/fast cold')
            retained_word, retained_tier = 'cold', 'cold'
        else:
            before_refusal = await ws.rpc('session.resume', session_id=sid)
            for response in [await acp.rpc('session/set_config_option', sessionId=sid, configId='service_tier', value='fast'),
                             await _slash(acp, sid, '/fast fast')]:
                assert 'error' in response and 'fast_unavailable' in json.dumps(response), response
            after_refusal = await ws.rpc('session.resume', session_id=sid)
            assert after_refusal['revision'] == before_refusal['revision']
            assert after_refusal['info']['service_tier'] == 'ultrafast'
            assert 'result' in await _slash(acp, sid, '/fast normal')
            retained_word, retained_tier = 'normal', ''
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
        if supported:
            assert _options(forked)['service_tier']['currentValue'] == retained_word
        child = forked['result']['sessionId']
        await _set(acp, child, 'reasoning_effort', 'low')
        await _set(acp, child, 'service_tier', 'normal')
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
            assert resumed['reasoning_effort']['currentValue'] == 'high'
            if supported:
                assert resumed['service_tier']['currentValue'] == retained_word
            loaded = _options(await acp.rpc('session/load', sessionId=child, cwd=str(tmp_path), mcpServers=[]))
            assert loaded['reasoning_effort']['currentValue'] == 'low'
            if supported:
                assert loaded['service_tier']['currentValue'] == 'normal'
            await _set(acp, child, 'reasoning_effort', 'medium')
            assert (await ws.rpc('session.info', session_id=sid))['reasoning_effort'] == 'high'
            assert (await ws.rpc('session.resume', session_id=sid))['messages'] == []
            assert (home / 'config.yaml').read_bytes() == cold_config
    assert model_peer.requests == [], 'Slash-only acceptance must not make model requests'
