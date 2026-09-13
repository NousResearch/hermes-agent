"""New ACP handoff advert and unmodified cold canonical ACP restoration."""
import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import psutil
import pytest

from tests.acp.test_gateway_sessions import daemon, editor, viewer, model_peer, control  # noqa: F401


@pytest.mark.linux_only
@pytest.mark.asyncio
@pytest.mark.parametrize('daemon', ['proposed-acp-descriptor'], indirect=True)
async def test_native_new_with_proposed_advert_executes_canonical_acp(daemon, tmp_path):
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        created = await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[])
        assert 'result' in created, created
        sid = created['result']['sessionId']
        result = await acp.rpc('session/prompt', sessionId=sid,
            prompt=[{'type': 'text', 'text': 'WS_SHARED fresh ACP'}])
        assert result['result']['stopReason'] == 'end_turn', result
        assert 'LOCAL_ACK_WS_SHARED' in json.dumps(acp.frames)
    async with viewer(daemon) as ws:
        info = await ws.rpc('session.info', session_id=sid)
        assert info['source'] == 'acp' and info['cwd'] == str(tmp_path)
        saved = await ws.rpc('session.resume', session_id=sid)
        assert 'WS_SHARED fresh ACP' in json.dumps(saved['messages'])
    print('ACP_FRESH_HANDOFF_RECEIPT=' + json.dumps({'session_id': sid, 'info': info,
        'only_injected_boundary': 'AuthorityConnection.describe acp source/capability advertisement'}))


@pytest.mark.linux_only
@pytest.mark.asyncio
async def test_cold_acp_load_retains_policy_and_executes_after_daemon_restart(daemon, tmp_path, model_peer):
    home, descriptor, env, root = daemon
    async with viewer(daemon) as ws:
        snapshot = await ws.rpc('session.create', source='acp', cwd=str(tmp_path),
            toolsets=['file'], request_id='cold-editor')
        sid = snapshot['session_id']
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        assert 'result' in await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
        assert 'result' in await acp.rpc('session/prompt', sessionId=sid,
            prompt=[{'type': 'text', 'text': 'WS_SHARED before cold restart'}])
    original_pid = descriptor['pid']
    owned = psutil.Process(original_pid)
    os.kill(original_pid, signal.SIGINT)
    await asyncio.to_thread(owned.wait, 20)
    import yaml
    config = yaml.safe_load((home / 'config.yaml').read_text())
    config['platform_toolsets'] = {'acp': ['terminal']}
    (home / 'config.yaml').write_text(json.dumps(config))
    with (tmp_path / 'restart.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=root, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 40
            current = {}
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    current = control(home, 'identify')
                    if current.get('state') == 'ready' and current['instance_id'] != descriptor['instance_id']:
                        break
                except (OSError, ValueError):
                    pass
                await asyncio.sleep(.1)
            assert current.get('state') == 'ready', (tmp_path / 'restart.log').read_text()
            cold = home, current, env, root
            async with editor(cold, tmp_path) as acp:
                await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
                loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
                assert 'result' in loaded, loaded
                assert 'WS_SHARED before cold restart' in json.dumps(acp.frames)
                reply = await acp.rpc('session/prompt', sessionId=sid,
                    prompt=[{'type': 'text', 'text': 'WS_SHARED after cold restart'}])
                assert 'result' in reply, reply
                assert 'LOCAL_ACK_WS_SHARED' in json.dumps(acp.frames)
            async with viewer(cold) as ws:
                info = await ws.rpc('session.info', session_id=sid)
                assert info['source'] == 'acp' and info['cwd'] == str(tmp_path)
                messages = (await ws.rpc('session.resume', session_id=sid))['messages']
                assert 'WS_SHARED after cold restart' in json.dumps(messages)
            tool_names = [{t['function']['name'] for t in r.get('tools', [])} for r in model_peer.requests]
            assert len(tool_names) == 2 and tool_names[0] == tool_names[1]
            assert 'write_file' in tool_names[1] and 'terminal' not in tool_names[1]
            print('ACP_COLD_RECEIPT=' + json.dumps({'session_id': sid, 'info': info,
                'original_pid': original_pid, 'restored_pid': process.pid, 'frozen_tools': sorted(tool_names[1])}))
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                await asyncio.to_thread(process.wait, 20)
