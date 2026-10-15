"""Held editor MCP connections follow the session's committed selection (#134889 ACP-8, ACP-10).

A viewed session keeps its editor servers connected between turns. Such a connection still sees
the server's changed tool list at the next admission start, and an empty selection closes it.
These run against a real gateway, real stdio MCP peers and a loopback model peer.
"""
import json
import sys

import pytest

from tests.acp_adapter.test_gateway_editor_mcp import (
    MCPModelPeer, _completed, _peers_running, _policies, initialize, prompt)
from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.waiting import async_wait_until

# The tool description is read on every tools/list, so one live process can change its tools.
PEER = '''import json, os, sys
from pathlib import Path
description = Path(os.environ['DESCRIPTION'])
for line in sys.stdin:
 r=json.loads(line); method=r.get('method'); ident=r.get('id')
 if ident is None: continue
 if method=='initialize': result={'protocolVersion':r['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'held-peer','version':'1'}}
 elif method=='tools/list': result={'tools':[{'name':'echo','description':description.read_text(),'inputSchema':{'type':'object','properties':{}}}]}
 elif method=='tools/call': result={'content':[{'type':'text','text':'OWNED_RESULT_'+description.read_text().replace(' ','_')}]}
 else: result={}
 print(json.dumps({'jsonrpc':'2.0','id':ident,'result':result}),flush=True)
'''


@pytest.fixture
def spec(tmp_path, model_peer):
    model_peer.RequestHandlerClass = MCPModelPeer
    (tmp_path / 'peer.py').write_text(PEER)
    (tmp_path / 'description').write_text('owned echo')
    return {'name': 'held', 'command': sys.executable, 'args': [str(tmp_path / 'peer.py')],
            'env': [{'name': 'DESCRIPTION', 'value': str(tmp_path / 'description')}]}


async def _viewed_turn(ws, spec, tmp_path, frames):
    """An acp session with *spec*, viewed by *ws*, after one settled turn through the editor tool."""
    created = await ws.rpc('session.create', request_id='held-owner', source='acp', cwd=str(tmp_path),
                           editor={'mcp_servers': [spec], 'edit_approval_policy': 'ask'})
    sid = created['session_id']
    await ws.rpc('session.resume', session_id=sid)
    await ws.rpc('prompt.submit', session_id=sid, input_id='first', text='Use owned echo')
    await async_wait_until(_completed(ws, frames, 'OWNED_RESULT_owned_echo'), timeout=30, what='first turn settled')
    return sid


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_held_server_changing_its_tools_refreezes_at_the_next_admission(daemon, spec, tmp_path):
    frames = []
    async with viewer(daemon) as ws:
        sid = await _viewed_turn(ws, spec, tmp_path, frames)
        assert _peers_running(daemon[1]['pid']), 'a viewed session holds its editor server between turns'
        # The same live process now lists a changed tool.
        (tmp_path / 'description').write_text('owned echo v2')
        await ws.rpc('prompt.submit', session_id=sid, input_id='second', text='Use owned echo')
        await async_wait_until(_completed(ws, frames, 'OWNED_RESULT_owned_echo_v2'), timeout=30,
                               what='turn after the drift settled')
    changed = [frame['params'] for frame in frames if frame['params'].get('type') == 'session.tools_changed']
    assert [event['payload']['mcp_servers'] for event in changed] == [['held']]
    (frozen,) = _policies(daemon[0], sid)
    assert 'owned echo v2' in json.dumps(frozen['schemas'])


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_empty_selection_closes_the_held_connections(daemon, spec, tmp_path):
    frames = []
    async with viewer(daemon) as ws:
        sid = await _viewed_turn(ws, spec, tmp_path, frames)
        assert _peers_running(daemon[1]['pid']), 'a viewed session holds its editor server between turns'
        idle = await ws.rpc('session.resume', session_id=sid)
        await ws.rpc('session.mutate', request_id='empty', session_id=sid, operation='editor_mcp',
                     payload={'mcp_servers': []}, expected_revision=idle['revision'],
                     expected_generation=idle['execution_generation'])
        # The viewer is still attached: only the committed empty selection may close the process.
        await async_wait_until(lambda: not _peers_running(daemon[1]['pid']), timeout=20,
                               what='held editor server closed by the empty selection')
    assert _policies(daemon[0], sid) == [None]


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_acp_load_with_no_servers_keeps_them_and_direct_editor_mcp_empty_removes_them(daemon, spec, tmp_path):
    """``session/load`` always carries ``mcpServers`` (the schema requires it), so a second editor with
    no MCP config loads with ``[]``: that keeps the session's servers for every viewer, as an omitted
    list on ``session/resume`` and ``[]`` on ``session/fork`` do. Removal is only the owner's fenced
    ``editor_mcp`` change with ``[]``, which also stops the held process."""
    async with editor(daemon, tmp_path) as acp:
        await initialize(acp)
        created = await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[spec])
        assert 'result' in created, created
        sid = created['result']['sessionId']
        assert 'result' in await prompt(acp, sid)
        assert _peers_running(daemon[1]['pid']), 'a viewed session holds its editor server between turns'
        resumed = await acp.rpc('session/resume', sessionId=sid, cwd=str(tmp_path))
        assert 'result' in resumed, resumed
        assert _peers_running(daemon[1]['pid']), 'an omitted list keeps the servers'
        frozen = _policies(daemon[0], sid)
        assert frozen != [None]
        loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
        assert 'result' in loaded, loaded
        assert _peers_running(daemon[1]['pid']), 'an empty load keeps the held server'
        assert _policies(daemon[0], sid) == frozen
        async with viewer(daemon) as ws:
            idle = await ws.rpc('session.resume', session_id=sid)
            await ws.rpc('session.mutate', request_id='remove', session_id=sid, operation='editor_mcp',
                         payload={'mcp_servers': []}, expected_revision=idle['revision'],
                         expected_generation=idle['execution_generation'])
            await async_wait_until(lambda: not _peers_running(daemon[1]['pid']), timeout=20,
                                   what='held editor server closed by the direct empty editor_mcp')
        assert _policies(daemon[0], sid) == [None]
