"""Native ACP stdio through ordinary gateway source, real model and MCP peers.

The editor's ``mcpServers`` is a request: the owner binds each session's servers under that
session's own scope, changes them only through the fenced ``editor_mcp`` mutation (or a re-freeze
at an admission start), and never writes a transport credential to state.db or events
(#134889 ACP-8, ACP-10, ACP-12).
"""
import asyncio
from http.server import BaseHTTPRequestHandler
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time

import psutil
import pytest

from hermes_cli.gateway_client import GatewayRPCError
from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.gateway.test_normal_runtime_boot import control
from tests.waiting import async_wait_until


class MCPModelPeer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(request)
        messages = request.get('messages', [])
        tail = messages[next((i for i in range(len(messages)-1, -1, -1)
                              if messages[i]['role'] == 'user'), 0):]
        results = [m for m in tail if m['role'] == 'tool']
        names = [t['function']['name'] for t in request.get('tools', [])
                 if t['function']['name'].startswith('mcp__')]
        if not names:
            names = re.findall(r'mcp__editor_[a-z0-9_]+__echo', json.dumps(results))
        finished = next((m['content'] for m in results if 'OWNED_RESULT_' in str(m['content'])), None)
        message = {'role': 'assistant', 'content': 'DONE ' + str(finished)}
        if not finished:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'mcp-fixture-' + str(len(results)), 'type': 'function', 'function': {
                    'name': ('tool_describe' if len(results) == 1 else 'tool_call') if names else 'tool_search',
                    'arguments': json.dumps(({'names': [names[0]]} if len(results) == 1 else
                        {'name': names[0], 'arguments': {}}) if names else {'queries': ['owned echo']})}}]}
        tool = 'tool_calls' in message
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if request.get('stream') else 'application/json')
        self.end_headers()
        if request.get('stream'):
            if tool:
                message['tool_calls'][0]['index'] = 0
            for delta, finish in [(message, None), ({}, 'tool_calls' if tool else 'stop')]:
                frame = {'id': 'local-mcp', 'object': 'chat.completion.chunk', 'created': 1,
                         'model': 'local-control', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                self.wfile.write(('data: ' + json.dumps(frame) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
        else:
            self.wfile.write(json.dumps({'id': 'local-mcp', 'object': 'chat.completion', 'created': 1,
                'model': 'local-control', 'choices': [{'index': 0, 'message': message,
                'finish_reason': 'tool_calls' if tool else 'stop'}]}).encode())
        self.wfile.flush()


@pytest.fixture
def specs(tmp_path, model_peer):
    model_peer.RequestHandlerClass = MCPModelPeer
    script = tmp_path / 'peer.py'
    script.write_text('''import json, os, sys, time
from pathlib import Path
for line in sys.stdin:
 r=json.loads(line); method=r.get('method'); ident=r.get('id')
 if ident is None: continue
 if method=='initialize': result={'protocolVersion':r['params']['protocolVersion'],'capabilities':{'tools':{}},'serverInfo':{'name':'owned-peer','version':'1'}}
 elif method=='tools/list': result={'tools':[{'name':'echo','description':'owned echo','inputSchema':{'type':'object','properties':{}}}]}
 elif method=='tools/call':
  marker=os.environ['MARKER']; gate=Path(os.environ['GATE'])
  Path(str(gate)+'.entered').write_text(marker)
  while not gate.exists(): time.sleep(.02)
  result={'content':[{'type':'text','text':'OWNED_RESULT_'+marker}]}
 else: result={}
 print(json.dumps({'jsonrpc':'2.0','id':ident,'result':result}),flush=True)
''')
    def spec(marker):
        return {'name': 'same-name', 'command': sys.executable, 'args': [str(script)], 'env': [
            {'name': 'MARKER', 'value': marker}, {'name': 'GATE', 'value': str(tmp_path / marker)},
            {'name': 'BORROWED_TOKEN', 'value': 'opaque-secret-' + marker}]}
    return spec('A'), spec('B')


def respec(spec, marker, tmp_path):
    """*spec*'s server with another marker, gate and borrowed secret (same name, same tools)."""
    changed = json.loads(json.dumps(spec))
    changed['env'] = [{'name': 'MARKER', 'value': marker}, {'name': 'GATE', 'value': str(tmp_path / marker)},
                      {'name': 'BORROWED_TOKEN', 'value': 'opaque-secret-' + marker}]
    return changed


async def initialize(acp):
    assert 'result' in await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})


async def prompt(acp, sid, text='Use owned echo'):
    return await acp.rpc('session/prompt', sessionId=sid, prompt=[{'type': 'text', 'text': text}])


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_native_opposing_editor_mcp_survives_viewer_exit(daemon, specs, tmp_path, model_peer):
    (tmp_path / 'A').write_text('release')
    async with editor(daemon, tmp_path) as b:
        await initialize(b)
        bc = await b.rpc('session/new', cwd=str(tmp_path), mcpServers=[specs[1]])
        assert 'result' in bc, bc
        bs = bc['result']['sessionId']
        pending = asyncio.create_task(prompt(b, bs))
        try:
            await async_wait_until(lambda: (tmp_path / 'B.entered').exists(), timeout=25,
                                   what="B's MCP tool call entered")
            async with editor(daemon, tmp_path) as a:
                await initialize(a)
                ac = await a.rpc('session/new', cwd=str(tmp_path), mcpServers=[specs[0]])
                assert 'result' in ac, ac
                sid = ac['result']['sessionId']
                assert 'result' in await prompt(a, sid)
                assert 'OWNED_RESULT_A' in json.dumps(a.frames)
                assert 'OWNED_RESULT_B' not in json.dumps(a.frames)
            # A has exited and its scoped server is released; B's real call is still waiting.
            (tmp_path / 'B').write_text('release')
            assert 'result' in await pending
            assert 'OWNED_RESULT_B' in json.dumps(b.frames)
            assert 'OWNED_RESULT_A' not in json.dumps(b.frames)
        finally:
            (tmp_path / 'B').touch()
            await asyncio.gather(pending, return_exceptions=True)
    async with viewer(daemon) as ws:
        for session_id in (sid, bs):
            saved = await ws.rpc('session.resume', session_id=session_id)
            assert 'OWNED_RESULT_' in json.dumps(saved['messages'])
    with sqlite3.connect(daemon[0] / 'state.db') as db:
        dump = '\n'.join(db.iterdump())
    assert all('opaque-secret-' + marker not in dump for marker in ('A', 'B'))
    print('ACP_MCP_NATIVE_RECEIPT=' + json.dumps({'sessions': [sid, bs],
        'model_requests': len(model_peer.requests), 'surviving_blocked_tool': 'OWNED_RESULT_B'}))


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_native_cold_editor_rebind_applies_changed_secret(daemon, specs, tmp_path, model_peer):
    """A changed editor config on load is the owner's ``editor_mcp`` rebind, not a refusal (#134889 ACP-10)."""
    (tmp_path / 'A').touch()
    (tmp_path / 'C').touch()
    async with editor(daemon, tmp_path) as acp:
        await initialize(acp)
        created = await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[specs[0]])
        assert 'result' in created, created
        sid = created['result']['sessionId']
        assert 'result' in await prompt(acp, sid)
    home, descriptor, env, root = daemon
    owned = psutil.Process(descriptor['pid'])
    os.kill(owned.pid, signal.SIGINT)
    await asyncio.to_thread(owned.wait, 20)
    with (tmp_path / 'cold.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=root, env=env,  # noqa: ASYNC220 -- Popen returns immediately; the test drives the child through its handle
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
            assert current.get('state') == 'ready'
            cold = home, current, env, root
            async with editor(cold, tmp_path) as acp:
                await initialize(acp)
                changed = respec(specs[0], 'C', tmp_path)
                loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[changed])
                assert 'result' in loaded, loaded
                acp.frames.clear()  # Old replay is not evidence of new execution.
                assert 'result' in await prompt(acp, sid, 'Use owned echo after restart')
                frames = json.dumps(acp.frames)
                assert 'OWNED_RESULT_C' in frames and 'OWNED_RESULT_A' not in frames
                # Same server, same schemas: the frozen manifest (prompt-cache prefix) did not move.
                assert 'Editor MCP tools changed' not in frames
                assert 'opaque-secret-' not in frames
            with sqlite3.connect(home / 'state.db') as db:
                dump = '\n'.join(db.iterdump())
            assert all('opaque-secret-' + marker not in dump for marker in 'AC')
            print('ACP_MCP_COLD_RECEIPT=' + json.dumps({'session_id': sid,
                'old_pid': descriptor['pid'], 'new_pid': process.pid, 'changed_secret_rebound': True}))
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                await asyncio.to_thread(process.wait, 20)


def _policies(home, *session_ids):
    with sqlite3.connect(home / 'state.db') as db:
        rows = [db.execute('SELECT value FROM state_meta WHERE key=?',
                           ('gateway.local_policy.v1:' + sid,)).fetchone()[0] for sid in session_ids]
        dump = '\n'.join(db.iterdump())
    assert 'opaque-secret-' not in dump
    return [json.loads(json.loads(row)['policy']['editor_mcp_json'] or 'null') for row in rows]


def _peers_running(pid):
    return [child for child in psutil.Process(pid).children(recursive=True)
            if any('peer.py' in part for part in child.cmdline())]


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_native_fork_binds_child_editor_mcp_in_its_own_scope(daemon, specs, tmp_path, model_peer):
    for marker in 'AC':
        (tmp_path / marker).write_text('release')
    async with editor(daemon, tmp_path) as acp:
        await initialize(acp)
        created = await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[specs[0]])
        assert 'result' in created, created
        sid = created['result']['sessionId']
        assert 'result' in await prompt(acp, sid)
        forked = await acp.rpc('session/fork', sessionId=sid, cwd=str(tmp_path),
                               mcpServers=[respec(specs[0], 'C', tmp_path)])
        assert 'result' in forked, forked
        child = forked['result']['sessionId']
        acp.frames.clear()
        assert 'result' in await prompt(acp, child)
        frames = json.dumps(acp.frames)
        assert 'OWNED_RESULT_C' in frames and 'OWNED_RESULT_A' not in frames
        # No list: the child gets a copy of the parent's servers, still under its own scope.
        copied = await acp.rpc('session/fork', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
        assert 'result' in copied, copied
        twin = copied['result']['sessionId']
        acp.frames.clear()
        assert 'result' in await prompt(acp, twin)
        assert 'OWNED_RESULT_A' in json.dumps(acp.frames)
        acp.frames.clear()
        assert 'result' in await prompt(acp, sid, 'Use owned echo again')
        frames = json.dumps(acp.frames)
        assert 'OWNED_RESULT_A' in frames and 'OWNED_RESULT_C' not in frames
        assert 'opaque-secret-' not in frames
    parent, forked_policy, twin_policy = _policies(daemon[0], sid, child, twin)
    assert len({parent['scope'], forked_policy['scope'], twin_policy['scope']}) == 3
    assert forked_policy['fingerprint'] != parent['fingerprint'] == twin_policy['fingerprint']


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_native_schema_drift_refreezes_at_the_next_turn(daemon, specs, tmp_path, model_peer):
    (tmp_path / 'A').write_text('release')
    async with editor(daemon, tmp_path) as acp:
        await initialize(acp)
        created = await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[specs[0]])
        assert 'result' in created, created
        sid = created['result']['sessionId']
        assert 'result' in await prompt(acp, sid)
    # The last viewer left: its connection closes. The server's schema then changes.
    await async_wait_until(lambda: not _peers_running(daemon[1]['pid']), timeout=20,
                           what='editor MCP peer released with the last viewer')
    script = tmp_path / 'peer.py'
    script.write_text(script.read_text().replace("'owned echo'", "'drifted echo'"))
    async with editor(daemon, tmp_path) as acp:
        await initialize(acp)
        loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[specs[0]])
        assert 'result' in loaded, loaded
        acp.frames.clear()
        reply = await prompt(acp, sid, 'Use owned echo after drift')
        assert 'result' in reply, reply
        frames = json.dumps(acp.frames)
        assert 'Editor MCP tools changed (servers: same-name)' in frames
        assert 'OWNED_RESULT_A' in frames and 'opaque-secret-' not in frames
    (frozen,) = _policies(daemon[0], sid)
    assert 'drifted echo' in json.dumps(frozen['schemas'])


def _completed(ws, frames, marker):
    """Predicate: a ``message.complete`` carrying *marker* arrived (its admission is terminal)."""
    def seen():
        while not ws.events.empty():
            frames.append(ws.events.get_nowait())
        return any(frame['params'].get('type') == 'message.complete' and marker in json.dumps(frame)
                   for frame in frames)
    return seen


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_editor_mcp_mutation_is_fenced_idle_only_and_secret_free(daemon, specs, tmp_path, model_peer):
    async with viewer(daemon) as ws:
        created = await ws.rpc('session.create', request_id='editor-owner', source='acp', cwd=str(tmp_path),
                               editor={'mcp_servers': [specs[0]], 'edit_approval_policy': 'ask'})
        sid = created['session_id']
        snapshot = await ws.rpc('session.resume', session_id=sid)
        changed = respec(specs[0], 'C', tmp_path)
        base = dict(session_id=sid, operation='editor_mcp', payload={'mcp_servers': [changed]})
        with pytest.raises(GatewayRPCError, match='invalid_params'):
            await ws.rpc('session.mutate', request_id='unfenced', expected_revision=snapshot['revision'], **base)
        with pytest.raises(GatewayRPCError, match='revision_conflict'):
            await ws.rpc('session.mutate', request_id='stale', expected_revision=snapshot['revision'] + 5,
                         expected_generation=snapshot['execution_generation'], **base)
        await ws.rpc('prompt.submit', session_id=sid, input_id='busy-turn', text='Use owned echo')
        await async_wait_until(lambda: (tmp_path / 'A.entered').exists(), timeout=25, what='turn inside the MCP tool')
        running = await ws.rpc('session.resume', session_id=sid)
        with pytest.raises(GatewayRPCError, match='session_busy'):
            await ws.rpc('session.mutate', request_id='busy', expected_revision=running['revision'],
                         expected_generation=running['execution_generation'], **base)
        (tmp_path / 'A').write_text('release')
        frames = []
        await async_wait_until(_completed(ws, frames, 'OWNED_RESULT_A'), timeout=30, what='first turn settled')
        idle = await ws.rpc('session.resume', session_id=sid)
        fence = dict(expected_revision=idle['revision'], expected_generation=idle['execution_generation'])
        applied = await ws.rpc('session.mutate', request_id='applied', **fence, **base)
        assert applied['tools_changed'] is False and applied['mcp_servers'] == ['same-name']
        assert applied['revision'] > idle['revision']
        assert await ws.rpc('session.mutate', request_id='applied', **fence, **base) == applied
        (tmp_path / 'C').write_text('release')
        await ws.rpc('prompt.submit', session_id=sid, input_id='after-rebind', text='Use owned echo')
        await async_wait_until(_completed(ws, frames, 'OWNED_RESULT_C'), timeout=30, what='rebound turn settled')
        events = json.dumps(frames)
        assert 'OWNED_RESULT_C' in events
        assert 'opaque-secret-' not in events and 'BORROWED_TOKEN' not in events
    from gateway.session_local_mcp import fingerprint
    (frozen,) = _policies(daemon[0], sid)
    assert frozen['fingerprint'] == fingerprint([changed])
