"""Native ACP stdio through ordinary gateway source, real model and MCP peers."""
import asyncio
from http.server import BaseHTTPRequestHandler
import json
import os
import re
import signal
import subprocess
import sys
import time

import psutil
import pytest

from tests.acp.test_gateway_sessions import daemon, editor, viewer, model_peer, control  # noqa: F401


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


async def initialize(acp):
    assert 'result' in await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})


async def prompt(acp, sid, text='Use owned echo'):
    return await acp.rpc('session/prompt', sessionId=sid, prompt=[{'type': 'text', 'text': text}])


@pytest.mark.linux_only
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
            async with asyncio.timeout(25):
                while not (tmp_path / 'B.entered').exists():
                    await asyncio.sleep(.05)
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
    import sqlite3
    with sqlite3.connect(daemon[0] / 'state.db') as db:
        dump = '\n'.join(db.iterdump())
    assert all('opaque-secret-' + marker not in dump for marker in ('A', 'B'))
    print('ACP_MCP_NATIVE_RECEIPT=' + json.dumps({'sessions': [sid, bs],
        'model_requests': len(model_peer.requests), 'surviving_blocked_tool': 'OWNED_RESULT_B'}))


@pytest.mark.linux_only
@pytest.mark.asyncio
async def test_native_cold_editor_rebind_refuses_changed_secret(daemon, specs, tmp_path, model_peer):
    (tmp_path / 'A').touch()
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
            assert current.get('state') == 'ready'
            cold = home, current, env, root
            async with editor(cold, tmp_path) as acp:
                await initialize(acp)
                changed = json.loads(json.dumps(specs[0]))
                changed['env'][-1]['value'] = 'changed-secret'
                refused = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[changed])
                assert 'acp_mcp_policy_conflict' in json.dumps(refused), refused
                loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[specs[0]])
                assert 'result' in loaded, loaded
                acp.frames.clear()  # Old replay is not evidence of new execution.
                assert 'result' in await prompt(acp, sid, 'Use owned echo after restart')
                assert 'OWNED_RESULT_A' in json.dumps(acp.frames)
            print('ACP_MCP_COLD_RECEIPT=' + json.dumps({'session_id': sid,
                'old_pid': descriptor['pid'], 'new_pid': process.pid, 'changed_secret_refused': True}))
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                await asyncio.to_thread(process.wait, 20)
