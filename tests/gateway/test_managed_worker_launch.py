"""The ordinary daemon, not a fixture launcher, owns the real tool worker."""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if not body.get('messages'):
            message = {'role': 'assistant', 'content': 'metadata'}
        elif any(m['role'] == 'user' and m.get('content') == 'RESUME_HISTORY' for m in body['messages']):
            self.server.requests.append(body)
            message = {'role': 'assistant', 'content': 'MANAGED_HISTORY_DONE'}
        elif (self.server.control_mode and any(m['role'] == 'tool' for m in body['messages'])
              and not any(m.get('name') == 'clarify' or m.get('tool_call_id') == 'managed-clarify' for m in body['messages'])):
            self.server.requests.append(body)
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'managed-clarify', 'type': 'function', 'function': {'name': 'clarify',
                'arguments': json.dumps({'question': 'Choose managed answer', 'choices': ['Alpha', 'Beta']})}}]}
        elif any(m['role'] == 'tool' for m in body['messages']):
            self.server.requests.append(body)
            self.server.blocked.set()
            self.server.release.wait(30)
            message = {'role': 'assistant', 'content': 'MANAGED_TOOL_DONE'}
        else:
            self.server.requests.append(body)
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'managed-tool', 'type': 'function', 'function': {'name': 'terminal',
                'arguments': json.dumps({'command': self.server.command, 'timeout': 10, **self.server.tool_args})}}]}
        choice = {'index': 0, 'message': message, 'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}
        frame = {'id': 'managed-model', 'model': 'managed-model', 'choices': [choice],
                 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        kind = 'application/json'
        if body.get('stream'):
            choice['delta'] = choice.pop('message')
            for index, call in enumerate(choice['delta'].get('tool_calls', [])):
                call['index'] = index
            payload = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        else:
            payload = json.dumps(frame).encode()
        try:
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


MCP_PEER = '''import json, os, sys
for line in sys.stdin:
    r = json.loads(line); method = r.get("method"); ident = r.get("id")
    if ident is None: continue
    if method == "initialize":
        result = {"protocolVersion": r["params"]["protocolVersion"], "capabilities": {"tools": {}},
                  "serverInfo": {"name": "owned-peer", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "owned echo",
                             "inputSchema": {"type": "object", "properties": {}}}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": ident, "result": result}), flush=True)
'''


def _model_saw_tool(request, name):
    """Direct definition or the tool_search deferred catalog (bridge description lists names)."""
    return any(name in json.dumps(t) for t in request.get('tools') or [])


@pytest.mark.linux_only
# The 'background' case probes a process the retired worker reparented to init: the test spawned it
# (through its own daemon), but pid_exists() on it is outside pytest's subtree for the live guard.
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize('worker_action', ['detach', 'kill', 'controls', 'stop', 'history', 'background', 'mcp'])
def test_ordinary_owner_launches_tool_worker_and_detach_does_not_cancel(tmp_path, worker_action):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.control_mode = worker_action in {'controls', 'stop'}
    target = home / 'delete-after-consent'
    target.mkdir()
    (target / 'owned.txt').write_text('owned')
    peer.command = 'rm -rf ' + str(target) if peer.control_mode else 'printf MANAGED_TOOL_EFFECT'
    peer.tool_args = {}
    gate = tmp_path / 'background-exit'
    if worker_action == 'background':
        # A session-owned background process is a turn-boundary survivor, not turn litter (F24).
        peer.command = f'printf MANAGED_TOOL_EFFECT; while test ! -e {gate}; do sleep .1; done'
        peer.tool_args = {'background': True}
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    config = {
        'gateway': {'multiplex_profiles': False, 'managed_workers': True},
        'model': {'provider': 'custom', 'default': 'managed-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
        'approvals': {'mode': 'manual'},
        'platform_toolsets': {'cli': ['terminal']}}
    if worker_action == 'mcp':
        # A configured stdio MCP server the owner discovered must reach the worker's model too (F25).
        (home / 'peer.py').write_text(MCP_PEER)
        config['mcp_servers'] = {'owned': {'command': sys.executable, 'args': [str(home / 'peer.py')]}}
        config['platform_toolsets']['cli'] = ['terminal', 'owned']
    (home / 'config.yaml').write_text(json.dumps(config))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    audit = tmp_path / 'sqlite-opens.jsonl'
    site = tmp_path / 'audit-site'
    site.mkdir()
    (site / 'sitecustomize.py').write_text(
        "import json, os, sys\ndef witness(event, args):\n"
        "    if event == 'sqlite3.connect':\n"
        f"        with open({str(audit)!r}, 'a', encoding='utf-8') as f:\n"
        "            f.write(json.dumps({'pid': os.getpid(), 'path': str(args[0])}) + '\\n')\n"
        "sys.addaudithook(witness)\n")
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=os.pathsep.join([str(site), str(root)]),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

    def query(sql, args=()):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute(sql, args).fetchall()

    async def exercise(desc, owner):
        import psutil
        async with websocket(home, desc) as ws:
            toolsets = ['terminal', 'clarify'] if peer.control_mode else ['terminal', 'owned'] if worker_action == 'mcp' else ['terminal']
            created = await rpc(ws, 'session.create', request_id='managed', source='cli', cwd=str(home),
                                model='managed-model', provider='custom', base_url=url, api_key='loopback-only',
                                toolsets=toolsets, ignore_rules=True)
            assert 'result' in created, created
            sid = created['result']['session_id']
            submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='managed-input', text='DO_MANAGED_TOOL')
            assert 'result' in submitted, submitted
            if peer.control_mode:
                async def prompt(kind):
                    async with asyncio.timeout(20):
                        while True:
                            snapshot = await rpc(ws, 'session.resume', session_id=sid)
                            found = next((p for p in snapshot['result']['prompts'] if p['kind'] == kind), None)
                            if found:
                                return found
                            await asyncio.sleep(.05)
                approval = await prompt('approval')
                assert target.exists()
                answered = await rpc(ws, 'approval.respond', session_id=sid,
                    execution_generation=approval['execution_generation'], prompt_id=approval['prompt_id'], choice='once')
                assert answered['result']['status'] == 'resolved', answered
                clarify = await prompt('clarify')
                assert not target.exists()
                if worker_action == 'stop':
                    stopped = await rpc(ws, 'session.interrupt', session_id=sid,
                                        execution_generation=clarify['execution_generation'])
                    assert 'result' in stopped, stopped
                    peer.release.set()
                    async with asyncio.timeout(15):
                        while query('SELECT status FROM session_admissions WHERE request_id=?', ('managed-input',)) == [('started',)]:
                            await asyncio.sleep(.05)
                    assert query('SELECT status FROM session_admissions') == [('terminal',)], (home / 'restart.log').read_text()
                    return
                answered = await rpc(ws, 'clarify.respond', session_id=sid,
                    execution_generation=clarify['execution_generation'], prompt_id=clarify['prompt_id'], answer='Beta')
                assert answered['result']['status'] == 'resolved', answered
            assert await asyncio.to_thread(peer.blocked.wait, 240), (home / 'restart.log').read_text()
            workers = query('SELECT execution_id,status FROM worker_executions WHERE session_id=?', (sid,))
            assert len(workers) == 1, workers
            children = [p for p in psutil.Process(owner.pid).children() if p.cmdline()[-2:] == ['-m', 'agent.managed_worker']]
            assert len(children) == 1, [(p.pid, p.cmdline()) for p in psutil.Process(owner.pid).children()]
            pid = children[0].pid
            opened = [json.loads(line) for line in audit.read_text().splitlines()]
            # Positive control: the witness hook is live in the owner, so worker silence is real.
            assert [r for r in opened if r['pid'] == owner.pid and 'state.db' in r['path']], opened
            assert not [r for r in opened if r['pid'] == pid and 'state.db' in r['path']], opened
            canonical_fds = [f.path for f in children[0].open_files() if Path(f.path).name in {'state.db', 'state.db-wal', 'state.db-shm'}]
            assert canonical_fds == [], canonical_fds
            if worker_action == 'kill':
                follower = await rpc(ws, 'prompt.submit', session_id=sid, input_id='follower', text='NEVER_REPLAY')
                assert 'result' in follower, follower
                children[0].kill()
                async with asyncio.timeout(10):
                    while query('SELECT status FROM session_admissions WHERE request_id=?', ('managed-input',)) == [('started',)]:
                        await asyncio.sleep(.05)
                assert query('SELECT request_id,status FROM session_admissions ORDER BY seq') == [
                    ('managed-input', 'unknown'), ('follower', 'queued')]
                restored = await rpc(ws, 'session.resume', session_id=sid)
                assert any(p['status'] == 'unknown' for p in restored['result']['pending']), restored
                assert not query("SELECT value FROM state_meta WHERE key LIKE 'gateway.admission.result.v1.%'")
                peer.release.set()
                print(json.dumps({'owner_pid': owner.pid, 'killed_worker_pid': pid, 'outcome': 'unknown', 'follower': 'queued'}))
                return
        # Viewer connection is gone while the real model still holds the turn.
        assert psutil.Process(pid).is_running()
        peer.release.set()
        async with asyncio.timeout(40):
            while query('SELECT status FROM session_admissions WHERE request_id=?', ('managed-input',)) != [('terminal',)]:
                await asyncio.sleep(.05)
        async with websocket(home, desc) as ws:
            restored = await rpc(ws, 'session.resume', session_id=sid)
            history = restored['result']['messages']
            assert 'MANAGED_TOOL_DONE' in json.dumps(history), restored
            replay = await rpc(ws, 'session.events.since', session_id=sid,
                               replay_epoch=restored['result']['replay_epoch'], last_sequence=0)
            tools = [e for e in replay['result']['events'] if e['type'] in {'tool.start', 'tool.complete'}]
            assert tools and tools[0]['type'] == 'tool.start' and tools[-1]['type'] == 'tool.complete', replay
            assert str(target) not in json.dumps(tools) and 'MANAGED_TOOL_EFFECT' not in json.dumps(tools), tools
        rows = query('SELECT role,content FROM messages WHERE session_id=? ORDER BY id', (sid,))
        assert sum(role == 'user' and 'DO_MANAGED_TOOL' in content for role, content in rows) == 1, rows
        assert sum(role == 'assistant' and 'MANAGED_TOOL_DONE' in (content or '') for role, content in rows) == 1, rows
        if peer.control_mode:
            assert any(role == 'tool' and 'Beta' in content for role, content in rows), rows
        elif worker_action == 'background':
            started = json.loads(next(content for role, content in rows if role == 'tool'))
            assert started['output'] == 'Background process started', started
            background = psutil.Process(started['pid'])
            assert background.is_running() and background.status() != psutil.STATUS_ZOMBIE, started
            assert not psutil.pid_exists(pid), 'worker interpreter still alive after settlement'
            # The next admission's worker adopts it from the profile checkpoint (process_manage).
            assert started['pid'] in [e['pid'] for e in json.loads((home / 'processes.json').read_text())]
            gate.touch()
            async with asyncio.timeout(10):
                while psutil.pid_exists(background.pid) and psutil.Process(background.pid).status() != psutil.STATUS_ZOMBIE:
                    await asyncio.sleep(.05)
        else:
            assert any(role == 'tool' and 'MANAGED_TOOL_EFFECT' in content for role, content in rows), rows
        assert query('SELECT status FROM worker_executions WHERE session_id=?', (sid,)) == [('terminal',)]
        assert query('SELECT COUNT(*) FROM session_turn_leases') == [(0,)]
        if worker_action == 'mcp':
            assert all(_model_saw_tool(r, 'mcp__owned__echo') for r in peer.requests), [r.get('tools') for r in peer.requests]
        assert len(peer.requests) == (3 if peer.control_mode else 2), json.dumps([
            {'model': r.get('model'), 'roles': [m['role'] for m in r['messages']],
             'user': [str(m.get('content'))[:120] for m in r['messages'] if m['role'] == 'user']} for r in peer.requests])
        print(json.dumps({'owner_pid': owner.pid, 'worker_pid': pid, 'worker_module': 'agent.managed_worker',
                          'model_requests': len(peer.requests), 'rows': rows, 'detach_survived': True,
                          'canonical_sqlite_opens': [], 'canonical_fds': canonical_fds}))
        if worker_action == 'history':
            config = json.loads((home / 'config.yaml').read_text())
            config['model']['default'] = 'CHANGED_PROFILE_MODEL'
            config['platform_toolsets']['cli'] = []
            (home / 'config.yaml').write_text(json.dumps(config))
            async with websocket(home, desc) as ws:
                await rpc(ws, 'session.resume', session_id=sid)
                next_turn = await rpc(ws, 'prompt.submit', session_id=sid, input_id='history', text='RESUME_HISTORY')
                assert 'result' in next_turn, next_turn
                async with asyncio.timeout(30):
                    while query('SELECT status FROM session_admissions WHERE request_id=?', ('history',)) != [('terminal',)]:
                        await asyncio.sleep(.05)
                restored = await rpc(ws, 'session.resume', session_id=sid)
                assert 'MANAGED_HISTORY_DONE' in json.dumps(restored['result']['messages']), restored
            request = peer.requests[-1]
            assert request['model'] == 'managed-model'
            assert request['messages'][0] == peer.requests[0]['messages'][0], 'cached prefix changed'
            assert request['tools'] == peer.requests[0]['tools'], 'frozen tools changed'
            assert any(m['role'] == 'tool' and 'MANAGED_TOOL_EFFECT' in m['content'] for m in request['messages'])
            assert query('SELECT status FROM worker_executions WHERE session_id=?', (sid,)) == [('terminal',), ('terminal',)]


    try:
        with daemon(root, home, env, barrier=False) as (owner, desc):
            asyncio.run(exercise(desc, owner))
    finally:
        gate.touch()
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
