"""An ordinary managed worker serves its session's frozen config, never the live profile file.

The owner hydrates the creation-time snapshot into the bootstrap; the child binds it before any
config import, so an edit to an already-selected MCP server's transport after creation reaches
only sessions created after the edit (andrexibiza N1). Plugins, MCP and hooks stay ordinary.
"""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import textwrap
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket

PEER = '''import json, sys
for line in sys.stdin:
    r = json.loads(line); ident = r.get("id")
    if ident is None: continue
    if r.get("method") == "initialize":
        result = {"protocolVersion": r["params"]["protocolVersion"], "capabilities": {"tools": {}},
                  "serverInfo": {"name": "peer", "version": "1"}}
    elif r.get("method") == "tools/list":
        result = {"tools": [{"name": "echo_" + sys.argv[1], "description": "x",
                             "inputSchema": {"type": "object", "properties": {}}}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": ident, "result": result}), flush=True)
'''


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('messages'):
            self.server.requests.append(body)
        choice = {'index': 0, 'delta': {'role': 'assistant', 'content': 'OK'}, 'finish_reason': 'stop'}
        frame = {'id': 'f', 'model': 'frozen-model', 'choices': [choice],
                 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        if body.get('stream'):
            payload, kind = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode(), 'text/event-stream'
        else:
            choice['message'] = choice.pop('delta')
            payload, kind = json.dumps(frame).encode(), 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.mark.platforms("linux")
def test_existing_session_keeps_frozen_mcp_transport_after_profile_edit(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'peer.py').write_text(PEER)
    config = {'gateway': {'multiplex_profiles': False, 'managed_workers': True},
              'model': {'provider': 'custom', 'default': 'frozen-model', 'base_url': url},
              'auxiliary': {'title_generation': {'enabled': False}},
              'mcp_servers': {'owned': {'command': sys.executable, 'args': [str(home / 'peer.py'), 'FROZEN_A']}},
              'platform_toolsets': {'cli': ['owned']}}
    (home / 'config.yaml').write_text(json.dumps(config))
    env = {**child_env(), **{k: os.environ[k] for k in ('TIRITH_ENABLED',) if k in os.environ}}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

    def status(request_id):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute('SELECT status FROM session_admissions WHERE request_id=?', (request_id,)).fetchall()

    async def create(ws, name):
        created = await rpc(ws, 'session.create', request_id=name, source='cli', cwd=str(home), model='frozen-model',
                            provider='custom', base_url=url, api_key='loopback-only', toolsets=['owned'], ignore_rules=True)
        assert 'result' in created, created
        return created['result']['session_id']

    async def tools_seen(ws, sid, request_id):
        before = len(peer.requests)
        submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id=request_id, text='HELLO')
        assert 'result' in submitted, submitted
        async with asyncio.timeout(120):
            while status(request_id) != [('terminal',)]:
                await asyncio.sleep(.05)
        tools = json.dumps([r.get('tools') for r in peer.requests[before:]])
        return [tag for tag in ('echo_FROZEN_A', 'echo_LIVE_B') if tag in tools]

    async def exercise(desc):
        async with websocket(home, desc) as ws:
            existing = await create(ws, 'existing')
            edited = json.loads((home / 'config.yaml').read_text())
            edited['mcp_servers']['owned']['args'] = [str(home / 'peer.py'), 'LIVE_B']
            (home / 'config.yaml').write_text(json.dumps(edited))
            fresh = await create(ws, 'fresh')
            assert await tools_seen(ws, existing, 'existing-turn') == ['echo_FROZEN_A']
            assert await tools_seen(ws, fresh, 'fresh-turn') == ['echo_LIVE_B']
    try:
        with daemon(root, home, env, barrier=False) as (_owner, desc):
            asyncio.run(exercise(desc))
    finally:
        peer.shutdown()
        peer.server_close()


def test_ordinary_worker_binding_freezes_readers_but_keeps_customizations(tmp_path):
    """The ordinary binding serves the snapshot to every config reader without the bypass policy
    (plugins still load), and a worker write-back lands only its own change on the live file."""
    root = Path(__file__).resolve().parents[2]
    home = tmp_path / 'home'
    plugin = home / 'plugins' / 'sentinel'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text('name: sentinel\nversion: 1.0.0\nkind: standalone\n', encoding='utf-8')
    (plugin / '__init__.py').write_text("import os; from pathlib import Path\nPath(os.environ['HERMES_HOME'], 'plugin-executed').touch()\ndef register(ctx): pass\n", encoding='utf-8')
    frozen = {'plugins': {'enabled': ['sentinel']}, 'agent': {'max_turns': 7}, 'command_allowlist': ['echo *'],
              'mcp_servers': {'x': {'command': 'FROZEN_A'}}}
    live = {**frozen, 'agent': {'max_turns': 99}, 'mcp_servers': {'x': {'command': 'LIVE_B'}},
            'display': {'personality': 'EDITED_AFTER_CREATE'}}
    (home / 'config.yaml').write_text(json.dumps(live), encoding='utf-8')
    script = tmp_path / 'probe.py'
    script.write_text(textwrap.dedent(f'''
        import json, os, sys
        sys.path.insert(0, {str(root)!r})
        from agent.managed_worker import bind_worker_policy
        bind_worker_policy({{'safe_mode': False, 'ignore_user_config': False,
                             'policy': {{'config_json': json.dumps({frozen!r})}}}})
        from agent.safe_worker_policy import safe_worker_enabled
        from hermes_cli.config import load_config, read_raw_config
        from hermes_cli.config_effective import load_user_config_effective
        from hermes_cli.plugins import discover_plugins
        from tools.mcp_tool_config import _load_mcp_config
        discover_plugins()
        from tools import approval
        approval.save_permanent_allowlist({{'echo *', 'ls *'}})
        print(json.dumps({{'safe': safe_worker_enabled(), 'turns': [load_config()['agent']['max_turns'],
            read_raw_config()['agent']['max_turns'], load_user_config_effective()['agent']['max_turns']],
            'mcp': _load_mcp_config()['x']['command'],
            'plugin': os.path.exists(os.path.join(os.environ['HERMES_HOME'], 'plugin-executed'))}}))
    '''), encoding='utf-8')
    env = {**child_env(), **{k: os.environ[k] for k in ('TIRITH_ENABLED',) if k in os.environ}}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(root))
    result = subprocess.run([sys.executable, str(script)], cwd=root, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt == {'safe': False, 'turns': [7, 7, 7], 'mcp': 'FROZEN_A', 'plugin': True}, receipt
    from hermes_cli.config import fast_safe_load
    saved = fast_safe_load((home / 'config.yaml').read_text(encoding='utf-8'))
    assert saved['command_allowlist'] == ['echo *', 'ls *']
    assert saved['agent']['max_turns'] == 99 and saved['mcp_servers']['x']['command'] == 'LIVE_B', saved
    assert saved['display']['personality'] == 'EDITED_AFTER_CREATE', saved
