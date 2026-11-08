"""Public ACP -> durable admission -> real cached agent/tool loop -> loopback provider."""
import asyncio
from contextlib import suppress
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from unittest.mock import AsyncMock

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, websocket


class EffortModel(Model):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if 'messages' not in body:
            self.send_error(404)
            return
        self.server.requests.append(body)
        user = next(m['content'] for m in reversed(body['messages']) if m['role'] == 'user')
        if body['messages'][-1]['role'] == 'tool':
            message, finish = {'role': 'assistant', 'content': 'done ' + user}, 'stop'
        else:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'read-' + str(len(self.server.requests)), 'type': 'function', 'function': {
                    'name': 'read_file', 'arguments': json.dumps({'path': str(self.server.target)})}}]}
            finish = 'tool_calls'
        payload = {'id': 'effort', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                   'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        if body.get('stream'):
            delta = dict(message)
            if 'tool_calls' in delta:
                delta['tool_calls'] = [dict(delta['tool_calls'][0], index=0)]
            payload['choices'][0] = {'index': 0, 'delta': delta, 'finish_reason': finish}
            raw, kind = ('data: ' + json.dumps(payload) + '\n\ndata: [DONE]\n\n').encode(), 'text/event-stream'
        else:
            raw, kind = json.dumps(payload).encode(), 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.mark.asyncio
async def test_public_acp_pin_routes_whole_cached_turn_and_keeps_picker_none(tmp_path, monkeypatch):
    from acp.schema import TextContent
    from acp_adapter.gateway_server import GatewayACPAgent
    from hermes_cli.gateway_client import GatewayClient
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    plugin = home / 'plugins' / 'turn-effort'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text('name: turn-effort\nversion: 1.0.0\ndescription: Offline test policy\nprovides_hooks: [select_turn_reasoning, on_session_start]\n')
    (plugin / '__init__.py').write_text('''
import json
from pathlib import Path
from hermes_constants import get_hermes_home

def select(**p):
    path = get_hermes_home() / 'selected.jsonl'
    with path.open('a') as f:
        f.write(json.dumps(p) + '\\n')
    if p['reasoning']['pinned'] and p['reasoning']['raw_effort'] == 'none':
        return 'high' if 'hard' in p['user_message'] else 'low'

def started(**p):
    with (get_hermes_home() / 'started.jsonl').open('a') as f:
        f.write(json.dumps(p) + '\\n')

def register(ctx):
    ctx.register_hook('select_turn_reasoning', select)
    ctx.register_hook('on_session_start', started)
''')
    peer = ThreadingHTTPServer(('127.0.0.1', 0), EffortModel)
    peer.requests, peer.target = [], home / 'fixture.txt'
    peer.target.write_text('REAL_TOOL_CONTENT')
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    config = {'gateway': {'multiplex_profiles': False},
              'model': {'provider': 'custom', 'default': 'wire-effort', 'base_url': url, 'api_mode': 'chat_completions'},
              'agent': {'reasoning_effort': 'none'}, 'platform_toolsets': {'acp': ['file']},
              'model_catalog': {'enabled': False}, 'auxiliary': {'title_generation': {'enabled': False}},
              'plugins': {'enabled': ['turn-effort']}}
    (home / 'config.yaml').write_text(json.dumps(config))
    env = child_env()
    root = Path(__file__).resolve().parents[2]
    env.update(HOME=str(user), HERMES_HOME=str(home), TMPDIR=str(tmp_path), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    monkeypatch.setenv('HERMES_HOME', str(home))
    try:
        with daemon(root, home, env, barrier=False) as (_, descriptor):
            async with websocket(home, descriptor) as ws, GatewayClient(ws) as client:
                acp = GatewayACPAgent()
                acp._conn = AsyncMock()
                acp._gateway = client
                async def connected():
                    return client
                monkeypatch.setattr(acp, '_client', connected)
                pump = asyncio.create_task(acp._events())
                try:
                    sid = (await acp.new_session(str(home), [])).session_id
                    await acp.set_config_option('reasoning_effort', sid, 'none')
                    for text in ('easy one', 'hard work', 'easy again'):
                        response = await acp.prompt([TextContent(type='text', text=text)], sid)
                        assert response.stop_reason == 'end_turn'
                        info = await client.rpc('session.info', session_id=sid)
                        assert info['reasoning_effort'] == 'none'
                    efforts = [r.get('reasoning_effort') or r.get('reasoning', {}).get('effort') for r in peer.requests]
                    assert efforts == ['low', 'low', 'high', 'high', 'low', 'low']
                    selected = [json.loads(line) for line in (home / 'selected.jsonl').read_text().splitlines()]
                    assert [p['reasoning']['raw_effort'] for p in selected] == ['none'] * 3
                    assert len({p['turn_id'] for p in selected}) == 3
                    assert len((home / 'started.jsonl').read_text().splitlines()) == 1
                    assert all('REAL_TOOL_CONTENT' in str(peer.requests[i]['messages']) for i in (1, 3, 5))
                    updates = [c.kwargs.get('update') for c in acp._conn.session_update.await_args_list]
                    assert 'high' not in [getattr(o, 'current_value', '') for u in updates
                                          for o in getattr(u, 'config_options', [])]
                    await acp.load_session(str(home), sid, [])
                    await acp.prompt([TextContent(type='text', text='easy resumed')], sid)
                    await acp.set_config_option('reasoning_effort', sid, 'default')
                    await acp.prompt([TextContent(type='text', text='hard after reset')], sid)
                    selected = [json.loads(line) for line in (home / 'selected.jsonl').read_text().splitlines()]
                    assert len(selected) == 5
                    assert selected[3]['reasoning']['pinned'] is True
                    assert selected[4]['reasoning']['pinned'] is False
                    assert selected[4]['reasoning']['source'] == 'default'
                    assert peer.requests[-3]['reasoning_effort'] == 'low'
                    assert peer.requests[-1]['reasoning_effort'] == 'none'
                finally:
                    pump.cancel()
                    with suppress(asyncio.CancelledError):
                        await pump
                    for task in list(acp._background):
                        task.cancel()
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
