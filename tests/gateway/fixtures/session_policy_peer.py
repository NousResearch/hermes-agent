"""Real concurrent source/model/tool/cwd launch proof; no external provider."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import traceback


class Peer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(body)
        messages = body.get('messages', [])
        done = any(m.get('role') == 'tool' for m in messages)
        if messages and not done:
            self.server.first_round.wait(timeout=20)
        message = {'role': 'assistant', 'content': 'POLICY_DONE'}
        if not done:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'owned', 'type': 'function', 'function': {'name': 'terminal',
                 'arguments': json.dumps({'command': 'pwd; printf owned > policy-proof.txt'})}}]}
        payload = json.dumps({'id': 'policy', 'object': 'chat.completion', 'model': body.get('model'),
                              'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop' if done else 'tool_calls'}],
                              'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}}).encode()
        content_type = 'application/json'
        if body.get('stream'):
            delta = dict(message)
            if delta.get('tool_calls'):
                delta['tool_calls'][0]['index'] = 0
            chunk = {'id': 'policy', 'object': 'chat.completion.chunk', 'model': body.get('model'),
                     'choices': [{'index': 0, 'delta': delta, 'finish_reason': 'stop' if done else 'tool_calls'}]}
            payload = ('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode()
            content_type = 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


async def probe(peer):
    import websockets
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from gateway.session_contract import SessionRef
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='policy')
    api = await start_gateway_api(runner)
    web_server.app.state.auth_required = True
    async def connect():
        ticket = mint_ticket(user_id='creator', provider='fixture')
        return await websockets.connect(f'ws://127.0.0.1:{api.socket.getsockname()[1]}/api/ws?ticket={ticket}')
    async def rpc(ws, method, **params):
        await ws.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
        async with asyncio.timeout(20):
            while True:
                result = json.loads(await ws.recv())
                if result.get('id') == method:
                    return result
    sockets, sessions = [], []
    before = dict(os.environ)
    config_before = Path(os.environ['HERMES_HOME'], 'config.yaml').read_bytes()
    try:
        for source in ('cli', 'tui', 'gui'):
            ws = await connect()
            sockets.append(ws)
            cwd = Path(os.environ['HERMES_HOME'], source)
            cwd.mkdir()
            params = dict(request_id=source, source=source, cwd=str(cwd), model='policy-' + source)
            if source == 'cli':
                params['toolsets'] = ['terminal']
            created = await rpc(ws, 'session.create', **params)
            assert 'result' in created, created
            sid = created['result']['session_id']
            sessions.append(sid)
            from gateway.session_policy import policy_for_source
            from hermes_state_runtime import RuntimeStoreError
            live_source = authority.sessions[sid].source
            adapter = runner._adapter_for_source(live_source)
            saved_policy = adapter.policies.pop(live_source.chat_id)
            try:
                policy_for_source(runner, live_source)
            except RuntimeStoreError as exc:
                assert exc.reason == 'storage_unavailable'
            else:
                raise AssertionError('missing recovered policy silently selected CLI')
            finally:
                adapter.policies[live_source.chat_id] = saved_policy
            again = await rpc(ws, 'session.create', **params)
            assert again['result']['session_id'] == sid
            conflict = await rpc(ws, 'session.create', **{**params, 'model': 'conflict'})
            assert 'error' in conflict, conflict
            accepted = await rpc(ws, 'prompt.submit', session_id=sid, input_id=source, text='Write owned cwd proof')
            assert 'result' in accepted, accepted
        for sid in sessions:
            async with asyncio.timeout(30):
                while authority.sessions[sid].task is None:
                    await asyncio.sleep(.01)
            await asyncio.wait_for(authority.sessions[sid].task, 40)
        for source, sid, ws in zip(('cli', 'tui', 'gui'), sessions, sockets):
            agent = authority.agent(SessionRef(authority.profile_id, sid))
            assert agent is not None, (source, list(runner._agent_cache), peer.requests, await rpc(ws, 'session.resume', session_id=sid))
            assert agent.platform == {'cli': 'cli', 'tui': 'tui', 'gui': 'desktop'}[source], vars(agent).get('platform')
            cwd = Path(os.environ['HERMES_HOME'], source)
            assert (cwd / 'policy-proof.txt').read_text() == 'owned'
            requests = [r for r in peer.requests if r.get('model') == 'policy-' + source]
            assert len(requests) >= 2, peer.requests
            assert any(str(cwd) in json.dumps(m) for r in requests for m in r['messages'] if m['role'] == 'tool')
            names = {t['function']['name'] for t in requests[0]['tools']}
            assert 'terminal' in names
            assert ('desktop_ui' in agent.enabled_toolsets) == (source == 'gui')
            from toolsets import resolve_toolset
            desktop_names = resolve_toolset('desktop_ui')
            surface_on_wire = any(name in json.dumps(requests[0]['tools']) for name in desktop_names)
            assert surface_on_wire == (source == 'gui'), (source, desktop_names, names)
            await ws.close()
            reconnected = await connect()
            sockets.append(reconnected)
            result = await rpc(reconnected, 'session.resume', session_id=sid)
            assert result['result']['session_id'] == sid
            assert authority.agent(SessionRef(authority.profile_id, sid)) is agent
        assert Path(os.environ['HERMES_HOME'], 'config.yaml').read_bytes() == config_before
        assert {k: os.environ.get(k) for k in ('HERMES_DESKTOP', 'TERMINAL_CWD', 'HERMES_TUI_TOOLSETS')} == {k: before.get(k) for k in ('HERMES_DESKTOP', 'TERMINAL_CWD', 'HERMES_TUI_TOOLSETS')}
        Path(os.environ['HERMES_HOME'], 'policy-receipt.json').write_text(json.dumps({'sources': ['cli', 'tui', 'gui'], 'requests': len(peer.requests), 'cwd_effects': True, 'same_agents': True, 'no_spill': True}))
    finally:
        for ws in sockets:
            await ws.close()
        await stop_gateway_api(api)


def main():
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    peer.requests = []
    peer.first_round = threading.Barrier(3)
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
        f'model:\n  default: policy-default\n  provider: custom\n  base_url: {url}\n'
        'terminal:\n  env_type: local\nstreaming:\n  enabled: false\n'
        'platform_toolsets:\n  cli: [terminal]\nauxiliary:\n  title_generation:\n    enabled: false\n')
    try:
        asyncio.run(probe(peer))
    finally:
        peer.shutdown()
        peer.server_close()


if __name__ == '__main__':
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        status = 1
    os._exit(status)
