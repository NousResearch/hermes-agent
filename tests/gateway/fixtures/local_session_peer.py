"""Owned loopback fixture: no native credentials or alternative agent runtime."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import sys
import threading
import traceback

from authority_controls_peer import ModelPeer


async def probe(peer, target, kind):
    import websockets
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionRef
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from hermes_cli import web_server
    from run_agent import AIAgent
    from tools.approval import resolve_gateway_approval

    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='local')
    api = await start_gateway_api(runner)
    port = api.socket.getsockname()[1]
    web_server.app.state.auth_required = True
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    sid = None
    sockets = []

    async def connect():
        ticket = mint_ticket(user_id='fixture-creator', provider='local-fixture')
        ws = await websockets.connect(f'ws://127.0.0.1:{port}/api/ws?ticket={ticket}')
        sockets.append(ws)
        return ws

    async def rpc(ws, method, **params):
        await ws.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
        async with asyncio.timeout(15):
            while True:
                frame = json.loads(await ws.recv())
                if frame.get('id') == method:
                    return frame

    try:
        a = await connect()
        for payload in ({'source': 'telegram'}, {'source': 'cron'}, {'internal': True},
                        {'profile': 'foreign'}, {'cwd': 'relative'}, {'yolo': True}):
            denied = await rpc(a, 'session.create', **payload)
            assert 'error' in denied, denied
        before_sessions = set(authority.sessions)
        runner._draining = True
        try:
            draining = await rpc(a, 'session.create', request_id='during-drain', source='cli')
            assert draining.get('error', {}).get('message') == 'runtime_draining', draining
            assert set(authority.sessions) == before_sessions
        finally:
            runner._draining = False
        created = await rpc(a, 'session.create', request_id='fresh', source='cli')
        assert 'result' in created, created
        sid = created['result']['session_id']
        assert sid == created['result']['stored_session_id']
        listed = await rpc(a, 'session.list', limit=10)
        assert any(row['session_id'] == sid for row in listed['result']['sessions']), listed
        info = await rpc(a, 'session.info', session_id=sid)
        assert info['result']['source'] == 'cli', info
        assert info['result']['lazy'] is True, info
        assert authority.db.get_session(sid) is not None
        # Private route identity cannot be reconstructed from its serialized source,
        # even under an unrelated messaging allow-all policy.
        from dataclasses import replace
        source = authority.sessions[sid].source
        assert runner._is_user_authorized_for_source(source)
        os.environ['GATEWAY_ALLOW_ALL_USERS'] = 'true'
        try:
            assert not runner._is_user_authorized_for_source(replace(source, role_authorized=True))
            assert not runner._is_user_authorized_for_source(replace(source, profile='foreign'))
        finally:
            os.environ.pop('GATEWAY_ALLOW_ALL_USERS')
        from gateway.session_local import create_local_session
        from gateway.session_contract import Principal
        from hermes_state_runtime import RuntimeStoreError
        for actor, reason in ((Principal('read-only', 'fixture', frozenset({'session:read'}), 'readonly'),
                               'permission_denied'),
                              (Principal('creator', 'foreign', frozenset({'session:create'}), 'foreign'),
                               'profile_mismatch')):
            try:
                create_local_session(authority, actor, {'api_key': 'denied-launch-key',
                    'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1', 'ignore_rules': True})
            except RuntimeStoreError as exc:
                assert exc.reason == reason
            else:
                raise AssertionError('restricted principal created a local route')
        repeated = await rpc(a, 'session.create', request_id='fresh', source='cli')
        assert repeated['result']['session_id'] == sid, repeated
        b = await connect()
        resumed = await rpc(b, 'session.resume', session_id=sid)
        assert resumed['result']['session_id'] == sid, resumed
        submitted = await rpc(a, 'prompt.submit', session_id=sid, input_id='first', text='Remove owned fixture')
        assert submitted.get('result', {}).get('status') == 'queued', submitted
        async with asyncio.timeout(25):
            while True:
                snapshot = await rpc(b, 'session.resume', session_id=sid)
                if snapshot['result'].get('prompts'):
                    break
                await asyncio.sleep(.03)
        assert target.exists(), 'effect preceded consent'
        ref = SessionRef(authority.profile_id, sid)
        agent = authority.agent(ref)
        assert isinstance(agent, AIAgent)
        prompt = snapshot['result']['prompts'][0]
        assert prompt['kind'] == kind, prompt
        await a.close()
        await b.close()
        assert target.exists()
        c = await connect()
        snapshot = await rpc(c, 'session.resume', session_id=sid)
        assert snapshot['result']['prompts'][0]['prompt_id'] == prompt['prompt_id']
        assert authority.agent(ref) is agent
        response = {'choice': 'once'} if kind == 'approval' else {'answer': 'green'}
        answer = await rpc(c, kind + '.respond', session_id=sid,
                           prompt_id=prompt['prompt_id'], execution_generation=prompt['execution_generation'], **response)
        assert answer.get('result', {}).get('status') == 'resolved', answer
        await asyncio.wait_for(authority.sessions[sid].task, 25)
        if kind == 'approval':
            assert not target.exists(), 'real owned terminal effect missing'
        else:
            assert any('green' in json.dumps(m) for messages in peer.requests
                       for m in messages if m['role'] == 'tool'), peer.requests
        final = await rpc(c, 'session.resume', session_id=sid)
        assert final['result']['session_id'] == sid
        assert any(m.get('content') == kind.upper() + '_FINISHED' for m in final['result']['messages']), final
        assert authority.agent(ref) is agent
        pong = await rpc(c, 'ping')
        assert pong.get('result', {}).get('pong') is True, pong
        described = await rpc(c, 'runtime.describe')
        assert described['result']['instance_id'] == authority.instance_id
        assert str(Path(os.environ['HERMES_HOME'])) not in json.dumps(described)
        # Public transport auth remains mandatory, even for metadata or fresh sessions.
        try:
            async with websockets.connect(f'ws://127.0.0.1:{port}/api/ws?token=forged') as bad:
                await rpc(bad, 'session.create', request_id='forged')
        except (websockets.exceptions.InvalidStatus, websockets.exceptions.ConnectionClosed):
            pass
        else:
            raise AssertionError('unauthenticated socket accepted')
        Path(os.environ['HERMES_HOME'], 'receipt.json').write_text(json.dumps({
            'terminal_effect': not target.exists(), 'human_response': True,
            'same_agent': True, 'detached_pending': True,
            'negative_controls': True, 'reconnected_identity': sid, 'model_requests': len(peer.requests)}))
    finally:
        if sid is not None:
            resolve_gateway_approval(authority.sessions[sid].route, 'deny', resolve_all=True)
            from tools import clarify_gateway
            clarify_gateway.clear_session(authority.sessions[sid].route)
            task = authority.sessions[sid].task
            if task is not None:
                await asyncio.wait_for(task, 30)
        for ws in sockets:
            await ws.close()
        await stop_gateway_api(api)


def main():
    target = Path(os.environ['HERMES_HOME'], 'owned-removal')
    target.mkdir()
    (target / 'owned.txt').write_text('disposable')
    kind = sys.argv[1] if len(sys.argv) > 1 else 'approval'
    from authority_clarify_peer import ModelPeer as ClarifyPeer
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ModelPeer if kind == 'approval' else ClarifyPeer)
    peer.requests = []
    peer.command = 'rm -r -- ' + shlex.quote(str(target))
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, TERMINAL_ENV='local')
    Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
        f'model:\n  default: local-control\n  provider: custom\n  base_url: {url}\n'
        'approvals:\n  mode: manual\n  timeout: 45\n'
        'streaming:\n  enabled: false\n'
        'auxiliary:\n  title_generation:\n    enabled: false\n')
    try:
        asyncio.run(probe(peer, target, kind))
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
