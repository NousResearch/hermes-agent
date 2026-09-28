"""Real HTTP/WS API execution probe with a loopback inference peer."""
import asyncio
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from http.server import ThreadingHTTPServer

from shared_authority_peer import ModelPeer


async def probe(peer):
    import aiohttp
    import websockets
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.run import GatewayRunner
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionRef
    from hermes_cli import web_server
    from hermes_state_runtime import list_session_admissions

    runner = GatewayRunner()
    await initialize_session_authority(runner, profile_id='default', instance_id='api-peer')
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'api-probe-owned-secret', 'port': 0}))
    adapter.gateway_runner = runner
    adapter.set_message_handler(runner._handle_message)
    runner.adapters[Platform.API_SERVER] = adapter
    assert await adapter.connect()
    api_port = adapter._site._server.sockets[0].getsockname()[1]
    handle = await start_gateway_api(runner)
    frames = []
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=35)) as client:
            url = f'http://127.0.0.1:{api_port}/v1/chat/completions'
            headers = {'Authorization': 'Bearer api-probe-owned-secret', 'X-Hermes-Session-Id': 'api-shared'}
            body = {'messages': [{'role': 'system', 'content': 'API_STABLE_SYSTEM'}, {'role': 'user', 'content': 'MESSAGING_WARM'}]}
            async with client.post(url, json=body) as denied:
                assert denied.status == 401
            async with client.post(url, json=body, headers=headers) as response:
                data = await response.json()
                assert response.status == 200, data
            authority = runner.session_authority
            ref = SessionRef('default', 'api-shared')
            assert ref.session_id in authority.sessions, 'API bypassed canonical authority'
            agent = authority.agent(ref)
            assert agent is not None
            port = handle.socket.getsockname()[1]
            async with websockets.connect(f'ws://127.0.0.1:{port}/api/ws?token={web_server._SESSION_TOKEN}') as ws:
                async def until(predicate):
                    async with asyncio.timeout(30):
                        while True:
                            frame = json.loads(await ws.recv())
                            frames.append(frame)
                            if predicate(frame):
                                return frame
                await until(lambda f: f.get('params', {}).get('type') == 'gateway.ready')
                await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'session.resume', 'params': {'session_id': ref.session_id}}))
                resumed = await until(lambda f: f.get('id') == 1)
                assert 'result' in resumed, resumed
                await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 2, 'method': 'prompt.submit', 'params': {'session_id': ref.session_id, 'input_id': 'ws-shared', 'text': 'WS_SHARED'}}))
                submitted = await until(lambda f: f.get('id') == 2)
                assert 'result' in submitted, submitted
                final = await until(lambda f: f.get('params', {}).get('type') == 'message.complete')
                assert 'LOCAL_ACK_WS_SHARED' in json.dumps(final), final
                assert authority.agent(ref) is agent, 'WS replaced the selected API agent'
                rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
                assert len(rows) == 2 and all(r['status'] == 'terminal' for r in rows), rows
                systems = [[m for m in req['messages'] if m['role'] in ('system', 'developer')] for req in peer.requests]
                assert systems[0] == systems[1] and 'API_STABLE_SYSTEM' in json.dumps(systems[0]), systems
                if os.environ.get('API_RUN'):
                    async with client.post(f'http://127.0.0.1:{api_port}/v1/runs',
                            json={'session_id': ref.session_id, 'input': 'RUN_ON_SHARED', 'instructions': 'API_STABLE_SYSTEM'},
                            headers={**headers, 'Idempotency-Key': 'run-shared'}) as response:
                        accepted = await response.json()
                        assert response.status == 202, accepted
                    run_id = accepted['run_id']
                    admitted = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
                    assert any(row['request_id'] == run_id for row in admitted), 'HTTP 202 preceded canonical admission'
                    async with asyncio.timeout(20):
                        while True:
                            async with client.get(f'http://127.0.0.1:{api_port}/v1/runs/{run_id}', headers=headers) as response:
                                status = await response.json()
                            if status.get('status') in ('completed', 'failed', 'cancelled'):
                                break
                            await asyncio.sleep(.02)
                    assert status['status'] == 'completed', status
                    assert authority.agent(ref) is agent
                    admitted = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
                    assert len(admitted) == 3 and all(row['status'] == 'terminal' for row in admitted), admitted
                if os.environ.get('API_RUN') == 'controls':
                    async def create_run(text):
                        async with client.post(f'http://127.0.0.1:{api_port}/v1/runs', json={
                            'session_id': ref.session_id, 'input': text, 'instructions': 'API_STABLE_SYSTEM'}, headers=headers) as response:
                            assert response.status == 202, await response.text()
                            return (await response.json())['run_id']
                    blocking = await create_run('BLOCK_FIFO')
                    assert await asyncio.to_thread(peer.blocked.wait, 10)
                    async with client.post(f'http://127.0.0.1:{api_port}/v1/runs/{blocking}/steer',
                            json={'input': 'SAFE_STEER'}, headers=headers) as response:
                        steered = await response.json()
                        assert response.status == 200 and steered['accepted'], steered
                    queued = await create_run('CANCEL_ONLY_QUEUED')
                    async with client.post(f'http://127.0.0.1:{api_port}/v1/runs/{queued}/stop', headers=headers) as response:
                        stopped = await response.json()
                        assert response.status == 200 and stopped['status'] == 'cancelled', stopped
                    pending_rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
                    assert next(r for r in pending_rows if r['request_id'] == queued)['outcome'] == 'cancelled'
                    assert next(r for r in pending_rows if r['request_id'] == blocking)['status'] == 'started'
                    peer.release.set()
                if os.environ.get('API_ADVANCED'):
                    await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 3, 'method': 'prompt.submit', 'params': {'session_id': ref.session_id, 'input_id': 'ws-block', 'text': 'BLOCK_FIFO'}}))
                    await until(lambda f: f.get('id') == 3)
                    assert await asyncio.to_thread(peer.blocked.wait, 10)
                    retry_headers = {**headers, 'Idempotency-Key': 'durable-api-retry'}
                    follow = {'messages': [{'role': 'system', 'content': 'API_STABLE_SYSTEM'}, {'role': 'user', 'content': 'FIFO_SECOND'}]}
                    async def request_follow(stream=False):
                        async with client.post(url, json={**follow, 'stream': stream}, headers=retry_headers) as response:
                            text = await response.text()
                            assert response.status == 200, text
                            return text
                    pending = asyncio.create_task(request_follow())
                    async with asyncio.timeout(10):
                        while not any(r['payload'].get('text') == 'FIFO_SECOND' for r in list_session_admissions(authority.db, session_id=ref.session_id)):
                            await asyncio.sleep(.01)
                    peer.release.set()
                    first_reply = json.loads(await pending)
                    assert first_reply['choices'][0]['message']['content']
                    count = len(peer.requests)
                    await request_follow()
                    streamed = await request_follow(stream=True)
                    assert 'data: [DONE]' in streamed and 'LOCAL_ACK' in streamed, streamed
                    assert len(peer.requests) == count, 'retry reexecuted canonical admission'
                    assert 'BLOCK_FIFO' in json.dumps(peer.requests[-1]['messages']), 'queued API used stale admission-time history'
                    assert authority.agent(ref) is agent
                Path(os.environ['HERMES_HOME'], 'receipt.json').write_text(json.dumps({'same_agent': True, 'rows': rows, 'http': data, 'frames': frames}))
    finally:
        await stop_gateway_api(handle)
        await adapter.disconnect()


if __name__ == '__main__':
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ModelPeer)
    peer.requests, peer.metadata_requests = [], []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base)
    Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
        f'model:\n  default: local-wire-stub\n  provider: custom\n  base_url: {base}\n'
        f'auxiliary:\n  title_generation:\n    enabled: false\nterminal:\n  cwd: {os.environ["HERMES_HOME"]}\n')
    code = 0
    try:
        asyncio.run(probe(peer))
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        peer.shutdown()
        peer.server_close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
