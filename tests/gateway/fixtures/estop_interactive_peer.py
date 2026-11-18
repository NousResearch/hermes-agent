"""GW-1 fixture: `hermes pause` holds every gateway turn, including interactive (TUI/Desktop/CLI) ones.

A real GatewayRunner + session authority + WS API in a fake HERMES_HOME, a loopback model. Phase 1
starts a turn that blocks in the model, engages the ESTOP and releases it: the running turn must
finish normally. Phase 2 submits a new interactive turn while paused: its completion must carry
``estop.paused_reply()`` (the reason and "run `hermes resume`") and the model is never called.
Phase 3 resumes and the next turn runs. Writes receipt.json into HERMES_HOME.
"""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import traceback


async def probe():
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tests.gateway.fixtures.local_recovery_probe import Model
    from agent import estop
    from websockets.asyncio.client import connect

    home = Path(os.environ['HERMES_HOME'])
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests, peer.blocked, peer.release = [], threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    (home / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'custom', 'default': 'loopback', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}))
    runner = GatewayRunner()
    await initialize_session_authority(runner, profile_id=str(home), instance_id='estop-probe')
    api = await start_gateway_api(runner)
    web_server.app.state.auth_required = True
    receipt = {}
    try:
        ticket = mint_ticket(user_id='owner', provider='fixture')
        async with connect(api.api_origin.replace('http:', 'ws:') + '/api/ws?ticket=' + ticket) as ws:
            pending = {}
            frames = asyncio.Queue()

            async def pump():
                async for raw in ws:
                    frame = json.loads(raw)
                    if 'id' in frame and frame['id'] in pending:
                        pending.pop(frame['id']).set_result(frame)
                    elif frame.get('method') == 'event':
                        await frames.put(frame['params'])
            pumper = asyncio.create_task(pump())

            async def rpc(method, **params):
                rid = f'{method}-{len(pending)}-{asyncio.get_running_loop().time()}'
                pending[rid] = asyncio.get_running_loop().create_future()
                await ws.send(json.dumps({'jsonrpc': '2.0', 'id': rid, 'method': method, 'params': params}))
                return await asyncio.wait_for(pending[rid], 30)

            async def complete():
                while True:
                    event = await asyncio.wait_for(frames.get(), 60)
                    if event.get('type') == 'message.complete':
                        return event.get('payload') or {}

            sid = (await rpc('session.create', request_id='estop', source='tui', cwd=str(home),
                             toolsets=[]))['result']['session_id']
            # Phase 1: a turn already running when the pause lands finishes.
            await rpc('prompt.submit', session_id=sid, input_id='running', text='BLOCK_STARTED')
            assert await asyncio.to_thread(peer.blocked.wait, 30), 'model never reached'
            estop.engage(reason='maintenance window')
            peer.release.set()
            receipt['running'] = await complete()
            # Phase 2: a new interactive turn while paused gets the named refusal; no model call.
            calls = len(peer.requests)
            receipt['paused_submit'] = (await rpc('prompt.submit', session_id=sid, input_id='paused',
                                                  text='NEW_WHILE_PAUSED')).get('result')
            receipt['paused'] = await complete()
            receipt['paused_model_calls'] = len(peer.requests) - calls
            receipt['expected_notice'] = estop.paused_reply()
            # Phase 3: resume; the next turn runs.
            estop.disengage()
            await rpc('prompt.submit', session_id=sid, input_id='after', text='AFTER_RESUME')
            receipt['after'] = await complete()
            pumper.cancel()
    finally:
        await stop_gateway_api(api)
        peer.shutdown()
    (home / 'receipt.json').write_text(json.dumps(receipt, default=str))


if __name__ == '__main__':
    status = 0
    try:
        asyncio.run(probe())
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
