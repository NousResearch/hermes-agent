"""Authenticated WS fixture: Ink ``!cmd`` on the shared owner; no inference."""
import asyncio
import json
import os
from pathlib import Path
import sys
import traceback


async def probe():
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tests.gateway.fixtures.local_recovery_probe import rpc
    from websockets.asyncio.client import connect

    home = Path(os.environ['HERMES_HOME'])
    work = home / 'session-checkout'
    work.mkdir()
    (home / 'config.yaml').write_text('gateway:\n  multiplex_profiles: false\nmodel:\n  provider: custom\n  default: no-inference\n  base_url: http://127.0.0.1:1/v1\nauxiliary:\n  title_generation:\n    enabled: false\n')
    # A credential the long-lived owner holds must never reach a user-typed child.
    os.environ['OPENROUTER_API_KEY'] = 'sk-or-v1-' + 'f' * 48
    runner = GatewayRunner()
    await initialize_session_authority(runner, profile_id=str(home), instance_id='shell')
    api = await start_gateway_api(runner)
    web_server.app.state.auth_required = True
    receipt = {'owner_cwd': os.getcwd(), 'session_cwd': str(work)}
    try:
        def socket(actor):
            return connect(api.api_origin.replace('http:', 'ws:') + '/api/ws?ticket='
                           + mint_ticket(user_id=actor, provider='fixture'))
        async with socket('owner') as ws:
            sid = (await rpc(ws, 'session.create', request_id='shell', source='tui', cwd=str(work),
                             toolsets=[]))['result']['session_id']
            ran = await rpc(ws, 'shell.exec', session_id=sid,
                            command='pwd; printf "key=%s" "${OPENROUTER_API_KEY:-}" >&2; exit 3')
            receipt['ran'] = ran.get('result', ran.get('error'))
            dangerous = await rpc(ws, 'shell.exec', session_id=sid, command='rm -rf /')
            receipt['dangerous'] = dangerous.get('error', dangerous.get('result'))
            receipt['sessionless'] = (await rpc(ws, 'shell.exec', command='pwd')).get('error', {}).get('code')
    finally:
        await stop_gateway_api(api)
    (home / 'receipt.json').write_text(json.dumps(receipt))


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
