"""Authenticated WS fixture: a browser pick's ``staged_upload`` receipt attached on the shared owner."""
import asyncio
import io
import json
import os
from pathlib import Path
import sys
import traceback


async def probe():
    from gateway.run import GatewayRunner
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from gateway.session_authority import initialize_session_authority
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from hermes_cli.staged_uploads import publish_staged_upload, staged_upload_descriptor
    from tests.gateway.fixtures.local_recovery_probe import rpc
    from websockets.asyncio.client import connect

    home = Path(os.environ['HERMES_HOME'])
    (home / 'config.yaml').write_text('gateway:\n  multiplex_profiles: false\nmodel:\n  provider: custom\n  default: no-inference\n  base_url: http://127.0.0.1:1/v1\nauxiliary:\n  title_generation:\n    enabled: false\n')
    runner = GatewayRunner()
    await initialize_session_authority(runner, profile_id=str(home), instance_id='staged-attach')
    api = await start_gateway_api(runner)
    web_server.app.state.auth_required = True
    receipt = {}

    def staged(profile_home, payload):
        # Exactly the Webapp upload route: publish into the profile home, return its receipt.
        path = publish_staged_upload(io.BytesIO(payload), profile_home, 'notes.txt')
        return staged_upload_descriptor(path, profile_home)

    try:
        ticket = mint_ticket(user_id='owner', provider='fixture')
        async with connect(api.api_origin.replace('http:', 'ws:') + '/api/ws?ticket=' + ticket) as ws:
            sid = (await rpc(ws, 'session.create', request_id='desktop', source='gui', toolsets=[]))['result']['session_id']

            async def attach(descriptor):
                reply = await rpc(ws, 'file.attach', name='notes.txt', session_id=sid, staged_upload=descriptor)
                return reply.get('result') or reply.get('error', {}).get('message')

            own = staged(home, b'BROWSER PAYLOAD')
            attached = await attach(own)
            receipt['own'] = {k: attached[k] for k in ('attached', 'uploaded', 'ref_text')} \
                if isinstance(attached, dict) else attached
            receipt['own_bytes'] = Path(attached['path']).read_text() if isinstance(attached, dict) else None
            # Another backend's receipt, a path outside the profile's upload dir, and a profile this
            # owner does not serve: none is copied, whatever the descriptor claims.
            receipt['foreign_install'] = await attach(dict(staged(home, b'x'), install_id='another-install'))
            secret = home / 'secret.txt'
            secret.write_text('SECRET')
            receipt['outside_uploads'] = await attach(dict(own, path=str(secret)))
            stranger = home.parent / 'stranger-profile'
            stranger.mkdir()
            receipt['unserved_profile'] = await attach(staged(stranger, b'y'))
            copies = home / 'attachments'
            receipt['copied'] = sorted(p.read_text() for p in copies.iterdir()) if copies.is_dir() else []
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
