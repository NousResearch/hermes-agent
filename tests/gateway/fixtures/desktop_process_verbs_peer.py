"""Authenticated WS fixture: Desktop's exact process/MCP frames on the shared owner; no inference."""
import asyncio
import json
import os
from pathlib import Path
import shlex
import sys
import traceback


async def probe():
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tests.gateway.fixtures.local_recovery_probe import rpc
    from tools.process_registry import process_registry
    from websockets.asyncio.client import connect

    home = Path(os.environ['HERMES_HOME'])
    (home / 'config.yaml').write_text('gateway:\n  multiplex_profiles: false\nmodel:\n  provider: custom\n  default: no-inference\n  base_url: http://127.0.0.1:1/v1\nauxiliary:\n  title_generation:\n    enabled: false\n')
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='desktop-verbs')
    api = await start_gateway_api(runner)
    web_server.app.state.auth_required = True
    receipt, processes = {}, []
    command = shlex.quote(sys.executable) + ' -c ' + shlex.quote('import time; time.sleep(60)')
    try:
        ticket = mint_ticket(user_id='owner', provider='fixture')
        async with connect(api.api_origin.replace('http:', 'ws:') + '/api/ws?ticket=' + ticket) as ws:
            sid = (await rpc(ws, 'session.create', request_id='desktop', source='gui', toolsets=[]))['result']['session_id']
            route = authority.sessions[sid].route
            for task, key in [(sid, route), (sid, route), ('foreign', 'agent:main:telegram:dm:1')]:
                processes.append(process_registry.spawn_local(command, cwd=str(home), task_id=task, owner_task_id=task,
                                                              session_key=key))
            own, other, foreign = processes
            # composer-status.ts: per-process Stop.
            kill = await rpc(ws, 'process.kill', process_id=own.id, session_id=sid)
            await asyncio.to_thread(own._completion_event.wait, 10)
            receipt['kill'] = kill.get('result', {}).get('status', kill.get('error'))
            receipt['foreign_kill'] = (await rpc(ws, 'process.kill', process_id=foreign.id, session_id=sid)).get('error', {}).get('message')
            # slash.ts /stop: the session id session.interrupt resolved.
            stop = await rpc(ws, 'process.stop', session_id=sid)
            await asyncio.to_thread(other._completion_event.wait, 10)
            receipt['stop'] = stop.get('result', stop.get('error'))
            receipt['exited'] = [p.exited for p in processes]
            # use-mcp-servers.ts / mcp.ts / repair.ts.
            reload = await rpc(ws, 'reload.mcp', confirm=True, session_id=sid)
            receipt['reload_mcp'] = reload.get('result', {}).get('status', reload.get('error'))
    finally:
        for proc in processes:
            if not proc.exited:
                process_registry.kill_process(proc.id)
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
