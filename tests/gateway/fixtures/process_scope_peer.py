"""Authenticated WS fixture: process-global sidecar verbs on the shared owner; no inference."""
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
    from tools.delegate_tool import is_spawn_paused
    from tools.process_registry import process_registry
    from websockets.asyncio.client import connect

    home = Path(os.environ['HERMES_HOME'])
    (home / 'config.yaml').write_text('gateway:\n  multiplex_profiles: false\nmodel:\n  provider: custom\n  default: no-inference\n  base_url: http://127.0.0.1:1/v1\nauxiliary:\n  title_generation:\n    enabled: false\n')
    (home / '.env').write_text('PROCESS_SCOPE_PROBE=from-dotenv\n')
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='process-scope')
    api = await start_gateway_api(runner)
    web_server.app.state.auth_required = True
    receipt, processes = {}, []
    command = shlex.quote(sys.executable) + ' -c ' + shlex.quote('import time; time.sleep(60)')
    try:
        ticket = mint_ticket(user_id='owner', provider='fixture')
        async with connect(api.api_origin.replace('http:', 'ws:') + '/api/ws?ticket=' + ticket) as ws:
            sid = (await rpc(ws, 'session.create', request_id='scope', source='tui', toolsets=[]))['result']['session_id']
            route = authority.sessions[sid].route
            # This session's own job (persisted too), another chat's job and another chat's persisted job.
            for task, key, persist in [(sid, route, False), (sid, route, True),
                                       ('foreign', 'agent:main:telegram:dm:1', False),
                                       ('foreign-persist', 'agent:main:discord:dm:2', True)]:
                processes.append(process_registry.spawn_local(command, cwd=str(home), task_id=task, owner_task_id=task,
                                                              session_key=key, persist_on_release=persist))
            stop = await rpc(ws, 'process.stop', session_id=sid)
            for proc in processes[:2]:
                await asyncio.to_thread(proc._completion_event.wait, 10)
            receipt['stop'] = stop.get('result', stop.get('error'))
            receipt['exited'] = [p.exited for p in processes]
            receipt['sessionless_stop'] = (await rpc(ws, 'process.stop')).get('error', {}).get('code')
            receipt['exited_after_sessionless'] = [p.exited for p in processes]
            pause = await rpc(ws, 'delegation.pause', paused=True)
            receipt['pause'] = pause.get('error', {}).get('code')
            receipt['spawn_paused'] = is_spawn_paused()
            os.environ.pop('PROCESS_SCOPE_PROBE', None)
            receipt['reload_env'] = (await rpc(ws, 'reload.env')).get('error', {}).get('code')
            receipt['environ_rewritten'] = 'PROCESS_SCOPE_PROBE' in os.environ
            receipt['reload_mcp'] = (await rpc(ws, 'reload.mcp', session_id=sid, confirm=True)).get('error', {}).get('code')
            receipt['agents_list'] = (await rpc(ws, 'agents.list')).get('error', {}).get('code')
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
