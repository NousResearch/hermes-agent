"""Dispatcher subprocess: submit one claim to its profile owner; never run an agent."""
import asyncio
import os
from pathlib import Path
import sys


async def run(params, board_db):
    from hermes_cli.gateway_client import connect_gateway, GatewayClientError
    if os.environ.get('HERMES_TUI_GATEWAY_URL'):
        raise GatewayClientError('Kanban requires the assigned local profile owner')
    async with connect_gateway() as client:
        accepted = await client.rpc('kanban.run', **dict(params, db=str(Path(board_db).resolve())))
        sid, receipt = accepted['session_id'], accepted['receipt']
        print(f'Session: {sid}', file=sys.stderr)
        while receipt['status'] not in {'terminal', 'unknown'}:
            await asyncio.sleep(.2)
            receipt = await client.rpc('prompt.receipt', session_id=sid, admission_id=receipt['admission_id'])
        if receipt['status'] == 'unknown':
            raise GatewayClientError('Kanban execution unknown; retained by owner')
        from gateway.session_kanban import worker_exit_code
        return worker_exit_code(board_db, params)


def main():
    try:
        params = {'board': os.environ['HERMES_KANBAN_BOARD'], 'task_id': os.environ['HERMES_KANBAN_TASK'],
            'run_id': int(os.environ['HERMES_KANBAN_RUN_ID']), 'claim_lock': os.environ['HERMES_KANBAN_CLAIM_LOCK']}
        board_db = os.environ['HERMES_KANBAN_DB']
        # Owner startup must never inherit this worker's task/tool/goal identity.
        for key in list(os.environ):
            if key.startswith('HERMES_KANBAN_') and key != 'HERMES_KANBAN_HOME':
                os.environ.pop(key)
        for key in ('HERMES_SESSION_SOURCE', 'HERMES_TENANT', 'TERMINAL_CWD'):
            os.environ.pop(key, None)
        return asyncio.run(run(params, board_db))
    except (KeyError, ValueError, OSError) as exc:
        print(f'Kanban owner refused: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
