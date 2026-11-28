"""Hold the old owner on either side of the finish commit; capture the real request."""

import asyncio
import json
import os
from pathlib import Path
import runpy

from gateway import session_worker


request = session_worker.worker_request
home = Path(os.environ['HERMES_HOME'])
is_successor = (home / 'before-finish.json').exists()


async def hold_finish(connection, ref, params, *, operation):
    if operation != 'persist' or params.get('operation') != 'execution.finish':
        return await request(connection, ref, params, operation=operation)
    identity = {k: params[k] for k in ('execution_id', 'sequence', 'epoch', 'pid')}
    if not is_successor:
        if (home / 'commit-before-loss').exists():
            real = await request(connection, ref, params, operation=operation)
            assert real == {'status': 'terminal'}, real
        (home / 'before-finish.tmp').write_text(json.dumps(identity))
        (home / 'before-finish.tmp').replace(home / 'before-finish.json')
        # The test replaces this owner before the worker receives its RPC reply.
        await asyncio.Future()
    result = await request(connection, ref, params, operation=operation)
    (home / 'after-finish.json').write_text(json.dumps(identity))
    return result


if __name__ == '__main__':
    session_worker.worker_request = hold_finish
    runpy.run_module('gateway.run', run_name='__main__')
