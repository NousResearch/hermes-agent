"""Deliver an accepted Kanban Stop through the existing authenticated owner protocol."""
import asyncio
from contextlib import closing
import json
import logging
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)


def _bindings(conn, task_id, event_id):
    return [{**json.loads(row[1]), 'run_id': row[0]} for row in conn.execute(
        "SELECT run_id,payload FROM task_events WHERE task_id=? AND kind='owner_admitted' AND id<? ORDER BY id",
        (task_id, event_id))]


def _checked_session(home, task_id, board_db, binding):
    from hermes_state_local import POLICY_PREFIX
    path = home / 'state.db'
    if Path(binding['db']).resolve() != path.resolve():
        raise RuntimeError('previous task owner belongs to another profile')
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
        conn.execute('PRAGMA query_only=ON')
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + binding['session_id'],)).fetchone()
    saved = json.loads(row[0]) if row else {}
    policy = saved.get('policy') or {}
    context = json.loads(policy.get('kanban_json') or '{}')
    if (Path(saved.get('profile_id', '')).resolve() != home.resolve() or policy.get('source') != 'kanban'
            or context.get('task_id') != task_id or context.get('run_id') != binding['run_id']
            or Path(context.get('db', '')).resolve() != board_db.resolve()):
        raise RuntimeError('previous task execution identity is unverified')
    return binding['session_id']


async def _deliver(home, task_id, board_db, bindings):
    from hermes_cli.gateway_client import connect_gateway
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    found = await asyncio.to_thread(discover_gateway_endpoint, home)
    if found.state != 'ready' or found.endpoint is None:
        raise RuntimeError('task blocked; process Stop not confirmed: native owner unavailable')
    async with connect_gateway(endpoint=found.endpoint) as client:
        for binding in bindings:
            sid = await asyncio.to_thread(_checked_session, home, task_id, board_db, binding)
            snapshot = await client.rpc('session.resume', session_id=sid)
            # Each native attempt has its own session. Only attempts recorded before
            # this accepted Stop are targets; a successor's session is never included.
            await client.rpc('session.interrupt', session_id=sid,
                             execution_generation=snapshot['execution_generation'])


def request_native_stop(conn, task_id, event_id):
    bindings = _bindings(conn, task_id, event_id)
    if not bindings:
        return
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import gateway_profile_home
    home = Path(gateway_profile_home(get_hermes_home()))
    board_db = Path(next(row[2] for row in conn.execute('PRAGMA database_list') if row[1] == 'main'))
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_deliver(home, task_id, board_db, bindings))
        return
    # Never block the owner loop on a call back into itself. The committed Stop
    # event and predecessor admission check preserve uncertainty if delivery fails.
    task = loop.create_task(_deliver(home, task_id, board_db, bindings))
    def completed(done):
        if not done.cancelled() and (error := done.exception()) is not None:
            logger.warning('Task %s is blocked; native process Stop is unconfirmed (%s)', task_id, type(error).__name__)
    task.add_done_callback(completed)
