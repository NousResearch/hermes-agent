"""Dispatcher subprocess: submit one claim to its profile owner; never run an agent."""
# Spawned as a bare ``sys.executable -m``: in a PM install that interpreter carries no dependencies
# until hermes_bootstrap selects the committed environment, so it must be the first import.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap":
        raise

import asyncio
import os
from pathlib import Path
import sys
import time

# A worker the dispatcher spawns while `hermes update` drains the owner (or while the new owner is
# still starting) waits for it instead of exiting: the condition is the host's, not the card's, and
# a failed exit here spends the card's failure budget on an update window. Nothing is submitted
# until an owner is ready, so waiting can never run the claim twice.
OWNER_WAIT_S = 180.0
_TRANSIENT_OWNER_STATES = frozenset({'draining', 'starting'})
# Past the board lock wait: the owner's session setup and receipt before and after the write.
ADMISSION_ANSWER_MARGIN_S = 60.0
# A dial that dies on the host's transport (an owner thrashing at the host's RAM ceiling refuses or
# stalls the websocket handshake) reports one of these; the condition is the host's, not the card's.
_TRANSPORT_FAILURE_PREFIXES = ('Gateway connection failed', 'Gateway connection timed out')


def _is_transport_failure(exc):
    """A refused or stalled dial: the ``OSError`` family (``TimeoutError`` included) or a
    transport-class client error. Owner refusals and received RPC answers are never this."""
    return isinstance(exc, OSError) or str(exc).startswith(_TRANSPORT_FAILURE_PREFIXES)


def admission_answer_budget():
    """Seconds to wait for ``kanban.run``'s answer: the longest the owner's admission write can
    wait on the board lock (``write_txn`` retries ``BEGIN IMMEDIATE`` ``_BUSY_MAX_RETRIES`` times,
    each under the busy timeout) plus ``ADMISSION_ANSWER_MARGIN_S``."""
    from hermes_cli import kanban_db_connect
    lock_wait = (kanban_db_connect._BUSY_MAX_RETRIES + 1) * kanban_db_connect._resolve_busy_timeout_ms() / 1000
    return lock_wait + ADMISSION_ANSWER_MARGIN_S


async def run(params, board_db):
    from hermes_cli.gateway_client import connect_gateway, GatewayClientError, GatewayUnavailableError
    if os.environ.get('HERMES_TUI_GATEWAY_URL'):
        raise GatewayClientError('Kanban requires the assigned local profile owner')
    deadline = time.monotonic() + OWNER_WAIT_S
    # Once a connection attempt yields a client, the claim can already be on the wire: nothing past
    # this point is ever retried, so one worker run can never send the work twice.
    submitted = False
    while True:
        try:
            async with connect_gateway() as client:
                submitted = True
                # kanban.run subscribes this socket to the card's session, but only receipts matter
                # here: discard the turn's events, or they fill the client's bounded queue and its
                # reader stops, with the next receipt behind it.
                drain = asyncio.create_task(_discard(client.events))
                try:
                    return await _submit_claim(client, params, board_db)
                finally:
                    drain.cancel()
        except GatewayUnavailableError as exc:
            if exc.state not in _TRANSIENT_OWNER_STATES or time.monotonic() >= deadline:
                raise
            await asyncio.sleep(1)
        except (GatewayClientError, OSError) as exc:
            # A refused or stalled dial before any submission is the host's condition, not the
            # card's: redial within the same OWNER_WAIT_S budget instead of spending the card's
            # failure budget on it. Past the connection — or past the budget — it still raises: the
            # owner may already hold the claim ('execution unknown; retained by owner') and the
            # work must never be sent again.
            if submitted or not _is_transport_failure(exc) or time.monotonic() >= deadline:
                raise
            await asyncio.sleep(1)


async def _discard(events):
    while True:
        await events.get()


async def _submit_claim(client, params, board_db):
    from hermes_cli.gateway_client import GatewayClientError
    db = str(await asyncio.to_thread(Path(board_db).resolve))
    # The owner answers only after it checks the claim and writes its ``owner_admitted`` marker
    # in one board write transaction. The board (the cross-profile dispatch bus) can make that
    # wait out its busy timeout, so the 30 s default left the owner admitting an attempt whose
    # submitter had exited (the later-bound worker's submitter-liveness watcher ended it, and the
    # admission was left ``unknown``). The wait is still finite: once the marker commits,
    # ``enforce_max_runtime`` and TTL reclaim skip the run (``owner_reclaim_paused``), so an owner
    # that stalls after admitting is bounded only here. A hang-up fails the wait sooner.
    accepted = await client.rpc('kanban.run', _timeout=admission_answer_budget(), **dict(params, db=db))
    sid, receipt = accepted['session_id'], accepted['receipt']
    print(f'Session: {sid}', file=sys.stderr)
    while receipt['status'] not in {'terminal', 'unknown'}:
        await asyncio.sleep(.2)
        receipt = await client.rpc('prompt.receipt', session_id=sid, admission_id=receipt['admission_id'])
    if receipt['status'] == 'unknown':
        raise GatewayClientError('Kanban execution unknown; retained by owner')
    from gateway.session_kanban import worker_result
    result = worker_result(board_db, params)
    # The attempt's reply ends the log, as a one-shot worker's did: crash diagnostics quote it.
    if result.get('last_output'):
        print(result['last_output'])
    return result.get('exit_code', 1)


def main():
    # The dead-worker sweep books this process's exit from the trailer in its log (it never
    # reaps us), so every exit path writes it — the same contract as a quiet one-shot worker.
    from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER
    rc = 1
    try:
        rc = _submit()
    finally:
        print(f'\n{KANBAN_WORKER_EXIT_TRAILER}{int(rc)}', file=sys.stderr, flush=True)
    return rc


def _submit():
    try:
        params = {'board': os.environ['HERMES_KANBAN_BOARD'], 'task_id': os.environ['HERMES_KANBAN_TASK'],
            'run_id': int(os.environ['HERMES_KANBAN_RUN_ID']), 'claim_lock': os.environ['HERMES_KANBAN_CLAIM_LOCK']}
        board_db = os.environ['HERMES_KANBAN_DB']
        # Owner startup must never inherit this worker's task/tool/goal identity.
        for key in list(os.environ):
            # The busy timeout is a host tuning knob, not task identity: the answer budget reads it.
            if key.startswith('HERMES_KANBAN_') and key not in ('HERMES_KANBAN_HOME', 'HERMES_KANBAN_BUSY_TIMEOUT_MS'):
                os.environ.pop(key)
        for key in ('HERMES_SESSION_SOURCE', 'HERMES_TENANT', 'TERMINAL_CWD'):
            os.environ.pop(key, None)
        return asyncio.run(run(params, board_db))
    except (KeyError, ValueError, OSError) as exc:
        print(f'Kanban owner refused: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
