"""Dispatcher subprocess: submit one claim to its profile owner; never run an agent."""
# Spawned as a bare ``sys.executable -m``: in a PM install that interpreter carries no dependencies
# until hermes_bootstrap selects the committed environment, so it must be the first import.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap" and not "hermes_bootstrap".startswith(f"{exc.name}."):
        raise

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Optional

# A worker that loses its owner (a dropped socket, a restart, `hermes update` draining it or a new
# owner still starting) waits for it instead of exiting: the condition is the host's, not the card's,
# and a failed exit here spends the card's failure budget and lets the owner end the turn. Asking
# again is safe: ``kanban.run`` is idempotent per attempt and ``prompt.receipt`` is a read.
# ``OWNER_WAIT_S`` is the longest the owner may stay silent between answers before the worker
# gives up (the first ``kanban.run`` answer has its own wait, ``admission_answer_budget``).
OWNER_WAIT_S = 180.0
# A fencing owner is not a silent one: while an update owns the install, the owner keeps
# ANSWERING — discovery verdicts ``draining``/``starting`` (``update_paused``) and in-band
# ``runtime_draining`` refusals — for the 10+ minutes an update pause can run. That stretch is
# bounded by the card's own ``max_runtime_seconds`` (read from the board) instead of the
# silent-owner wait, so one attempt survives a full update wave instead of dying inside it; a
# card with no ``max_runtime_seconds`` falls back to this default.
OWNER_FENCE_DEFAULT_S = 3600.0
_TRANSIENT_OWNER_STATES = frozenset({'draining', 'starting'})
# In-band owner refusals that leave the attempt intact: a draining owner admits nothing new, and a
# retired authority's successor holds the admission the next connection reads.
_TRANSIENT_RPC_REASONS = frozenset({'runtime_draining', 'runtime_retired'})
# Control-socket answers a restarting owner gives (an ``inaccessible`` owner's reason, from
# discovery or the ticket fetch): a reset or half-written answer, or a replaced owner refusing a
# stale instance. Unsafe nodes or peers, a refused authorization, a bad pointer or a failed start
# stay final.
_RESTART_CONTROL_REASONS = frozenset({'invalid_control_peer', 'incomplete_control_response',
                                      'invalid_control_response'})
# Past the board lock wait: the owner's session setup and receipt before and after the write.
ADMISSION_ANSWER_MARGIN_S = 60.0


def admission_answer_budget():
    """Seconds to wait for ``kanban.run``'s answer: the longest the owner's admission write can
    wait on the board lock (``write_txn`` retries ``BEGIN IMMEDIATE`` ``_BUSY_MAX_RETRIES`` times,
    each under the busy timeout) plus ``ADMISSION_ANSWER_MARGIN_S``."""
    from hermes_cli import kanban_db_connect
    lock_wait = (kanban_db_connect._BUSY_MAX_RETRIES + 1) * kanban_db_connect._resolve_busy_timeout_ms() / 1000
    return lock_wait + ADMISSION_ANSWER_MARGIN_S


class OwnerFenceTimeout(Exception):
    """The owner kept advertising its fence (a ``draining``/``starting`` discovery verdict, or an
    in-band ``runtime_draining`` refusal) past the card's fence budget: the update wave won and
    this attempt never ran anything of the card's. Infrastructure of the host, not a card
    failure — ``_submit`` converts it into the sweep's distinguishable drain-sentinel exit."""

    def __init__(self, waited_s: float, detail: str):
        super().__init__(f'Kanban owner draining for {waited_s:.0f}s: {detail}')
        self.waited_s, self.detail = waited_s, detail


def _fence(exc):
    """The name of the fence *exc* proves the owner still advertises and answers, else None.

    A fence is the owner's OWN verdict or refusal while it is working (discovery
    ``draining``/``starting``, in-band ``runtime_draining``). A lost transport or a garbled
    control answer is silence — nothing to extend a wait with."""
    from hermes_cli.gateway_client import GatewayRPCError, GatewayUnavailableError
    if isinstance(exc, GatewayRPCError):
        return 'runtime_draining' if str(exc) == 'runtime_draining' else None
    if isinstance(exc, GatewayUnavailableError):
        if exc.state in _TRANSIENT_OWNER_STATES:
            return exc.reason_code or exc.state or 'draining'
    return None


def _card_max_runtime_seconds(task_id, board_db):
    """The card's ``max_runtime_seconds`` straight from the board (read-only), else None."""
    try:
        import sqlite3
        from contextlib import closing
        db = str(Path(board_db).resolve())
        with closing(sqlite3.connect(f'file:{db}?mode=ro', uri=True)) as conn:
            row = conn.execute(
                'SELECT max_runtime_seconds FROM tasks WHERE id = ?', (task_id,)).fetchone()
        return float(row[0]) if row and row[0] else None
    except Exception:
        return None


def _fence_budget_s(task_id, board_db):
    """Seconds a worker keeps waiting while its owner keeps advertising one fence.

    Never below ``OWNER_WAIT_S``: a card whose budget is smaller must not lose the recovery
    window a silent owner would have gotten — the fence budget only ever lengthens the wait."""
    limit = _card_max_runtime_seconds(task_id, board_db)
    return max(OWNER_WAIT_S, limit if limit is not None else OWNER_FENCE_DEFAULT_S)


@dataclass
class _Claim:
    """What the owner has said about this attempt, carried across connections."""
    answered: float  # monotonic time of the owner's last answer
    session_id: Optional[str] = None
    admission_id: Optional[str] = None

    def silent_for(self):
        return time.monotonic() - self.answered


async def run(params, board_db):
    from agent.retry_utils import jittered_backoff
    from hermes_cli.gateway_client import connect_gateway, GatewayClientError, outside_restart_wait
    from websockets.exceptions import WebSocketException
    if os.environ.get('HERMES_TUI_GATEWAY_URL'):
        raise GatewayClientError('Kanban requires the assigned local profile owner')
    # A lost socket is not a lost attempt, and exiting here would let ``_watch_submitter`` end the
    # owner's in-flight turn, so the next connection resumes whatever the last one reached. The
    # wait is bounded from the owner's last answer: an owner that keeps answering keeps the
    # worker, and one that stays silent for ``OWNER_WAIT_S`` fails the worker honestly. A fence
    # the owner itself advertises (an update wave) is an answer, and waits out the card's fence
    # budget instead — an update pause routinely runs longer than ``OWNER_WAIT_S``.
    claim = _Claim(answered=time.monotonic())
    failures = 0
    fenced_since = None  # when the owner began advertising the fence it still advertises
    fence_budget = None  # the stretch's wait budget, resolved once when its fence appears
    while True:
        seen = claim.answered
        try:
            # Never into a restart's after-turn wait: it cut this socket so the turn's submitter
            # waits for the NEXT owner, and a worker polling its receipt inside the wait keeps the
            # turn it waits on alive, holding `hermes update` until restart_after_turn_timeout.
            with outside_restart_wait():
                async with connect_gateway() as client:
                    # kanban.run subscribes this socket to the card's session, but only receipts
                    # matter here: discard the turn's events, or they fill the client's bounded
                    # queue and its reader stops, with the next receipt behind it.
                    drain = asyncio.create_task(_discard(client.events))
                    try:
                        receipt = await _await_receipt(client, params, board_db, claim)
                    finally:
                        drain.cancel()
        # A restart also surfaces as a refused or reset dial, a failed handshake, or an answer
        # that never came.
        except (GatewayClientError, WebSocketException, OSError, TimeoutError) as exc:
            if not _restartable(exc):
                raise
            # WebSocket errors can embed remote bodies; only our own messages are quoted.
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            detail = exc if isinstance(exc, GatewayClientError) else type(exc).__name__ + (
                f' (HTTP {status})' if status else '')
            if _fence(exc) is not None:
                # A fence verdict or refusal is an owner answer (it proves the owner lives and
                # names WHY it cannot serve this attempt): refresh the silent window to THIS
                # answer and let this stretch wait out the card's fence budget. The budget is
                # resolved once per stretch and anchored to the fence's START (fenced_since),
                # so the stretch ends when the wave outlasts it even as every answer keeps the
                # silent window fresh. When the fence stops being advertised (owner back, or
                # the wave turned into plain silence by losing its connect path) the
                # silent-owner rule takes over from the LAST fence answer — the first
                # post-wave non-fence error must never be judged against an age the fence
                # answers kept clearing.
                fenced_since = fenced_since if fenced_since is not None else time.monotonic()
                claim.answered = time.monotonic()
                if fence_budget is None:
                    fence_budget = _fence_budget_s(params['task_id'], board_db)
                if time.monotonic() - fenced_since >= fence_budget:
                    raise OwnerFenceTimeout(time.monotonic() - fenced_since, detail) from exc
            else:
                fenced_since, fence_budget = None, None
                if claim.silent_for() >= OWNER_WAIT_S:
                    raise GatewayClientError(
                        f'Kanban owner silent for {claim.silent_for():.0f}s: {detail}') from exc
        else:
            return _exit_code(receipt, params, board_db)
        failures = 1 if claim.answered != seen else failures + 1
        # Jittered (1-2 s first, up to 16 s): workers of one dead owner do not all redial, and so
        # start a replacement, in the same instant.
        await asyncio.sleep(jittered_backoff(failures, base_delay=1.0, max_delay=8.0, jitter_ratio=1.0))


def _restartable(exc):
    """False when *exc* is the owner's own final answer; True for a restarting owner or a lost transport."""
    from hermes_cli.gateway_client import GatewayRPCError, GatewayUnavailableError
    if isinstance(exc, GatewayRPCError):
        return str(exc) in _TRANSIENT_RPC_REASONS
    if isinstance(exc, GatewayUnavailableError):
        return exc.state in _TRANSIENT_OWNER_STATES or exc.reason_code in _RESTART_CONTROL_REASONS
    return True


async def _discard(events):
    while True:
        await events.get()


async def _await_receipt(client, params, board_db, claim):
    """Submit the claim (or, on a later connection, re-read the admission it got) and return its
    settled receipt, recording each owner answer on *claim*."""
    if claim.admission_id is None:
        db = str(await asyncio.to_thread(Path(board_db).resolve))
        # The owner answers only after it checks the claim and writes its ``owner_admitted`` marker
        # in one board write transaction. The board (the cross-profile dispatch bus) can make that
        # wait out its busy timeout, so the 30 s default left the owner admitting an attempt whose
        # submitter had exited (the later-bound worker's submitter-liveness watcher ended it, and the
        # admission was left ``unknown``). The wait is still finite: once the marker commits,
        # ``enforce_max_runtime`` and TTL reclaim skip the run (``owner_reclaim_paused``), so an owner
        # that stalls after admitting is bounded by this wait and ``OWNER_WAIT_S``.
        accepted = await client.rpc('kanban.run', _timeout=admission_answer_budget(), **dict(params, db=db))
        receipt = accepted['receipt']
        claim.session_id, claim.admission_id = accepted['session_id'], receipt['admission_id']
        print(f'Session: {claim.session_id}', file=sys.stderr)
    else:
        receipt = await _receipt(client, claim)
    claim.answered = time.monotonic()
    while receipt['status'] not in {'terminal', 'unknown'}:
        await asyncio.sleep(.2)
        receipt = await _receipt(client, claim)
        claim.answered = time.monotonic()
    return receipt


def _exit_code(receipt, params, board_db):
    from hermes_cli.gateway_client import GatewayClientError
    if receipt['status'] == 'unknown':
        raise GatewayClientError('Kanban execution unknown; retained by owner')
    from gateway.session_kanban import worker_result
    result = worker_result(board_db, params)
    # The attempt's reply ends the log, as a one-shot worker's did: crash diagnostics quote it.
    if result.get('last_output'):
        print(result['last_output'])
    return result.get('exit_code', 1)


async def _receipt(client, claim):
    return await client.rpc('prompt.receipt', session_id=claim.session_id, admission_id=claim.admission_id)


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
    except OwnerFenceTimeout as exc:
        # The dead-worker sweep books this exit code as an infrastructure requeue (a spawn
        # refusal's accounting) — it must never read as a card failure.
        print(f'Kanban owner fenced past the wait budget: {exc}', file=sys.stderr)
        from hermes_cli.kanban_db import KANBAN_OWNER_DRAIN_EXIT_CODE
        return KANBAN_OWNER_DRAIN_EXIT_CODE
    except (KeyError, ValueError, OSError) as exc:
        print(f'Kanban owner refused: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
