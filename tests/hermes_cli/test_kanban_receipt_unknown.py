"""t_0b026e13: a worker that finished its work must not be booked as a crash.

Two ledgers exist for one Kanban attempt: the owner's ``prompt.receipt`` (the submitter's
witness) and the board's ``worker_result`` event (the durable result). An owner restart
fences the admission ``unknown`` while the board may already hold the attempt's terminal
result — the submitter used to raise on the receipt alone and the dead-worker sweep booked
``crashed`` twice, spending the card's retry budget (``crashed x2 -> gave_up``).

The fix has two halves, both pinned here:

1. ``kanban_worker_client._submit_claim`` reads the board back before calling anything
   failed: a recorded ``worker_result`` is returned as-is (exit 0 -> clean_exit class,
   not crashed); with nothing on the board the process exits
   ``KANBAN_RECEIPT_UNKNOWN_EXIT_CODE`` instead of a bare rc=1 crash.
2. The dead-worker sweep books that sentinel as ``receipt_unknown``: the card is requeued
   to its source phase WITHOUT touching ``consecutive_failures`` (the host's restart
   window is not the card's failure), mirroring the quota-wall (``rate_limited``) channel.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_worker_client
from hermes_cli.gateway_client import GatewayClient
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

PARAMS = {'board': 'b', 'task_id': 't1', 'run_id': 7, 'claim_lock': 'FinnMBP.lan:owner'}


# --- submitter side: read the board back before calling the attempt failed ---------------------

class _UnknownReceiptOwner:
    """Admits the claim and answers the receipt ``unknown`` — the shape an owner restart
    leaves behind (the admission is fenced while the board keeps whatever the worker wrote)."""

    def __init__(self):
        self.inbox, self.sent = asyncio.Queue(), []

    async def send(self, raw):
        req = json.loads(raw)
        self.sent.append(req)
        if req['method'] == 'kanban.run':
            await self.inbox.put(json.dumps({'id': req['id'], 'result': {
                'session_id': 's1', 'receipt': {'status': 'unknown', 'admission_id': 'a1'}}}))
        elif req['method'] == 'prompt.receipt':
            await self.inbox.put(json.dumps({'id': req['id'], 'result': {
                'status': 'unknown', 'admission_id': 'a1'}}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.inbox.get()


def _board_with_worker_result(path: Path, *, task_id='t1', run_id=7, claim_lock='FinnMBP.lan:owner',
                              exit_code=0, last_output='', pid=4242) -> None:
    """The slice of a board ``worker_result()`` reads: its ``task_events`` table."""
    conn = sqlite3.connect(path)
    conn.execute(
        'CREATE TABLE task_events (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, '
        'run_id INTEGER, kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL)')
    conn.execute('INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?,?,?,?,?)',
                 (task_id, run_id, 'worker_result',
                  json.dumps({'pid': pid, 'claim_lock': claim_lock, 'exit_code': exit_code,
                              'last_output': last_output}), int(time.time())))
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_a_settled_result_survives_a_lost_receipt(tmp_path: Path, capsys):
    """t_0b026e13: the owner recorded exit 0, then the receipt died across the owner handoff.
    The submitter must return 0 (the board's result), not raise into a phantom crash."""
    board = tmp_path / 'kanban.db'
    _board_with_worker_result(board, exit_code=0, last_output='rework done, 17/17 verified')
    owner = _UnknownReceiptOwner()
    async with GatewayClient(owner) as client:
        code = await kanban_worker_client._submit_claim(client, PARAMS, str(board))
    assert code == 0
    assert 'rework done, 17/17 verified' in capsys.readouterr().out
    # The readback is the only extra RPC-free step: one submission, no receipt polling on an
    # already-unknown answer.
    assert [r['method'] for r in owner.sent] == ['kanban.run']


@pytest.mark.asyncio
async def test_a_nonzero_settled_result_is_reported_as_recorded(tmp_path: Path):
    """The board's word is returned as-is — a recorded failure stays a failure (it is not
    laundered into a clean exit by the lost receipt)."""
    board = tmp_path / 'kanban.db'
    _board_with_worker_result(board, exit_code=1, last_output='')
    async with GatewayClient(_UnknownReceiptOwner()) as client:
        code = await kanban_worker_client._submit_claim(client, PARAMS, str(board))
    assert code == 1


@pytest.mark.asyncio
async def test_a_receipt_without_a_board_result_is_a_lost_receipt(tmp_path: Path):
    """Nothing on the board: the outcome is unknowable, so the submitter must not invent a
    failure either — it raises the dedicated lost-receipt error."""
    board = tmp_path / 'kanban.db'
    # The owner's result for a DIFFERENT claim must not be adopted: same run, other lock.
    _board_with_worker_result(board, exit_code=0, claim_lock='FinnMBP.lan:previous-owner')
    async with GatewayClient(_UnknownReceiptOwner()) as client:
        with pytest.raises(kanban_worker_client.KanbanReceiptLost):
            await kanban_worker_client._submit_claim(client, PARAMS, str(board))


def test_the_client_exits_the_receipt_sentinel(monkeypatch, capsys):
    """The lost-receipt path exits ``KANBAN_RECEIPT_UNKNOWN_EXIT_CODE`` (79), not the rc=1
    that the dead-worker sweep books as a crash."""
    for key, value in {'HERMES_KANBAN_BOARD': 'b', 'HERMES_KANBAN_TASK': 't1',
                       'HERMES_KANBAN_RUN_ID': '7', 'HERMES_KANBAN_CLAIM_LOCK': 'FinnMBP.lan:owner',
                       'HERMES_KANBAN_DB': '/nonexistent/kanban.db'}.items():
        monkeypatch.setenv(key, value)

    async def lost(params, board_db):
        raise kanban_worker_client.KanbanReceiptLost('no worker_result on the board')

    monkeypatch.setattr(kanban_worker_client, 'run', lost)
    assert kanban_worker_client._submit() == kb.KANBAN_RECEIPT_UNKNOWN_EXIT_CODE
    assert 'Kanban receipt lost' in capsys.readouterr().err


def test_main_writes_the_receipt_sentinel_to_the_trailer(monkeypatch, capsys):
    """The durable witness the dead-worker sweep reads must carry the sentinel too."""
    monkeypatch.setattr(kanban_worker_client, '_submit', lambda: kb.KANBAN_RECEIPT_UNKNOWN_EXIT_CODE)
    assert kanban_worker_client.main() == kb.KANBAN_RECEIPT_UNKNOWN_EXIT_CODE
    assert f'{KANBAN_WORKER_EXIT_TRAILER}{kb.KANBAN_RECEIPT_UNKNOWN_EXIT_CODE}' in capsys.readouterr().err


# --- sweep side: the sentinel is a neutral requeue, the owner's exit 0 stays clean -------------

@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_KANBAN_CRASH_GRACE_SECONDS', '0')
    monkeypatch.setattr(kb, '_pid_alive', lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


def _dead_worker_with_log(conn, tid: str, pid: int, rc: int) -> None:
    """Claim ``tid`` for a worker that already exited ``rc`` and wrote its log — never reaped here."""
    host = kb._claimer_id().split(':', 1)[0]
    kb.claim_task(conn, tid, claimer=f'{host}:w{pid}')
    conn.execute(
        'UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?',
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'a', encoding='utf-8') as f:
        f.write(f'Kanban receipt lost: no worker_result on the board\n\n{KANBAN_WORKER_EXIT_TRAILER}{rc}\n')


def _dead_pid() -> int:
    proc = subprocess.Popen(['true'])
    proc.wait()
    return proc.pid


def test_a_lost_receipt_is_requeued_without_counting_a_failure(kanban_home):
    """Acceptance 2: unknown + no worker_result -> ``consecutive_failures`` untouched, the
    card is back at ``ready`` with the reason on the board, and the tick reports it on its
    own channel (not as a crash)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='t', assignee='a')
        _dead_worker_with_log(conn, tid, 70001, kb.KANBAN_RECEIPT_UNKNOWN_EXIT_CODE)

        crashed = kbd.detect_crashed_workers(conn)

        ev = conn.execute('SELECT kind FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1',
                          (tid,)).fetchone()
        run = conn.execute('SELECT outcome, error, metadata FROM task_runs WHERE task_id=? '
                           'ORDER BY id DESC LIMIT 1', (tid,)).fetchone()
        task = kb.get_task(conn, tid)
        assert ev['kind'] == 'receipt_unknown'
        assert run['outcome'] == 'receipt_unknown'
        assert kb._json_dict(run['metadata']).get('exit_code') == kb.KANBAN_RECEIPT_UNKNOWN_EXIT_CODE
        assert task.status == 'ready'
        assert task.consecutive_failures == 0
        assert 'receipt' in (task.last_failure_error or '')
        assert tid not in crashed
        assert tid in kbd.detect_crashed_workers._last_receipt_unknown
        assert kb._retry_status_for_run(conn, tid) == 'ready'


def test_repeated_lost_receipts_never_trip_the_breaker(kanban_home):
    """A card behind a restarting owner may requeue many times; none of it is the card's
    failure, so even ``failure_limit=2`` must never block it."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='loop', assignee='a')
        for i in range(5):
            _dead_worker_with_log(conn, tid, 72000 + i, kb.KANBAN_RECEIPT_UNKNOWN_EXIT_CODE)
            kbd.detect_crashed_workers(conn)
            kb.recompute_ready(conn, failure_limit=2)
        task = kb.get_task(conn, tid)
        assert task.status == 'ready'
        assert task.consecutive_failures == 0


def test_an_owner_recorded_exit_0_is_booked_clean_not_crashed(kanban_home):
    """Acceptance 1 (board half): the owner settled the attempt with exit 0 before the
    receipt was lost. The sweep must book the clean_exit class (protocol violation), never
    a crash, and leave the retry budget alone."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title='settled', assignee='w')
        host = kb._claimer_id().split(':', 1)[0]
        task = kb.claim_task(conn, tid, claimer=f'{host}:owner')
        assert task is not None
        submitter, interpreter = _dead_pid(), _dead_pid()
        kbd._set_worker_pid(conn, tid, submitter)
        with kb.write_txn(conn):
            kb._append_event(conn, tid, 'worker_bound',
                             {'pid': interpreter, 'claim_lock': task.claim_lock, 'started_at': 'x'},
                             run_id=task.current_run_id)
            kb._append_event(conn, tid, 'worker_result',
                             {'pid': interpreter, 'claim_lock': task.claim_lock, 'exit_code': 0,
                              'last_output': 'done'}, run_id=task.current_run_id)

        kbd.detect_crashed_workers(conn)

        ev = conn.execute('SELECT kind FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1',
                          (tid,)).fetchone()
        run = conn.execute('SELECT outcome, error, metadata FROM task_runs WHERE id=?',
                           (task.current_run_id,)).fetchone()
        row = kb.get_task(conn, tid)
        # The clean_exit class: booked as the protocol violation (rc 0 while running), never
        # as a bare "pid N not alive" crash, and the unified retry budget is untouched.
        assert ev['kind'] == 'protocol_violation'
        assert kb._json_dict(run['metadata']).get('protocol_violation') is True
        assert kb._json_dict(run['metadata']).get('exit_code') == 0
        assert 'not alive' not in (run['error'] or '')
        assert row.consecutive_failures == 0
        assert row.status == 'ready'
