"""A reclaim snapshot may stop and release only the execution it observed."""
from __future__ import annotations

import time

import pytest

from hermes_cli import kanban_claims as claims
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as db
from hermes_cli import kanban_worker_identity as identity


@pytest.fixture
def board(tmp_path):
    path = tmp_path / 'claims.db'
    writer = db.connect(path)
    peer = db.connect(path)
    try:
        yield writer, peer
    finally:
        peer.close()
        writer.close()


def _task_row(conn, task_id):
    return dict(conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone())


def _claim(conn, phase='ready'):
    task_id = kb.create_task(conn, title='generation-bound recovery', assignee='builder')
    if phase == 'review':
        with db.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
    lock = kb._claimer_id()
    claim = claims.claim_review_task if phase == 'review' else claims.claim_task
    assert claim(conn, task_id, claimer=lock) is not None
    with db.write_txn(conn):
        conn.execute(
            'UPDATE tasks SET worker_pid = 424242, worker_pid_started_at = 17, '
            "worker_scope = 'observed.scope', worker_registered_at = 1, "
            'last_heartbeat_at = ?, claim_expires = ?, consecutive_failures = 3 WHERE id = ?',
            (int(time.time()) - 7200, int(time.time()) - 20, task_id),
        )
    return task_id


def _move_execution(peer, task_id, change):
    before = _task_row(peer, task_id)
    if change == 'run':
        # A single dispatcher legitimately reuses its claim lock across attempts.
        # Reclaim must distinguish those attempts even with identical PID identity.
        with db.write_txn(peer):
            peer.execute(
                "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
                'worker_scope=NULL, reclaim_reserved_at=NULL WHERE id=?',
                (task_id,),
            )
        assert claims.claim_task(peer, task_id, claimer=before['claim_lock']) is not None
        with db.write_txn(peer):
            peer.execute(
                'UPDATE tasks SET worker_pid=?, worker_pid_started_at=?, worker_scope=?, '
                'worker_registered_at=1, last_heartbeat_at=?, claim_expires=?, '
                "consecutive_failures=7, last_failure_error='successor evidence' WHERE id=?",
                (before['worker_pid'], before['worker_pid_started_at'], before['worker_scope'],
                 before['last_heartbeat_at'], before['claim_expires'], task_id),
            )
    elif change == 'pid':
        with db.write_txn(peer):
            peer.execute('UPDATE tasks SET worker_pid = worker_pid + 1 WHERE id = ?', (task_id,))
    elif change == 'heartbeat':
        # Models an old-version writer that may not know the reservation column.
        with db.write_txn(peer):
            peer.execute('UPDATE tasks SET last_heartbeat_at = ? WHERE id = ?',
                         (int(time.time()), task_id))
    return _task_row(peer, task_id)


_CASES = [(mode, 'current', phase) for mode in ('manual', 'ttl') for phase in ('ready', 'review')]
_CASES += [(mode, point, change) for mode in ('manual', 'ttl')
           for point in ('reserve', 'recheck', 'final') for change in ('run', 'pid', 'heartbeat')]
_CASES += [('manual', 'wrong_run', 'run')]
_CASES += [('manual', 'parked', phase) for phase in ('ready', 'review')]
_CASES += [('manual', 'reservation', state) for state in ('active', 'expired', 'replacement_token')]


@pytest.mark.parametrize(('mode', 'point', 'change'), _CASES,
                         ids=lambda value: value)
def test_reclaim_preserves_exact_execution_across_interleavings(board, monkeypatch, mode, point, change):
    conn, peer = board
    task_id = _claim(conn, change if point == 'current' else 'ready')
    original = _task_row(conn, task_id)
    if point == 'parked':
        # A deferred handoff keeps its scope even when an older writer exposes
        # a ready/review status. A direct claim cannot bypass dispatch's fence.
        with db.write_txn(peer):
            peer.execute('UPDATE tasks SET status=?, claim_lock=NULL WHERE id=?', (change, task_id))
        claim = claims.claim_review_task if change == 'review' else claims.claim_task
        assert claim(conn, task_id) is None
        final = _task_row(peer, task_id)
        assert final['status'] == change
        assert final['worker_scope'] == original['worker_scope']
        assert final['current_run_id'] == original['current_run_id']
        assert peer.execute('SELECT ended_at FROM task_runs WHERE id=?',
                            (original['current_run_id'],)).fetchone()['ended_at'] is None
        return
    if point == 'reservation':
        now = int(time.time())
        first = claims.reserve_reclaim(conn, task_id, original, now=now)
        assert first is not None
        if change == 'active':
            assert claims.reserve_reclaim(peer, task_id, _task_row(peer, task_id), now=now + 1) is None
            assert _task_row(peer, task_id)['reclaim_reserved_at'] == first['reclaim_reserved_at']
            return
        if change == 'expired':
            second = claims.reserve_reclaim(peer, task_id, _task_row(peer, task_id),
                                            now=first['claim_expires'] + 1)
            assert second is not None
        else:
            # Old-version controllers might replace only the reservation token.
            # The first controller must compare that token, not merely non-NULL.
            with db.write_txn(peer):
                peer.execute('UPDATE tasks SET reclaim_reserved_at=? WHERE id=?', (now + 1, task_id))
            second = _task_row(peer, task_id)
        assert second['reclaim_reserved_at'] != first['reclaim_reserved_at']
        assert claims._recheck_reclaim_reservation(
            conn, task_id, first['claim_lock'], first['claim_expires'], first,
        ) is False
        assert _task_row(peer, task_id)['reclaim_reserved_at'] == second['reclaim_reserved_at']
        return
    moved = None
    stopped = []
    monkeypatch.setattr(identity, '_run_worker_alive', lambda row: (False, 'fixture_worker_dead'))

    def move():
        nonlocal moved
        moved = _move_execution(peer, task_id, change)

    def terminate(pid, lock, **kwargs):
        # A second connection can make progress at the irreversible boundary:
        # process termination must not hold the SQLite writer transaction.
        assert not conn.in_transaction
        stopped.append((pid, lock))
        if point == 'final':
            move()
        return {'host_local': True, 'termination_attempted': True, 'terminated': True,
                'scope_unit': 'observed.scope', 'scope_stopped': True}

    monkeypatch.setattr(identity, '_terminate_reclaimed_worker', terminate)
    if point in ('reserve', 'recheck'):
        symbol = 'reserve_reclaim' if point == 'reserve' else '_recheck_reclaim_reservation'
        original_boundary = getattr(claims, symbol)

        def race(*args, **kwargs):
            move()
            return original_boundary(*args, **kwargs)

        monkeypatch.setattr(claims, symbol, race)
    if point == 'wrong_run':
        move()

    if mode == 'manual':
        kwargs = {'expected_run_id': original['current_run_id']} if point == 'wrong_run' else {}
        result = claims.reclaim_task(conn, task_id, **kwargs)
    else:
        result = claims.release_stale_claims(conn)
    final = _task_row(peer, task_id)

    if point == 'current':
        assert result == 1
        assert stopped == [(original['worker_pid'], original['claim_lock'])]
        assert final['status'] == change
        assert final['claim_lock'] is None
        assert final['worker_pid'] is None
        run = peer.execute('SELECT outcome, ended_at FROM task_runs WHERE id=?',
                           (original['current_run_id'],)).fetchone()
        assert run['outcome'] == 'reclaimed'
        assert run['ended_at'] is not None
    else:
        assert result == 0
        assert moved is not None
        assert final['status'] == 'running'
        for column in ('current_run_id', 'claim_lock', 'worker_pid', 'worker_pid_started_at',
                       'worker_scope', 'last_heartbeat_at', 'consecutive_failures', 'last_failure_error'):
            assert final[column] == moved[column], column
        assert stopped == ([(original['worker_pid'], original['claim_lock'])] if point == 'final' else [])
        run = peer.execute('SELECT ended_at FROM task_runs WHERE id=?',
                           (final['current_run_id'],)).fetchone()
        assert run['ended_at'] is None
