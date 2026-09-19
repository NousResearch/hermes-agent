"""Contraprovas de concorrencia para o estacionamento por cota."""
import pytest
from tests.hermes_cli.test_kanban_quota_budget import board, quota_exit
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_quota import park_exhausted_quota


@pytest.mark.parametrize('claim', ['running', 'worker', 'lock', 'run'])
def test_parking_never_clears_a_live_claim(board, claim):
    tid = kb.create_task(board, title='nao interromper', assignee='default')
    quota_exit(board, tid, 'ready')
    quota_exit(board, tid, 'ready')
    updates = {
        'running': "status='running'",
        'worker': 'worker_pid=123',
        'lock': "claim_lock='outro-dono'",
        'run': 'current_run_id=(SELECT max(id) FROM task_runs)',
    }
    with kb.write_txn(board):
        board.execute(f'UPDATE tasks SET {updates[claim]} WHERE id=?', (tid,))
    before = tuple(board.execute('SELECT * FROM tasks WHERE id=?', (tid,)).fetchone())
    assert not park_exhausted_quota(board, tid, 'ready')
    assert before == tuple(board.execute('SELECT * FROM tasks WHERE id=?', (tid,)).fetchone())


def test_parking_is_idempotent_and_emits_one_visible_reason(board):
    tid = kb.create_task(board, title='bloqueio visivel', assignee='default')
    quota_exit(board, tid, 'review')
    quota_exit(board, tid, 'review')
    assert park_exhausted_quota(board, tid, 'review')
    assert not park_exhausted_quota(board, tid, 'review')
    events = board.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='blocked'", (tid,)).fetchall()
    assert len(events) == 1
    assert 'quota retry budget exhausted' in events[0]['payload']
