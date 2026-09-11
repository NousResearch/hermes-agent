"""Pass-2 behavioral regressions against real candidate SQLite APIs."""
import sqlite3

import pytest
from hermes_cli import kanban_db as kb
from tests.hermes_cli.test_kanban_authority_history import conn, enrolled, history


def rows(db):
    tables = ('tasks', 'task_runs', 'task_events', 'authority_history',
              'authority_task_bindings', 'authority_owner_bindings', 'authority_run_bindings',
              'authority_source_bindings', 'authority_history_meta', 'authority_history_removal')
    return {t: [tuple(r) for r in db.execute(f'SELECT * FROM {t}')] for t in tables}


def test_credential_swap_with_event_rolls_back_every_row(conn):
    cap, task = enrolled(conn)
    run = kb.claim_task(conn, task, claimer='private-bearer-token').current_run_id
    before = rows(conn)
    with pytest.raises(ValueError):
        with kb.write_txn(conn):
            conn.execute('UPDATE tasks SET claim_lock=? WHERE id=?', ('replacement', task))
            conn.execute('UPDATE task_runs SET claim_lock=? WHERE id=?', ('replacement', run))
            kb._append_event(conn, task, 'heartbeat', run_id=run)
    assert rows(conn) == before


def test_emitted_running_without_run_rolls_back_every_row(conn):
    cap, task = enrolled(conn)
    before = rows(conn)
    with pytest.raises(ValueError):
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='running' WHERE id=?", (task,))
            kb._append_event(conn, task, 'status')
    assert rows(conn) == before


def test_last_run_observation_must_match_commit(conn):
    cap, task = enrolled(conn)
    run = kb.claim_task(conn, task, claimer='private-bearer-token').current_run_id
    before = rows(conn)
    with pytest.raises(ValueError):
        with kb.write_txn(conn):
            conn.execute('UPDATE task_runs SET started_at=started_at+1 WHERE id=?', (run,))
            kb._append_event(conn, task, 'heartbeat', run_id=run)
            conn.execute('UPDATE task_runs SET started_at=started_at+1 WHERE id=?', (run,))
            kb._append_event(conn, task, 'heartbeat', run_id=run)
            conn.execute('UPDATE task_runs SET started_at=started_at-1 WHERE id=?', (run,))
    assert rows(conn) == before


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('additional', [False, True])
def test_coherent_bypass_refused(conn, explicit, additional):
    cap, task = enrolled(conn)
    run = kb.claim_task(conn, task, claimer='private-bearer-token').current_run_id
    before = rows(conn)
    db = conn
    if additional:
        db = sqlite3.connect(conn.execute('PRAGMA database_list').fetchone()[2], isolation_level=None)
    try:
        if explicit:
            db.execute('BEGIN IMMEDIATE')
        with pytest.raises((sqlite3.DatabaseError, ValueError)):
            db.execute('UPDATE tasks SET claim_lock=? WHERE id=?', ('replacement', task))
            db.execute('UPDATE task_runs SET claim_lock=? WHERE id=?', ('replacement', run))
        if db.in_transaction:
            db.rollback()
    finally:
        if additional:
            db.close()
    assert rows(conn) == before


@pytest.mark.parametrize('field', ['claim_lock', 'started_at'])
def test_misleading_observation_even_when_final_state_unchanged(conn, field):
    cap, task = enrolled(conn)
    run = kb.claim_task(conn, task, claimer='private-bearer-token').current_run_id
    before = rows(conn)
    old = conn.execute(f'SELECT {field} FROM task_runs WHERE id=?', (run,)).fetchone()[0]
    replacement = 'replacement' if field == 'claim_lock' else old + 1
    with pytest.raises(ValueError):
        with kb.write_txn(conn):
            conn.execute(f'UPDATE task_runs SET {field}=? WHERE id=?', (replacement, run))
            kb._append_event(conn, task, 'heartbeat', run_id=run)
            conn.execute(f'UPDATE task_runs SET {field}=? WHERE id=?', (old, run))
    assert rows(conn) == before


def test_orphan_open_run_rolls_back_every_row(conn):
    cap, task = enrolled(conn)
    run = kb.claim_task(conn, task, claimer='private-bearer-token').current_run_id
    before = rows(conn)
    with pytest.raises(ValueError):
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, claim_expires=NULL WHERE id=?", (task,))
            kb._append_event(conn, task, 'status', run_id=run)
    assert rows(conn) == before
