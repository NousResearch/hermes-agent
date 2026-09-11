"""Bounded lifecycle matrix: no worker launch; only process boundaries mocked."""
import sqlite3
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_db_dispatch import _set_worker_pid as _kbd_set_worker_pid
from tests.hermes_cli.test_kanban_authority_history import conn, enrolled, history
from tests.hermes_cli.test_kanban_history_pass2 import rows


@pytest.mark.parametrize('mode', ['unowned', 'synthetic', 'active'])
def test_schedule_unblock_retains_exact_run(conn, mode):
    cap, task = enrolled(conn)
    run = None
    if mode == 'active':
        run = kb.claim_task(conn, task, claimer='private-bearer-token').current_run_id
    assert kb.schedule_task(conn, task, reason=None if mode == 'unowned' else 'wait for time')
    record = history(conn, cap)['records'][-1]
    assert record['kind'] == 'scheduled'
    assert record['task_state']['status'] == 'scheduled'
    if mode == 'unowned':
        assert record['run_state'] is None and record['owner'] is None
    else:
        assert record['run_state']['status'] == 'scheduled'
        assert record['run_state']['ended_at'] is not None
        if mode == 'active':
            assert record['run_state']['id'] == run
            assert record['owner']['owner_ref'] == 'owner-1'
        else:
            assert record['owner'] is None
    assert kb.unblock_task(conn, task)
    assert history(conn, cap)['records'][-1]['task_state']['status'] == 'ready'


def test_prospective_scheduled_task_can_enroll(conn):
    cap = kb.enroll_authority_history(conn)
    task = kb.create_task(conn, title='scheduled', assignee='worker')
    assert kb.schedule_task(conn, task)
    kb.bind_authority_task(conn, task, repo_id='repo', project_id='project')
    assert history(conn, cap)['records'][-1]['kind'] == 'history_bound'


@pytest.mark.parametrize('enroll', [False, True])
def test_active_delete_refusal_preserves_ordinary_behavior(conn, enroll):
    if enroll:
        cap, task = enrolled(conn)
    else:
        task = kb.create_task(conn, title='ordinary', assignee='worker')
        kb.recompute_ready(conn)
    kb.claim_task(conn, task, claimer='private-bearer-token')
    before = rows(conn)
    if enroll:
        with pytest.raises(ValueError, match='active'):
            kb.delete_task(conn, task)
        assert not kb.delete_archived_task(conn, task)
        assert rows(conn) == before
    else:
        assert kb.delete_task(conn, task)


@pytest.mark.parametrize('archive', [False, True])
def test_terminal_delete_journal_fault_is_atomic(conn, archive):
    cap, task = enrolled(conn)
    kb.claim_task(conn, task, claimer='private-bearer-token')
    kb.archive_task(conn, task) if archive else kb.complete_task(conn, task)
    before = rows(conn)
    conn.execute("CREATE TRIGGER delete_fault BEFORE INSERT ON authority_history BEGIN SELECT RAISE(ABORT,'delete journal fault'); END")
    with pytest.raises(sqlite3.IntegrityError, match='delete journal fault'):
        (kb.delete_archived_task if archive else kb.delete_task)(conn, task)
    assert rows(conn) == before


@pytest.mark.parametrize('action,kind,terminal', [
    ('extend', 'claim_extended', False), ('defer', 'reclaim_deferred', False),
    ('timeout', 'timed_out', True), ('stale', 'stale', True),
    ('crash', 'crashed', True), ('rate', 'rate_limited', True),
    ('protocol', 'protocol_violation', True),
])
@pytest.mark.parametrize('fault', [False, True])
def test_real_runtime_lifecycle_matrix(conn, monkeypatch, action, kind, terminal, fault):
    cap, task = enrolled(conn)
    token = kb._claimer_id()
    kb.bind_authority_owner(conn, task, claimer=token, consumer_id='consumer', runtime_id='instance', owner_ref='matrix-owner')
    claimed = kb.claim_task(conn, task, claimer=token, ttl_seconds=-1)
    run = claimed.current_run_id
    # Fixture state in an audited transaction; no time mocking or worker spawn.
    with kb.write_txn(conn):
        conn.execute('UPDATE tasks SET started_at=1, max_runtime_seconds=1, claim_expires=1 WHERE id=?', (task,))
        conn.execute('UPDATE task_runs SET started_at=1, claim_expires=1 WHERE id=?', (run,))
        kb._append_event(conn, task, 'heartbeat', run_id=run)
    _kbd_set_worker_pid(conn, task, 987654321)
    monkeypatch.setattr(kb, '_pid_alive', lambda pid: action == 'extend')
    monkeypatch.setattr(kb, '_terminate_reclaimed_worker', lambda *a, **kw: {
        'termination_attempted': True, 'host_local': True, 'terminated': action != 'defer'})
    monkeypatch.setattr(kbd, '_classify_worker_exit', lambda pid: {
        'crash': ('nonzero_exit', 1), 'rate': ('rate_limited', kb.KANBAN_RATE_LIMIT_EXIT_CODE),
        'protocol': ('clean_exit', 0)}[action])
    signals = []
    def invoke():
        if action in {'extend', 'defer'}: return kb.release_stale_claims(conn)
        if action == 'timeout': return kb.enforce_max_runtime(conn, signal_fn=lambda *args: signals.append(args))
        if action == 'stale': return kb.detect_stale_running(conn, stale_timeout_seconds=1)
        return kb.detect_crashed_workers(conn)
    before = rows(conn)
    if fault:
        conn.execute("CREATE TRIGGER matrix_fault BEFORE INSERT ON authority_history BEGIN SELECT RAISE(ABORT,'matrix fault'); END")
        with pytest.raises(sqlite3.IntegrityError, match='matrix fault'): invoke()
        assert rows(conn) == before
        return
    invoke()
    record = next(r for r in reversed(history(conn, cap)['records']) if r['kind'] == kind)
    assert record['run_state']['id'] == run
    assert record['owner']['owner_ref'] == 'matrix-owner'
    assert (record['run_state']['ended_at'] is not None) == terminal
    assert record['task_state']['status'] == ('ready' if terminal else 'running')
    if not terminal:
        assert record['task_state']['claim_expires'] == record['run_state']['claim_expires']
        assert record['task_state']['claim_expires'] > 1
    if action == 'timeout': assert signals and signals[0][0] == 987654321


def test_typed_dependency_and_block_loop_use_real_runtime(conn):
    cap, task = enrolled(conn)
    parent = kb.create_task(conn, title='parent')
    kb.claim_task(conn, task, claimer='private-bearer-token')
    kb.link_tasks(conn, parent, task)
    assert kb.block_task(conn, task, reason='dependency', kind='dependency')
    record = history(conn, cap)['records'][-1]
    assert record['kind'] == 'dependency_wait'
    assert record['task_state']['status'] == 'todo'
    assert record['owner']['owner_ref'] == 'owner-1'
    kb.recompute_ready(conn)
    assert kb.get_task(conn, task).status == 'todo'
    kb.complete_task(conn, parent)
    assert kb.get_task(conn, task).status == 'ready'
    for attempt in range(kb.BLOCK_RECURRENCE_LIMIT):
        claim = kb.claim_task(conn, task, claimer='private-bearer-token')
        assert kb.block_task(conn, task, reason='input', kind='needs_input', expected_run_id=claim.current_run_id)
        record = history(conn, cap)['records'][-1]
        assert record['run_state']['id'] == claim.current_run_id
        assert record['owner']['owner_ref'] == 'owner-1'
        if attempt + 1 < kb.BLOCK_RECURRENCE_LIMIT:
            assert kb.unblock_task(conn, task)
    assert record['kind'] == 'block_loop_detected'
    assert record['task_state']['status'] == 'triage'


@pytest.mark.parametrize('archive', [True, False], ids=['archive', 'destructive'])
def test_failed_board_removal_retries_with_enrollment_fenced(tmp_path, monkeypatch, archive):
    from pathlib import Path
    import shutil

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    kb.create_board('reserved')
    directory = kb.board_dir('reserved')
    owner, operation = (Path, 'rename') if archive else (shutil, 'rmtree')
    real = getattr(owner, operation)

    def fail(path, *args, **kwargs):
        if Path(path) == directory:
            raise OSError('injected removal failure')
        return real(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(owner, operation, fail)
        with pytest.raises(OSError, match='injected removal failure'):
            kb.remove_board('reserved', archive=archive)
    assert directory.is_dir()
    with kb.connect_closing(board='reserved') as db:
        assert db.execute('SELECT * FROM authority_history_removal').fetchall()
        with pytest.raises(ValueError, match='remov'):
            kb.enroll_authority_history(db)
        task = kb.create_task(db, title='still ordinary')
        assert kb.get_task(db, task)

    result = kb.remove_board('reserved', archive=archive)
    assert result['action'] == ('archived' if archive else 'deleted')
    assert not directory.exists()
    if archive:
        target = Path(result['new_path'])
        assert target.is_dir()
        with kb.connect_closing(target / 'kanban.db') as db:
            assert kb.get_task(db, task).title == 'still ordinary'
            assert len(db.execute('SELECT * FROM authority_history_removal').fetchall()) == 1
            with pytest.raises(ValueError, match='remov'):
                kb.enroll_authority_history(db)
    else:
        assert result['new_path'] == ''
