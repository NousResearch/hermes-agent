"""Real candidate imports and disposable SQLite authority-history contracts."""
import json
import sqlite3

import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_dispatch import _record_task_failure


@pytest.fixture
def conn(tmp_path):
    db = kbc.connect(tmp_path / 'board.db')
    yield db
    db.close()


def test_history_is_explicitly_opt_in_and_prospective(conn):
    task = kb.create_task(conn, title='before enrollment')
    assert callable(getattr(kb, 'enroll_authority_history', None)), 'supported enrollment API missing'
    assert kb.authority_history_capability(conn) is None
    cap = kb.enroll_authority_history(conn)
    assert cap['version'] == 1
    assert cap['incarnation']
    assert cap['coverage_start'] == 1
    assert kb.enroll_authority_history(conn) == cap
    assert kb.read_authority_history(conn, incarnation=cap['incarnation'])['records'] == []
    assert kb.get_task(conn, task).title == 'before enrollment'


def enrolled(conn):
    cap = kb.enroll_authority_history(conn)
    task = kb.create_task(conn, title='isolated task', assignee='worker')
    assert callable(getattr(kb, 'bind_authority_task', None)), 'immutable task binding API missing'
    kb.bind_authority_task(conn, task, repo_id='repo', project_id='project')
    kb.bind_authority_owner(conn, task, claimer='private-bearer-token',
                            consumer_id='consumer', runtime_id='runtime', owner_ref='owner-1')
    kb.recompute_ready(conn)
    return cap, task


def history(conn, cap):
    return kb.read_authority_history(conn, incarnation=cap['incarnation'])


def test_claim_renew_complete_retained_after_gc_and_delete(conn):
    cap, task = enrolled(conn)
    claimed = kb.claim_task(conn, task, claimer='private-bearer-token')
    assert claimed is not None
    assert kb.heartbeat_claim(conn, task, claimer='private-bearer-token', ttl_seconds=3000)
    assert kb.complete_task(conn, task, result='private-bearer-token')
    before = history(conn, cap)['records']
    assert {'claimed', 'claim_renewed', 'completed'} <= {r['kind'] for r in before}
    assert 'private-bearer-token' not in json.dumps(before)
    assert kb.gc_events(conn, older_than_seconds=-1) > 0
    assert kb.delete_task(conn, task)
    after = history(conn, cap)
    assert after['records'][:-1] == before
    assert after['records'][-1]['kind'] == 'deleted'
    assert after['records'][-1]['task_state']['status'] == 'deleted'
    assert after['highwater'] == len(after['records'])


def test_unknown_owner_rolls_back_grant(conn):
    cap, task = enrolled(conn)
    before = history(conn, cap)
    with pytest.raises(ValueError, match='owner'):
        kb.claim_task(conn, task, claimer='not-bound')
    assert kb.get_task(conn, task).status == 'ready'
    assert kb.list_runs(conn, task) == []
    assert history(conn, cap) == before


def test_journal_failure_rolls_back_authority(conn):
    cap, task = enrolled(conn)
    before = history(conn, cap)
    conn.execute("CREATE TRIGGER fault BEFORE INSERT ON authority_history BEGIN SELECT RAISE(ABORT, 'journal fault'); END")
    with pytest.raises(sqlite3.IntegrityError, match='journal fault'):
        kb.claim_task(conn, task, claimer='private-bearer-token')
    assert kb.get_task(conn, task).status == 'ready'
    assert kb.list_runs(conn, task) == []
    assert history(conn, cap) == before


def test_bindings_cannot_change_or_backfill_running_task(conn):
    cap, task = enrolled(conn)
    with pytest.raises(ValueError):
        kb.bind_authority_task(conn, task, repo_id='other', project_id='project')
    with pytest.raises(ValueError):
        kb.bind_authority_owner(conn, task, claimer='private-bearer-token',
                                consumer_id='other', runtime_id='runtime', owner_ref='owner-1')
    ordinary = kb.create_task(conn, title='already running', assignee='worker')
    kb.recompute_ready(conn)
    kb.claim_task(conn, ordinary, claimer='ordinary')
    with pytest.raises(ValueError):
        kb.bind_authority_task(conn, ordinary, repo_id='repo', project_id='project')
    assert kb.authority_history_capability(conn) == cap


@pytest.mark.parametrize('after', [-1, True, 999999])
def test_reader_rejects_unknown_cursor(conn, after):
    cap = kb.enroll_authority_history(conn)
    with pytest.raises(ValueError):
        kb.read_authority_history(conn, incarnation=cap['incarnation'], after=after)


def test_reader_rejects_wrong_incarnation(conn):
    kb.enroll_authority_history(conn)
    with pytest.raises(ValueError):
        kb.read_authority_history(conn, incarnation='unknown')


@pytest.mark.parametrize('table', ['authority_history', 'authority_history_meta',
                                  'authority_task_bindings', 'authority_owner_bindings'])
def test_immutable_sql_records(conn, table):
    cap, task = enrolled(conn)
    kb.claim_task(conn, task, claimer='private-bearer-token')
    with pytest.raises(sqlite3.IntegrityError, match='immutable'):
        conn.execute(f'DELETE FROM {table}')
    assert history(conn, cap)['records']


def test_migration_reopen_preserves_exact_records(conn):
    cap, task = enrolled(conn)
    kb.claim_task(conn, task, claimer='private-bearer-token')
    before = history(conn, cap)
    path = conn.execute('PRAGMA database_list').fetchone()[2]
    kb.init_db(__import__('pathlib').Path(path))
    other = kbc.connect(__import__('pathlib').Path(path))
    try:
        assert history(other, cap) == before
    finally:
        other.close()


@pytest.mark.parametrize('archive', [True, False])
def test_enrolled_board_removal_refused(tmp_path, monkeypatch, archive):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    kb.create_board('history')
    with kbc.connect_closing(board='history') as db:
        kb.enroll_authority_history(db)
    with pytest.raises(ValueError, match='enrolled'):
        kb.remove_board('history', archive=archive)
    assert kb.board_exists('history')


def test_enrollment_cannot_race_board_removal(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    kb.create_board('race')
    directory = kb.board_dir('race')
    real_rename = Path.rename
    reached = []
    def interleaved_rename(path, target):
        if path == directory:
            with kbc.connect_closing(directory / 'kanban.db') as db:
                with pytest.raises(ValueError, match='remov'):
                    kb.enroll_authority_history(db)
            reached.append(True)
        return real_rename(path, target)
    monkeypatch.setattr(Path, 'rename', interleaved_rename)
    kb.remove_board('race')
    assert reached == [True]


def test_two_processes_exactly_one_claim(conn):
    import subprocess
    import sys
    from pathlib import Path
    cap, task = enrolled(conn)
    path = conn.execute('PRAGMA database_list').fetchone()[2]
    probe = Path(__file__).resolve().parent / 'claim_probe.py'
    children = [subprocess.Popen([sys.executable, '-B', str(probe), path, task, 'claim'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    results = [p.communicate(timeout=40) for p in children]
    assert [p.returncode for p in children] == [0, 0], results
    assert sorted(out.strip() for out, err in results) == ['LOST', 'WON'], results
    assert len([r for r in history(conn, cap)['records'] if r['kind'] == 'claimed']) == 1
    assert len(kb.list_runs(conn, task)) == 1


@pytest.mark.parametrize('mode', ['before', 'after'])
def test_abrupt_process_exit_commit_boundary(conn, mode):
    import subprocess
    import sys
    from pathlib import Path
    cap, task = enrolled(conn)
    before = history(conn, cap)
    path = conn.execute('PRAGMA database_list').fetchone()[2]
    probe = Path(__file__).resolve().parent / 'claim_probe.py'
    child = subprocess.run([sys.executable, '-B', str(probe), path, task, mode],
                           capture_output=True, text=True, timeout=40)
    assert child.returncode == 23, child.stderr
    if mode == 'before':
        assert kb.get_task(conn, task).status == 'ready'
        assert kb.list_runs(conn, task) == []
        assert history(conn, cap) == before
    else:
        assert kb.get_task(conn, task).status == 'running'
        assert len(kb.list_runs(conn, task)) == 1
        assert history(conn, cap)['records'][-1]['kind'] == 'claimed'
        assert kb.claim_task(conn, task, claimer='private-bearer-token') is None
        assert len(kb.list_runs(conn, task)) == 1


@pytest.mark.parametrize('kind', ['alien_transition'])
def test_unknown_event_rolls_back(conn, kind):
    cap, task = enrolled(conn)
    before = history(conn, cap)
    with pytest.raises(ValueError, match='unknown'):
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (task,))
            kb._append_event(conn, task, kind)
    assert history(conn, cap) == before
    assert kb.get_task(conn, task).status == 'ready'


def test_defensive_repair_is_refused_for_enrolled_tasks(conn):
    cap, task = enrolled(conn)
    claimed = kb.claim_task(conn, task, claimer='private-bearer-token')
    # Simulate administrator corruption explicitly; normal SQL is now refused.
    guard = conn.execute("SELECT sql FROM sqlite_master WHERE name='authority_writer_tasks_update'").fetchone()[0]
    conn.execute('DROP TRIGGER authority_writer_tasks_update')
    conn.execute("UPDATE tasks SET status='ready', claim_lock=NULL WHERE id=?", (task,))
    conn.execute(guard)
    before = history(conn, cap)
    with pytest.raises(ValueError, match='invariant|repair'):
        kb.claim_task(conn, task, claimer='private-bearer-token')
    assert history(conn, cap) == before
    assert kb.get_run(conn, claimed.current_run_id).ended_at is None


@pytest.mark.parametrize('record', ['{}', '{broken', '{"kind":"alien"}'])
def test_malformed_record_fails_closed(conn, record):
    cap, task = enrolled(conn)
    # Explicit administrator corruption, not a supported insert bypass.
    guard = conn.execute("SELECT sql FROM sqlite_master WHERE name='authority_history_owned_insert'").fetchone()[0]
    conn.execute('DROP TRIGGER authority_history_owned_insert')
    try:
        conn.execute('INSERT INTO authority_history(version,record) VALUES(1,?)', (record,))
    finally:
        conn.execute(guard)
    with pytest.raises(ValueError):
        history(conn, cap)


def test_missing_guards_block_writes_and_reopen(conn):
    cap, task = enrolled(conn)
    conn.execute('DROP TRIGGER authority_history_no_delete')
    with pytest.raises(ValueError):
        kb.heartbeat_claim(conn, task, claimer='private-bearer-token')
    path = conn.execute('PRAGMA database_list').fetchone()[2]
    with pytest.raises(ValueError):
        kb.init_db(__import__('pathlib').Path(path))


@pytest.mark.parametrize('action,kind,status', [
    ('reclaim', 'reclaimed', 'ready'), ('archive', 'archived', 'archived'),
    ('block', 'blocked', 'blocked'), ('spawn_failure', 'spawn_failed', 'ready'),
    ('spawn_give_up', 'gave_up', 'blocked'), ('stale', 'reclaimed', 'ready'),
    ('heartbeat', 'heartbeat', 'running'), ('delete_active', 'deleted', 'deleted'),
])
def test_actual_lifecycle_transitions(conn, monkeypatch, action, kind, status):
    cap, task = enrolled(conn)
    claimed = kb.claim_task(conn, task, claimer='private-bearer-token', ttl_seconds=10)
    if action == 'reclaim':
        assert kb.reclaim_task(conn, task)
    elif action == 'archive':
        assert kb.archive_task(conn, task)
    elif action == 'block':
        assert kb.block_task(conn, task, reason='human review')
    elif action == 'spawn_failure':
        assert not _record_task_failure(
            conn, task, 'test failure', outcome='spawn_failed',
            failure_limit=3, release_claim=True, end_run=True)
    elif action == 'spawn_give_up':
        assert _record_task_failure(
            conn, task, 'test failure', outcome='spawn_failed',
            failure_limit=1, release_claim=True, end_run=True)
    elif action == 'stale':
        monkeypatch.setattr(kb.time, 'time', lambda: claimed.claim_expires + 1)
        assert kb.release_stale_claims(conn) == 1
    elif action == 'heartbeat':
        assert kbd.heartbeat_worker(conn, task, expected_run_id=claimed.current_run_id)
    elif action == 'delete_active':
        before = history(conn, cap)
        with pytest.raises(ValueError, match='active'):
            kb.delete_task(conn, task)
        assert history(conn, cap) == before
        assert kb.get_task(conn, task).current_run_id == claimed.current_run_id
        return
    record = history(conn, cap)['records'][-1]
    assert record['kind'] == kind
    assert record['task_state']['status'] == status
    if action != 'delete_active':
        assert record['owner']['owner_ref'] == 'owner-1'
    if status not in {'running', 'deleted'}:
        assert record['run_state']['ended_at'] is not None


def test_review_claim_uses_same_durable_owner_binding(conn):
    cap, task = enrolled(conn)
    # Exercise the existing review claim API with the same setup its dispatcher
    # receives; this setup is not claimed as coverage of the dashboard writer.
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (task,))
        kb._append_event(conn, task, 'status')
    claimed = kb.claim_review_task(conn, task, claimer='private-bearer-token')
    assert claimed is not None
    record = history(conn, cap)['records'][-1]
    assert record['kind'] == 'claimed'
    assert record['run_state']['id'] == claimed.current_run_id
    assert record['owner']['runtime_id'] == 'runtime'


def test_archive_delete_retains_tombstone(conn):
    cap, task = enrolled(conn)
    kb.claim_task(conn, task, claimer='private-bearer-token')
    kb.archive_task(conn, task)
    assert kb.delete_archived_task(conn, task)
    record = history(conn, cap)['records'][-1]
    assert record['kind'] == 'deleted'
    assert record['task_state']['status'] == 'deleted'


def test_reader_pagination_replays_exact_bytes(conn):
    cap, task = enrolled(conn)
    kb.claim_task(conn, task, claimer='private-bearer-token')
    full = history(conn, cap)
    records, cursor = [], 0
    while cursor < full['highwater']:
        page = kb.read_authority_history(conn, incarnation=cap['incarnation'], after=cursor, limit=1)
        records.extend(page['records'])
        cursor = page['next_sequence']
    assert records == full['records']


def test_dashboard_direct_status_has_durable_history(conn):
    import importlib.util
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / 'plugins/kanban/dashboard/plugin_api.py'
    spec = importlib.util.spec_from_file_location('authority_dashboard_probe', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cap, task = enrolled(conn)
    kb.claim_task(conn, task, claimer='private-bearer-token')
    assert module._set_status_direct(conn, task, 'ready')
    record = history(conn, cap)['records'][-1]
    assert record['kind'] == 'status'
    assert record['task_state']['status'] == 'ready'
    assert record['run_state']['ended_at'] is not None
    assert record['owner']['owner_ref'] == 'owner-1'
    child = kb.create_task(conn, title='child', assignee='worker')
    kb.bind_authority_task(conn, child, repo_id='repo', project_id='project')
    kb.link_tasks(conn, task, child)
    kb.complete_task(conn, task)
    assert kb.get_task(conn, child).status == 'ready'
    assert module._set_status_direct(conn, task, 'todo')
    assert kb.get_task(conn, child).status == 'todo'
    latest = [r for r in history(conn, cap)['records'] if r['binding']['task_id'] == child][-1]
    assert latest['kind'] == 'status'
    assert latest['task_state']['status'] == 'todo'


def test_unemitted_authority_transition_fails_closed(conn):
    cap, task = enrolled(conn)
    before = history(conn, cap)
    with pytest.raises(ValueError, match='unjournaled'):
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (task,))
    assert kb.get_task(conn, task).status == 'ready'
    assert history(conn, cap) == before


def test_run_only_unemitted_transition_fails_closed(conn):
    cap, task = enrolled(conn)
    claimed = kb.claim_task(conn, task, claimer='private-bearer-token')
    before = history(conn, cap)
    with pytest.raises(ValueError, match='unjournaled'):
        with kb.write_txn(conn):
            conn.execute('UPDATE task_runs SET claim_expires=claim_expires+100 WHERE id=?', (claimed.current_run_id,))
    assert history(conn, cap) == before
