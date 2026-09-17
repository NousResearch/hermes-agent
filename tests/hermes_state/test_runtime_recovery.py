import json
import sqlite3

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


def legacy(path, rows):
    with sqlite3.connect(path) as conn:
        conn.execute('''CREATE TABLE admissions(seq INTEGER PRIMARY KEY, admission_id TEXT,
            target_session_id TEXT, root TEXT, lineage TEXT, payload TEXT, status TEXT,
            outcome TEXT, owner TEXT, generation INTEGER)''')
        conn.executemany('INSERT INTO admissions VALUES(?,?,?,?,?,?,?,?,?,?)', rows)


def row(seq, target, status='queued'):
    return (seq, f'old-{seq}', target, 's', json.dumps(['s', target]), json.dumps({'text': f'input-{seq}', 'intent': 'queue'}), status, 'completed' if status == 'terminal' else None, 'dead', seq if status == 'started' else None)


def test_frozen_legacy_import_preserves_work_without_branch_theft(tmp_path):
    assert callable(getattr(rt, 'import_legacy_session_admissions', None)), 'canonical legacy migration missing'
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='test')
        db.end_session('s', 'compression')
        db.create_session('tip', source='test', parent_session_id='s')
        db.create_session('fork', source='test', parent_session_id='s', model_config={'_branched_from': 's'})
        source = tmp_path / 'prompt-admissions.db'
        unbound = list(row(2, 's', 'started'))
        unbound[-1] = None  # Legacy claim and generation binding were separate commits.
        legacy(source, [row(1, 's'), tuple(unbound), row(3, 'fork'), row(4, 's', 'terminal')])
        with sqlite3.connect(source) as conn:
            conn.execute('CREATE TABLE executions(root TEXT PRIMARY KEY, generation INTEGER NOT NULL)')
            conn.execute("INSERT INTO executions VALUES('s', 40)")
        before = source.read_bytes()  # source is closed and drained, not a live database
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        args = dict(epoch=epoch, source_path=source, principal_id='legacy-human', writers_drained=True)
        assert rt.import_legacy_session_admissions(db, **args) == 4
        assert rt.import_legacy_session_admissions(db, **args) == 0
        assert source.read_bytes() == before
        tip = rt.list_session_admissions(db, session_id='tip', pending_only=False)
        assert [r['admission_id'] for r in tip] == ['old-1', 'old-2', 'old-4']
        assert [r['status'] for r in tip] == ['queued', 'unknown', 'terminal']
        assert [r['admission_id'] for r in rt.list_session_admissions(db, session_id='fork')] == ['old-3']
        with pytest.raises(rt.RuntimeStoreError, match='unknown_execution'):
            rt.claim_session_input(db, epoch=epoch, session_id='tip')
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            rt.resolve_unknown_session_input(db, epoch=epoch, admission_id='old-2', generation=99)
        unknown_generation = tip[1]['generation']
        assert type(unknown_generation) is int and unknown_generation > 40
        rt.resolve_unknown_session_input(db, epoch=epoch, admission_id='old-2', generation=unknown_generation)
        claimed = rt.claim_session_input(db, epoch=epoch, session_id='tip')
        assert claimed['admission_id'] == 'old-1' and claimed['generation'] > unknown_generation
        with sqlite3.connect(source) as conn:
            conn.execute("UPDATE admissions SET payload='{}' WHERE seq=1")
        with pytest.raises(rt.RuntimeStoreError, match='admission_conflict'):
            rt.import_legacy_session_admissions(db, **args)
    finally:
        db.close()


def test_legacy_import_failure_rolls_back_rows_and_marker(tmp_path):
    assert callable(getattr(rt, 'import_legacy_session_admissions', None)), 'canonical legacy migration missing'
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='test')
        source = tmp_path / 'prompt-admissions.db'
        legacy(source, [row(1, 's'), row(2, 'missing')])
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        args = dict(epoch=epoch, source_path=source, principal_id='human', writers_drained=True)
        with pytest.raises(rt.RuntimeStoreError, match='invalid_params'):
            rt.import_legacy_session_admissions(db, **(args | {'writers_drained': False}))
        with pytest.raises(rt.RuntimeStoreError, match='not_found'):
            rt.import_legacy_session_admissions(db, **args)
        assert rt.list_session_admissions(db, session_id='s') == []
        assert db._conn.execute("SELECT value FROM state_meta WHERE key='gateway.prompt_admissions_import.v1'").fetchone() is None
        db.create_session('missing', source='test')
        assert rt.import_legacy_session_admissions(db, **args) == 2
    finally:
        db.close()
