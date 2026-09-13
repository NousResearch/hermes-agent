"""Strict retained-stream regression cases; corruption is explicit test-only SQL."""
import json
import sqlite3
import pytest
from hermes_cli import kanban_db as kb
from tests.hermes_cli.test_kanban_authority_history import conn, enrolled, history
from tests.hermes_cli.test_kanban_history_pass2 import rows


def corrupt_insert(db, table, sql, params=()):
    """Administrator-only corruption: remove one guard, then restore exact SQL.

    Never enable the production owned-writer gate for an unsupported insertion.
    Reader/grant assertions below run with every production guard restored.
    """
    name = table + '_owned_insert'
    guard = db.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()[0]
    db.execute(f'DROP TRIGGER {name}')
    try:
        db.execute(sql, params)
    finally:
        db.execute(guard)


def corrupt_record(db, sequence, text):
    guard = db.execute("SELECT sql FROM sqlite_master WHERE name='authority_history_no_update'").fetchone()[0]
    db.execute('DROP TRIGGER authority_history_no_update')
    db.execute('UPDATE authority_history SET record=? WHERE sequence=?', (text, sequence))
    db.execute(guard)


@pytest.mark.parametrize('corruption', ['duplicate', 'bool_version', 'float_version',
    'null_run_id', 'zero_run_id', 'null_start', 'wrong_binding', 'wrong_owner',
    'missing_owner', 'no_current_run', 'extra_nested', 'nan', 'source_reuse',
    'invented_source', 'wrong_kind', 'wrong_observed'])
def test_corrupt_retained_record_refuses_read_and_new_authority(conn, corruption):
    cap, task = enrolled(conn)
    kb.claim_task(conn, task, claimer='private-bearer-token')
    kb.complete_task(conn, task)
    all_records = history(conn, cap)['records']
    grant = next(r for r in all_records if r['kind'] == 'claimed')
    sequence = grant['sequence']
    data = {k: v for k, v in grant.items() if k not in {'sequence', 'version'}}
    if corruption == 'bool_version': data['binding']['version'] = True
    elif corruption == 'float_version': data['owner']['version'] = 1.0
    elif corruption == 'null_run_id': data['run_state']['id'] = None
    elif corruption == 'zero_run_id': data['run_state']['id'] = 0
    elif corruption == 'null_start': data['run_state']['started_at'] = None
    elif corruption == 'wrong_binding': data['binding']['repo_id'] = 'unrelated'
    elif corruption == 'wrong_owner': data['owner']['runtime_id'] = 'unrelated'
    elif corruption == 'missing_owner': data['owner'] = None
    elif corruption == 'no_current_run': data['task_state']['current_run_id'] = None
    elif corruption == 'extra_nested': data['owner']['extra'] = 'unexpected'
    elif corruption == 'nan': data['task_state']['claim_expires'] = float('nan')
    elif corruption == 'source_reuse': data['source_event_id'] = all_records[0]['source_event_id']
    elif corruption == 'invented_source': data['source_event_id'] = 9999999
    elif corruption == 'wrong_kind': data['kind'] = 'heartbeat'
    elif corruption == 'wrong_observed': data['observed_at'] += 1
    # Operational source GC must not erase provenance needed by the reader.
    conn.execute('DELETE FROM task_events')
    text = json.dumps(data)
    if corruption == 'duplicate': text = '{"kind":"heartbeat",' + text[1:]
    corrupt_record(conn, sequence, text)
    before = rows(conn)
    # Corruption before a requested page cannot disappear behind a cursor.
    with pytest.raises(ValueError):
        kb.read_authority_history(conn, incarnation=cap['incarnation'], after=sequence, limit=1)
    with pytest.raises(ValueError):
        kb.recompute_ready(conn)
    assert rows(conn) == before


@pytest.mark.parametrize('table', ['authority_history_meta', 'authority_task_bindings',
    'authority_owner_bindings', 'authority_run_bindings', 'authority_history'])
@pytest.mark.parametrize('additional', [False, True])
def test_replace_cannot_rewrite_immutable_identity(conn, table, additional):
    cap, task = enrolled(conn)
    kb.claim_task(conn, task, claimer='private-bearer-token')
    before = rows(conn)
    db = conn if not additional else sqlite3.connect(conn.execute('PRAGMA database_list').fetchone()[2], isolation_level=None)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            db.execute(f'INSERT OR REPLACE INTO {table} SELECT * FROM {table} LIMIT 1')
    finally:
        if additional: db.close()
    assert rows(conn) == before


def test_zero_based_sequence_cannot_masquerade_as_complete(conn):
    cap, task = enrolled(conn)
    record = conn.execute('SELECT record FROM authority_history ORDER BY sequence LIMIT 1').fetchone()[0]
    highwater = conn.execute('SELECT MAX(sequence) FROM authority_history').fetchone()[0]
    corrupt_insert(conn, 'authority_history', 'INSERT INTO authority_history(sequence,version,record) VALUES(0,1,?)', (record,))
    corrupt_insert(conn, 'authority_history', 'INSERT INTO authority_history(sequence,version,record) VALUES(?,1,?)', (highwater + 2, record))
    with pytest.raises(ValueError): history(conn, cap)


def test_orphan_binding_is_not_unenrolled_capability(conn):
    corrupt_insert(conn, 'authority_task_bindings', "INSERT INTO authority_task_bindings VALUES('orphan',1,'repo','project')")
    with pytest.raises(ValueError): kb.authority_history_capability(conn)


def test_altered_authority_table_schema_is_refused(conn):
    cap, task = enrolled(conn)
    conn.execute('ALTER TABLE authority_owner_bindings ADD COLUMN unsupported TEXT')
    with pytest.raises(ValueError): kb.authority_history_capability(conn)
    with pytest.raises(ValueError): kb.claim_task(conn, task, claimer='private-bearer-token')


def test_unused_malformed_owner_binding_blocks_grant(conn):
    cap, task = enrolled(conn)
    corrupt_insert(conn, 'authority_owner_bindings', "INSERT INTO authority_owner_bindings VALUES(?, 'not-a-digest', 1, 'consumer', 'runtime', 'owner')", (task,))
    before = rows(conn)
    with pytest.raises(ValueError): kb.claim_task(conn, task, claimer='private-bearer-token')
    assert rows(conn) == before


def test_foreign_task_binding_in_a_record_is_refused(conn):
    cap, task = enrolled(conn)
    kb.claim_task(conn, task, claimer='private-bearer-token')
    other = kb.create_task(conn, title='other')
    kb.bind_authority_task(conn, other, repo_id='other-repo', project_id='project')
    record = next(r for r in history(conn, cap)['records'] if r['kind'] == 'claimed')
    sequence = record.pop('sequence')
    record.pop('version')
    record['binding'] = dict(conn.execute('SELECT * FROM authority_task_bindings WHERE task_id=?', (other,)).fetchone())
    corrupt_record(conn, sequence, json.dumps(record))
    with pytest.raises(ValueError): history(conn, cap)


@pytest.mark.parametrize('additional', [False, True])
@pytest.mark.parametrize('table,sql', [
    ('history', "INSERT INTO authority_history(version,record) VALUES(1,'{}')"),
    ('binding', "INSERT INTO authority_task_bindings VALUES('forged',1,'repo','project')"),
])
def test_raw_insert_cannot_claim_owned_authority(conn, additional, table, sql):
    enrolled(conn)
    before = rows(conn)
    db = conn if not additional else sqlite3.connect(conn.execute('PRAGMA database_list').fetchone()[2], isolation_level=None)
    try:
        db.execute('BEGIN IMMEDIATE')
        with pytest.raises(sqlite3.DatabaseError): db.execute(sql)
        db.rollback()
    finally:
        if additional: db.close()
    assert rows(conn) == before


def test_capture_retains_observed_source_timestamp(conn):
    cap, task = enrolled(conn)
    kb.claim_task(conn, task, claimer='private-bearer-token')
    for record in history(conn, cap)['records']:
        observed = conn.execute('SELECT created_at FROM task_events WHERE id=?', (record['source_event_id'],)).fetchone()[0]
        assert record.get('observed_at') == observed
