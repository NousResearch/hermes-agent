"""A runtime census must not mistake a TEMP shadow for durable queued work."""

import json
import time

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as runtime


@pytest.mark.parametrize('kind', ['view', 'table'])
def test_snapshot_refuses_temp_relation_hiding_a_durable_admission(tmp_path, kind):
    db = SessionDB(db_path=tmp_path / 'state.db')
    shadow = False
    try:
        db.create_session('primary', source='test')
        epoch = runtime.begin_runtime_epoch(db, instance_id='instance-a')
        admitted = runtime.admit_session_input(
            db, epoch=epoch, principal_id='synthetic-owner', session_id='primary',
            request_id='before-shadow', payload={'text': 'retained synthetic work'},
        )
        options = {'epoch': epoch, 'max_records': 128, 'max_bytes': 2 * 1024 * 1024}

        def snapshot():
            return runtime.snapshot_runtime_admission_workers(
                db, deadline=time.monotonic() + 2.0, **options)

        assert snapshot()['counts']['admissions_by_status']['queued'] == 1
        original = dict(db._conn.execute(
            'SELECT * FROM main.session_admissions WHERE admission_id=?',
            (admitted['admission_id'],),
        ).fetchone())
        sql = {
            'view': 'CREATE TEMP VIEW SESSION_ADMISSIONS AS SELECT * FROM main.session_admissions WHERE 0',
            'table': 'CREATE TEMP TABLE SESSION_ADMISSIONS AS SELECT * FROM main.session_admissions WHERE 0',
        }[kind]
        db._execute_write(lambda conn: conn.execute(sql))
        shadow = True
        assert db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
        assert dict(db._conn.execute(
            'SELECT * FROM main.session_admissions WHERE admission_id=?',
            (admitted['admission_id'],),
        ).fetchone()) == original
        observed = None
        refusal = None
        refusal_field = None
        try:
            observed = snapshot()
        except runtime.RuntimeStoreError as exc:
            refusal, refusal_field = exc.reason, exc.field
        assert not db._conn.in_transaction
        db._execute_write(lambda conn: conn.execute(f'DROP {kind} temp.SESSION_ADMISSIONS'))
        shadow = False
        assert dict(db._conn.execute(
            'SELECT * FROM main.session_admissions WHERE admission_id=?',
            (admitted['admission_id'],),
        ).fetchone()) == original
        assert snapshot()['counts']['admissions_by_status']['queued'] == 1
        assert runtime.admit_session_input(
            db, epoch=epoch, principal_id='synthetic-owner', session_id='primary',
            request_id='after-shadow', payload={'text': 'new synthetic work'},
        )['status'] == 'queued'
        assert snapshot()['counts']['admissions_by_status']['queued'] == 2
        print(json.dumps({'diagnostic': 'runtime-temp-shadow', 'kind': kind,
                          'durableQueuedRows': 1,
                          'returnedCounts': observed.get('counts') if observed else None,
                          'refusal': refusal, 'refusalField': refusal_field,
                          'originalRowRetained': True, 'restoredWriterSucceeded': True}), flush=True)
        assert refusal == 'snapshot_schema'
        assert refusal_field == 'session_admissions'
    finally:
        try:
            if shadow:
                db._execute_write(lambda conn: conn.execute(f'DROP {kind} temp.SESSION_ADMISSIONS'))
        finally:
            db.close()
