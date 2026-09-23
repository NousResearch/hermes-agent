import pytest
import sqlite3

from hermes_state import SessionDB
import hermes_state_runtime as rt


def setup_worker(tmp_path):
    assert callable(getattr(rt, 'register_worker_execution', None)), 'canonical worker receipts missing'
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = rt.begin_runtime_epoch(db, instance_id='boot')
    assignment = dict(execution_id='worker', session_id='s', generation=0)
    rt.register_worker_execution(db, epoch=epoch, **assignment, kind='compute', adoption_secret='private-fixture')
    return db, epoch, assignment


def test_worker_receipts_require_exact_sequence_assignment_and_adoption(tmp_path):
    db, epoch, assignment = setup_worker(tmp_path)
    try:
        assert rt.claim_session_input(db, epoch=epoch, session_id='s') is None
        args = dict(epoch=epoch, **assignment, sequence=1, role='assistant', content='kept output')
        result = rt.persist_worker_message(db, **args)
        assert rt.persist_worker_message(db, **args) == result
        assert len(db.get_messages('s')) == 1
        for change, reason in [({'content': 'different'}, 'admission_conflict'), ({'sequence': 3}, 'invalid_params'), ({'session_id': 'other'}, 'permission_denied'), ({'generation': 1}, 'stale_generation')]:
            with pytest.raises(rt.RuntimeStoreError, match=reason):
                rt.persist_worker_message(db, **(args | change))
        new_epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        rt.recover_session_inputs(db, epoch=new_epoch)
        with pytest.raises(rt.RuntimeStoreError, match='stale_epoch'):
            rt.persist_worker_message(db, **args)
        with pytest.raises(rt.RuntimeStoreError, match='stale_epoch'):
            rt.persist_worker_message(db, **(args | {'epoch': new_epoch}))
        with pytest.raises(rt.RuntimeStoreError, match='permission_denied'):
            rt.adopt_worker_execution(db, epoch=new_epoch, **assignment, adoption_secret='wrong')
        adopted = rt.adopt_worker_execution(db, epoch=new_epoch, **assignment, adoption_secret='private-fixture')
        assert adopted['owner_epoch'] == new_epoch and 'adoption_digest' not in adopted
        assert rt.persist_worker_message(db, **(args | {'epoch': new_epoch})) == result
        rt.persist_worker_message(db, **(args | {'epoch': new_epoch, 'sequence': 2, 'content': 'next'}))
        finished = rt.finish_worker_execution(db, epoch=new_epoch, **assignment)
        db.close()
        db = SessionDB(db_path=tmp_path / 'state.db')
        before = db._conn.total_changes
        assert rt.finish_worker_execution(db, epoch=new_epoch, **assignment) == finished
        assert rt.persist_worker_message(db, **(args | {'epoch': new_epoch})) == result
        assert db._conn.total_changes == before
        assert 'adoption_digest' not in finished
        with pytest.raises(rt.RuntimeStoreError, match='admission_conflict'):
            rt.persist_worker_message(db, **(args | {'epoch': new_epoch, 'content': 'changed'}))
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            rt.adopt_worker_execution(db, epoch=new_epoch, **assignment, adoption_secret='private-fixture')
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            rt.persist_worker_message(db, **(args | {'epoch': new_epoch, 'sequence': 3}))
        assert [r['content'] for r in db.get_messages('s')] == ['kept output', 'next']
        rt.admit_session_input(db, epoch=new_epoch, principal_id='human', session_id='s', request_id='next', payload={})
        assert rt.claim_session_input(db, epoch=new_epoch, session_id='s')['generation'] > assignment['generation']
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            rt.persist_worker_message(db, **(args | {'epoch': new_epoch}))
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            rt.finish_worker_execution(db, epoch=new_epoch, **assignment)
    finally:
        db.close()


@pytest.mark.parametrize('content', ['once', 'a' + chr(0xD800) + 'b'])
def test_worker_mutation_and_receipt_roll_back_together(tmp_path, content):
    db, epoch, assignment = setup_worker(tmp_path)
    try:
        db._conn.execute("CREATE TRIGGER reject_receipt BEFORE INSERT ON worker_receipts BEGIN SELECT RAISE(ABORT, 'receipt fixture'); END")
        db.create_session('direct', source='test')
        db.append_message('direct', 'assistant', content)
        expected_content = db.get_messages('direct')[0]['content']
        args = dict(epoch=epoch, **assignment, sequence=1, role='assistant', content=content)
        with pytest.raises(sqlite3.IntegrityError, match='receipt fixture'):
            rt.persist_worker_message(db, **args)
        assert db.get_messages('s') == []
        assert db.get_session('s')['message_count'] == 0
        assert db._conn.execute("SELECT last_sequence FROM worker_executions WHERE execution_id='worker'").fetchone()[0] == 0
        db._conn.execute('DROP TRIGGER reject_receipt')
        receipt = rt.persist_worker_message(db, **args)
        db.close()
        db = SessionDB(db_path=tmp_path / 'state.db')
        assert rt.persist_worker_message(db, **args) == receipt
        assert db.get_messages('s')[0]['content'] == expected_content
        if content != expected_content:
            with pytest.raises(rt.RuntimeStoreError, match='admission_conflict'):
                rt.persist_worker_message(db, **(args | {'content': expected_content}))
        assert len(db.get_messages('s')) == db.get_session('s')['message_count'] == 1
    finally:
        db.close()


@pytest.mark.parametrize('closed', [False, True])
def test_worker_append_preserves_compression_write_guards(tmp_path, closed):
    from hermes_state_errors import CompressionSessionClosedError

    db, epoch, assignment = setup_worker(tmp_path)
    try:
        args = dict(epoch=epoch, **assignment, sequence=1, role='assistant', content='before closure')
        receipt = rt.persist_worker_message(db, **args)
        assert db.try_acquire_compression_lock('s', 'compressor')
        if closed:
            db.end_session('s', end_reason='compression')
            db.create_session('tip', source='test', parent_session_id='s')
        before = db.get_session('s')
        messages = db.get_messages('s')
        next_args = args | {'sequence': 2, 'content': 'new output'}
        if closed:
            with pytest.raises(CompressionSessionClosedError):
                db.append_message('s', 'assistant', 'ordinary')
            with pytest.raises(CompressionSessionClosedError):
                rt.persist_worker_message(db, **next_args)
            assert db.get_session('s') == before
            assert db.get_messages('s') == messages
            assert db._conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0] == 1
            assert db._conn.execute('SELECT last_sequence FROM worker_executions').fetchone()[0] == 1
        else:
            db.append_message('s', 'assistant', 'ordinary')
            rt.persist_worker_message(db, **next_args)
            assert [r['content'] for r in db.get_messages('s')] == ['before closure', 'ordinary', 'new output']
        db.close()
        db = SessionDB(db_path=tmp_path / 'state.db')
        assert rt.persist_worker_message(db, **args) == receipt
    finally:
        db.close()
